import numpy as np
import pandas as pd
import sklearn.metrics
import scipy.optimize
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestClassifier
from scipy import pi,sqrt,exp
from scipy.special import erf
from scipy.stats import skewnorm
import os
import pickle
# import matplotlib2tikz
import collections
import itertools

from strouse_ib import IB
import CvCalibrationModel
import ItrCalculatorProb
import time


class ModelTrainer(object):
    def __init__(self):
        self.recordings = []
        self.eeg = []
        self.look_back_length = None
        self.cross_validation_folds = None
        self.training_recordings = []
        self.testing_recordings = []
        self.training_data = None
        self.training_labels = None
        self.testing_data = None
        self.testing_labels = None
        self.testing_roc = None
        self.training_roc = None
        self.training_prc = None
        self.testing_prc = None
        self.model = None
        self.lda_model = None
        self.transition_model = None
        self.thresholds = None
        self.min_max = None
        self.random_forest_model = None
        self.cv_model = None
        self.features_to_use = None

    def setEeg(self, eeg):
        self.eeg = eeg

    def setup(self, look_back_length, feature_names, data, labels, n_classes):
        self.recordings = data
        self.labels = labels
        # Check before testing!!
        self.t_use_ml = False
        self.t_use_maf_on_features = False
        self.t_use_maf_on_probas = False and self.t_use_ml
        self.t_normalise_probas = False and self.t_use_ml
        self.t_matrix_builder_types = [False]  # [True, False]
        self.n_classes = n_classes
        self.t_remove_samples_features = True and self.t_use_maf_on_features
        self.t_remove_samples_probas = True and self.t_use_maf_on_probas
        self.t_feature_maf = self.getMafLength(self.t_use_maf_on_features)
        self.t_proba_maf = self.getMafLength(self.t_use_maf_on_probas)
        self.t_precisions_bounded = True
        self.t_predictions_bounded = True
        self.itr_calculator_prob = ItrCalculatorProb.ItrCalculatorProb(
            window_length=1,
            step=0.125,
            feature_maf_length=self.t_feature_maf,
            proba_maf_length=self.t_proba_maf,
            look_back_length=1 if self.t_use_ml is False else look_back_length,
            n_targets=self.n_classes,
        )

        self.features_to_use = feature_names
        self.look_back_length = look_back_length
        self.training_recordings = self.recordings[:]
        self.cv_model = CvCalibrationModel.TrainingModel()
        self.cv_model.setup(self.features_to_use, self.look_back_length, self.recordings, self.t_matrix_builder_types)

    def getConfusionMatrix(self, prediction, labels, label_order):
        return sklearn.metrics.confusion_matrix(labels, prediction, labels=label_order)

    def getThresholdConfusionMatrix(self, prediction, labels, label_order):
        return sklearn.metrics.confusion_matrix(labels, prediction, labels=list(label_order)+["None"])

    def plotChange(self, data, labels, index, color, plot_count, target_count):
        x = np.arange(0, len(data))
        plt.subplot(plot_count, 1, index+1)
        decision = data.T[index]
        plt.plot(x, decision, color=color)
        plt.plot(x, (labels == index % target_count + 1)*decision.max() + (1-(labels == index % target_count + 1))*decision.min(), "r--", color=color)

    def plotAllChanges(self, data, labels, thresholds):
        plt.figure()
        colors = ["red", "green", "blue"]
        plot_count = data.shape[1]
        target_count = len(colors)
        for i in range(plot_count):
            self.plotChange(data, labels, i, colors[i%target_count], plot_count, target_count)
            plt.plot([0, data.shape[0]], [thresholds[i], thresholds[i]], color=colors[i%target_count])

    def splitTrainingData(self):
        data_split = []
        labels_split = []
        for recording, labels in zip(self.training_recordings, self.labels):
            data, labels = self.cv_model.getConcatenatedMatrix([recording], labels)
            data_split.append(data)
            labels_split.append(labels)
        return data_split, labels_split

    def splitAndRollData(self):
        data_split = []
        labels_split = []
        for recording in self.training_recordings:
            data, labels = self.cv_model.getConcatenatedMatrix([recording])
            data_split.append(self.applyRoll(data))
            labels_split.append(labels)
        return data_split, labels_split

    def allExceptOne(self, data, index):
        return [x for i, x in enumerate(data) if i != index]

    def predictProbaCv(self, model, split_data, split_labels):
        folds = len(split_data)
        predictions = []
        for i in range(folds):
            data = np.concatenate(self.allExceptOne(split_data, i), axis=0)
            labels = np.concatenate(self.allExceptOne(split_labels, i), axis=0)
            model.fit(data, labels)
            predictions.append(model.predictProba(split_data[i], self.t_proba_maf, self.t_normalise_probas))
        return predictions

    def calculatePredictionProbability(self, confusion_matrix):
        if not isinstance(confusion_matrix, float):
            matrix_sum = float(confusion_matrix.sum())
            return (matrix_sum-confusion_matrix.sum(axis=0)[-1])/matrix_sum

    def calculateAccuracy(self, confusion_matrix):
        if not isinstance(confusion_matrix, float):
            return float(np.trace(confusion_matrix)) / confusion_matrix.sum()

    def calculateAccuracyIgnoringLastColumn(self, confusion_matrix):
        if not isinstance(confusion_matrix, float):
            return float(np.trace(confusion_matrix))/(confusion_matrix.sum()-confusion_matrix.sum(axis=0)[-1])

    def addLastRowColumn(self, confusion_matrix):
        confusion_matrix = list(map(list, confusion_matrix)) + [[0.0 for _ in range(len(confusion_matrix))]]
        map(lambda x: x.append(0.0), confusion_matrix)
        return np.array(confusion_matrix)

    def printConfusionMatrixData(self, confusion_matrix):
        accuracy = self.calculateAccuracyIgnoringLastColumn(confusion_matrix)
        prediction_probability = self.calculatePredictionProbability(confusion_matrix)
        print("ITR from matrix per prediction")
        print(self.calculate_mi(confusion_matrix))
        print("ITR from matrix per min")
        print(self.calculate_mi(confusion_matrix)*60.0/self.itr_calculator_prob.mdt(prediction_probability))
        print("Standard ITR:")
        print(self.itr_calculator_prob.itrBitPerMin(accuracy, prediction_probability))
        print("Accuracy:")
        print(accuracy)
        print("MDT:")
        print(self.itr_calculator_prob.mdt(prediction_probability))
        print("Made " + str((confusion_matrix.sum()-confusion_matrix.sum(axis=0)[-1])) + " predictions out of " + str(confusion_matrix.sum()) + " possible.")
        print(confusion_matrix)

    def modifySplitLabels(self, use_maf, split_labels):
        if use_maf:
            return map(lambda x: x[1:-1], split_labels)
        else:
            return split_labels

    def getMafLength(self, use_maf):
        return 3 if use_maf else 1

    def fitAndPredictProbaSingle(self, model, data, labels):
        model.fit(data, labels)
        return model.predictProba(data, self.t_proba_maf, self.t_normalise_probas)

    def fitAndPredictProbaCv(self, model, data, labels):
        cv_predictions = np.array(self.predictProbaCv(model, data, labels))
        model.fit(np.concatenate(data, 0), np.concatenate(labels, 0))
        return cv_predictions

    def fitAndPredictProba(self, n_folds, model, data, labels):
        if n_folds == 1:
            return [self.fitAndPredictProbaSingle(model, data[0], labels[0])]
        else:
            return self.fitAndPredictProbaCv(model, data, labels)

    def calculateTrainingFeatures(self, n_folds, model, data, labels):
        if self.t_use_ml:
            return self.fitAndPredictProba(n_folds, model, data, labels)
        else:
            return data

    def calculateTestingFeatures(self, model, data):
        if self.t_use_ml:
            return model.predictProba(data, self.t_proba_maf, self.t_normalise_probas)
        else:
            return data

    def applyRoll(self, data):
        return np.roll(data, 1, axis=1)

    def applyMovingAverage(self, data, n):
        return np.transpose(map(lambda x: self.moving_average(x, n), np.transpose(data)))

    def moving_average(self, a, n) :
        ret = np.cumsum(a, dtype=float)
        ret[n:] = ret[n:] - ret[:-n]
        return ret[n - 1:] / n

    def removeSamples(self, data, samples_to_remove):
        return map(lambda (x, s): np.delete(x, s, axis=0), zip(data, samples_to_remove))

    def removeSamplesBeforeAfterClassChange(self, split_data, split_labels):
        class_change = map(lambda x: np.where(x[:-1] != x[1:])[0], split_labels)
        samples_to_remove = map(lambda y: np.concatenate(map(lambda x: [x, x+1], y)), class_change)
        # print map(lambda (x,y): list(x[i] for i in y), zip(split_labels, samples_to_remove))
        return (
            np.array(self.removeSamples(split_data, samples_to_remove)),
            np.array(self.removeSamples(split_labels, samples_to_remove))
        )

    def removeClass(self, split_data, split_labels, c):
        new_data = []
        new_labels = []
        for data, labels in zip(split_data, split_labels):
            class_indices = np.where(labels != c)
            new_data.append(data[class_indices])
            new_labels.append(labels[class_indices])
        return new_data, new_labels

    def func(self, x, alpha, mu, sigma):  # , c, a):
        normpdf = (1.0/(sigma*np.sqrt(2*pi)))*np.exp(-(np.power((x-mu),2)/(2.0*np.power(sigma,2))))
        normcdf = (0.5*(1.0+erf((alpha*((x-mu)/sigma))/(np.sqrt(2.0)))))
        return 2.0*normpdf*normcdf

    def classify3new(self, bin_edges, result_full_label, shift, threshold,  features):
        idxs = [np.searchsorted(bin_edges[i], f) for i, f in enumerate(features)]
        if shift == 0:
            return result_full_label[tuple(idxs)]
        else:
            if self.n_classes == 4:
                labels = np.pad(result_full_label,shift,mode="edge")[idxs[0]:idxs[0]+2*shift+1, idxs[1]:idxs[1]+2*shift+1, idxs[2]:idxs[2]+2*shift+1, idxs[3]:idxs[3]+2*shift+1]
            elif self.n_classes == 5:
                labels = np.pad(result_full_label, shift, mode="edge")[idxs[0]:idxs[0] + 2 * shift + 1,
                         idxs[1]:idxs[1] + 2 * shift + 1, idxs[2]:idxs[2] + 2 * shift + 1,
                         idxs[3]:idxs[3] + 2 * shift + 1, idxs[4]:idxs[4] + 2 * shift + 1]
            elif self.n_classes == 3:
                labels = np.pad(result_full_label,shift,mode="edge")[idxs[0]:idxs[0]+2*shift+1, idxs[1]:idxs[1]+2*shift+1, idxs[2]:idxs[2]+2*shift+1]
            else:
                raise Exception("not implemented!")

            counts = np.bincount(np.array(labels.flatten(),dtype=np.int))
            max_idx = np.argmax(counts)
            max_val = counts[max_idx]
            if max_val >= threshold:
                return max_idx
            else:
                return 0

    def calculate_mi(self, confusion_matrix):
        mi = 0
        confusion_matrix = (confusion_matrix[:-1,:-1])/float(confusion_matrix[:-1,:-1].sum())
        row_sums = confusion_matrix.sum(1)
        col_sums = confusion_matrix.sum(0)
        for r, row in enumerate(confusion_matrix):
            for c, p in enumerate(row):
                if p != 0:
                    mi += p*np.log2(float(p)/(row_sums[r]*col_sums[c]))
        return mi

    def start(self, subject, make_plots):
        split_data, split_labels = self.splitTrainingData()
        training_confusion_matrices = 0.0
        testing_confusion_matrices = 0.0

        if self.t_use_maf_on_features:
            split_data = map(lambda x: self.applyMovingAverage(x, self.t_feature_maf), split_data)
            split_labels = np.array(self.modifySplitLabels(self.t_use_maf_on_features, split_labels))
        if self.t_remove_samples_features:
            split_data, split_labels = self.removeSamplesBeforeAfterClassChange(split_data, split_labels)
        if self.t_use_maf_on_probas:
            split_labels_proba = self.modifySplitLabels(self.t_use_maf_on_probas, split_labels)
        else:
            split_labels_proba = split_labels
        assert len(split_data) > 1
        testing_predictions = []
        print("Starting 5-fold cross-validation")
        accuracies_trca = []
        itrs_trca = []
        itrs_standard_trca = []
        mdts_trca = []
        predictions_made_trca = []
        accuracies_my = []
        for test_data_index in range(len(split_data)):
            start_time = time.time()
            fold_nr = str(test_data_index+1)
            print("Starting fold " + fold_nr)
            split_training_data = self.allExceptOne(split_data, test_data_index)
            split_training_labels = self.allExceptOne(split_labels, test_data_index)
            split_training_labels_proba = self.allExceptOne(split_labels_proba, test_data_index)
            training_data = np.concatenate(split_training_data, 0)
            training_labels = np.concatenate(split_training_labels, 0)
            testing_data = split_data[test_data_index]
            testing_labels = split_labels[test_data_index]
            testing_labels_proba = split_labels_proba[test_data_index]
            n_folds = len(split_training_data)
            tr_prediction = self.calculateTrainingFeatures(n_folds, self.cv_model, split_training_data, split_training_labels)
            if self.t_remove_samples_probas:
                tr_prediction, split_training_labels_proba = self.removeSamplesBeforeAfterClassChange(tr_prediction, split_training_labels_proba)

            all_training_data = np.concatenate(tr_prediction, 0)
            all_training_lables = np.concatenate(split_training_labels_proba, 0)
            all_training_data = [all_training_data[np.where(all_training_lables == i)] for i in range(self.n_classes)]
            all_training_lables1 = [all_training_lables[np.where(all_training_lables == i)] for i in range(self.n_classes)]

            min_data_length = np.min([d.shape[0] for d in all_training_data])
            all_training_data = [d[:min_data_length] for d in all_training_data]
            all_training_lables1 = [d[:min_data_length] for d in all_training_lables1]
            all_training_lables = np.concatenate(all_training_lables1)

            # (class, feature_value, feature_index)
            all_training_data = np.array(all_training_data)

            testing_prediction = self.calculateTestingFeatures(self.cv_model, testing_data)
            if self.t_remove_samples_probas:
                testing_prediction, testing_labels_proba = self.removeSamplesBeforeAfterClassChange(
                    [testing_prediction], [testing_labels_proba])
                testing_prediction, testing_labels_proba = testing_prediction[0], testing_labels_proba[0]
            testing_predictions.append(testing_prediction)

            features = np.concatenate(all_training_data, axis=0)

            n_classes = self.n_classes
            assert all_training_data.shape[0] == n_classes

            skew = True
            plot_hist_skew = False
            plot_3d = False
            save_folder = None

            shift = 1
            thresh = 25

            n_bins = 30
            same_range = False

            min_features_f = np.min(np.min(all_training_data, axis=0), axis=0)
            max_features_f = np.max(np.max(all_training_data, axis=0), axis=0)
            print(min_features_f, max_features_f)

            n_binss = []
            test = []
            for i in range(n_classes):
                n_binss_row = []
                test_row = []
                for j in range(n_classes):
                    edges = np.histogram_bin_edges(all_training_data[i,:,j], "sturges", (min_features_f[j], max_features_f[j]))
                    n_binss_row.append(len(edges))
                    test_row.append(len(np.histogram_bin_edges(all_training_data[i,:,j], "fd")))
                n_binss.append(n_binss_row)
                test.append(test_row)
            n_binss = np.array(n_binss)
            n_binss = np.array(np.round(np.mean(n_binss, axis=1)), np.int)
            print("test", n_binss, np.round(np.mean(test,axis=1)))

            bin_edges = []
            for f in range(n_classes):
                bin_edges.append(np.linspace(min_features_f[f], max_features_f[f], n_binss[f]))

            histogram = []
            min_features = []
            max_features = []
            parameters = []
            for i in range(n_classes):
                hist_row = []
                min_features_row = []
                max_features_row = []
                parameters_row = []
                for j in range(n_classes):
                    data = all_training_data[i,:,j]
                    hist, _ = np.histogram(data, bins=bin_edges[j], density=True)
                    hist_row.append(np.array([0] + list(hist) + [0]))
                    min_features_row.append(np.min(data))
                    max_features_row.append(np.max(data))
                    if skew:
                        param = skewnorm.fit(data)
                        parameters_row.append(param)
                histogram.append(hist_row)
                min_features.append(min_features_row)
                max_features.append(max_features_row)
                parameters.append(parameters_row)
            min_features = np.array(min_features)
            max_features = np.array(max_features)
            histogram = [[histogram[i][j]/np.sum(histogram[i][j]) for j in range(n_classes)] for i in range(n_classes)]
            parameters = np.array(parameters)

            if plot_hist_skew:
                for c in range(n_classes):
                    for f in range(n_classes):
                        plt.subplot(n_classes, n_classes, c*n_classes+f+1)
                        plt.xlim(np.min(min_features), np.max(max_features))
                        plt.title(str(c)+" "+str(f))
                        # plt.ylim(0, 0.6)
                        plt.hist(all_training_data[c,:,f], bins=bin_edges[f], density=True)
                        points = np.linspace(min_features[c][f], max_features[c][f], 100)
                        plt.plot(points, self.func(points, *parameters[c][f]))
                plt.show()

            if same_range:
                x = np.linspace(np.min(min_features), np.max(max_features), n_bins)
                intervals = [(-float("inf"), x[0])] + [(l, h) for l, h in zip(x[:-1], x[1:])] + [(x[-1], float("inf"))]
                intervals = np.array(intervals)

                intervals_all = []
                for i1, v1 in enumerate(intervals):
                    for i2, v2 in enumerate(intervals):
                        for i3, v3 in enumerate(intervals):
                            intervals_all.append((intervals[i1], intervals[i2], intervals[i3]))
                # (, feature_idx, (low, high))
                intervals_all = np.array(intervals_all)

                if plot_3d:
                    plot_location = intervals_all.copy()
                    plot_location[np.where(plot_location == -np.inf)] = x[0]
                    plot_location = plot_location[:, :, 0]
            else:
                pass

            if skew:
                discretised = []
                for i, row in enumerate(parameters):
                    discretised_row = []
                    for j, param in enumerate(row):
                        cdf_values = []
                        if same_range:
                            y = skewnorm.cdf(x, *param)
                        else:
                            y = skewnorm.cdf(bin_edges[j], *param)
                        cdf_values.append(0)
                        cdf_values.extend(y)
                        cdf_values.append(1)
                        result = []
                        for v1, v2 in zip(cdf_values[:-1], cdf_values[1:]):
                            result.append(v2 - v1)
                        discretised_row.append(result)
                    discretised.append(discretised_row)
                discretised = np.array(discretised)


            probas = []
            for i, row in enumerate(discretised if skew else histogram):
                proba_row = np.round(np.prod(np.stack(np.meshgrid(row[1], row[0], *row[2:]), -1).reshape(-1, n_classes), axis=1)/n_classes, 8)
                probas.append(proba_row)

            probas = np.transpose(probas)
            probas = probas/probas.sum()

            ds = IB.dataset(pxy=probas)

            betas = [100]
            alpha = 1
            hts = []
            iyts = []
            htxs = []
            for beta in betas:
                fit_param = pd.DataFrame(data={'alpha': [alpha], 'zeroLtol': float("inf"), 'Tmax': n_classes, 'clamp': False,
                                               'betas': [beta], 'beta_search': False})#, 'cthresh': 5})
                fit_param['repeats'] = 1
                metrics_conv, dist_conv, metrics_sw, dist_sw = IB.IB(ds, fit_param, keep_steps=False, sw_dist_to_keep={})

                if alpha == 1:
                    hts.append(metrics_conv["ixt"][0])
                if alpha == 0:
                    hts.append(metrics_conv["ht"][0])
                iyts.append(metrics_conv["iyt"][0])
                htxs.append(metrics_conv["ht_x"][0])

                result = np.array(dist_conv["qt_x"][0])

                result_label = np.zeros(result.shape[1])-1
                for i in range(n_classes):
                    result_label[np.where(result[i] == 1)] = i

                if ds.zx is not None:
                    result_full_label = np.zeros(probas.shape[0])
                    result_full_label[ds.zx] = -1
                    result_full_label[[i for i in range(probas.shape[0]) if i not in ds.zx]] = result_label
                else:
                    result_full_label = result_label
                reshape_result_full_label = result_full_label.reshape(tuple(n_binss+1))

                if plot_3d:
                    fig = plt.figure()
                    ax = fig.add_subplot(111, projection='3d')
                    pos = np.transpose(plot_location[np.where(result_full_label == 0)])
                    ax.scatter(pos[0], pos[1], pos[2], c='blue')
                    pos = np.transpose(plot_location[np.where(result_full_label == 1)])
                    ax.scatter(pos[0], pos[1], pos[2], c='red')
                    pos = np.transpose(plot_location[np.where(result_full_label == 2)])
                    ax.scatter(pos[0], pos[1], pos[2], c='green')
                    pd.DataFrame(plot_location[np.where(result_full_label == 0)], columns=["x","y","z"]).to_csv("3d1.csv")
                    pd.DataFrame(plot_location[np.where(result_full_label == 1)], columns=["x","y","z"]).to_csv("3d2.csv")
                    pd.DataFrame(plot_location[np.where(result_full_label == 2)], columns=["x","y","z"]).to_csv("3d3.csv")
                    plt.show()

                itrs_row = []
                itrs_standard_row = []
                accuracies_row = []
                mdts_row = []
                predictions_made_row = []

                predicted_training_labels = map(lambda x: self.classify3new(bin_edges, reshape_result_full_label+1, shift, thresh, x), features)
                training_confusion_matrix = sklearn.metrics.confusion_matrix(all_training_lables+1,
                                                                             predicted_training_labels,
                                                                             labels=[i+1 for i in range(n_classes)] + [0])

                permutations = list(itertools.permutations([i for i in range(n_classes)]))

                itrss = [self.calculateAccuracyIgnoringLastColumn(training_confusion_matrix[:,list(perm) + [n_classes]]) for perm in permutations]
                best_itr = np.argmax(itrss)
                best_perm = list(permutations[best_itr]) + [n_classes]

                predicted_testing_labels = map(lambda x: self.classify3new(bin_edges, reshape_result_full_label+1, shift, thresh, x), testing_prediction)

                training_confusion_matrices += training_confusion_matrix[:,best_perm]
                testing_confusion_matrix = sklearn.metrics.confusion_matrix(testing_labels+1,
                                                                            predicted_testing_labels,
                                                                            labels=[i + 1 for i in range(n_classes)] + [0])
                testing_confusion_matrices += testing_confusion_matrix[:,best_perm]

                confusion_matrix = testing_confusion_matrix[:,best_perm]
                accuracy = self.calculateAccuracyIgnoringLastColumn(confusion_matrix)
                prediction_probability = self.calculatePredictionProbability(confusion_matrix)
                mdt = self.itr_calculator_prob.mdt(prediction_probability)
                accuracies_my.append(accuracy)

                itrs_row.append(self.calculate_mi(confusion_matrix) * 60.0 / mdt)
                itrs_standard_row.append(self.itr_calculator_prob.itrBitPerMin(accuracy, prediction_probability))
                accuracies_row.append(accuracy)
                mdts_row.append(mdt)
                predictions_made_row.append(confusion_matrix.sum() - confusion_matrix.sum(axis=0)[-1])

                print(time.time()-start_time)
                if save_folder is not None:
                    pickle.dump((itrs_row, itrs_standard_row, accuracies_row, mdts_row, predictions_made_row), open(save_folder + str(subject) + "_" + str(test_data_index) + ".p", "w"))

    def checkDataAndPlotTestingCurve(self, curve):
        plt.clf()
        curve.plot()

    def checkDataAndPlotTrainingCurve(self, curves, curve_name):
        for i, curve in enumerate(curves):
            plt.clf()
            curve.plot(i)
            plt.savefig(os.path.join(os.pardir, "plots", curve_name + "_train_fold" + str(i+1)))
