
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D

from scipy.stats import skewnorm
import sklearn.metrics

import os
import itertools

import IB


def predict(bin_edges, result_full_label, shift, threshold, features, n_classes):
    idxs = [np.searchsorted(bin_edges[i], f) for i, f in enumerate(features)]
    if shift == 0:
        return result_full_label[tuple(idxs)]
    else:
        if n_classes == 3:
            labels = np.pad(result_full_label, shift, mode="edge")[idxs[0]:idxs[0] + 2 * shift + 1,
                     idxs[1]:idxs[1] + 2 * shift + 1, idxs[2]:idxs[2] + 2 * shift + 1]
        elif n_classes == 4:
            labels = np.pad(result_full_label, shift, mode="edge")[idxs[0]:idxs[0] + 2 * shift + 1,
                     idxs[1]:idxs[1] + 2 * shift + 1, idxs[2]:idxs[2] + 2 * shift + 1, idxs[3]:idxs[3] + 2 * shift + 1]
        elif n_classes == 5:
            labels = np.pad(result_full_label, shift, mode="edge")[idxs[0]:idxs[0] + 2 * shift + 1,
                     idxs[1]:idxs[1] + 2 * shift + 1, idxs[2]:idxs[2] + 2 * shift + 1,
                     idxs[3]:idxs[3] + 2 * shift + 1, idxs[4]:idxs[4] + 2 * shift + 1]
        else:
            raise Exception("not implemented!")

        counts = np.bincount(np.array(labels.flatten(), dtype=np.int))
        max_idx = np.argmax(counts)
        max_val = counts[max_idx]
        if max_val >= threshold:
            return max_idx
        else:
            return 0


def standard_itr_per_trial(a, n_classes):
    if a == 1:
        return np.log2(n_classes)
    elif a == 0:
        return np.log2(n_classes)+np.log2(1.0/(n_classes-1))
    else:
        return np.log2(n_classes)+a*np.log2(a)+(1-a)*np.log2((1.0-a)/(n_classes-1))


def mdt_from_prediction_prob(p, window_length, step_length):
    return window_length + (1.0/p - 1)*step_length


def standard_itr_from_confusion_matrix(confusion_matrix, window_length, step_length, n_classes):
    a = accuracy_from_confusion_matrix(confusion_matrix)
    itr = standard_itr_per_trial(a, n_classes)
    p = prediction_probability_from_confusion_matrix(confusion_matrix)
    mdt = mdt_from_prediction_prob(p, window_length, step_length)
    if p == 0:
        return 0
    else:
        return itr*60.0/mdt


def mi_from_confusion_matrix(confusion_matrix, n_classes):
    c_and_p = confusion_matrix/confusion_matrix.sum()
    m = prediction_probability_from_confusion_matrix(confusion_matrix)
    c_and_p = c_and_p/m
    c_and_p = np.delete(c_and_p, n_classes, 1)
    c_and_p = np.delete(c_and_p, n_classes, 0)
    p = c_and_p.sum(axis=0)
    c = c_and_p.sum(axis=1)
    itr = 0
    for i in range(n_classes):
        for j in range(n_classes):
            if c_and_p[i][j] != 0:
                itr += c_and_p[i][j]*np.log2(c_and_p[i][j]/(c[i]*p[j]))
    return itr


def itr_mi_from_confusion_matrix(confusion_matrix, window_length, step_length, n_classes):
    p = prediction_probability_from_confusion_matrix(confusion_matrix)
    mdt = mdt_from_prediction_prob(p, window_length, step_length)
    mi = mi_from_confusion_matrix(confusion_matrix, n_classes)
    return mi*60/mdt


def accuracy_from_confusion_matrix(confusion_matrix):
    return np.trace(confusion_matrix)/(confusion_matrix.sum()-confusion_matrix.sum(axis=0)[-1])


def prediction_probability_from_confusion_matrix(confusion_matrix):
    return (confusion_matrix.sum()-confusion_matrix.sum(axis=0)[-1])/confusion_matrix.sum()


def read_data(subject, recordings, data_folder):
    all_data_for_subject = []
    labels_for_subject = []
    for recording in recordings:
        input_file_name = os.path.join(os.pardir, os.pardir, data_folder, "feature_data", "sub" + subject + "rec" + recording + ".csv")
        data = pd.read_csv(input_file_name)
        features = data.iloc[:, 1:16].to_numpy()
        labels = data["label"].to_numpy()
        labels = np.delete(labels, list(range(113, 120)) + list(range(233, 240)), axis=0)
        features = np.delete(features, list(range(113, 120)) + list(range(233, 240)), axis=0)

        all_data_for_subject.append(features)
        labels_for_subject.append(labels)

    all_data_for_subject = np.array(all_data_for_subject)
    labels_for_subject = np.array(labels_for_subject)
    return all_data_for_subject, labels_for_subject


def calculate_ratios(data_row, n_classes):
    return np.array(
        [data_row[i:i + n_classes] / np.sum(data_row[i:i + n_classes]) for i in range(0, len(data_row), n_classes)]
    ).flatten()


def scale_data(all_data_for_subject, training_data, test_data, n_features, n_classes):
    for i in range(0, n_features, n_classes):
        minf = np.min(all_data_for_subject[:, :, i:i + n_classes])
        maxf = np.max(all_data_for_subject[:, :, i:i + n_classes])
        training_data[:, :, i:i + n_classes] = (training_data[:, :, i:i + n_classes] - minf) / (maxf - minf) + 1
        test_data[:, i:i + n_classes] = (test_data[:, i:i + n_classes] - minf) / (maxf - minf) + 1
    return training_data, test_data


def add_ratios_as_features(training_data, test_data, n_trials, n_samples, n_features, n_classes, add_ratios):
    if add_ratios:
        training_ratios = np.array(list(map(lambda x: calculate_ratios(x, n_classes), training_data.reshape(-1, n_features))))
        test_ratios = np.array(list(map(lambda x: calculate_ratios(x, n_classes), test_data.reshape(-1, n_features))))

        n_new_features = n_features * 2

        training_data = np.concatenate([training_data, training_ratios.reshape((n_trials - 1, n_samples, n_features))], axis=2)
        test_data = np.concatenate([test_data, test_ratios], axis=1)
    else:
        n_new_features = n_features
    return n_new_features, training_data, test_data


def apply_lda(training_data, training_labels, test_data, feature_selector, lda_model, n_new_features, n_trials, do_lda, do_lda_separately):
    if do_lda:
        if do_lda_separately:
            lda_features = []
            for i in range(n_trials - 1):
                lda_training_data = np.delete(training_data, i, 0).reshape(-1, n_new_features)
                lda_training_labels = np.delete(training_labels, i, 0).reshape(-1)
                lda_prediction_data = training_data[i, :, :].reshape(-1, n_new_features)
                feature_selector.fit(lda_training_data, lda_training_labels)
                lda_model.fit(feature_selector.transform(lda_training_data), lda_training_labels)
                lda_features.append(lda_model.decision_function(feature_selector.transform(lda_prediction_data)))
            lda_features = np.array(lda_features)

            new_train_features = lda_features
            feature_selector.fit(training_data.reshape(-1, n_new_features), training_labels.reshape(-1))
            lda_model.fit(feature_selector.transform(training_data.reshape(-1, n_new_features)), training_labels.reshape(-1))
            new_test_features = lda_model.decision_function(feature_selector.transform(test_data.reshape(-1, n_new_features)))
        else:
            lda_training_data = training_data.reshape(-1, n_new_features)
            lda_training_labels = training_labels.reshape(-1)
            feature_selector.fit(lda_training_data, lda_training_labels)
            lda_model.fit(feature_selector.transform(lda_training_data), lda_training_labels)
            new_train_features = lda_model.decision_function(feature_selector.transform(lda_training_data))
            new_test_features = lda_model.decision_function(feature_selector.transform(test_data.reshape(-1, n_new_features)))
        return new_train_features, new_test_features
    else:
        return training_data, test_data


def calculate_skew_norm_params(train_features_given_class, n_classes):
    return np.array([
            [
                skewnorm.fit(train_features_given_class[class_i,:,feature_i])
                for feature_i in range(n_classes)
            ] for class_i in range(n_classes)
        ])


def plot_histogram_and_skew_norm(train_features_given_class, skew_norm_params, n_classes, do_skew_norm_plots):
    if do_skew_norm_plots:
        all_min = np.min(train_features_given_class)
        all_max = np.max(train_features_given_class)
        for class_i in range(n_classes):
            for feature_i in range(n_classes):
                plt.subplot(3, 3, 1 + feature_i + n_classes * class_i)
                plt.hist(train_features_given_class[class_i, :, feature_i], density=True)
                minf = np.min(train_features_given_class[class_i, :, feature_i])
                maxf = np.max(train_features_given_class[class_i, :, feature_i])
                x = np.linspace(minf, maxf)
                plt.plot(x, skewnorm.pdf(x, *skew_norm_params[class_i][feature_i]))
                plt.xlim((all_min, all_max))
                plt.ylim((0, 0.5))
                plt.title("C=" + str(class_i + 1) + " F=" + str(feature_i + 1))
        plt.show()


def plot_3d(bin_edges, result_full_label, n_classes, do_3d_plot):
    if n_classes == 3 and do_3d_plot:
        intervals = []
        for j in range(n_classes):
            intervals.append(
                [(bin_edges[j][0] - (bin_edges[j][1] - bin_edges[j][0]), bin_edges[j][0])] +
                [(l, h) for l, h in zip(bin_edges[j][:-1], bin_edges[j][1:])] +
                [(bin_edges[j][-1], bin_edges[j][-1] + (bin_edges[j][1] - bin_edges[j][0]))]
            )
        intervals = np.array(intervals)

        intervals_all = []
        for i1, v1 in enumerate(intervals[0]):
            for i2, v2 in enumerate(intervals[1]):
                for i3, v3 in enumerate(intervals[2]):
                    intervals_all.append((intervals[0][i1], intervals[1][i2], intervals[2][i3]))
        # (, feature_idx, (low, high))
        intervals_all = np.array(intervals_all)

        plot_location = intervals_all.copy()
        plot_location = plot_location[:, :, 0]

        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        pos = np.transpose(plot_location[np.where(result_full_label == 0)])
        ax.scatter(pos[0], pos[1], pos[2], c='blue')
        pos = np.transpose(plot_location[np.where(result_full_label == 1)])
        ax.scatter(pos[0], pos[1], pos[2], c='red')
        pos = np.transpose(plot_location[np.where(result_full_label == 2)])
        ax.scatter(pos[0], pos[1], pos[2], c='green')
        # for class_i in range(3):
        #     result = ""
        #     for row in plot_location[np.where(result_full_label == class_i)]:
        #         result += " ".join(map(lambda x: str(np.round(x, 2)), row)) + "\n"
        #     open("3d" + str(class_i) + ".txt", "w").write(result)
        plt.show()


def train_classifier(alpha, beta, probas, n_classes):
    ds = IB.dataset(pxy=probas)

    fit_param = pd.DataFrame(data={'alpha': [alpha], 'zeroLtol': float("inf"), 'Tmax': 3, 'clamp': False,
                                   'betas': [beta], 'beta_search': False, 'cthresh': 5})
    fit_param['repeats'] = 1
    metrics_conv, dist_conv, metrics_sw, dist_sw = IB.IB(ds, fit_param, keep_steps=False, sw_dist_to_keep={})

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

    return result_full_label


def discretise_features(train_features_given_class, bin_edges, skew_norm_params, n_classes, use_skew):
    if use_skew:
        discretised = []
        for i, row in enumerate(skew_norm_params):
            discretised_row = []
            for j, param in enumerate(row):
                cdf_values = []
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
    else:
        discretised = [[[0] + list(np.histogram(train_features_given_class[i,:,j], bins=bin_edges[j])[0]) + [0] for j in range(n_classes)] for i in range(n_classes)]
        discretised = [[discretised[i][j] / np.sum(discretised[i][j]) for j in range(n_classes)] for i in range(n_classes)]
    return discretised


def calculate_feature_probabilities(discretised, n_classes):
    probas = []
    for i, row in enumerate(discretised):
        proba_row = np.round(np.prod(np.stack(np.meshgrid(*row, indexing="ij"), -1).reshape(-1, n_classes), axis=1) / n_classes, 8)
        probas.append(proba_row)
    probas = np.transpose(probas)
    return probas / probas.sum()


def find_label_permutation(classifier, train_features_given_class, training_labels, bin_edges, shift, thresh, n_classes):
    predicted_training_labels = list(map(lambda x: predict(bin_edges, classifier + 1, shift, thresh, x, n_classes), train_features_given_class.reshape(-1, n_classes)))
    training_confusion_matrix = sklearn.metrics.confusion_matrix(
        training_labels.reshape(-1),
        predicted_training_labels,
        labels=[i+1 for i in range(n_classes)] + [0]
    )
    permutations = list(itertools.permutations([i for i in range(n_classes)]))
    accuracies = [accuracy_from_confusion_matrix(training_confusion_matrix[:,list(perm) + [n_classes]]) for perm in permutations]
    best_accuracy = np.argmax(accuracies)
    return list(permutations[best_accuracy]) + [n_classes]


def calculate_bin_edges(train_features_given_class, method, n_classes, n_bins):
    min_features_f = np.min(np.min(train_features_given_class, axis=0), axis=0)
    max_features_f = np.max(np.max(train_features_given_class, axis=0), axis=0)
    if n_bins is None:
        n_edges = np.array([int(np.round(np.mean([len(np.histogram_bin_edges(train_features_given_class[i, :, j], method, (min_features_f[j], max_features_f[j]))) for i in range(n_classes)]))) for j in range(n_classes)])
    else:
        n_edges = np.array([n_bins + 1] * n_classes)
    bin_edges = [np.linspace(min_features_f[f], max_features_f[f], n_edges[f]) for f in range(n_classes)]
    return n_edges, bin_edges


