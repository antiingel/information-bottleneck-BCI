from experiments1.itr_optimisation import *

from sklearn.feature_selection import SelectFpr
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis


subjects = [str(i+1) for i in range(4)]
recordings = [str(i+1) for i in range(5)]

np.random.seed(200)

n_classes = 3
window_length = 1
step_length = 0.125

n_bins = None
shift_and_thresh = {"1": (2, 125), "2": (4, 668), "3": (4, 680), "4": (4, 710)}

do_feature_selection = False
add_ratios = False
do_lda = True
do_lda_separately = True
do_skew_norm_plots = False

treat_as_online_signal = True

use_skew = True
do_3d_plot = False


for subject in subjects:

    all_data_for_subject, labels_for_subject = read_data(subject, recordings, "data")

    n_trials, n_samples, n_features = all_data_for_subject.shape

    assert n_classes == len(np.unique(labels_for_subject))

    lda_model = LinearDiscriminantAnalysis()

    feature_selector = SelectFpr(alpha=5e-2 if do_feature_selection else 1)

    shift, thresh = shift_and_thresh[subject]

    accuracies = []
    mdts = []
    prediction_counts = []
    standard_itrs = []
    mi_itrs = []
    prediction_count = []

    for cv_index in range(n_trials):

        training_data = np.delete(all_data_for_subject, cv_index, 0).copy()
        test_data = all_data_for_subject[cv_index,:,:].copy()

        training_labels = np.delete(labels_for_subject, cv_index, 0)
        test_labels = labels_for_subject[cv_index,:]

        training_data, test_data = scale_data(all_data_for_subject, training_data, test_data, n_features, n_classes)

        n_new_features, training_data, test_data = add_ratios_as_features(training_data, test_data, n_trials, n_samples, n_features, n_classes, add_ratios)

        new_train_features, new_test_features = apply_lda(training_data, training_labels, test_data, feature_selector, lda_model, n_new_features, n_trials, do_lda, do_lda_separately)

        train_features_given_class = np.array([new_train_features.reshape(-1, n_classes)[np.where(training_labels.reshape(-1)==i+1)] for i in range(n_classes)])
        training_labels = np.array([training_labels.reshape(-1)[np.where(training_labels.reshape(-1)==i+1)] for i in range(n_classes)])

        skew_norm_params = calculate_skew_norm_params(train_features_given_class, n_classes)

        plot_histogram_and_skew_norm(train_features_given_class, skew_norm_params, n_classes, do_skew_norm_plots)

        n_edges, bin_edges = calculate_bin_edges(train_features_given_class, "fd", n_classes, n_bins)

        discretised_features = discretise_features(train_features_given_class, bin_edges, skew_norm_params, n_classes, use_skew)

        feature_probas = calculate_feature_probabilities(discretised_features, n_classes)

        classifier = train_classifier(1, 100, feature_probas, n_classes)

        plot_3d(bin_edges, classifier, n_classes, do_3d_plot)

        classifier = classifier.reshape(tuple(n_edges + 1))

        best_perm = find_label_permutation(classifier, train_features_given_class, training_labels, bin_edges, shift, thresh, n_classes)

        mi_itr, standard_itr, accuracy, mdt, prediction_count = evaluate_performance(
            new_test_features, classifier, bin_edges, best_perm, shift, thresh, test_labels, n_classes, window_length, step_length, n_samples, treat_as_online_signal
        )

        accuracies.append(accuracy)
        prediction_counts.append(prediction_count)
        mi_itrs.append(mi_itr)
        standard_itrs.append(standard_itr)
        mdts.append(mdt)

    print("Results for subject " + str(subject) + ":")
    print(np.mean(accuracies), accuracies)
    print(np.mean(mdts), mdts)
    print(np.mean(mi_itrs), mi_itrs)
    print(np.mean(standard_itrs), standard_itrs)
    print(np.mean(prediction_counts), prediction_counts)
