import scipy.io
import numpy as np
import pandas as pd
import os
from itr_optimisation import ModelTrainer2

frequencies = [str(i+1) for i in range(40)]
subjects = [str(i+1) for i in range(35)]
recordings = [str(i+1) for i in range(6)]
electrodes = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
# 1: Pz, 2: PO5,3: PO3, 4: POz, 5: PO4, 6: PO6, 7: O1, 8: Oz, and 9: O2

feature_names = ["1_('O1', 'O2')_CCA_('O1', 'O2')_Sum_" + str(frequency) for frequency in frequencies]

n_classes = 3

for subject in subjects:
    print("\nOptimising for subject " + str(subject))
    all_data_for_subject = []
    labels_for_subject = []
    for recording in recordings:
        input_file_name = os.path.join(os.pardir, os.pardir, "dataset2", "S" + subject + "_cv_" + recording + "_ens_0.mat")
        data = scipy.io.loadmat(input_file_name)["rhos"]  # (22, 40, 40) = (sample, correct_class, class_feature)
        n_samples = data.shape[0]
        data = np.concatenate(tuple(data[:,c,:n_classes] for c in range(n_classes)))
        labels = np.concatenate([[j] * n_samples for j in range(n_classes)])
        features_dictionary = [{key: value for key, value in zip(feature_names, feature_vector)} for feature_vector in data]

        all_data_for_subject.append(features_dictionary)
        labels_for_subject.append(labels)

    trainer = ModelTrainer_40_rhos.ModelTrainer()
    trainer.setup(
        1,
        feature_names,
        all_data_for_subject,
        labels_for_subject,
        n_classes
    )
    trainer.start(subject, make_plots=False)
    # raw_input("\nPress enter to start optimising for next subject")
