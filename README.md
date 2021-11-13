
# Information Bottleneck as Optimisation Method for SSVEP-Based BCI

## Introduction

This repository contains code for applying information bottleneck to optimise steady-state visual evoked potential (SSVEP) based brain computer interfaces (BCIs). This approach should work in other types of BCIs as well. Information bottleneck is information-theoretic optimisation method for solving specific optimisation task. Here, information bottleneck is used to find an optimal classification rule for a BCI. Optimality is viewed in terms of the standard performance measure for BCIs, the information transfer rate (ITR).

The algorithm is introduced in the [article](https://doi.org/10.3389/fnhum.2021.675091): Anti Ingel and Raul Vicente. "Information Bottleneck as Optimisation Method for SSVEP-Based BCI".  Frontiers in Human Neuroscience 15 (2021). Please cite this article when using the code.

## Requirements

The code is written in Python and has been tested with Python 3.7. Running the code requires packages `numpy`, `scipy`, `sklearn`, `pandas`, `matplotlib`. Code for calculating information bottleneck is needed to run the algorithm. Please download `IB.py` file from [this repository](https://github.com/antiingel/information-bottleneck) and add it to the `src` folder before trying to run the code.

## Getting started

The repository contains code for running the algorithm on two different datasets.

### Running the algorithm on Dataset 1

Before trying to run the code, please download the required dataset from [here](https://drive.google.com/drive/folders/12Wu6377sfgYZ2qpOw_WUtgVYO97maDlu?usp=sharing). This dataset contains only the extracted features. The original dataset from which these features are calculated can be found [here](http://www.bakardjian.com/work/ssvep_data_Bakardjian.html), but this is not necessary to run the code.

Extract `data` folder to the same folder that contains `src`. Now optimisation procedure can be run using the script `experiments1/4_optimise_itr.py`.

Downloaded features can also be calculated from the original dataset. For that download the original dataset and put data into `data/original_data` folder. The extracted features can be calculated from the original data by running the files `1_original_data_to_csv.py`, `2_generate_eeg_data.py`, `3_generate_feature_data.py`.

### Running the algorithm on Dataset 2

Before trying to run the code, please download the required dataset from [here](https://drive.google.com/file/d/1JmVFwb75GlzH9zGUZxixpxQ17SinOPQX/view?usp=sharing). This dataset contains only the extracted features. The original dataset from which these features are calculated can be found [here](http://bci.med.tsinghua.edu.cn/download.html), but this is not necessary to run the code.

Extract `data2` folder to the same folder that contains `src`. Now optimisation procedure can be run using the script `experiments2/optimise_itr.py`.

Downloaded features can also be calculated from the original dataset. For that code from [this repository](https://github.com/mnakanishi/TRCA-SSVEP) was used.
