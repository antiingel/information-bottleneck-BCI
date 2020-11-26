

## Introduction

TODO

## Experiments on Dataset 1

Before trying to run the code, please download the required dataset from [here](https://drive.google.com/file/d/1JmVFwb75GlzH9zGUZxixpxQ17SinOPQX/view?usp=sharing). This dataset contains only the extracted features. The original dataset from which these features are calculated can be found [here](http://www.bakardjian.com/work/ssvep_data_Bakardjian.html).

Extract `dataset1` folder to the same folder that contains `src`. Now optimisation procedure can be run using the script `experiments1/4_optimise_itr`. 

Downloaded features can also be calculated from the original dataset. For that download the original dataset and put data into `dataset1/original_data` folder. Then

1. `experiments1/1_original_data_to_csv.py` can be used to generate files into `original_data_as_csv` folder from the original files in `MATLAB` format in folder `original_data`.
2. `experiments1/2_generate_eeg_data.py` can be used to generate files into `eeg_data` folder from the files in folder `original_data_as_csv`.
3. `experiments1/3_generate_feature_data.py` can be used to generate files in `feature_data` folder from the files in folder `eeg_data`.


## Experiments on Dataset 2

Before trying to run the code, please download the required dataset from [here](https://drive.google.com/file/d/1JmVFwb75GlzH9zGUZxixpxQ17SinOPQX/view?usp=sharing) if you have not already. This dataset contains only the extracted features. The original dataset from which these features are calculated can be found [here](http://bci.med.tsinghua.edu.cn/download.html).

Extract `dataset2` folder to the same folder that contains `src`.
We also need code for calculating information bottleneck. For that you need code from [here](https://github.com/djstrouse/information-bottleneck). Download `IB.py` from this repository and put this file into `src` folder.

Now optimisation procedure can be run using the script `experiments2/optimise_itr`. 

Downloaded features can also be calculated from the original dataset. For that we used code from [here](https://github.com/mnakanishi/TRCA-SSVEP).
