<div align="center">
<h2>LLM-as-a-Coauthor: Can Mixed Human-Written and Machine-Generated Text Be Detected?</h2>

[![Paper](https://img.shields.io/badge/Paper-%F0%9F%8E%93-lightgrey?style=flat-square)](https://arxiv.org/abs/2401.05952) [![Dataset](https://img.shields.io/badge/Dataset-%F0%9F%92%BE-green?style=flat-square)](https://huggingface.co/datasets/shuaishuaicdp/MixSet)

<img src="https://img.shields.io/github/last-commit/Dongping-Chen/MixSet?style=flat-square&color=5D6D7E" alt="git-last-commit" />
<img src="https://img.shields.io/github/commit-activity/m/Dongping-Chen/MixSet?style=flat-square&color=5D6D7E" alt="GitHub commit activity" />
<img src="https://img.shields.io/github/languages/top/Dongping-Chen/MixSet?style=flat-square&color=5D6D7E" alt="GitHub top language" />


<img src="figures/outline.jpg">
<img src="figures/self_bleu.jpg">
<p align="center">

</p>
</div>

## Updates & News
- [14/04/2024] ⭐ We revise the data structure and update huggingface version of our dataset. 
- [15/03/2024] 🔥 MixSet is accepted to NAACL'24 Findings!
- [11/01/2024] 🌊 Our [paper](https://arxiv.org/abs/2401.05952) and [dataset](https://huggingface.co/datasets/shuaishuaicdp/MixSet) are released! 

## Table of content
- [Dataset: MixSet](#dataset-mixset)
  - [Overview](#overview)
    - [Dataset Location](#dataset-location)
- [Usage](#usage)
  - [Benchmark your MGT/MixText detector](#benchmark-your-mgtmixtext-detector)
  - [Experiment Reproduce](#experiment-reproduce)
    - [Prerequisites](#prerequisites)
    - [Installation](#installation)
    - [Experiment 1](#experiment-1)
    - [Experiment 2](#experiment-2)
    - [Experiment 3](#experiment-3)
      - [Storage Requirements for Experiments 3 and 4 Scripts](#storage-requirements-for-experiments-3-and-4-scripts)
    - [Experiment 4](#experiment-4)
    - [Script Parameters Description](#script-parameters-description)
- [Contact](#contact)
- [Acknowledgments](#acknowledgments)
- [Citation](#citation)
# Dataset: MixSet

## Overview
The MixSet dataset is a comprehensive collection designed for advanced Machine Learning experiments. It's structured to support a variety of tasks including MGT classification in the era of LLMs, natural language understanding, and more.

### Dataset Location
The dataset is located in the `./data/MixSet/` directory relative to the project's root. Ensure that this path exists and contains the necessary data files before running any scripts that depend on the MixSet dataset.

# Usage

## Benchmark your MGT/MixText detector
Please refer to `./data/MixSet/README.md` for our MixSet data structure and how to leverage our dataset with ease.

## Experiment Reproduce

### Prerequisites

- Python = 3.9
- Other dependencies specified in `requirements.txt`
  
### Installation
To set up your environment to run the code, follow these steps:

1. **Clone the Repository:**

```shell
git clone https://github.com/Dongping-Chen/MixSet.git
cd MixSet
```

2. **Create and Activate a Virtual Environment (optional but recommended) and Install the Required Packages:**

```shell
conda create --name mixset python=3.9
conda activate mixset
pip install -r requirements.txt
```

3. **Download Datasets**
To download the pure MGT datasets, please refer to [this link](https://1drv.ms/u/s!AivM2GUMbPYyjkgx6us826N6_j2P?e=yriuqR), then move the dataset folders to `<YOUR PATH>/MixSet/data/MGT_datasets/`.
To download the pure HWT datasets, please refer to [this link](https://1drv.ms/u/s!AivM2GUMbPYyjkl-aRs1m_l5X9kW?e=xu2QjU), then move the dataset folders to `<YOUR PATH>/MixSet/data/pure_processed_HWT/`.

4. **Download Checkpoints of GPT-Sentinel**
Download the pre-trained [GPT-Sentinel t5-small](https://1drv.ms/u/s!AivM2GUMbPYyjkqfT_3Ri-fpnifX?e=eqG1t7) and put to `<YOUR PATH>/MixSet/`.


### Experiment 1
To reproduce the first experiments, run:
```shell
./Ex1_run.sh
```
You should run GPT-Zero by:
```shell
./Ex1_run_GPTzero
```
As for [Ghostbuster](https://github.com/vivek3141/ghostbuster), we will update the code as soon as possible.

### Experiment 2
To reproduce the second experiment for binary classification, run:
```shell
./Ex2_binary_run
```
To reproduce the second experiment for three-class classification, run:
```shell
./Ex2_three_class_run
```


### Experiment 3
To reproduce the third experiment for operation-wise transfer learning, run:
```shell
./Ex3_operation_train.sh
./Ex3_operation_test.sh
```
To reproduce the third experiment for LLM-wise transfer learning, run:
```shell
./Ex3_LLM_transfer.sh
```
#### Storage Requirements for Experiments 3 and 4 Scripts

Please be aware that the scripts for Experiments 3 and 4 require storing trained checkpoints in the folder path. This may occupy more than 20GB of space. It is essential to ensure that you have sufficient storage available on your device. Failing to allocate the necessary space might lead to interruptions during the code execution. We highly recommend checking and freeing up adequate space before running these scripts to ensure a smooth and uninterrupted experience.

### Experiment 4
To reproduce the fourth experiment for the ablation study, run:
```shell
./Ex4_auto_train.sh
./Ex4_auto_test.sh
```

### Script Parameters Description

Below are the parameters used in the script along with their descriptions:

- `--Mixcase_filename`: Specifies the filename for the MixText data. Default is `None`.
- `--MGT_only_GPT`: If set, the script will only use MGT (Model Generated Text) from GPT-family models.
- `--test_only`: If set, the script will only perform testing, skipping any training procedures.
- `--train_threshold`: Specifies the threshold for training. Default is `10000`.
- `--no_auc`: If set, the script will only calculate the MixText scenarios, which means no Area Under the ROC Curve (AUC) metrics.
- `--only_supervised`: If set, the script will perform only supervised learning without any unsupervised techniques.
- `--train_with_mixcase`: If set, the script will include MixText data in the training process.
- `--seed`: Sets the seed for random number generation to ensure reproducibility. Default is `0`.
- `--ckpt_dir`: Specifies the directory to save checkpoints. Default is `"./ckpt"`.
- `--log_name`: Specifies the name of the log file. Default is `'Log'`.
- `--mixcase_threshold`: Sets the threshold for considering data as MixText. Default is `0.8`.
- `--transfer_filename`: Specifies the filename for transfer learning. Default is `None`.
- `--three_classes`: If set, the script will use a three-class classification scheme instead of binary classification.
- `--finetune`: If set, the script will fine-tune the supervised model.
- `--mixcase_as_mgt`: If set, MixText data will be treated as Model Generated Text (MGT).

# Contact
For any issues, questions, or suggestions related to the MixSet dataset, feel free to contact [me](mailto:dongpingchen0612@gmail.com) or open an issue in the project's repository.

# Acknowledgments
Part of the code is borrowed from [MGTBench](https://github.com/xinleihe/MGTBench).
The corresponding author [Lichao Sun](james.lichao.sun@gmail.com) is supported by the National Science Foundation Grants CRII-2246067.

# Citation

```
@misc{zhang2024llmasacoauthor,
      title={LLM-as-a-Coauthor: Can Mixed Human-Written and Machine-Generated Text Be Detected?}, 
      author={Qihui Zhang and Chujie Gao and Dongping Chen and Yue Huang and Yixin Huang and Zhenyang Sun and Shilin Zhang and Weiye Li and Zhengyan Fu and Yao Wan and Lichao Sun},
      year={2024},
      eprint={2401.05952},
      archivePrefix={arXiv},
      primaryClass={cs.CL}
}
```