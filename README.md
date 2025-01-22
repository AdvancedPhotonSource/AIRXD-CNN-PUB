# AIRXD-CNN

This repository contains a lightweight convolutional neural network designed for the identification of artifacts in x-ray images. The sister repo, AIRXD, is available at: https://github.com/AdvancedPhotonSource/AIRXD-ML-PUB

---

## Table of Contents
1. [Installation](#installation)
2. [Usage](#usage)
3. [Examples](#examples)

---

## Installation

This set of steps should require just a single .yml file, but there were pytorch consistency issues that were run into when dependencies for all the packages were being solved simultaneously. We therefore need to install the deep learning packages (pytorch, torchvision, dlsia) first in a specific order before filling in the rest of the packages.

### Prerequisites
Ensure you have the following installed:
- [Miniconda/Anaconda](https://docs.conda.io/projects/conda/en/latest/user-guide/install/index.html)
- Python 3.8+ (compatible with the provided environment file)

### Setting Up the Environment
1. Clone this repository:
   ```bash
   git clone https://github.com/yourusername/xray-artifact-cnn.git
   cd xray-artifact-cnn

2. Create conda environment from scratch using python 3.11:
   ```bash
    conda env create -n airxd_cnn python=3.11

3. Activate environment:
   ```bash
    conda activate airxd_cnn

4. Manually install pytorch first (this was causing initial issues with the yml file):
   ```bash
   conda install pytorch torchvision pytorch-cuda=11.8 -c pytorch -c nvidia

5. Manually install dlsia
   ```bash
   pip install dlsia

6. Update remaining dependencies with .yml file
   ```bash
   conda env update --file minimalist_env.yml

7. You're done! Everything should be ready to use.
---

## Usage

The original model can be trained with some variant of vanilla_train.py. All hyperparameter sweeps were done by building on top of this initial training code.
Currently, the training hyperparameters are set at the optimal numbers we found (see manuscript).

To run, simply use: `python vanilla.py`.

NOTE: You need pretty high local memory requirements to run this due to the size of all the image patches (> 100 GB). For a more lightweight implementation, see below.

---

## Examples

Use the notebook `example_training.ipynb` to go through the different steps of the training process, including the mutual information pruning and island masking steps. There is a secondary lightweight neural network that doesn't have nearly has high memory requirements that can achieve similar levels of performance. You can also skip straight ahead to Steps 3-4 in the notebook to use our best-trained models to evaluate the test set.
