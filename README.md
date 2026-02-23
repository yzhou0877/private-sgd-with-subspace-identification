# Bypassing the Ambient Dimension: Private SGD with Gradient Subspace Identification

Official implementation of the paper **“Bypassing the Ambient Dimension: Private SGD with Gradient Subspace Identification.”**

## Overview

This repository contains the Python implementation of the proposed DPD-SGD algorithm with gradient subspace identification. Experiments are provided for both MNIST and Fashion-MNIST datasets.

## Environment Setup

Create a virtual environment using:

```bash
conda env create -f environment.yml
```

## Running Experiments

### Fashion-MNIST

Main file:
```
fashion-mnist-cnn/run_nn_fmnist_sgd_proj.py
```

Run DPD-SGD with:

```bash
python3 run_nn_fmnist_sgd_proj.py \
    --stdev 18 \
    --batch-size 250 \
    --lr 0.01 \
    --micro-size 5 \
    --proj-dim 70 \
    --num-valid 100 \
    --proj-epoch 15
```

### MNIST

Main file:
```
mnist-cnn/run_nn_mnist_sgd_proj.py
```

Run DPD-SGD with:

```bash
python3 run_nn_mnist_sgd_proj.py \
    --stdev 14 \
    --batch-size 250 \
    --lr 0.2 \
    --micro-size 1 \
    --proj-dim 50 \
    --num-valid 100 \
    --proj-epoch 0
```

## Parameter Description

- `--stdev`: Standard deviation of Gaussian noise  
- `--batch-size`: Training batch size  
- `--lr`: Learning rate  
- `--micro-size`: Micro-batch size  
- `--proj-dim`: Projection dimension (k in DPD-SGD)  
- `--num-valid`: Size of the public validation dataset  
- `--proj-epoch`: Epoch number to start projection  


