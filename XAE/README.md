# XAE

## Introduction

This is the implementation of auto-encoder based generative models in PyTorch.

Currently, MMD and GAN version of [Wasserstein Auto-Encoders](https://arxiv.org/abs/1711.01558) is implemented.

## Requirement

* python 3
* PyTorch >= 1.9
* torchvision

## Train Step

Sample running code is in `exp_mnist` directory. Architectecture of a model is defined in `train.py`. CUDA_VISIBLE_DEVICES is defined in `run.sh`. Other configurations are defined in `config/train_config.cfg`.

To run the model in Linux terminal, run `run.sh`. You can also train the model with Jupyter notebook at `train.ipynb`.

## Development

To add new models, define new classes in `model.py` inheriting classes in `_base_model.py`.
