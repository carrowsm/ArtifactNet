# DAnet: A 3-Dimensional Convolutional Neural Network for Dental Artifact Removal


DAnet is an open-source software package for training neural networks to remove dental artifacts (DAs) from CT scans. The models in this package use PyTorch and the training scripts use the [PyTorch-Lightning framework](https://github.com/PyTorchLightning/pytorch-lightning).

## Usage
This package is designed to train three different networks.
### Pix2Pix for artificial DA removal
The [pix2pix](https://phillipi.github.io/pix2pix/) network can be trained by running the pix2pix.py script with paired data.

### 2-Dimensional CycleGAN
We attempted to reproduce the results from [Nakao et al.](https://arxiv.org/pdf/1911.08105.pdf) who used the [cycleGAN](https://junyanz.github.io/CycleGAN/) architecture to remove DAs from 3D CT scan volumes. This approach did not use a truly 3D CNN, but a 2D CNN with the z-axis given as input channels to the network (analogous to colour channels in a 2D imageNet CNN).

### 3-Dimensional CycleGAN
We propose a novel 3D cycleGAN architecture to remove artifacts with convolutions along all three axes.
