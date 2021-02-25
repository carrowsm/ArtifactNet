# ArtifactNet: A 3-Dimensional Convolutional Neural Network for Dental Artifact Removal
Metal implants in computed tomography (CT) scans can cause large streak artifacts which obscure large portions of images. These artifacts are particularly common in the mouth where dental implants and fillings cause dental artifacts (DA) which obscure structures in the head and neck, negatively impacting the quality of imaging for head and neck cancers.

ArtifactNet is an open-source software package for training neural networks to remove dental artifacts (DAs) from CT scans. The models in this package use PyTorch and the training scripts use the [PyTorch-Lightning framework](https://github.com/PyTorchLightning/pytorch-lightning).

The main model in the package is a 3-dimensional adaptation of the [cycleGAN](https://junyanz.github.io/CycleGAN/) model for unpaired image-to-image translation. This means that ArtifactNet requires a set of 3D CT scans with artifacts and a set without artifacts for training, where each set can consist of a different cohort of patients. Other architectures like [pix2pix](https://phillipi.github.io/pix2pix/), often used in medical image domain translation require pairs of images from the same patient.

## Usage
This package can be used to train three different networks:
### Pix2Pix for artificial DA removal
The [pix2pix](https://phillipi.github.io/pix2pix/) network can be trained by running the pix2pix.py script with paired data.

### 2-Dimensional CycleGAN
We attempted to reproduce the results from [Nakao et al.](https://arxiv.org/pdf/1911.08105.pdf) who used the [cycleGAN](https://junyanz.github.io/CycleGAN/) architecture to remove DAs from 3D CT scan volumes. This approach did not use a truly 3D CNN, but a 2D CNN with the z-axis given as input channels to the network (analogous to colour channels in a 2D imageNet CNN).

### 3-Dimensional CycleGAN
We propose a novel 3D cycleGAN architecture to remove artifacts with convolutions along all three axes.
