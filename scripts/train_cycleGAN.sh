#!/bin/bash
#SBATCH -t 0-12:00:00
#SBATCH --mem=220G
#SBATCH -J ArtifactNet
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:1
#SBATCH --output=DA_removal_GAN_3D.out
#SBATCH --ntasks-per-node=1


echo 'Starting Shell Script'

source /cluster/home/carrowsm/.bashrc
conda activate DAnet

# Change working directory to top of module
cd ../

# Python script we are running
path=/cluster/home/carrowsm/ArtifactNet/cycleGAN.py

# Paths to data and logs
csv_path="/cluster/home/carrowsm/ArtifactNet/datasets/train_labels.csv"
# img_path="/cluster/projects/radiomics/Temp/RADCURE-npy/img"
img_path="/cluster/projects/radiomics/Temp/colin/isotropic_npy/images"
log_path="/cluster/home/carrowsm/logs/artifact_net/remove/cycleGAN"

# Hyperparameters for training the model
epochs=50                                # Number of epochs for training
learn_rate=0.0008                        # Initial rate for the trainer
batch_size=1                             # Batch size for trainer
aug_factor=1                             # Number of times to augment each image
num_gpus=1                               # Number of GPUs to use for training

echo 'Started python script.'
python $path \
--csv_path=$csv_path \
--img_dir=$img_path \
--log_dir=$log_path \
--batch_size=$batch_size \
--lr=$learn_rate \
--augmentation_factor=$aug_factor \
--n_gpus=$num_gpus
echo 'Python script finished.'
