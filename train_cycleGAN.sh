#!/bin/bash
#SBATCH -t 0-20:00:00
#SBATCH --mem=200G
#SBATCH -J ArtifactNet
#SBATCH -c 10
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:4
#SBATCH --output=DA_removal_GAN.out
#SBATCH --ntasks-per-node=1


echo 'Starting Shell Script'

source /cluster/home/carrowsm/.bashrc
conda activate DAnet

# Python script we are running
path=/cluster/home/carrowsm/ArtifactNet/gan.py

# Paths to data and logs
csv_path = "/cluster/home/carrowsm/data/radcure_DA_labels.csv"
img_path = "/cluster/projects/radiomics/Temp/RADCURE-npy/img"
log_path = "/cluster/home/carrowsm/logs/artifact_net/remove/cycleGAN"

# Hyperparameters for training the model
epochs=50                                # Number of epochs for training
learn_rate=0.0002                        # Initial rate for the trainer
batch_size=1                             # Batch size for trainer
aug_factor=1


echo 'Started python script.'
python $path --csv_path=$csv_path --img_dir=$img_path --log_dir=$log_path --batch_size=$batch_size --lr=$learn_rate --augmentation_factor=$aug_factor
echo 'Python script finished.'
