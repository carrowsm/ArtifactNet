#!/bin/bash
#SBATCH -t 1-00:00:00
#SBATCH --mem=100G
#SBATCH -J ArtifactNet
#SBATCH -c 33
#SBATCH -N 1
#SBATCH --account=radiomics_gpu
#SBATCH --partition=gpu_radiomics
#SBATCH --gres=gpu:v100:1
#SBATCH --output=strongORweak_to_none.out
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
img_path="/cluster/projects/radiomics/RADCURE-images/"
log_path="/cluster/home/carrowsm/logs/cycleGAN"
cache_path="/cluster/projects/radiomics/Temp/colin/cyclegan_data/2-1-1mm_nrrd/"

### EDIT BELOW ###
# Hyperparameters for training the model
learn_rate=0.0002                       # Initial rate for the trainer
batch_size=10                           # Batch size for trainer
num_gpus=1                              # Number of GPUs to use for training
num_cpus=32                             # Number of workers for dataloaders
num_filters=32                          # Number of input filters for the model
### ---------- ###

echo 'Started python script.'
python $path \
--csv_path $csv_path \
--img_dir $img_path \
--log_dir $log_path \
--cache_dir $cache_path \
--batch_size $batch_size \
--lr $learn_rate \
--n_gpus $num_gpus \
--n_cpus $num_cpus \
--n_filters $num_filters \
--image_size 8 256 256 \
--img_domain_x 2 1 \
--img_domain_y 0 \
--cnn_layers 3
echo 'Python script finished.'
