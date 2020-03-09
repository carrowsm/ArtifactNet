from argparse import ArgumentParser


### EDIT THESE PATHS ###
# Remote
img_dir = "/cluster/home/carrowsm/data/Simulated_DA/"
img_suffix = ".npy"                 # This string follows the patient ID in the filename
log_dir = "/cluster/home/carrowsm/logs/artifact_net/remove/"

# Local
# img_path = "/home/colin/Documents/BHKLab/data/Artifact_Net_Training/trg_data"
# img_suffix = "_img.npy"                 # This string follows the patient ID in the filename
# log_dir = "/home/colin/Documents/BHKLab/data/Artifact_Net_Training/logs"
### ---------------- ###

parser = ArgumentParser()

parser.add_argument("--csv_path", type=str, default="",
help="Path to the CSV containing all image DA statuses and DA locations")
parser.add_argument("--img_dir", default=img_path, type=str, help="Path to the input image data.")
parser.add_argument("--img_suffix", default=img_suffix, type=str)

parser.add_argument("--data_type", default="phantom", type=str,
help="'phantom' if we are using artificial input images, 'real' if we are using patient CT scans.")
parser.add_argument("--log_dir", default=log_dir, type=str, help='Where to save results.')


### Args for model training ###
parser.add_argument("--batch_size", type=int, default=5, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=None,
                    help="Dimension of the image that it given to the generator. \
 By default, the dataloader image size will be used.")

parser.add_argument("--augmentation_factor", type=int, default=1,
                     help="Factor by which to augment the data with random transforms.")


# Helper function to get arguments in other scripts
def get_args():
    args, unparsed = parser.parse_known_args()

    return args, unparsed
