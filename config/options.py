from argparse import ArgumentParser


### EDIT THESE PATHS ###
img_dir = "/cluster/home/carrowsm/data/Simulated_DA/"
log_dir = "/cluster/home/carrowsm/logs/artifact_net/remove/"
### ---------------- ###

parser = ArgumentParser()

# Paths to data and logs
parser.add_argument("--csv_path", type=str, default="",
help="Path to the CSV containing all image DA statuses and DA locations")
parser.add_argument("--img_dir", default=img_dir, type=str, help="Path to the input image data.")
parser.add_argument("--log_dir", default=log_dir, type=str, help='Where to save results.')


### Hyperparams for model training ###
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--augmentation_factor", type=int, default=1,
                     help="Factor by which to augment the data with random transforms.")


# Helper function to get arguments in other scripts
def get_args():
    args, unparsed = parser.parse_known_args()

    return args, unparsed
