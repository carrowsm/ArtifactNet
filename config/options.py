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
parser.add_argument("--max_num_epochs", type=int, default=100,
                    help="Maximum number of epochs to train for.")

### Model selection and training parameters ###
parser.add_argument("--half_precision", type=bool, default=False,
                    help="If true 16-bit precision will be used. Default is False (use 32-bit)")
parser.add_argument("--architecture", type=str, default="cycleGAN_3D",
                    help="'cycleGAN_2D', 'cycleGAN_3D', or 'pix2pix' are accepted.")


parser.add_argument("--n_gpus", type=int, default=1,
                    help="Number of GPUs to use for training")

# Helper function to get arguments in other scripts
def get_args():
    args, unparsed = parser.parse_known_args()

    return args, unparsed
