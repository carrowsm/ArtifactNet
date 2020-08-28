from argparse import ArgumentParser


### EDIT THESE PATHS ###
img_dir = "/cluster/home/carrowsm/data/Simulated_DA/"
log_dir = "/cluster/home/carrowsm/logs/artifact_net/remove/"
cache   = "/cluster/projects/radiomics/Temp/colin/isotropic_nrrd/"
### ---------------- ###

parser = ArgumentParser()

# Paths to data and logs
parser.add_argument("--csv_path", type=str, default="",
                    help="Path to the CSV containing all image DA statuses and DA locations")
parser.add_argument("--img_dir", default=img_dir, type=str, help="Path to the input image data.")
parser.add_argument("--log_dir", default=log_dir, type=str, help='Where to save results.')
parser.add_argument("--cache_dir", default=cache, type=str, help="Where to cache images for training.")

### Hyperparams for model training ###
parser.add_argument("--batch_size", type=int, default=1, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--weight_decay", type=float, default=1.0e-4, help="Adam: decay of weights.")
parser.add_argument("--augmentation_factor", type=int, default=1,
                     help="Factor by which to augment the data with random transforms.")
parser.add_argument("--max_num_epochs", type=int, default=100,
                    help="Maximum number of epochs to train for.")
parser.add_argument("--n_gpus", type=int, default=1,
                    help="Number of GPUs to use for training")
parser.add_argument("--n_cpus", type=int, default=1,
                    help="Number of parallel workers to use for data loading.")

### Model selection and training parameters ###
parser.add_argument("--half_precision", type=bool, default=False,
                    help="If true 16-bit precision will be used. Default is False (use 32-bit)")
parser.add_argument("--architecture", type=str, default="cycleGAN_3D",
                    help="'cycleGAN_2D', 'cycleGAN_3D', or 'pix2pix' are accepted.")
parser.add_argument("--n_filters", type=int, default=32,
                    help="Number of input filters to use in generator and discriminator.")
parser.add_argument("--cnn_layers", type=int, default=3,
                    help="Number of convolutional layers for the discriminator.")
parser.add_argument("--image_size", type=int, nargs="*", default=[16,256,256],
                    help="The size of the image in pixels. If a list of length 3, the image will \
be 3D with shape [z, y, x]. If a list of length 2, the image with be 2D with shape [y, x].")
parser.add_argument("--img_domain_x", type=str, nargs="*", default=[2],
                    help="One or two arguments representing the images to use for domain X. If \
two arguments are given, the union of the groups will be used. 2 is 'strong', 1 is 'weak', 0 is no-DA.")
parser.add_argument("--img_domain_y", type=str, nargs="*", default=[1],
                    help="One or two arguments representing the images to use for domain Y. If \
two arguments are given, the union of the groups will be used.")
parser.add_argument("--checkpoint", type=str, default="None",
                    help="The path to a checkpoint file. If 'None', training will start from scratch.")



# Helper function to get arguments in other scripts
def get_args():
    args, unparsed = parser.parse_known_args()

    return args, unparsed
