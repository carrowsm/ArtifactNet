from argparse import ArgumentParser


### EDIT THESE PATHS ###
img_path = "/cluster/projects/bhklab/RADCURE/img/"
img_suffix = "_img.npy"                 # This string follows the patient ID in the filename
label_path = "/cluster/home/carrowsm/logs/label/reza_artifact_labels.csv"
log_dir = "/cluster/home/carrowsm/logs/label/"
### ---------------- ###

parser = ArgumentParser()

parser.add_argument("--img_dir", default=img_path, type=str, help="Path to the input image data.")
parser.add_argument("--img_suffix", default=img_suffix, type=str)

parser.add_argument("--data_type", default="phantom", type=str,
help="'phantom' if we are using artificial input images, 'real' if we are using patient CT scans.")
parser.add_argument("--label_dir", default=label_path, type=str, help='Path to a CSV containing image labels.')
parser.add_argument("--logdir", default=log_dir, type=str, help='Where to save results.')


parser.add_argument("--norm_lower", default=0.64, type=float)
parser.add_argument("--norm_upper", default=0.86, type=float)

parser.add_argument("--test", action='store_true', help="If the test option is given, code will only process a few images.")

parser.add_argument("--ncpu", default=None, type=int, help="Number of CPUs to use.")


### Args for model training ###
parser.add_argument("--batch_size", type=int, default=64, help="size of the batches")
parser.add_argument("--lr", type=float, default=0.0002, help="adam: learning rate")
parser.add_argument("--b1", type=float, default=0.5,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--b2", type=float, default=0.999,
                    help="adam: decay of first order momentum of gradient")
parser.add_argument("--latent_dim", type=int, default=None,
                    help="Dimension of the image that it given to the generator.\
 By default, the dataloader image size will be used.")


# Helper function to get arguments in other scripts
def get_args():
    args, unparsed = parser.parse_known_args()

    return args, unparsed
