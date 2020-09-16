import os
from argparse import ArgumentParser

import time
import numpy as np
import pandas as pd
import torch
import torchvision
import json

from torch.utils.data import DataLoader

import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

from cycleGAN import GAN
from data.data_loader import load_image_data_frame, UnpairedDataset
from data.transforms import ToTensor, Normalize
from data.postprocessing import PostProcessor

import matplotlib.pyplot as plt

""" Use this script to 'clean' a specific set of DA images using the 3D-cycleGAN
model from cycleGAN.

A script to generate and save images using one of the pre-trained generators
from cycleGAN for DA reduction.
"""


def prepare_data(img_list_csv) :
    """Get a list of patient IDs to be cleaned
    """
    df = pd.read_csv(img_list_csv, dtype=str).set_index("patient_id")
    x_df = df[df["has_artifact"].isin(["2", "1"])] # Limit df to 'dirty' images
    y_df = df[df["has_artifact"].isin(["0"])] # Limit df to 'dirty' images
    return x_df, y_df



def load_model(module: pl.LightningModule, checkpoint_path: str) :
    """ Load a pytorchlightning module from checkpoint"""
    model = module.load_from_checkpoint(checkpoint_path)
    model.eval()
    return model

def process_one_image() :
    pass


def main(args) :
    # Save a file containing info on the model used to create these results
    info_file_path = os.path.join(args.out_img_dir, "model_info.json")
    info_dict = {"path_to_data_csv" : args.csv_path,
                 "path_to_saved_model" : args.checkpoint,}
    with open(info_file_path, 'w') as outfile:
        json.dump(info_dict, outfile, indent=4)


    # Get list of patient IDs
    x_df, y_df = prepare_data(args.csv_path) # X is DA+, Y is DA-

    # Define transforms to normalize input data
    transform = torchvision.transforms.Compose([
                        Normalize(-1000.0, 1000.0),
                        ToTensor()])

    # Initialize the dataloader
    dataset = UnpairedDataset(x_df, y_df,
                              image_dir=args.in_img_dir,
                              cache_dir=args.cache_dir,
                              file_type="nrrd",
                              image_size=[8, 256,256],
                              dim=3,
                              transform=transform,
                              num_workers=args.n_cpus)
    dataloader = DataLoader(dataset, batch_size=1, shuffle=False,
                            num_workers=args.n_cpus)
    print("Dataloaders created.")
    print(f"Generating {len(dataset)} clean images.")

    # Initialize the postprocessor
    postprocess = PostProcessor(input_dir=args.in_img_dir,
                                output_dir=args.out_img_dir,
                                output_spacing=[1, 1, 3],
                                input_file_type="nrrd",
                                output_file_type="nrrd")

    # Load the model from checkpoint
    model = load_model(GAN, args.checkpoint)
    print("Model loaded")

    if torch.cuda.is_available() :
        n_gpus = torch.cuda.device_count()
        print(f"{n_gpus} GPUs are available")
        device = torch.device('cuda')
    else :
        device = torch.device('cpu')

    # Move the model to GPU
    model.to(torch.device('cpu'))
    generator = model.g_y.to(device)
    del model # Free up memory


    # Start Test loop
    with torch.no_grad() :
        for i, data in enumerate(dataloader) :
            t0 = time.time()
            x, y = data # Get img X (DA+) and a random unpaired Y (DA-) img
            gen_y = generator(x.to(device)) # Translate DA+ img to DA-

            # Postprocess the resulting torch tensor.
            # This re-inserts the generated image back into the full DICOM and
            # saves it in NRRD format
            patient_id = dataset.x_ids[i]
            img_center_x = float(x_df.at[patient_id, "img_center_x"])
            img_center_y = float(x_df.at[patient_id, "img_center_y"])
            img_center_z = float(x_df.at[patient_id, "img_center_z"])
            postprocess(gen_y.to(torch.device('cpu')),
                        patient_id,
                        [img_center_x, img_center_y, img_center_z])

            print(f"Image {patient_id} processed in {time.time() - t0} s.")





if __name__ == '__main__':
    # Defaut Paths
    module_path = "/cluster/home/carrowsm/ArtifactNet/"
    # csv_path = os.path.join(module_path, "datasets/oar_segment_imgs.csv")
    csv_path = os.path.join(module_path, "datasets/radcure_challenge_test.csv")
    # img_dir = "/cluster/projects/radiomics/RADCURE-images/"        # Raw DICOM images
    img_dir = "/cluster/projects/radiomics/RADCURE-challenge/data/test/images"
    # log_dir = "/cluster/projects/radiomics/Temp/colin/oar_test_clean"
    log_dir = "/cluster/projects/radiomics/Temp/colin/radcure_challenge_test"
    # cache = "/cluster/projects/radiomics/Temp/colin/isotropic_nrrd/unpaired"
    cache = "/cluster/home/carrowsm/img-cache"
    checkpoint_path = "/cluster/home/carrowsm/logs/artifact_net/remove/cycleGAN/\
8_256_256px/2_1-0/version_2/checkpoints/epoch=48.ckpt"

    parser = ArgumentParser()

    # Paths to data and logs
    parser.add_argument("--csv_path", type=str, default=csv_path,
        help="Path to the CSV containing all image DA statuses and DA locations")
    parser.add_argument("--in_img_dir", default=img_dir, type=str,
                        help="Path to the input image data (DICOM).")
    parser.add_argument("--cache_dir", default=cache, type=str,
                        help="Where to cache images for model input.")
    parser.add_argument("--out_img_dir", default=log_dir, type=str,
                        help='Where to save clean output images.')

    ### Hyperparams for model testing ###
    parser.add_argument("--n_gpus", type=int, default=1,
                        help="Number of GPUs to use for training")
    parser.add_argument("--n_cpus", type=int, default=1,
                        help="Number of parallel workers to use for data loading.")

    ### Model selection and training parameters ###
    parser.add_argument("--checkpoint", type=str, default=checkpoint_path,
    help="The path to a checkpoint file. If 'None', training will start from scratch.")

    args, unparsed = parser.parse_known_args()


    main(args)
