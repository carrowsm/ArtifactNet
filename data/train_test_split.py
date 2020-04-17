""" A script to create a train/validation set and a seperate test set for RadCure data"""

import os
import numpy as np
import pandas as pd

from data_loader import load_image_data_frame
from sklearn.model_selection import train_test_split


def main(csv_path, save_dir="", test_size=0.10) :
    """ Load a csv containing all data labels. Create the train/test split
    and output one DF for each train and test set, with same format as
    original dataframe."""

    # Load data in a DataFrame
    df = pd.read_csv(csv_path, dtype=str)
    df.set_index("p_index", inplace=True) # Set index

    # Split the patient IDs into a train and test set
    # Stratify the split based on number of each DA class
    train_ids, test_ids = train_test_split(df["patient_id"].values,
                                           test_size=test_size,
                                           stratify=df["has_artifact"].values)
    # Make each list of patient IDs into its own data frame
    train_df = df[df["patient_id"].isin(train_ids)]
    test_df = df[df["patient_id"].isin(test_ids)]

    # Save each dataframe as a CSV
    train_df.to_csv(os.path.join(save_dir, "train_labels.csv"))
    test_df.to_csv(os.path.join(save_dir, "test_labels.csv"))


if __name__ == '__main__':
    # Path to full data labels and path to save split data
    in_path = "/cluster/home/carrowsm/data/radcure_DA_labels.csv"
    out_path ="/cluster/home/carrowsm/ArtifactNet/data/labels/"

    main(in_path, save_dir=out_path, test_size=0.10)
