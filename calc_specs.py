import numpy as np
import pandas as pd
import os

'''A script to calculate the mean, median and standard deviation of the entire dataset'''



def parse_files(path) :
    """Get a list of all the files in the dataset, (list all
    npy files in path/*)"""
    file_list = []
    nb_patients = 0

    for file in os.listdir(path) :
        extension = file.split(".")[-1]
        if extension == "npy" :
            # Add the file name to the list
            file_list.append(file)
            nb_patients = nb_patients + 1

    return file_list, nb_patients

def load_stack(path) :
    """Load one image file and return numpy array of image stack"""
    data = np.load(path)
    return data



if __name__ == "__main__" :

    img_path = "/cluster/projects/radiomics/Temp/RADCURE-npy/img"


    patient_list, nb_patients = parse_files(img_path)

    # Data frame containing stat for each patient
    df = pd.DataFrame(columns=["file", "N", "mean", "median", "std"])
    df["file"] = patient_list



    for i in df.index :
        file_name = df["file"].loc[i]
        full_path = os.path.join(img_path, file_name)

        # Get the image
        image = load_stack(full_path)

        # Calculate stats
        df["N"].loc[i]      = np.size(image)
        df["mean"].loc[i]   = np.mean(image)
        df["median"].loc[i] = np.median(image)
        df["std"].loc[i]    = np.std(image)

        print(f"Patient {file_name.split('_')[0]} analysed")


    print(df)
    df.dropna(inplace=True)

    # calculate total data set mean and std
    means = df["mean"].values
    stds  = df["std"].values
    Ns = df["N"].values

    """Calculate the pooled mean and pooled standard deviation
    Formulae taken from :
    https://en.wikipedia.org/wiki/Pooled_variance"""

    total_mean = np.sum(Ns * means) / np.sum(Ns)
    total_std  = np.sum((Ns - 1) * stds) / np.sum(Ns - 1)

    print("Pooled Mean Pixel Intensity: ", total_mean)
    print("Pooled Standard Deviation Pixel Intensity: ", total_std)

    # Save the data frame as a CSV
    df.to_csv("RADCURE_stats.csv")
