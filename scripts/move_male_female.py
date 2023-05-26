import pandas as pd
import os
import shutil

# Read the CSV file
df = pd.read_csv("../data/mixtec_figures.csv")

# Set the file locations
fileOrigin = "../data/unlabeled_codices/codex_vindobonensis/cutouts/"
fileDest = "../data/labeled_figures/codex_vindobonensis/seated/"

# Iterate through the file names
for ff in df['file_name'].tolist():
    file_codex = df['codex'].loc[df['file_name'] == ff]

    # Check if the value in the "codex" column is equal to "codex_nutall"
    if str(file_codex.values[0]) == "codex_vindobonensis":

        # Get the gender from the CSV file
        folder = df['seated'].loc[df['file_name'] == ff].values[0]

        # Move the file to the destination folder
        shutil.copy(fileOrigin + ff, fileDest + folder + "/")

print("Files have been copied successfully!")