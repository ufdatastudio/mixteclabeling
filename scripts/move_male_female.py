import pandas as pd
import os
import shutil

## Variable to sort
column_name = 'gender'

# Read the CSV file
df_v = pd.read_csv("../data/CSVs/codex_vindobonensis.csv")

df_n = pd.read_csv("../data/CSVs/codex_nuttall.csv")

df_s = pd.read_csv("../data/CSVs/codex_selden.csv")

df_list = []

df_list.append(df_v)
df_list.append(df_n)
df_list.append(df_s)

# Set the file locations
fileOrigin = "../data/unlabeled_codices/"
fileDest = "../data/labeled_figures/"

file_seen_dict = {}

# iterate through codices:
for df_codex in df_list:

    # Iterate through the file names of particular codex
    for ff in sorted(df_codex['file_name'].tolist()):
        print(ff)
    
        # Get the codex from the CSV file
        codex_folder = df_codex['codex'].loc[df_codex['file_name'] == ff].values[0]
        
        # Get the get the specific class from the CSV file
        class_folder = df_codex[column_name].loc[df_codex['file_name'] == ff].values[0]
    
        # Move the file to the destination folder
        shutil.copy(fileOrigin + codex_folder + "/" + "cutouts/" + ff, 
        fileDest + codex_folder + "/" + column_name + "/" + class_folder + "/")
        
        

print("Files have been copied successfully!")
