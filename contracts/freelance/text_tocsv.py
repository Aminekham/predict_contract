import os
import pandas as pd
import re
from glob import glob

# Specify the folder containing the .txt files
folder_path = "./"

# Get a list of all .txt files in the folder
txt_files = glob(os.path.join(folder_path, '*.txt'))

# Create an empty list to store DataFrames
dfs = []

# Iterate over each .txt file
for txt_file in txt_files:
    # Read the text file
    with open(txt_file, 'r', encoding='utf-8') as file:
        lines = file.read().split('*')

    # Remove non-printable characters
    lines = [re.sub(r'[﻿]', '', line) for line in lines]

    # Create a DataFrame from the text file
    df = pd.DataFrame({'Content': lines})
    
    # Remove empty rows
    df = df[df['Content'].str.strip() != '']
    
    # Reindex the DataFrame
    df.index = range(1, len(df) + 1)

    # Append the DataFrame to the list
    dfs.append(df)

# Concatenate all DataFrames in the list into a single DataFrame
all_data = pd.concat(dfs, ignore_index=True)

# Get the directory of one of the input files
output_folder = os.path.dirname(txt_files[0])

# Save the combined DataFrame to a CSV file in the same folder
output_file_path = os.path.join(output_folder, 'output_combined.csv')
all_data.to_csv(output_file_path, index=True, encoding='utf-8')
