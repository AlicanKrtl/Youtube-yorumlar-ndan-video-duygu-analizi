import os
import pandas as pd

# Directory containing the CSV files
folder_path = '/home/alican/Documents/Studies/begüm_proje'

# List to store DataFrames from CSV files
dfs = []

# Iterate over files in the directory
for file in os.listdir(folder_path):
    if file.endswith('.csv'):
        # Read CSV file into a DataFrame
        df = pd.read_csv(os.path.join(folder_path, file))
        
        # Append DataFrame to the list
        dfs.append(df)

# Check if all DataFrames have the same column keys
column_keys = set(dfs[0].columns)
for df in dfs[1:]:
    if set(df.columns) != column_keys:
        raise ValueError("Not all CSV files have the same columns keys.")

# Merge all DataFrames
merged_df = pd.concat(dfs, ignore_index=True)

# Save the merged DataFrame to a new CSV file
merged_df.to_csv('merged_data.csv', index=False)

print("Merged DataFrame saved to 'merged_data.csv'")
