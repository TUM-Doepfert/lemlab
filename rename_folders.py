"""The file adds a suffix to all folders in a given directory"""
import os

# Define the directory where your folders are located
# regions: countryside, rural, suburban, urban
# scenarios: woLEM_rtc_naive, woLEM_mpc_naive, wLEM_mpc_perfect, wLEM_mpc_naive
directory_path = os.path.join('.', 'simulation_results', 'rural', 'woLEM_mpc_naive')

# Define the string to be added to the end of each folder
suffix_separator = "_"
suffix_to_add = suffix_separator + directory_path.split(os.sep)[-1]
suffix_parts = suffix_to_add.split(suffix_separator)[1:]

# List all folders in the directory
folders = next(os.walk(directory_path))[1]

# Iterate through the folders and rename them
for folder in folders:
    # Check if the folder already contains the suffix or parts of it
    if not folder.endswith(suffix_to_add):
        # Remove any existing parts of the suffix
        folder_wo_suffix = folder
        for part in suffix_parts:
            string = f'{suffix_separator}{part}'
            folder_wo_suffix = folder_wo_suffix.replace(string, "")

        # Add the complete suffix
        new_folder_name = folder_wo_suffix + suffix_to_add

        # Create the full path of the old and new folder names
        old_path = os.path.join(directory_path, folder)
        new_path = os.path.join(directory_path, new_folder_name)

        # Rename the folder
        os.rename(old_path, new_path)
