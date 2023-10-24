"""Checks which type of simulation the folders are and puts them into the respective folder."""

import os
import shutil

source_folder = './'  # specify the path to your main folder here if it's different

woLEM_destination = os.path.join(source_folder, 'woLEM')
wLEM_destination = os.path.join(source_folder, 'wLEM')
woLEM_mpc_destination = os.path.join(source_folder, 'woLEM_mpc')

# Create the destination folders if they don't exist
if not os.path.exists(woLEM_destination):
    os.makedirs(woLEM_destination)

if not os.path.exists(wLEM_destination):
    os.makedirs(wLEM_destination)

if not os.path.exists(woLEM_mpc_destination):
    os.makedirs(woLEM_mpc_destination)

# Iterate through all items in the source folder
for item in os.listdir(source_folder):
    item_path = os.path.join(source_folder, item)

    # Check if the item is a directory and not the destination folders themselves
    if os.path.isdir(item_path) and item not in ['woLEM', 'wLEM', 'woLEM_mpc']:
        # Move directories ending with 'woLEM' to 'woLEM' folder
        if item.endswith('_woLEM'):
            shutil.move(item_path, os.path.join(woLEM_destination, item))
        elif item.endswith('_woLEM_mpc'):
            shutil.move(item_path, os.path.join(woLEM_mpc_destination, item))
        # Move all other directories to 'wLEM' folder
        else:
            shutil.move(item_path, os.path.join(wLEM_destination, item))

print("Folders moved successfully!")
