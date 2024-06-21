"""Deletes everything except db_snapshot for all folders as this is the only relevant one."""

import os
import shutil
from tqdm import tqdm

# Directory to start the search
# regions: countryside, rural, suburban, urban
# scenarios: woLEM_rtc_naive, woLEM_mpc_naive, wLEM_mpc_perfect, wLEM_mpc_naive
start_directory = './simulation_results/rural/woLEM_mpc_naive'

# Count failures
failures = 0

# raise 'This script is dangerous. Please check the code before executing it.'

# Iterate through all folders and subfolders
folder_root, subfolders, _ = next(os.walk(start_directory))
for subfolder in tqdm(subfolders, desc="Processing Folders"):
    path_sf = os.path.join(folder_root, subfolder)
    subsubfolders = next(os.walk(path_sf))[1]
    for subsubfolder in subsubfolders:
        path_ssf = os.path.join(path_sf, subsubfolder)
        # Delete everything except db_snapshot folder
        if subsubfolder != 'db_snapshot':
            try:
                shutil.rmtree(path_ssf)
            except Exception as e:
                failures += 1
        # Delete file positions_market_ex_ante_archive.csv in db_snapshot folder
        elif subsubfolder == 'db_snapshot':
            try:
                os.remove(os.path.join(path_ssf, 'positions_market_ex_ante_archive.csv'))
            except Exception as e:
                failures += 1

print(f"Finished deleting everything except db_snapshot for all folders. {failures} failures.")
