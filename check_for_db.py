"""Checks if the scenarios have the db_snapshot folder which means they ran through successfully."""

import os


def check_subfolders_for_db_snapshot(folder_path):
    """
    Checks all subfolders of the given folder to see if they contain a folder named 'db_snapshot'.
    Prints the names of subfolders that do not contain 'db_snapshot'.
    """
    if not os.path.exists(folder_path):
        print(f"The folder '{folder_path}' does not exist.")
        return

    subfolders_without_db_snapshot = []
    for entry in os.listdir(folder_path):
        subfolder_path = os.path.join(folder_path, entry)
        if os.path.isdir(subfolder_path):
            if not 'db_snapshot' in os.listdir(subfolder_path):
                subfolders_without_db_snapshot.append(entry)

    if subfolders_without_db_snapshot:
        print("Subfolders without 'db_snapshot':")
        for subfolder in subfolders_without_db_snapshot:
            print(subfolder)
    else:
        print("All subfolders contain 'db_snapshot'.")

# Example usage
folder_path = "./simulation_results"
check_subfolders_for_db_snapshot(folder_path)
