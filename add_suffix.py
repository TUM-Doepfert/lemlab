import os

def add_suffix_to_folders(parent_folder, suffix="_mpc"):
    """
    Add a specified suffix to each folder within the given parent folder.

    Args:
    parent_folder (str): The path to the parent folder.
    suffix (str): The suffix to add to each folder. Defaults to "_mpc".
    """
    for item in os.listdir(parent_folder):
        item_path = os.path.join(parent_folder, item)
        if os.path.isdir(item_path):
            new_name = item_path + suffix
            os.rename(item_path, new_name)
            # print(f"Renamed '{item_path}' to '{new_name}'")

# Run
# path = './rural/woLEM_mpc'
add_suffix_to_folders(path)
