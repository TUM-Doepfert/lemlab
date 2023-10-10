"""This script removes all plants by removing them from the plant dict.
This needs to be copied into the scenario folder and executed there."""
import os
import json
import pandas as pd


# Get a list of all prosumers
prosumers = next(os.walk('./prosumer'))[1]

# Component to remove
remove_component = 'pv'


# Go through each prosumer
for prosumer in prosumers:
    # Load plant json file
    with open(f'./prosumer/{prosumer}/config_plants.json', 'r') as f:
        plants = json.load(f)

    new_plants = plants.copy()

    # Go through each plant
    for plant, info in plants.items():
        # Check if plant is the component to remove
        if info['type'] == remove_component:
            # Remove the plant
            new_plants.pop(plant)

    # Save the new plant dict
    with open(f'./prosumer/{prosumer}/config_plants.json', 'w') as f:
        json.dump(new_plants, f, indent=4)

print(f'All {remove_component} plants were removed.')
