"""This script resizes all battery plants by changing their timeseries file (raw_data).
This needs to be copied into the scenario folder and executed there."""
import os
import json
import pandas as pd


# Get a list of all prosumers
prosumers = next(os.walk('./prosumer'))[1]

# Resize factor and column to resize
resize_factor = 1.5  # factor to resize the battery capacity depending on the pv capacity


# Go through each prosumer
for prosumer in prosumers:
    # Load plant json file
    with open(f'./prosumer/{prosumer}/config_plants.json', 'r') as f:
        plants = json.load(f)

    # Get power of pv plants
    pv_power = sum([int(info['power']) for plant, info in plants.items() if info['type'] == 'pv'])

    # Go through each plant and find first battery (if there are more they will be ignored)
    for plant, info in plants.items():
        if info['type'] == 'bat':
        # Change the capacity and power of the battery
            plants[plant]['capacity'] = int(pv_power * resize_factor)
            plants[plant]['power'] = int(pv_power * resize_factor)
            break

    # Save the plant json file
    with open(f'./prosumer/{prosumer}/config_plants.json', 'w') as f:
        json.dump(plants, f, indent=4)

print(f'All battery plants were resized by a factor of {resize_factor}.')
