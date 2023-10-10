"""This script resizes all pv plants by changing their timeseries file (raw_data).
This needs to be copied into the scenario folder and executed there."""
import os
import json
import pandas as pd


# Get a list of all prosumers
prosumers = next(os.walk('./prosumer'))[1]

# Resize factor and column to resize
resize_factor = 1
resize_column = 'heat'
resize_component = 'hp'


# Go through each prosumer
for prosumer in prosumers:
    # Load plant json file
    with open(f'./prosumer/{prosumer}/config_plants.json', 'r') as f:
        plants = json.load(f)

    # List of pv plant names
    comp_plants = [plant for plant, info in plants.items() if info['type'] == resize_component]

    # Go through each plant
    for comp in comp_plants:
        # Load the timeseries file
        ts = pd.read_feather(f'./prosumer/{prosumer}/raw_data_{comp}.ft').set_index('timestamp')
        # Get the maximum hp power
        # max_power = plants[comp]['power']
        # Make sure that no value exceeds the maximum power
        # ts = ts.clip(lower=-max_power)
        # Resize the timeseries
        ts[resize_column] *= resize_factor
        # Save the timeseries file
        ts.reset_index().to_feather(f'./prosumer/{prosumer}/raw_data_{comp}.ft')

print(f'All {resize_component} plants were resized by a factor of {resize_factor}.')
