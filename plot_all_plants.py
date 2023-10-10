"""Plots the plants of the prosumer. Needs to be in the prosumer directory."""

import pandas as pd
import matplotlib.pyplot as plt
import os
import json

# Plant type to plot
plant_type = 'pv'

# Get all prosumers
prosumers = next(os.walk('.'))[1]

# Create df for all prosumers
df_all = pd.DataFrame()
num_plants = 0

# Loop through all prosumers
for prosumer in prosumers:
    # Load plant config
    with open(f'{prosumer}/config_plants.json', 'r') as f:
        plant_config = json.load(f)

    # Get all plants of the desired type
    plants = [plant for plant, info in plant_config.items() if info['type'] == plant_type]

    # Create df for plotting
    df_plot = pd.DataFrame()

    # Loop through all plants
    for plant in plants:
        # Concatenate all plant data
        df = pd.read_feather(f'{prosumer}/raw_data_{plant}.ft').set_index('timestamp')
        df_plot = pd.concat([df_plot, df], axis=1)

    # Add info to df for all prosumers
    df_all = pd.concat([df_all, df_plot.sum(axis=1)], axis=1)

    # Make index datetime
    df_plot.index = pd.to_datetime(df_plot.index, unit='s')

    # Plot
    df_plot.sum(axis=1).plot()
    plt.title(f'{prosumer}: {plant_type} {len(plants)}x')
    plt.show()

    # Count number of plants
    num_plants += len(plants)

# Make index datetime
df_all.index = pd.to_datetime(df_all.index, unit='s')

# Plot
df_all.sum(axis=1).plot()
plt.title(f'All prosumers: {plant_type} {num_plants}x')
plt.show()
