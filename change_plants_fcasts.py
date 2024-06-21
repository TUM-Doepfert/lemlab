"""This script changes the forecast method of the plants."""
import os
import json

# Path to scenarios
path = './to_simulate/urban/wLEM_perfect'

# Dict containing the fcast method for each plant type
fcasts = {
    'hh': 'perfect',
    'pv': 'perfect',
    'hp': 'perfect',
    'ev': 'perfect',
}

# Get a list of all scenarios
scenarios = next(os.walk(path))[1]

# Go through each scenario
for scenario in scenarios:

    # Get a list of all prosumers
    prosumers = next(os.walk(os.path.join(path, scenario, 'prosumer')))[1]

    # Go through each prosumer
    for prosumer in prosumers:

        # Load plant json file
        plant_path = os.path.join(path, scenario, 'prosumer', prosumer, 'config_plants.json')
        with open(plant_path, 'r') as f:
            plants = json.load(f)

        # Go through each plant
        for plant, info in plants.items():
            # Check the type of the plant
            if info['type'] in fcasts:
                # Set the forecast method
                info['fcast'] = fcasts[info['type']]
                # Update the plants file
                plants[plant] = info

        # Save the updated plants file
        with open(plant_path, 'w') as f:
            json.dump(plants, f, indent=4)
