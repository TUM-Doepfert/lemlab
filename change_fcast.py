"""This script sets all prosumers's fcast. This needs to be copied into the scenario folder and executed there."""
import os
import json

path = './scenarios/countryside/woLEM'

# Get a list of all scenarios
scenarios = next(os.walk(path))[1]
print(scenarios)
exit()


# Get a list of all prosumers
prosumers = next(os.walk('./prosumer'))[1]

# Go through each prosumer
for prosumer in prosumers:
    # Load plant json file
    with open(f'./prosumer/{prosumer}/config_plants.json', 'r') as f:
        plants = json.load(f)

    # Go through each plant
    for plant, info in plants.items():
        # Check if the plant is pv
        if info['type'] == 'pv':
            # Set the pv to not controllable
            info['controllable'] = False
            # Update the plants file
            plants[plant] = info

    # Save the updated plants file
    with open(f'./prosumer/{prosumer}/config_plants.json', 'w') as f:
        json.dump(plants, f, indent=4)
