"""This script sets all prosumers's parameter param to a random value in val. This needs to be copied into the scenario folder and executed there."""
import os
import json
import random

# Parameter to change
params = ['controller_strategy', 'mpc_price_fcast']
vals = [['mpc_opt'], ['flat']]

# Path to scenarios
path = './scenarios/countryside/woLEM_mpc'

# Get a list of all scenarios
scenarios = next(os.walk(path))[1]

for scenario in scenarios:

    # Get a list of all prosumers
    path_pros = os.path.join(path, scenario, 'prosumer')
    prosumers = next(os.walk(path_pros))[1]

    # Go through each prosumer
    for prosumer in prosumers:
        # Load plant json file
        with open(os.path.join(path_pros, prosumer, 'config_account.json'), 'r') as f:
            config = json.load(f)

        #print(prosumer)
        #print(config[param])

        for param, val in zip(params, vals):
            # Change chosen parameter and choose random value from val
            config[param] = random.choice(val)

        #print(config[param])
        #exit()

        # Save the updated plants file
        with open(os.path.join(path_pros, prosumer, 'config_account.json'), 'w') as f:
            json.dump(config, f, indent=4)
