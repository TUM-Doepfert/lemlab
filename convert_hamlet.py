"""This script runs all the lemlab scenarios specified in the scenario folder.
The scenarios already need to be converted from the hamlet formet"""

from hamlet2lemlab import H2l
import os
from tqdm import tqdm


def main(hamlet_scenario: str = './test', lemlab_scenario: str = './scenarios'):
    # Get a list of all scenarios to simulate
    scenarios = next(os.walk(hamlet_scenario))[1]

    # Prepare the progressbar
    pbar = tqdm(total=len(scenarios))

    # Go through each scenario
    for scenario in scenarios:
        # Set the progressbar description
        string = f"Converting {scenario}"
        pbar.set_description(string)

        # Set scenario paths
        hamlet_path = os.path.join(hamlet_scenario, scenario)
        lemlab_path_wLEM = os.path.join(lemlab_scenario, f'{scenario}')
        lemlab_path_woLEM = os.path.join(lemlab_scenario, f'{scenario}_woLEM')

        # Convert the scenario from hamlet to lemlab format once with LEM and once without LEM
        H2l(hamlet_path, lemlab_path_wLEM, lem=True, weather="weather.ft").convert()
        H2l(hamlet_path, lemlab_path_woLEM, lem=False, weather="weather.ft").convert()

        # Update the progressbar
        pbar.update()

    # Close the progressbar
    pbar.close()


if __name__ == "__main__":
    main()
