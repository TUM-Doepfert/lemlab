"""This script runs all the lemlab scenarios specified in the scenario folder.
The scenarios already need to be converted from the hamlet formet"""

from hamlet2lemlab import h2l
import os
from tqdm import tqdm


def main(hamlet_scenario: str = './hamlet_scenarios', lemlab_scenario: str = './scenarios'):
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
        lemlab_path = os.path.join(lemlab_scenario, scenario)

        # Convert the scenario from hamlet to lemlab format
        h2l(hamlet_path, lemlab_path, scenario, weather="weather.ft").convert()

        # Update the progressbar
        pbar.update()

        break

    # Close the progressbar
    pbar.close()


if __name__ == "__main__":
    main()
