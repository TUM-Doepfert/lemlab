from hamlet2lemlab import h2l
from lemlab import ScenarioExecutor
import os
from tqdm import tqdm


def main(hamlet_scenario: str = './hamlet_scenarios', lemlab_scenario: str = './scenarios',
         lemlab_results: str = './simulation_results'):
    # Get a list of all scenarios to simulate
    scenarios = next(os.walk(hamlet_scenario))[1]

    # Prepare the progressbar
    pbar = tqdm(total=len(scenarios))

    # Go through each scenario
    for scenario in scenarios:
        # Set the progressbar description
        pbar.set_description(f"Converting {scenario}")

        # Set scenario paths
        hamlet_path = os.path.join(hamlet_scenario, scenario)
        lemlab_path = os.path.join(lemlab_scenario, scenario)

        # Convert the scenario from hamlet to lemlab format
        h2l().convert(hamlet_path, lemlab_path, scenario)

        # Set the progressbar description
        pbar.set_description(f"Running {scenario}")

        # Run the scenario
        simulation = ScenarioExecutor(path_scenario=lemlab_path,
                                      path_results=os.path.join(lemlab_results, scenario))
        simulation.run()

        # Update the progressbar
        pbar.update()

    # Close the progressbar
    pbar.close()


if __name__ == "__main__":
    main()
