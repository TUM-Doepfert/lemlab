from lemlab import ScenarioExecutor
import os

# in this example we execute the demo scenario

if __name__ == "__main__":
    sim_name = "Countryside_pv_1.0_hp_1.0_ev_1.0_winter_moreheat"

    simulation = ScenarioExecutor(path_scenario=f"../scenarios/{sim_name}",
                                  path_results=f"../simulation_results/{sim_name}")
    simulation.run()

    # Next steps: Run the simulation and get the controllers and forecasters to work.

# if __name__ == "__main__":
#     # Get a list of all scenarios to simulate
#     scenarios = next(os.walk('../scenarios'))[1]
#     for scenario in scenarios:
#         sim_name = scenario
#
#         simulation = ScenarioExecutor(path_scenario=f"../scenarios/{sim_name}",
#                                       path_results=f"../simulation_results/{sim_name}")
#         simulation.run()

