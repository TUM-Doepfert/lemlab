from lemlab import ScenarioExecutor
import os

# in this example we execute the demo scenario

if __name__ == "__main__":
    # sim_name = "Countryside_pv_1.0_hp_1.0_ev_1.0_winter_woLEM_mpc"
    # sim_names = ["countryside/wLEM/Countryside_pv_0.5_hp_0.5_ev_1.0_winter",
    #               "countryside/woLEM/Countryside_pv_0.5_hp_0.5_ev_1.0_winter_woLEM",
    #               "countryside/wLEM/Countryside_pv_0.5_hp_0.5_ev_1.0_summer",
    #               "countryside/woLEM/Countryside_pv_0.5_hp_0.5_ev_1.0_summer_woLEM",
    #               "countryside/wLEM/Countryside_pv_0.5_hp_0.5_ev_1.0_transition",
    #               "countryside/woLEM/Countryside_pv_0.5_hp_0.5_ev_1.0_transition_woLEM",
    #               ]
    sim_names = ["countryside/wLEM/Countryside_pv_1.0_hp_0.0_ev_0.0_summer"]
    sim_names = ["test_sim"]

    for sim_name in sim_names:
        sim = sim_name.split("/")[-1]

        simulation = ScenarioExecutor(path_scenario=f"../scenarios/{sim_name}",
                                      path_results=f"../simulation_results/{sim}")
        simulation.run(hamlet=False)

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

