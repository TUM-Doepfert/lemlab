import lemlab


if __name__ == "__main__":
    sim_name = "base_neuburg"

    simulation = lemlab.ScenarioExecutor(path_scenario=f"../scenarios/{sim_name}",
                                         path_results=f"../simulation_results/{sim_name}")
    simulation.run()
