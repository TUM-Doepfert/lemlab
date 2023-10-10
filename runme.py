from lemlab import ScenarioExecutor

# in this example we execute the demo scenario

if __name__ == "__main__":
    sim_name = "test"

    simulation = ScenarioExecutor(path_scenario=f"./scenarios/{sim_name}",
                                  path_results=f"./simulation_results/{sim_name}")
    simulation.run()

    # Next steps: Run the simulation and get the controllers and forecasters to work.
