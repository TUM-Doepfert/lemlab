import lemlab
import os

# this example demonstrates the use of the scenario analyzer library
# by showing some simple plots of the results of the demo simulation


def plot_powerflow_for_all_scenarios(path: str, folders: list, sim_name: str, limits: dict = None):
    for folder in folders:
        sims = [sim for sim in next(os.walk(f"{path}/{folder}"))[1] if sim.startswith(sim_name)]
        for sim in sims:
            print(sim)
            if limits:
                season = [season for season in limits.keys() if season in sim][0]
                limit = limits[season]
            else:
                limit = None
            path_results = f"{path}/{folder}/{sim}"

            analysis = lemlab.ScenarioAnalyzer(path_results=path_results,
                                               show_figures=True,
                                               save_figures=False)
            analysis.plot_virtual_feeder_flow(limits=limit)


def plot_mcp_for_lem_scenarios(path: str, folders: list, sim_name: str):
    for folder in folders:
        if folder.str.endswith('woLEM'):
            continue
        sims = [sim for sim in next(os.walk(f"{path}/{folder}"))[1] if sim.startswith(sim_name)]
        for sim in sims:
            print(sim)
            path_results = f"{path}/{folder}/{sim}"

            analysis = lemlab.ScenarioAnalyzer(path_results=path_results,
                                               show_figures=True,
                                               save_figures=False)
            analysis.plot_mcp()


def plot_detailed_powerflow_for_all_scenarios(path: str, folders: list, sim_name: str, limits: dict = None):
    for folder in folders:
        sims = [sim for sim in next(os.walk(f"{path}/{folder}"))[1] if sim.startswith(sim_name)]
        for sim in sims:
            if limits:
                season = [season for season in limits.keys() if season in sim][0]
                limit = limits[season]
            else:
                limit = None

            print(sim)

            path_results = f"{path}/{folder}/{sim}"

            analysis = lemlab.ScenarioAnalyzer(path_results=path_results,
                                               show_figures=True,
                                               save_figures=False)
            analysis.plot_detailed_virtual_feeder_flow(limits=limit, name=sim)


def plot_household_for_all_scenarios(path: str, folders: list, sim_name: str, household: int, limits: dict = None):
    for folder in folders:
        sims = [sim for sim in next(os.walk(f"{path}/{folder}"))[1] if sim.startswith(sim_name)]
        for sim in sims:
            if limits:
                season = [season for season in limits.keys() if season in sim][0]
                limit = limits[season]
            else:
                limit = None

            print(sim)

            path_results = f"{path}/{folder}/{sim}"

            analysis = lemlab.ScenarioAnalyzer(path_results=path_results,
                                               show_figures=True,
                                               save_figures=False)
            analysis.plot_household(id_user=household, limits=limit)


if __name__ == "__main__":

    path = './all_lemlab_results/Land'

    sim_name = "Countryside_pv_1.0_hp_1.0_ev_1.0"

    folders = ['results', 'results_woLEM']

    limits = {
        "summer": [1563832800, 1564437600],
        "transition": [1569276000, 1569880800],
        "winter": [1573513200, 1574118000],
    }

    household = 6

    # plot_detailed_powerflow_for_all_scenarios(path, folders, sim_name, limits)

    # plot_powerflow_for_all_scenarios(path, folders, sim_name, limits)

    plot_household_for_all_scenarios(path, folders, sim_name, household, limits)
