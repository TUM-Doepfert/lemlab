import lemlab

# this example demonstrates the use of the scenario analyzer library
# by showing some simple plots of the results of the demo simulation

if __name__ == "__main__":

    sim_name = 'Countryside_pv_0.5_hp_0.5_ev_1.0_summer'

    limits = {
        "winter": [1573513200, 1574118000],
        "summer": [1563832800, 1564437600],
        "transition": [1569276000, 1569880800],
    }

    limit = [limits[season] for season in limits if season in sim_name][0]

    analysis = lemlab.ScenarioAnalyzer(path_results=f"../simulation_results/{sim_name}",
                                       show_figures=True,
                                       save_figures=False)

    analysis.plot_detailed_virtual_feeder_flow(limits=limit)
    # analysis.plot_virtual_feeder_flow(limits=limit)

    # analysis.plot_mcp()

    # for pro in range(1, 12):
    #    analysis.plot_household(id_user=pro)