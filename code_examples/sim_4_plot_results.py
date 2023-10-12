import lemlab

# this example demonstrates the use of the scenario analyzer library
# by showing some simple plots of the results of the demo simulation

if __name__ == "__main__":

    sim_name = 'Countryside_pv_1.0_hp_1.0_ev_1.0_winter_woLEM'

    analysis = lemlab.ScenarioAnalyzer(path_results=f"../simulation_results/{sim_name}",
                                       show_figures=True,
                                       save_figures=False)

    analysis.plot_detailed_virtual_feeder_flow()

    #analysis.plot_virtual_feeder_flow()
    #for pro in range(1, 12):
    #    analysis.plot_household(id_user=pro)
    #
    # for pro in range(1, 12):
    #     sim_name = "Rural_pv_1.0_hp_0.25_ev_0.0_summer"
    #
    #     analysis = lemlab.ScenarioAnalyzer(path_results=f"../all_lemlab_results/Dorf/results/{sim_name}",
    #                                        show_figures=True,
    #                                        save_figures=False)
    #     try:
    #        analysis.plot_household(id_user=pro)
    #     except IndexError:
    #        pass
    #
    #     sim_name = "Land_pv_0.5_hp_0.5_ev_0.5_week1_woLEM"
    #
    #     analysis = lemlab.ScenarioAnalyzer(path_results=f"../simulation_results/{sim_name}",
    #                                        show_figures=True,
    #                                        save_figures=False)
    #     try:
    #        analysis.plot_household(id_user=pro)
    #     except IndexError:
    #        pass