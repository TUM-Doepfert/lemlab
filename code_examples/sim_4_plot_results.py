import lemlab

# this example demonstrates the use of the scenario analyzer library
# by showing some simple plots of the results of the demo simulation

if __name__ == "__main__":
    params = {
        "topology": "urban",
        "pv": 0.25 * 4,
        "ev": 0.25 * 4,
        "hp": 0.25 * 4,
        "season": "transition",
        "wLEM": False,
    }

    seasons = ['summer', 'transition', 'winter']
    # seasons = ['transition']
    wlem = [True, False]

    for season in seasons:
        for lem in wlem:
            params['season'] = season
            params['wLEM'] = lem

            if params["wLEM"]:
                sim_name = f"{params['topology']}/wLEM_mpc_naive/{params['topology'].capitalize()}_pv_{params['pv']}_hp_{params['hp']}_ev_{params['ev']}_{params['season']}_wLEM_mpc_naive"
            else:
                sim_name = f"{params['topology']}/woLEM_rtc_naive/{params['topology'].capitalize()}_pv_{params['pv']}_hp_{params['hp']}_ev_{params['ev']}_{params['season']}_woLEM_rtc_naive"

            limits = {
                "winter": [1573513200, 1574118000],
                "summer": [1563832800, 1564437600],
                "transition": [1569276000, 1569880800],
            }

            limit = [limits[season] for season in limits if season in sim_name][0]

            analysis = lemlab.ScenarioAnalyzer(path_results=f"../simulation_results/{sim_name}",
                                               show_figures=True,
                                               save_figures=False)

            # Plot powerflow
            analysis.plot_detailed_virtual_feeder_flow(limits=limit)

            # Plot market clearing price (only for wLEM)
            # if 'woLEM' not in sim_name:
            #   analysis.plot_mcp(limits=limit)


            # Plot household
            # analysis.plot_household(id_user=3)
            # Plots households
            #for pro in range(1, 12):
            #    analysis.plot_household(id_user=pro)
