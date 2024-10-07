import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats



# Colors are chosen to accommodate colorblind people
COLORS = {
    "pv": "#FFA500",
    "ev": "#1E88E5",
    "hp": "#DE0051",
}


def calc_peakpower(info: dict, meter_file: str, readings_file: str):
    # Get IDs of all main meters (1=utility with multiple submeters, 2=utility meter)
    df_meter_info = pd.read_csv(meter_file, index_col=0)
    list_main_meters = list(df_meter_info[df_meter_info['type_meter'].isin(
        ["grid meter", "virtual grid meter"])]['id_meter'].unique())

    # Get power flows of all meters in list_main_meters
    df_meter_readings_delta = pd.read_csv(readings_file, index_col=0)

    # Get power flows of all meters in list_main_meters
    df_results = df_meter_readings_delta[df_meter_readings_delta['id_meter'].isin(list_main_meters)]

    # Group by timestamp and sum all power flows
    df_results = df_results.groupby('ts_delivery').sum(numeric_only=True)
    df_results['energy_in'] = df_results['energy_in'].astype('int64') * 1 / 250  # to convert from Wh per 15 min to kW
    df_results['energy_out'] = df_results['energy_out'].astype('int64') * 1 / 250  # to convert from Wh per 15 min to kW

    # Rename columns
    df_results.rename(columns={'energy_in': "negative_flow_kW", 'energy_out': "positive_flow_kW"}, inplace=True)
    df_results["negative_flow_kW"] = - df_results["negative_flow_kW"]
    df_results["net_flow_kW"] = df_results["positive_flow_kW"] + df_results["negative_flow_kW"]
    try:
        df_results.drop(columns=['id_meter'], inplace=True)
    except KeyError:
        pass
    df_results = df_results.sort_index()

    # Calculate peak power
    max_magnitude_index = df_results['net_flow_kW'].abs().idxmax()
    peak_power = df_results.loc[max_magnitude_index, 'net_flow_kW']

    # Add peak power to info dictionary
    info['peak_power'] = round(peak_power, 2)

    return info


def calc_avgenergyprice(info: dict, market_file: str):
    # Get the market data
    df_market = pd.read_csv(market_file, index_col=0)

    df_market["costs_€"] = df_market['qty_energy_traded'] * df_market['price_energy_market_uniform'] * \
                           1 / 1000000000  # to convert from sigma to €

    # Create dataframe for the results
    df_results = pd.DataFrame(columns=["timestamp", "total_cost_€", "total_energy_kWh", "avg_price_€/kWh"])

    # Gather values form df_market_results and calculate weighted average
    df_results["timestamp"] = sorted(df_market['ts_delivery'].unique())
    df_results.set_index("timestamp", inplace=True)
    df_temp = df_market.groupby('ts_delivery').sum(numeric_only=True)
    df_results["total_cost_€"] = df_temp["costs_€"]
    df_results["total_energy_kWh"] = df_temp['qty_energy_traded'] * 1 / 1000  # to convert from Wh to kWh
    df_results["avg_price_€/kWh"] = df_results["total_cost_€"] / df_results["total_energy_kWh"]

    # Add average energy price to info dictionary
    info['average_energy_price'] = df_results["total_cost_€"].sum() / df_results["total_energy_kWh"].sum()
    info['average_energy_price'] = round(info['average_energy_price'], 4)

    return info


def create_data(src: str):
    # TODO: Limit data to 1 week
    # Initiate a list to store information about each scenario
    scenario_info_list = []

    # Info about week and season correlation
    season_info = {
        1: 'Winter',
        2: 'Summer',
        3: 'Spring/Autumn',
    }

    # Iterate through all scenario folders
    for scenario_folder in next(os.walk(src))[1]:
        # Path to scenario folder
        scenario_path = os.path.join(src, scenario_folder)

        # Split the scenario_folder string
        split_values = scenario_folder.split("_")

        # Extract relevant values using dictionary keys
        info = {
            "grid_topology": split_values[0],
            "pv_penetration": float(split_values[2]),
            "hp_penetration": float(split_values[4]),
            "ev_penetration": float(split_values[6]),
            "week": int(split_values[7][4:]),
            "season": None,
            "lem": False if "woLEM" in scenario_folder else True,
            "average_energy_price": None,
            "peak_power": None,
        }

        # Determine season based on the week number
        info['season'] = season_info[info['week']]

        # Path to scenario data
        data_path = os.path.join(scenario_path, "db_snapshot")

        # Specify the path to the transactions file
        meter_file = os.path.join(data_path, "info_meter.csv")
        readings_file = os.path.join(data_path, "readings_meter_delta.csv")
        info = calc_peakpower(info, meter_file, readings_file)

        # Calculate average energy price and peak power
        market_file = os.path.join(data_path, "results_market_ex_ante_pda.csv")
        info = calc_avgenergyprice(info, market_file)

        # Add info to the list
        scenario_info_list.append(info)

    # Create a DataFrame from the list
    df = pd.DataFrame(scenario_info_list)

    return df


# Create a function to draw pie charts at specified coordinates
def create_pie_chart(ax, x: float, y: float, val: float, color: str, radius: float = 0.1, plot_inner_circle: bool = False):
    # Plot background of pie chart as white circle
    ax.scatter(x, y, marker='o', s=1000 * radius, c='white', edgecolors='black', linewidths=0.1)

    # Create the marker for the pie chart
    mx = [0] + np.cos(np.linspace(540/360, -2 * np.pi * val + 540/360, 100)).tolist()
    my = [0] + np.sin(np.linspace(540/360, -2 * np.pi * val + 540/360, 100)).tolist()
    xy = np.column_stack([mx, my])

    # Plot share of the pie chart
    ax.scatter(x, y, marker=xy, s=1000 * radius, c=color)

    # Plot inner circle of pie chart as white circle to create doughnut
    if plot_inner_circle:
        ax.scatter(x, y, marker='o', s=1000 * radius / 4, c='white', edgecolors='black', linewidths=0.1)


def create_price_power_plot(df: pd.DataFrame, save: bool = False):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():
        df_top = df[df["grid_topology"] == topology]

        # TODO: Add a part that takes the weeks into account

        # Create a dataframe that contains all rows with LEM and without LEM
        df_with_lem = df_top[df_top["lem"] == True]
        df_wo_lem = df_top[df_top["lem"] == False]

        # Merge the DataFrames based on the columns representing PV, HP, and EV penetration
        df_m = pd.merge(df_with_lem, df_wo_lem, on=["pv_penetration", "hp_penetration", "ev_penetration"],
                      suffixes=('_with_lem', '_without_lem'))

        # Compute the ratios for average_energy_price and peak_power
        df_m["average_energy_price_ratio"] = df_m["average_energy_price_with_lem"] / df_m["average_energy_price_without_lem"]
        df_m["peak_power_ratio"] = df_m["peak_power_with_lem"] / df_m["peak_power_without_lem"]

        # Drop the unnecessary columns and keep the relevant columns
        df_m = df_m[["grid_topology_with_lem", "pv_penetration", "hp_penetration", "ev_penetration",
                 "average_energy_price_ratio", "peak_power_ratio"]]

        # Rename the 'grid_topology_with_lem' column to 'grid_topology'
        df_m.rename(columns={"grid_topology_with_lem": "grid_topology"}, inplace=True)

        # Create the scatter plot with customized markers
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Plot the pie charts for each scenario
        for i, row in df_m.iterrows():

            x_val = row["average_energy_price_ratio"]
            y_val = row["peak_power_ratio"]

            # Create the pie charts for each device
            create_pie_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5*2)
            create_pie_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25*2)
            create_pie_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1*2,
                             plot_inner_circle=True)

        # Add the dashed line and the text
        # ax.plot(np.linspace(0, 2, 10), np.linspace(0, 2, 10), linestyle='--', color='black', linewidth=1)
        # ax.text(1.1, 1, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$", ha='right', va='top', fontsize=10)

        # Add matrix lines and text
        ax.plot(np.linspace(0, 2, 10), np.linspace(1, 1, 10), linestyle='--', color='black', linewidth=1)
        ax.plot(np.linspace(1, 1, 10), np.linspace(0, 2, 10), linestyle='--', color='black', linewidth=1)
        ax.text(0.1, 0.1, f"lower costs \nlower power", ha='left', va='bottom', fontsize=12)
        ax.text(0.1, 1.9, f"lower costs \nhigher power", ha='left', va='top', fontsize=12)
        ax.text(1.9, 0.1, f"higher costs \nlower power", ha='right', va='bottom', fontsize=12)
        ax.text(1.9, 1.9, f"higher costs \nhigher power", ha='right', va='top', fontsize=12)


        # Set axis labels and title
        plt.xlabel("Average Energy Price - with LEM / without LEM")
        plt.ylabel("Peak Power - with LEM / without LEM")
        # plt.title("Comparison of Parameters with and without LEM")

        # Create legend handles and labels for the customized markers
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color=COLORS['pv'], label='PV', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color=COLORS['ev'], label='EV', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color=COLORS['hp'], label='HP', markersize=10, linestyle='None'),
            # plt.Line2D([0], [0], color='black', label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
            #            linestyle='--'),
        ]

        # Show the legend with customized markers
        ax.legend(handles=legend_elements, loc='right')

        # Set x and y axis and aspect ratio
        ax.set_xlim(0, 2)  # max(df_m["average_energy_price_ratio"].max(), df_m["peak_power_ratio"].max()) + 0.1)
        ax.set_ylim(0, 2)  #max(df_m["average_energy_price_ratio"].max(), df_m["peak_power_ratio"].max()) + 0.1)
        ax.set(aspect="equal")

        # Tighten the layout
        plt.tight_layout()

        if save:
            # Save the plot
            plt.savefig(f'./figures/{topology}_price_power.png', dpi=300)

        # Show the plot
        plt.show()


def create_price_plot(df: pd.DataFrame, save: bool = False):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():
        df_top = df[df["grid_topology"] == topology]

        # TODO: Add a part that takes the weeks into account

        # Create a dataframe that contains all rows with LEM and without LEM
        df_with_lem = df_top[df_top["lem"] == True]
        df_wo_lem = df_top[df_top["lem"] == False]

        # Merge the DataFrames based on the columns representing PV, HP, and EV penetration
        df_m = pd.merge(df_with_lem, df_wo_lem, on=["pv_penetration", "hp_penetration", "ev_penetration"],
                        suffixes=('_with_lem', '_without_lem'))

        # Compute the ratios for average_energy_price and peak_power
        df_m["average_energy_price_ratio"] = df_m["average_energy_price_with_lem"] / df_m["average_energy_price_without_lem"]
        # df["peak_power_ratio"] = df["peak_power_with_lem"] / df["peak_power_without_lem"]

        # Drop the unnecessary columns and keep the relevant columns
        df_m = df_m[["grid_topology_with_lem", "pv_penetration", "hp_penetration", "ev_penetration",
                 "average_energy_price_ratio", "average_energy_price_with_lem", "average_energy_price_without_lem"]]

        # Rename the 'grid_topology_with_lem' column to 'grid_topology'
        df_m.rename(columns={"grid_topology_with_lem": "grid_topology"}, inplace=True)

        # Create the scatter plot with customized markers
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Plot the pie charts for each scenario
        for i, row in df_m.iterrows():

            x_val = row["average_energy_price_with_lem"]
            y_val = row["average_energy_price_ratio"]

            # Create the pie charts for each device
            create_pie_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5*2)
            create_pie_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25*2)
            create_pie_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1*2,
                             plot_inner_circle=True)

        # Add the dashed line and the text
        ax.plot(np.linspace(0, 0.3, 10), np.linspace(1, 1, 10), linestyle='--', color='black', linewidth=1)
        # ax.text(1.1, 1, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$", ha='right', va='top', fontsize=10)


        # Set axis labels and title
        plt.xlabel("Average Energy Price - with LEM")
        plt.ylabel("Average Energy Price - with LEM / without LEM")
        # plt.title("Comparison of Parameters with and without LEM")

        # Create legend handles and labels for the customized markers
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color=COLORS['pv'], label='PV', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color=COLORS['ev'], label='EV', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color=COLORS['hp'], label='HP', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], color='black', label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
                       linestyle='--'),
        ]

        # Show the legend with customized markers
        ax.legend(handles=legend_elements, loc='upper left')

        # Set x and y axis and aspect ratio
        ax.set_xlim(0, df_m["average_energy_price_with_lem"].max() + 0.1)
        ax.set_ylim(0, 1.2)

        # Add ticks and tick labels to axes
        ax.set_xticks(np.append(np.arange(0, max(ax.get_xticks()), 0.05), [0.24]))
        # ax.set_xticks(np.arange(0, max(ax.get_xticks()), 0.05))
        # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        ax.set_xticklabels([f'{x:.2f} €/kWh' for x in ax.get_xticks()])
        # ax.set_yticklabels(["0", "0.2", "0.4", "0.6", "0.8", "1.0", "1.2"])
        # ax.set(aspect="equal")

        # Tighten the layout
        plt.tight_layout()

        if save:
            # Save the plot
            plt.savefig(f'./figures/{topology}_price.png', dpi=300)

        # Show the plot
        plt.show()


def create_power_plot(df: pd.DataFrame, save: bool = False):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():
        df_top = df[df["grid_topology"] == topology]

        # TODO: Add a part that takes the weeks into account

        # Create a dataframe that contains all rows with LEM and without LEM
        df_with_lem = df_top[df_top["lem"] == True]
        df_wo_lem = df_top[df_top["lem"] == False]

        # Merge the DataFrames based on the columns representing PV, HP, and EV penetration
        df_m = pd.merge(df_with_lem, df_wo_lem, on=["pv_penetration", "hp_penetration", "ev_penetration"],
                        suffixes=('_with_lem', '_without_lem'))

        # Compute the ratios for average_energy_price and peak_power
        # df["average_energy_price_ratio"] = df["average_energy_price_with_lem"] / df["average_energy_price_without_lem"]
        df_m["peak_power_ratio"] = df_m["peak_power_with_lem"] / df_m["peak_power_without_lem"]

        # Drop the unnecessary columns and keep the relevant columns
        df_m = df_m[["grid_topology_with_lem", "pv_penetration", "hp_penetration", "ev_penetration",
                 "peak_power_ratio", "peak_power_with_lem", "peak_power_without_lem"]]

        # Rename the 'grid_topology_with_lem' column to 'grid_topology'
        df_m.rename(columns={"grid_topology_with_lem": "grid_topology"}, inplace=True)

        # Create the scatter plot with customized markers
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Plot the pie charts for each scenario
        for i, row in df_m.iterrows():

            x_val = row["peak_power_without_lem"]
            y_val = row["peak_power_with_lem"]

            # Create the pie charts for each device
            create_pie_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5*2)
            create_pie_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25*2)
            create_pie_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1*2,
                             plot_inner_circle=True)

        # Add the dashed line and the text
        ax.plot(np.linspace(0, max(df_m["peak_power_without_lem"].max(), df_m["peak_power_with_lem"].max()) + 5, 10),
                np.linspace(0, max(df_m["peak_power_without_lem"].max(), df_m["peak_power_with_lem"].max()) + 5, 10),
                linestyle='--', color='black', linewidth=1)
        # ax.text(1.1, 1, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$", ha='right', va='top', fontsize=10)


        # Set axis labels and title
        plt.xlabel("Peak Power - without LEM")
        plt.ylabel("Peak Power - with LEM")
        # plt.title("Comparison of Parameters with and without LEM")

        # Create legend handles and labels for the customized markers
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color=COLORS['pv'], label='PV', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color=COLORS['ev'], label='EV', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color=COLORS['hp'], label='HP', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], color='black', label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
                       linestyle='--'),
        ]

        # Show the legend with customized markers
        ax.legend(handles=legend_elements, loc='upper left')

        # Set x and y axis and aspect ratio
        ax.set_xlim(0, max(df_m["peak_power_without_lem"].max(), df_m["peak_power_with_lem"].max()) + 5)
        ax.set_ylim(0, max(df_m["peak_power_without_lem"].max(), df_m["peak_power_with_lem"].max()) + 5)

        # Add ticks and tick labels to axes
        ax.set_xticks(np.arange(0, max(ax.get_xticks()), 5))
        ax.set_yticks(np.arange(0, max(ax.get_yticks()), 5))
        # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
        ax.set_xticklabels([f'{x:.0f} kW' for x in ax.get_xticks()])
        ax.set_yticklabels([f'{y:.0f} kW' for y in ax.get_yticks()])
        ax.set(aspect="equal")

        # Tighten the layout
        plt.tight_layout()

        if save:
            # Save the plot
            plt.savefig(f'./figures/{topology}_power.png', dpi=300)

        # Show the plot
        plt.show()


def create_indepth_plot(df: pd.DataFrame, save: bool = False):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():
        df_top = df[df["grid_topology"] == topology]

        # TODO: Add a part that takes the weeks into account

        # Create a dataframe that contains all rows with LEM and without LEM
        df_with_lem = df_top[df_top["lem"] == True]
        df_wo_lem = df_top[df_top["lem"] == False]

        # Merge the DataFrames based on the columns representing PV, HP, and EV penetration
        df_m = pd.merge(df_with_lem, df_wo_lem, on=["pv_penetration", "hp_penetration", "ev_penetration"],
                        suffixes=('_with_lem', '_without_lem'))

        # Compute the ratios for average_energy_price and peak_power
        df_m["average_energy_price_ratio"] = df_m["average_energy_price_with_lem"] / df_m["average_energy_price_without_lem"]
        df_m["peak_power_ratio"] = df_m["peak_power_with_lem"] / df_m["peak_power_without_lem"]

        # Rename the 'grid_topology_with_lem' column to 'grid_topology' and drop unnecessary columns
        df_m.rename(columns={"grid_topology_with_lem": "grid_topology"}, inplace=True)
        df_m.drop(columns='grid_topology_without_lem', inplace=True)

        # Define the grid topologies and device types
        device_types = COLORS.keys()

        # Create the figure with subplots
        fig, axs = plt.subplots(3, 2, figsize=(12, 18))
        fig.subplots_adjust(hspace=0.4)

        # Plot for average energy price
        for i, device in enumerate(device_types):
            ax = axs[i, 0]
            data = [df_m.loc[df_m[device + "_penetration"] == x, "average_energy_price_ratio"].values
                    for x in df_m[device + "_penetration"].unique()]
            c = COLORS[device]
            ax.boxplot(data, positions=df_m[device + "_penetration"].unique(), widths=0.1, patch_artist=True,
                       boxprops=dict(facecolor='white', color=c),
                       capprops=dict(color=c),
                       whiskerprops=dict(color=c),
                       flierprops=dict(color=c, markeredgecolor=c),
                       medianprops=dict(color=c),)
            avg = [sum(x) / len(x) for x in data]
            z = np.polyfit(df_m[device + "_penetration"].unique(), avg, 1)
            p = np.poly1d(z)
            ax.plot(np.linspace(-.1, 1.1, 100), p(np.linspace(-.1, 1.1, 100)), color='black', linewidth=2, alpha=0.7)
            ax.text(1, 1.5, f"y = {z[0]:.2f}x + {z[1]:.2f}", ha='right', va='top', fontsize=10)
            res = stats.linregress(np.linspace(-.1, 1.1, 100), p(np.linspace(-.1, 1.1, 100)))
            ax.plot(np.linspace(-.1, 1.1, 10), np.linspace(1, 1, 10), linestyle='--', color='black', linewidth=1)
            ax.set_xlabel(f"{device.upper()} Penetration Level")
            ax.set_ylabel("")
            ax.set_title(f"Effect of {device.upper()} on Average Energy Price")
            ax.set_xlim(-.1, 1.1)
            ax.set_ylim(0)

        # Plot for peak power
        for i, device in enumerate(device_types):
            ax = axs[i, 1]
            data = [df_m.loc[df_m[device + "_penetration"] == x, "peak_power_ratio"].values
                    for x in df_m[device + "_penetration"].unique()]
            c = COLORS[device]
            ax.boxplot(data, positions=df_m[device + "_penetration"].unique(), widths=0.1, patch_artist=True,
                       boxprops=dict(facecolor='white', color=c),
                       capprops=dict(color=c),
                       whiskerprops=dict(color=c),
                       flierprops=dict(color=c, markeredgecolor=c),
                       medianprops=dict(color=c),)
            avg = [sum(x) / len(x) for x in data]
            z = np.polyfit(df_m[device + "_penetration"].unique(), avg, 1)
            p = np.poly1d(z)
            ax.plot(np.linspace(-.1, 1.1, 100), p(np.linspace(-.1, 1.1, 100)), color='black', linewidth=2, alpha=0.7)
            ax.text(1, 1.5, f"y = {z[0]:.2f}x + {z[1]:.2f}", ha='right', va='top', fontsize=10)
            res = stats.linregress(np.linspace(-.1, 1.1, 100), p(np.linspace(-.1, 1.1, 100)))
            ax.plot(np.linspace(-.1, 1.1, 10), np.linspace(1, 1, 10), linestyle='--', color='black', linewidth=1)
            ax.set_xlabel(f"{device.upper()} Penetration Level")
            ax.set_ylabel("")
            ax.set_title(f"Effect of {device.upper()} on Peak Power")
            ax.set_xlim(-.1, 1.1)
            ax.set_ylim(0)

        # Tighten the layout
        plt.tight_layout()

        if save:
            # Save the plot
            plt.savefig(f'./figures/{topology}_indepth.png', dpi=300)

        # Show the plot
        plt.show()

if __name__ == "__main__":
    # Folder that contains simulation results
    path = './results'

    # # Create the file containing all information
    # df = create_data(src=path)
    #
    # # Save the dataframe to a csv file
    # df.to_csv('./validation_dataset.csv')
    #
    # # Load the dataframe from a csv file
    # df = pd.read_csv('./validation_dataset.csv', index_col=0)

    # DUMMY DATA
    # Define the grid topologies and penetration levels
    grid_topologies = ["Countryside", "Rural", "Suburban", "Urban"]
    penetration_levels = [0, 0.25, 0.5, 0.75, 1]

    # Create an empty list to store the generated data
    data_with_lem = []
    data_wolem = []

    # Generate data for each grid topology and penetration level with and without LEM
    for topology in grid_topologies:
        for pv_penetration in penetration_levels:
            for hp_penetration in penetration_levels:
                for ev_penetration in penetration_levels:
                    for lem_status in [True, False]:
                        data = {
                            "grid_topology": topology,
                            "pv_penetration": pv_penetration,
                            "hp_penetration": hp_penetration,
                            "ev_penetration": ev_penetration,
                            "lem": lem_status,
                            "average_energy_price": np.random.uniform(0.1, 0.2),
                            "peak_power": np.random.randint(20, 40),
                        }
                        if lem_status:
                            data_with_lem.append(data)
                        else:
                            data_wolem.append(data)

    # Create dataframes for LEM and non-LEM cases
    df_with_lem = pd.DataFrame(data_with_lem)
    df_wolem = pd.DataFrame(data_wolem)

    # Combine both dataframes
    df = pd.concat([df_with_lem, df_wolem], ignore_index=True)

    # Create plot comparing price and power with and without LEM
    create_price_power_plot(df, save=True)

    # # Create plot comparing price with and without LEM
    # create_price_plot(df)#, save=True)
    #
    # # Create plot comparing power with and without LEM
    # create_power_plot(df)#, save=True)
    #
    # # Create in-depth plot for each device type
    # create_indepth_plot(df)#, save=True)
