import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
from scipy import stats
from pprint import pprint
from tqdm import tqdm
import logging
import math

# Configure the logging module
logging.basicConfig(
    filename='create_data.log',
    level=logging.ERROR,
    format='%(asctime)s %(levelname)s: %(message)s',
    datefmt='%Y-%m-%d %H:%M:%S'
)



# Colors are chosen to accommodate colorblind people
COLORS = {
    "pv": "#FFA500",
    "ev": "#1E88E5",
    "hp": "#DE0051",
}


def calc_peakpower(info: dict, meter_file: str, readings_file: str, limits: list = None):
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

    # Calculate peak power and add to info dictionary
    max_magnitude_index = df_results['net_flow_kW'].abs().idxmax()
    info['peak_power'] = df_results.loc[max_magnitude_index, 'net_flow_kW']
    # info['peak_power'] = round(info['peak_power'], 2)  # Left out now to have more precise values

    return info


def calc_avgenergyprice(info: dict, market_file: str, limits: list = None):
    # Get the market data
    df_market = pd.read_csv(market_file, index_col=0)

    # Limit the market data to the specified limits if available
    if limits:
        df_market = df_market[(limits[0] <= df_market['ts_delivery']) & (df_market['ts_delivery'] < limits[1])]

    # Drop all rows where qty_energy is positive as it is only about the bought energy
    df_market = df_market[df_market['qty_energy'] < 0]

    # Drop all rows where the retailer bought energy as it would distort the average energy price for the prosumers
    df_market = df_market[df_market['id_user'] != 'retailer01']

    # Drop all rows that contain levies as they are not part of the energy price
    df_market = df_market[~df_market['type_transaction'].str.contains('levies')]

    # Sum the columns to obtain the total energy traded and the total costs
    df_market = df_market.sum(numeric_only=True)

    # Calculate the average energy price, convert it from sigma to € and add it to the info dictionary
    info['average_energy_price'] = df_market['delta_balance'] / df_market['qty_energy'] * 1 / 1000000

    return info


def create_data(src: str, limits_dict: dict = None):
    # Initiate a list to store information about each scenario
    scenario_info_list = []

    # Get all scenario folders
    topologies = next(os.walk(src))[1]

    # Outer tqdm bar
    for topology in tqdm(topologies, desc="Creating dataset...", unit='topology', position=0, leave=True):
        topology_path = os.path.join(src, topology)

        # Middle tqdm bar
        markets = next(os.walk(topology_path))[1]
        for market in tqdm(markets, desc=f"Processing {topology}...", unit='market', position=1, leave=False):
            market_path = os.path.join(topology_path, market)

            # Inner tqdm bar
            scenario_folders = next(os.walk(market_path))[1]
            for scenario_folder in tqdm(scenario_folders, desc=f"Processing {market}...", unit='scenario', position=2,
                                        leave=False):
                # Path to scenario folder
                scenario_path = os.path.join(market_path, scenario_folder)

                # Split the scenario_folder string
                split_values = scenario_folder.split("_")

                # Extract relevant values using dictionary keys
                info = {
                    "grid_topology": split_values[0],
                    "pv_penetration": float(split_values[2]),
                    "hp_penetration": float(split_values[4]),
                    "ev_penetration": float(split_values[6]),
                    "season": split_values[7].lower(),
                    "lem": False if "woLEM" in scenario_folder else True,
                    "average_energy_price": None,
                    "peak_power": None,
                }

                # Set the limits that are to be included in the calculations
                if limits_dict:
                    limits = limits_dict[info["season"]]
                else:
                    limits = None

                # Path to scenario data
                data_path = os.path.join(scenario_path, "db_snapshot")

                # Specify the paths to the relevant files
                meter_file = os.path.join(data_path, "info_meter.csv")
                readings_file = os.path.join(data_path, "readings_meter_delta.csv")
                market_file = os.path.join(data_path, "logs_transactions.csv")

                # Calculate average energy price and peak power
                try:
                    info = calc_avgenergyprice(info, market_file, limits)
                    info = calc_peakpower(info, meter_file, readings_file, limits)
                except Exception as e:
                    # Log error
                    logging.error(f"Error in {scenario_folder}: {e}")

                # Add info to the list
                scenario_info_list.append(info)

    # Create a DataFrame from the list
    df = pd.DataFrame(scenario_info_list)

    return df


# Create a function to draw pie charts at specified coordinates
def create_pie_chart(ax, x: float, y: float, val: float, color: str, radius: float = 0.1, plot_inner_circle: bool = False):

    # Alpha value for the pie chart
    alpha = 0.7

    # Plot background of pie chart as white circle
    ax.scatter(x, y, marker='o', s=1000 * radius, c='white', edgecolors='black', linewidths=0.1, alpha=alpha)

    # Create the marker for the pie chart
    mx = [0] + np.cos(np.linspace(540/360, -2 * np.pi * val + 540/360, 100)).tolist()
    my = [0] + np.sin(np.linspace(540/360, -2 * np.pi * val + 540/360, 100)).tolist()
    xy = np.column_stack([mx, my])

    # Plot share of the pie chart
    ax.scatter(x, y, marker=xy, s=1000 * radius, c=color, alpha=alpha)

    # Plot inner circle of pie chart as white circle to create doughnut
    if plot_inner_circle:
        ax.scatter(x, y, marker='o', s=1000 * radius / 4, c='white', edgecolors='black', linewidths=0.1, alpha=alpha)


def format_df_for_plots(df: pd.DataFrame, topology, weights: dict = None):

    # Reduce the dataframe to the specified topology
    df_top = df[df["grid_topology"] == topology]


    # Weight the average energy price
    sum_weights = 0
    if weights:
        for season, weight in weights.items():
            df_top.loc[df_top['season'] == season, 'average_energy_price'] *= weight
            sum_weights += weight

    # TODO: Continue from here. Filter first that only the scenarios with all six seasons are included. The rest should be fine.

    # Create a dataframe that contains all rows with LEM and without LEM
    df = (df_top.groupby(['grid_topology', 'pv_penetration', 'hp_penetration', 'ev_penetration', 'lem']).
          filter(lambda group: set(group['season']) == {'summer', 'transition', 'winter'}).
          # filter(lambda group: set(group['lem']) == {True, False}).
          reset_index(drop=True))
    df = (df.groupby(['grid_topology', 'pv_penetration', 'hp_penetration', 'ev_penetration']).
          # filter(lambda group: set(group['season']) == {'summer', 'transition', 'winter'}).
          filter(lambda group: set(group['lem']) == {True, False}).
          reset_index(drop=True))
    df['peak_power'].fillna(0, inplace=True)

    # Grouping by the relevant columns
    grouped = df.groupby(['grid_topology', 'pv_penetration', 'hp_penetration', 'ev_penetration', 'lem'])

    # Calculate sum of average_energy_price
    sum_energy_price = grouped['average_energy_price'].sum() / sum_weights

    # Find the row with max absolute peak_power
    idx = grouped['peak_power'].apply(lambda x: x.abs().idxmax())
    max_peak_power = df.loc[idx, 'peak_power']
    max_peak_season = df.loc[idx, 'season']

    # Construct the resulting dataframe
    result_df = sum_energy_price.reset_index()
    result_df['peak_power'] = max_peak_power.values
    result_df['season'] = max_peak_season.values

    # Create a dataframe that contains all rows with LEM and without LEM
    df_with_lem = result_df[result_df["lem"] == True]
    df_wo_lem = result_df[result_df["lem"] == False]

    # Merge the DataFrames based on the columns representing PV, HP, and EV penetration
    df_m = pd.merge(df_with_lem, df_wo_lem, on=["pv_penetration", "hp_penetration", "ev_penetration"],
                    suffixes=('_with_lem', '_without_lem'))

    # Compute the ratios for average_energy_price and peak_power
    df_m["average_energy_price_ratio"] = df_m["average_energy_price_with_lem"] / df_m[
        "average_energy_price_without_lem"]
    df_m["peak_power_ratio"] = abs(df_m["peak_power_with_lem"] / df_m["peak_power_without_lem"])

    # # Drop the unnecessary columns and keep the relevant columns
    # df_m = df_m[["grid_topology_with_lem", "pv_penetration", "hp_penetration", "ev_penetration",
    #          "average_energy_price_ratio", "peak_power_ratio"]]

    # Rename the 'grid_topology_with_lem' column to 'grid_topology'
    df_m.rename(columns={"grid_topology_with_lem": "grid_topology"}, inplace=True)

    # Save results to csv
    df_m.to_csv(f'./results/{topology}_price_power.csv')

    return df_m

def create_price_power_plot(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        # Format the dataframe for plotting
        df_m = format_df_for_plots(df, topology, weights)

        # Reduce the dataframe to the specified penetration rates
        rates = [0.0, 0.5, 1.0]
        df_m = df_m[df_m["pv_penetration"].isin(rates)
                    & df_m["hp_penetration"].isin(rates)
                    & df_m["ev_penetration"].isin(rates)]

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
        # plt.show()

        plt.close()

def create_price_plot(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        # Format the dataframe for plotting
        df_m = format_df_for_plots(df, topology, weights)

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
            # plt.Line2D([0], [0], color='black', label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
            #            linestyle='--'),
        ]

        # Add a vertical line that represents the average energy price without LEM
        ax.axvline(df_m["average_energy_price_without_lem"].mean(), color='black', linestyle='--', linewidth=1)
        # Add text next to it to specify what the line stands for
        ax.text(df_m["average_energy_price_without_lem"].mean() + 0.01, 0.1, "Average Energy Price without LEM",
                ha='left', va='bottom', fontsize=12)

        # Show the legend with customized markers
        ax.legend(handles=legend_elements, loc='upper left')

        # Set x and y axis and aspect ratio
        ax.set_xlim(0, df_m["average_energy_price_with_lem"].max() + 0.1)
        ax.set_ylim(0, 1.2)

        # Add ticks and tick labels to axes
        ax.set_xticks(np.arange(0, max(ax.get_xticks()), 0.05))
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
        # plt.show()

def plot_power_plot(df_m: pd.DataFrame, topology, rate, comp, save: bool = False, max_power: float = None):
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Plot the pie charts for each scenario
    for i, row in df_m.iterrows():
        x_val = abs(row["peak_power_without_lem"])
        y_val = abs(row["peak_power_with_lem"])

        # Create the pie charts for each device
        create_pie_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5 * 2)
        create_pie_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25 * 2)
        create_pie_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1 * 2,
                         plot_inner_circle=True)

    # Add the dashed lines and the text
    if not max_power:
        max_power = max(abs(df_m["peak_power_without_lem"]).max(), abs(df_m["peak_power_with_lem"]).max())
    # LEM/noLEM = 1
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power, 10),
            linestyle='--', color=(0.4, 0.4, 0.4), linewidth=2)
    # ax.text(max_power, max_power, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
    #         ha='left', va='center', fontsize=10)
    # LEM/noLEM = 0.75
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power * 0.75, 10),
            linestyle='--', color=(0.4, 0.5, 0.6), linewidth=2)
    # ax.text(max_power, max_power * 0.75, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.75$",
    #         ha='left', va='center',                fontsize=10)
    # LEM/noLEM = 0.5
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power * 0.5, 10),
            linestyle='--', color=(0.5, 0.6, 0.7), linewidth=2)
    # ax.text(max_power, max_power * 0.5, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.5$",
    #         ha='left', va='center',                fontsize=10)
    # LEM/noLEM = 0.25
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power * 0.25, 10),
            linestyle='--', color=(0.6, 0.7, 0.8), linewidth=2)
    # ax.text(max_power, max_power * 0.25, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.25$",
    #         ha='left', va='center',                fontsize=10)

    # Set axis labels and title
    plt.xlabel("Absolute Peak Power - without LEM")
    plt.ylabel("Absolute Peak Power - with LEM")
    # plt.title("Comparison of Parameters with and without LEM")

    # Create legend handles and labels for the customized markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=COLORS['pv'], label='PV', markersize=10, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color=COLORS['ev'], label='EV', markersize=10, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color=COLORS['hp'], label='HP', markersize=10, linestyle='None'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
                   color=(0.4, 0.4, 0.4), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.75$",
                   color=(0.4, 0.5, 0.6), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.5$",
                   color=(0.5, 0.6, 0.7), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.25$",
                   color=(0.6, 0.7, 0.8), linestyle='--'),
    ]

    # Show the legend with customized markers
    ax.legend(handles=legend_elements, loc='upper left')

    # Stepsize in kW
    max_ticks = 12  # max number of ticks on x and y axis
    ticks = np.inf  # initialize ticks with infinity
    stepsize = 0   # initialize stepsize with 0
    while ticks > max_ticks:
        if stepsize < 100:
            stepsize += 25
        elif stepsize < 200:
            stepsize += 50
        elif stepsize < 500:
            stepsize += 100
        elif stepsize < 1000:
            stepsize += 200
        ticks = math.ceil(max_power / stepsize)

    # Set x and y axis and aspect ratio
    ax.set_xlim(0, stepsize * math.ceil(max_power/stepsize))
    ax.set_ylim(0, stepsize * math.ceil(max_power/stepsize))

    # Add ticks and tick labels to axes
    ax.set_xticks(np.arange(0, max(ax.get_xlim()) + 1, stepsize))
    ax.set_yticks(np.arange(0, max(ax.get_ylim()) + 1, stepsize))
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xticklabels([f'{x:.0f}kW' for x in ax.get_xticks()])
    ax.set_yticklabels([f'{y:.0f}kW' for y in ax.get_yticks()])
    ax.set(aspect="equal")

    # Tighten the layout
    plt.tight_layout()

    if save:
        # Save the plot
        plt.savefig(f'./figures/{topology}_power_{rate}_{comp}.png', dpi=300)

    # Show the plot
    # plt.show()

    # Close the plot again
    plt.close()

def create_power_plot(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        # Format the dataframe for plotting
        df_temp = format_df_for_plots(df, topology, weights)

        # Get max_power beforehand to ensure scaling is the same
        max_power = max(abs(df_temp["peak_power_without_lem"]).max(), abs(df_temp["peak_power_with_lem"]).max())

        # Plot all scenarios
        components = ['pv_penetration', 'hp_penetration', 'ev_penetration']
        rates = list(set(df_temp['pv_penetration']))
        for comp in components:
            for rate in rates:
                df_m = df_temp[df_temp[comp] == rate]

                # Create the scatter plot
                plot_power_plot(df_m, topology, rate, comp, save, max_power)

        # Plot one with all datapoints
        plot_power_plot(df_temp, topology, 'all', 'all', save, max_power)

def plot_indepth_plot(df_m: pd.DataFrame, topology: str, save: bool = False):

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
            ax.text(1, 0.9, f"y = {z[0]:.2f}x + {z[1]:.2f}", ha='right', va='top', fontsize=10)
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
            ax.text(1, 0.9, f"y = {z[0]:.2f}x + {z[1]:.2f}", ha='right', va='top', fontsize=10)
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
        # plt.show()

        plt.close()

def create_indepth_plot(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        # Format the dataframe for plotting
        df_m = format_df_for_plots(df, topology, weights)

        # Create the figure with subplots
        plot_indepth_plot(df_m, topology, save)


if __name__ == "__main__":
    # Folder that contains simulation results
    path = './all_lemlab_results'

    # Weighting of the seasons
    weights = {
        "winter": 1,
        "summer": 1,
        "transition": 2,
    }

    # Limits for the time period that is to be included in the calculations [x, y)
    # Limits are from Tue to Mon since Soner's input data was off by a day XD
    limits = {
        "winter": [1573513200, 1574118000],
        "summer": [1563832800, 1564437600],
        "transition": [1569276000, 1569880800],
    }


    # # Create the file containing all information
    # df = create_data(src=path, limits_dict=limits)
    #
    # # Save the dataframe to a csv file
    # df.to_csv('./validation_dataset.csv')
    # exit()

    # Load the dataframe from a csv file
    df = pd.read_csv('./validation_dataset.csv', index_col=0)

    # Currently .show commented out

    # Create plot comparing price and power with and without LEM
    # create_price_power_plot(df, show=False, save=True, weights=weights)

    # Create plot comparing price with and without LEM
    # create_price_plot(df, show=False, save=True, weights=weights)

    # Create plot comparing power with and without LEM
    create_power_plot(df, show=False, save=True, weights=weights)

    # Create in-depth plot for each device type
    # create_indepth_plot(df, show=False, save=True, weights=weights)