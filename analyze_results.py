import os
import time
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from scipy import stats
from pprint import pprint
from tqdm import tqdm
import logging
import math
import warnings
import itertools
import ast
import lemlab.db_connection.db_param as db_p
font = {'size': 16}

matplotlib.rc('font', **font)

# Turn off Pandas deprecation warnings
warnings.filterwarnings('ignore', category=FutureWarning)
warnings.filterwarnings('ignore', category=DeprecationWarning)

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

# Prices in €/kWh
LEVIES = 0.2297
MARKET_BUY = 0.0827
MARKET_SELL = 0.147


# Notes:
# # TODO: Load all results dfs and concatenate them to see which scenarios all hold true for (0,0) and (1,0)
# # TODO: Create a figure that shows all the scenarios that are in the same quadrant (or half)
# #  among all topologies (maybe with a bit of slack)

def calc_peakpower(info: dict, meter_file: str, readings_file: str, limits: list = None, max_loads: float = 0.1):
    # Get IDs of all main meters (1=utility with multiple submeters, 2=utility meter)
    df_meter_info = pd.read_csv(meter_file, index_col=0)
    list_main_meters = list(df_meter_info[df_meter_info['type_meter'].isin(
        ["grid meter", "virtual grid meter"])]['id_meter'].unique())

    # Get power flows of all meters in list_main_meters
    df_meter_readings_delta = pd.read_csv(readings_file, index_col=0)

    # Limit the readings to the specified limits if available
    if limits:
        df_meter_readings_delta = df_meter_readings_delta[(limits[0] <= df_meter_readings_delta['ts_delivery'])
                                                          & (df_meter_readings_delta['ts_delivery'] < limits[1])]

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
    # Ensure that the rows are in chronological order
    df_results = df_results.sort_index()

    # Calculate peak power and add to info dictionary
    max_magnitude_index = df_results['net_flow_kW'].abs().idxmax()
    info['peak_power'] = df_results.loc[max_magnitude_index, 'net_flow_kW']
    # info['peak_power'] = round(info['peak_power'], 2)  # Left out now to have more precise values

    # Save the maximum loads to the info dictionary (x % of the highest net loads)
    info["max_loads"] = df_results['net_flow_kW'].abs().sort_values(ascending=False).iloc[:math.ceil(len(df_results) * max_loads)].to_list()

    return info


def calc_energy(info: dict, meter_file: str, readings_file: str, limits: list = None):

    # Dict to be filled with the energy values
    energy = {
        'main': None,
        'hh': None,
        'pv': None,
        'bat': None,
        'hp': None,
        'ev': None,
    }

    # Load meter information
    df_meters = pd.read_csv(meter_file, index_col=0, dtype={"id_user": str})

    # Group by meter type
    meter_types = df_meters[db_p.INFO_ADDITIONAL].unique()
    df_meters = df_meters.groupby(db_p.INFO_ADDITIONAL)

    # Get meter readings
    df_meter_readings = pd.read_csv(readings_file, index_col=0)

    # Limit the meter readings to the specified limits if available
    if limits:
        df_meter_readings = df_meter_readings[(limits[0] <= df_meter_readings[db_p.TS_DELIVERY])
                                              & (df_meter_readings[db_p.TS_DELIVERY] < limits[1])]

    # Create new df for the results with the timestamp as index
    df_results = pd.DataFrame(index=df_meter_readings[db_p.TS_DELIVERY].unique())

    # Loop through the different meter types
    for meter_type, df_meter_type in df_meters:

        # Get all meter IDs of the current meter type
        list_meter_ids = set(df_meter_type[db_p.ID_METER])

        # Get the meter readings of the current meter type
        df_meter_readings_type = df_meter_readings[df_meter_readings[db_p.ID_METER].isin(list_meter_ids)]

        # Group by timestamp
        df_meter_readings_type = df_meter_readings_type.groupby(db_p.TS_DELIVERY).sum()

        # Calculate net energy flow
        df_meter_readings_type['energy'] = df_meter_readings_type['energy_out'] - df_meter_readings_type['energy_in']

        # Add the energy flows to the results df
        # df_results[f'{meter_type}_in'] = df_meter_readings_type['energy_in']
        # df_results[f'{meter_type}_out'] = df_meter_readings_type['energy_out']
        df_results[meter_type] = df_meter_readings_type['energy']

    # Convert all values from Wh per 15 min to kW
    df_results *= 1 / 250

    # Rename main meter to main
    df_results = df_results.rename(columns={'main meter': 'main'})

    # Separate into positive and negative values
    df_pos = df_results.clip(lower=0)
    df_neg = df_results.clip(upper=0)

    # Assign the results to the dict

    energy_info = {'energy_in': 0,
                   'energy_out':0,}
    for key in energy:
        if key in df_results.columns:
            val_in, val_out = df_pos[key].sum().round().astype(int), df_neg[key].sum().round().astype(int)
        else:
            val_in, val_out = [0, 0]
        # Add the key specific values
        energy[key] = [val_in, val_out]
        energy_info[f'{key}_in'] = val_in
        energy_info[f'{key}_out'] = val_out
        # Add the values to energy_in and energy_out
        if key != 'main':
            energy_info['energy_in'] += energy[key][0]
            energy_info['energy_out'] += energy[key][1]

    info['energy'] = energy
    info.update(energy_info)

    return info


def calc_avg_market_price_wlem(df_market):

    # Change ts_delivery to datetime
    df_market['ts_delivery'] = pd.to_datetime(df_market['ts_delivery'], unit='s')

    # Drop all rows where the retailer bought energy as it would distort the average energy price for the prosumers
    df_market = df_market[df_market['id_user'] != 'retailer01']

    # Drop all irrelevant columns
    cols_to_drop = ['delta_balance', 't_update_balance', 'share_quality_na', 'share_quality_local', 'share_quality_green_local']
    df_market = df_market.drop(columns=cols_to_drop)
    # Sort by timestamp and by user
    df_market = df_market.sort_values(by=['ts_delivery', 'id_user'])
    # Drop all rows where the sum of qty_energy that is in the rows where type_transaction is 'market' for each user and timestamp is greater than 0
    # Create a mask for rows where 'type_transaction' is 'market'
    mask = df_market['type_transaction'] == 'market'
    # Use the mask to filter the DataFrame and perform the groupby operation
    grouped = df_market[mask].groupby(['ts_delivery', 'id_user'])['qty_energy'].sum()
    # Create a DataFrame from the grouped Series
    df_grouped = grouped.reset_index()
    # Filter the original DataFrame using the grouped DataFrame
    df_market = df_market.merge(df_grouped[df_grouped['qty_energy'] < 0], on=['ts_delivery', 'id_user'], suffixes=(None, '_sum'))
    # Change the type transaction from balancing to market if the qty_energy is positive and change the sign of the energy price
    df_market.loc[(df_market['qty_energy'] > 0) & (df_market['type_transaction'] == 'balancing'), 'price_energy_market'] = -df_market['price_energy_market']
    df_market.loc[(df_market['qty_energy'] > 0) & (df_market['type_transaction'] == 'balancing'), 'type_transaction'] = 'market'
    # Recalculate the qty energy sum for each type of transaction
    df_market['qty_energy_sum'] = df_market.groupby(['ts_delivery', 'id_user', 'type_transaction'])['qty_energy'].transform('sum')
    # Calculate the sum of each market transaction for each user and timestamp
    df_market['qty_energy_sum'] = df_market.groupby(['ts_delivery', 'id_user', 'type_transaction'])['qty_energy'].transform('sum')
    # Calculate the weighted average energy price for each user and timestamp
    df_market['price_energy_market_weighted'] = df_market['price_energy_market'] * df_market['qty_energy'] / df_market['qty_energy_sum']

    # Drop all rows that contain levies as they are not part of the energy price
    df_market = df_market[~df_market['type_transaction'].str.contains('levies')]

    # Group entries by user, transaction type and timestamp
    df_market = df_market.groupby(['ts_delivery', 'id_user', 'type_transaction']).sum(numeric_only=True)
    # Expand the index to columns
    df_market = df_market.reset_index()
    # Drop the columns that are not needed
    df_market = df_market.drop(columns=['price_energy_market', 'qty_energy_sum'])
    # Drop all rows where market energy >= 0
    df_market = df_market[df_market['qty_energy'] < 0]
    # Drop all rows where the weighted average energy price negative
    df_market = df_market[df_market['price_energy_market_weighted'] >= 0]
    # Round the weighted average energy price
    df_market['price_energy_market_weighted'] = df_market['price_energy_market_weighted'].round().astype(int)

    # Convert sigma to €
    df_market['price_energy_market_weighted'] /= 1000000  # 1000000 to convert from sigma to €

    # Convert Wh to kWh
    df_market['qty_energy'] /= 1000  # 1000 to convert from Wh to kWh

    # Add levies directly to the energy price as they are constant (in €/kWh)
    df_market['price_pu'] = (df_market['price_energy_market_weighted'] + LEVIES)

    # Add the energy price to the dataframe
    df_market['price_total'] = df_market['price_pu'] * df_market['qty_energy']

    return df_market


def calc_avg_market_price_wolem(df_market):

    # Change ts_delivery to datetime
    df_market['ts_delivery'] = pd.to_datetime(df_market['ts_delivery'], unit='s')

    # Drop all rows where the retailer bought energy as it would distort the average energy price for the prosumers
    df_market = df_market[df_market['id_user'] != 'retailer01']

    # Drop all irrelevant columns
    cols_to_drop = ['delta_balance', 't_update_balance', 'share_quality_na', 'share_quality_local', 'share_quality_green_local']
    df_market = df_market.drop(columns=cols_to_drop)
    # Sort by timestamp and by user
    df_market = df_market.sort_values(by=['ts_delivery', 'id_user'])
    # Drop all rows where the energy quantity is greater than 0 (only keep the energy that is bought)
    df_market = df_market[df_market['qty_energy'] < 0]
    # Remove the levies as they are added later
    df_market = df_market[~df_market['type_transaction'].str.contains('levies')]

    # Convert sigma to €
    df_market['price_energy_market'] /= 1000000  # 1000000 to convert from sigma to €

    # Convert Wh to kWh
    df_market['qty_energy'] /= 1000  # 1000 to convert from Wh to kWh

    # Add levies directly to the energy price as they are constant (in €/kWh)
    df_market['price_pu'] = (df_market['price_energy_market'] + LEVIES)

    # Add the energy price to the dataframe
    df_market['price_total'] = df_market['price_pu'] * df_market['qty_energy']

    return df_market


def add_selfused_energy(df_market, df_meter, df_readings):

    # Create dataframe to store the self-used energy information
    df_selfused = pd.DataFrame(columns=df_market.columns)
    df_selfused['id_user'] = df_meter['id_user'].unique()
    df_selfused['price_energy_market'] = MARKET_BUY
    df_selfused['price_pu'] = MARKET_BUY
    df_selfused['type_transaction'] = 'self-used'

    # Filter for generation and consumption (everything that is not main meter or pv)
    df_gen = df_meter[df_meter['info_additional'] == 'pv']
    df_con = df_meter[(df_meter['info_additional'] != 'pv') & (df_meter['info_additional'] != 'main meter')]

    # Merge df_readings with df_gen and df_con to get generation and consumption readings
    df_gen_readings = pd.merge(df_readings, df_gen, on='id_meter')
    df_con_readings = pd.merge(df_readings, df_con, on='id_meter')

    # Group by 'ts_delivery' and sum 'energy_out' and 'energy_in' to get total generation and consumption for each timestamp
    df_gen_grouped = df_gen_readings.groupby(['ts_delivery', 'id_user'])['energy_out'].sum()
    df_con_grouped = df_con_readings.groupby(['ts_delivery', 'id_user'])['energy_in'].sum()

    # Calculate 'self_used' energy by taking the minimum of generation and consumption for each timestamp
    df_selfused_energy = pd.DataFrame({'self_used': np.minimum(df_gen_grouped, df_con_grouped)})

    # Group by 'id_user' and sum 'self_used' energy to get total self-used energy for each user
    df_selfused_energy_grouped = df_selfused_energy.groupby('id_user')['self_used'].sum()

    # Assign 'qty_energy' and calculate 'price_total' for each user
    df_selfused.set_index('id_user', inplace=True)
    df_selfused.loc[df_selfused_energy_grouped.index, 'qty_energy'] = -df_selfused_energy_grouped / 1e3
    df_selfused['price_total'] = df_selfused['price_pu'] * df_selfused['qty_energy']
    # Reset the index
    df_selfused.reset_index(inplace=True)

    # Append the self-owned energy information to the market dataframe
    df_market = pd.concat([df_market, df_selfused], ignore_index=True)

    # Sort by timestamp and by user
    df_market = df_market.sort_values(by=['ts_delivery', 'id_user'])

    return df_market


def calc_avgenergyprice(info: dict, market_file: str, meter_file: str, readings_file: str, limits: list = None):
    # Get the market data
    df_market = pd.read_csv(market_file, index_col=0)

    # Get the meter data
    df_meter = pd.read_csv(meter_file, index_col=0)

    # Get the readings data
    df_readings = pd.read_csv(readings_file, index_col=0)

    # Limit the market data to the specified limits if available
    if limits:
        df_market = df_market[(limits[0] <= df_market['ts_delivery']) & (df_market['ts_delivery'] < limits[1])]
        df_readings = df_readings[(limits[0] <= df_readings['ts_delivery']) & (df_readings['ts_delivery'] < limits[1])]

    # Clean market data to obtain the average energy price
    if 'wLEM' in market_file:
        # Calculation is more complex as agents can buy and sell energy at all times and sometimes have to correct
        # their balance by selling previously bought energy. This has to be taken into account when calculating the
        # average energy price as otherwise the average energy price would be distorted upwards.
        df_market = calc_avg_market_price_wlem(df_market)
    elif 'woLEM' in market_file:
        df_market = calc_avg_market_price_wolem(df_market)
    else:
        raise ValueError("The market file must contain either 'wLEM' or 'woLEM'.")

    # Add rows with information about the self-used energy to the market data
    df_market = add_selfused_energy(df_market, df_meter, df_readings)

    # Calculate the average energy price, convert it from sigma to € and add it to the info dictionary
    info['average_energy_price'] = df_market['price_total'].sum() / df_market['qty_energy'].sum()

    return info


def calc_market_balancing_energy(info: dict, market_file: str, limits: list = None):
    # Get the market data
    df_market = pd.read_csv(market_file, index_col=0)

    # Limit the market data to the specified limits if available
    if limits:
        df_market = df_market[(limits[0] <= df_market['ts_delivery']) & (df_market['ts_delivery'] < limits[1])]

    # Exclude the retailer from the market data
    df_market = df_market[df_market['id_user'] != 'retailer01']

    for key in ['in', 'out']:

        # Get all market trades with negative energy amount (energy purchases)
        if key == 'in':
            df_m = df_market[df_market['qty_energy'] > 0]
        elif key == 'out':
            df_m = df_market[df_market['qty_energy'] < 0]
        else:
            raise ValueError(f"Key must be either 'in' or 'out', but is {key}.")

        # Group by transaction type
        df_m = df_m.groupby(by='type_transaction').sum()

        if len(df_m) == 0:
            # If there are no market trades of the specified type, the energy is 0
            info[f"energy_market_{key}"] = 0
            info[f"energy_balancing_{key}"] = 0

        # Add info to dict
        try:
            info[f"energy_market_{key}"] = int(abs(df_m.loc['market', 'qty_energy'] / 1000))
            info[f"energy_balancing_{key}"] = int(abs(df_m.loc['balancing', 'qty_energy'] / 1000))
        except KeyError:
            try:
                # If there is no market, the balancing energy is the market energy
                info[f"energy_market_{key}"] = int(abs(df_m.loc['balancing', 'qty_energy'] / 1000))
                info[f"energy_balancing_{key}"] = 0
            except KeyError:
                info[f"energy_market_{key}"] = 0
                info[f"energy_balancing_{key}"] = 0

        # info["costs_market"] = int(abs(df_market.loc['market', 'delta_balance'] / 1000000 / 1000))
        # info["costs_balancing"] = int(abs(df_market.loc['balancing', 'delta_balance'] / 1000000 / 1000))

    info["energy_market"] = abs(info["energy_market_in"]) + abs(info["energy_market_out"])
    info["energy_balancing"] = abs(info["energy_balancing_in"]) + abs(info["energy_balancing_out"])

    return info


def get_economic_data(info: dict, market_file: str, meter_file: str, readings_file: str, limits: list = None):

    # Get average energy price
    info = calc_avgenergyprice(info=info, market_file=market_file, meter_file=meter_file,
                               readings_file=readings_file, limits=limits)

    # Get amount of market and balancing energy
    info = calc_market_balancing_energy(info=info, market_file=market_file, limits=limits)

    return info


def create_data(src: str, limits_dict: dict = None, max_loads: float = 0.1):
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
                    "controller": split_values[-2],
                    "fcast": split_values[-1],
                    "average_energy_price": None,
                    "peak_power": None,
                    "max_loads": None,
                    "energy": None,
                    "energy_in": None,
                    "energy_out": None,
                    "energy_market": None,
                    "energy_balancing": None,
                    "costs_market": None,
                    "costs_balancing": None,
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
                    info = get_economic_data(info=info, market_file=market_file, meter_file=meter_file,
                                               readings_file=readings_file, limits=limits)
                    # TODO: info = get_technical_data()
                    info = calc_peakpower(info, meter_file, readings_file, limits, max_loads)
                    info = calc_energy(info, meter_file, readings_file, limits)
                except Exception as e:
                    # Log error
                    logging.error(f"Error in {scenario_folder}: {e}")

                    raise e

                # Add info to the list
                scenario_info_list.append(info)

    # Create a DataFrame from the list
    df = pd.DataFrame(scenario_info_list)

    return df


# Create a function to draw pie charts at specified coordinates
def create_donut_chart(ax, x: float, y: float, val: float, color: str, radius: float = 0.1, plot_inner_circle: bool = False):

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


def create_pie_chart(ax, x: float, y: float, vals: list, colors: list, radius: float = 1, stepsize: float = 0.25, plot_inner_circle: bool = False):
    """
    Create a pie chart with each segment divided into rings that represent different penetration levels.

    Parameters:
    ax (matplotlib.axes._subplots.AxesSubplot): The subplot to draw on.
    x (float): The x-coordinate of the center of the pie chart.
    y (float): The y-coordinate of the center of the pie chart.
    vals (list): The values for each segment in the pie chart.
    colors (list): The colors for each segment.
    radius (float): The overall radius of the pie chart.
    stepsize (float): The step size to increase the radius of each ring.
    plot_inner_circle (bool): If True, plot an inner circle to create a doughnut shape.
    """

    # Number of segments and the corresponding values
    n = 1 / len(vals)
    list_n = [n * i for i in range(len(vals) + 1)]

    # Alpha value for the pie chart
    alpha = 0.7

    # Plot segments
    for idx, val in enumerate(vals):
        # Plot each ring
        for i in range(4, 0 , -1):
            # Calculate the radius of the ring
            ring_radius = radius * stepsize * i
            # Create the marker for the pie chart
            mx = [0] + np.cos(np.linspace((2 * np.pi * list_n[idx] + np.pi/2) ,
                                          (2 * np.pi * list_n[idx + 1] + np.pi/2), 100)).tolist()
            my = [0] + np.sin(np.linspace((2 * np.pi * list_n[idx] + np.pi/2),
                                          (2 * np.pi * list_n[idx + 1] + np.pi/2), 100)).tolist()
            xy = np.column_stack([mx, my])

            # Plot share of the pie chart
            if val >= stepsize * i:
                color = colors[idx]
            else:
                color = 'white'
            ax.scatter(x, y, marker=xy, s=1000 * (ring_radius ** 2),
                       c=color, edgecolors='black', linewidths=0.3, alpha=alpha)


def format_df_for_plots(df: pd.DataFrame, topology, weights: dict = None, overwrite: bool = False,
                        wlem: list = ('mpc', 'naive'), wolem: list = ('rtc', 'naive'), energy_key: 'in, out' = None):

    # Check if the file already exists, and it should not be overwritten and return it right away
    if os.path.isfile(f'./results/{topology}_price_power.csv') and not overwrite:
        return pd.read_csv(f'./results/{topology}_price_power.csv')

    # Reduce the dataframe to the specified topology
    df_top = df[df["grid_topology"] == topology]

    # Reduce the dataframe by the type of controller and forecast
    df_top = df_top[(df_top["lem"] == True) & (df_top["controller"] == wlem[0]) & (df_top["fcast"] == wlem[1])
                    | (df_top["lem"] == False) & (df_top["controller"] == wolem[0]) & (df_top["fcast"] == wolem[1])]

    # Create a dataframe that contains all scenarios with LEM and without LEM and all seasons (thus 6 rows per group)
    df = (df_top.groupby(['grid_topology', 'pv_penetration', 'hp_penetration', 'ev_penetration', 'lem']).
          filter(lambda group: set(group['season']) == {'summer', 'transition', 'winter'}).
          reset_index(drop=True))
    df = (df.groupby(['grid_topology', 'pv_penetration', 'hp_penetration', 'ev_penetration']).
          filter(lambda group: set(group['lem']) == {True, False}).
          reset_index(drop=True))
    df['peak_power'].fillna(0, inplace=True)

    # Turn the max loads and energy column into lists as it is loaded as a string
    try:
        df['max_loads'] = df['max_loads'].apply(lambda x: ast.literal_eval(x))
        df['energy'] = df['energy'].apply(lambda x: ast.literal_eval(x))
    except ValueError:
        raise ValueError("The max_loads and energy columns could not be converted to lists. "
                         "There seems to be a nan value in the column.")

    # Expand the dict in the energy column
    # Function to expand the 'energy' column into separate columns
    def expand_energy(energy_dict, key):
        return pd.Series({
            f'energy_{key}_pos': energy_dict[key][0],
            f'energy_{key}_neg': energy_dict[key][1],
            f'energy_{key}_net': energy_dict[key][0] + energy_dict[key][1],
        })

    # Apply the function to each row in the DataFrame for all keys in the 'energy' dictionary
    for key in df['energy'][0].keys():
        expanded_energy = df['energy'].apply(lambda x: expand_energy(x, key))
        # Concatenate the expanded_energy DataFrame with the original DataFrame
        df = pd.concat([df, expanded_energy], axis=1)

    # Add new columns to the DataFrame
    # Share of balancing energy of total energy
    if energy_key:
        df['share_balancing_energy'] = df[f'energy_balancing_{energy_key}'] / (df[f'energy_market_{energy_key}'] + df[f'energy_balancing_{energy_key}'])
    else:
        df['share_balancing_energy'] = df[f'energy_balancing'] / (df[f'energy_market'] + df[f'energy_balancing'])
    df['share_balancing_energy'] = df['share_balancing_energy'].round(5)
    # Set the share of balancing energy to 0 if there is no LEM
    df.loc[df['lem'] == False, 'share_balancing_energy'] = 0
    # Add the balancing energy to the market energy if there is no LEM
    if energy_key:
        df.loc[df['lem'] == False, f'energy_market_{energy_key}'] += df.loc[df['lem'] == False, f'energy_balancing_{energy_key}']
    else:
        df.loc[df['lem'] == False, f'energy_market'] += df.loc[df['lem'] == False, f'energy_balancing']
    # Set the balancing energy to 0 if there is no LEM
    if energy_key:
        df.loc[df['lem'] == False, f'energy_balancing_{energy_key}'] = 0
    else:
        df.loc[df['lem'] == False, f'energy_balancing'] = 0
    # Normalized market energy
    df['normalized_market_energy'] = 0
    for lem_status in [True, False]:
        if energy_key:
            df.loc[df['lem'] == lem_status, 'normalized_market_energy'] = df.loc[df['lem'] == lem_status, f'energy_market_{energy_key}'] / \
                                                                      df.loc[df['lem'] == lem_status, f'energy_market_{energy_key}'].max()
        else:
            df.loc[df['lem'] == lem_status, 'normalized_market_energy'] = df.loc[df['lem'] == lem_status, f'energy_market'] / \
                                                                      df.loc[df['lem'] == lem_status, f'energy_market'].max()
        df['normalized_market_energy'] = df['normalized_market_energy'].round(5)
    # Ratio of generation and demand
    df['ratio_gen_dem'] = abs(df['energy_in'] / df['energy_out'])
    # Ratio of hp and hh
    df['ratio_hp_hh'] = abs(df['energy_hp_net'] / df['energy_hh_net'])
    # Ratio of pv and hp
    df['ratio_pv_hp'] = abs(df['energy_pv_net'] / df['energy_hp_net'])

    # Drop the original 'energy' column
    df = df.drop('energy', axis=1)

    # Save the raw results to csv
    df.to_csv(f'./results/{topology}_raw.csv')

    # Weight the average energy price, the balancing ratio and normalized market energy and the energy columns
    # according to the weights
    df.reset_index(inplace=True, drop=True)
    sum_weights = 0
    if weights:
        for season, weight in weights.items():
            df.loc[df['season'] == season, 'average_energy_price'] *= weight
            df.loc[df['season'] == season, 'share_balancing_energy'] *= weight
            df.loc[df['season'] == season, 'normalized_market_energy'] *= weight
            df.loc[df['season'] == season, 'ratio_gen_dem'] *= weight
            df.loc[df['season'] == season, 'ratio_hp_hh'] *= weight
            df.loc[df['season'] == season, 'ratio_pv_hp'] *= weight
            sum_weights += weight

    # Extend the lists of max loads according to the weights
    if weights:
        for season, weight in weights.items():
            df.loc[df['season'] == season, f'max_loads'] = df[f'max_loads'] * weight

    # Grouping by the relevant columns
    grouped = df.groupby(['grid_topology', 'pv_penetration', 'hp_penetration', 'ev_penetration', 'lem'])

    # Calculate sum of average_energy_price
    sum_energy_price = grouped['average_energy_price'].sum() / sum_weights

    # Calculate the average share of balancing energy
    avg_share_balancing_energy = grouped['share_balancing_energy'].sum() / sum_weights

    # Calculate the average normalized market energy
    avg_normalized_market_energy = grouped['normalized_market_energy'].sum() / sum_weights

    # Calculate the weighted columns
    result_df = grouped.agg(
        average_energy_price=('average_energy_price', lambda x: round(x.sum() / sum_weights, 4)),  # average_energy_price
        avg_share_balancing_energy=('share_balancing_energy', lambda x: round(x.sum() / sum_weights, 4)),  # average share of balancing energy
        avg_normalized_market_energy=('normalized_market_energy', lambda x: round(x.sum() / sum_weights, 4)),  # average normalized market energy
        avg_ratio_gen_dem=('ratio_gen_dem', lambda x: round(x.sum() / sum_weights, 4)),  # average ratio of generation and demand
        avg_ratio_hp_hh=('ratio_hp_hh', lambda x: round(x.sum() / sum_weights, 4)),  # average ratio of hp and hh
        avg_ratio_pv_hp=('ratio_pv_hp', lambda x: round(x.sum() / sum_weights, 4)),  # average ratio of pv and hp
    )

    # Find the row with max absolute peak_power
    idx = grouped['peak_power'].apply(lambda x: x.abs().idxmax())
    max_peak_power = df.loc[idx, 'peak_power']
    max_peak_season = df.loc[idx, 'season']

    # Combine the max loads to one list per scenario
    def combine_max_loads(series):
        return [item for sublist in series for item in sublist]
    df_loads = grouped['max_loads'].agg(combine_max_loads).reset_index()
    # Sort the list in descending order and take the 1/x of the sum of the weights
    df_loads['max_loads'] = df_loads['max_loads'].apply(lambda x: sorted(x, reverse=True)[:math.ceil(len(x) / sum_weights)])
    # Compute the mean of the max loads
    df_loads['max_loads'] = df_loads['max_loads'].apply(lambda x: round(np.mean(x), 2))

    # Construct the result dataframe
    result_df = result_df.reset_index()
    result_df['peak_power'] = max_peak_power.values
    result_df['season'] = max_peak_season.values
    # Match max loads on pv, hp, and ev and if lem is True or False
    result_df = pd.merge(result_df, df_loads, on=['grid_topology', 'pv_penetration', 'hp_penetration', 'ev_penetration', 'lem'])

    # Create a dataframe that contains all rows with LEM and without LEM
    df_with_lem = result_df[result_df["lem"] == True]
    df_wo_lem = result_df[result_df["lem"] == False]

    # Merge the DataFrames based on the columns representing PV, HP, and EV penetration
    df_m = pd.merge(df_with_lem, df_wo_lem, on=["pv_penetration", "hp_penetration", "ev_penetration"],
                    suffixes=('_with_lem', '_without_lem'))

    # Compute the ratios for average_energy_price, peak_power and max_loads
    df_m["average_energy_price_ratio"] = df_m["average_energy_price_with_lem"] / df_m[
        "average_energy_price_without_lem"]
    df_m["peak_power_ratio"] = abs(df_m["peak_power_with_lem"] / df_m["peak_power_without_lem"])
    df_m["max_loads_ratio"] = df_m["max_loads_with_lem"] / df_m["max_loads_without_lem"]

    # # Drop the unnecessary columns and keep the relevant columns
    # df_m = df_m[["grid_topology_with_lem", "pv_penetration", "hp_penetration", "ev_penetration",
    #          "average_energy_price_ratio", "peak_power_ratio"]]

    # Rename the 'grid_topology_with_lem' column to 'grid_topology'
    df_m.rename(columns={"grid_topology_with_lem": "grid_topology"}, inplace=True)

    # Save results to csv
    df_m.to_csv(f'./results/{topology}_price_power.csv')

    return df_m


def plot_price_power(df_m: pd.DataFrame, topology, rate, comp, show: bool = False, save: bool = False, power: str = 'max_loads'):

        # Use these lines in conjunction with make_all=False if you want to limit the circles to some certain shares
        # Reduce the dataframe to the specified penetration rates
        # rates = [0.0, 0.5, 1.0]
        # df_m = df_m[df_m["pv_penetration"].isin(rates)
        #             & df_m["hp_penetration"].isin(rates)
        #             & df_m["ev_penetration"].isin(rates)]

        # Use these lines in conjunction with make_all=True if you want to limit the circles to some certain shares
        # If you want another share per type, then just add it as element to the list
        # params = {
        #     "pv": [0.25 * 1],
        #     "ev": [0.25 * 4],
        #     "hp": [0.25 * 1],
        # }
        # df_m = df_m[(df_m["pv_penetration"].isin(params['pv']))
        #             & (df_m["hp_penetration"].isin(params['hp']))
        #             & (df_m["ev_penetration"].isin(params['ev']))]

        # Use this if you want to reduce the dataframe to a certain penetration rate
        # Note: In the analysis PV needs to be greater than 0
        df_m = df_m[(df_m["pv_penetration"] > 0)]
        # df_m = df_m[(df_m["average_energy_price_ratio"] < 0.8)]
        df_m = df_m[(df_m[f"{power}_ratio"] < .9)]
        if len(df_m) == 0:
            print('0')
            return
        print(len(df_m))

        # df_m = df_m[(df_m["pv_penetration"] > 0) & (df_m["pv_penetration"] < 1)
        #             & (df_m["hp_penetration"] > 0) & (df_m["hp_penetration"] < 1)
        #             & (df_m["ev_penetration"] > 0) & (df_m["ev_penetration"] < 1)]

        # Count the number of scenarios per category
        cat = {
            'lc_lp': len(df_m[(df_m['average_energy_price_ratio'] < 1) & (df_m[f'{power}_ratio'] < 1)]),
            'hc_lp': len(df_m[(df_m['average_energy_price_ratio'] > 1) & (df_m[f'{power}_ratio'] < 1)]),
            'lc_hp': len(df_m[(df_m['average_energy_price_ratio'] < 1) & (df_m[f'{power}_ratio'] > 1)]),
            'hc_hp': len(df_m[(df_m['average_energy_price_ratio'] > 1) & (df_m[f'{power}_ratio'] > 1)]),
            'total': len(df_m),
        }

        # Create the scatter plot with customized markers
        plt.figure(figsize=(10, 10))
        ax = plt.gca()

        # Plot the pie charts for each scenario
        plot_type = 'pie'
        for i, row in df_m.iterrows():

            x_val = row["average_energy_price_ratio"]
            y_val = row[f"{power}_ratio"]

            if plot_type == 'donut':

                # Create the donut charts for each device
                create_donut_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5 * 2)
                create_donut_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25 * 2)
                create_donut_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1 * 2,
                                   plot_inner_circle=True)
            elif plot_type == 'pie':
                # Values and colors for the pie chart
                vals = [row["hp_penetration"], row["ev_penetration"], row["pv_penetration"]]  # HP, EV, PV order
                colors = [COLORS['hp'], COLORS['ev'], COLORS['pv']]

                # Create the donut charts
                create_pie_chart(ax, x_val, y_val, vals, colors)

        # Add matrix lines and text
        ax.plot(np.linspace(0, 2, 10), np.linspace(1, 1, 10), linestyle='--', color='black', linewidth=1)
        ax.plot(np.linspace(1, 1, 10), np.linspace(0, 2, 10), linestyle='--', color='black', linewidth=1)
        ax.text(0.775, 0.7, f"lower costs \nlower power\nshare: {int(round(cat['lc_lp'] / cat['total'] * 100))}%", ha='left', va='bottom', fontsize=12)
        ax.text(0.775, 1.15, f"lower costs \nhigher power\nshare: {int(round(cat['lc_hp'] / cat['total'] * 100))}%", ha='left', va='top', fontsize=12)
        ax.text(1.075, 0.7, f"higher costs \nlower power\nshare: {int(round(cat['hc_lp'] / cat['total'] * 100))}%", ha='right', va='bottom', fontsize=12)
        ax.text(1.075, 1.15, f"higher costs \nhigher power\nshare: {int(round(cat['hc_hp'] / cat['total'] * 100))}%", ha='right', va='top', fontsize=12)


        # Set axis labels and title
        plt.xlabel("Average Energy Price - with LEM / without LEM")
        plt.ylabel(f"Peak Power - with LEM / without LEM")

        # Create legend handles and labels for the customized markers
        legend_elements = [
            plt.Line2D([0], [0], marker='o', color=COLORS['pv'], label='PV', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color=COLORS['ev'], label='EV', markersize=10, linestyle='None'),
            plt.Line2D([0], [0], marker='o', color=COLORS['hp'], label='HP', markersize=10, linestyle='None'),
        ]

        # Show the legend with customized markers
        ax.legend(handles=legend_elements, loc='right')

        # Set x and y axis and aspect ratio
        ax.set_xlim(0.75, 1.1)  # max(df_m["average_energy_price_ratio"].max(), df_m["peak_power_ratio"].max()) + 0.1)
        ax.set_ylim(0.65, 1.2)  #max(df_m["average_energy_price_ratio"].max(), df_m["peak_power_ratio"].max()) + 0.1)
        # ax.set(aspect="equal")

        # Title
        plt.title(f"{topology.capitalize()}")

        # Tighten the layout
        # plt.tight_layout()

        if save:
            # Save the plot
            plt.savefig(f'./figures/{topology}_price_power_{rate}_{comp}.png', dpi=300)

        # Show the plot
        if show:
            plt.show()

        plt.close()


def create_price_power(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None, power: str = 'max_loads', make_all: bool = False):
    # df_all = pd.DataFrame(columns=df.columns)
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        # Format the dataframe for plotting
        df_temp = format_df_for_plots(df, topology, weights)

        # Append the formatted dataframe to the df_all dataframe
        # df_all = pd.concat([df_all, df_temp], ignore_index=True)

        # Plot all scenarios
        components = ['pv_penetration', 'hp_penetration', 'ev_penetration']
        rates = list(set(df_temp['pv_penetration']))
        if make_all:
            for comp in components:
                for rate in rates:
                    df_m = df_temp[df_temp[comp] == rate]

                    # Create the scatter plot
                    plot_price_power(df_m, topology, rate, comp, show, save, power)

        # Plot one with all datapoints
        plot_price_power(df_temp, topology, 'all', 'all', show, save, power)

    # TODO: Use df_all to create a plot that shows all scenarios that are in the same quadrant (or half) among all topologies (maybe with a bit of slack)
    # Put this into new function instead of here.


def plot_price(df_m: pd.DataFrame, topology, rate, comp, show: bool = False, save: bool = False, limits: list = None):
    # Use this if you want to reduce the dataframe to a certain penetration rate
    # df_m = df_m[(df_m["pv_penetration"] > 0) & (df_m["pv_penetration"] < 1)
    #                 & (df_m["hp_penetration"] > 0) & (df_m["hp_penetration"] < 1)
    #                 & (df_m["ev_penetration"] > 0) & (df_m["ev_penetration"] < 1)]
    # Note: In the analysis PV needs to be greater than 0
    df_m = df_m[(df_m["pv_penetration"] > 0)]

    # Create the scatter plot with customized markers
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Plot the pie charts for each scenario
    plot_type = 'pie'
    for i, row in df_m.iterrows():
        x_val = row["average_energy_price_without_lem"]
        y_val = row["average_energy_price_with_lem"]

        # # Create the pie charts for each device
        # create_donut_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5 * 2)
        # create_donut_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25 * 2)
        # create_donut_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1 * 2,
        #                    plot_inner_circle=True)

        if plot_type == 'donut':

            # Create the donut charts for each device
            create_donut_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5 * 2)
            create_donut_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25 * 2)
            create_donut_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1 * 2,
                               plot_inner_circle=True)
        elif plot_type == 'pie':
            # Values and colors for the pie chart
            vals = [row["hp_penetration"], row["ev_penetration"], row["pv_penetration"]]  # HP, EV, PV order
            colors = [COLORS['hp'], COLORS['ev'], COLORS['pv']]

            # Create the donut charts
            create_pie_chart(ax, x_val, y_val, vals, colors)

    # Add the dashed lines and the text
    if not limits:
        max_price = max(abs(df_m["average_energy_price_without_lem"]).max(),
                        abs(df_m["average_energy_price_with_lem"]).max())
    else:
        max_price = limits[1]
    # LEM/noLEM = 1.5
    ax.plot(np.linspace(0, 2 * max_price, 10), np.linspace(0, 2 * max_price * 1.2, 10),
            linestyle='--', color=(0.1, 0.1, 0.1), linewidth=2)
    # ax.text(max_power, max_power, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1.5$",
    #         ha='left', va='center', fontsize=10)
    # LEM/noLEM = 1.25
    ax.plot(np.linspace(0, 2 * max_price, 10), np.linspace(0, 2 * max_price * 1.1, 10),
            linestyle='--', color=(0.25, 0.25, 0.25), linewidth=2)
    # ax.text(max_power, max_power, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1.25$",
    #         ha='left', va='center', fontsize=10)
    # LEM/noLEM = 1
    ax.plot(np.linspace(0, 2 * max_price, 10), np.linspace(0, 2 * max_price, 10),
            linestyle='--', color=(0.4, 0.4, 0.4), linewidth=2)
    # ax.text(max_power, max_power, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
    #         ha='left', va='center', fontsize=10)
    # LEM/noLEM = 0.75
    ax.plot(np.linspace(0, 2 * max_price, 10), np.linspace(0, 2 * max_price * 0.9, 10),
            linestyle='--', color=(0.4, 0.5, 0.6), linewidth=2)
    # ax.text(max_power, max_power * 0.75, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.75$",
    #         ha='left', va='center',                fontsize=10)
    # LEM/noLEM = 0.5
    ax.plot(np.linspace(0, 2 * max_price, 10), np.linspace(0, 2 * max_price * 0.8, 10),
            linestyle='--', color=(0.5, 0.6, 0.7), linewidth=2)
    # ax.text(max_power, max_power * 0.5, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.5$",
    #         ha='left', va='center',                fontsize=10)

    # Set axis labels and title
    plt.xlabel("Average Energy Price - without LEM")
    plt.ylabel("Average Energy Price - with LEM")
    # plt.title("Comparison of Parameters with and without LEM")

    # Create legend handles and labels for the customized markers
    legend_elements = [
        plt.Line2D([0], [0], marker='o', color=COLORS['pv'], label='PV', markersize=10, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color=COLORS['ev'], label='EV', markersize=10, linestyle='None'),
        plt.Line2D([0], [0], marker='o', color=COLORS['hp'], label='HP', markersize=10, linestyle='None'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1.2$",
                   color=(0.1, 0.1, 0.1), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1.1$",
                   color=(0.25, 0.25, 0.25), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
                   color=(0.4, 0.4, 0.4), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.9$",
                   color=(0.4, 0.5, 0.6), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.8$",
                   color=(0.5, 0.6, 0.7), linestyle='--'),
    ]

    # # Add a vertical line that represents the average energy price without LEM
    # ax.axvline(df_m["average_energy_price_without_lem"].mean(), color='black', linestyle='--', linewidth=1)
    # # Add text next to it to specify what the line stands for
    # ax.text(df_m["average_energy_price_without_lem"].mean() + 0.01, 0.1, "Average Energy Price without LEM",
    #         ha='left', va='bottom', fontsize=12)

    # Show the legend with customized markers
    ax.legend(handles=legend_elements, loc='upper left')

    # Set x and y axis and aspect ratio
    bottom, top = 0.05 * math.floor(limits[0] / 0.05), 0.05 * math.ceil(limits[1] / 0.05)
    ax.set_xlim(bottom, top)
    ax.set_ylim(bottom, top)

    # Add ticks and tick labels to axes
    ax.set_xticks(np.arange(min(ax.get_xticks()), max(ax.get_xticks()), 0.05))
    ax.set_yticks(np.arange(min(ax.get_yticks()), max(ax.get_yticks()), 0.05))
    ax.set_xticklabels([f'{x:.2f} €/kWh' for x in ax.get_xticks()])
    ax.set_yticklabels([f'{y:.2f} €/kWh' for y in ax.get_yticks()])
    ax.set(aspect="equal")

    # Title
    plt.title(f"{topology.capitalize()}")

    # Tighten the layout
    plt.tight_layout()
    plt.grid()

    if save:
        # Save the plot
        plt.savefig(f'./figures/{topology}_price_{rate}_{comp}.png', dpi=300)

    # Show the plot
    if show:
        plt.show()

    # Close the plot again
    plt.close()


def create_price(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None, make_all: bool = False):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        # Format the dataframe for plotting
        df_temp = format_df_for_plots(df, topology, weights)

        # Get max_power beforehand to ensure scaling is the same
        limits = [min(abs(df_temp[f"average_energy_price_without_lem"]).min(),
                      abs(df_temp[f"average_energy_price_with_lem"]).min()),
                  max(abs(df_temp[f"average_energy_price_without_lem"]).max(),
                      abs(df_temp[f"average_energy_price_with_lem"]).max())
                 ]

        # Plot all scenarios
        components = ['pv_penetration', 'hp_penetration', 'ev_penetration']
        rates = list(set(df_temp['pv_penetration']))
        if make_all:
            for comp in components:
                for rate in rates:
                    df_m = df_temp[df_temp[comp] == rate]

                    # Create the scatter plot
                    plot_price(df_m, topology, rate, comp, show, save, limits)

        # Plot one with all datapoints
        plot_price(df_temp, topology, 'all', 'all', show, save, limits)


def create_price_plot_fixed(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        if topology != 'rural':
            continue

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
            create_donut_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5 * 2)
            create_donut_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25 * 2)
            create_donut_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1 * 2,
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

        # Title
        plt.title(f"{topology.capitalize()}")

        # Tighten the layout
        plt.tight_layout()

        if save:
            # Save the plot
            plt.savefig(f'./figures/{topology}_price.png', dpi=300)

        # Show the plot
        if show:
            plt.show()


def plot_power(df_m: pd.DataFrame, topology, rate, comp, show: bool = False, save: bool = False, limits: list = None, power: str = 'max_loads'):
    # Use this if you want to reduce the dataframe to a certain penetration rate
    #df_m = df_m[(df_m["season_with_lem"]) == 'transition']
    # df_m = df_m[(df_m["pv_penetration"] > 0) & (df_m["pv_penetration"] < 1)
    #             & (df_m["hp_penetration"] > 0) & (df_m["hp_penetration"] < 1)
    #             & (df_m["ev_penetration"] > 0) & (df_m["ev_penetration"] < 1)]
    # Note: In the analysis PV needs to be greater than 0
    df_m = df_m[(df_m["pv_penetration"] > 0)]

    # I used this part to print parts of the plot separately. Put it back in and move the rest, if you want to use it.
    # print(len(df_m))
    # df_temp = df_m.copy()
    # limlim = [(0, 0), (70, 70), (100, 110), (180, 180), (250, 260), (300, 300), (400, 400)]
    # limits = None
    # limlim = [(0, 0), (200, 200), (300, 300), (400, 450), (500, 650), (720, 900)]
    # for idx, lim in enumerate(limlim[1:]):
    #     lim_prev = limlim[idx]
    #     df_m = df_temp[
    #         (df_temp["max_loads_with_lem"] > lim_prev[0])
    #         &           (df_temp["max_loads_with_lem"] < lim[0])
    #                    & (df_temp["max_loads_without_lem"] > lim_prev[1])
    #                    & (df_temp["max_loads_without_lem"] < lim[1])]
    #
    #     print(len(df_m))


    # Create the scatter plot with customized markers
    plt.figure(figsize=(10, 10))
    ax = plt.gca()

    # Plot the pie charts for each scenario
    plot_type = 'pie'
    for i, row in df_m.iterrows():
        x_val = abs(row[f"{power}_without_lem"])
        y_val = abs(row[f"{power}_with_lem"])

        # # Create the pie charts for each device
        # create_donut_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5 * 2)
        # create_donut_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25 * 2)
        # create_donut_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1 * 2,
        #                    plot_inner_circle=True)

        if plot_type == 'donut':

            # Create the donut charts for each device
            create_donut_chart(ax, x_val, y_val, row["hp_penetration"], color=COLORS['hp'], radius=0.5 * 2)
            create_donut_chart(ax, x_val, y_val, row["ev_penetration"], color=COLORS['ev'], radius=0.25 * 2)
            create_donut_chart(ax, x_val, y_val, row["pv_penetration"], color=COLORS['pv'], radius=0.1 * 2,
                               plot_inner_circle=True)
        elif plot_type == 'pie':
            # Values and colors for the pie chart
            vals = [row["hp_penetration"], row["ev_penetration"], row["pv_penetration"]]  # HP, EV, PV order
            colors = [COLORS['hp'], COLORS['ev'], COLORS['pv']]

            # Create the donut charts
            create_pie_chart(ax, x_val, y_val, vals, colors)

    # Add the dashed lines and the text
    if not limits:
        max_power = max(abs(df_m[f"{power}_without_lem"]).max(), abs(df_m[f"{power}_with_lem"]).max())
        min_power = min(abs(df_m[f"{power}_without_lem"]).min(), abs(df_m[f"{power}_with_lem"]).min())
    else:
        max_power = limits[1]
        min_power = 0
    # LEM/noLEM = 1.5
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power * 1.3, 10),
            linestyle='--', color=(0.1, 0.1, 0.1), linewidth=2)
    # ax.text(max_power, max_power, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1.5$",
    #         ha='left', va='center', fontsize=10)
    # LEM/noLEM = 1.25
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power * 1.15, 10),
            linestyle='--', color=(0.25, 0.25, 0.25), linewidth=2)
    # ax.text(max_power, max_power, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1.25$",
    #         ha='left', va='center', fontsize=10)
    # LEM/noLEM = 1
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power, 10),
            linestyle='--', color=(0.4, 0.4, 0.4), linewidth=2)
    # ax.text(max_power, max_power, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
    #         ha='left', va='center', fontsize=10)
    # LEM/noLEM = 0.75
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power * 0.85, 10),
            linestyle='--', color=(0.4, 0.5, 0.6), linewidth=2)
    # ax.text(max_power, max_power * 0.75, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.75$",
    #         ha='left', va='center',                fontsize=10)
    # LEM/noLEM = 0.5
    ax.plot(np.linspace(0, 2 * max_power, 10), np.linspace(0, 2 * max_power * 0.7, 10),
            linestyle='--', color=(0.5, 0.6, 0.7), linewidth=2)
    # ax.text(max_power, max_power * 0.5, r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.5$",
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
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1.3$",
                   color=(0.1, 0.1, 0.1), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1.15$",
                   color=(0.25, 0.25, 0.25), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 1$",
                   color=(0.4, 0.4, 0.4), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.85$",
                   color=(0.4, 0.5, 0.6), linestyle='--'),
        plt.Line2D([0], [0], label=r"$\frac{ {\mathrm{with\ LEM}}}{{\mathrm{no\ LEM}}} = 0.7$",
                   color=(0.5, 0.6, 0.7), linestyle='--'),
    ]

    # Show the legend with customized markers
    ax.legend(handles=legend_elements, loc='upper left')

    # Stepsize in kW
    max_ticks = 12  # max number of ticks on x and y axis
    ticks = np.inf  # initialize ticks with infinity
    stepsize = 0   # initialize stepsize with 0
    while ticks > max_ticks:
        if stepsize <= 100:
            stepsize += 25
        elif stepsize <= 200:
            stepsize += 50
        elif stepsize <= 500:
            stepsize += 100
        elif stepsize <= 1000:
            stepsize += 200

        ticks = math.ceil((max_power - min_power) * 1.2 / stepsize)  # add 5% to ensure radius of pie chart is not cut off

    # Set x and y-axis and aspect ratio
    ax.set_xlim(stepsize * math.floor(min_power/stepsize), stepsize * math.ceil(max_power/stepsize))
    ax.set_ylim(stepsize * math.floor(min_power/stepsize), stepsize * math.ceil(max_power/stepsize))

    # Add ticks and tick labels to axes
    ax.set_xticks(np.arange(min(ax.get_xlim()), max(ax.get_xlim()) + 1, stepsize))
    ax.set_yticks(np.arange(min(ax.get_ylim()), max(ax.get_ylim()) + 1, stepsize))
    # ax.set_yticks([0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2])
    ax.set_xticklabels([f'{x:.0f}kW' for x in ax.get_xticks()])
    ax.set_yticklabels([f'{y:.0f}kW' for y in ax.get_yticks()])
    ax.set(aspect="equal")

    # Title
    plt.title(f"{topology.capitalize()}")

    # Tighten the layout
    plt.tight_layout()
    plt.grid()

    if save:
        # Save the plot
        plt.savefig(f'./figures/{topology}_power_{rate}_{comp}.png', dpi=300)

    # Show the plot
    if show:
        plt.show()

    # Close the plot again
    plt.close()


def create_power(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None, power: str = 'max_loads', make_all: bool = False):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        # Format the dataframe for plotting
        df_temp = format_df_for_plots(df, topology, weights)

        # Get limits beforehand to ensure scaling is the same
        limits = [min(abs(df_temp[f"{power}_without_lem"]).max(), abs(df_temp[f"{power}_with_lem"]).max()),
                  max(abs(df_temp[f"{power}_without_lem"]).max(), abs(df_temp[f"{power}_with_lem"]).max())]

        # Plot all scenarios
        components = ['pv_penetration', 'hp_penetration', 'ev_penetration']
        rates = list(set(df_temp['pv_penetration']))
        if make_all:
            for comp in components:
                for rate in rates:
                    df_m = df_temp[df_temp[comp] == rate]

                    # Create the scatter plot
                    plot_power(df_m, topology, rate, comp, show, save, limits)

        # Plot one with all datapoints
        plot_power(df_temp, topology, 'all', 'all', show, save, limits)


def plot_indepth(df_m: pd.DataFrame, topology: str, show: bool = False, save: bool = False, power: str = 'max_loads'):
        # Use this if you want to reduce the dataframe to a certain penetration rate
        #df_m = df_m[(df_m["pv_penetration"] > 0)]  # & (df_m["pv_penetration"] <= 0.5)]
        # df_m = df_m[(df_m["pv_penetration"] > 0) & (df_m["pv_penetration"] < 1)
        #             & (df_m["hp_penetration"] > 0) & (df_m["hp_penetration"] < 1)
        #             & (df_m["ev_penetration"] > 0) & (df_m["ev_penetration"] < 1)]
        # Note: In the analysis PV needs to be greater than 0
        df_m = df_m[(df_m["pv_penetration"] > 0)]

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
            # Ausgleichsgerade
            # ax.plot(np.linspace(-.1, 1.1, 100), p(np.linspace(-.1, 1.1, 100)), color='black', linewidth=2, alpha=0.7)
            # Ausgleichsgeradenformel
            # ax.text(1, 0.9, f"y = {z[0]:.2f}x + {z[1]:.2f}", ha='right', va='top', fontsize=10)
            res = stats.linregress(np.linspace(-.1, 1.1, 100), p(np.linspace(-.1, 1.1, 100)))
            ax.plot(np.linspace(-.1, 1.1, 10), np.linspace(1, 1, 10), linestyle='--', color='black', linewidth=1)
            ax.set_xlabel(f"{device.upper()} Penetration Level")
            ax.set_ylabel("")
            ax.set_title(f"Effect of {device.upper()} on Average Energy Price")
            ax.set_xlim(-.1, 1.1)
            # ax.set_ylim(0.8, 1.1)
            ax.set_ylim(.7, 1.15)

        # Plot for peak power
        for i, device in enumerate(device_types):
            ax = axs[i, 1]
            data = [df_m.loc[df_m[device + "_penetration"] == x, f"{power}_ratio"].values
                    for x in df_m[device + "_penetration"].unique()]
            c = COLORS[device]
            bp = ax.boxplot(data, positions=df_m[device + "_penetration"].unique(), widths=0.1, patch_artist=True,
                       boxprops=dict(facecolor='white', color=c),
                       capprops=dict(color=c),
                       whiskerprops=dict(color=c),
                       flierprops=dict(color=c, markeredgecolor=c),
                       medianprops=dict(color=c),)
            # print(topology)
            # for idx in range(len(bp['medians'])):
            #     print(bp['medians'][idx].get_ydata())
            avg = [sum(x) / len(x) for x in data]
            z = np.polyfit(df_m[device + "_penetration"].unique(), avg, 1)
            p = np.poly1d(z)
            # Ausgleichsgerade
            # ax.plot(np.linspace(-.1, 1.1, 100), p(np.linspace(-.1, 1.1, 100)), color='black', linewidth=2, alpha=0.7)
            # Ausgleichsgeradenformel
            # ax.text(1, 0.9, f"y = {z[0]:.2f}x + {z[1]:.2f}", ha='right', va='top', fontsize=10)
            res = stats.linregress(np.linspace(-.1, 1.1, 100), p(np.linspace(-.1, 1.1, 100)))
            ax.plot(np.linspace(-.1, 1.1, 10), np.linspace(1, 1, 10), linestyle='--', color='black', linewidth=1)
            ax.set_xlabel(f"{device.upper()} Penetration Level")
            ax.set_ylabel("")
            ax.set_title(f"Effect of {device.upper()} on Peak Power")
            ax.set_xlim(-.1, 1.1)
            ax.set_ylim(.7, 1.15)

        # Title
        plt.title(f"{topology.capitalize()}")

        # Tighten the layout
        plt.tight_layout()

        if save:
            # Save the plot
            plt.savefig(f'./figures/{topology}_indepth.png', dpi=300)

        # Show the plot
        if show:
            plt.show()

        plt.close()


def create_indepth(df: pd.DataFrame, show: bool = True, save: bool = False, weights: dict = None, power: str = 'max_loads'):
    # Loop through the grid topologies
    for topology in df["grid_topology"].unique():

        # Format the dataframe for plotting
        df_m = format_df_for_plots(df, topology, weights)

        # Create the figure with subplots
        plot_indepth(df_m, topology, show, save, power)


if __name__ == "__main__":
    # Folder that contains simulation results
    path = './simulation_results'

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
    # Set the maximum load share that is to be included in the calculations
    max_loads = 0.15  # typical values: 0.10 and 0.15

    # Create the file containing all information
    # df = create_data(src=path, limits_dict=limits, max_loads=max_loads)

    # Save the dataframe to a csv file
    # df.to_csv(f'./validation_dataset_{max_loads}.csv')
    # exit()

    # Load the dataframe from a csv file
    df = pd.read_csv(f'./validation_dataset_{max_loads}.csv', index_col=0)

    power = 'max_loads'  # options: 'max_loads', 'peak_power'

    show = True
    save = False
    all = False

    # Create plot comparing price and power with and without LEM
    create_price_power(df, show=show, save=save, weights=weights, power=power, make_all=all)

    # Create plot comparing price with and without LEM
    # create_price(df, show=show, save=save, weights=weights, make_all=all)

    # Create plot comparing power with and without LEM
    # create_power(df, show=show, save=save, weights=weights, power=power, make_all=all)

    # Create in-depth plot for each device type
    # create_indepth(df, show=show, save=save, weights=weights, power=power)
