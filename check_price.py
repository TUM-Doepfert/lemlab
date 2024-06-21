"""This script is also in analyze_results.py"""

import pandas as pd
import matplotlib.pyplot as plt

LEVIES = 0.2297
MARKET_BUY = 0.0827
MARKET_SELL = 0.147

def calc_avg_market_price(df_market):

    # Change ts_delivery to datetime
    df_market['ts_delivery'] = pd.to_datetime(df_market['ts_delivery'], unit='s')

    # Drop all rows where qty_energy is positive as it is only about the bought energy
    df_market = df_market[df_market['qty_energy'] < 0]

    # Drop all rows where the retailer bought energy as it would distort the average energy price for the prosumers
    df_market = df_market[df_market['id_user'] != 'retailer01']

    # Drop all rows that contain levies as they are not part of the energy price
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
    df_main = df_meter[(df_meter['info_additional'] == 'main meter')]  # not really needed
    df_gen = df_meter[df_meter['info_additional'] == 'pv']
    df_con = df_meter[(df_meter['info_additional'] != 'pv') & (df_meter['info_additional'] != 'main meter')]

    # Iterate over all prosumers
    for prosumer in df_selfused['id_user']:
        # Create dataframe to store the self-used energy information
        df_temp = pd.DataFrame(columns=['ts_delivery', "generation", "consumption", "self_used"])
        df_temp['ts_delivery'] = df_readings['ts_delivery'].unique()
        df_temp.set_index('ts_delivery', drop=True, inplace=True)
        # Get the main meter id of the prosumer
        main_id = df_main[df_main['id_user'] == prosumer]['id_meter'].values[0]
        # Get the generation and consumption meter ids of the prosumer
        gen_ids = df_gen[df_gen['id_user'] == prosumer]['id_meter'].values
        con_ids = df_con[df_con['id_user'] == prosumer]['id_meter'].values

        # Get the readings that belong to the prosumer for generation and consumption
        # df_pro_main = df_readings[df_readings['id_meter'] == main_id].set_index('ts_delivery', drop=True)
        df_pro_gen = df_readings[df_readings['id_meter'].isin(gen_ids)]
        df_pro_gen = df_pro_gen.groupby('ts_delivery').sum(numeric_only=True)
        df_pro_con = df_readings[df_readings['id_meter'].isin(con_ids)]
        df_pro_con = df_pro_con.groupby('ts_delivery').sum(numeric_only=True)

        # Assign the generation and consumption to the dataframe
        df_temp['generation'] = df_pro_gen['energy_out']
        df_temp['consumption'] = df_pro_con['energy_in']

        # Calculate the self-used energy
        df_temp.loc[df_temp['generation'] <= df_temp['consumption'], 'self_used'] = df_temp['generation']
        df_temp.loc[df_temp['generation'] > df_temp['consumption'], 'self_used'] = df_temp['consumption']

        # Assign the self-used energy to the self-used dataframe and calculate costs
        df_selfused.loc[df_selfused['id_user'] == prosumer, 'qty_energy'] = -df_temp['self_used'].sum() / 1e3
        df_selfused.loc[df_selfused['id_user'] == prosumer, 'price_total'] = (
                df_selfused.loc[df_selfused['id_user'] == prosumer, 'price_pu']
                * df_selfused.loc[df_selfused['id_user'] == prosumer, 'qty_energy'])

    # Append the self-owned energy information to the market dataframe
    df_market = df_market.append(df_selfused, ignore_index=True)

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
    df_market = calc_avg_market_price(df_market)

    # Add rows with information about the self-used energy to the market data
    df_market = add_selfused_energy(df_market, df_meter, df_readings)

    # Calculate the average energy price, convert it from sigma to € and add it to the info dictionary
    info['average_energy_price'] = df_market['price_total'].sum() / df_market['qty_energy'].sum()

    return info

if __name__ == "__main__":
    # Define the path to the data
    market_file = './simulation_results/countryside/wLEM/Countryside_pv_1.0_hp_0.25_ev_0.0_summer/db_snapshot/logs_transactions.csv'
    meter_file = './simulation_results/countryside/wLEM/Countryside_pv_1.0_hp_0.25_ev_0.0_summer/db_snapshot/info_meter.csv'
    readings_file = './simulation_results/countryside/wLEM/Countryside_pv_1.0_hp_0.25_ev_0.0_summer/db_snapshot/readings_meter_delta.csv'


    # Limits for the time period that is to be included in the calculations [x, y)
    # Limits are from Tue to Mon since Soner's input data was off by a day XD
    limits = [1563832800, 1564437600]

    # Calculate the average energy price
    info = calc_avgenergyprice(info={}, market_file=market_file, meter_file=meter_file, readings_file=readings_file,
                               limits=limits)

    # Print the average energy price
    print(f"Average energy price: {round(info['average_energy_price'], 3)} €/kWh")