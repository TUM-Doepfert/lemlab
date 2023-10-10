import pandas as pd

def calc_avgenergyprice(info: dict, market_file: str, limits: list = None):
    # Get the market data
    df_market = pd.read_csv(market_file, index_col=0)

    # Limit the market data to the specified limits if available
    if limits:
        df_market = df_market[(limits[0] <= df_market['ts_delivery']) & (df_market['ts_delivery'] < limits[1])]

    print(len(df_market['ts_delivery'].unique()) * 0.25)
    exit()

    # TODO: Probably best to check the share of balancing vs market for each time step to see why it is so high

    # TODO: Try out a version where the average energy price is the average of both bought and own generated and
    #  consumed energy where the consumed energy is evaluated at 8.2 ct/kWh. This might bring more variance and
    #  be more realistic

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

if __name__ == "__main__":
    # Define the path to the market data
    market_file = './all_lemlab_results/Dorf/results/Rural_pv_1.0_hp_0.25_ev_0.0_summer/db_snapshot/logs_transactions.csv'

    # Limits for the time period that is to be included in the calculations [x, y)
    # Limits are from Tue to Mon since Soner's input data was off by a day XD
    limits = [1563832800, 1564437600]

    # Calculate the average energy price
    info = calc_avgenergyprice(info={}, market_file=market_file, limits=limits)

    # Print the result
    print(info['average_energy_price'])