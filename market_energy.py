import pandas as pd

def calc_market_balancing_energy(info: dict, market_file: str, limits: list = None):
    # Get the market data
    df_market = pd.read_csv(market_file, index_col=0)

    # Limit the market data to the specified limits if available
    if limits:
        df_market = df_market[(limits[0] <= df_market['ts_delivery']) & (df_market['ts_delivery'] < limits[1])]

    # Get all market trades with negative energy amount (energy purchases)
    print(len(df_market))
    df_market = df_market[df_market['qty_energy'] < 0]
    print(len(df_market))

    # Exclude the retailer from the market data
    df_market = df_market[df_market['id_user'] != 'retailer01']
    print(len(df_market))
    print(df_market.head(10).to_string())
    print(df_market.tail(10).to_string())
    exit()
    # Group by transaction type
    df_market = df_market.groupby(by='type_transaction').sum()

    # Add info to dict
    try:
        info["energy_market"] = int(abs(df_market.loc['market', 'qty_energy'] / 1000))
        info["energy_balancing"] = int(abs(df_market.loc['balancing', 'qty_energy'] / 1000))
    except KeyError:
        # If there is no market, the balancing energy is the market energy
        info["energy_market"] = int(abs(df_market.loc['balancing', 'qty_energy'] / 1000))
        info["energy_balancing"] = 0
    # info["costs_market"] = int(abs(df_market.loc['market', 'delta_balance'] / 1000000 / 1000))
    # info["costs_balancing"] = int(abs(df_market.loc['balancing', 'delta_balance'] / 1000000 / 1000))

    return info

if __name__ == "__main__":
    info = {}
    market_file = './simulation_results/countryside/wLEM_mpc_naive/countryside_pv_1.0_hp_1.0_ev_1.0_winter_wLEM_mpc_naive/db_snapshot/logs_transactions.csv'
    limits = {
        "winter": [1573513200, 1574118000],
        "summer": [1563832800, 1564437600],
        "transition": [1569276000, 1569880800],
    }
    limits = limits['winter']

    calc_market_balancing_energy(info, market_file, limits)