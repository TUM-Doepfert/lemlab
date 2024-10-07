import pandas as pd

df = pd.read_csv('weather.csv')

df.to_feather('weather.ft')