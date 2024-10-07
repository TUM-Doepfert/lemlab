import pandas as pd


topology = 'Suburban'
df = pd.read_excel(f'{topology}_raw.xlsx', sheet_name=f'{topology}_raw', index_col=0)

print(df['max_loads'][0])
print(type(df['max_loads'][0]))
df['max_loads'] = df['max_loads'].apply(lambda x: x[1:-1].split(',')).apply(lambda x: [round(float(i), 1) for i in x])

df['avg_max_loads'] = df['max_loads'].apply(lambda x: round(sum(x) / len(x), 1))

print(df['max_loads'][0])
print(type(df['max_loads'][0]))
print(type(df['max_loads'][0][0]))

print(df['avg_max_loads'][0])

df.to_excel(f'{topology}_avg_max_loads.xlsx', sheet_name=f'{topology}_avg_max_loads')