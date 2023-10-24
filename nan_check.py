import os
import pandas as pd
from tqdm import tqdm

path = './scenarios/urban'

folders = next(os.walk(path))[1]

counter = 0
for folder in tqdm(folders, desc="Processing Scenarios", unit="scenario(s)"):
# for folder in folders:
    agents_path = os.path.join(path, folder, 'prosumer')
    agents = next(os.walk(agents_path))[1]
    for agent in agents:
        agent_path = os.path.join(agents_path, agent)
        files = [file for file in next(os.walk(agent_path))[2] if file.endswith('.ft') and file.startswith('raw_data')]
        for file in files:
            file_path = os.path.join(agent_path, file)
            df = pd.read_feather(file_path)

            # Check if there are NaN values
            if df.isnull().values.any():
                counter += 1

            # Interpolate NaN values for the DataFrame
            #df = df.interpolate(method='linear')

            # Use backward fill for the first row if there are still NaN values
            #df.iloc[0] = df.iloc[0].fillna(method='bfill')

            # Use backward fill for dataframe
            df = df.fillna(method='bfill')

            # Check if NaN are still there
            if df.isnull().values.any():
                print(f'NaN in {file} of {agent} in {folder}')
            else:
                df.to_feather(file_path)

print(f'Found {counter} files with NaN values')

