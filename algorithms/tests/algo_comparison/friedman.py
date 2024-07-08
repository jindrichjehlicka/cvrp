import pandas as pd
import glob
import os
from scipy.stats import friedmanchisquare
import numpy as np

folder_path = './'
csv_files = glob.glob(os.path.join(folder_path, '*.csv'))
dataframes = []

for file in csv_files:
    df = pd.read_csv(file)
    dataframes.append(df)

combined_df = pd.concat(dataframes, ignore_index=True)

combined_df['Iteration'] = combined_df['Iteration'].astype(float)

unique_columns = ['Algorithm', 'Instance', 'Run']
with_iterations = combined_df[combined_df['Iteration'].notna()]
without_iterations = combined_df[combined_df['Iteration'].isna()]

if not with_iterations.empty:
    last_iteration_idx = with_iterations.groupby(unique_columns)['Iteration'].idxmax()
    last_iteration_df = with_iterations.loc[last_iteration_idx]
else:
    last_iteration_df = pd.DataFrame()

final_df = pd.concat([last_iteration_df, without_iterations], ignore_index=True)

cost_data = []
algorithms = final_df['Algorithm'].unique()
max_length = max(final_df['Algorithm'].value_counts())

for algo in algorithms:
    cost_diff = final_df[final_df['Algorithm'] == algo]['Cost Difference'].dropna().values
    if len(cost_diff) < max_length:
        cost_diff = np.pad(cost_diff, (0, max_length - len(cost_diff)), constant_values=np.nan)
    cost_data.append(cost_diff)

cost_data = np.array(cost_data).T
cost_data = cost_data[~np.isnan(cost_data).any(axis=1)]

stat, p_value = friedmanchisquare(*cost_data.T)

print(f'Friedman Test: statistic={stat}, p-value={p_value:.999}')
