import pandas as pd
import glob
import os
from scipy.stats import wilcoxon
from itertools import combinations

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

algorithms = final_df['Algorithm'].unique()

results = []
for algo1, algo2 in combinations(algorithms, 2):
    cost1 = final_df[final_df['Algorithm'] == algo1]['Cost Difference'].dropna()
    cost2 = final_df[final_df['Algorithm'] == algo2]['Cost Difference'].dropna()

    min_len = min(len(cost1), len(cost2))
    cost1 = cost1.iloc[:min_len]
    cost2 = cost2.iloc[:min_len]

    if len(cost1) > 0 and len(cost2) > 0:
        stat, p_value = wilcoxon(cost1, cost2)
        results.append((algo1, algo2, stat, p_value))
    else:
        results.append((algo1, algo2, None, None))

for result in results:
    if result[2] is not None and result[3] is not None:
        print(
            f'Wilcoxon Signed-Rank Test between {result[0]} and {result[1]}: statistic={result[2]}, p-value={result[3]}')
    else:
        print(f'Wilcoxon Signed-Rank Test between {result[0]} and {result[1]}: insufficient data')
