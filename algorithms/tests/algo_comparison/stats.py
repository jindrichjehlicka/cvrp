import pandas as pd
import glob
import os
from scipy.stats import skew, kurtosis

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

grouped = final_df.groupby('Algorithm')

agg_funcs = {
    'Cost Difference': [
        'mean', 'std', 'median', 'min', 'max',
        lambda x: x.max() - x.min(),
        'var',
        lambda x: skew(x),
        lambda x: kurtosis(x, fisher=True),
        lambda x: x.quantile(0.95),
        lambda x: x.quantile(0.05),
        lambda x: (x.std() / x.mean())
    ],
    'Time': [
        'mean', 'std', 'median', 'min', 'max',
        lambda x: x.max() - x.min(),
        'var',
        lambda x: skew(x),
        lambda x: kurtosis(x, fisher=True),
        lambda x: x.quantile(0.95),
        lambda x: x.quantile(0.05),
        lambda x: (x.std() / x.mean())
    ]
}

desc_stats = grouped.agg(agg_funcs)

desc_stats.columns = [
    'Cost Difference_mean', 'Cost Difference_std', 'Cost Difference_median', 'Cost Difference_min', 'Cost Difference_max',
    'Cost Difference_Range', 'Cost Difference_var', 'Cost Difference_Skewness', 'Cost Difference_Kurtosis',
    'Cost Difference_95th Percentile', 'Cost Difference_5th Percentile', 'Cost Difference_CV',
    'Time_mean', 'Time_std', 'Time_median', 'Time_min', 'Time_max',
    'Time_Range', 'Time_var', 'Time_Skewness', 'Time_Kurtosis',
    'Time_95th Percentile', 'Time_5th Percentile', 'Time_CV'
]

print(desc_stats)
desc_stats.to_csv('summary_statistics.csv', index=True)

