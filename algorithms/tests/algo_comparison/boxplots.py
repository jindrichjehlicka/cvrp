import pandas as pd
import glob
import os
import matplotlib.pyplot as plt
import seaborn as sns

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

plt.figure(figsize=(12, 6))
sns.boxplot(x='Algorithm', y='Cost Difference', data=final_df)
plt.title('Box Plot of Cost Difference by Algorithm')
plt.ylabel('Cost Difference')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplot_cost_difference.png')
plt.show()

plt.figure(figsize=(12, 6))
sns.boxplot(x='Algorithm', y='Time', data=final_df)
plt.yscale('log')
plt.title('Box Plot of Time by Algorithm (Log Scale)')
plt.ylabel('Time (seconds, log scale)')
plt.xlabel('Algorithm')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('boxplot_time_log_scale.png')
plt.show()
