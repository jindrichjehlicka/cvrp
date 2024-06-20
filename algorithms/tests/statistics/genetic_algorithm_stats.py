import pandas as pd
import numpy as np
from scipy.stats import chi2, f_oneway
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import ParameterGrid

# Load the data
data = pd.read_csv('../../results/ga_params_results.csv')

# Parse the 'Parameters' column
data['Parameters'] = data['Parameters'].apply(eval)

# Extract the parameter values into separate columns
data['population_size'] = data['Parameters'].apply(lambda x: x['population_size'])
data['generations'] = data['Parameters'].apply(lambda x: x['generations'])

# Convert Parameters dict to a frozenset for grouping
data['Parameters'] = data['Parameters'].apply(lambda x: frozenset(x.items()))

# Calculate the closeness to optimal
data['closeness_to_optimal'] = (data['Total Cost'] - data['Optimal Cost']) / data['Optimal Cost'] * 100


# Calculate Mahalanobis distance for outlier detection
def mahalanobis_distances(df, cols):
    mean = df[cols].mean().values
    cov = np.cov(df[cols].values.T)
    inv_covmat = np.linalg.inv(cov)
    distances = df[cols].apply(lambda x: mahalanobis(x, mean, inv_covmat), axis=1)
    return distances


data['mahalanobis'] = mahalanobis_distances(data, ['Total Cost', 'Execution Time (s)'])
data['p_value'] = 1 - chi2.cdf(data['mahalanobis'], df=2)
threshold = 0.01  # Significance level

data_filtered = data[data['p_value'] > threshold]

# Aggregate the filtered data
aggregated = data_filtered.groupby(['Instance', 'Parameters']).agg({
    'Total Cost': ['mean', 'std', 'median'],
    'Execution Time (s)': ['mean', 'std', 'median'],
    'Difference': ['mean', 'std', 'median'],
    'closeness_to_optimal': ['mean', 'std', 'median']
}).reset_index()

aggregated.columns = [' '.join(col).strip() for col in aggregated.columns.values]

# Extract population_size and generations for plotting
aggregated['population_size'] = aggregated['Parameters'].apply(lambda x: dict(x)['population_size'])
aggregated['generations'] = aggregated['Parameters'].apply(lambda x: dict(x)['generations'])

# Plot the results
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population_size', y='closeness_to_optimal mean', hue='generations')
plt.title('Mean Closeness to Optimal by Parameters')
plt.ylabel('Mean Closeness to Optimal (%)')
plt.xlabel('Population Size')
plt.legend(title='Generations')
plt.savefig('mean_closeness_to_optimal.png')

plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population_size', y='Execution Time (s) mean', hue='generations')
plt.title('Mean Execution Time by Parameters')
plt.ylabel('Mean Execution Time (s)')
plt.xlabel('Population Size')
plt.legend(title='Generations')
plt.savefig('mean_execution_time.png')

plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population_size', y='Difference mean', hue='generations')
plt.title('Mean Difference by Parameters')
plt.ylabel('Mean Difference')
plt.xlabel('Population Size')
plt.legend(title='Generations')
plt.savefig('mean_difference.png')

plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population_size', y='closeness_to_optimal median', hue='generations')
plt.title('Median Closeness to Optimal by Parameters')
plt.ylabel('Median Closeness to Optimal (%)')
plt.xlabel('Population Size')
plt.legend(title='Generations')
plt.savefig('median_closeness_to_optimal.png')

plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population_size', y='Execution Time (s) median', hue='generations')
plt.title('Median Execution Time by Parameters')
plt.ylabel('Median Execution Time (s)')
plt.xlabel('Population Size')
plt.legend(title='Generations')
plt.savefig('median_execution_time.png')

plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population_size', y='Difference median', hue='generations')
plt.title('Median Difference by Parameters')
plt.ylabel('Median Difference')
plt.xlabel('Population Size')
plt.legend(title='Generations')
plt.savefig('median_difference.png')


# Perform F-tests
def perform_f_tests(df, group_col, target_col):
    groups = df.groupby(group_col)[target_col].apply(list)
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


f_stat_closeness, p_value_closeness = perform_f_tests(data_filtered, 'Parameters', 'closeness_to_optimal')
f_stat_time, p_value_time = perform_f_tests(data_filtered, 'Parameters', 'Execution Time (s)')
f_stat_difference, p_value_difference = perform_f_tests(data_filtered, 'Parameters', 'Difference')

# Print F-test results
print(f"F-test for Closeness to Optimal: F-statistic = {f_stat_closeness}, p-value = {p_value_closeness}")
print(f"F-test for Execution Time: F-statistic = {f_stat_time}, p-value = {p_value_time}")
print(f"F-test for Difference: F-statistic = {f_stat_difference}, p-value = {p_value_difference}")

# Print aggregated statistics
print(aggregated)

# Save the plots
plt.savefig('aggregated_statistics.png')
plt.show()
