import pandas as pd
import numpy as np
from scipy.stats import chi2, f_oneway
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('../../results/aco_algorithm_comparison_results.csv')

# Assuming 'Parameters' is a string representation of a dictionary, we need to parse it
data['Parameters'] = data['Parameters'].apply(eval)

# Extract the parameter values into separate columns
data['num_ants'] = data['Parameters'].apply(lambda x: x['num_ants'])
data['iterations'] = data['Parameters'].apply(lambda x: x['iterations'])
data['decay'] = data['Parameters'].apply(lambda x: x['decay'])

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

# Extract num_ants, iterations, and decay for plotting
aggregated['num_ants'] = aggregated['Parameters'].apply(lambda x: dict(x)['num_ants'])
aggregated['iterations'] = aggregated['Parameters'].apply(lambda x: dict(x)['iterations'])
aggregated['decay'] = aggregated['Parameters'].apply(lambda x: dict(x)['decay'])

# Plot the results
# Mean closeness to optimal by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='num_ants', y='closeness_to_optimal mean', hue='iterations')
plt.title('Mean Closeness to Optimal by Parameters')
plt.ylabel('Mean Closeness to Optimal (%)')
plt.xlabel('Number of Ants')
plt.legend(title='Iterations')
plt.savefig('aco_mean_closeness_to_optimal.png')

# Mean execution time by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='num_ants', y='Execution Time (s) mean', hue='iterations')
plt.title('Mean Execution Time by Parameters')
plt.ylabel('Mean Execution Time (s)')
plt.xlabel('Number of Ants')
plt.legend(title='Iterations')
plt.savefig('aco_mean_execution_time.png')

# Mean difference by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='num_ants', y='Difference mean', hue='iterations')
plt.title('Mean Difference by Parameters')
plt.ylabel('Mean Difference')
plt.xlabel('Number of Ants')
plt.legend(title='Iterations')
plt.savefig('aco_mean_difference.png')

# Median closeness to optimal by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='num_ants', y='closeness_to_optimal median', hue='iterations')
plt.title('Median Closeness to Optimal by Parameters')
plt.ylabel('Median Closeness to Optimal (%)')
plt.xlabel('Number of Ants')
plt.legend(title='Iterations')
plt.savefig('aco_median_closeness_to_optimal.png')

# Median execution time by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='num_ants', y='Execution Time (s) median', hue='iterations')
plt.title('Median Execution Time by Parameters')
plt.ylabel('Median Execution Time (s)')
plt.xlabel('Number of Ants')
plt.legend(title='Iterations')
plt.savefig('aco_median_execution_time.png')

# Median difference by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='num_ants', y='Difference median', hue='iterations')
plt.title('Median Difference by Parameters')
plt.ylabel('Median Difference')
plt.xlabel('Number of Ants')
plt.legend(title='Iterations')
plt.savefig('aco_median_difference.png')


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
plt.savefig('aco_aggregated_statistics.png')
plt.show()
