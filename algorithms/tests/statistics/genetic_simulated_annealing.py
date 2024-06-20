import pandas as pd
import numpy as np
from scipy.stats import chi2, f_oneway
from scipy.spatial.distance import mahalanobis
import matplotlib.pyplot as plt
import seaborn as sns

# Load the data
data = pd.read_csv('../results/ga_sa_hybrid_results.csv')

# Assuming 'Parameters' is a string representation of a dictionary, we need to parse it
data['Parameters'] = data['Parameters'].apply(eval)

# Extract the parameter values into separate columns
data['population_size'] = data['Parameters'].apply(lambda x: x['population_size'])
data['generations'] = data['Parameters'].apply(lambda x: x['generations'])
data['max_iterations'] = data['Parameters'].apply(lambda x: x['max_iterations'])
data['cooling_rate'] = data['Parameters'].apply(lambda x: x['cooling_rate'])

# Convert Parameters dict to a tuple for grouping
data['Parameters'] = data['Parameters'].apply(lambda x: tuple(sorted(x.items())))

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
aggregated = data_filtered.groupby(['population_size', 'generations', 'max_iterations', 'cooling_rate']).agg({
    'Total Cost': ['mean', 'median'],
    'Execution Time (s)': ['mean', 'median'],
    'closeness_to_optimal': ['mean', 'median']
}).reset_index()

aggregated.columns = ['_'.join(col).strip() if col[1] else col[0] for col in aggregated.columns.values]
aggregated.columns = [col.replace('_', ' ') for col in aggregated.columns]

# Plot the results
# Mean closeness to optimal by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population size', y='closeness to optimal mean', hue='cooling rate')
plt.title('Mean Closeness to Optimal by Parameters')
plt.ylabel('Mean Closeness to Optimal (%)')
plt.xlabel('Population Size')
plt.legend(title='Cooling Rate')
plt.savefig('ga_sa_mean_closeness_to_optimal.png')

# Mean execution time by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population size', y='Execution Time (s) mean', hue='cooling rate')
plt.title('Mean Execution Time by Parameters')
plt.ylabel('Mean Execution Time (s)')
plt.xlabel('Population Size')
plt.legend(title='Cooling Rate')
plt.savefig('ga_sa_mean_execution_time.png')

# Median closeness to optimal by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population size', y='closeness to optimal median', hue='cooling rate')
plt.title('Median Closeness to Optimal by Parameters')
plt.ylabel('Median Closeness to Optimal (%)')
plt.xlabel('Population Size')
plt.legend(title='Cooling Rate')
plt.savefig('ga_sa_median_closeness_to_optimal.png')

# Median execution time by parameters
plt.figure(figsize=(10, 6))
sns.barplot(data=aggregated, x='population size', y='Execution Time (s) median', hue='cooling rate')
plt.title('Median Execution Time by Parameters')
plt.ylabel('Median Execution Time (s)')
plt.xlabel('Population Size')
plt.legend(title='Cooling Rate')
plt.savefig('ga_sa_median_execution_time.png')


# Perform F-tests
def perform_f_tests(df, group_col, target_col):
    groups = df.groupby(group_col)[target_col].apply(list)
    f_stat, p_value = f_oneway(*groups)
    return f_stat, p_value


f_stat_closeness, p_value_closeness = perform_f_tests(data_filtered, 'Parameters', 'closeness_to_optimal')
f_stat_time, p_value_time = perform_f_tests(data_filtered, 'Parameters', 'Execution Time (s)')

# Print F-test results
print(f"F-test for Closeness to Optimal: F-statistic = {f_stat_closeness}, p-value = {p_value_closeness}")
print(f"F-test for Execution Time: F-statistic = {f_stat_time}, p-value = {p_value_time}")

# Print aggregated statistics
print(aggregated)

# Save the plots
plt.savefig('ga_sa_aggregated_statistics.png')
plt.show()
