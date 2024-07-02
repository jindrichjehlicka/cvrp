import glob

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Define the file pattern for merging
file_pattern = './tabu_search_epoch_data_chunk_*.csv'
file_paths = glob.glob(file_pattern)

# Load and concatenate all matching CSV files
all_data = pd.concat([pd.read_csv(file_path) for file_path in file_paths])

# Extract relevant columns
parameters = all_data[['max_iterations', 'tabu_size', 'neighborhood_size']]
epoch_data_raw = all_data['epoch_data']

# Convert the string representation of lists to actual lists
epoch_data_processed = epoch_data_raw.apply(lambda x: eval(x))

# Add the processed epoch data to the DataFrame
all_data['epoch_data_processed'] = epoch_data_processed

import numpy as np

def normalize_epoch_data(array):
    array = np.array(array)
    # Convert to positive if necessary
    array = np.abs(array)
    max_val = np.max(array)
    # Normalize the differences between 0 and 1 based on max value
    normalized_array = array / max_val
    return normalized_array

# Normalize the epoch data
all_data['normalized_epoch_data'] = all_data['epoch_data_processed'].apply(normalize_epoch_data)

# Group by parameters
grouped = all_data.groupby(['max_iterations', 'tabu_size', 'neighborhood_size'])

# Generate plots for each group
for name, group in grouped:
    plt.figure(figsize=(10, 6))
    plt.title(f'Max Iterations: {name[0]}, Tabu Size: {name[1]}, Neighborhood Size: {name[2]}')

    all_runs = []
    max_generations = max([len(normalized_array) for normalized_array in group['normalized_epoch_data']])

    for normalized_array in group['normalized_epoch_data']:
        all_runs.append(
            np.pad(normalized_array, (0, max_generations - len(normalized_array)), 'constant', constant_values=np.nan))

    all_runs = np.array(all_runs)
    mean_run = np.nanmean(all_runs, axis=0)
    std_run = np.nanstd(all_runs, axis=0)

    generations = np.arange(max_generations)

    for individual_run in all_runs:
        plt.plot(generations, individual_run, alpha=0.2, color='lightgrey', zorder=1)

    mean_run = np.nan_to_num(mean_run, nan=np.nan)
    std_run = np.nan_to_num(std_run, nan=np.nan)

    plt.plot(generations, mean_run, label='Mean', color='blue', linewidth=2, zorder=2)
    plt.fill_between(generations, mean_run - std_run, mean_run + std_run, color='blue', alpha=0.2, label='Std Dev', zorder=2)

    # Label the plot
    plt.xlabel('Iterations')
    plt.ylabel('Normalized Cost')
    plt.legend()
    plt.grid(True)
    plt.show()
