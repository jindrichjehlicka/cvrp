import glob
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import ast

# Define the file pattern for merging
file_pattern = './simulated_annealing_epoch_data_chunk_*.csv'
file_paths = glob.glob(file_pattern)

all_data = pd.concat([pd.read_csv(file_path) for file_path in file_paths])

# Extract relevant columns
parameters = all_data[['max_iterations', 'initial_temperature', 'cooling_rate']]
epoch_data_raw = all_data['epoch_data']

# Process the epoch data from string to list of floats
epoch_data_processed = epoch_data_raw.apply(ast.literal_eval)
all_data['epoch_data_processed'] = epoch_data_processed

# Define a function to normalize the epoch data
def normalize_epoch_data(array):
    array = np.array(array)
    array = np.abs(array)
    max_val = np.max(array)
    # Normalize the differences between 0 and 1 based on max value
    normalized_array = array / max_val
    return normalized_array

# Normalize the epoch data
all_data['normalized_epoch_data'] = all_data['epoch_data_processed'].apply(normalize_epoch_data)

# Group by parameters
grouped = all_data.groupby(['max_iterations', 'initial_temperature', 'cooling_rate'])

# Generate plots for each group
for name, group in grouped:
    plt.figure(figsize=(10, 6))
    plt.title(f'Max Iterations: {name[0]}, Initial Temperature: {name[1]}, Cooling Rate: {name[2]}')
    all_runs = []

    max_generations = max([len(normalized_array) for normalized_array in group['normalized_epoch_data']])

    for normalized_array in group['normalized_epoch_data']:
        all_runs.append(
            np.pad(normalized_array, (0, max_generations - len(normalized_array)), 'constant', constant_values=np.nan)
        )

    all_runs = np.array(all_runs)
    mean_run = np.nanmean(all_runs, axis=0)
    std_run = np.nanstd(all_runs, axis=0)

    generations = np.arange(max_generations)

    for individual_run in all_runs:
        plt.plot(generations, individual_run, alpha=0.2, color='lightgrey', zorder=1)

    # Plot the mean and standard deviation on top
    plt.plot(generations, mean_run, label='Mean', color='blue', linewidth=2, zorder=2)
    plt.fill_between(generations, mean_run - std_run, mean_run + std_run, color='blue', alpha=0.2, label='Std Dev', zorder=2)

    # Label the plot
    plt.xlabel('Iterations')
    plt.ylabel('Normalized Cost')
    plt.legend()
    plt.grid(True)
    plt.show()
