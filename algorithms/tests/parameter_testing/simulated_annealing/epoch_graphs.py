import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob

all_data = pd.read_csv('./simulated_annealing_epoch_data_20240627_194850.csv')

# Extract relevant columns
parameters = all_data[['max_iterations', 'initial_temperature', 'cooling_rate']]
epoch_data_raw = all_data['epoch_data']

epoch_data_processed = epoch_data_raw.apply(lambda x: eval(x))

all_data['epoch_data_processed'] = epoch_data_processed


# Define a function to normalize the epoch data
def normalize_epoch_data(array):
    array = np.array(array)
    min_val = np.min(array)
    max_val = np.max(array)
    normalized_array = (array - min_val) / (max_val - min_val)
    # # Invert the normalized values to go from 1 to 0
    # normalized_array = 1 - normalized_array
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

    max_generations = max([len(normalized_array[0]) for normalized_array in group['normalized_epoch_data']])

    for normalized_array in group['normalized_epoch_data']:
        for individual_run in normalized_array:
            all_runs.append(
                np.pad(individual_run, (0, max_generations - len(individual_run)), 'constant', constant_values=np.nan))

    all_runs = np.array(all_runs)
    mean_run = np.nanmean(all_runs, axis=0)
    std_run = np.nanstd(all_runs, axis=0)

    generations = np.arange(max_generations)

    for individual_run in all_runs:
        plt.plot(generations, individual_run, alpha=0.02, color='lightgrey', zorder=1)

    # Plot the mean and standard deviation on top
    plt.plot(generations, mean_run, label='Mean', color='blue', linewidth=2, zorder=2)
    plt.fill_between(generations, mean_run - std_run, mean_run + std_run, color='blue', alpha=0.2, label='Std Dev',
                     zorder=2)

    # Label the plot
    plt.xlabel('Iterations')
    plt.ylabel('Normalized Cost')
    plt.legend()
    plt.grid(True)
    plt.show()
