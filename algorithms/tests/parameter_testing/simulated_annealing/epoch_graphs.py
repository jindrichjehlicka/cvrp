import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the epoch data
epoch_filename = 'simulated_annealing_epoch_data_20240627_194850.csv'
epoch_df = pd.read_csv(epoch_filename)

# Ensure 'epoch_data' column is properly evaluated
epoch_df['epoch_data'] = epoch_df['epoch_data'].apply(eval)


def normalize_epoch_data(epoch_data):
    epoch_data = np.array(epoch_data)
    initial_cost = epoch_data[:, 0]
    normalized_data = (epoch_data.T / initial_cost).T
    return normalized_data.tolist()


epoch_df['normalized_epoch_data'] = epoch_df['epoch_data'].apply(normalize_epoch_data)

grouped = epoch_df.groupby(['max_iterations', 'initial_temperature', 'cooling_rate'])

plt.figure(figsize=(12, 8))

for (params, group) in grouped:
    mean_normalized_data = np.mean(np.vstack(group['normalized_epoch_data'].values), axis=0)
    std_normalized_data = np.std(np.vstack(group['normalized_epoch_data'].values), axis=0)
    iterations = np.arange(len(mean_normalized_data))

    label = f'Iter: {params[0]}, Temp: {params[1]}, Cool: {params[2]}'
    plt.plot(iterations, mean_normalized_data, label=label)
    plt.fill_between(iterations, mean_normalized_data - std_normalized_data, mean_normalized_data + std_normalized_data,
                     alpha=0.2)

plt.title('Convergence Plot for Simulated Annealing')
plt.xlabel('Iterations')
plt.ylabel('Normalized Cost')
plt.ylim(0, 1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()
