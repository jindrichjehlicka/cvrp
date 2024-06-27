import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the epoch data
epoch_filename = './simulated_annealing_epoch_data_20240627_194850.csv'
epoch_df = pd.read_csv(epoch_filename)

epoch_df['epoch_data'] = epoch_df['epoch_data'].apply(eval)


def normalize_epoch_data(epoch_data):
    epoch_data = np.array(epoch_data)
    initial_cost = epoch_data[:, 0]  # Use the initial cost of each run for normalization
    normalized_data = (epoch_data.T / initial_cost).T
    return normalized_data.tolist()


epoch_df['normalized_epoch_data'] = epoch_df['epoch_data'].apply(normalize_epoch_data)

grouped = epoch_df.groupby('max_iterations')

for max_iter, group in grouped:
    plt.figure(figsize=(12, 8))

    for idx, row in group.iterrows():
        normalized_epoch_data = np.array(row['normalized_epoch_data'])
        iterations = np.arange(len(normalized_epoch_data[0]))

        mean_normalized_data = np.mean(normalized_epoch_data, axis=0)
        std_normalized_data = np.std(normalized_epoch_data, axis=0)

        label = f'Temp: {row["initial_temperature"]}, Cool: {row["cooling_rate"]}'
        plt.plot(iterations, mean_normalized_data, label=label)
        plt.fill_between(iterations, mean_normalized_data - std_normalized_data,
                         mean_normalized_data + std_normalized_data, alpha=0.2)

    plt.title(f'Convergence Plot for Simulated Annealing (Max Iterations: {max_iter})')
    plt.xlabel('Iterations')
    plt.ylabel('Normalized Cost')
    plt.ylim(0, 1.1)
    plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
    plt.tight_layout()
    plt.show()
