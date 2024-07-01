import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the epoch data
filename = 'genetic_algorithm_epoch_data_chunk_1_20240701_190030.csv'
epoch_df = pd.read_csv(filename)

# Ensure 'epoch_data' column is properly evaluated
epoch_df['epoch_data'] = epoch_df['epoch_data'].apply(eval)

# Function to normalize epoch data
def normalize_epoch_data(epoch_data):
    epoch_data = np.array(epoch_data)
    initial_cost = epoch_data[:, 0]
    normalized_data = (epoch_data.T / initial_cost).T
    return normalized_data.tolist()

# Normalize the epoch data
epoch_df['normalized_epoch_data'] = epoch_df['epoch_data'].apply(normalize_epoch_data)

# Plot the normalized data
plt.figure(figsize=(12, 8))

for idx, row in epoch_df.iterrows():
    normalized_epoch_data = np.array(row['normalized_epoch_data'])
    mean_normalized_data = np.mean(normalized_epoch_data, axis=0)
    std_normalized_data = np.std(normalized_epoch_data, axis=0)
    generations = np.arange(len(mean_normalized_data))

    label = f"Pop: {row['population_size']}, Gen: {row['generations']}, Mut: {row['mutation_rate']}"
    plt.plot(generations, mean_normalized_data, label=label)
    plt.fill_between(generations, mean_normalized_data - std_normalized_data, mean_normalized_data + std_normalized_data, alpha=0.2)

plt.title('Normalized Convergence Plot')
plt.xlabel('Generations')
plt.ylabel('Normalized Cost')
plt.ylim(0, 1.1)
plt.legend(loc='upper right', bbox_to_anchor=(1.3, 1))
plt.tight_layout()
plt.show()
