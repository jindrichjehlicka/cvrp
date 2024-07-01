import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Load the specific dataset
file_path = './ga_sa_hybrid_epoch_data_chunk_10_20240701_110812.csv'
epoch_df = pd.read_csv(file_path)

# Evaluate the 'epoch_data' string into actual lists
epoch_df['epoch_data'] = epoch_df['epoch_data'].apply(eval)

# Function to normalize epoch data
def normalize_epoch_data(epoch_data):
    epoch_data = np.array(epoch_data)
    initial_cost = epoch_data[0]
    normalized_data = epoch_data / initial_cost
    return normalized_data.tolist()

# Apply normalization
epoch_df['normalized_epoch_data'] = epoch_df['epoch_data'].apply(normalize_epoch_data)

# Helper function to align lengths of lists
def align_lengths(data, length):
    return [x + [x[-1]] * (length - len(x)) if len(x) < length else x[:length] for x in data]

# Plot the normalized data for inspection
plt.figure(figsize=(12, 8))

for idx, row in epoch_df.iterrows():
    normalized_epoch_data = np.array(row['normalized_epoch_data'])
    generations = list(range(len(normalized_epoch_data)))

    plt.plot(generations, normalized_epoch_data, alpha=0.5)

plt.title('Normalized Convergence Plot')
plt.xlabel('Generations')
plt.ylabel('Normalized Cost')
plt.ylim(0, 1.1)
plt.tight_layout()
plt.show()
