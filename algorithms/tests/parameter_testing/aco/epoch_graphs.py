
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

# Load the epoch data
epoch_df = pd.read_csv('genetic_algorithm_epoch_data_20240623_155728.csv')

# Ensure 'epoch_data' column is properly evaluated
epoch_df['epoch_data'] = epoch_df['epoch_data'].apply(eval)


# Normalize the epoch data
def normalize_epoch_data(row):
    epoch_data = np.array(row['epoch_data'])
    initial_cost = epoch_data[0]
    normalized_data = 1 - (epoch_data / initial_cost)
    return normalized_data.tolist()


epoch_df['normalized_epoch_data'] = epoch_df.apply(normalize_epoch_data, axis=1)

# Generate plots for each parameter set
plt.figure(figsize=(10, 6))

for idx, row in epoch_df.iterrows():
    normalized_epoch_data = np.array(row['normalized_epoch_data'])
    generations = list(range(len(normalized_epoch_data)))

    plt.plot(generations, normalized_epoch_data,
             label=f'Pop: {row["population_size"]}, Gen: {row["generations"]}, Mut: {row["mutation_rate"]}')
    plt.fill_between(generations, normalized_epoch_data, alpha=0.2)

plt.title('Normalized Convergence Plot')
plt.xlabel('Generations')
plt.ylabel('Normalized Improvement')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
