import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

epoch_filename = 'tabu_search_epoch_data_20240624_150042.csv'
epoch_df = pd.read_csv(epoch_filename)

epoch_df['epoch_data'] = epoch_df['epoch_data'].apply(eval)


def normalize_epoch_data(row):
    epoch_data = np.array(row['epoch_data'])
    initial_cost = epoch_data[0]
    normalized_data = 1 - (epoch_data / initial_cost)
    return normalized_data.tolist()


epoch_df['normalized_epoch_data'] = epoch_df.apply(normalize_epoch_data, axis=1)

plt.figure(figsize=(12, 8))

for idx, row in epoch_df.iterrows():
    normalized_epoch_data = np.array(row['normalized_epoch_data'])
    iterations = list(range(len(normalized_epoch_data)))

    plt.plot(iterations, normalized_epoch_data,
             label=f'Iter: {row["max_iterations"]}, Tabu: {row["tabu_size"]}, Neigh: {row["neighborhood_size"]}')
    plt.fill_between(iterations, normalized_epoch_data, alpha=0.2)

plt.title('Normalized Convergence Plot for Tabu Search')
plt.xlabel('Iterations')
plt.ylabel('Normalized Improvement')
plt.legend(loc='upper left', bbox_to_anchor=(1, 1))
plt.tight_layout()
plt.show()
