import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import glob
import os

output_dir = 'graphs'
os.makedirs(output_dir, exist_ok=True)

file_pattern = './ga_sa_hybrid_epoch_data_chunk_2_*.csv'
file_paths = glob.glob(file_pattern)

all_data = pd.concat([pd.read_csv(file_path) for file_path in file_paths])

parameters = all_data[['population_size', 'generations', 'initial_temperature', 'cooling_rate']]
epoch_data_raw = all_data['epoch_data']

epoch_data_processed = epoch_data_raw.apply(lambda x: eval(x))

all_data['epoch_data_processed'] = epoch_data_processed

def normalize_epoch_data(array):
    array = np.array(array)
    array = np.abs(array)
    max_val = np.max(array)
    normalized_array = array / max_val if max_val != 0 else array
    return normalized_array

all_data['normalized_epoch_data'] = all_data['epoch_data_processed'].apply(normalize_epoch_data)

grouped = all_data.groupby(['population_size', 'generations', 'initial_temperature', 'cooling_rate'])

for name, group in grouped:
    plt.figure(figsize=(10, 6))
    plt.title(f'Population: {name[0]}, Generations: {name[1]}, Temperature: {name[2]}, Cooling Rate: {name[3]}')
    all_runs = []

    max_generations = max([len(normalized_array) for normalized_array in group['normalized_epoch_data']])

    for normalized_array in group['normalized_epoch_data']:
        all_runs.append(np.pad(normalized_array, (0, max_generations - len(normalized_array)), 'constant', constant_values=np.nan))

    all_runs = np.array(all_runs)
    mean_run = np.nanmean(all_runs, axis=0)
    std_run = np.nanstd(all_runs, axis=0)

    generations = np.arange(max_generations)

    for individual_run in all_runs:
        plt.plot(generations, individual_run, alpha=0.02, color='lightgrey', zorder=1)

    plt.plot(generations, mean_run, label='Mean', color='blue', linewidth=2, zorder=2)
    plt.fill_between(generations, mean_run - std_run, mean_run + std_run, color='blue', alpha=0.2, label='Std Dev', zorder=2)

    plt.xlabel('Generations')
    plt.ylabel('Normalized Optimal Cost / Cost')
    plt.legend()
    plt.grid(True)

    file_name = f'Population_{name[0]}_Generations_{name[1]}_Temperature_{name[2]}_CoolingRate_{name[3]}.png'
    plt.savefig(os.path.join(output_dir, file_name))
    plt.close()
