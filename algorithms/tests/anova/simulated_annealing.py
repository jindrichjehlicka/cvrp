import pandas as pd
import numpy as np
from scipy.stats import f_oneway
import glob


# Load data from multiple CSV files
def load_data(file_pattern):
    file_paths = glob.glob(file_pattern)
    all_data = pd.concat([pd.read_csv(file_path) for file_path in file_paths])
    all_data['epoch_data'] = all_data['epoch_data'].apply(lambda x: eval(x))
    return all_data


# Extract final scores
def extract_final_scores(df):
    final_scores = []
    for _, row in df.iterrows():
        key = (row['max_iterations'], row['initial_temperature'], row['cooling_rate'])
        final_score = row['epoch_data'][-1]
        final_scores.append({
            'max_iterations': row['max_iterations'],
            'initial_temperature': row['initial_temperature'],
            'cooling_rate': row['cooling_rate'],
            'final_score': final_score
        })
    return pd.DataFrame(final_scores)


# Group by parameter and calculate summary statistics
def group_by_parameter(final_scores):
    grouped = final_scores.groupby(['max_iterations', 'initial_temperature', 'cooling_rate']).agg(
        mean_final_score=('final_score', 'mean'),
        std_final_score=('final_score', 'std')
    ).reset_index()
    return grouped


# Perform ANOVA
def perform_anova(grouped):
    anova_results = {}
    # Group by max_iterations
    for max_iter in grouped['max_iterations'].unique():
        groups = [group['mean_final_score'].values for name, group in
                  grouped[grouped['max_iterations'] == max_iter].groupby('initial_temperature')]
        if len(groups) > 1:
            anova_results[f'max_iterations_{max_iter}'] = f_oneway(*groups)

    # Group by initial_temperature
    for temp in grouped['initial_temperature'].unique():
        groups = [group['mean_final_score'].values for name, group in
                  grouped[grouped['initial_temperature'] == temp].groupby('cooling_rate')]
        if len(groups) > 1:
            anova_results[f'initial_temperature_{temp}'] = f_oneway(*groups)

    # Group by cooling_rate
    for rate in grouped['cooling_rate'].unique():
        groups = [group['mean_final_score'].values for name, group in
                  grouped[grouped['cooling_rate'] == rate].groupby('max_iterations')]
        if len(groups) > 1:
            anova_results[f'cooling_rate_{rate}'] = f_oneway(*groups)

    return anova_results


# Main function
def main(file_pattern):
    df = load_data(file_pattern)
    final_scores = extract_final_scores(df)
    grouped = group_by_parameter(final_scores)

    # Debug prints
    print("Final Scores DataFrame:")
    print(final_scores.head())

    print("\nGrouped Summary:")
    print(grouped.to_string(index=False))

    anova_results = perform_anova(grouped)

    print("\nANOVA Results:")
    for key, result in anova_results.items():
        print(f"{key}: F-statistic = {result.statistic:.6f}, P-value = {result.pvalue:.15f}")


if __name__ == "__main__":
    file_pattern = '../parameter_testing/simulated_annealing/simulated_annealing_epoch_data_chunk_*.csv'
    main(file_pattern)
