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
        key = (row['max_iterations'], row['tabu_size'], row['neighborhood_size'])
        final_score = row['epoch_data'][-1]
        final_scores.append({
            'max_iterations': row['max_iterations'],
            'tabu_size': row['tabu_size'],
            'neighborhood_size': row['neighborhood_size'],
            'final_score': final_score
        })
    return pd.DataFrame(final_scores)


# Group by parameter and calculate summary statistics
def group_by_parameter(final_scores):
    grouped = final_scores.groupby(['max_iterations', 'tabu_size', 'neighborhood_size']).agg(
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
                  grouped[grouped['max_iterations'] == max_iter].groupby('tabu_size')]
        if len(groups) > 1:
            anova_results[f'max_iterations_{max_iter}'] = f_oneway(*groups)

    # Group by tabu_size
    for tabu in grouped['tabu_size'].unique():
        groups = [group['mean_final_score'].values for name, group in
                  grouped[grouped['tabu_size'] == tabu].groupby('neighborhood_size')]
        if len(groups) > 1:
            anova_results[f'tabu_size_{tabu}'] = f_oneway(*groups)

    # Group by neighborhood_size
    for neighborhood in grouped['neighborhood_size'].unique():
        groups = [group['mean_final_score'].values for name, group in
                  grouped[grouped['neighborhood_size'] == neighborhood].groupby('max_iterations')]
        if len(groups) > 1:
            anova_results[f'neighborhood_size_{neighborhood}'] = f_oneway(*groups)

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
        print(f"{key}: F-statistic = {result.statistic:.6f}, P-value = {result.pvalue:.15}")


if __name__ == "__main__":
    file_pattern = '../parameter_testing/tabu_search/tabu_search_epoch_data_chunk_*.csv'
    main(file_pattern)