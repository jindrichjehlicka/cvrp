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
        key = (row['population_size'], row['generations'], row['mutation_rate'])
        final_score = [epoch[-1] for epoch in row['epoch_data']]
        for score in final_score:
            final_scores.append({
                'population_size': row['population_size'],
                'generations': row['generations'],
                'mutation_rate': row['mutation_rate'],
                'final_score': score
            })
    return pd.DataFrame(final_scores)


# Group by parameter and calculate summary statistics
def group_by_parameter(final_scores):
    grouped = final_scores.groupby(['population_size', 'generations', 'mutation_rate']).agg(
        mean_final_score=('final_score', 'mean'),
        std_final_score=('final_score', 'std')
    ).reset_index()
    return grouped


# Perform ANOVA
def perform_anova(grouped):
    anova_results = {}
    # Group by population_size
    for size in grouped['population_size'].unique():
        groups = [group['mean_final_score'].values for name, group in
                  grouped[grouped['population_size'] == size].groupby('generations')]
        anova_results[f'population_size_{size}'] = f_oneway(*groups)

    # Group by generations
    for gen in grouped['generations'].unique():
        groups = [group['mean_final_score'].values for name, group in
                  grouped[grouped['generations'] == gen].groupby('mutation_rate')]
        anova_results[f'generations_{gen}'] = f_oneway(*groups)

    # Group by mutation_rate
    for rate in grouped['mutation_rate'].unique():
        groups = [group['mean_final_score'].values for name, group in
                  grouped[grouped['mutation_rate'] == rate].groupby('population_size')]
        anova_results[f'mutation_rate_{rate}'] = f_oneway(*groups)

    return anova_results


# Main function
def main(file_pattern):
    df = load_data(file_pattern)
    final_scores = extract_final_scores(df)
    grouped = group_by_parameter(final_scores)
    anova_results = perform_anova(grouped)

    # Formatting results for diploma thesis
    print("Grouped Summary:")
    print(grouped.to_string(index=False))

    print("\nANOVA Results:")
    for key, result in anova_results.items():
        print(f"{key}: F-statistic = {result.statistic:.3f}, P-value = {result.pvalue:.3f}")


if __name__ == "__main__":
    file_pattern = '../parameter_testing/genetic/genetic_algorithm_epoch_data_chunk_*.csv'
    main(file_pattern)
