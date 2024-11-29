import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt


def perform_correlation_analysis(data):
    """
    Perform Pearson correlation analysis focusing on correlations with injury column

    Parameters:
        data (pd.DataFrame): Input dataframe with daily data

    Returns:
        pd.Series: Correlations with injury column
        pd.Series: P-values for correlations with injury
    """
    # Get correlations with injury column
    # Calculate p-values for injury correlations
    injury_correlations = []
    p_values = []
    for col in data.columns:
        correlation, p_value = pearsonr(data[col], data['injury'])
        injury_correlations.append(correlation)
        p_values.append(p_value)

    p_values_series = pd.Series(p_values, index=data.columns)
    injury_correlations_series = pd.Series(
        injury_correlations, index=data.columns)

    return injury_correlations_series, p_values_series


if __name__ == "__main__":
    data = pd.read_csv('clustering/filtered_data_without_outliers.csv')
    # data = pd.read_csv('data/day_approach_maskedID_timeseries.csv')

    # Group data by Athlete ID without aggregation
    athlete_groups = {athlete_id: group for athlete_id,
                      group in data.groupby('Athlete ID')}

    # Filter for athletes with at least one injury
    athlete_groups = {athlete_id: group for athlete_id, group in athlete_groups.items()
                      if group['injury'].sum() > 0}
    print(f"\nAnalyzing {len(athlete_groups)} athletes with injury history")

    athletes_with_small_correlation = []
    for athlete_id, group in athlete_groups.items():
        print(f"Correlation analysis for athlete {athlete_id}:")

        # Remove Date, and Athlete ID columns
        group = group.drop(columns=['Date', 'Athlete ID'])

        injury_correlations, p_values_series = perform_correlation_analysis(
            group)

        # Check if any correlation is less than 0.4
        injury_correlations = injury_correlations.iloc[:-1]
        if injury_correlations.abs().max() < 0.4:
            athletes_with_small_correlation.append(athlete_id)

    print(f"Athletes with small correlation: {
          len(athletes_with_small_correlation)}")

    # Remove athletes with small correlation
    top_performing_data = data[~data['Athlete ID'].isin(
        athletes_with_small_correlation)]
    
    # correlation on whole dataset
    injury_correlations, p_values_series = perform_correlation_analysis(
        top_performing_data)

    plt.figure(figsize=(12, 6))
    injury_correlations.sort_values().plot(kind='bar')
    plt.title('Feature Correlations with Injury (Whole Dataset)')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig('correlation_analysis/whole_dataset.png')
    plt.show()
    plt.close()

    # Group data by Athlete ID without aggregation
    athlete_groups = {athlete_id: group for athlete_id,
                      group in top_performing_data.groupby('Athlete ID')}

    for athlete_id, group in athlete_groups.items():
        print(f"Correlation analysis for athlete {athlete_id}:")

        group = group.drop(columns=['Date', 'Athlete ID'])

        injury_correlations, p_values_series = perform_correlation_analysis(
            group)

        # filter out columns with correlation less than 0.5
        injury_correlations = injury_correlations.drop('injury')
        high_corr_indices = injury_correlations[injury_correlations.abs() >= 0.5].index
        injury_correlations = injury_correlations[high_corr_indices]
        p_values_series = p_values_series[high_corr_indices]

        # Skip if there are no correlations
        if len(injury_correlations) == 0:
            continue

        plt.figure(figsize=(12, 6))
        injury_correlations.sort_values().plot(kind='bar')
        plt.title(f'Feature Correlations with Injury (Athlete {athlete_id})')
        plt.xlabel('Features')
        plt.ylabel('Correlation Coefficient')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'correlation_analysis/athlete_{athlete_id}.png')
        plt.close()

        # plot p-values
        plt.figure(figsize=(12, 6))
        p_values_series.sort_values().plot(kind='bar')
        plt.title(f'P-values for Feature Correlations with Injury (Athlete {athlete_id})')
        plt.xlabel('Features')
        plt.ylabel('P-value')
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        plt.savefig(f'correlation_analysis/athlete_{athlete_id}_pvalues.png')
        plt.close()
