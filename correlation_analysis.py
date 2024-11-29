import pandas as pd
import numpy as np
from scipy.stats import pearsonr
import matplotlib.pyplot as plt
import seaborn as sns


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
    injury_correlations = data.corr(method='pearson')['injury']

    # Calculate p-values for injury correlations
    p_values = []
    for col in data.columns:
        _, p_value = pearsonr(data[col], data['injury'])
        p_values.append(p_value)

    p_values_series = pd.Series(p_values, index=data.columns)

    # Plot correlation bar chart
    plt.figure(figsize=(12, 6))
    injury_correlations.drop('injury').sort_values().plot(kind='bar')
    plt.title('Feature Correlations with Injury')
    plt.xlabel('Features')
    plt.ylabel('Correlation Coefficient')
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()

    return injury_correlations, p_values_series


if __name__ == "__main__":
    data = pd.read_csv('data/day_approach_maskedID_timeseries.csv')

    # Filter columns from previous days
    keep_cols = data.columns[data.columns.str.endswith('.6') |
                             data.columns.isin(['Athlete ID', 'injury', 'Date'])]
    data = data.loc[:, keep_cols]

    # Group data by Athlete ID without aggregation
    athlete_groups = {athlete_id: group for athlete_id,
                      group in data.groupby('Athlete ID')}

    # Filter for athletes with at least one injury
    athlete_groups = {athlete_id: group for athlete_id, group in athlete_groups.items()
                      if group['injury'].sum() > 0}
    print(f"\nAnalyzing {len(athlete_groups)} athletes with injury history")

    for athlete_id, group in athlete_groups.items():
        print(f"Correlation analysis for athlete {athlete_id}:")
        injury_correlations, p_values_series = perform_correlation_analysis(
            group)
        print(injury_correlations)
        print(p_values_series)
