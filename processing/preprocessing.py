import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, MinMaxScaler
from sklearn.impute import SimpleImputer

def preprocess_data(input_file, output_file):
    # Load the data
    data = pd.read_csv(input_file)

    # Separate features and target
    X = data.drop(columns=['injury'])
    y = data['injury']

    # Count the number of unique injured athletes
    injured_athletes = data['Athlete ID'].nunique()
    print(f"Number of unique injured athletes: {injured_athletes}")

    # Impute missing values with the mean
    imputer = SimpleImputer(strategy='mean')
    X_imputed = imputer.fit_transform(X)

    # Normalize the features to a range between 0 and 1
    scaler = MinMaxScaler()
    X_normalized = scaler.fit_transform(X_imputed)

    # Convert to DataFrame to maintain column names
    X_preprocessed = pd.DataFrame(X_normalized, columns=X.columns)

    # Save the preprocessed data to a new CSV file
    preprocessed_data = X_preprocessed.copy()
    preprocessed_data['injury'] = y
    preprocessed_data.to_csv(output_file, index=False)

    print(f"Preprocessed data saved to {output_file}")

# Example usage
if __name__ == "__main__":
    preprocess_data('data/week_approach_maskedID_timeseries.csv', 'processing/preprocessed_week.csv')
