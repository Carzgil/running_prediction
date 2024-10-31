import os
import pandas as pd
from sklearn.linear_model import LassoCV
from sklearn.preprocessing import StandardScaler

# Load the data
data = pd.read_csv('data/day_approach_maskedID_timeseries.csv')

# Separate features and target
X = data.drop(columns=['injury'])
y = data['injury']

# Binary transformation for features with many zeros
X_binary = X.applymap(lambda x: 1 if x != 0 else 0)

# Aggregate features: count of non-zero entries
X['non_zero_count'] = X_binary.sum(axis=1)

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Fit Lasso model with increased iterations
lasso = LassoCV(cv=5, random_state=42, max_iter=10000)
lasso.fit(X_scaled, y)

# Get the coefficients
lasso_coefficients = pd.Series(lasso.coef_, index=X.columns)

# Select features with non-zero coefficients
selected_features = lasso_coefficients[lasso_coefficients != 0].index.tolist()

# Save the selected features to a CSV file in the 'analysis' folder
os.makedirs('analysis/daily', exist_ok=True)
selected_features_df = pd.DataFrame(selected_features, columns=['Selected Features'])
selected_features_df.to_csv('analysis/daily/selected_features_lasso.csv', index=False)

print("Selected features using Lasso saved to 'analysis/daily/selected_features_lasso.csv'")



