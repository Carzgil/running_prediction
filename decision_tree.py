import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier, plot_tree
import matplotlib.pyplot as plt
import os

# Load the preprocessed data
data = pd.read_csv('processing/preprocessed_day.csv')

# Load the selected features
selected_features_df = pd.read_csv('analysis/daily/selected_features_nn.csv')
selected_features = selected_features_df['Feature'].tolist()

# Ensure the selected features are in the dataset
X = data[selected_features]
y = data['injury']

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the Decision Tree Classifier
decision_tree = DecisionTreeClassifier(random_state=42)

# Train the model
decision_tree.fit(X_train, y_train)

# Visualize the decision tree
plt.figure(figsize=(20, 10))
plot_tree(decision_tree, feature_names=selected_features, class_names=['No Injury', 'Injury'], filled=True, rounded=True)
plt.title("Decision Tree for Injury Prediction (Selected Features)")

# Save the plot as a PNG file
os.makedirs('analysis', exist_ok=True)
plt.savefig('analysis/decision_tree_injury_day.png')
plt.close()

print("Decision tree visualization saved to 'analysis/decision_tree_injury_day.png'")
