import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import os

# Load the data
data = pd.read_csv('data/day_approach_maskedID_timeseries.csv')

# Separate features and target
X = data.drop(columns=['injury'])
y = data['injury']

# Standardize the features
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
y_tensor = torch.tensor(y.values, dtype=torch.float32)

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

# Create DataLoader
train_dataset = TensorDataset(X_train, y_train)
train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)

# Define a simple neural network model
class SimpleNN(nn.Module):
    def __init__(self, input_dim):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc3 = nn.Linear(32, 1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.sigmoid(self.fc3(x))
        return x

# Initialize the model, loss function, and optimizer
model = SimpleNN(X_train.shape[1])
criterion = nn.BCELoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# Move model to GPU if available
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
model.to(device)

# Training loop
for epoch in range(10):  # Number of epochs
    model.train()
    for inputs, targets in train_loader:
        inputs, targets = inputs.to(device), targets.to(device)
        
        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs.squeeze(), targets)
        
        # Backward pass and optimization
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    
    print(f'Epoch {epoch+1}, Loss: {loss.item()}')

# Evaluate feature importance (simple approach)
feature_importance = model.fc1.weight.abs().mean(dim=0).cpu().detach().numpy()
important_features = np.argsort(feature_importance)[-10:]  # Top 10 features

# Map indices to feature names
selected_feature_names = [X.columns[i] for i in important_features]

# Save the selected features to a CSV file in the 'analysis' folder
os.makedirs('analysis/weekly', exist_ok=True)
selected_features_df = pd.DataFrame(selected_feature_names, columns=['Feature'])
selected_features_df.to_csv('analysis/weekly/selected_features_nn.csv', index=False)

# Save the model's state dictionary
model_path = 'models/daily/simple_nn_model.pth'
torch.save(model.state_dict(), model_path)

print(f"Model saved to {model_path}")
