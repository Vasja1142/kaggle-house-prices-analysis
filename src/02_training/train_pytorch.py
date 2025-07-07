import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error

# 1. Load and preprocess data
def load_data(train_path, test_path):
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    X = train_df.drop('SalePrice', axis=1)
    y = train_df['SalePrice']

    # Align columns - crucial for consistent feature sets
    train_cols = X.columns
    test_cols = test_df.columns
    
    missing_in_test = set(train_cols) - set(test_cols)
    for c in missing_in_test:
        test_df[c] = 0 # Or appropriate default/imputation
    
    missing_in_train = set(test_cols) - set(train_cols)
    for c in missing_in_train:
        X[c] = 0 # Or appropriate default/imputation

    test_df = test_df[train_cols] # Ensure order and presence

    # Handle potential NaN values by filling with 0 (simple approach, consider more sophisticated)
    X = X.fillna(0)
    test_df = test_df.fillna(0)

    # Scale numerical features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    test_scaled = scaler.transform(test_df)

    return X_scaled, y.values, test_scaled

# 2. Define the PyTorch Model
class TwoLayerNet(nn.Module):
    def __init__(self, input_size, hidden_size, output_size=1, dropout_rate=0.5):
        super(TwoLayerNet, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(dropout_rate)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.dropout(x)
        x = self.fc2(x)
        return x

# 3. Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs=100):
    for epoch in range(num_epochs):
        model.train()
        for inputs, targets in train_loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets.view(-1, 1))
            loss.backward()
            optimizer.step()

        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for inputs, targets in val_loader:
                outputs = model(inputs)
                loss = criterion(outputs, targets.view(-1, 1))
                val_loss += loss.item()
        
        val_loss /= len(val_loader)
        # print(f'Epoch {epoch+1}/{num_epochs}, Validation Loss: {val_loss:.4f}')
    return val_loss

# Main execution
if __name__ == "__main__":
    train_path = '/home/vasja1142/projects/kaggle-house-prices-analysis/data/processed/train_processed.csv'
    test_path = '/home/vasja1142/projects/kaggle-house-prices-analysis/data/processed/test_processed.csv'

    X_scaled, y_values, test_scaled = load_data(train_path, test_path)

    # Convert to PyTorch tensors
    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)
    y_tensor = torch.tensor(y_values, dtype=torch.float32)
    test_tensor = torch.tensor(test_scaled, dtype=torch.float32)

    # Split data for training and validation
    X_train, X_val, y_train, y_val = train_test_split(X_tensor, y_tensor, test_size=0.2, random_state=42)

    # Create DataLoader
    train_dataset = TensorDataset(X_train, y_train)
    val_dataset = TensorDataset(X_val, y_val)

    train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=64, shuffle=False)

    input_size = X_train.shape[1]
    hidden_size = 128 # Example hidden size
    learning_rate = 0.001 # Example learning rate
    num_epochs = 100

    model = TwoLayerNet(input_size, hidden_size)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    print("Starting training...")
    final_val_loss = train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs)
    print(f"Training finished. Final Validation Loss: {final_val_loss:.4f}")

    # Make predictions (example)
    model.eval()
    with torch.no_grad():
        predictions = model(test_tensor).numpy().flatten()
    
    # print("Sample predictions:", predictions[:5])