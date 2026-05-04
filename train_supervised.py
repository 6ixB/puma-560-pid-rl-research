import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import TensorDataset, DataLoader
import pandas as pd
import numpy as np

from lstm_pid_simulation import LSTMPIDTuner

def load_and_prep_data(csv_file='offline_pid_dataset.csv', window_size=10):
    df = pd.read_csv(csv_file)
    print(f"Loaded dataset with {len(df)} steps.")
    
    # 24 features: 6 angles, 6 velocities, 6 errors, 6 integrals
    qs = df[[f'q_{i}' for i in range(6)]].values
    qds = df[[f'qd_{i}' for i in range(6)]].values
    errs = df[[f'err_{i}' for i in range(6)]].values
    integrals = df[[f'integral_{i}' for i in range(6)]].values
    
    features = np.concatenate([qs, qds, errs, integrals], axis=1).astype(np.float32)
    
    # 18 targets: delta_Kp, delta_Ki, delta_Kd
    targets = np.zeros((len(df), 18), dtype=np.float32)
    for i in range(6):
        targets[:, i*3] = df[f'delta_Kp_{i}'].values
        targets[:, i*3+1] = df[f'delta_Ki_{i}'].values
        targets[:, i*3+2] = df[f'delta_Kd_{i}'].values
        
    # Scale targets back to [-1, 1] range based on max bounds
    # Max: Kp=50, Ki=5, Kd=10
    scales = np.array([50.0, 5.0, 10.0] * 6, dtype=np.float32)
    targets_scaled = targets / scales
    # Clamp to avoid going out of Tanh limits due to outliers
    targets_scaled = np.clip(targets_scaled, -1.0, 1.0)
    
    # Create sliding windows
    X = []
    y = []
    
    for i in range(len(features) - window_size):
        X.append(features[i:i+window_size, :])
        y.append(targets_scaled[i+window_size-1, :]) # Predict ideal target for the LAST step in the window
        
    X = torch.tensor(np.array(X), dtype=torch.float32)
    y = torch.tensor(np.array(y), dtype=torch.float32)
    
    return X, y

def train_supervised(epochs=100, batch_size=32, lr=1e-3, load_existing=False):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Training on device: {device}")
    
    window_size = 10
    X, y = load_and_prep_data(window_size=window_size)
    
    dataset = TensorDataset(X, y)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    
    model = LSTMPIDTuner(
        input_size=24, 
        hidden_size=64, 
        num_layers=2, 
        output_size=18, 
        window_size=window_size
    ).to(device)
    
    if load_existing:
        try:
            model.load_state_dict(torch.load('lstm_supervised_weights.pth'))
            print("Loaded existing weights.")
        except Exception as e:
            print("Could not load existing weights, starting fresh.")
            
    # We use MSELoss against the unscaled Tanh outputs
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr)
    
    print("Starting Supervised Training Loop...")
    for epoch in range(epochs):
        model.train()
        epoch_loss = 0.0
        
        for batch_X, batch_y in dataloader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()
            
            # forward() returns unscaled raw outputs in range [-1, 1] through Tanh
            predictions = model(batch_X)
            
            loss = criterion(predictions, batch_y)
            loss.backward()
            
            optimizer.step()
            epoch_loss += loss.item()
            
        avg_loss = epoch_loss / len(dataloader)
        if (epoch + 1) % 10 == 0 or epoch == 0:
            print(f"Epoch [{epoch+1}/{epochs}], MSE Loss: {avg_loss:.6f}")
            
    torch.save(model.state_dict(), 'lstm_supervised_weights.pth')
    print("Training finished. Weights saved to 'lstm_supervised_weights.pth'")

if __name__ == "__main__":
    train_supervised(epochs=100)
