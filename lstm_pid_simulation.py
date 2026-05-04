import torch
import torch.nn as nn
import torch.nn.functional as F

class LSTMPIDTuner(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers, output_size, window_size=10):
        super(LSTMPIDTuner, self).__init__()
        
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.window_size = window_size
        
        # 1. DCNN Component: 1D Convolution over time series sliding window
        # Input shape expected by Conv1d: (batch_size, in_channels, sequence_length)
        self.conv1 = nn.Conv1d(in_channels=input_size, out_channels=32, kernel_size=3, padding=1)
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=3, padding=1)
        
        # 2. LSTM Component
        # We will transpose back to (batch_size, sequence_length, 64) for LSTM
        self.lstm = nn.LSTM(input_size=64, hidden_size=hidden_size, num_layers=num_layers, batch_first=True)
        
        # 3. Actor Head (Outputs PID deltas)
        self.actor_fc1 = nn.Linear(hidden_size, 128)
        self.actor_fc2 = nn.Linear(128, output_size)
        self.tanh = nn.Tanh()
        
        # 4. Critic Head (Outputs State Value for RL)
        self.critic_fc1 = nn.Linear(hidden_size, 128)
        self.critic_fc2 = nn.Linear(128, 1)

    def extract_features(self, x):
        # x shape: (batch_size, sequence_length, input_features)
        x_conv = x.transpose(1, 2) # (batch_size, channels, length)
        
        x_conv = F.relu(self.conv1(x_conv))
        x_conv = F.relu(self.conv2(x_conv))
        
        x_lstm_in = x_conv.transpose(1, 2) # (batch_size, length, channels)
        
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x_lstm_in, (h0, c0))
        return out[:, -1, :] # Extract the last time step

    def forward(self, x):
        """Actor forward pass: Returns unscaled deltas in [-1, 1]"""
        last_out = self.extract_features(x)
        
        actor_out = F.relu(self.actor_fc1(last_out))
        actor_out = self.actor_fc2(actor_out)
        return self.tanh(actor_out)

    def get_deltas(self, x, max_kp=50.0, max_ki=5.0, max_kd=10.0):
        """Returns scaled PID deltas based on defined maximum adjustments."""
        unscaled_deltas = self.forward(x) # Shape: (batch, 18)
        
        device = unscaled_deltas.device
        scales = torch.tensor([max_kp, max_ki, max_kd] * 6, device=device)
        
        return unscaled_deltas * scales

    def get_action_and_value(self, x):
        """Returns both Actor action and Critic state value. Used during training."""
        last_out = self.extract_features(x)
        
        # Actor
        actor_out = F.relu(self.actor_fc1(last_out))
        actor_out = self.actor_fc2(actor_out)
        action = self.tanh(actor_out)
        
        # Critic
        critic_out = F.relu(self.critic_fc1(last_out))
        value = self.critic_fc2(critic_out)
        
        return action, value

# --- Example Usage ---
if __name__ == "__main__":
    features_per_step = 24
    window_size = 10

    model = LSTMPIDTuner(input_size=features_per_step, hidden_size=64, num_layers=2, output_size=18, window_size=window_size)

    dummy_history = torch.randn(1, window_size, features_per_step)

    predicted_deltas = model(dummy_history)
    scaled_deltas = model.get_deltas(dummy_history)
    action, value = model.get_action_and_value(dummy_history)
    print("Predicted Delta Gains shape:", predicted_deltas.shape)
    print("Scaled Deltas example:", scaled_deltas[0, :3].detach().numpy())
    print("Critic Value shape:", value.shape)