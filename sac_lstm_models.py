import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_shape, action_dim, max_size=100000):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        self.state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, *state_shape), dtype=np.float32)
        self.reward = np.zeros((max_size, 1), dtype=np.float32)
        self.not_done = np.zeros((max_size, 1), dtype=np.float32)

    def add(self, state, action, next_state, reward, done):
        self.state[self.ptr] = state
        self.action[self.ptr] = action
        self.next_state[self.ptr] = next_state
        self.reward[self.ptr] = reward
        self.not_done[self.ptr] = 1. - done

        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        ind = np.random.randint(0, self.size, size=batch_size)

        return (
            torch.FloatTensor(self.state[ind]).to(device),
            torch.FloatTensor(self.action[ind]).to(device),
            torch.FloatTensor(self.next_state[ind]).to(device),
            torch.FloatTensor(self.reward[ind]).to(device),
            torch.FloatTensor(self.not_done[ind]).to(device)
        )

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        out, (h_n, c_n) = self.lstm(x)
        return out[:, -1, :]

class SACActor(nn.Module):
    def __init__(self, state_dim=42, action_dim=6, max_action=None, hidden_dim=256):
        super(SACActor, self).__init__()
        
        self.lstm = LSTMFeatureExtractor(state_dim, hidden_dim, num_layers=2)
        self.l1 = nn.Linear(hidden_dim + state_dim, 256)
        self.l2 = nn.Linear(256, 128)
        
        self.mu = nn.Linear(128, action_dim)
        self.log_std = nn.Linear(128, action_dim)
        
        if max_action is None:
            max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
        self.max_action = torch.FloatTensor(max_action).to(device)
        
        self.log_std_min = -20
        self.log_std_max = 2

    def forward(self, state_history):
        lstm_features = self.lstm(state_history)
        current_state = state_history[:, -1, :]
        
        x = torch.cat([lstm_features, current_state], dim=1)
        x = F.relu(self.l1(x))
        x = F.relu(self.l2(x))
        
        mu = self.mu(x)
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mu, log_std

    def sample(self, state_history):
        mu, log_std = self.forward(state_history)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mu, std)
        x_t = normal.rsample()  # for reparameterization trick
        y_t = torch.tanh(x_t)
        action = y_t * self.max_action
        
        # Enforcing Action Bound
        log_prob = normal.log_prob(x_t)
        log_prob -= torch.log(self.max_action * (1 - y_t.pow(2)) + 1e-6)
        log_prob = log_prob.sum(1, keepdim=True)
        
        return action, log_prob

class SACCritic(nn.Module):
    def __init__(self, state_dim=42, action_dim=6, hidden_dim=256):
        super(SACCritic, self).__init__()
        
        self.lstm1 = LSTMFeatureExtractor(state_dim, hidden_dim, num_layers=2)
        self.lstm2 = LSTMFeatureExtractor(state_dim, hidden_dim, num_layers=2)
        
        self.l1 = nn.Linear(hidden_dim + state_dim + action_dim, 256)
        self.l2 = nn.Linear(256, 128)
        self.l3 = nn.Linear(128, 1)

        self.l4 = nn.Linear(hidden_dim + state_dim + action_dim, 256)
        self.l5 = nn.Linear(256, 128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, state_history, action):
        lstm_feat1 = self.lstm1(state_history)
        lstm_feat2 = self.lstm2(state_history)
        current_state = state_history[:, -1, :]
        
        sa1 = torch.cat([lstm_feat1, current_state, action], dim=1)
        sa2 = torch.cat([lstm_feat2, current_state, action], dim=1)

        q1 = F.relu(self.l1(sa1))
        q1 = F.relu(self.l2(q1))
        q1 = self.l3(q1)

        q2 = F.relu(self.l4(sa2))
        q2 = F.relu(self.l5(q2))
        q2 = self.l6(q2)
        return q1, q2
