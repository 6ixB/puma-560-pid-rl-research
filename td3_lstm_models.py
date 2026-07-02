import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    """Standard TD3 Replay Buffer storing augmented observation s_tilde"""
    def __init__(self, state_dim, action_dim, max_size=1000000):
        self.ptr, self.size, self.max_size = 0, 0, max_size
        self.state = np.zeros((max_size, state_dim), dtype=np.float32)
        self.action = np.zeros((max_size, action_dim), dtype=np.float32)
        self.next_state = np.zeros((max_size, state_dim), dtype=np.float32)
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

# ============================================================================
# Algorithm 2 & 5: Temporal Observer Components
# ============================================================================
class LSTMTemporalObserver(nn.Module):
    def __init__(self, input_dim=40, hidden_dim=256):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.hidden_dim = hidden_dim
        
    def forward(self, x, h=None, c=None):
        """Line 7: LSTMFORWARD(X_t, h, c) -> h_t, c_t"""
        self.lstm.flatten_parameters()
        x_tensor = torch.FloatTensor(x).unsqueeze(0).to(device) # Batch size 1
        
        if h is not None and c is not None:
            h_tns = torch.FloatTensor(h).unsqueeze(0).unsqueeze(0).to(device)
            c_tns = torch.FloatTensor(c).unsqueeze(0).unsqueeze(0).to(device)
            _, (h_next, c_next) = self.lstm(x_tensor, (h_tns, c_tns))
        else:
            _, (h_next, c_next) = self.lstm(x_tensor)
            
        # Return exact NumPy arrays for Algorithm 5 state concatenation
        return h_next.squeeze().detach().cpu().numpy(), c_next.squeeze().detach().cpu().numpy()

# ============================================================================
# Algorithm 3: TD3 Actor and Critic
# ============================================================================
class TD3Actor(nn.Module):
    def __init__(self, state_dim=302, action_dim=6):
        super().__init__()
        # Layer Normalization added to all hidden layers
        self.l1 = nn.Linear(state_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.l3 = nn.Linear(128, action_dim)
        
    def forward(self, state):
        x = F.relu(self.ln1(self.l1(state)))
        x = F.relu(self.ln2(self.l2(x)))
        # Action Normalization: Actor strictly outputs in [-1, 1]. No scaling here.
        return torch.tanh(self.l3(x))

class TD3Critic(nn.Module):
    def __init__(self, state_dim=302, action_dim=6):
        super().__init__()
        # Q1 Architecture with LayerNorm
        self.l1 = nn.Linear(state_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        self.l2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        self.l3 = nn.Linear(128, 1)

        # Q2 Architecture with LayerNorm
        self.l4 = nn.Linear(state_dim + action_dim, 256)
        self.ln4 = nn.LayerNorm(256)
        self.l5 = nn.Linear(256, 128)
        self.ln5 = nn.LayerNorm(128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, state, action):
        # Action Normalization: Critic expects action to already be in [-1, 1]. No division here.
        sa = torch.cat([state, action], dim=1)

        q1 = F.relu(self.ln1(self.l1(sa)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = self.l3(q1)

        q2 = F.relu(self.ln4(self.l4(sa)))
        q2 = F.relu(self.ln5(self.l5(q2)))
        q2 = self.l6(q2)
        return q1, q2
        
    def Q1(self, state, action):
        sa = torch.cat([state, action], dim=1)
        q1 = F.relu(self.ln1(self.l1(sa)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = self.l3(q1)
        return q1