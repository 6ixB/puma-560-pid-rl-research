import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ReplayBuffer:
    def __init__(self, state_shape, action_dim, max_size=1000000, memmap_dir=None):
        self.max_size = max_size
        self.ptr = 0
        self.size = 0
        
        if memmap_dir is not None:
            import os
            os.makedirs(memmap_dir, exist_ok=True)
            self.state = np.memmap(os.path.join(memmap_dir, 'state.dat'), dtype=np.float32, mode='w+', shape=(max_size, *state_shape))
            self.action = np.memmap(os.path.join(memmap_dir, 'action.dat'), dtype=np.float32, mode='w+', shape=(max_size, action_dim))
            self.next_state = np.memmap(os.path.join(memmap_dir, 'next_state.dat'), dtype=np.float32, mode='w+', shape=(max_size, *state_shape))
            self.reward = np.memmap(os.path.join(memmap_dir, 'reward.dat'), dtype=np.float32, mode='w+', shape=(max_size, 1))
            self.not_done = np.memmap(os.path.join(memmap_dir, 'not_done.dat'), dtype=np.float32, mode='w+', shape=(max_size, 1))
        else:
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

        # Pre-allocate batch arrays to prevent memory allocation and fragmentation leaks
        if not hasattr(self, '_batch_state') or self._batch_state.shape[0] != batch_size:
            self._batch_state = np.empty((batch_size, *self.state.shape[1:]), dtype=np.float32)
            self._batch_action = np.empty((batch_size, self.action.shape[1]), dtype=np.float32)
            self._batch_next_state = np.empty((batch_size, *self.next_state.shape[1:]), dtype=np.float32)
            self._batch_reward = np.empty((batch_size, 1), dtype=np.float32)
            self._batch_not_done = np.empty((batch_size, 1), dtype=np.float32)
            
            # Pre-allocate numpy-to-torch wrappers (shares memory, zero allocation after first time)
            self._np_state_wrapper = torch.from_numpy(self._batch_state)
            self._np_action_wrapper = torch.from_numpy(self._batch_action)
            self._np_next_state_wrapper = torch.from_numpy(self._batch_next_state)
            self._np_reward_wrapper = torch.from_numpy(self._batch_reward)
            self._np_not_done_wrapper = torch.from_numpy(self._batch_not_done)
            
            # Pre-allocate PyTorch device tensors
            self._batch_state_t = torch.empty((batch_size, *self.state.shape[1:]), device=device, dtype=torch.float32)
            self._batch_action_t = torch.empty((batch_size, self.action.shape[1]), device=device, dtype=torch.float32)
            self._batch_next_state_t = torch.empty((batch_size, *self.next_state.shape[1:]), device=device, dtype=torch.float32)
            self._batch_reward_t = torch.empty((batch_size, 1), device=device, dtype=torch.float32)
            self._batch_not_done_t = torch.empty((batch_size, 1), device=device, dtype=torch.float32)

        # Copy data directly into the pre-allocated buffers
        np.take(self.state, ind, axis=0, out=self._batch_state)
        np.take(self.action, ind, axis=0, out=self._batch_action)
        np.take(self.next_state, ind, axis=0, out=self._batch_next_state)
        np.take(self.reward, ind, axis=0, out=self._batch_reward)
        np.take(self.not_done, ind, axis=0, out=self._batch_not_done)

        # Copy to pre-allocated PyTorch device tensors in-place using cached wrappers (zero allocations!)
        self._batch_state_t.copy_(self._np_state_wrapper)
        self._batch_action_t.copy_(self._np_action_wrapper)
        self._batch_next_state_t.copy_(self._np_next_state_wrapper)
        self._batch_reward_t.copy_(self._np_reward_wrapper)
        self._batch_not_done_t.copy_(self._np_not_done_wrapper)

        return (
            self._batch_state_t,
            self._batch_action_t,
            self._batch_next_state_t,
            self._batch_reward_t,
            self._batch_not_done_t
        )

class LSTMFeatureExtractor(nn.Module):
    def __init__(self, input_dim, hidden_dim=256, num_layers=3):
        super(LSTMFeatureExtractor, self).__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, batch_first=True)
        
    def forward(self, x):
        # Force weights to be contiguous to prevent cuDNN memory reallocation/fragmentation
        self.lstm.flatten_parameters()
        # x is (batch, seq_len, input_dim)
        out, (h_n, c_n) = self.lstm(x)
        # return the last time step's output feature
        return out[:, -1, :]

class TD3Actor(nn.Module):
    def __init__(self, state_dim=42, action_dim=6, max_action=None, hidden_dim=256):
        super(TD3Actor, self).__init__()
        
        # Extract temporal features from the last 20 timesteps (window_size) of state history
        self.lstm = LSTMFeatureExtractor(state_dim, hidden_dim, num_layers=2)
        
        # Context dimension is now embedded directly in the 44-dim state vector
        context_dim = 0
        
        # Layer 1: Combines LSTM temporal features + instantaneous current state + context vector
        self.l1 = nn.Linear(hidden_dim + state_dim + context_dim, 256)
        # LayerNorm 1: Stabilizes the activations to prevent exploding gradients
        self.ln1 = nn.LayerNorm(256)
        
        # Layer 2: Hidden processing layer
        self.l2 = nn.Linear(256, 128)
        # LayerNorm 2: Stabilizes the activations before the final output layer
        self.ln2 = nn.LayerNorm(128)
        
        # Layer 3: Output layer generating raw action values (bounded by Tanh later)
        self.l3 = nn.Linear(128, action_dim)
        
        if max_action is None:
            max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
        self.register_buffer('max_action', torch.FloatTensor(max_action))
        
    def forward(self, state_history):
        lstm_features = self.lstm(state_history)
        current_state = state_history[:, -1, :]
        
        x = torch.cat([lstm_features, current_state], dim=1)
        x = F.relu(self.ln1(self.l1(x)))
        x = F.relu(self.ln2(self.l2(x)))
        # Tanh bounds output to [-1, 1], we multiply by max_action limits
        x = torch.tanh(self.l3(x)) 
        return x * self.max_action

class TD3Critic(nn.Module):
    def __init__(self, state_dim=42, action_dim=6, hidden_dim=256):
        super(TD3Critic, self).__init__()
        
        # Shared LSTM for Twin Q-networks (Optional: can use separate LSTMs, but shared is faster and often stable if tuned well. We'll use separate for pure TD3)
        self.lstm1 = LSTMFeatureExtractor(state_dim, hidden_dim, num_layers=2)
        self.lstm2 = LSTMFeatureExtractor(state_dim, hidden_dim, num_layers=2)
        
        # Context dimension is now embedded directly in the 44-dim state vector
        context_dim = 0
        
        # Max action for normalization
        max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
        self.register_buffer('max_action', torch.FloatTensor(max_action))
        
        # --- Q1 Architecture ---
        # Layer 1: Combines LSTM temporal features + instantaneous current state + context vector + ACTION
        self.l1 = nn.Linear(hidden_dim + state_dim + context_dim + action_dim, 256)
        self.ln1 = nn.LayerNorm(256)
        
        # Layer 2: Hidden processing layer
        self.l2 = nn.Linear(256, 128)
        self.ln2 = nn.LayerNorm(128)
        
        # Layer 3: Output layer generating the Q1-value (Expected future reward)
        self.l3 = nn.Linear(128, 1)

        # --- Q2 Architecture (Twin Critic) ---
        # Identical to Q1, used to prevent overestimation bias in continuous control
        self.l4 = nn.Linear(hidden_dim + state_dim + context_dim + action_dim, 256)
        self.ln4 = nn.LayerNorm(256)
        self.l5 = nn.Linear(256, 128)
        self.ln5 = nn.LayerNorm(128)
        self.l6 = nn.Linear(128, 1)

    def forward(self, state_history, action):
        lstm_feat1 = self.lstm1(state_history)
        lstm_feat2 = self.lstm2(state_history)
        current_state = state_history[:, -1, :]
        
        norm_action = action / self.max_action
        
        sa1 = torch.cat([lstm_feat1, current_state, norm_action], dim=1)
        sa2 = torch.cat([lstm_feat2, current_state, norm_action], dim=1)

        q1 = F.relu(self.ln1(self.l1(sa1)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = self.l3(q1)

        q2 = F.relu(self.ln4(self.l4(sa2)))
        q2 = F.relu(self.ln5(self.l5(q2)))
        q2 = self.l6(q2)
        return q1, q2

    def Q1(self, state_history, action):
        lstm_feat1 = self.lstm1(state_history)
        current_state = state_history[:, -1, :]
        
        norm_action = action / self.max_action
        
        sa1 = torch.cat([lstm_feat1, current_state, norm_action], dim=1)

        q1 = F.relu(self.ln1(self.l1(sa1)))
        q1 = F.relu(self.ln2(self.l2(q1)))
        q1 = self.l3(q1)
        return q1
