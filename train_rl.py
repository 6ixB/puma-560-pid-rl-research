import torch
import torch.optim as optim
from torch.distributions import Normal
import numpy as np

from rl_env import Puma560Env
from lstm_pid_simulation import LSTMPIDTuner

def train():
    print("Initializing RL Tuning Pipeline...")
    window_size = 10
    
    # 1. Initialize Custom Gym Wrapper
    env = Puma560Env(dt=0.01, window_size=window_size, max_steps=400)
    
    # 2. Setup DCNN+LSTM Custom Network
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = LSTMPIDTuner(input_size=24, hidden_size=64, num_layers=2, output_size=18, window_size=window_size).to(device)
    optimizer = optim.Adam(model.parameters(), lr=5e-4) # Learning rate
    
    # RL Hyperparameters
    action_std = 0.5            # Determines exploration variance
    gamma = 0.99                # Discount factor for future rewards
    epochs = 50                 # Number of simulated trajectories to run
    
    best_error = float('inf')
    
    for epoch in range(epochs):
        state = env.reset()
        
        log_probs = []
        values = []
        rewards = []
        
        done = False
        while not done:
            # Format state from numpy -> torch. FloatTensor with shape (batch_size, sequence, features)
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            
            # Forward pass to get Action (18 PID Parameters) and Critic's Value estimate
            action_mean, value = model.get_action_and_value(state_tensor)
            
            # Noise-based exploration mechanism (Allows network to try new PID values)
            dist = Normal(action_mean, action_std)
            action = dist.sample()
            
            log_prob = dist.log_prob(action).sum(dim=-1)
            
            # Strip computation graph and enforce non-negative gains for physics simulator array
            action_np = torch.clamp(action, min=0.0).detach().cpu().numpy()[0]
            
            # Step the Puma 560 Simulation
            next_state, reward, done, info = env.step(action_np)
            
            log_probs.append(log_prob)
            values.append(value)
            rewards.append(reward)
            
            state = next_state
            
        # End of Trajectory: Value/Advantage computation (A2C Logic)
        returns = []
        R = 0
        for r in reversed(rewards):
            R = r + gamma * R
            returns.insert(0, R)
            
        returns = torch.tensor(returns).to(device)
        returns = (returns - returns.mean()) / (returns.std() + 1e-8)
        
        log_probs = torch.cat(log_probs)
        values = torch.cat(values).squeeze()
        
        # The Advantage estimates how much better the action performed than average
        advantage = returns - values.detach()
        
        actor_loss = -(log_probs * advantage).mean()
        critic_loss = (returns - values).pow(2).mean()
        loss = actor_loss + 0.5 * critic_loss
        
        # Optimize global policy
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Simple checkpoint save condition
        final_tracking_error = info['error']
        model_saved = ""
        if final_tracking_error < best_error:
            best_error = final_tracking_error
            torch.save(model.state_dict(), "best_model.pt")
            model_saved = "(New Best Tracking Model Saved!)"
        
        print(f"Epoch {epoch+1:03d}/{epochs} | Ep Reward: {sum(rewards):7.1f} | Final L2 Error: {final_tracking_error:6.3f} {model_saved}")
        
        # Anneal Exploration (Optionally lower noise over time as agent learns)
        action_std = max(0.05, action_std * 0.98)

if __name__ == "__main__":
    train()
