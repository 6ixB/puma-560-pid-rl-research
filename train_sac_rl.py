import argparse
import copy
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rl_env import Puma560Env
from sac_lstm_models import SACActor, SACCritic, ReplayBuffer, device

class SACAgent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        alpha=0.2  # Entropy coefficient
    ):
        self.actor = SACActor(state_dim, action_dim, max_action).to(device)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = SACCritic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.max_action_tensor = torch.FloatTensor(max_action).to(device)
        self.discount = discount
        self.tau = tau
        self.alpha = alpha

    def select_action(self, state, evaluate=False):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            if evaluate:
                mu, _ = self.actor(state)
                action = torch.tanh(mu) * self.max_action_tensor
            else:
                action, _ = self.actor.sample(state)
        return action.cpu().data.numpy().flatten()
        
    def get_critic_uncertainty(self, state, action):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        action = torch.FloatTensor(action).unsqueeze(0).to(device)
        with torch.no_grad():
            q1, q2 = self.critic(state, action)
            return torch.abs(q1 - q2).mean().item()

    def train(self, replay_buffer, batch_size=100):
        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            next_action, next_log_prob = self.actor.sample(next_state)
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2) - self.alpha * next_log_prob
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Actor Update
        pi, log_prob = self.actor.sample(state)
        q1_pi, q2_pi = self.critic(state, pi)
        min_q_pi = torch.min(q1_pi, q2_pi)
        
        actor_loss = ((self.alpha * log_prob) - min_q_pi).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Soft update target networks
        for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return critic_loss.item(), actor_loss.item()

    def save(self, filename):
        torch.save(self.critic.state_dict(), filename + "_critic")
        torch.save(self.critic_optimizer.state_dict(), filename + "_critic_optimizer")
        torch.save(self.actor.state_dict(), filename + "_actor")
        torch.save(self.actor_optimizer.state_dict(), filename + "_actor_optimizer")

    def load(self, filename):
        self.critic.load_state_dict(torch.load(filename + "_critic"))
        self.critic_optimizer.load_state_dict(torch.load(filename + "_critic_optimizer"))
        self.critic_target = copy.deepcopy(self.critic)

        self.actor.load_state_dict(torch.load(filename + "_actor"))
        self.actor_optimizer.load_state_dict(torch.load(filename + "_actor_optimizer"))


def eval_policy(agent, baseline_setting='A', eval_episodes=5):
    env = Puma560Env(dt=0.001, rl_decimation=10, lstm_decimation=5, window_size=20, baseline_setting=baseline_setting)
    avg_reward = 0.
    avg_error = 0.
    for ep in range(eval_episodes):
        state = env.reset(episode=300) 
        done = False
        while not done:
            action = agent.select_action(state, evaluate=True)
            state, reward, terminated, truncated, info = env.step(action, episode=300)
            avg_reward += reward
            done = terminated or truncated
        avg_error += info['error']
        
    avg_reward /= eval_episodes
    avg_error /= eval_episodes
    return avg_reward, avg_error


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--max_episodes", default=1000, type=int)
    parser.add_argument("--start_timesteps", default=1000, type=int)
    parser.add_argument("--eval_freq", default=10, type=int)
    parser.add_argument("--max_timesteps", default=500, type=int)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--baseline_setting", default="A", type=str, choices=["A", "B"])
    parser.add_argument("--save_freq", default=0, type=int, help="Save a checkpoint every this many episodes (0 to disable)")
    parser.add_argument("--early_stopping_patience", default=0, type=int, help="Stop training if no improvement for this many episodes (0 to disable)")
    args = parser.parse_args()

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"sac_puma560_Setting_{args.baseline_setting}_{timestamp}"
    writer = SummaryWriter(log_dir=f"./runs/{run_name}")

    env = Puma560Env(dt=0.001, rl_decimation=10, lstm_decimation=5, window_size=20, baseline_setting=args.baseline_setting)
    
    state_dim = 42
    action_dim = 6
    max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
    
    agent = SACAgent(state_dim, action_dim, max_action)
    
    if args.load_model != "":
        print(f"Loading checkpoint from: {args.load_model}")
        agent.load(args.load_model)

    replay_buffer = ReplayBuffer((20, state_dim), action_dim)
    
    best_error = float('inf')
    total_steps = 0
    patience_counter = 0
    
    print("Starting SAC + LSTM Training...")
    
    for episode in range(1, args.max_episodes + 1):
        state = env.reset(episode=episode)
        ep_reward = 0
        ep_critic_loss = []
        ep_actor_loss = []
        ep_uncertainties = []
        
        for t in range(args.max_timesteps):
            total_steps += 1
            
            if total_steps < args.start_timesteps:
                action = np.random.uniform(-max_action, max_action)
                critic_uncertainty = 0.0
            else:
                action = agent.select_action(state, evaluate=False)
                critic_uncertainty = agent.get_critic_uncertainty(state, action)
                
            ep_uncertainties.append(critic_uncertainty)
                
            # Note: We pass critic_uncertainty here to potentially use for Critic-Aware Safety Cage
            next_state, reward, terminated, truncated, info = env.step(action, episode=episode, critic_uncertainty=critic_uncertainty)
            done_bool = terminated or truncated
            
            replay_buffer.add(state, action, next_state, reward, float(done_bool))
            
            state = next_state
            ep_reward += reward
            
            if total_steps >= args.start_timesteps:
                c_loss, a_loss = agent.train(replay_buffer, args.batch_size)
                ep_critic_loss.append(c_loss)
                ep_actor_loss.append(a_loss)
                    
            if done_bool:
                break
                
        tracking_error = info['error']
        avg_c_loss = np.mean(ep_critic_loss) if len(ep_critic_loss) > 0 else 0
        avg_a_loss = np.mean(ep_actor_loss) if len(ep_actor_loss) > 0 else 0
        avg_uncertainty = np.mean(ep_uncertainties) if len(ep_uncertainties) > 0 else 0
        
        writer.add_scalar("Train/Reward", ep_reward, episode)
        writer.add_scalar("Train/Error", tracking_error, episode)
        writer.add_scalar("Train/Critic_Loss", avg_c_loss, episode)
        writer.add_scalar("Train/Actor_Loss", avg_a_loss, episode)
        writer.add_scalar("Train/Critic_Uncertainty", avg_uncertainty, episode)

        saved_str = ""
        
        if episode % args.eval_freq == 0:
            eval_reward, eval_error = eval_policy(agent, args.baseline_setting)
            writer.add_scalar("Eval/Reward", eval_reward, episode)
            writer.add_scalar("Eval/Error", eval_error, episode)
            
            if eval_error < best_error:
                best_error = eval_error
                patience_counter = 0
                if args.save_model:
                    checkpoint_dir = f"checkpoints/Setting_{args.baseline_setting}"
                    if not os.path.exists(checkpoint_dir):
                        os.makedirs(checkpoint_dir)
                    agent.save(f"{checkpoint_dir}/sac_best")
                saved_str = f" (Saved Best Eval: {best_error:.3f})"
            else:
                patience_counter += args.eval_freq
                
        if args.save_freq > 0 and episode % args.save_freq == 0:
            checkpoint_dir = f"checkpoints/Setting_{args.baseline_setting}"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            agent.save(f"{checkpoint_dir}/sac_ep_{episode}")
            saved_str += f" (Saved Ep {episode} Checkpoint)"
                
        print(f"Ep {episode:03d} | Steps: {t+1:03d} | Total: {total_steps} | Tr. Rwd: {ep_reward:7.1f} | Tr. Err: {tracking_error:6.3f} | C_Loss: {avg_c_loss:5.2f}{saved_str}")
        
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered after {episode} episodes. No improvement for {patience_counter} episodes.")
            break

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    main()
