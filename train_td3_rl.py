import argparse
import copy
import os
import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter

from rl_env import Puma560Env, Puma560EnvTD3
from td3_lstm_models import TD3Actor, TD3Critic, ReplayBuffer, device

class TD3Agent:
    def __init__(
        self,
        state_dim,
        action_dim,
        max_action,
        discount=0.99,
        tau=0.005,
        policy_noise=0.2,
        noise_clip=0.5,
        policy_freq=2
    ):
        self.actor = TD3Actor(state_dim, action_dim, max_action).to(device)
        self.actor_target = copy.deepcopy(self.actor)
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)

        self.critic = TD3Critic(state_dim, action_dim).to(device)
        self.critic_target = copy.deepcopy(self.critic)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)

        self.max_action = max_action
        self.max_action_tensor = torch.FloatTensor(max_action).to(device)
        self.discount = discount
        self.tau = tau
        self.policy_noise = policy_noise
        self.noise_clip = noise_clip
        self.policy_freq = policy_freq

        self.total_it = 0

    def select_action(self, state):
        state = torch.FloatTensor(state).unsqueeze(0).to(device)
        with torch.no_grad():
            action = self.actor(state).cpu().data.numpy().flatten()
        return action

    def train(self, replay_buffer, batch_size=100):
        self.total_it += 1

        state, action, next_state, reward, not_done = replay_buffer.sample(batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.policy_noise).clamp(-self.noise_clip, self.noise_clip)
            scaled_noise = noise * self.max_action_tensor
            
            next_action = (self.actor_target(next_state) + scaled_noise).clamp(-self.max_action_tensor, self.max_action_tensor)

            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            target_Q = torch.min(target_Q1, target_Q2)
            target_Q = reward + not_done * self.discount * target_Q

        current_Q1, current_Q2 = self.critic(state, action)
        critic_loss = F.mse_loss(current_Q1, target_Q) + F.mse_loss(current_Q2, target_Q)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        if self.total_it % self.policy_freq == 0:
            actor_loss = -self.critic.Q1(state, self.actor(state)).mean()
            
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()

            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            return critic_loss.item(), actor_loss.item()
            
        return critic_loss.item(), 0.0

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
        self.actor_target = copy.deepcopy(self.actor)


def eval_policy(agent, baseline_setting='A', eval_episodes=5):
    env = Puma560EnvTD3(dt=0.001, rl_decimation=10, lstm_decimation=5, window_size=20, baseline_setting=baseline_setting)
    avg_reward = 0.
    avg_error = 0.
    for ep in range(eval_episodes):
        # 300 simulates late-stage safety cage alpha for strict evaluation
        state = env.reset(episode=300) 
        done = False
        while not done:
            action = agent.select_action(state)
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
    parser.add_argument("--expl_noise", default=0.1, type=float)
    parser.add_argument("--batch_size", default=64, type=int)
    parser.add_argument("--save_model", action="store_true", default=True)
    parser.add_argument("--load_model", default="")
    parser.add_argument("--baseline_setting", default="A", type=str, choices=["A", "B"])
    parser.add_argument("--save_freq", default=0, type=int, help="Save a checkpoint every this many episodes (0 to disable)")
    parser.add_argument("--early_stopping_patience", default=0, type=int, help="Stop training if no improvement for this many episodes (0 to disable)")
    args = parser.parse_args()

    import time
    timestamp = time.strftime("%Y%m%d-%H%M%S")
    run_name = f"td3_puma560_Setting_{args.baseline_setting}_{timestamp}"
    writer = SummaryWriter(log_dir=f"./runs/{run_name}")

    env = Puma560EnvTD3(dt=0.001, rl_decimation=10, lstm_decimation=5, window_size=20, baseline_setting=args.baseline_setting)
    
    state_dim = 42
    action_dim = 6
    max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
    
    agent = TD3Agent(state_dim, action_dim, max_action)
    
    if args.load_model != "":
        print(f"Loading checkpoint from: {args.load_model}")
        agent.load(args.load_model)

    replay_buffer = ReplayBuffer((20, state_dim), action_dim)
    
    best_error = float('inf')
    total_steps = 0
    patience_counter = 0
    
    print("Starting TD3 + LSTM Training...")
    
    for episode in range(1, args.max_episodes + 1):
        state = env.reset(episode=episode)
        ep_reward = 0
        ep_critic_loss = []
        ep_actor_loss = []
        
        for t in range(args.max_timesteps):
            total_steps += 1
            
            # Select action randomly or according to policy
            if total_steps < args.start_timesteps:
                action = np.random.uniform(-max_action, max_action)
            else:
                action = agent.select_action(state)
                noise = np.random.normal(0, max_action * args.expl_noise, size=action_dim)
                action = np.clip(action + noise, -max_action, max_action)
                
            next_state, reward, terminated, truncated, info = env.step(action, episode=episode)
            done_bool = terminated or truncated
            
            replay_buffer.add(state, action, next_state, reward, float(done_bool))
            
            state = next_state
            ep_reward += reward
            
            if total_steps >= args.start_timesteps:
                c_loss, a_loss = agent.train(replay_buffer, args.batch_size)
                ep_critic_loss.append(c_loss)
                if a_loss != 0.0:
                    ep_actor_loss.append(a_loss)
                    
            if done_bool:
                break
                
        tracking_error = info['error']
        avg_c_loss = np.mean(ep_critic_loss) if len(ep_critic_loss) > 0 else 0
        avg_a_loss = np.mean(ep_actor_loss) if len(ep_actor_loss) > 0 else 0
        
        # Logging to TensorBoard
        writer.add_scalar("Train/Reward", ep_reward, episode)
        writer.add_scalar("Train/Error", tracking_error, episode)
        writer.add_scalar("Train/Critic_Loss", avg_c_loss, episode)
        writer.add_scalar("Train/Actor_Loss", avg_a_loss, episode)
        writer.add_scalar("Train/Alpha", env.safety.alpha, episode)

        saved_str = ""
        
        # Evaluation Loop
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
                    agent.save(f"{checkpoint_dir}/td3_best")
                saved_str = f" (Saved Best Eval: {best_error:.3f})"
            else:
                patience_counter += args.eval_freq
                
        # Periodic Checkpointing
        if args.save_freq > 0 and episode % args.save_freq == 0:
            checkpoint_dir = f"checkpoints/Setting_{args.baseline_setting}"
            if not os.path.exists(checkpoint_dir):
                os.makedirs(checkpoint_dir)
            agent.save(f"{checkpoint_dir}/td3_ep_{episode}")
            saved_str += f" (Saved Ep {episode} Checkpoint)"
                
        print(f"Ep {episode:03d} | Steps: {t+1:03d} | Total: {total_steps} | Tr. Rwd: {ep_reward:7.1f} | Tr. Err: {tracking_error:6.3f} | C_Loss: {avg_c_loss:5.2f}{saved_str}")
        
        if args.early_stopping_patience > 0 and patience_counter >= args.early_stopping_patience:
            print(f"Early stopping triggered after {episode} episodes. No improvement for {patience_counter} episodes.")
            break

if __name__ == "__main__":
    if not os.path.exists("checkpoints"):
        os.makedirs("checkpoints")
    main()
