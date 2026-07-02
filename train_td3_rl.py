import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.tensorboard import SummaryWriter
import copy
import time
import os

from rl_env import Puma560EnvTD3
from td3_lstm_models import TD3Actor, TD3Critic, ReplayBuffer, LSTMTemporalObserver, device

def evaluate_policy(agent, env, eval_episodes=5):
    agent.actor.eval()
    
    total_reward = 0
    total_error = 0
    
    c_t = np.array([0.0, 1.0, 1.0, 0.5, 0.0, 0.0]) 
    max_action = np.array([15., 20., 15., 5., 5., 3.])
    
    for _ in range(eval_episodes):
        X_t = env.reset()
        h, c_hidden = np.zeros(256), np.zeros(256)
        
        q_ref, qd_ref, _ = env._get_reference(env.t_total)
        tau_pid, e, ed = env.pid.compute(q_ref, env.q, qd_ref, env.qd)
        
        done = False
        step_count = 0
        
        while not done and step_count < 500:
            step_count += 1
            
            h_t, c_hidden_t = agent.temporal_observer(X_t, h, c_hidden)
            s_t = X_t[-1]
            s_tilde_t = np.concatenate([s_t, h_t, c_t])
            
            with torch.no_grad():
                a_t_norm = agent.actor(torch.FloatTensor(s_tilde_t).unsqueeze(0).to(device)).cpu().numpy().flatten()
            
            a_t_env = a_t_norm * max_action
            
            tau_rl_safe, _ = env.safety_cage.apply(
                tau_rl_raw=a_t_env, 
                tau_pid=tau_pid, 
                e=e, ed=ed, 
                alpha=0.30, 
                qd=env.qd, 
                M_q=env.get_M()
            )
            
            s_t_next, next_tau_pid, next_e, next_ed = env.execute_inner_loop(tau_rl_safe)
            reward, done = env.compute_reward(next_e, a_t_env, tau_rl_safe)
            
            total_reward += reward
            total_error += np.linalg.norm(next_e)
            
            X_t = np.array(env.S)
            tau_pid, e, ed = next_tau_pid, next_e, next_ed
            h, c_hidden = h_t, c_hidden_t
            
    agent.actor.train()
    return total_reward / eval_episodes, total_error / (eval_episodes * 500)


class TD3Agent:
    """Algorithm 3 Logic encapsulated"""
    def __init__(self):
        # Noise scaled down to match normalized [-1.0, 1.0] action space
        self.sigma_explore = 0.1
        self.sigma_target = 0.2
        self.c_noise_clip = 0.5
        self.batch_size = 256  # 1. Batch size is confirmed to be 256
        self.discount = 0.99
        self.tau = 0.005
        self.d_policy_freq = 2
        
        # Line 1: Initialize actor pi_theta, critics Q_phi1, Q_phi2, replay buffer D
        # State dim is 40 (s_t) + 256 (h_t) + 6 (c_t) = 302
        self.actor = TD3Actor(state_dim=302, action_dim=6).to(device)
        self.critic = TD3Critic(state_dim=302, action_dim=6).to(device)
        
        # Line 2: Initialize target networks
        self.actor_target = copy.deepcopy(self.actor)
        self.critic_target = copy.deepcopy(self.critic)
        
        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=3e-4)
        self.critic_optimizer = torch.optim.Adam(self.critic.parameters(), lr=3e-4)
        
        self.replay_buffer = ReplayBuffer(state_dim=302, action_dim=6)
        # Override the default 42 from the paper's typo to the actual 40
        self.temporal_observer = LSTMTemporalObserver(input_dim=40).to(device)
        self.total_it = 0

    def select_action(self, s_tilde_t):
        state = torch.FloatTensor(s_tilde_t).unsqueeze(0).to(device)
        with torch.no_grad():
            return self.actor(state).detach().cpu().numpy().flatten()

    def update_td3(self):
        """Lines 15-20: UPDATETD3(D) {Per Equations (17)-(21)}"""
        self.total_it += 1
        state, action, next_state, reward, not_done = self.replay_buffer.sample(self.batch_size)

        with torch.no_grad():
            noise = (torch.randn_like(action) * self.sigma_target).clamp(-self.c_noise_clip, self.c_noise_clip)
            # Action Normalization: Target action bounded to [-1.0, 1.0]
            next_action = (self.actor_target(next_state) + noise).clamp(-1.0, 1.0)
            
            # Line 15: Compute target value y
            target_Q1, target_Q2 = self.critic_target(next_state, next_action)
            y = reward + not_done * self.discount * torch.min(target_Q1, target_Q2)

        current_Q1, current_Q2 = self.critic(state, action)
        
        # Line 16: Update critics
        critic_loss = F.mse_loss(current_Q1, y) + F.mse_loss(current_Q2, y)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        
        # 2. Gradient Clipping added for the Critic to prevent loss explosion
        torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=1.0)
        self.critic_optimizer.step()

        # Line 17: if t mod d == 0 then
        if self.total_it % self.d_policy_freq == 0:
            # Line 18: Update actor
            q1_out = self.critic.Q1(state, self.actor(state))
            actor_loss = -q1_out.mean()
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            
            # 2. Gradient Clipping added for the Actor 
            torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=1.0)
            self.actor_optimizer.step()

            # Line 19: Soft update targets
            for param, target_param in zip(self.critic.parameters(), self.critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for param, target_param in zip(self.actor.parameters(), self.actor_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
                
            return critic_loss.item(), actor_loss.item()
        return critic_loss.item(), 0.0

def main():
    """Algorithm 5: Complete Proposed Framework: PID + LSTM + TD3 + Safety Cage"""
    print("Starting exact execution of Algorithm 5...")
    
    # Require: Safety cage bounds and ramp
    alpha_start = 0.05
    alpha_end = 0.30
    N_ramp = 300
    N_episodes = 1000
    T_max = 500
    
    env = Puma560EnvTD3()
    agent = TD3Agent()
    c_t = np.array([0.0, 1.0, 1.0, 0.5, 0.0, 0.0]) # Context dimension matching visual
    max_action = np.array([15., 20., 15., 5., 5., 3.]) # Real physical limits
    
    run_name = f"algo5_exact_run_{time.strftime('%Y%m%d-%H%M%S')}"
    writer = SummaryWriter(log_dir=f"./runs/{run_name}")

    # 3: for episode k = 1 to N_episodes do
    for k in range(1, N_episodes + 1):
        
        # 4: alpha <- min(alpha_start + 0.25 * k / N_ramp, alpha_end)
        alpha = min(alpha_start + 0.25 * (k / N_ramp), alpha_end)
        
        # 5: Reset environment, obtain s_tilde_0 sequence
        X_t = env.reset()
        
        # 2: Initialize LSTM hidden state h <- 0, cell state c <- 0
        h, c_hidden = np.zeros(256), np.zeros(256)
        
        # Extract initial PID and error states needed for first step
        q_ref, qd_ref, _ = env._get_reference(env.t_total)
        tau_pid, e, ed = env.pid.compute(q_ref, env.q, qd_ref, env.qd)
        
        ep_reward = 0
        ep_c_loss, ep_a_loss = [], []

        # 6: for step t = 0 to T_max do
        for t in range(T_max):
            
            # 7: LSTMFORWARD(X_t, h, c) -> h_t, c_t
            h_t, c_hidden_t = agent.temporal_observer(X_t, h, c_hidden)
            
            # 8: Form augmented observation s_tilde_t <- [s_t, h_t, c_t]^T
            s_t = X_t[-1]
            s_tilde_t = np.concatenate([s_t, h_t, c_t])
            
            # 9: Select NORMALIZED action a_t_norm <- pi_theta(s_tilde_t) + noise
            a_t_norm = agent.select_action(s_tilde_t)
            a_t_norm = a_t_norm + np.random.normal(0, agent.sigma_explore, size=6)
            a_t_norm = np.clip(a_t_norm, -1.0, 1.0)
            
            # Scale to physical bounds immediately before passing to the environment
            a_t_env = a_t_norm * max_action
            
            # (Line 10 handled by inner loop implicitly)
            
            # 11: SAFETYCAGE(a_t_env, tau_PID, alpha) -> tau_RL_safe
            tau_rl_safe, V_t = env.safety_cage.apply(
                tau_rl_raw=a_t_env, 
                tau_pid=tau_pid, 
                e=e, ed=ed, 
                alpha=alpha, 
                qd=env.qd, 
                M_q=env.get_M()
            )
            
            # 12-13: Execute tau_total, observe next state and reward (handled in inner loop)
            s_t_next, next_tau_pid, next_e, next_ed = env.execute_inner_loop(tau_rl_safe)
            reward, done = env.compute_reward(next_e, a_t_env, tau_rl_safe)
            
            # Prepare next augmented state to store transition correctly
            X_t_next = np.array(env.S)
            h_next, c_hidden_next = agent.temporal_observer(X_t_next, h_t, c_hidden_t)
            s_tilde_t_next = np.concatenate([s_t_next, h_next, c_t])
            
            # 14: Store (s_tilde_t, a_t_norm, r_t, s_tilde_t_next, done) in D
            # Note: We store the NORMALIZED action (a_t_norm) bounded to [-1, 1]
            agent.replay_buffer.add(s_tilde_t, a_t_norm, s_tilde_t_next, reward, done)
            
            # 15: if |D| > batch_size then
            if agent.replay_buffer.size > agent.batch_size:
                # 16: UPDATETD3(D) {Per Equations (17)-(21)}
                c_l, a_l = agent.update_td3()
                ep_c_loss.append(c_l)
                if a_l != 0.0: ep_a_loss.append(a_l)
                
            # Advance loop variables
            X_t = X_t_next
            tau_pid, e, ed = next_tau_pid, next_e, next_ed
            h, c_hidden = h_t, c_hidden_t
            ep_reward += reward
            
            # 22: if done then (break implied by environment termination)
            if done: break
            
        # Logging
        tracking_error = np.linalg.norm(e)
        avg_c = np.mean(ep_c_loss) if ep_c_loss else 0
        
        # EVALUATION PHASE
        if k % 10 == 0:
            eval_reward, eval_error = evaluate_policy(agent, env)
            writer.add_scalar('Eval/Reward', eval_reward, k)
            writer.add_scalar('Eval/Error', eval_error, k)
            
        # Save checkpoints every 100 episodes
        if k % 100 == 0:
            os.makedirs("checkpoints", exist_ok=True)
            torch.save(agent.actor.state_dict(), f"checkpoints/actor_ep_{k}.pth")
            torch.save(agent.critic.state_dict(), f"checkpoints/critic_ep_{k}.pth")
            
        # Advanced Monitoring
        writer.add_scalar("Train/Reward", ep_reward, k)
        writer.add_scalar("Train/Error", tracking_error, k)
        writer.add_scalar("Train/Critic_Loss", avg_c, k)
        writer.add_scalar("Train/Alpha", alpha, k)
        writer.add_scalar("Safety/Raw_vs_Safe_Distance", np.mean((a_t_env - tau_rl_safe)**2), k)
        writer.add_scalar("Safety/Lyapunov_V", V_t, k)
        
        writer.flush()
        
        # 3. Periodic Replay Buffer Action: Clear buffer every 50 episodes
        if k % 50 == 0:
            agent.replay_buffer.ptr = 0
            agent.replay_buffer.size = 0
            print(f"  [Action] Replay Buffer flushed at episode {k} to clear stale curriculum data.")
        
        print(f"Ep {k:03d} | Rwd: {ep_reward:7.1f} | Err: {tracking_error:6.3f} | α: {alpha:.3f} | C_Loss: {avg_c:5.2f}")

    # Save final model
    os.makedirs("checkpoints", exist_ok=True)
    torch.save(agent.actor.state_dict(), "checkpoints/td3_best_actor.pth")
    print("Final model saved to checkpoints/td3_best_actor.pth")

if __name__ == "__main__":
    main()