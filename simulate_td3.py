import argparse
import numpy as np
import matplotlib.pyplot as plt
import torch

from rl_env import Puma560Env, Puma560EnvTD3
from td3_lstm_models import TD3Actor, device
from trajectory import StaticTrajectory

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--baseline_setting", default="A", type=str, choices=["A", "B"], help="Default setting if not comparing.")
    parser.add_argument("--checkpoint", default=None, type=str, help="Specific checkpoint path for the default run.")
    parser.add_argument("--compare", nargs="+", help="List of 'Setting:CheckpointPath:Label' to compare. e.g. A:checkpoints/Setting_A/td3_best_actor:Run1")
    args = parser.parse_args()

    # Step input to 90 degree starting from 0 degree for all joints
    step_trajectory = StaticTrajectory(np.deg2rad(np.ones(6) * 90.0))
    
    runs = []
    if args.compare:
        for comp in args.compare:
            parts = comp.split(":")
            if len(parts) == 3:
                runs.append({"setting": parts[0], "checkpoint": parts[1], "label": parts[2]})
            elif len(parts) == 2:
                runs.append({"setting": parts[0], "checkpoint": parts[1], "label": parts[1]})
            else:
                print(f"Invalid compare format: {comp}. Use Setting:CheckpointPath:Label")
                return
    else:
        ckpt = args.checkpoint if args.checkpoint else f"checkpoints/Setting_{args.baseline_setting}/td3_best_actor"
        runs.append({"setting": args.baseline_setting, "checkpoint": ckpt, "label": f"Setting {args.baseline_setting}"})

    state_dim = 42
    action_dim = 6
    max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
    
    results = []

    for run in runs:
        print(f"Running evaluation simulation for {run['label']}...")
        env = Puma560EnvTD3(dt=0.001, rl_decimation=10, lstm_decimation=5, window_size=20, baseline_setting=run['setting'], trajectory=step_trajectory)
        actor = TD3Actor(state_dim, action_dim, max_action).to(device)
        
        try:
            actor.load_state_dict(torch.load(run['checkpoint'], map_location=device))
            actor.eval()
            print(f"Successfully loaded checkpoint: {run['checkpoint']}")
        except FileNotFoundError:
            print(f"Error: Could not find checkpoint at {run['checkpoint']}.")
            continue

        state = env.reset(episode=300) # Full safety cage active
        
        t_steps = []
        q_values = [[] for _ in range(6)]
        sp_values = [[] for _ in range(6)]
        err_values = [[] for _ in range(6)]
        tau_rl_values = [[] for _ in range(6)]
        
        done = False
        step_idx = 0
        
        while not done:
            with torch.no_grad():
                state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
                action = actor(state_tensor).cpu().data.numpy().flatten()
                
            state, reward, terminated, truncated, info = env.step(action, episode=300)
            # Ignore truncation during this test so the simulation doesn't stop immediately
            # due to the large initial 90-degree error (which is > 0.5 radians).
            done = terminated
            
            t_steps.append(step_idx * 0.01) # RL step is 10ms
            current_q = env.history[-1, :6]
            current_sp = env.history[-1, 30:36]
            current_err = env.history[-1, 12:18]
            
            for i in range(6):
                q_values[i].append(np.rad2deg(current_q[i]))
                sp_values[i].append(np.rad2deg(current_sp[i]))
                err_values[i].append(np.rad2deg(current_err[i]))
                tau_rl_values[i].append(action[i])
                
            step_idx += 1
            
        results.append({
            "label": run["label"],
            "t_steps": t_steps,
            "q_values": q_values,
            "sp_values": sp_values,
            "err_values": err_values,
            "tau_rl_values": tau_rl_values
        })

    if not results:
        print("No results to plot.")
        return

    # Plotting for all 6 joints
    print("Plotting results for all joints...")
    
    # Use 3 columns: Angle, Error, Torque
    fig, axes = plt.subplots(6, 3, figsize=(20, 20), sharex=True)
    
    for joint_idx in range(6):
        ax_q = axes[joint_idx, 0]
        ax_err = axes[joint_idx, 1]
        ax_tau = axes[joint_idx, 2]
        
        # Plot Setpoint
        ax_q.plot(results[0]["t_steps"], results[0]["sp_values"][joint_idx], '--', label="Setpoint", color='black')
        
        for res in results:
            ax_q.plot(res["t_steps"], res["q_values"][joint_idx], label=f"Angle ({res['label']})")
            ax_err.plot(res["t_steps"], res["err_values"][joint_idx], label=f"Error ({res['label']})")
            ax_tau.plot(res["t_steps"], res["tau_rl_values"][joint_idx], label=f"Torque ({res['label']})")
            
        ax_q.set_title(f"Joint {joint_idx+1} Tracking")
        ax_q.set_ylabel("Angle (deg)")
        ax_q.grid()
        if joint_idx == 0: ax_q.legend()
            
        ax_err.set_title(f"Joint {joint_idx+1} Error")
        ax_err.set_ylabel("Error (deg)")
        ax_err.grid()
        if joint_idx == 0: ax_err.legend()

        ax_tau.set_title(f"Joint {joint_idx+1} TD3 Torque")
        ax_tau.set_ylabel("Torque (Nm)")
        ax_tau.grid()
        if joint_idx == 0: ax_tau.legend()
            
        if joint_idx == 5:
            ax_q.set_xlabel("Time (s)")
            ax_err.set_xlabel("Time (s)")
            ax_tau.set_xlabel("Time (s)")

    plt.tight_layout()
    plt.savefig("simulation_results.png")
    print("Saved plot to 'simulation_results.png' in case the window doesn't appear.")
    try:
        plt.show(block=True)
    except Exception as e:
        print(f"Could not open plot window: {e}")

def run_td3_simulation(baseline_setting, duration, dt, q0_rad, trajectory):
    env = Puma560EnvTD3(dt=0.001, rl_decimation=10, lstm_decimation=5, window_size=20, baseline_setting=baseline_setting, trajectory=trajectory)
    
    state_dim = 42
    action_dim = 6
    max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
    
    actor = TD3Actor(state_dim, action_dim, max_action).to(device)
    
    checkpoint_path = f"checkpoints/Setting_{baseline_setting}/td3_best_actor"
    try:
        actor.load_state_dict(torch.load(checkpoint_path, map_location=device))
        actor.eval()
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Missing checkpoint: {checkpoint_path}. Please train this setting first!") from e
        
    state = env.reset(episode=300)
    if q0_rad is not None:
        env.q = np.array(q0_rad)
        
    t_steps = []
    q_values = [[] for _ in range(6)]
    err_values = [[] for _ in range(6)]
    u_values = [[] for _ in range(6)]
    torque_values = [[] for _ in range(6)]
    p_values = [[] for _ in range(6)]
    i_values = [[] for _ in range(6)]
    d_values = [[] for _ in range(6)]
    setpoint_values = [[] for _ in range(6)]
    
    step_idx = 0
    rl_step_time = 0.001 * 10 # 10ms
    max_steps = int(duration / rl_step_time)
    
    while step_idx < max_steps:
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            action = actor(state_tensor).cpu().data.numpy().flatten()
            
        state, reward, terminated, truncated, info = env.step(action, episode=300)
        
        t_curr = step_idx * rl_step_time
        t_steps.append(t_curr)
        
        current_q = env.history[-1, :6]
        current_sp = env.history[-1, 30:36]
        current_err = env.history[-1, 12:18]
        
        for i in range(6):
            q_values[i].append(np.rad2deg(current_q[i]))
            setpoint_values[i].append(np.rad2deg(current_sp[i]))
            err_values[i].append(np.rad2deg(current_err[i]))
            u_values[i].append(action[i])
            torque_values[i].append(action[i])
            p_values[i].append(0.0)
            i_values[i].append(0.0)
            d_values[i].append(0.0)
            
        step_idx += 1
        
    return t_steps, q_values, err_values, u_values, torque_values, p_values, i_values, d_values, setpoint_values

if __name__ == "__main__":
    main()
