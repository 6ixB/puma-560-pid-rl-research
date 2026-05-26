import numpy as np
import torch
import roboticstoolbox as rtb
import matplotlib.pyplot as plt

from generate_offline_data import get_baseline_gains
from lstm_pid_simulation import LSTMPIDTuner
from trajectory import SineTrajectory
from pid_controller import PIDController, plot_pid_controller_output

def simulate_with_lstm(model_path='lstm_supervised_weights.pth', duration=10.0, dt=0.01, baseline_setting='A'):
    print(f"Loading Supervised LSTM Model and simulating with Baseline Setting {baseline_setting}...")
    
    window_size = 10
    model = LSTMPIDTuner(input_size=24, hidden_size=64, num_layers=2, output_size=18, window_size=window_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    
    baseline_gains = get_baseline_gains(baseline_setting)
    robot = rtb.models.DH.Puma560()
    
    q = np.zeros(6, dtype=np.float64)
    qd = np.zeros(6, dtype=np.float64)
    
    trajectory = SineTrajectory([
        (45.0, 0.2, 0.0, 0.0),
        (30.0, 0.3, 0.0, 45.0),
        (20.0, 0.4, 0.0, 90.0),
        (45.0, 0.2, 0.0, 0.0),
        (45.0, 0.2, 0.0, 0.0),
        (45.0, 0.2, 0.0, 0.0)
    ])
    
    # Init PIDs
    pids = [
        PIDController(Kp=np.float64(g.Kp), Ki=np.float64(g.Ki), Kd=np.float64(g.Kd))
        for g in baseline_gains
    ]
    
    t_steps = np.arange(0, duration, dt)
    history = np.zeros((window_size, 24), dtype=np.float32)
    
    q_values = [[] for _ in range(6)]
    u_values = [[] for _ in range(6)]
    setpoints = [[] for _ in range(6)]
    p_values = [[] for _ in range(6)]
    i_values = [[] for _ in range(6)]
    d_values = [[] for _ in range(6)]
    
    print("Simulating...")
    for t in t_steps:
        sp_current = trajectory.get_setpoint(t)
        
        # 1. Update Target Error and Features (History Shift)
        for i in range(6):
            pids[i].setpoint = sp_current[i]
            
        err = sp_current - q
        integrals = np.array([pid.integral for pid in pids], dtype=np.float64)
        features = np.concatenate([q, qd, err, integrals]).astype(np.float32)
        
        history[:-1] = history[1:]
        history[-1] = features
        
        # 2. Get scaled Deltas from LSTM
        with torch.no_grad():
            hist_tensor = torch.tensor(history).unsqueeze(0) # add batch dim
            scaled_deltas = model.get_deltas(hist_tensor)[0].numpy()
            
        # 3. Apply baseline + deltas
        for i in range(6):
            base = baseline_gains[i]
            pids[i].Kp = base.Kp + scaled_deltas[i*3+0]
            pids[i].Ki = base.Ki + scaled_deltas[i*3+1]
            pids[i].Kd = base.Kd + scaled_deltas[i*3+2]
            
        # 4. Standard control flow
        tau_pid = np.array([pids[i].update(q[i], dt)[0] for i in range(6)])
        
        M = robot.inertia(q)
        C = robot.coriolis(q, qd)
        G = robot.gravload(q)
        
        qdd = np.linalg.inv(M) @ (tau_pid - C @ qd - G)
        qd = qd + qdd * dt
        q = q + qd * dt
        
        # Track for plotting
        for i in range(6):
            q_values[i].append(np.rad2deg(q[i]))
            u_values[i].append(tau_pid[i])
            setpoints[i].append(np.rad2deg(sp_current[i]))
            p_values[i].append(pids[i].p_term)
            i_values[i].append(pids[i].i_term)
            d_values[i].append(pids[i].d_term)
            
    print("Simulation complete! Plotting results for Joint 2 (Shoulder)...")
    
    # We plot using the existing function. 
    # Notice we pass list of lists for setpoints only because it expects (t_steps, q_values, u_values, setpoints, ...)
    # Wait, in pid_controller.py `plot_pid_controller_output` expects setpoints to purely be a flat array for a static trajectory OR list of lists. Actually, wait.
    # The original function signature: plot_pid_controller_output(t_steps, q_values, u_values, setpoints, p_values, i_values, d_values)
    # Inside it uses `np.rad2deg(setpoints[joint_to_plot])`, so `setpoints` must be scalar.
    # Wait, check `pid_controller.py`: `ax1.axhline(np.rad2deg(setpoints[joint_to_plot]))`. Yes, it expects constant setpoint per joint.
    # We have a sine trajectory, so let's just make our own minor plot here.
    
    joint_to_plot = 1  # 0-indexed, so 1 is Joint 2
    
    plt.figure(figsize=(10, 8))
    plt.subplot(3, 1, 1)
    plt.plot(t_steps, setpoints[joint_to_plot], 'r--', label='Target (J2)')
    plt.plot(t_steps, q_values[joint_to_plot], 'b', label='LSTM Response (J2)')
    plt.ylabel('Response (deg)')
    plt.title('Residual Gain Tuning Performance (LSTM + Baseline)')
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 2)
    plt.plot(t_steps, p_values[joint_to_plot], 'r', label='P-term')
    plt.plot(t_steps, d_values[joint_to_plot], 'b', label='D-term')
    plt.plot(t_steps, i_values[joint_to_plot], 'g', label='I-term')
    plt.ylabel('PID Output')
    plt.legend()
    plt.grid()
    
    plt.subplot(3, 1, 3)
    plt.plot(t_steps, u_values[joint_to_plot], 'k', label='Total Torque')
    plt.xlabel('Time (s)')
    plt.ylabel('Torque')
    plt.legend()
    plt.grid()
    
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    simulate_with_lstm(baseline_setting='A')
