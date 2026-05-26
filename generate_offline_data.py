import numpy as np
import pandas as pd
from scipy.optimize import minimize
from roboticstoolbox import models
import matplotlib.pyplot as plt

from trajectory import SineTrajectory
from pid_controller import PIDValue, PIDController

def get_baseline_gains(setting='A'):
    if setting == 'A':
        return [
            PIDValue(Kp=800.0, Ki=50.0, Kd=40.0),
            PIDValue(Kp=1200.0, Ki=80.0, Kd=60.0),
            PIDValue(Kp=800.0, Ki=50.0, Kd=40.0),
            PIDValue(Kp=200.0, Ki=10.0, Kd=15.0),
            PIDValue(Kp=200.0, Ki=10.0, Kd=15.0),
            PIDValue(Kp=100.0, Ki=5.0, Kd=8.0)
        ]
    elif setting == 'B':
        return [
            PIDValue(Kp=652.4, Ki=41.2, Kd=32.6),
            PIDValue(Kp=978.6, Ki=61.8, Kd=48.9),
            PIDValue(Kp=652.4, Ki=41.2, Kd=32.6),
            PIDValue(Kp=163.1, Ki=10.3, Kd=8.2),
            PIDValue(Kp=163.1, Ki=10.3, Kd=8.2),
            PIDValue(Kp=81.15, Ki=5.1, Kd=4.1)
        ]
    else:
        raise ValueError(f"Unknown setting: {setting}")

def optimize_step(q, qd, sp_next, dt, M, C, G, errors, integrals, derivatives, tau_pid_base):
    M_inv = np.linalg.inv(M)
    C_qd_G = C @ qd + G
    
    def objective(x):
        dKp = x[0::3]
        dKi = x[1::3]
        dKd = x[2::3]
        
        tau_delta = dKp * errors + dKi * integrals + dKd * derivatives
        tau_total = tau_pid_base + tau_delta
        
        qdd = M_inv @ (tau_total - C_qd_G)
        qd_next = qd + qdd * dt
        q_next = q + qd_next * dt
        
        err = sp_next - q_next
        mse = np.sum(err**2) * 1e4
        
        reg_Kp = np.sum((dKp / 50.0)**2)
        reg_Ki = np.sum((dKi / 5.0)**2)
        reg_Kd = np.sum((dKd / 10.0)**2)  # Max Kd adjustment kept at 10.0
        
        reg_cost = 0.1 * (reg_Kp + reg_Ki + reg_Kd) 
        
        return mse + reg_cost

    x0 = np.zeros(18)
    bounds = []
    for _ in range(6):
        bounds.extend([(-50.0, 50.0), (-5.0, 5.0), (-10.0, 10.0)])
        
    result = minimize(objective, x0, method='L-BFGS-B', bounds=bounds)
    return result.x if result.success else x0

def generate_dataset(duration=10.0, dt=0.01, baseline_setting='A'):
    print(f"Generating Offline Dataset using Baseline Setting {baseline_setting}...")
    baseline_gains = get_baseline_gains(baseline_setting)
    robot = models.DH.Puma560()
    
    q = np.zeros(6)
    qd = np.zeros(6)
    
    trajectory = SineTrajectory([
        (45.0, 0.2, 0.0, 0.0),
        (30.0, 0.3, 0.0, 45.0),
        (20.0, 0.4, 0.0, 90.0),
        (45.0, 0.2, 0.0, 0.0),
        (45.0, 0.2, 0.0, 0.0),
        (45.0, 0.2, 0.0, 0.0)
    ])
    
    pids = [
        PIDController(Kp=np.float64(g.Kp), Ki=np.float64(g.Ki), Kd=np.float64(g.Kd))
        for g in baseline_gains
    ]
    
    t_steps = np.arange(0, duration, dt)
    dataset = []
    
    for t in t_steps:
        # Get next sp to optimize for
        sp_current = trajectory.get_setpoint(t)
        sp_next = trajectory.get_setpoint(min(t + dt, duration))
        
        for i in range(6):
            pids[i].setpoint = sp_current[i]
            
        M = robot.inertia(q)
        C = robot.coriolis(q, qd)
        G = robot.gravload(q)
        
        errors = np.zeros(6)
        integrals = np.zeros(6)
        derivatives = np.zeros(6)
        tau_pid_base = np.zeros(6)
        
        for i in range(6):
            # manual calculation to avoid updating PID state before optimization
            err = sp_current[i] - q[i]
            integral = (pids[i].integral * pids[i].leak_rate) + err * dt
            integral = max(min(integral, 1e6), -1e6)
            derivative = (err - pids[i].prev_error) / dt
            derivative = max(min(derivative, 1e6), -1e6)
            
            p_term = pids[i].Kp * err
            i_term = pids[i].Ki * integral
            d_term = pids[i].Kd * derivative
            tau = p_term + i_term + d_term
            
            errors[i] = err
            integrals[i] = integral
            derivatives[i] = derivative
            tau_pid_base[i] = tau

        # Optimize for Deltas
        optimal_deltas = optimize_step(q, qd, sp_next, dt, M, C, G, errors, integrals, derivatives, tau_pid_base)
        
        # Log to Dataset
        row = {
            'time': t,
        }
        for i in range(6):
            row[f'q_{i}'] = q[i]
            row[f'qd_{i}'] = qd[i]
            row[f'sp_{i}'] = sp_current[i]
            row[f'err_{i}'] = errors[i]
            row[f'integral_{i}'] = integrals[i]
            row[f'derivative_{i}'] = derivatives[i]
            
            row[f'delta_Kp_{i}'] = optimal_deltas[i*3]
            row[f'delta_Ki_{i}'] = optimal_deltas[i*3+1]
            row[f'delta_Kd_{i}'] = optimal_deltas[i*3+2]
            
        dataset.append(row)
        
        # Step simulation forward using the IDEAL total torque (baseline + delta)
        # This keeps the trajectory tightly bound to what the LSTM will learn
        dKp = optimal_deltas[0::3]
        dKi = optimal_deltas[1::3]
        dKd = optimal_deltas[2::3]
        tau_delta = dKp * errors + dKi * integrals + dKd * derivatives
        tau_final = tau_pid_base + tau_delta
        
        qdd = np.linalg.inv(M) @ (tau_final - C @ qd - G)
        qd = qd + qdd * dt
        q = q + qd * dt
        
        # Update actual PID states for next loop
        for i in range(6):
            pids[i].update(q[i], dt)
            
        if t * 100 % 100 == 0:
            print(f"Time {t:.2f}s computed. Mean absolute error: {np.mean(np.abs(sp_current - q)):.5f}")

    df = pd.DataFrame(dataset)
    df.to_csv('offline_pid_dataset.csv', index=False)
    print("Dataset saved to 'offline_pid_dataset.csv'")

if __name__ == "__main__":
    generate_dataset()
