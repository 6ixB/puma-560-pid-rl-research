import numpy as np
import time
from scipy.optimize import minimize

from pid_controller import PIDValue, run_pid_controller

def objective(x, duration, dt):

    pid_values = []
    for i in range(6):
        kp, ki, kd = x[i*3], x[i*3+1], x[i*3+2]
        pid_values.append(PIDValue(Kp=np.float64(kp), Ki=np.float64(ki), Kd=np.float64(kd)))

    setpoints_rad = np.zeros(6, dtype=np.float64)
    q0_rad = np.zeros(6, dtype=np.float64)  
    
    try:
        (
            t_steps,
            q_values,
            error_values,
            u_values,
            torque_values,
            p_values,
            i_values,
            d_values,
            setpoint_values,
        ) = run_pid_controller(
            setpoints=setpoints_rad,
            pid_values=pid_values,
            duration=duration,
            dt=dt,
            q0=q0_rad,
        )
        
        total_cost = 0.0
        
        for joint_idx in range(6):
            joint_errors = np.array(error_values[joint_idx])
            
            mse = np.mean(joint_errors ** 2)
            
            effort_penalty = 0.0001 * np.mean(np.array(u_values[joint_idx]) ** 2)
            
            weight = 10.0 if joint_idx < 3 else 1.0 
            
            total_cost += (mse + effort_penalty) * weight

        return total_cost

    except Exception as e:
        return 1e9


def run_optimization():
    print("Starting PID optimization to maintain 0 degree rest position...")
    
    duration = 20.0
    dt = 0.01 
    
    x0 = np.array([
        100.0, 1.0, 10.0,  # Joint 1
        100.0, 1.0, 10.0,  # Joint 2
        100.0, 1.0, 10.0,  # Joint 3
        50.0,  0.5, 5.0,   # Joint 4
        50.0,  0.5, 5.0,   # Joint 5
        50.0,  0.5, 5.0    # Joint 6
    ])

    bounds = []
    for i in range(6):
        bounds.extend([(0.0, 500.0), (0.0, 50.0), (0.0, 100.0)])
        
    start_time = time.time()
    
    print("Running L-BFGS-B minimization...")
    result = minimize(
        objective, 
        x0, 
        args=(duration, dt),
        method='L-BFGS-B', 
        bounds=bounds,
        options={'maxiter': 500, 'disp': True}
    )
    
    end_time = time.time()
    
    print(f"\nOptimization Finished in {end_time - start_time:.2f} seconds!")
    print(f"Final Objective Score: {result.fun:.4f}")
    
    if result.success:
        print("Optimization Converged Successfully.")
    else:
        print(f"Optimization Stopped: {result.message}")
        
    print("\n--- Optimized PID Values ---")
    for i in range(6):
        kp, ki, kd = result.x[i*3], result.x[i*3+1], result.x[i*3+2]
        print(f"Joint {i+1}: Kp={kp:.2f}, Ki={ki:.2f}, Kd={kd:.2f}")

if __name__ == "__main__":
    run_optimization()