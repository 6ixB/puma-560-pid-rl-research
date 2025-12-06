import numpy as np
import roboticstoolbox as rtb

def run_forward_dynamics(torque_values, duration_s, dt, q_initial_deg, qd_initial_deg):
    """
    Runs a forward dynamics simulation.
    Inputs are in Degrees, Calculation in Radians, Returns Degrees.
    """
    robot = rtb.models.DH.Puma560()
    n_joints = robot.n
    
    steps = int(duration_s / dt)
    if steps <= 0:
        raise ValueError("Duration and dt result in zero or negative steps.")
    t_vec = np.linspace(0, duration_s, steps)

    # Convert Inputs: Degrees -> Radians
    q = np.array(q_initial_deg) * (np.pi / 180.0)
    qd = np.array(qd_initial_deg) * (np.pi / 180.0)
    tau_applied = np.array(torque_values) # Torque is Nm

    if q.shape[0] != n_joints or qd.shape[0] != n_joints or tau_applied.shape[0] != n_joints:
        raise ValueError(f"Inputs must all be of length {n_joints}")

    q_history = []
    qd_history = []
    qdd_history = []

    for i in range(steps):
        qdd = robot.accel(q, qd, tau_applied)
        q_history.append(q)
        qd_history.append(qd)
        qdd_history.append(qdd)
        
        # Euler Integration
        qd = qd + qdd * dt
        q = q + qd * dt
        
    # Convert Results: Radians -> Degrees
    q_deg = np.array(q_history) * (180.0 / np.pi)
    qd_deg = np.array(qd_history) * (180.0 / np.pi)
    qdd_deg = np.array(qdd_history) * (180.0 / np.pi)

    return (t_vec, q_deg, qd_deg, qdd_deg)

def run_inverse_dynamics(q_initial_deg, q_target_deg, duration_s, dt):
    """
    Runs an inverse dynamics simulation.
    Generates a trajectory from Initial to Target.
    Returns Motion (in Degrees) and Torque (in Nm).
    """
    robot = rtb.models.DH.Puma560()
    n_joints = robot.n

    t_vec = np.arange(0, duration_s + dt, dt)
    
    # Convert Inputs: Degrees -> Radians
    q_initial_rad = np.array(q_initial_deg) * (np.pi / 180.0)
    q_target_rad = np.array(q_target_deg) * (np.pi / 180.0)
    
    if q_initial_rad.shape[0] != n_joints:
        raise ValueError(f"Initial values must have {n_joints} items.")
    if q_target_rad.shape[0] != n_joints:
        raise ValueError(f"Target values must have {n_joints} items.")

    # Generate a joint-space trajectory (jtraj uses quintic polynomial)
    traj = rtb.jtraj(q_initial_rad, q_target_rad, t_vec)
    
    # Extract motion profiles (Radians)
    q_rad = traj.q
    qd_rad = traj.qd
    qdd_rad = traj.qdd
    
    # Calculate required torques using Recursive Newton-Euler (requires Radians)
    torques = robot.rne(q_rad, qd_rad, qdd_rad)
    
    # Convert Motion Results: Radians -> Degrees
    q_deg = q_rad * (180.0 / np.pi)
    qd_deg = qd_rad * (180.0 / np.pi)
    qdd_deg = qdd_rad * (180.0 / np.pi)

    return (t_vec, q_deg, qd_deg, qdd_deg, torques)
