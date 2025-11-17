import numpy as np
import roboticstoolbox as rtb

def run_forward_dynamics(torque_values, duration_s, dt, q_initial, qd_initial):
    """
    Runs a forward dynamics simulation and returns the results.
    """
    robot = rtb.models.DH.Puma560()
    n_joints = robot.n
    
    steps = int(duration_s / dt)
    if steps <= 0:
        raise ValueError("Duration and dt result in zero or negative steps.")
    t_vec = np.linspace(0, duration_s, steps)

    q = np.array(q_initial)
    qd = np.array(qd_initial)
    tau_applied = np.array(torque_values)

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
        qd = qd + qdd * dt
        q = q + qd * dt
        
    return (t_vec, 
            np.array(q_history), 
            np.array(qd_history), 
            np.array(qdd_history))

def run_inverse_dynamics(q_target_deg, duration_s, dt):
    """
    Runs an inverse dynamics simulation and returns the results.
    """
    robot = rtb.models.DH.Puma560()
    n_joints = robot.n

    t_vec = np.arange(0, duration_s + dt, dt)
    q0 = np.zeros(n_joints)
    
    # Convert target from degrees to radians
    q_target_rad = np.array(q_target_deg) * (np.pi / 180.0)
    
    if q_target_rad.shape[0] != n_joints:
        raise ValueError(f"Target position must have {n_joints} values.")

    # Generate a joint-space trajectory
    traj = rtb.jtraj(q0, q_target_rad, t_vec)
    
    # Extract motion profiles
    q = traj.q
    qd = traj.qd
    qdd = traj.qdd
    
    # Calculate required torques using Recursive Newton-Euler
    torques = robot.rne(q, qd, qdd)
    
    return (t_vec, q, qd, qdd, torques)