import numpy as np
from roboticstoolbox import models

# ============================================================================
# PUMA 560 Dynamic Parameters (6 Joints)
# ============================================================================
class PUMA560Params:
    J = np.array([4.0, 6.0, 4.5, 1.5, 1.0, 0.8])
    B = np.array([0.06, 0.08, 0.07, 0.04, 0.04, 0.03])
    Fc_pos = np.array([0.60, 0.80, 0.70, 0.40, 0.40, 0.30])
    Fc_neg = np.array([0.65, 0.85, 0.75, 0.42, 0.42, 0.32])
    vs = np.array([0.005, 0.006, 0.005, 0.004, 0.004, 0.003])
    Fs = np.array([0.20, 0.25, 0.22, 0.15, 0.15, 0.12])
    G_coeff = np.array([0.0, 50.0, 30.0, 0.0, 0.0, 0.0])
    tau_max = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
    q_limits = np.array([
        [-2.967, 2.967], [-0.785, 2.356], [-2.356, 2.356],
        [-3.142, 3.142], [-2.094, 2.094], [-3.142, 3.142]
    ])

def compute_friction(qd, params):
    tau_f = np.zeros(6)
    for i in range(6):
        vel = qd[i]
        abs_vel = abs(vel)
        Fc = params.Fc_pos[i] if vel > 0.001 else (params.Fc_neg[i] if vel < -0.001 else 0)
        Fv_term = params.B[i] * vel
        stribeck = params.Fs[i] * np.exp(-(abs_vel / params.vs[i])**2) * np.sign(vel) if (1e-6 < abs_vel < 3 * params.vs[i]) else 0
        tau_f[i] = Fc * np.sign(vel) + Fv_term + stribeck if abs_vel > 0.001 else 0
    return tau_f

# ============================================================================
# Algorithm 1: PID Control
# ============================================================================
class FastPIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp, self.Ki, self.Kd = Kp, Ki, Kd
        self.dt = dt
        self.e_int = np.zeros(6)  # Line 1: Initialize e_int
        self.e_max = 10.0
    
    def compute(self, q_ref, q, qd_ref, qd):
        e = q_ref - q               # Line 4: Compute tracking error
        ed = qd_ref - qd            # Line 5: Compute velocity error
        self.e_int += e * self.dt   # Line 6: Update integral error
        self.e_int = np.clip(self.e_int, -self.e_max, self.e_max) # Line 7: Anti-windup
        tau_pid = self.Kp @ e + self.Ki @ self.e_int + self.Kd @ ed # Line 8
        return tau_pid, e, ed

# ============================================================================
# Algorithm 4: Lyapunov-Based Safety Cage
# ============================================================================
class LyapunovSafetyCage:
    def __init__(self, Kd, Ki):
        self.Lambda = np.diag([8.0, 10.0, 8.0, 5.0, 5.0, 4.0])
        self.Kd, self.Ki = Kd, Ki
        self.Pmax = 50.0
        self.beta = 0.1
        self.eps = 1e-6
        self.tau_max = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
        
    def apply(self, tau_rl_raw, tau_pid, e, ed, alpha, qd, M_q):
        # 1-3: Sliding surface and Lyapunov
        s = ed + self.Lambda @ e
        V_t = 0.5 * (s.T @ M_q @ s) + 0.5 * (e.T @ self.Ki @ e)
        
        tau_pid_norm = np.linalg.norm(tau_pid)
        tau_rl_norm = np.linalg.norm(tau_rl_raw)
        
        # 4-10: 1. Magnitude Bound (Primary)
        if tau_rl_norm > alpha * tau_pid_norm:
            tau_safe = tau_rl_raw * (alpha * tau_pid_norm) / (tau_rl_norm + self.eps)
        else:
            tau_safe = tau_rl_raw.copy()
            
        # 11-15: 2. Direction Agreement Check
        if np.dot(tau_safe, tau_pid) < -self.beta * (tau_pid_norm**2):
            proj = np.dot(tau_safe, tau_pid) / (tau_pid_norm**2 + self.eps)
            tau_safe = tau_safe - proj * tau_pid
            
        # 16-20: 3. Energy Bound
        P_RL = abs(np.dot(tau_safe, qd))
        if P_RL > self.Pmax:
            tau_safe = tau_safe * (self.Pmax / (P_RL + self.eps))
            
        # 21-26: 4. Lyapunov Stability Condition Enforcement
        lambda_min = np.min(np.diag(self.Kd))
        Delta_max = 1.0
        numerator = lambda_min * np.linalg.norm(s) - Delta_max
        alpha_max = max(0.0, numerator / (tau_pid_norm + self.eps))
        
        if alpha_max < alpha:
            tau_safe = tau_safe * (alpha_max / (alpha + self.eps))
            
        # 27: Apply torque limits
        tau_safe = np.clip(tau_safe, -self.tau_max, self.tau_max)
        return tau_safe, V_t

# ============================================================================
# Environment (Algorithm 2 Inner Loops)
# ============================================================================
class Puma560EnvTD3:
    def __init__(self, dt=0.001, window_size=20, rl_decimation=20):
        self.dt = dt 
        self.rl_decimation = rl_decimation 
        self.window_size = window_size
        self.params = PUMA560Params()
        self.robot = models.DH.Puma560()
        
        # Setting A baseline
        Kp = np.diag([800.0, 1200.0, 800.0, 200.0, 200.0, 100.0])
        Ki = np.diag([50.0, 80.0, 50.0, 10.0, 10.0, 5.0])
        Kd = np.diag([40.0, 60.0, 40.0, 15.0, 15.0, 8.0])
        self.pid = FastPIDController(Kp, Ki, Kd, dt)
        self.safety_cage = LyapunovSafetyCage(Kd, Ki)
        
        self.q = np.zeros(6)
        self.qd = np.zeros(6)
        self.t_total = 0.0
        
    def _get_reference(self, t):
        # Dual-tone training trajectory matching research defaults
        q_ref = 0.5 * np.sin(2*np.pi*0.5*t)*np.ones(6) + 0.2 * np.sin(2*np.pi*0.3*t)*np.ones(6)
        qd_ref = 0.5*np.pi*np.cos(np.pi*t)*np.ones(6) + 0.12*np.pi*np.cos(0.6*np.pi*t)*np.ones(6)
        qdd_ref = -0.5*(np.pi**2)*np.sin(np.pi*t)*np.ones(6) - 0.072*(np.pi**2)*np.sin(0.6*np.pi*t)*np.ones(6)
        return q_ref, qd_ref, qdd_ref

    def get_M(self):
        M = np.empty((6, 6))
        for j in range(6):
            qdd_dummy = np.zeros(6)
            qdd_dummy[j] = 1.0
            M[:, j] = self.robot.rne(self.q, np.zeros(6), qdd_dummy, gravity=[0, 0, 0])
        return M

    def reset(self):
        self.q = np.zeros(6)
        self.qd = np.zeros(6)
        self.t_total = 0.0
        self.pid.e_int = np.zeros(6)
        self.S = []
        
        # Warmup buffer to fill T=20
        for _ in range(self.window_size):
            s_t, _, _, _ = self.execute_inner_loop(np.zeros(6))
        return np.array(self.S)

    def execute_inner_loop(self, tau_rl_safe):
        """Algorithm 2: Lines 4-8 (1 kHz updates inside 20ms RL step)"""
        for _ in range(self.rl_decimation):
            self.t_total += self.dt
            q_ref, qd_ref, qdd_ref = self._get_reference(self.t_total)
            
            tau_pid, e, ed = self.pid.compute(q_ref, self.q, qd_ref, self.qd)
            tau_total = tau_pid + tau_rl_safe # Line 6: Total torque
            
            # Physics Step (Line 7)
            tau_fric = compute_friction(self.qd, self.params)
            C_G = self.robot.rne(self.q, self.qd, np.zeros(6))
            M = self.get_M()
            qdd = np.linalg.inv(M) @ (tau_total - C_G - tau_fric)
            self.qd += qdd * self.dt
            self.q = np.clip(self.q + self.qd * self.dt, self.params.q_limits[:,0], self.params.q_limits[:,1])
            
        # Line 9: Build instantaneous state
        s_t = np.concatenate([self.q, self.qd, e, ed, self.pid.e_int, qdd_ref])
        
        # Line 10: Append and trim
        self.S.append(s_t)
        if len(self.S) > self.window_size:
            self.S.pop(0)
            
        return s_t, tau_pid, e, ed

    def compute_reward(self, e_final, ed_final, tau_residual, tau_total):
        # Normalized tracking reward to ensure stable gradients
        # Weights derived from Inverse Baseline PID MSE to guarantee equal learning distribution
        weights = np.array([0.0725, 0.1059, 0.4290, 2.4099, 2.1500, 0.8327])
        w = weights / np.mean(weights)
        
        r_track = -1.0 * np.sum(w * (e_final**2))
        r_energy = -0.01 * np.sum((tau_residual/self.params.tau_max)**2) - 0.001 * np.sum((tau_total/self.params.tau_max)**2)
        r_stab = 1.0 if np.max(np.abs(e_final)) < 0.01 else 0.0
        
        reward = float(np.clip(r_track + r_energy + r_stab, -20.0, 20.0))
        truncated = np.max(np.abs(e_final)) > 3.0
        if truncated: reward -= 20.0
        return reward, truncated