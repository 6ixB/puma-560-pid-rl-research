import numpy as np
from roboticstoolbox import models

# ============================================================================
# PUMA 560 Dynamic Parameters (6 Joints)
# ============================================================================

class PUMA560Params:
    """Dynamic parameters for PUMA 560 all 6 joints"""
    
    J = np.array([4.0, 6.0, 4.5, 1.5, 1.0, 0.8])
    B = np.array([0.06, 0.08, 0.07, 0.04, 0.04, 0.03])
    
    Fc_pos = np.array([0.60, 0.80, 0.70, 0.40, 0.40, 0.30])
    Fc_neg = np.array([0.65, 0.85, 0.75, 0.42, 0.42, 0.32])
    
    vs = np.array([0.005, 0.006, 0.005, 0.004, 0.004, 0.003])
    Fs = np.array([0.20, 0.25, 0.22, 0.15, 0.15, 0.12])
    
    G_coeff = np.array([0.0, 50.0, 30.0, 0.0, 0.0, 0.0])
    tau_max = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
    
    q_limits = np.array([
        [-2.967, 2.967],
        [-0.785, 2.356],
        [-2.356, 2.356],
        [-3.142, 3.142],
        [-2.094, 2.094],
        [-3.142, 3.142]
    ])

def compute_friction(qd, params):
    tau_f = np.zeros(6)
    for i in range(6):
        vel = qd[i]
        abs_vel = abs(vel)
        if vel > 0.001:
            Fc = params.Fc_pos[i]
        elif vel < -0.001:
            Fc = params.Fc_neg[i]
        else:
            Fc = 0
            
        Fv_term = params.B[i] * vel
        
        if abs_vel < 3 * params.vs[i] and abs_vel > 1e-6:
            stribeck = params.Fs[i] * np.exp(-(abs_vel / params.vs[i])**2) * np.sign(vel)
        else:
            stribeck = 0
            
        tau_f[i] = Fc * np.sign(vel) + Fv_term + stribeck if abs_vel > 0.001 else 0
    return tau_f

class FastPIDController:
    def __init__(self, Kp, Ki, Kd, dt):
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.dt = dt
        self.integral = np.zeros(6)
        self.integral_limit = 10.0
    
    def compute(self, e, ed):
        self.integral += e * self.dt
        self.integral = np.clip(self.integral, -self.integral_limit, self.integral_limit)
        return self.Kp @ e + self.Ki @ self.integral + self.Kd @ ed
    
    def reset(self):
        self.integral = np.zeros(6)

class MotorDynamics:
    def __init__(self, Ra, La, Kt, Kb, G, Imax, dt):
        self.Ra, self.La, self.Kt, self.Kb, self.G, self.Imax, self.dt = Ra, La, Kt, Kb, G, Imax, dt
        self.current = 0.0
    
    def compute_torque(self, voltage, omega_joint):
        omega_motor = self.G * omega_joint
        v_bemf = self.Kb * omega_motor
        di_dt = ((voltage - v_bemf) - self.Ra * self.current) / self.La
        self.current = np.clip(self.current + di_dt * self.dt, -self.Imax, self.Imax)
        return self.Kt * self.current * self.G
    
    def compute_voltage(self, desired_torque, omega_joint):
        req_i = desired_torque / (self.Kt * self.G)
        v = self.Ra * req_i + self.Kb * self.G * omega_joint
        return np.clip(v, -30.0, 30.0)
    
    def reset(self):
        self.current = 0.0

class SafetyCage:
    def __init__(self, Kd, Ki, Kp):
        self.Lambda = np.diag([8.0, 10.0, 8.0, 5.0, 5.0, 4.0])
        self.lambda_min_Kd = np.min(np.diag(Kd))
        self.Delta_max = 1.0
        self.Pmax = 50.0
        self.alpha = 0.30
        self.beta = 0.1
    
    def update_alpha(self, episode, total_episodes=300, critic_uncertainty=0.0):
        # Disabled Curriculum: alpha is constant
        self.alpha = 0.30
        return self.alpha
        
    def compute_alpha_max(self, s, tau_pid):
        s_norm, tau_norm = np.linalg.norm(s), np.linalg.norm(tau_pid)
        if s_norm < 1e-6 or tau_norm < 1e-6: return 1.0
        num = self.lambda_min_Kd * s_norm - self.Delta_max
        return 0.0 if num <= 0 else np.clip(num / tau_norm, 0.0, 1.0)
        
    def apply(self, tau_raw, tau_pid, e, ed):
        tau_safe = tau_raw.copy()
        s = ed + self.Lambda @ e
        tau_pid_norm = np.linalg.norm(tau_pid)
        
        if tau_pid_norm > 1e-6:
            tau_norm = np.linalg.norm(tau_safe)
            if tau_norm > self.alpha * tau_pid_norm:
                tau_safe = tau_safe * (self.alpha * tau_pid_norm) / tau_norm
                
        if np.dot(tau_safe, tau_pid) < -self.beta * tau_pid_norm**2:
            tau_safe -= (np.dot(tau_safe, tau_pid) / (tau_pid_norm**2 + 1e-6)) * tau_pid
            
        power = abs(np.dot(tau_safe, ed))
        if power > self.Pmax:
            tau_safe *= (self.Pmax / power)
            
        a_max = self.compute_alpha_max(s, tau_pid)
        if a_max < self.alpha:
            tau_safe *= (a_max / self.alpha)
            
        return np.clip(tau_safe, -PUMA560Params.tau_max, PUMA560Params.tau_max)

class Puma560Env:
    """
    Refactored Environment for multi-rate RL.
    1ms PID, 5ms LSTM History Sampling, 10ms RL Step.
    Returns: (history_buffer_for_lstm, state_t, reward, done)
    """
    def __init__(self, dt=0.001, window_size=20, rl_decimation=10, lstm_decimation=5, baseline_setting='A', trajectory=None):
        self.dt = dt # 1ms
        self.rl_decimation = rl_decimation # 10 steps = 10ms
        self.lstm_decimation = lstm_decimation # 5 steps = 5ms
        self.window_size = window_size
        self.trajectory = trajectory
        
        self.params = PUMA560Params()
        self.robot = models.DH.Puma560()
        
        if baseline_setting == 'A':
            self.Kp = np.diag([800.0, 1200.0, 800.0, 200.0, 200.0, 100.0])
            self.Ki = np.diag([50.0, 80.0, 50.0, 10.0, 10.0, 5.0])
            self.Kd = np.diag([40.0, 60.0, 40.0, 15.0, 15.0, 8.0])
        elif baseline_setting == 'B':
            self.Kp = np.diag([652.4, 978.6, 652.4, 163.1, 163.1, 81.15])
            self.Ki = np.diag([41.2, 61.8, 41.2, 10.3, 10.3, 5.1])
            self.Kd = np.diag([32.6, 48.9, 32.6, 8.2, 8.2, 4.1])
        else:
            self.Kp = np.diag([521.9, 782.9, 521.9, 130.5, 130.5, 65.2])
            self.Ki = np.diag([104.4, 156.6, 104.4, 26.1, 26.1, 13.0])
            self.Kd = np.diag([26.1, 39.1, 26.1, 6.5, 6.5, 3.3])
            
        self.pid = FastPIDController(self.Kp, self.Ki, self.Kd, dt)
        self.safety = SafetyCage(self.Kd, self.Ki, self.Kp)
        
        motor_params = [
            (1.2, 0.002, 0.35, 0.35, 62.6, 10.0), (1.3, 0.0025, 0.38, 0.38, 62.6, 10.0),
            (1.4, 0.003, 0.40, 0.40, 62.6, 10.0), (1.5, 0.003, 0.42, 0.42, 62.6, 8.0),
            (1.6, 0.0035, 0.45, 0.45, 62.6, 8.0), (1.7, 0.0035, 0.48, 0.48, 62.6, 6.0)
        ]
        self.motors = [MotorDynamics(*p, dt) for p in motor_params]
        
        self.q = np.zeros(6)
        self.qd = np.zeros(6)
        self.e_int = np.zeros(6)
        self.tau_residual_prev = np.zeros(6)
        self.tau_raw_prev = np.zeros(6)
        self.step_count = 0
        self.episode_length = 0
        
        self.q_ref = np.zeros(6)
        self.qd_ref = np.zeros(6)
        
        self.history = np.zeros((self.window_size, 44), dtype=np.float32)

    def _update_reference(self, t):
        if self.trajectory is not None:
            q_ref_new = self.trajectory.get_setpoint(t)
            if t == 0:
                self.qd_ref = np.zeros(6)
            else:
                self.qd_ref = (q_ref_new - getattr(self, 'q_ref', q_ref_new)) / self.dt
            self.q_ref = q_ref_new
        else:
            # Randomized training trajectory
            traj_type = getattr(self, 'traj_type', 'sine')
            if traj_type == 'sine':
                amp = getattr(self, 'traj_amp', 0.5)
                freq = getattr(self, 'traj_freq', 1.0)
                phase = getattr(self, 'traj_phase', 0.0)
                
                q_ref_val = amp * np.sin(np.pi * freq * t + phase) + 0.2 * np.sin(0.6 * np.pi * freq * t + phase)
                qd_ref_val = amp * np.pi * freq * np.cos(np.pi * freq * t + phase) + 0.2 * 0.6 * np.pi * freq * np.cos(0.6 * np.pi * freq * t + phase)
                self.q_ref = np.ones(6) * q_ref_val
                self.qd_ref = np.ones(6) * qd_ref_val
            elif traj_type == 'static':
                self.q_ref = getattr(self, 'traj_target', np.zeros(6))
                self.qd_ref = np.zeros(6)
            elif traj_type == 'step':
                step_time = getattr(self, 'traj_step_time', 2.5)
                if t < step_time:
                    self.q_ref = getattr(self, 'traj_target_initial', np.zeros(6))
                else:
                    self.q_ref = getattr(self, 'traj_target_final', np.zeros(6))
                self.qd_ref = np.zeros(6)

    def _get_state(self):
        e = self.q_ref - self.q
        ed = self.qd_ref - self.qd
        
        t_curr = self.step_count * self.rl_decimation * self.dt
        phase_sin = np.sin(2 * np.pi * 0.5 * t_curr)
        phase_cos = np.cos(2 * np.pi * 0.5 * t_curr)
        
        # Manually scale inputs to sit roughly between -1.0 and 1.0
        norm_q = self.q / np.pi
        norm_qd = self.qd / 5.0
        norm_e = e / 1.0 
        norm_ed = ed / 5.0
        norm_e_int = self.e_int / 10.0
        norm_q_ref = self.q_ref / np.pi
        norm_tau_prev = self.tau_residual_prev / 20.0
        
        state = np.concatenate([norm_q, norm_qd, norm_e, norm_ed, norm_e_int, norm_q_ref, norm_tau_prev, [phase_sin, phase_cos]])
        return state.astype(np.float32)

    def _dynamics(self, state, tau):
        q, qd = state[:6], state[6:12]
        tau_fric = compute_friction(qd, self.params)
        
        # 1. Coriolis + Gravity bias term calculated in 1 single low-level RNE call (zero allocation & leak-free)
        C_qd_plus_G = self.robot.rne(q, qd, np.zeros(6))
        
        # 2. Inertia matrix M calculated column-by-column using direct RNE calls to bypass high-level memory leaks
        M = np.empty((6, 6))
        for j in range(6):
            qdd_dummy = np.zeros(6)
            qdd_dummy[j] = 1.0
            M[:, j] = self.robot.rne(q, np.zeros(6), qdd_dummy, gravity=[0, 0, 0])
            
        qdd = np.linalg.inv(M) @ (tau - C_qd_plus_G - tau_fric)
        return np.concatenate([qd, qdd])

    def reset(self, episode=0):
        self.q = np.zeros(6)
        self.qd = np.zeros(6)
        self.e_int = np.zeros(6)
        self.tau_residual_prev = np.zeros(6)
        self.tau_raw_prev = np.zeros(6)
        self.step_count = 0
        self.episode_length = 0
        
        self.pid.reset()
        for m in self.motors: m.reset()
        
        # Randomize trajectory params
        self.traj_type = np.random.choice(['sine', 'static', 'step'])
        if self.traj_type == 'sine':
            self.traj_amp = np.random.uniform(0.3, 0.7)
            self.traj_freq = np.random.uniform(0.8, 1.2)
            self.traj_phase = np.random.uniform(0, 2*np.pi)
        elif self.traj_type == 'static':
            self.traj_target = np.random.uniform(-1.0, 1.0, size=6)
        elif self.traj_type == 'step':
            self.traj_target_initial = np.random.uniform(-1.0, 1.0, size=6)
            self.traj_target_final = np.random.uniform(-1.0, 1.0, size=6)
            self.traj_step_time = np.random.uniform(1.0, 4.0)
        
        self._update_reference(0.0)
        self.q = np.copy(self.q_ref)  # Spawn robot exactly on the trajectory start point
        s0 = self._get_state()
        self.history = np.tile(s0, (self.window_size, 1))
        
        return np.copy(self.history)

    def step(self, tau_residual_raw, episode=0, critic_uncertainty=0.0):
        # Safety cage acts once per RL step (10ms)
        e_init = self.q_ref - self.q
        ed_init = self.qd_ref - self.qd
        tau_pid_init = self.pid.compute(e_init, ed_init)
        
        # Inverse Dynamics Feedforward
        tau_ff_init = self.params.G_coeff * np.sin(self.q) + compute_friction(self.qd, self.params) + 0.05 * np.abs(self.qd) * self.qd
        tau_baseline_init = tau_pid_init + tau_ff_init
        
        self.safety.update_alpha(episode, 300, critic_uncertainty=critic_uncertainty)
        # Pass critic_uncertainty if provided via kwargs (default 0 for now)
        tau_residual = self.safety.apply(tau_residual_raw, tau_baseline_init, e_init, ed_init)
        self.tau_residual_prev = tau_residual
        
        total_tau_pid = np.zeros(6)
        
        # Inner loop: 10 steps of 1ms each
        for i in range(1, self.rl_decimation + 1):
            t_curr = (self.step_count * self.rl_decimation + i) * self.dt
            self._update_reference(t_curr)
            
            e_sub = self.q_ref - self.q
            ed_sub = self.qd_ref - self.qd
            tau_pid_sub = self.pid.compute(e_sub, ed_sub)
            total_tau_pid += tau_pid_sub
            
            tau_ff_sub = self.params.G_coeff * np.sin(self.q) + compute_friction(self.qd, self.params) + 0.05 * np.abs(self.qd) * self.qd
            tau_total = tau_pid_sub + tau_ff_sub + tau_residual
            
            for j, motor in enumerate(self.motors):
                v = motor.compute_voltage(tau_total[j], self.qd[j])
                motor.compute_torque(v, self.qd[j])
                
            qdd = self._dynamics(np.concatenate([self.q, self.qd]), tau_total)[6:12]
            self.qd += qdd * self.dt
            self.q = np.clip(self.q + self.qd * self.dt, self.params.q_limits[:, 0], self.params.q_limits[:, 1])
            
            # LSTM samples every 5ms
            if i % self.lstm_decimation == 0:
                self.e_int = np.clip(self.e_int + e_sub * (self.dt * self.lstm_decimation), -10.0, 10.0)
                self.history[:-1] = self.history[1:]
                self.history[-1] = self._get_state()
                
        self.step_count += 1
        self.episode_length += 1
        
        e_final = self.q_ref - self.q
        ed_final = self.qd_ref - self.qd
        
        # Curriculum Penalty Decay Disabled (Environment is frozen at final difficulty)
        w_energy = 0.001
        w_jerk = 0.1
        
        # ====================================================================
        # REWARD FUNCTION BREAKDOWN
        # ====================================================================
        
        # 1. Tracking Reward (r_track)
        # Penalizes the agent for failing to follow the reference trajectory.
        # - Weight tracking errors inversely to their baseline MSE to balance multi-joint learning.
        # - We normalize the weights so their mean is 1.0. This prevents the total reward magnitude
        # - from blowing up and hitting the [-20.0, 20.0] clip limit, which destroys learning gradients.
        raw_weights = np.array([2.5, 1.0, 5.5, 75.0, 82.0, 18.0])
        joint_weights = raw_weights / np.mean(raw_weights)
        r_track = -10.0 * np.sum(joint_weights * (e_final**2)) - 1.0 * np.sum(joint_weights * (ed_final**2)) - 0.5 * np.sum(joint_weights * (self.e_int**2))
        
        # 2. Energy Reward (r_energy)
        # Penalizes the agent for using excessive torque/power.
        # - Normalize torques so the energy penalty is proportional to the joint's maximum capability.
        # - This ensures tiny wrist joints aren't disproportionately punished compared to massive shoulder joints.
        max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
        norm_tau = tau_residual / max_action
        norm_total_tau = (tau_residual + total_tau_pid/10.0) / max_action
        r_energy = -w_energy * np.sum(norm_tau**2) - (w_energy/10.0) * np.sum(norm_total_tau**2)
        
        # 3. Smoothness Reward (r_smoothness)
        # Penalizes "jerk" or sudden massive changes in torque from one timestep to the next.
        # - Also normalized by max_action to ensure fair punishment across all joints.
        norm_tau_prev = self.tau_residual_prev / max_action
        r_smoothness = -w_jerk * np.sum((norm_tau - norm_tau_prev)**2)
        
        # 4. Stability Reward (r_stability)
        # Gives a flat +1.0 bonus if the agent perfectly tracks the trajectory (all errors < 0.01 radians).
        r_stability = 1.0 if np.max(np.abs(e_final)) < 0.01 else 0.0
        
        # 5. Action L2 Regularization Penalty
        r_action_l2 = -0.01 * np.sum((tau_residual_raw / max_action)**2)
        
        reward = float(np.clip(r_track + r_energy + r_smoothness + r_stability + r_action_l2, -20.0, 20.0))
        
        terminated = self.episode_length >= 500
        truncated = np.max(np.abs(e_final)) > 3.0
        if truncated: reward -= 10.0
        elif terminated: reward += 10.0
            
        return np.copy(self.history), reward, terminated, truncated, {'error': np.linalg.norm(e_final), 'actual_action': tau_residual}

class Puma560EnvTD3(Puma560Env):
    """
    Environment subclass specifically for TD3, perfectly mimicking the state
    truncation found in the research script (s_t[:42]). This uses qd_ref instead
    of tau_residual_prev. It also overrides the decimation and physics step 
    to match the research script's lack of feedforward torque.
    """
    def __init__(self, **kwargs):
        # Force research parameters
        kwargs['rl_decimation'] = 20
        kwargs['lstm_decimation'] = 20
        super().__init__(**kwargs)

    def _get_state(self):
        e = self.q_ref - self.q
        ed = self.qd_ref - self.qd
        # Research script concatenates [q, qd, e, ed, e_int, q_ref, qd_ref, tau_residual_prev] 
        # and then takes [:42], which exactly captures everything up to qd_ref (6 * 7 = 42).
        
        t_curr = self.step_count * self.rl_decimation * self.dt
        phase_sin = np.sin(2 * np.pi * 0.5 * t_curr)
        phase_cos = np.cos(2 * np.pi * 0.5 * t_curr)
        
        # Manually scale inputs to sit roughly between -1.0 and 1.0
        norm_q = self.q / np.pi
        norm_qd = self.qd / 5.0
        norm_e = e / 1.0
        norm_ed = ed / 5.0
        norm_e_int = self.e_int / 10.0
        norm_q_ref = self.q_ref / np.pi
        norm_qd_ref = getattr(self, 'qd_ref', np.zeros(6)) / 5.0
        
        state = np.concatenate([
            norm_q, 
            norm_qd, 
            norm_e, 
            norm_ed, 
            norm_e_int, 
            norm_q_ref, 
            norm_qd_ref,
            [phase_sin, phase_cos]
        ])
        return state.astype(np.float32)

    def step(self, tau_residual_raw, episode=0, **kwargs):
        e_init = self.q_ref - self.q
        ed_init = self.qd_ref - self.qd
        tau_pid_init = self.pid.compute(e_init, ed_init)
        
        # RESEARCH: No tau_ff. Safety cage gets raw pid.
        self.safety.update_alpha(episode, 300) # Simple ramp, no critic_uncertainty penalty
        tau_residual = self.safety.apply(tau_residual_raw, tau_pid_init, e_init, ed_init)
        self.tau_residual_prev = tau_residual
        
        total_tau_pid = np.zeros(6)
        
        for i in range(1, self.rl_decimation + 1):
            t_curr = (self.step_count * self.rl_decimation + i) * self.dt
            self._update_reference(t_curr)
            
            e_sub = self.q_ref - self.q
            ed_sub = self.qd_ref - self.qd
            tau_pid_sub = self.pid.compute(e_sub, ed_sub)
            total_tau_pid += tau_pid_sub
            
            # RESEARCH: No tau_ff
            tau_total = tau_pid_sub + tau_residual
            
            for j, motor in enumerate(self.motors):
                v = motor.compute_voltage(tau_total[j], self.qd[j])
                motor.compute_torque(v, self.qd[j])
                
            qdd = self._dynamics(np.concatenate([self.q, self.qd]), tau_total)[6:12]
            self.qd += qdd * self.dt
            self.q = np.clip(self.q + self.qd * self.dt, self.params.q_limits[:, 0], self.params.q_limits[:, 1])
            
            # LSTM sampling
            if i % self.lstm_decimation == 0:
                self.e_int = np.clip(self.e_int + e_sub * (self.dt * self.lstm_decimation), -10.0, 10.0) # RESEARCH: limit is 10.0
                self.history[:-1] = self.history[1:]
                self.history[-1] = self._get_state()
                
        self.step_count += 1
        self.episode_length += 1
        
        e_final = self.q_ref - self.q
        ed_final = self.qd_ref - self.qd
        
        # RESEARCH: Static reward weights
        w_energy = 0.01
        w_total = 0.001
        w_jerk = 0.1
        
        # ====================================================================
        # REWARD FUNCTION BREAKDOWN
        # ====================================================================
        
        # 1. Tracking Reward (r_track)
        # Penalizes the agent for failing to follow the reference trajectory.
        # - Weight tracking errors inversely to their baseline MSE to balance multi-joint learning.
        raw_weights = np.array([2.5, 1.0, 5.5, 75.0, 82.0, 18.0])
        joint_weights = raw_weights / np.mean(raw_weights)
        r_track = -10.0 * np.sum(joint_weights * (e_final**2)) - 1.0 * np.sum(joint_weights * (ed_final**2)) - 0.5 * np.sum(joint_weights * (self.e_int**2))
        
        # 2. Energy Reward (r_energy)
        # Penalizes the agent for using excessive torque/power.
        # - Normalize torques so the energy penalty is proportional to the joint's maximum capability.
        # - This ensures tiny wrist joints aren't disproportionately punished compared to massive shoulder joints.
        max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
        norm_tau = tau_residual / max_action
        norm_total_tau = (tau_residual + total_tau_pid/self.rl_decimation) / max_action
        r_energy = -w_energy * np.sum(norm_tau**2) - w_total * np.sum(norm_total_tau**2)
        
        # 3. Smoothness Reward (r_smoothness)
        # Penalizes "jerk" or sudden massive changes in torque from one timestep to the next.
        # - Also normalized by max_action to ensure fair punishment across all joints.
        norm_tau_raw_prev = self.tau_raw_prev / max_action
        norm_tau_raw = tau_residual_raw / max_action
        r_smoothness = -w_jerk * np.sum((norm_tau_raw - norm_tau_raw_prev)**2)
        
        # 4. Stability Reward (r_stability)
        # Gives a flat +1.0 bonus if the agent perfectly tracks the trajectory (all errors < 0.01 radians).
        r_stability = 1.0 if np.max(np.abs(e_final)) < 0.01 else 0.0
        
        # 5. Cage Penalty
        w_cage = 0.05
        r_cage = -w_cage * np.sum((norm_tau_raw - norm_tau)**2)
        
        # 6. Action L2 Regularization Penalty
        r_action_l2 = -0.01 * np.sum((tau_residual_raw / max_action)**2)
        
        reward = float(np.clip(r_track + r_energy + r_smoothness + r_stability + r_cage + r_action_l2, -20.0, 20.0))
        
        terminated = self.episode_length >= 500
        truncated = np.max(np.abs(e_final)) > 3.0
        if truncated: reward -= 10.0
        elif terminated: reward += 10.0
            
        self.tau_raw_prev = tau_residual_raw
        return np.copy(self.history), reward, terminated, truncated, {'error': np.linalg.norm(e_final), 'actual_action': tau_residual}
