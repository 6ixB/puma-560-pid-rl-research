import numpy as np
import roboticstoolbox as rtb
from pid_controller import PIDController

class Puma560Env:
    """
    A unified Reinforcement Learning environment wrapping the Puma 560 Robot and PID controller.
    Facilitates step-by-step optimization using Deep Neural Networks.
    """
    def __init__(self, dt=0.01, window_size=10, max_steps=500):
        self.dt = dt
        self.window_size = window_size
        self.max_steps = max_steps
        self.robot = rtb.models.DH.Puma560()
        
    def reset(self, target_deg=None, randomize=True):
        """Resets the robot to the initial point and starts a new trajectory target."""
        if randomize:
            # Domain Randomization: prevent network overfitting to 0,0,0
            self.target_rad = np.random.uniform(-np.pi/2, np.pi/2, 6).astype(np.float64)
            self.q = np.random.uniform(-np.pi/2, np.pi/2, 6).astype(np.float64)
            self.qd = np.random.uniform(-0.1, 0.1, 6).astype(np.float64)
        else:
            if target_deg is None:
                self.target_rad = np.deg2rad(np.array([45, 90, -45, 30, 0, 0], dtype=np.float64))
            else:
                self.target_rad = np.deg2rad(np.array(target_deg, dtype=np.float64))
                
            self.q = np.zeros(6, dtype=np.float64)
            self.qd = np.zeros(6, dtype=np.float64)
        
        # Instantiate continuous PIDs with leak_rate=0.99 for leaky integral carry-over
        self.pids = [PIDController(Kp=0.0, Ki=0.0, Kd=0.0, setpoint=self.target_rad[i], leak_rate=0.99) for i in range(6)]
        self.current_step = 0
        
        # History buffer: features = [q (6), qd (6), error (6), integral (6)] = 24 elements
        self.history = np.zeros((self.window_size, 24), dtype=np.float32)
        
        self._update_history()
        return np.copy(self.history)
        
    def _update_history(self):
        error = self.target_rad - self.q
        integrals = np.array([pid.integral for pid in self.pids], dtype=np.float64)
        features = np.concatenate([self.q, self.qd, error, integrals]).astype(np.float32)
        
        # Shift history sequence backwards
        self.history[:-1] = self.history[1:]
        # Insert current step at the end of the history window
        self.history[-1] = features
        
    def step(self, pid_gains):
        """
        Executes a simulation step dt given 18 continuous PID variables.
        pid_gains array shape: (18,) corresponding to (Kp, Ki, Kd) * 6 joints
        """
        for i in range(6):
            self.pids[i].Kp = pid_gains[i * 3 + 0]
            self.pids[i].Ki = pid_gains[i * 3 + 1]
            self.pids[i].Kd = pid_gains[i * 3 + 2]
            
        # Get torques from PIDs
        tau_pid = np.array([self.pids[i].update(self.q[i], self.dt)[0] for i in range(6)])
        
        # Run forward dynamics (Euler Integration)
        M = self.robot.inertia(self.q)
        C = self.robot.coriolis(self.q, self.qd)
        G = self.robot.gravload(self.q)
        
        qdd = np.linalg.inv(M) @ (tau_pid - C @ self.qd - G)
        
        self.qd += qdd * self.dt
        self.q += self.qd * self.dt
        
        self.current_step += 1
        self._update_history()
        
        # Calculate Reward Structure
        error_rad = self.target_rad - self.q
        
        # Penalty parameters (Tweakable)
        tracking_penalty = np.sum(error_rad**2)
        energy_penalty = 0.001 * np.sum(tau_pid**2)
        
        reward = -tracking_penalty - energy_penalty
        done = self.current_step >= self.max_steps
        
        return np.copy(self.history), reward, done, {"error": np.linalg.norm(error_rad)}
