import numpy as np
from numpy import float64 as f64
import matplotlib.pyplot as plt
from roboticstoolbox import models
from numpy.typing import NDArray


class PIDController:
    def __init__(self, Kp: f64, Ki: f64, Kd: f64, setpoint: f64 = f64(0)) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.integral = 0.0
        self.prev_error = 0.0

    def update(self, measurement: f64, dt: f64) -> f64:
        error = self.setpoint - measurement
        self.integral += error * dt
        self.integral = max(min(self.integral, 1e6), -1e6)
        derivative = (error - self.prev_error) / dt
        derivative = max(min(derivative, 1e6), -1e6)
        self.prev_error = error
        return (self.Kp * error) + (self.Ki * self.integral) + (self.Kd * derivative)


# Load PUMA 560 model
robot = models.DH.Puma560()

# Initial joint configuration (radians)
q: NDArray[f64] = np.zeros(6)
qd: NDArray[f64] = np.zeros(6)

# Setpoints
setpoints: NDArray[f64] = np.deg2rad([0, 45, 0, 0, 0, 0])

# One PID per joint
pids: list[PIDController] = [
    PIDController(Kp=f64(40), Ki=f64(5), Kd=f64(15), setpoint=sp) for sp in setpoints
]

# Simulation parameters
t_steps: NDArray[f64] = np.linspace(0, 10, 1000)
dt: f64 = t_steps[1] - t_steps[0]

# Data storage
q_values: list[list[f64]] = [[] for _ in range(6)]
u_values: list[list[f64]] = [[] for _ in range(6)]

# ---------------- Simulation Loop ----------------
for _ in t_steps:
    # Robot dynamics terms
    M: NDArray[f64] = robot.inertia(q)
    C: NDArray[f64] = robot.coriolis(q, qd)
    G: NDArray[f64] = robot.gravload(q)

    # PID torques (feedback)
    tau_pid: NDArray[f64] = np.array(
        [pids[i].update(q[i], dt) for i in range(6)])

    # --- Gravity compensation (feed-forward) ---
    tau_vector: NDArray[f64] = tau_pid + G

    # Forward dynamics: M qdd = tau - C qd - G
    qdd: NDArray[f64] = np.linalg.inv(M) @ (tau_vector - C @ qd - G)

    # Integrate
    qd += qdd * dt
    q += qd * dt

    # Log
    for i in range(6):
        q_values[i].append(np.rad2deg(q[i]))
        u_values[i].append(tau_vector[i])

# ---------------- Plot: angle+setpoint (top), torque (bottom) ----------------
joint_to_plot = 5  # 0..5

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

# Top: angle & setpoint
ax1.plot(t_steps, q_values[joint_to_plot], 'b',
         label=f"Joint {joint_to_plot + 1} Angle")
ax1.axhline(np.rad2deg(setpoints[joint_to_plot]),
            color='r', linestyle='--', label="Setpoint")
ax1.set_ylabel("Angle (deg)")
ax1.set_title(f"Joint {joint_to_plot + 1} Tracking")
ax1.legend(loc="upper right")
ax1.grid()

# Bottom: torque
ax2.plot(t_steps, u_values[joint_to_plot], 'g', label="Torque")
ax2.set_xlabel("Time (s)")
ax2.set_ylabel("Torque (Nm)")
ax2.legend(loc="upper right")
ax2.grid()

plt.tight_layout()
plt.show()
