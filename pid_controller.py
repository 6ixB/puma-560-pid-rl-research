from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from roboticstoolbox import models


@dataclass
class PIDValue:
    Kp: f64
    Ki: f64
    Kd: f64


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


def run_pid_controller(setpoints: NDArray[f64], pid_values: list[PIDValue]):
    robot = models.DH.Puma560()

    q: NDArray[f64] = np.zeros(6)
    qd: NDArray[f64] = np.zeros(6)

    pids: list[PIDController] = [
        PIDController(
            Kp=f64(pid_values[i].Kp),
            Ki=f64(pid_values[i].Ki),
            Kd=f64(pid_values[i].Kd),
            setpoint=setpoints[i],
        )
        for i in range(len(setpoints))
    ]

    t_steps: NDArray[f64] = np.linspace(0, 10, 1000)
    dt: f64 = t_steps[1] - t_steps[0]

    q_values: list[list[f64]] = [[] for _ in range(6)]
    u_values: list[list[f64]] = [[] for _ in range(6)]

    # ---------------- Simulation Loop ----------------
    for _ in t_steps:
        M: NDArray[f64] = robot.inertia(q)  # pyright: ignore[reportAttributeAccessIssue]
        C: NDArray[f64] = robot.coriolis(q, qd)  # pyright: ignore[reportAttributeAccessIssue]
        G: NDArray[f64] = robot.gravload(q)  # pyright: ignore[reportAttributeAccessIssue]

        tau_pid: NDArray[f64] = np.array([pids[i].update(q[i], dt) for i in range(6)])

        tau_vector: NDArray[f64] = tau_pid
        # tau_vector: NDArray[f64] = tau_pid + G

        qdd: NDArray[f64] = np.linalg.inv(M) @ (tau_vector - C @ qd - G)

        qd += qdd * dt
        q += qd * dt

        for i in range(6):
            q_values[i].append(np.rad2deg(q[i]))
            u_values[i].append(tau_vector[i])

    return t_steps, q_values, u_values


def plot_pid_controller_output(t_steps, q_values, u_values, setpoints):
    # ---------------- Plot: angle+setpoint (top), torque (bottom) ----------------
    joint_to_plot = 5  # 0..5

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8), sharex=True)

    # Top: angle & setpoint
    ax1.plot(
        t_steps, q_values[joint_to_plot], "b", label=f"Joint {joint_to_plot + 1} Angle"
    )
    ax1.axhline(
        np.rad2deg(setpoints[joint_to_plot]),
        color="r",
        linestyle="--",
        label="Setpoint",
    )
    ax1.set_ylabel("Angle (deg)")
    ax1.set_title(f"Joint {joint_to_plot + 1} Tracking")
    ax1.legend(loc="upper right")
    ax1.grid()

    # Bottom: torque
    ax2.plot(t_steps, u_values[joint_to_plot], "g", label="Torque")
    ax2.set_xlabel("Time (s)")
    ax2.set_ylabel("Torque (Nm)")
    ax2.legend(loc="upper right")
    ax2.grid()

    plt.tight_layout()
    plt.show()
