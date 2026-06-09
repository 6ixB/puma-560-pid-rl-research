from dataclasses import dataclass

import matplotlib.pyplot as plt
import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray
from roboticstoolbox import models

from trajectory import TrajectoryGenerator, StaticTrajectory


@dataclass
class PIDValue:
    Kp: f64
    Ki: f64
    Kd: f64


class PIDController:
    def __init__(self, Kp: f64, Ki: f64, Kd: f64, setpoint: f64 = f64(0), leak_rate: f64 = 1.0) -> None:
        self.Kp = Kp
        self.Ki = Ki
        self.Kd = Kd
        self.setpoint = setpoint
        self.leak_rate = leak_rate
        self.integral = 0.0
        self.prev_error = 0.0
        self.p_term = 0.0
        self.i_term = 0.0
        self.d_term = 0.0

    def update(self, measurement: f64, dt: f64) -> tuple[f64, f64]:
        error = self.setpoint - measurement
        self.integral = (self.integral * self.leak_rate) + error * dt
        self.integral = max(min(self.integral, 1e6), -1e6)
        derivative = (error - self.prev_error) / dt
        derivative = max(min(derivative, 1e6), -1e6)
        self.prev_error = error
        self.p_term = self.Kp * error
        self.i_term = self.Ki * self.integral
        self.d_term = self.Kd * derivative
        output = self.p_term + self.i_term + self.d_term
        return output, error


def run_pid_controller(
    setpoints: NDArray[f64],
    pid_values: list[PIDValue],
    duration: float = 10.0,
    dt: float = 0.01,
    q0: NDArray[f64] | None = None,
    trajectory: TrajectoryGenerator | None = None,
):
    robot = models.DH.Puma560()

    if q0 is not None:
        q: NDArray[f64] = np.array(q0, dtype=f64)
    else:
        q: NDArray[f64] = np.zeros(6, dtype=f64)
        
    qd: NDArray[f64] = np.zeros(6, dtype=f64)

    pids: list[PIDController] = [
        PIDController(
            Kp=f64(pid_values[i].Kp),
            Ki=f64(pid_values[i].Ki),
            Kd=f64(pid_values[i].Kd),
            setpoint=setpoints[i],
        )
        for i in range(len(setpoints))
    ]

    # Fall back to static trajectory when no explicit generator is given
    if trajectory is None:
        trajectory = StaticTrajectory(setpoints)

    t_steps: NDArray[f64] = np.arange(0, duration, dt)
    dt: f64 = f64(dt)

    q_values: list[list[f64]] = [[] for _ in range(6)]
    error_values: list[list[f64]] = [[] for _ in range(6)]
    u_values: list[list[f64]] = [[] for _ in range(6)]
    torque_values: list[list[f64]] = [[] for _ in range(6)]
    p_values: list[list[f64]] = [[] for _ in range(6)]
    i_values: list[list[f64]] = [[] for _ in range(6)]
    d_values: list[list[f64]] = [[] for _ in range(6)]
    setpoint_values: list[list[f64]] = [[] for _ in range(6)]

    # ---------------- Simulation Loop ----------------
    for t_now in t_steps:
        # Update each PID controller's setpoint from the trajectory
        current_sp = trajectory.get_setpoint(float(t_now))
        for i in range(6):
            pids[i].setpoint = current_sp[i]
        M: NDArray[f64] = robot.inertia(q)  # pyright: ignore[reportAttributeAccessIssue]
        C: NDArray[f64] = robot.coriolis(q, qd)  # pyright: ignore[reportAttributeAccessIssue]
        G: NDArray[f64] = robot.gravload(q)  # pyright: ignore[reportAttributeAccessIssue]

        pid_outputs = [pids[i].update(q[i], dt) for i in range(6)]
        
        # Unpack control signals (u) and errors
        tau_pid: NDArray[f64] = np.array([out[0] for out in pid_outputs])
        errors: list[f64] = [out[1] for out in pid_outputs]

        tau_vector: NDArray[f64] = tau_pid
        # tau_vector: NDArray[f64] = tau_pid + G

        qdd: NDArray[f64] = np.linalg.inv(M) @ (tau_vector - C @ qd - G)

        qd += qdd * dt
        q += qd * dt

        for i in range(6):
            q_values[i].append(np.rad2deg(q[i]))
            error_values[i].append(errors[i])
            u_values[i].append(tau_pid[i])
            torque_values[i].append(tau_vector[i])
            p_values[i].append(pids[i].p_term)
            i_values[i].append(pids[i].i_term)
            d_values[i].append(pids[i].d_term)
            setpoint_values[i].append(np.rad2deg(current_sp[i]))

    return t_steps, q_values, error_values, u_values, torque_values, p_values, i_values, d_values, setpoint_values


def plot_pid_controller_output(t_steps, q_values, u_values, setpoints, p_values, i_values, d_values):
    # ---------------- Plot: angle+setpoint, components, torque ----------------
    joint_to_plot = 5  # 0..5

    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(10, 12), sharex=True, dpi=100)

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
    ax1.set_ylabel("Response (deg)")
    ax1.set_title(f"Joint {joint_to_plot + 1} Tracking")
    ax1.legend(loc="upper right")
    ax1.grid()

    # Middle: P, I, D components
    ax2.plot(t_steps, p_values[joint_to_plot], "r", label="P Term")
    ax2.plot(t_steps, i_values[joint_to_plot], "g", label="I Term")
    ax2.plot(t_steps, d_values[joint_to_plot], "b", label="D Term")
    ax2.set_ylabel("Control Components")
    ax2.legend(loc="upper right")
    ax2.grid()

    # Bottom: torque
    ax3.plot(t_steps, u_values[joint_to_plot], "g", label="Total Torque")
    ax3.set_xlabel("Time (s)")
    ax3.set_ylabel("Torque (Nm)")
    ax3.legend(loc="upper right")
    ax3.grid()

    plt.tight_layout()
    plt.show()
