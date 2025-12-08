import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas


class MplCanvas(FigureCanvas):
    """
    Matplotlib canvas containing up to 12 subplots:
        For each joint J1..J6 in PID mode:
            Row 1: Angle (deg) + Setpoint (deg)
            Row 2: Torque (Nm)

    For FD/ID:
        Only the first 6 subplots (one per joint) are shown.
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # 12 rows, 1 column (we will hide 6 of them when not in PID mode)
        self.fig, self.axs = plt.subplots(
            12, 1, figsize=(width, height), dpi=dpi, sharex=True
        )
        super().__init__(self.fig)
        self.setParent(parent)
        self.setup_initial_plots()

    def setup_initial_plots(self):
        self.fig.suptitle("Robotics Simulation Results")

        # Initialize first 6 axes (default FD/ID view)
        for i in range(6):
            ax = self.axs[i]
            ax.set_ylabel(f"J{i + 1}")
            ax.grid(True)

        # Hide the extra 6 axes by default (used only for PID)
        for i in range(6, 12):
            self.axs[i].set_visible(False)

        self.axs[5].set_xlabel("Time (s)")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # pyright: ignore[reportArgumentType]
        self.draw()

    def plot_fd_results(self, t, q, qd, qdd):
        # Show first 6 axes
        for i in range(6):
            self.axs[i].set_visible(True)
        for i in range(6, 12):
            self.axs[i].set_visible(False)
            self.axs[i].clear()

        for i in range(6):
            ax = self.axs[i]
            ax.clear()

            ax.plot(t, q[:, i], label="Position (deg)", color="tab:blue")

            ax.set_ylabel(f"J{i + 1} (Deg)")
            ax.grid(True)
            ax.legend(loc="upper right", fontsize="small")

            # AUTOSCALE
            ax.relim()
            ax.autoscale_view()

        self.axs[5].set_xlabel("Time (s)")
        self.fig.suptitle("Forward Dynamics: Joint Positions (Degrees)")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # pyright: ignore[reportArgumentType]
        self.draw()

    def plot_id_results(self, t, q, qd, qdd, torques, active_modes):
        data_map = {
            "tau": (torques, "Nm", "tab:red"),
            "q": (q, "Deg", "tab:blue"),
            "qd": (qd, "Deg/s", "tab:orange"),
            "qdd": (qdd, "Deg/s^2", "tab:green"),
        }

        # Show first 6 axes
        for i in range(6):
            self.axs[i].set_visible(True)
            self.axs[i].clear()
            self.axs[i].grid(True)

        for i in range(6, 12):
            self.axs[i].set_visible(False)
            self.axs[i].clear()

        if not active_modes:
            self.draw()
            return

        for mode in active_modes:
            if mode in data_map:
                data, unit, color = data_map[mode]
                for i in range(6):
                    ax = self.axs[i]
                    ax.plot(t, data[:, i], label=mode, color=color)
                    ax.set_ylabel(f"J{i + 1} ({unit})")
                    ax.legend(loc="upper right", fontsize="small")

        # AUTOSCALE for all 6 axes
        for i in range(6):
            ax = self.axs[i]
            ax.relim()
            ax.autoscale_view()

        self.axs[5].set_xlabel("Time (s)")
        self.fig.suptitle(f"Inverse Dynamics: {', '.join(active_modes)}")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # pyright: ignore[reportArgumentType]
        self.draw()

    def plot_pc_results(self, t_steps, q_values, u_values, setpoints):
        # Show all 12
        for i in range(12):
            self.axs[i].set_visible(True)

        for j in range(6):
            angle_ax = self.axs[j * 2]
            torque_ax = self.axs[j * 2 + 1]

            angle_ax.clear()
            torque_ax.clear()

            angle_ax.grid(True)
            torque_ax.grid(True)

            # Angle
            angle_ax.plot(
                t_steps, q_values[j], label=f"J{j + 1} Angle (deg)", color="tab:blue"
            )
            angle_ax.axhline(
                np.rad2deg(setpoints[j]),
                linestyle="--",
                color="tab:red",
                label="Setpoint",
            )
            angle_ax.set_ylabel(f"J{j + 1} (deg)")
            angle_ax.legend(loc="upper right", fontsize="small")

            # Torque
            torque_ax.plot(t_steps, u_values[j], label="Torque (Nm)", color="tab:green")
            torque_ax.set_ylabel(f"Tau {j + 1}")
            torque_ax.legend(loc="upper right", fontsize="small")

            # AUTOSCALE for each subplot
            angle_ax.relim()
            angle_ax.autoscale_view()
            torque_ax.relim()
            torque_ax.autoscale_view()

        self.axs[-1].set_xlabel("Time (s)")
        self.fig.suptitle("PID Controller: Angle + Torque for All Joints")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])  # pyright: ignore[reportArgumentType]
        self.draw()
