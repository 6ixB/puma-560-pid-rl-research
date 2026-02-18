import matplotlib.pyplot as plt
import numpy as np
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure

# Apply a clean style for better aesthetics
try:
    plt.style.use("seaborn-v0_8-whitegrid")
except OSError:
    plt.style.use("ggplot")  # Fallback


class MplCanvas(FigureCanvas):
    """
    Matplotlib canvas for displaying robotics simulation results.
    Dynamically adjusts layout based on the plotting mode:
      - FD/ID (6 joints): 3 rows x 2 columns grid.
      - PID (12 plots): 6 rows x 2 columns (Angle vs Torque per joint).
    """

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        # Create figure with constrained_layout for automatic spacing
        self.fig = Figure(figsize=(width, height), dpi=dpi, constrained_layout=True)
        super().__init__(self.fig)
        self.setParent(parent)

        # Initial placeholder setup
        self.setup_initial_plots()

    def setup_initial_plots(self):
        """Initial empty state or default view (FD/ID layout)."""
        self.fig.clear()
        self.fig.suptitle("Robotics Simulation Results", fontsize=14, weight="bold")
        
        # Default to 3x2 grid for 6 joints
        axs = self.fig.subplots(3, 2)
        self.axs = axs.flatten()

        for i, ax in enumerate(self.axs):
            ax.set_ylabel(f"J{i + 1}")
            ax.grid(True)
            if i >= 4:  # Bottom row
                ax.set_xlabel("Time (s)")

        self.draw()

    def _reset_figure(self, rows, cols, suptitle):
        """Helper to clear figure and create new subplots grid."""
        self.fig.clear()
        self.fig.suptitle(suptitle, fontsize=12, weight="bold")
        axs = self.fig.subplots(rows, cols, sharex=True)
        self.axs = axs.flatten()
        return self.axs

    def plot_fd_results(self, t, q, qd, qdd):
        """Plots Forward Dynamics results (Joint Positions)."""
        # Layout: 3 Rows x 2 Cols (Compact 6-joint view)
        self._reset_figure(3, 2, "Forward Dynamics: Joint Positions (Degrees)")

        for i in range(6):
            if i >= len(self.axs): break
            ax = self.axs[i]
            
            ax.plot(t, q[:, i], label="Position", color="#1f77b4", linewidth=1.5)
            ax.set_ylabel(f"J{i + 1} (Deg)", fontsize=9)
            ax.legend(loc="upper right", fontsize="x-small", frameon=True)
            ax.grid(True, linestyle="--", alpha=0.7)

        # Set X-label only on bottom row (indices 4 and 5)
        for i in range(4, 6):
            self.axs[i].set_xlabel("Time (s)", fontsize=10)

        self.draw()

    def plot_id_results(self, t, q, qd, qdd, torques, active_modes):
        """Plots Inverse Dynamics results based on selected modes."""
        title_modes = ", ".join(active_modes) if active_modes else "None"
        self._reset_figure(3, 2, f"Inverse Dynamics: {title_modes}")

        if not active_modes:
            self.draw()
            return

        # Data mapping: (Data, Unit, Color)
        data_map = {
            "tau": (torques, "Nm", "#d62728"),      # Red
            "q": (q, "Deg", "#1f77b4"),             # Blue
            "qd": (qd, "Deg/s", "#ff7f0e"),         # Orange
            "qdd": (qdd, "Deg/s^2", "#2ca02c"),     # Green
        }

        # Plot selected modes overlayed
        for mode in active_modes:
            if mode in data_map:
                data, unit, color = data_map[mode]
                for i in range(6):
                    ax = self.axs[i]
                    ax.plot(t, data[:, i], label=mode, color=color, linewidth=1.5)
                    # Note: Y-label shows the last plotted unit if mixed, 
                    # but legend helps distinguish.
                    if len(active_modes) == 1:
                        ax.set_ylabel(f"J{i + 1} ({unit})", fontsize=9)
                    else:
                        ax.set_ylabel(f"J{i + 1}", fontsize=9)
                    
                    ax.legend(loc="upper right", fontsize="x-small", frameon=True)
                    ax.grid(True, linestyle="--", alpha=0.7)

        # Set X-label only on bottom row
        for i in range(4, 6):
            self.axs[i].set_xlabel("Time (s)", fontsize=10)

        self.draw()

    def plot_pc_results(
        self, t_steps, q_values, error_values, u_values, torque_values, setpoints
    ):
        """Plots PID Controller results: Angle, Error, Control, Torque per joint."""
        # Layout: 6 Rows x 4 Cols
        # Cols: Angle | Error | Control u(k) | Torque
        self._reset_figure(6, 4, "PID Controller: Angle | Error | Control | Torque")

        for j in range(6):
            # Indices for the 4 columns in the j-th row
            idx_angle = j * 4
            idx_error = j * 4 + 1
            idx_ctrl = j * 4 + 2
            idx_torque = j * 4 + 3

            ax_angle = self.axs[idx_angle]
            ax_error = self.axs[idx_error]
            ax_ctrl = self.axs[idx_ctrl]
            ax_torque = self.axs[idx_torque]

            # --- 1. Angle Plot ---
            ax_angle.plot(
                t_steps,
                q_values[j],
                label=f"J{j + 1} Angle",
                color="#1f77b4",
                linewidth=1.5,
            )
            # Setpoint line
            sp_deg = np.rad2deg(setpoints[j])
            ax_angle.axhline(
                sp_deg,
                linestyle="--",
                color="#d62728",
                label=f"Ref {sp_deg:.1f}",
                linewidth=1.2,
                alpha=0.8,
            )
            ax_angle.set_ylabel(f"J{j + 1} (deg)", fontsize=9)
            ax_angle.legend(loc="upper right", fontsize="x-small")
            ax_angle.grid(True, linestyle="--", alpha=0.7)

            # --- 2. Error Plot ---
            ax_error.plot(
                t_steps,
                error_values[j],
                label="Error",
                color="#ff7f0e",
                linewidth=1.5,
            )
            ax_error.set_ylabel("Err", fontsize=9)
            ax_error.legend(loc="upper right", fontsize="x-small")
            ax_error.grid(True, linestyle="--", alpha=0.7)

            # --- 3. Control Signal u(k) Plot ---
            ax_ctrl.plot(
                t_steps,
                u_values[j],
                label="u(k)",
                color="#9467bd",
                linewidth=1.5,
            )
            ax_ctrl.set_ylabel("u(k)", fontsize=9)
            ax_ctrl.legend(loc="upper right", fontsize="x-small")
            ax_ctrl.grid(True, linestyle="--", alpha=0.7)

            # --- 4. Torque Plot ---
            ax_torque.plot(
                t_steps,
                torque_values[j],
                label="Torque",
                color="#2ca02c",
                linewidth=1.5,
            )
            ax_torque.set_ylabel("Nm", fontsize=9)
            ax_torque.legend(loc="upper right", fontsize="x-small")
            ax_torque.grid(True, linestyle="--", alpha=0.7)

            # Hide X labels for all except bottom row
            if j < 5:
                for ax in [ax_angle, ax_error, ax_ctrl, ax_torque]:
                    ax.tick_params(labelbottom=False)

        # Bottom row X labels
        for k in range(4):
            self.axs[-4 + k].set_xlabel("Time (s)", fontsize=10)

        self.draw()

    def plot_tuning_results(self, experiments: list[dict], title: str):
        """
        Plots comparison of multiple PID experiments for a specific joint.
        experiments: List of dicts {'t': t_steps, 'q': q_values, 'label': str}
        """
        self.fig.clear()
        self.fig.suptitle(title, fontsize=12, weight="bold")
        ax = self.fig.add_subplot(111)

        # Plot each experiment
        for exp in experiments:
            ax.plot(exp["t"], exp["q"], label=exp["label"], linewidth=2)

        # Draw Setpoint (Assuming 45 degrees for Joint 2 as per demo)
        ax.axhline(45, color="r", linestyle="--", label="Setpoint (45 deg)", alpha=0.7)

        ax.set_xlabel("Time (s)")
        ax.set_ylabel("Joint 2 Angle (deg)")
        ax.legend(loc="best")
        ax.grid(True)

        self.draw()
