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

    def plot_pc_results(self, t_steps, q_values, u_values, setpoints):
        """Plots PID Controller results: Angle (Left) vs Torque (Right) per joint."""
        # Layout: 6 Rows x 2 Cols
        # Row i, Col 0: Joint i Angle
        # Row i, Col 1: Joint i Torque
        self._reset_figure(6, 2, "PID Controller: Angle (Left) vs Torque (Right)")

        for j in range(6):
            # Left Plot (Angle) -> Index 2*j
            angle_ax = self.axs[j * 2]
            # Right Plot (Torque) -> Index 2*j + 1
            torque_ax = self.axs[j * 2 + 1]

            # --- Angle Plot ---
            angle_ax.plot(
                t_steps, q_values[j], label=f"J{j + 1} Angle", color="#1f77b4", linewidth=1.5
            )
            # Setpoint line
            sp_deg = np.rad2deg(setpoints[j])
            angle_ax.axhline(
                sp_deg, linestyle="--", color="#d62728", label=f"Ref {sp_deg:.1f}", linewidth=1.2, alpha=0.8
            )
            angle_ax.set_ylabel(f"J{j + 1} (deg)", fontsize=9)
            angle_ax.legend(loc="upper right", fontsize="x-small")
            angle_ax.grid(True, linestyle="--", alpha=0.7)

            # --- Torque Plot ---
            torque_ax.plot(t_steps, u_values[j], label="Torque", color="#2ca02c", linewidth=1.5)
            torque_ax.set_ylabel(f"Tau {j + 1} (Nm)", fontsize=9)
            torque_ax.legend(loc="upper right", fontsize="x-small")
            torque_ax.grid(True, linestyle="--", alpha=0.7)
            
            # Remove internal X labels for cleaner look (except bottom row)
            if j < 5:
                # sharex=True handles ticks, but let's ensure labels are off
                angle_ax.tick_params(labelbottom=False)
                torque_ax.tick_params(labelbottom=False)

        # Bottom row X labels
        self.axs[-2].set_xlabel("Time (s)", fontsize=10)
        self.axs[-1].set_xlabel("Time (s)", fontsize=10)

        self.draw()
