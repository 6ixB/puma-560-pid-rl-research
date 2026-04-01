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
        if not isinstance(axs, np.ndarray):
            self.axs = [axs]
        else:
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
        self, t_steps, q_values, error_values, u_values, torque_values, p_values, i_values, d_values, setpoints, active_joints, active_vars
    ):
        """Plots PID Controller results: Overlay of selected variables for active joints, separated by joint."""
        n_joints = len(active_joints)
        if n_joints == 0:
            self._reset_figure(1, 1, "PID Controller: No Joints Selected")
            self.draw()
            return
            
        self._reset_figure(n_joints, 1, "PID Controller: Variables per Joint")

        # Color map for variables
        colors = {
            "Angle": "#1f77b4",
            "Setpoint": "#d62728",
            "Error": "#ff7f0e",
            "U(k)": "#9467bd",
            "Torque(Nm)": "#2ca02c",
            "P": "#e377c2",
            "I": "#8c564b",
            "D": "#17becf"
        }

        # Instead of linestyles by joint, we use solid lines for each individual joint graph
        ls = "-"

        for idx, j in enumerate(active_joints):
            ax = self.axs[idx]
            has_plotted = False

            if "Angle" in active_vars:
                ax.plot(t_steps, q_values[j], label=f"Output", color=colors["Angle"], linestyle=ls, linewidth=1.5)
                has_plotted = True
            
            if "Setpoint" in active_vars:
                ax.plot(t_steps, setpoints[j], linestyle="--", color=colors["Setpoint"], label=f"Setpoint", linewidth=1.2, alpha=0.8)
                has_plotted = True

            if "Error" in active_vars:
                ax.plot(t_steps, error_values[j], label=f"Err", color=colors["Error"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "U(k)" in active_vars:
                ax.plot(t_steps, u_values[j], label=f"U(k)", color=colors["U(k)"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "Torque(Nm)" in active_vars:
                ax.plot(t_steps, torque_values[j], label=f"Torque", color=colors["Torque(Nm)"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "P" in active_vars:
                ax.plot(t_steps, p_values[j], label=f"P", color=colors["P"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "I" in active_vars:
                ax.plot(t_steps, i_values[j], label=f"I", color=colors["I"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "D" in active_vars:
                ax.plot(t_steps, d_values[j], label=f"D", color=colors["D"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if has_plotted:
                ax.set_ylabel(f"J{j+1}", fontsize=12)
                # Only show legend on the top plot to save space, or show on all if preferred
                pass  # We will do it per-axis but position it well
                ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize="large")
                ax.grid(True, linestyle="--", alpha=0.7)

        # Set X label only on the bottom axis
        if n_joints > 0:
            self.axs[-1].set_xlabel("Time (s)", fontsize=12)

        # Better layout for out-of-bounds legend
        self.fig.tight_layout(rect=[0, 0, 0.85, 1])

        self.draw()

    def plot_pc_results_animated(
        self, t_steps, q_values, error_values, u_values, torque_values, p_values, i_values, d_values, setpoints, active_joints, active_vars, frame_idx
    ):
        """Animates PID Controller results: Plotted up to frame_idx."""
        n_joints = len(active_joints)
        if n_joints == 0:
            return
            
        t_sub = t_steps[:frame_idx]

        # Use the same color map
        colors = {
            "Angle": "#1f77b4",
            "Setpoint": "#d62728",
            "Error": "#ff7f0e",
            "U(k)": "#9467bd",
            "Torque(Nm)": "#2ca02c",
            "P": "#e377c2",
            "I": "#8c564b",
            "D": "#17becf"
        }
        ls = "-"

        for idx, j in enumerate(active_joints):
            ax = self.axs[idx]
            ax.clear()

            has_plotted = False

            if "Angle" in active_vars:
                ax.plot(t_sub, q_values[j][:frame_idx], label=f"Angle", color=colors["Angle"], linestyle=ls, linewidth=1.5)
                has_plotted = True
            
            if "Setpoint" in active_vars:
                ax.plot(t_sub, setpoints[j][:frame_idx], linestyle="--", color=colors["Setpoint"], label=f"Setpoint", linewidth=1.2, alpha=0.8)
                has_plotted = True

            if "Error" in active_vars:
                ax.plot(t_sub, error_values[j][:frame_idx], label=f"Err", color=colors["Error"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "U(k)" in active_vars:
                ax.plot(t_sub, u_values[j][:frame_idx], label=f"U(k)", color=colors["U(k)"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "Torque(Nm)" in active_vars:
                ax.plot(t_sub, torque_values[j][:frame_idx], label=f"Torque", color=colors["Torque(Nm)"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "P" in active_vars:
                ax.plot(t_sub, p_values[j][:frame_idx], label=f"P", color=colors["P"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "I" in active_vars:
                ax.plot(t_sub, i_values[j][:frame_idx], label=f"I", color=colors["I"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if "D" in active_vars:
                ax.plot(t_sub, d_values[j][:frame_idx], label=f"D", color=colors["D"], linestyle=ls, linewidth=1.5)
                has_plotted = True

            if has_plotted:
                ax.set_ylabel(f"J{j+1}", fontsize=12)
                ax.set_xlim(t_steps[0], t_steps[-1])
                
                # Determine min and max Y for better scaling
                y_min = float("inf")
                y_max = float("-inf")
                if "Angle" in active_vars:
                    y_min = min(y_min, min(q_values[j]))
                    y_max = max(y_max, max(q_values[j]))
                if "Setpoint" in active_vars:
                    y_min = min(y_min, min(setpoints[j]))
                    y_max = max(y_max, max(setpoints[j]))
                if "Error" in active_vars:
                    y_min = min(y_min, min(error_values[j]))
                    y_max = max(y_max, max(error_values[j]))
                if "U(k)" in active_vars:
                    y_min = min(y_min, min(u_values[j]))
                    y_max = max(y_max, max(u_values[j]))
                if "Torque(Nm)" in active_vars:
                    y_min = min(y_min, min(torque_values[j]))
                    y_max = max(y_max, max(torque_values[j]))
                if "P" in active_vars:
                    y_min = min(y_min, min(p_values[j]))
                    y_max = max(y_max, max(p_values[j]))
                if "I" in active_vars:
                    y_min = min(y_min, min(i_values[j]))
                    y_max = max(y_max, max(i_values[j]))
                if "D" in active_vars:
                    y_min = min(y_min, min(d_values[j]))
                    y_max = max(y_max, max(d_values[j]))
                
                margin = (y_max - y_min) * 0.1
                if margin == 0:
                    margin = 1.0
                ax.set_ylim(y_min - margin, y_max + margin)

                ax.legend(loc="center left", bbox_to_anchor=(1.05, 0.5), fontsize="large")
                ax.grid(True, linestyle="--", alpha=0.7)

        # Set X label only on the bottom axis
        if n_joints > 0:
            self.axs[-1].set_xlabel("Time (s)", fontsize=12)

        self.fig.tight_layout(rect=[0, 0, 0.85, 1])
        self.draw_idle()

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
