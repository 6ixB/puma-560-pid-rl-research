import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class MplCanvas(FigureCanvas):
    """
    A Matplotlib canvas widget to embed in a PySide6 application.
    """
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axs = plt.subplots(6, 1, figsize=(width, height), dpi=dpi, sharex=True)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setup_initial_plots()

    def setup_initial_plots(self):
        """Set initial labels and formatting."""
        self.fig.suptitle("Robotics Simulation Results")
        for i in range(6):
            self.axs[i].set_ylabel(f"J{i+1}")
            self.axs[i].grid(True)
        self.axs[5].set_xlabel("Time (s)")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.draw()
        
    def plot_fd_results(self, t, q, qd, qdd):
        """
        Plots Forward Dynamics results.
        q, qd, qdd are expected in DEGREES.
        """
        for i in range(6):
            self.axs[i].clear()
            self.axs[i].plot(t, q[:, i], label='Position (deg)', color='tab:blue')
            # Uncomment below if you want to see velocity too
            # self.axs[i].plot(t, qd[:, i], label='Velocity (deg/s)', color='tab:orange', linestyle='--')
            
            self.axs[i].set_ylabel(f"J{i+1} (Deg)")
            self.axs[i].grid(True)
            self.axs[i].legend(loc='upper right', fontsize='small')

        self.axs[5].set_xlabel("Time (s)")
        self.fig.suptitle("Forward Dynamics: Joint Positions (Degrees)")
        self.draw()
        
    def plot_id_results(self, t, q, qd, qdd, torques, active_modes):
        """
        Plots Inverse Dynamics results based on the active checkboxes.
        active_modes: List of strings e.g. ["tau", "q"]
        """
        # Data definition
        data_map = {
            "tau": (torques, "Nm", 'tab:red'),
            "q":   (q,       "Deg", 'tab:blue'),
            "qd":  (qd,      "Deg/s", 'tab:orange'),
            "qdd": (qdd,     "Deg/s^2", 'tab:green')
        }

        # Clear axes
        for i in range(6):
            self.axs[i].clear()
            self.axs[i].grid(True)

        if not active_modes:
            self.draw()
            return

        # Plot selected data
        for mode in active_modes:
            if mode in data_map:
                data, unit, color = data_map[mode]
                for i in range(6):
                    self.axs[i].plot(t, data[:, i], label=mode, color=color)
                    # We only set the label for the last plotted item to avoid overwriting,
                    # but typically you might want a more complex label logic.
                    # Simple approach: just show unit of last plotted item.
                    self.axs[i].set_ylabel(f"J{i+1} ({unit})")
                    self.axs[i].legend(loc='upper right', fontsize='small')

        self.axs[5].set_xlabel("Time (s)")
        self.fig.suptitle(f"Inverse Dynamics: {', '.join(active_modes)}")
        self.draw()