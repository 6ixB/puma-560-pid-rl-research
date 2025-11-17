import matplotlib.pyplot as plt
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas

class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig, self.axs = plt.subplots(6, 1, figsize=(width, height), dpi=dpi, sharex=True)
        super(MplCanvas, self).__init__(self.fig)
        self.setParent(parent)
        self.setup_initial_plots()

    def setup_initial_plots(self):
        """Set initial labels and formatting."""
        self.fig.suptitle("Robotics Simulation")
        for i in range(6):
            self.axs[i].set_ylabel(f"J{i+1}")
            self.axs[i].grid(True)
        self.axs[5].set_xlabel("Time (s)")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95])
        self.draw()
        
    def plot_fd_results(self, t, q, qd, qdd):
        """
        Clears the old plots and draws new forward dynamics results.
        """
        for i in range(6):
            self.axs[i].clear()
            self.axs[i].plot(t, q[:, i], label='Position (q)')
            self.axs[i].plot(t, qd[:, i], label='Velocity (qd)', linestyle='--')
            self.axs[i].set_ylabel(f"J{i+1} State")
            self.axs[i].grid(True)
            self.axs[i].legend(loc='upper right', fontsize='small')

        self.axs[5].set_xlabel("Time (s)")
        self.fig.suptitle("Forward Dynamics Results (Position & Velocity)")
        self.draw()
        
    def plot_id_results(self, t, q, torques):
        """
        Clears the old plots and draws new inverse dynamics results.
        Plots Torque and Position (on a twin axis)
        """
        for i in range(6):
            self.axs[i].clear()
            
            # Plot Torque on the left Y-axis
            color = 'tab:blue'
            self.axs[i].set_ylabel(f"J{i+1} Torque (Nm)", color=color, fontsize='small')
            self.axs[i].plot(t, torques[:, i], label='Torque (Nm)', color=color)
            self.axs[i].tick_params(axis='y', labelcolor=color)
            self.axs[i].grid(True)

            # Create a second Y-axis for Position
            ax2 = self.axs[i].twinx()
            color = 'tab:red'
            ax2.set_ylabel('Position (rad)', color=color, fontsize='small')
            ax2.plot(t, q[:, i], label='Position (rad)', color=color, linestyle='--')
            ax2.tick_params(axis='y', labelcolor=color)

        self.axs[5].set_xlabel("Time (s)")
        self.fig.suptitle("Inverse Dynamics Results (Torque & Position)")
        self.fig.tight_layout(rect=[0, 0.03, 1, 0.95]) # Re-run tight_layout
        self.draw()
