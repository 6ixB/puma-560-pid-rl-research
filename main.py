import sys

import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from numpy import float64 as f64
from PySide6 import QtCore, QtGui, QtWidgets

from canvas import MplCanvas
from dynamics import run_forward_dynamics, run_inverse_dynamics
from pid_controller import PIDValue, run_pid_controller
from trajectory import TrajectoryMode, StaticTrajectory, WaypointTrajectory, SineTrajectory
import roboticstoolbox as rtb
from matplotlib.figure import Figure
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
class MainWindow(QtWidgets.QMainWindow):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("PUMA 560 Simulator (Degrees)")
        self.setGeometry(100, 100, 1400, 900)

        main_widget = QtWidgets.QWidget()
        self.setCentralWidget(main_widget)
        main_layout = QtWidgets.QHBoxLayout(main_widget)

        # --- Left-side Tabbed Control Panel ---
        self.tabs = QtWidgets.QTabWidget()
        self.tabs.setMaximumWidth(400)

        self.tab_fd = QtWidgets.QWidget()
        self.tab_id = QtWidgets.QWidget()
        self.tab_pc = QtWidgets.QWidget()
        self.tab_tuning = QtWidgets.QWidget()

        self.tabs.addTab(self.tab_fd, "Forward Dynamics")
        self.tabs.addTab(self.tab_id, "Inverse Dynamics")
        self.tabs.addTab(self.tab_pc, "PID Controller")
        self.tabs.addTab(self.tab_tuning, "PID Tuning")

        main_layout.addWidget(self.tabs)

        # --- Populate Tabs ---
        self.setup_fd_tab()
        self.setup_id_tab()
        self.setup_pc_tab()
        self.setup_tuning_tab()

        # --- Right-side Plotting Area ---
        splitter = QtWidgets.QSplitter(QtCore.Qt.Orientation.Vertical)

        # Top area: 2D Plots
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)

        self.plot_canvas = MplCanvas(self, width=8, height=6, dpi=100)
        toolbar = NavigationToolbar(self.plot_canvas, self)

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.plot_canvas)
        splitter.addWidget(plot_widget)

        # Bottom area: 3D Robot View
        robot_widget = QtWidgets.QWidget()
        robot_layout = QtWidgets.QVBoxLayout(robot_widget)
        
        # Create a matplotlib figure for RTB
        self.fig_3d = Figure(figsize=(8, 4), dpi=100)
        self.canvas_3d = FigureCanvas(self.fig_3d)
        
        # Mock the manager for roboticstoolbox compatibility
        class DummyManager:
            def set_window_title(self, title):
                pass
        self.canvas_3d.manager = DummyManager()
        self.fig_3d.number = 1
        
        robot_layout.addWidget(self.canvas_3d)
        splitter.addWidget(robot_widget)

        main_layout.addWidget(splitter)
        
        # Initialize Robot
        self.robot = rtb.models.DH.Puma560()
        # Setup the plot via plotting method but passing False to block ensures we don't freeze
        self._init_3d_plot()

    def _init_3d_plot(self):
        # Initial robot pose
        self.env = self.robot.plot([0, 0, 0, 0, 0, 0], backend='pyplot', fig=self.fig_3d, block=False)
        self.canvas_3d.draw()

    def animate_3d(self, t, q_trajectory, full_results=None):
        """Helper to animate the robot 3D view given a trajectory"""
        if q_trajectory is None or len(q_trajectory) == 0:
            return
            
        # Optional: extract results for animating the plot
        if full_results:
            (q_values, error_values, u_values, torque_values, p_values, i_values, d_values, setpoints, active_joints, active_vars) = full_results
            
            # Reset figure once before animation starts
            n_joints = len(active_joints)
            if n_joints > 0:
                self.plot_canvas._reset_figure(n_joints, 1, "PID Controller: Variables per Joint")
            
        # Decide downsampling to aim for roughly ~30fps visual if simulation is 100Hz
        dt = t[1] - t[0] if len(t) > 1 else 0.01
        target_fps = 30.0
        step_sz = max(1, int(1.0 / (target_fps * dt)))
        
        for i in range(0, len(q_trajectory), step_sz):
            self.robot.q = np.deg2rad(q_trajectory[i])
            try:
                # The returned env from plot() handles updates
                if self.env:
                    self.env.step(0.01)
                self.canvas_3d.draw_idle()
                
                # Animate Graph if active data is provided
                if full_results and n_joints > 0:
                    # Provide up to i+1 elements
                    self.plot_canvas.plot_pc_results_animated(
                        t, q_values, error_values, u_values, torque_values, p_values, i_values, d_values, setpoints, active_joints, active_vars, frame_idx=i+1
                    )
            except Exception as e:
                print(f"Animation step error: {e}")
            QtCore.QCoreApplication.processEvents()
            
        # Final frame
        final_idx = len(q_trajectory)
        if full_results and n_joints > 0 and i != final_idx - 1:
             self.plot_canvas.plot_pc_results_animated(
                t, q_values, error_values, u_values, torque_values, p_values, i_values, d_values, setpoints, active_joints, active_vars, frame_idx=final_idx
             )

    def _create_joint_input_group(self, title, defaults):
        group_box = QtWidgets.QGroupBox(title)
        form_layout = QtWidgets.QFormLayout(group_box)
        entries = []
        for i in range(6):
            entry = QtWidgets.QLineEdit(str(defaults[i]))
            entry.setValidator(QtGui.QDoubleValidator())
            form_layout.addRow(f"Joint {i + 1}:", entry)
            entries.append(entry)
        return group_box, entries

    def _get_joint_values(self, entries):
        return [float(entry.text()) for entry in entries]

    def setup_fd_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_fd)

        # Initial Position (q0)
        q0_box, self.fd_q0_entries = self._create_joint_input_group(
            "Initial Position (q0) [Deg]", [0.0] * 6
        )

        # Initial Velocity (qd0)
        qd0_box, self.fd_qd0_entries = self._create_joint_input_group(
            "Initial Velocity (qd0) [Deg/s]", [0.0] * 6
        )

        # Applied Torques (tau)
        tau_box, self.fd_tau_entries = self._create_joint_input_group(
            "Applied Torques (tau) [Nm]", [10, 20, 5, 1, 1, 1]
        )

        # Simulation Parameters
        params_box = QtWidgets.QGroupBox("Simulation Parameters")
        params_layout = QtWidgets.QFormLayout(params_box)
        self.fd_duration_entry = QtWidgets.QLineEdit("0.8")
        self.fd_dt_entry = QtWidgets.QLineEdit("0.01")
        self.fd_duration_entry.setValidator(QtGui.QDoubleValidator(0.1, 100.0, 2))
        self.fd_dt_entry.setValidator(QtGui.QDoubleValidator(0.0001, 1.0, 4))
        params_layout.addRow("Duration (s):", self.fd_duration_entry)
        params_layout.addRow("Time Step (dt):", self.fd_dt_entry)

        # Run Button
        self.fd_run_button = QtWidgets.QPushButton("Run Forward Dynamics")
        self.fd_run_button.setStyleSheet("padding: 10px; font-weight: bold;")
        self.fd_run_button.clicked.connect(self.on_run_fd)

        # Scroll Area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.addWidget(q0_box)
        scroll_layout.addWidget(qd0_box)
        scroll_layout.addWidget(tau_box)
        scroll_layout.addWidget(params_box)
        scroll_layout.addWidget(self.fd_run_button)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)

        layout.addWidget(scroll)

    def setup_id_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_id)

        # Initial Values
        q0_box, self.id_q0_entries = self._create_joint_input_group(
            "Initial Values (q_init) [Deg]", [0.0] * 6
        )

        # Target Values
        q_target_box, self.id_q_target_entries = self._create_joint_input_group(
            "Target Values (q_target) [Deg]", [45, 90, 90, 30, 0, 0]
        )

        # Simulation Parameters
        params_box = QtWidgets.QGroupBox("Simulation Parameters")
        params_layout = QtWidgets.QFormLayout(params_box)
        self.id_duration_entry = QtWidgets.QLineEdit("2.0")
        self.id_dt_entry = QtWidgets.QLineEdit("0.01")
        self.id_duration_entry.setValidator(QtGui.QDoubleValidator(0.1, 100.0, 2))
        self.id_dt_entry.setValidator(QtGui.QDoubleValidator(0.0001, 1.0, 4))
        params_layout.addRow("Duration (s):", self.id_duration_entry)
        params_layout.addRow("Time Step (dt):", self.id_dt_entry)

        # Monitor Toggles (Checkboxes)
        plot_box = QtWidgets.QGroupBox("Monitor Variables")
        plot_layout = QtWidgets.QHBoxLayout(plot_box)

        self.id_check_tau = QtWidgets.QCheckBox("tau")
        self.id_check_q = QtWidgets.QCheckBox("q")
        self.id_check_qd = QtWidgets.QCheckBox("qd")
        self.id_check_qdd = QtWidgets.QCheckBox("qdd")

        # Set default checked
        self.id_check_tau.setChecked(True)

        plot_layout.addWidget(self.id_check_tau)
        plot_layout.addWidget(self.id_check_q)
        plot_layout.addWidget(self.id_check_qd)
        plot_layout.addWidget(self.id_check_qdd)

        # Run Button
        self.id_run_button = QtWidgets.QPushButton("Run Inverse Dynamics")
        self.id_run_button.setStyleSheet("padding: 10px; font-weight: bold;")
        self.id_run_button.clicked.connect(self.on_run_id)

        # Scroll Area
        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)
        scroll_layout.addWidget(q0_box)
        scroll_layout.addWidget(q_target_box)
        scroll_layout.addWidget(params_box)
        scroll_layout.addWidget(plot_box)
        scroll_layout.addWidget(self.id_run_button)
        scroll_layout.addStretch()
        scroll.setWidget(scroll_content)

        layout.addWidget(scroll)

    def setup_pc_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_pc)

        joints_widget = QtWidgets.QWidget()
        joints_layout = QtWidgets.QVBoxLayout(joints_widget)
        joints_layout.setContentsMargins(0, 0, 0, 0)

        self.pc_q0_entries = []
        self.pc_setpoint_entries = []
        self.pc_kp_entries = []
        self.pc_ki_entries = []
        self.pc_kd_entries = []

        for i in range(6):
            group_box = QtWidgets.QGroupBox(f"Joint {i + 1}")
            form_layout = QtWidgets.QFormLayout(group_box)
            form_layout.setVerticalSpacing(2)

            q0_entry = QtWidgets.QLineEdit("0.0")
            q0_entry.setValidator(QtGui.QDoubleValidator())
            self.pc_q0_entries.append(q0_entry)
            form_layout.addRow("Start Angle [Deg]:", q0_entry)

            sp_entry = QtWidgets.QLineEdit("0.0")
            sp_entry.setValidator(QtGui.QDoubleValidator())
            self.pc_setpoint_entries.append(sp_entry)
            form_layout.addRow("Setpoint [Deg]:", sp_entry)

            kp_entry = QtWidgets.QLineEdit("1.0")
            kp_entry.setValidator(QtGui.QDoubleValidator())
            self.pc_kp_entries.append(kp_entry)
            form_layout.addRow("Kp:", kp_entry)

            ki_entry = QtWidgets.QLineEdit("0.0")
            ki_entry.setValidator(QtGui.QDoubleValidator())
            self.pc_ki_entries.append(ki_entry)
            form_layout.addRow("Ki:", ki_entry)

            kd_entry = QtWidgets.QLineEdit("0.0")
            kd_entry.setValidator(QtGui.QDoubleValidator())
            self.pc_kd_entries.append(kd_entry)
            form_layout.addRow("Kd:", kd_entry)

            joints_layout.addWidget(group_box)

        # ---- Trajectory Mode Selector ----
        traj_mode_box = QtWidgets.QGroupBox("Trajectory Mode")
        traj_mode_layout = QtWidgets.QVBoxLayout(traj_mode_box)
        self.pc_traj_combo = QtWidgets.QComboBox()
        for mode in TrajectoryMode:
            self.pc_traj_combo.addItem(mode.value)
        traj_mode_layout.addWidget(self.pc_traj_combo)

        # Stacked widget to swap between trajectory parameter panels
        self.pc_traj_stack = QtWidgets.QStackedWidget()

        # Page 0: Static — no extra UI needed, just a label
        static_page = QtWidgets.QWidget()
        static_lbl = QtWidgets.QLabel("Uses the per-joint Setpoint values above.")
        static_lbl.setWordWrap(True)
        QtWidgets.QVBoxLayout(static_page).addWidget(static_lbl)
        self.pc_traj_stack.addWidget(static_page)

        # Page 1: Waypoints
        wp_page = QtWidgets.QWidget()
        wp_layout = QtWidgets.QVBoxLayout(wp_page)
        self.pc_wp_joint_combo = QtWidgets.QComboBox()
        for i in range(6):
            self.pc_wp_joint_combo.addItem(f"Joint {i+1}")
        wp_layout.addWidget(self.pc_wp_joint_combo)

        self.pc_wp_tables: list[QtWidgets.QTableWidget] = []
        self.pc_wp_table_stack = QtWidgets.QStackedWidget()
        for i in range(6):
            table = QtWidgets.QTableWidget(2, 2)
            table.setHorizontalHeaderLabels(["Time (s)", "Angle (Deg)"])
            table.horizontalHeader().setStretchLastSection(True)
            table.setItem(0, 0, QtWidgets.QTableWidgetItem("0.0"))
            table.setItem(0, 1, QtWidgets.QTableWidgetItem("0.0"))
            table.setItem(1, 0, QtWidgets.QTableWidgetItem("10.0"))
            table.setItem(1, 1, QtWidgets.QTableWidgetItem("0.0"))
            self.pc_wp_tables.append(table)
            self.pc_wp_table_stack.addWidget(table)
        wp_layout.addWidget(self.pc_wp_table_stack)
        self.pc_wp_joint_combo.currentIndexChanged.connect(
            self.pc_wp_table_stack.setCurrentIndex
        )

        wp_btn_layout = QtWidgets.QHBoxLayout()
        self.pc_wp_add_btn = QtWidgets.QPushButton("Add Row")
        self.pc_wp_rm_btn = QtWidgets.QPushButton("Remove Row")
        wp_btn_layout.addWidget(self.pc_wp_add_btn)
        wp_btn_layout.addWidget(self.pc_wp_rm_btn)
        wp_layout.addLayout(wp_btn_layout)

        self.pc_wp_add_btn.clicked.connect(self._wp_add_row)
        self.pc_wp_rm_btn.clicked.connect(self._wp_remove_row)

        self.pc_traj_stack.addWidget(wp_page)

        # Page 2: Sine Wave
        sine_page = QtWidgets.QWidget()
        sine_layout = QtWidgets.QVBoxLayout(sine_page)
        self.pc_sine_amp = []
        self.pc_sine_freq = []
        self.pc_sine_offset = []
        self.pc_sine_phase = []
        for i in range(6):
            grp = QtWidgets.QGroupBox(f"Joint {i+1}")
            fl = QtWidgets.QFormLayout(grp)
            fl.setVerticalSpacing(2)
            amp = QtWidgets.QLineEdit("0.0")
            amp.setValidator(QtGui.QDoubleValidator())
            freq = QtWidgets.QLineEdit("0.5")
            freq.setValidator(QtGui.QDoubleValidator())
            off = QtWidgets.QLineEdit("0.0")
            off.setValidator(QtGui.QDoubleValidator())
            ph = QtWidgets.QLineEdit("0.0")
            ph.setValidator(QtGui.QDoubleValidator())
            fl.addRow("Amplitude (Deg):", amp)
            fl.addRow("Frequency (Hz):", freq)
            fl.addRow("Offset (Deg):", off)
            fl.addRow("Phase (Deg):", ph)
            self.pc_sine_amp.append(amp)
            self.pc_sine_freq.append(freq)
            self.pc_sine_offset.append(off)
            self.pc_sine_phase.append(ph)
            sine_layout.addWidget(grp)
        self.pc_traj_stack.addWidget(sine_page)

        traj_mode_layout.addWidget(self.pc_traj_stack)
        self.pc_traj_combo.currentIndexChanged.connect(
            self.pc_traj_stack.setCurrentIndex
        )

        self.pc_run_button = QtWidgets.QPushButton("Run PID Controller")
        self.pc_run_button.setStyleSheet("padding: 10px; font-weight: bold;")
        self.pc_run_button.clicked.connect(self.on_run_pc)

        # Joint toggles
        joint_box = QtWidgets.QGroupBox("Show Joints")
        joint_layout = QtWidgets.QGridLayout(joint_box)
        self.pc_joint_checks = []
        for i in range(6):
            chk = QtWidgets.QCheckBox(f"J{i+1}")
            chk.setChecked(i == 1)  # Default Joint 2 enabled
            self.pc_joint_checks.append(chk)
            joint_layout.addWidget(chk, i // 3, i % 3)

        # Variable toggles
        var_box = QtWidgets.QGroupBox("Show Variables")
        var_layout = QtWidgets.QGridLayout(var_box)
        self.pc_var_checks = {}
        
        var_names = ["Angle", "Setpoint", "Error", "U(k)", "Torque(Nm)", "P", "I", "D"]
        for idx, v in enumerate(var_names):
            chk = QtWidgets.QCheckBox(v)
            chk.setChecked(v in ["Angle", "Setpoint"]) # sensible defaults
            self.pc_var_checks[v] = chk
            var_layout.addWidget(chk, idx // 3, idx % 3)

        # Simulation Parameters
        params_box = QtWidgets.QGroupBox("Simulation Parameters")
        params_layout = QtWidgets.QFormLayout(params_box)
        self.pc_duration_entry = QtWidgets.QLineEdit("10.0")
        self.pc_steps_entry = QtWidgets.QLineEdit("1000")
        self.pc_duration_entry.setValidator(QtGui.QDoubleValidator(0.1, 100.0, 2))
        self.pc_steps_entry.setValidator(QtGui.QIntValidator(10, 10000))
        params_layout.addRow("Duration (s):", self.pc_duration_entry)
        params_layout.addRow("Steps:", self.pc_steps_entry)

        self.pc_animate_check = QtWidgets.QCheckBox("Enable Live Animation")
        self.pc_animate_check.setChecked(True)
        params_layout.addRow("", self.pc_animate_check)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)

        scroll_layout.addWidget(joints_widget)
        scroll_layout.addWidget(traj_mode_box)
        scroll_layout.addWidget(params_box)
        scroll_layout.addWidget(joint_box)
        scroll_layout.addWidget(var_box)
        scroll_layout.addWidget(self.pc_run_button)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

    # ---- Waypoint table helpers ----
    def _wp_add_row(self):
        table = self.pc_wp_tables[self.pc_wp_joint_combo.currentIndex()]
        row = table.rowCount()
        table.insertRow(row)
        table.setItem(row, 0, QtWidgets.QTableWidgetItem(""))
        table.setItem(row, 1, QtWidgets.QTableWidgetItem(""))

    def _wp_remove_row(self):
        table = self.pc_wp_tables[self.pc_wp_joint_combo.currentIndex()]
        if table.rowCount() > 1:
            table.removeRow(table.rowCount() - 1)

    def setup_tuning_tab(self):
        layout = QtWidgets.QVBoxLayout(self.tab_tuning)

        # Description Label
        desc = QtWidgets.QLabel(
            "Demonstrate the effect of changing PID gains on Joint 2.\n"
            "Runs 3 experiments and plots the comparison."
        )
        desc.setWordWrap(True)
        layout.addWidget(desc)

        # Radio Buttons Group
        self.tuning_group = QtWidgets.QButtonGroup(self)
        
        self.rb_kp = QtWidgets.QRadioButton("Effect of Kp (Proportional)")
        self.rb_ki = QtWidgets.QRadioButton("Effect of Ki (Integral)")
        self.rb_kd = QtWidgets.QRadioButton("Effect of Kd (Derivative)")
        
        self.rb_kp.setChecked(True)

        self.tuning_group.addButton(self.rb_kp)
        self.tuning_group.addButton(self.rb_ki)
        self.tuning_group.addButton(self.rb_kd)

        layout.addWidget(self.rb_kp)
        layout.addWidget(self.rb_ki)
        layout.addWidget(self.rb_kd)

        # Run Button
        self.tuning_run_button = QtWidgets.QPushButton("Run Tuning Demo")
        self.tuning_run_button.setStyleSheet("padding: 10px; font-weight: bold;")
        self.tuning_run_button.clicked.connect(self.on_run_tuning)
        layout.addWidget(self.tuning_run_button)

        layout.addStretch()

    def on_run_fd(self):
        self.fd_run_button.setText("Running...")
        self.fd_run_button.setEnabled(False)
        QtCore.QCoreApplication.processEvents()

        try:
            q0 = self._get_joint_values(self.fd_q0_entries)
            qd0 = self._get_joint_values(self.fd_qd0_entries)
            torques = self._get_joint_values(self.fd_tau_entries)
            duration = float(self.fd_duration_entry.text())
            dt = float(self.fd_dt_entry.text())

            (t, q, qd, qdd) = run_forward_dynamics(torques, duration, dt, q0, qd0)
            self.plot_canvas.plot_fd_results(t, q, qd, qdd)
            self.animate_3d(t, q)

        except Exception as e:
            self.show_error_message("Simulation Error", f"An error occurred:\n{e}")
        finally:
            self.fd_run_button.setText("Run Forward Dynamics")
            self.fd_run_button.setEnabled(True)

    def on_run_id(self):
        self.id_run_button.setText("Running...")
        self.id_run_button.setEnabled(False)
        QtCore.QCoreApplication.processEvents()

        try:
            q_init_deg = self._get_joint_values(self.id_q0_entries)
            q_target_deg = self._get_joint_values(self.id_q_target_entries)
            duration = float(self.id_duration_entry.text())
            dt = float(self.id_dt_entry.text())

            # Gather checked variables
            active_modes = []
            if self.id_check_tau.isChecked():
                active_modes.append("tau")
            if self.id_check_q.isChecked():
                active_modes.append("q")
            if self.id_check_qd.isChecked():
                active_modes.append("qd")
            if self.id_check_qdd.isChecked():
                active_modes.append("qdd")

            if not active_modes:
                self.show_error_message(
                    "Plot Error", "Please select at least one variable to monitor."
                )
                return

            (t, q, qd, qdd, torques) = run_inverse_dynamics(
                q_init_deg, q_target_deg, duration, dt
            )
            self.plot_canvas.plot_id_results(t, q, qd, qdd, torques, active_modes)
            self.animate_3d(t, q)

        except Exception as e:
            self.show_error_message("Simulation Error", f"An error occurred:\n{e}")
        finally:
            self.id_run_button.setText("Run Inverse Dynamics")
            self.id_run_button.setEnabled(True)

    def _build_trajectory(self, setpoints_rad, duration):
        """Build a TrajectoryGenerator from the current UI state."""
        mode_text = self.pc_traj_combo.currentText()
        mode = TrajectoryMode(mode_text)

        if mode == TrajectoryMode.WAYPOINT:
            waypoints: list[list[tuple[float, float]]] = []
            for i in range(6):
                table = self.pc_wp_tables[i]
                joint_wps: list[tuple[float, float]] = []
                for row in range(table.rowCount()):
                    t_item = table.item(row, 0)
                    a_item = table.item(row, 1)
                    if t_item and a_item and t_item.text() and a_item.text():
                        joint_wps.append((float(t_item.text()), float(a_item.text())))
                if not joint_wps:
                    joint_wps = [(0.0, 0.0), (duration, 0.0)]
                waypoints.append(joint_wps)
            return WaypointTrajectory(waypoints)

        elif mode == TrajectoryMode.SINE_WAVE:
            params: list[tuple[float, float, float, float]] = []
            for i in range(6):
                amp = float(self.pc_sine_amp[i].text())
                freq = float(self.pc_sine_freq[i].text())
                off = float(self.pc_sine_offset[i].text())
                ph = float(self.pc_sine_phase[i].text())
                params.append((amp, freq, off, ph))
            return SineTrajectory(params)

        else:  # STATIC
            return StaticTrajectory(setpoints_rad)

    def on_run_pc(self):
        self.pc_run_button.setText("Running...")
        self.pc_run_button.setEnabled(False)
        QtCore.QCoreApplication.processEvents()

        try:
            # 1) Read starting angles and setpoints from UI (in degrees) and convert to radians
            q0_deg = np.array(
                self._get_joint_values(self.pc_q0_entries),
                dtype=np.float64,
            )
            q0_rad = np.deg2rad(q0_deg)

            setpoints_deg = np.array(
                self._get_joint_values(self.pc_setpoint_entries),
                dtype=np.float64,
            )
            setpoints_rad = np.deg2rad(setpoints_deg)

            # 2) Read PID gains per joint from UI and build PIDValue list
            pid_values: list[PIDValue] = []
            for i in range(6):
                Kp = float(self.pc_kp_entries[i].text())
                Ki = float(self.pc_ki_entries[i].text())
                Kd = float(self.pc_kd_entries[i].text())
                pid_values.append(PIDValue(Kp=f64(Kp), Ki=f64(Ki), Kd=f64(Kd)))
                
            duration = float(self.pc_duration_entry.text())
            steps = int(self.pc_steps_entry.text())

            # 3) Build trajectory generator from UI
            trajectory = self._build_trajectory(setpoints_rad, duration)

            # 4) Run simulation
            (
                t_steps,
                q_values,
                error_values,
                u_values,
                torque_values,
                p_values,
                i_values,
                d_values,
                setpoint_values,
            ) = run_pid_controller(
                setpoints=setpoints_rad,
                pid_values=pid_values,
                duration=duration,
                steps=steps,
                q0=q0_rad,
                trajectory=trajectory,
            )

            active_joints = [i for i, chk in enumerate(self.pc_joint_checks) if chk.isChecked()]
            active_vars = [v for v, chk in self.pc_var_checks.items() if chk.isChecked()]

            # 5) Plot on the embedded canvas
            if self.pc_animate_check.isChecked():
                # q_values from pc is shape (6, N), animate_3d expects (N, 6)
                q_traj = np.array(q_values).T
                full_results = (q_values, error_values, u_values, torque_values, p_values, i_values, d_values, setpoint_values, active_joints, active_vars)
                self.animate_3d(t_steps, q_traj, full_results=full_results)
            else:
                self.plot_canvas.plot_pc_results(
                    t_steps=t_steps,
                    q_values=q_values,
                    error_values=error_values,
                    u_values=u_values,
                    torque_values=torque_values,
                    p_values=p_values,
                    i_values=i_values,
                    d_values=d_values,
                    setpoints=setpoint_values,
                    active_joints=active_joints,
                    active_vars=active_vars,
                )
                self.robot.q = np.array(q_values)[:, -1]
                if self.env:
                    self.env.step(0.01)
                self.canvas_3d.draw_idle()

        except Exception as e:
            self.show_error_message("PID Controller Error", f"An error occurred:\n{e}")
        finally:
            self.pc_run_button.setText("Run PID Controller")
            self.pc_run_button.setEnabled(True)

    def on_run_tuning(self):
        self.tuning_run_button.setText("Running Experiments...")
        self.tuning_run_button.setEnabled(False)
        QtCore.QCoreApplication.processEvents()

        try:
            # Setup for Joint 2 (index 1) experiment
            joint_idx = 1
            setpoints_rad = np.zeros(6)
            setpoints_rad[joint_idx] = np.deg2rad(45.0) # Target 45 degrees

            experiments = []
            title = ""

            # Define gains based on selection
            if self.rb_kp.isChecked():
                title = "Effect of Proportional Gain (Kp)"
                # (Kp, Ki, Kd) tuples
                configs = [
                    (10, 0, 0, "Kp=10 (Sluggish/Error)"),
                    (50, 0, 0, "Kp=50 (Better)"),
                    (200, 0, 0, "Kp=200 (Overshoot)"),
                ]
            elif self.rb_ki.isChecked():
                title = "Effect of Integral Gain (Ki)"
                # Fix Kp=20, vary Ki
                configs = [
                    (20, 0, 0, "Ki=0 (Steady Error)"),
                    (20, 50, 0, "Ki=50 (Corrects Error)"),
                    (20, 100, 0, "Ki=100 (Oscillations)"),
                ]
            else: # Kd effect
                title = "Effect of Derivative Gain (Kd)"
                # High Kp=200, vary Kd
                configs = [
                    (200, 0, 0, "Kd=0 (Overshoot)"),
                    (200, 0, 10, "Kd=10 (Damped)"),
                    (200, 0, 50, "Kd=50 (Overdamped)"),
                ]

            # Run experiments
            for kp, ki, kd, label in configs:
                # Build PID values list
                pid_values = []
                for i in range(6):
                    if i == joint_idx:
                        pid_values.append(PIDValue(Kp=f64(kp), Ki=f64(ki), Kd=f64(kd)))
                    else:
                        # Stiff defaults for others
                        pid_values.append(PIDValue(Kp=f64(100), Ki=f64(0), Kd=f64(10)))

                # Run Sim (shorter duration for interactive feel)
                t_steps, q_values, *rest = run_pid_controller(
                    setpoints_rad, pid_values, duration=3.0, steps=300
                )
                # q_values shape is (6, N), transform it for 3D animation
                q_traj = np.array(q_values).T
                
                experiments.append({
                    "t": t_steps,
                    "q": q_values[joint_idx], # Extract Joint 2
                    "label": label
                })

            # Plot Results
            self.plot_canvas.plot_tuning_results(experiments, title)
            # Animate the final experiment run as a representation
            if experiments:
                self.animate_3d(t_steps, q_traj)

        except Exception as e:
            self.show_error_message("Tuning Demo Error", f"An error occurred:\n{e}")
        finally:
            self.tuning_run_button.setText("Run Tuning Demo")
            self.tuning_run_button.setEnabled(True)

    def show_error_message(self, title, message):
        msg_box = QtWidgets.QMessageBox(self)
        msg_box.setIcon(QtWidgets.QMessageBox.Icon.Critical)
        msg_box.setWindowTitle(title)
        msg_box.setText(message)
        msg_box.exec()


if __name__ == "__main__":
    app = QtWidgets.QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec())
