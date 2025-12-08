import sys

import numpy as np
from matplotlib.backends.backend_qtagg import NavigationToolbar2QT as NavigationToolbar
from numpy import float64 as f64
from PySide6 import QtCore, QtGui, QtWidgets

from canvas import MplCanvas
from dynamics import run_forward_dynamics, run_inverse_dynamics
from pid_controller import PIDValue, run_pid_controller


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

        self.tabs.addTab(self.tab_fd, "Forward Dynamics")
        self.tabs.addTab(self.tab_id, "Inverse Dynamics")
        self.tabs.addTab(self.tab_pc, "PID Controller")

        main_layout.addWidget(self.tabs)

        # --- Populate Tabs ---
        self.setup_fd_tab()
        self.setup_id_tab()
        self.setup_pc_tab()

        # --- Right-side Plotting Area ---
        plot_widget = QtWidgets.QWidget()
        plot_layout = QtWidgets.QVBoxLayout(plot_widget)

        self.plot_canvas = MplCanvas(self, width=8, height=10, dpi=100)
        toolbar = NavigationToolbar(self.plot_canvas, self)

        plot_layout.addWidget(toolbar)
        plot_layout.addWidget(self.plot_canvas)

        main_layout.addWidget(plot_widget)

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

        setpoint_box, self.pc_setpoint_entries = self._create_joint_input_group(
            "Setpoints [Deg]", [0.0] * 6
        )

        kp_box, self.pc_kp_entries = self._create_joint_input_group(
            "Proportional Gain (Kp)", [1.0] * 6
        )

        ki_box, self.pc_ki_entries = self._create_joint_input_group(
            "Integral Gain (Ki)", [0.0] * 6
        )

        kd_box, self.pc_kd_entries = self._create_joint_input_group(
            "Derivative Gain (Kd)", [0.0] * 6
        )

        self.pc_run_button = QtWidgets.QPushButton("Run PID Controller")
        self.pc_run_button.setStyleSheet("padding: 10px; font-weight: bold;")
        self.pc_run_button.clicked.connect(self.on_run_pc)

        scroll = QtWidgets.QScrollArea()
        scroll.setWidgetResizable(True)
        scroll_content = QtWidgets.QWidget()
        scroll_layout = QtWidgets.QVBoxLayout(scroll_content)

        scroll_layout.addWidget(setpoint_box)
        scroll_layout.addWidget(kp_box)
        scroll_layout.addWidget(ki_box)
        scroll_layout.addWidget(kd_box)
        scroll_layout.addWidget(self.pc_run_button)
        scroll_layout.addStretch()

        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)

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

        except Exception as e:
            self.show_error_message("Simulation Error", f"An error occurred:\n{e}")
        finally:
            self.id_run_button.setText("Run Inverse Dynamics")
            self.id_run_button.setEnabled(True)

    def on_run_pc(self):
        self.pc_run_button.setText("Running...")
        self.pc_run_button.setEnabled(False)
        QtCore.QCoreApplication.processEvents()

        try:
            # 1) Read setpoints from UI (in degrees) and convert to radians
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

            # 3) Run simulation
            t_steps, q_values, u_values = run_pid_controller(
                setpoints=setpoints_rad,
                pid_values=pid_values,
            )

            # 4) Plot on the embedded canvas
            self.plot_canvas.plot_pc_results(
                t_steps=t_steps,
                q_values=q_values,
                u_values=u_values,
                setpoints=setpoints_rad,
            )

        except Exception as e:
            self.show_error_message("PID Controller Error", f"An error occurred:\n{e}")
        finally:
            self.pc_run_button.setText("Run PID Controller")
            self.pc_run_button.setEnabled(True)

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
