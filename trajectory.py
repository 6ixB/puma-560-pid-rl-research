"""
Trajectory generators for dynamic PID setpoints.

Provides three modes:
  - StaticTrajectory:   fixed setpoint per joint (original behavior)
  - WaypointTrajectory: piecewise-linear interpolation between (time, angle) pairs
  - SineTrajectory:     sinusoidal setpoint per joint
"""

from enum import Enum

import numpy as np
from numpy import float64 as f64
from numpy.typing import NDArray


class TrajectoryMode(Enum):
    STATIC = "Static"
    WAYPOINT = "Waypoints"
    SINE_WAVE = "Sine Wave"


class TrajectoryGenerator:
    """Base class – subclasses must implement get_setpoint."""

    def get_setpoint(self, t: float) -> NDArray[f64]:
        """Return a 6-element setpoint vector **in radians** at time *t*."""
        raise NotImplementedError

    def get_info(self) -> dict:
        """Return human-readable metadata about the trajectory."""
        return {}


class StaticTrajectory(TrajectoryGenerator):
    """Constant setpoint for each joint (replicates the original behaviour)."""

    def __init__(self, setpoints_rad: NDArray[f64]) -> None:
        self._sp = np.array(setpoints_rad, dtype=f64)

    def get_setpoint(self, t: float) -> NDArray[f64]:  # noqa: ARG002
        return self._sp

    def get_info(self) -> dict:
        return {
            "Mode": "Static",
            "Setpoints (Deg)": [round(float(np.rad2deg(s)), 2) for s in self._sp]
        }


class WaypointTrajectory(TrajectoryGenerator):
    """
    Piecewise-linear interpolation between user-defined waypoints.

    Parameters
    ----------
    waypoints : list[list[tuple[float, float]]]
        Per-joint list of (time_s, angle_deg) pairs.
        Each inner list must be sorted by time and contain ≥ 1 entry.
        Values outside the time range are clamped to the nearest endpoint.
    """

    def __init__(self, waypoints: list[list[tuple[float, float]]]) -> None:
        self._times: list[NDArray[f64]] = []
        self._angles_rad: list[NDArray[f64]] = []
        for joint_wps in waypoints:
            sorted_wps = sorted(joint_wps, key=lambda p: p[0])
            times = np.array([p[0] for p in sorted_wps], dtype=f64)
            angles_deg = np.array([p[1] for p in sorted_wps], dtype=f64)
            self._times.append(times)
            self._angles_rad.append(np.deg2rad(angles_deg))

    def get_setpoint(self, t: float) -> NDArray[f64]:
        sp = np.zeros(6, dtype=f64)
        for i in range(6):
            if i < len(self._times):
                sp[i] = np.interp(t, self._times[i], self._angles_rad[i])
            # joints without waypoints default to 0
        return sp

    def get_info(self) -> dict:
        summary = []
        for i in range(len(self._times)):
            summary.append(f"J{i+1}: {len(self._times[i])} waypoints")
        return {
            "Mode": "Waypoints",
            "Summary": ", ".join(summary)
        }


class SineTrajectory(TrajectoryGenerator):
    """
    Sinusoidal trajectory: offset + amplitude * sin(2π * freq * t + phase).

    All angular parameters are supplied **in degrees** and stored internally
    in radians so that ``get_setpoint`` returns radians.

    Parameters
    ----------
    params : list[tuple[float, float, float, float]]
        Per-joint (amplitude_deg, frequency_hz, offset_deg, phase_deg).
    """

    def __init__(self, params: list[tuple[float, float, float, float]]) -> None:
        # Store (amp_rad, freq_hz, offset_rad, phase_rad) per joint
        self._params: list[tuple[f64, f64, f64, f64]] = []
        for amp_deg, freq, offset_deg, phase_deg in params:
            self._params.append((
                f64(np.deg2rad(amp_deg)),
                f64(freq),
                f64(np.deg2rad(offset_deg)),
                f64(np.deg2rad(phase_deg)),
            ))

    def get_setpoint(self, t: float) -> NDArray[f64]:
        sp = np.zeros(6, dtype=f64)
        for i, (amp, freq, offset, phase) in enumerate(self._params):
            sp[i] = offset + amp * np.sin(2.0 * np.pi * freq * t + phase)
        return sp

    def get_info(self) -> dict:
        summary = []
        for i, (amp, freq, offset, phase) in enumerate(self._params):
            if abs(amp) > 1e-6:
                summary.append(f"J{i+1}: A={np.rad2deg(amp):.1f}°, f={freq:.1f}Hz, off={np.rad2deg(offset):.1f}°, ph={np.rad2deg(phase):.1f}°")
        return {
            "Mode": "Sine Wave",
            "Params": " | ".join(summary) if summary else "All zeros"
        }
