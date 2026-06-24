"""
Rise time analysis algorithms for PUMA 560 step response evaluation.

Metrics computed per joint
--------------------------
  rise_time_10_90  : Elapsed time from the 10 % to the 90 % final-value crossing (s).
  rise_time_0_100  : Elapsed time from the first 0-crossing to the setpoint crossing (s).
  delay_time       : Time at which the response first crosses 50 % of the setpoint (s).
  settling_time_2% : Last instant the response was outside the ±2 % tolerance band (s).
  settling_time_5% : Same for ±5 % tolerance (s).
  percent_overshoot: (peak - setpoint) / |setpoint| × 100  (0 if no overshoot).
  peak_time        : Time at which the absolute maximum / minimum occurs (s).
  peak_value       : Value at peak_time (rad).
  steady_state_error: |mean(last 10 % of trace) − setpoint| (rad).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# ─────────────────────────────────────────────────────────────────────────────
# Data container
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class StepResponseMetrics:
    joint_idx: int              # 0-based joint index
    setpoint_rad: float         # commanded step value (rad)
    final_value_rad: float      # mean of last 10 % of trace (rad)
    rise_time_10_90: Optional[float]    # s
    rise_time_0_100: Optional[float]    # s
    delay_time: Optional[float]         # s  (50 % crossing)
    settling_time_2pct: Optional[float] # s
    settling_time_5pct: Optional[float] # s
    percent_overshoot: float            # %
    peak_time: Optional[float]          # s
    peak_value_rad: float               # rad
    steady_state_error_rad: float       # rad


# ─────────────────────────────────────────────────────────────────────────────
# Low-level helpers
# ─────────────────────────────────────────────────────────────────────────────

def _linear_crossing(
    time: np.ndarray,
    signal: np.ndarray,
    threshold: float,
) -> Optional[float]:
    """
    Return the linearly-interpolated time of the *first* crossing of `threshold`
    in either direction.  Returns None if no crossing is found.
    """
    for k in range(len(signal) - 1):
        a, b = float(signal[k]), float(signal[k + 1])
        # Zero-product test: one side is on or below, the other is on or above.
        if (a - threshold) * (b - threshold) <= 0 and (b - a) != 0.0:
            frac = (threshold - a) / (b - a)
            return float(time[k]) + frac * (float(time[k + 1]) - float(time[k]))
    return None


# ─────────────────────────────────────────────────────────────────────────────
# Public metric functions
# ─────────────────────────────────────────────────────────────────────────────

def compute_rise_time(
    time: np.ndarray,
    response: np.ndarray,
    setpoint: float,
    low_pct: float = 0.10,
    high_pct: float = 0.90,
) -> tuple[Optional[float], Optional[float]]:
    """
    Compute 10-90 % and 0-100 % rise times.

    Parameters
    ----------
    time, response : Same-length 1-D arrays.
    setpoint       : Target step value (rad).  Must be non-zero.
    low_pct        : Lower percentage threshold (default 0.10).
    high_pct       : Upper percentage threshold (default 0.90).

    Returns
    -------
    (rise_time_10_90, rise_time_0_100) in seconds; None if threshold not crossed.
    """
    if abs(setpoint) < 1e-9:
        return None, None

    t_lo = _linear_crossing(time, response, low_pct * setpoint)
    t_hi = _linear_crossing(time, response, high_pct * setpoint)

    if t_lo is not None and t_hi is not None and t_hi > t_lo:
        rt_10_90: Optional[float] = t_hi - t_lo
    else:
        rt_10_90 = None

    # 0–100 %: from the first 0-crossing (or time[0]) to first setpoint crossing.
    t_zero = _linear_crossing(time, response, 0.0)
    t_full = _linear_crossing(time, response, setpoint)
    if t_full is not None:
        origin = t_zero if t_zero is not None else float(time[0])
        rt_0_100: Optional[float] = t_full - origin
    else:
        rt_0_100 = None

    return rt_10_90, rt_0_100


def compute_settling_time(
    time: np.ndarray,
    response: np.ndarray,
    setpoint: float,
    tolerance: float = 0.02,
) -> Optional[float]:
    """
    Return the latest time at which the response was outside the
    ±`tolerance`·|setpoint| band around `setpoint`.

    Returns
    -------
    float : time of the sample *after* the last out-of-band sample.
    time[0] if the response is always within band.
    None    if the response never enters the band.
    """
    if abs(setpoint) < 1e-9:
        return None

    band = tolerance * abs(setpoint)
    outside = np.abs(response - setpoint) > band

    if not np.any(outside):
        return float(time[0])   # always within band

    last_out = int(np.where(outside)[0][-1])
    next_in = last_out + 1

    if next_in >= len(time):
        return None             # never fully settled within the record

    return float(time[next_in])


def compute_overshoot(response: np.ndarray, setpoint: float) -> float:
    """
    Percent overshoot relative to |setpoint|.
    Returns 0 when no overshoot occurs.
    """
    if abs(setpoint) < 1e-9:
        return 0.0

    if setpoint > 0:
        peak = float(np.max(response))
        return max(0.0, (peak - setpoint) / setpoint * 100.0)
    else:
        peak = float(np.min(response))
        return max(0.0, (setpoint - peak) / abs(setpoint) * 100.0)


def compute_peak_time(
    time: np.ndarray,
    response: np.ndarray,
    setpoint: float,
) -> tuple[Optional[float], float]:
    """
    Return (peak_time, peak_value).
    For positive setpoints the peak is the maximum; negative uses the minimum.
    """
    if setpoint >= 0:
        idx = int(np.argmax(response))
    else:
        idx = int(np.argmin(response))
    return float(time[idx]), float(response[idx])


# ─────────────────────────────────────────────────────────────────────────────
# Main analysis entry point
# ─────────────────────────────────────────────────────────────────────────────

def analyze_step_response(
    time: np.ndarray,
    response: np.ndarray,
    setpoint: float,
    joint_idx: int = 0,
) -> StepResponseMetrics:
    """
    Compute all step-response performance metrics for one joint.

    Parameters
    ----------
    time      : 1-D monotone time array (seconds).
    response  : 1-D joint angle array (radians), same length as `time`.
    setpoint  : Commanded step angle (radians).
    joint_idx : 0-based joint label (default 0).

    Returns
    -------
    StepResponseMetrics dataclass populated with every metric.
    """
    if len(time) < 2 or len(response) < 2:
        raise ValueError("time and response must each contain at least 2 samples.")
    if len(time) != len(response):
        raise ValueError("time and response must have the same length.")

    # Steady-state estimate: mean of the last 10 % of the trace.
    tail_start = max(1, int(0.90 * len(response)))
    final_value = float(np.mean(response[tail_start:]))
    ss_error = float(abs(final_value - setpoint))

    rt_10_90, rt_0_100 = compute_rise_time(time, response, setpoint)

    delay_time = (
        _linear_crossing(time, response, 0.5 * setpoint)
        if abs(setpoint) > 1e-9 else None
    )

    st_2pct = compute_settling_time(time, response, setpoint, tolerance=0.02)
    st_5pct = compute_settling_time(time, response, setpoint, tolerance=0.05)

    overshoot = compute_overshoot(response, setpoint)
    peak_time, peak_value = compute_peak_time(time, response, setpoint)

    return StepResponseMetrics(
        joint_idx=joint_idx,
        setpoint_rad=float(setpoint),
        final_value_rad=final_value,
        rise_time_10_90=rt_10_90,
        rise_time_0_100=rt_0_100,
        delay_time=delay_time,
        settling_time_2pct=st_2pct,
        settling_time_5pct=st_5pct,
        percent_overshoot=overshoot,
        peak_time=peak_time,
        peak_value_rad=peak_value,
        steady_state_error_rad=ss_error,
    )


# ─────────────────────────────────────────────────────────────────────────────
# Formatting
# ─────────────────────────────────────────────────────────────────────────────

def format_metrics_table(
    metrics_list: list[StepResponseMetrics],
    controller_name: str,
) -> str:
    """
    Render a list of StepResponseMetrics as a fixed-width ASCII table.

    Columns: Joint | SP(°) | Final(°) | RT10-90(s) | RT0-100(s) | ST2%(s) | OS(%) | SS-Err(°)
    """
    SEP = "=" * 92

    def fv(x: Optional[float], w: int, d: int = 4) -> str:
        return f"{'N/A':>{w}}" if x is None else f"{x:>{w}.{d}f}"

    header = (
        f"\n{SEP}\n"
        f"  {controller_name}  --  Step Response Metrics\n"
        f"{SEP}\n"
        f"  {'Joint':^6}  {'SP(deg)':>7}  {'Final(deg)':>10}  {'RT10-90(s)':>11}"
        f"  {'RT0-100(s)':>11}  {'ST 2pct(s)':>10}  {'OS(pct)':>8}  {'SS-Err(deg)':>11}\n"
        f"  {'-'*90}"
    )

    rows = [header]
    for m in metrics_list:
        sp_d    = np.rad2deg(m.setpoint_rad)
        final_d = np.rad2deg(m.final_value_rad)
        sse_d   = np.rad2deg(m.steady_state_error_rad)
        rows.append(
            f"  {'J' + str(m.joint_idx + 1):^6}  {sp_d:>7.1f}  {final_d:>10.3f}"
            f"  {fv(m.rise_time_10_90,  11, 4)}  {fv(m.rise_time_0_100, 11, 4)}"
            f"  {fv(m.settling_time_2pct, 10, 4)}  {m.percent_overshoot:>8.1f}"
            f"  {sse_d:>11.5f}"
        )

    rows.append(f"  {'-'*90}")
    return "\n".join(rows)
