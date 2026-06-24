"""
Rise time performance analysis — PID-Only vs PID+TD3 on PUMA 560.

Runs step response simulations for all 6 joints, computes rise time metrics,
generates two comparison figures, and exports results to CSV.

Usage
-----
  python run_rise_time_analysis.py                   # PID+RL, 25°, Setting A, 5 s
  python run_rise_time_analysis.py --step-deg 30     # 30-degree step
  python run_rise_time_analysis.py --no-rl           # PID-only (no checkpoint needed)
  python run_rise_time_analysis.py --setting B       # PID Setting B checkpoints
  python run_rise_time_analysis.py --duration 8      # longer simulation window

Notes
-----
- Truncation (large initial error) is *ignored* during evaluation so the full
  step response is always recorded regardless of step size.
- Puma560EnvTD3 forces rl_decimation=20 and lstm_decimation=20; one RL step
  therefore spans 20 ms (20 × 1 ms PID micro-steps).
- The PID-only baseline is simulated by sending zero residual torques, so the
  environment physics are identical between the two modes.
"""

from __future__ import annotations

import argparse
import csv
import os
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import torch

from rl_env import Puma560EnvTD3
from td3_lstm_models import TD3Actor, device
from trajectory import StaticTrajectory
from rise_time_analysis import (
    StepResponseMetrics,
    analyze_step_response,
    format_metrics_table,
)

# ─────────────────────────────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────────────────────────────

JOINT_NAMES = [
    "J1 Waist",
    "J2 Shoulder",
    "J3 Elbow",
    "J4 Wrist-Roll",
    "J5 Wrist-Pitch",
    "J6 Wrist-Yaw",
]

# Puma560EnvTD3 forces rl_decimation = 20  →  one RL step = 20 ms.
RL_STEP_DT = 0.001 * 20

PID_COLOR = "#1f77b4"
RL_COLOR  = "#d62728"
SP_COLOR  = "#2ca02c"


# ─────────────────────────────────────────────────────────────────────────────
# Simulation
# ─────────────────────────────────────────────────────────────────────────────

def run_step_simulation(
    step_rad: np.ndarray,
    baseline_setting: str,
    duration_s: float,
    actor: Optional[TD3Actor] = None,
) -> tuple[np.ndarray, list[np.ndarray]]:
    """
    Simulate a step input on all 6 joints simultaneously.

    Parameters
    ----------
    step_rad         : 6-element target angle array (rad).
    baseline_setting : PID gain set — 'A' or 'B'.
    duration_s       : Wall-clock duration of the simulation (s).
    actor            : Pre-loaded TD3Actor for PID+RL mode.
                       Pass None for PID-only (zero residual torques).

    Returns
    -------
    time : 1-D time array (s).
    q    : List of 6 arrays with per-joint angle traces (rad).
    """
    env = Puma560EnvTD3(
        dt=0.001,
        window_size=20,
        baseline_setting=baseline_setting,
        trajectory=StaticTrajectory(step_rad),
    )

    max_steps = int(duration_s / RL_STEP_DT)
    state = env.reset(episode=300)  # episode=300 → full safety-cage alpha

    time_log: list[float] = []
    q_log: list[list[float]] = [[] for _ in range(6)]

    for k in range(max_steps):
        if actor is not None:
            with torch.no_grad():
                t_in = torch.FloatTensor(state).unsqueeze(0).to(device)
                action: np.ndarray = actor(t_in).cpu().numpy().flatten()
        else:
            action = np.zeros(6, dtype=np.float32)

        state, _, terminated, _truncated, _ = env.step(action, episode=300)
        # Truncation (large transient error) is intentionally ignored so the
        # full response trace is captured for rise/settling time computation.

        time_log.append((k + 1) * RL_STEP_DT)
        q_now = env.history[-1, :6]   # most-recent sample in LSTM history
        for i in range(6):
            q_log[i].append(float(q_now[i]))

        if terminated:
            break

    return np.array(time_log), [np.array(q_log[i]) for i in range(6)]


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _metrics_text(m: StepResponseMetrics, color: str) -> str:
    """Short annotation string drawn on a step-response subplot."""
    rt  = f"{m.rise_time_10_90:.4f} s" if m.rise_time_10_90  is not None else "N/A"
    st  = f"{m.settling_time_2pct:.4f} s" if m.settling_time_2pct is not None else "N/A"
    sse = f"{np.rad2deg(m.steady_state_error_rad):.4f} deg"
    return f"RT10-90 = {rt}\nST +/-2% = {st}\nOS      = {m.percent_overshoot:.1f} %\nSS-Err  = {sse}"


def plot_step_responses(
    time_pid: np.ndarray,
    q_pid: list[np.ndarray],
    metrics_pid: list[StepResponseMetrics],
    step_rad: np.ndarray,
    time_rl: Optional[np.ndarray],
    q_rl: Optional[list[np.ndarray]],
    metrics_rl: Optional[list[StepResponseMetrics]],
    setting: str,
) -> plt.Figure:
    """
    One subplot per joint: PID and PID+TD3 overlaid on the same axes.
    Horizontal guides at 10 %, 90 %, and 100 % of the setpoint.
    ±2 % settling band shaded in green.
    """
    has_rl = (time_rl is not None) and (q_rl is not None) and (metrics_rl is not None)
    fig, axes = plt.subplots(6, 1, figsize=(10, 4 * 6), constrained_layout=True)

    for i, ax in enumerate(axes):
        sp_deg = np.rad2deg(step_rad[i])
        sp_rad = float(step_rad[i])

        # Step-response trace — PID
        ax.plot(time_pid, np.rad2deg(q_pid[i]),
                color=PID_COLOR, linewidth=1.6, label="PID-Only", zorder=3)

        # Step-response trace — PID+TD3
        if has_rl:
            ax.plot(time_rl, np.rad2deg(q_rl[i]),
                    color=RL_COLOR, linewidth=1.6, label="PID+TD3", zorder=4)

        # Setpoint line
        ax.axhline(sp_deg, color=SP_COLOR, linestyle="--", linewidth=1.2,
                   label="Setpoint", zorder=2)

        # Horizontal guides: 10 %, 90 %
        if abs(sp_rad) > 1e-9:
            for pct, ls in [(0.10, ":"), (0.90, "--")]:
                ax.axhline(pct * sp_deg, color="grey", linewidth=0.7,
                           linestyle=ls, alpha=0.5)

        # ±2 % settling band
        if abs(sp_rad) > 1e-9:
            band = 0.02 * abs(sp_deg)
            ax.axhspan(sp_deg - band, sp_deg + band,
                       alpha=0.08, color=SP_COLOR, zorder=1)

        # Annotation box — PID metrics
        pid_ann = _metrics_text(metrics_pid[i], PID_COLOR)
        ax.text(0.01, 0.97, pid_ann, transform=ax.transAxes,
                ha="left", va="top", fontsize=7.5, color=PID_COLOR,
                bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec=PID_COLOR),
                zorder=5)

        # Annotation box — RL metrics
        if has_rl:
            rl_ann = _metrics_text(metrics_rl[i], RL_COLOR)
            ax.text(0.22, 0.97, rl_ann, transform=ax.transAxes,
                    ha="left", va="top", fontsize=7.5, color=RL_COLOR,
                    bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.8, ec=RL_COLOR),
                    zorder=5)

        ax.set_title(
            f"{JOINT_NAMES[i]}  —  Step to {sp_deg:.1f}°  (Setting {setting})",
            fontsize=9, pad=4,
        )
        ax.set_ylabel("Angle (°)", fontsize=8)
        ax.set_xlabel("Time (s)", fontsize=8)
        ax.legend(fontsize=7.5, loc="lower right")
        ax.grid(True, alpha=0.25)

    return fig


def plot_metric_barchart(
    metrics_pid: list[StepResponseMetrics],
    metrics_rl: list[StepResponseMetrics],
    setting: str,
) -> plt.Figure:
    """
    Side-by-side bar charts: Rise Time, Settling Time, Overshoot, SS-Error.
    """
    joints = [f"J{i+1}" for i in range(6)]
    x = np.arange(6)
    w = 0.38

    def safe(lst: list[Optional[float]]) -> list[float]:
        return [v if v is not None else 0.0 for v in lst]

    rt_pid  = safe([m.rise_time_10_90   for m in metrics_pid])
    rt_rl   = safe([m.rise_time_10_90   for m in metrics_rl])
    st_pid  = safe([m.settling_time_2pct for m in metrics_pid])
    st_rl   = safe([m.settling_time_2pct for m in metrics_rl])
    os_pid  = [m.percent_overshoot              for m in metrics_pid]
    os_rl   = [m.percent_overshoot              for m in metrics_rl]
    sse_pid = [np.rad2deg(m.steady_state_error_rad) for m in metrics_pid]
    sse_rl  = [np.rad2deg(m.steady_state_error_rad) for m in metrics_rl]

    fig, axes = plt.subplots(1, 4, figsize=(18, 4.5), constrained_layout=True)

    panels = [
        (rt_pid,  rt_rl,  "Rise Time (10–90 %)",  "Time (s)"),
        (st_pid,  st_rl,  "Settling Time (±2 %)",  "Time (s)"),
        (os_pid,  os_rl,  "Percent Overshoot",      "%"),
        (sse_pid, sse_rl, "Steady-State Error",     "Degrees (°)"),
    ]

    for ax, (y_pid, y_rl, title, ylabel) in zip(axes, panels):
        b1 = ax.bar(x - w / 2, y_pid, w, label="PID-Only", color=PID_COLOR, alpha=0.85)
        b2 = ax.bar(x + w / 2, y_rl,  w, label="PID+TD3",  color=RL_COLOR,  alpha=0.85)
        ax.set_title(title, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=8)
        ax.set_xticks(x)
        ax.set_xticklabels(joints, fontsize=8)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.25, axis="y")

        for bar_group in (b1, b2):
            for rect in bar_group:
                h = rect.get_height()
                if h > 1e-6:
                    ax.text(
                        rect.get_x() + rect.get_width() / 2.0,
                        h + 0.002 * max(max(y_pid), max(y_rl), 1e-6),
                        f"{h:.3f}",
                        ha="center", va="bottom", fontsize=6, rotation=45,
                    )

    fig.suptitle(
        f"Rise Time Metric Comparison  —  Setting {setting}",
        fontsize=12, y=1.02,
    )
    return fig


# ─────────────────────────────────────────────────────────────────────────────
# CSV export
# ─────────────────────────────────────────────────────────────────────────────

def save_csv(
    metrics_pid: list[StepResponseMetrics],
    metrics_rl: Optional[list[StepResponseMetrics]],
    csv_path: str,
) -> None:
    """Write all metrics to a flat CSV file."""
    fieldnames = [
        "controller", "joint",
        "setpoint_deg", "final_value_deg",
        "rise_time_10_90_s", "rise_time_0_100_s", "delay_time_s",
        "settling_time_2pct_s", "settling_time_5pct_s",
        "percent_overshoot",
        "peak_time_s", "peak_value_deg",
        "steady_state_error_deg",
    ]

    def to_row(ctrl: str, m: StepResponseMetrics) -> dict:
        return {
            "controller":           ctrl,
            "joint":                f"J{m.joint_idx + 1}",
            "setpoint_deg":         round(np.rad2deg(m.setpoint_rad),          4),
            "final_value_deg":      round(np.rad2deg(m.final_value_rad),        4),
            "rise_time_10_90_s":    m.rise_time_10_90,
            "rise_time_0_100_s":    m.rise_time_0_100,
            "delay_time_s":         m.delay_time,
            "settling_time_2pct_s": m.settling_time_2pct,
            "settling_time_5pct_s": m.settling_time_5pct,
            "percent_overshoot":    round(m.percent_overshoot,                  4),
            "peak_time_s":          m.peak_time,
            "peak_value_deg":       round(np.rad2deg(m.peak_value_rad),         4),
            "steady_state_error_deg": round(np.rad2deg(m.steady_state_error_rad), 6),
        }

    rows = [to_row("PID-Only", m) for m in metrics_pid]
    if metrics_rl:
        rows += [to_row("PID+TD3", m) for m in metrics_rl]

    with open(csv_path, "w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


# ─────────────────────────────────────────────────────────────────────────────
# Comparison table (stdout)
# ─────────────────────────────────────────────────────────────────────────────

def _print_comparison(
    metrics_pid: list[StepResponseMetrics],
    metrics_rl:  list[StepResponseMetrics],
) -> None:
    """Print a delta-table: improvement (+) or degradation (−) per joint."""
    SEP = "-" * 80
    print(f"\n{SEP}")
    print("  Improvement Summary (positive = RL is faster / more accurate)")
    print(SEP)
    print(
        f"  {'Joint':<18}  {'D Rise-Time':>13}  "
        f"{'D Settle-2%':>13}  {'D Overshoot':>12}  {'D SS-Error':>12}"
    )
    print(f"  {'-'*76}")

    for pm, rm in zip(metrics_pid, metrics_rl):
        name = JOINT_NAMES[pm.joint_idx]

        if pm.rise_time_10_90 and rm.rise_time_10_90:
            d_rt = (pm.rise_time_10_90 - rm.rise_time_10_90) / pm.rise_time_10_90 * 100
            drt_s = f"{d_rt:+.1f} %"
        else:
            drt_s = "N/A"

        if pm.settling_time_2pct and rm.settling_time_2pct:
            d_st = (pm.settling_time_2pct - rm.settling_time_2pct) / pm.settling_time_2pct * 100
            dst_s = f"{d_st:+.1f} %"
        else:
            dst_s = "N/A"

        d_os  = rm.percent_overshoot - pm.percent_overshoot
        d_sse = (
            np.rad2deg(pm.steady_state_error_rad) - np.rad2deg(rm.steady_state_error_rad)
        )

        print(
            f"  {name:<18}  {drt_s:>13}  {dst_s:>13}"
            f"  {d_os:>+11.1f} %  {d_sse:>+11.4f} deg"
        )

    print(f"  {'-'*76}")
    print("  Note: D Rise-Time/Settle > 0 means RL is faster; D OS < 0 means less overshoot.")


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="Rise time analysis — PID-Only vs PID+TD3 on PUMA 560.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--step-deg", type=float, default=25.0, metavar="DEG",
        help="Step amplitude applied to all joints (degrees).",
    )
    parser.add_argument(
        "--setting", default="A", choices=["A", "B"],
        help="PID gain setting (A = stronger, B = ~80 %% of A).",
    )
    parser.add_argument(
        "--duration", type=float, default=5.0, metavar="SECS",
        help="Simulation duration (seconds).",
    )
    parser.add_argument(
        "--no-rl", action="store_true",
        help="Run PID-only simulation, skip the RL evaluation.",
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Override TD3 actor checkpoint path.",
    )
    args = parser.parse_args()

    step_rad = np.ones(6) * np.deg2rad(args.step_deg)

    print(f"\n{'=' * 60}")
    print("  PUMA 560  --  Rise Time Analysis")
    print(f"{'=' * 60}")
    print(f"  Step size  : {args.step_deg:.1f} deg  ({np.deg2rad(args.step_deg):.4f} rad)")
    print(f"  Setting    : {args.setting}")
    print(f"  Duration   : {args.duration:.1f} s")
    print(f"  RL step dt : {RL_STEP_DT * 1000:.0f} ms")
    print(f"  Max steps  : {int(args.duration / RL_STEP_DT)}")
    print(f"{'=' * 60}")

    # -- 1. PID-Only simulation
    print(f"\n[1/2] PID-Only step simulation ...")
    time_pid, q_pid = run_step_simulation(
        step_rad, args.setting, args.duration, actor=None
    )
    metrics_pid = [
        analyze_step_response(time_pid, q_pid[i], float(step_rad[i]), i)
        for i in range(6)
    ]
    print(format_metrics_table(metrics_pid, f"PID-Only  (Setting {args.setting})"))

    # ── 2. PID+TD3 simulation ─────────────────────────────────────────────────
    time_rl:    Optional[np.ndarray]             = None
    q_rl:       Optional[list[np.ndarray]]       = None
    metrics_rl: Optional[list[StepResponseMetrics]] = None

    if not args.no_rl:
        ckpt = args.checkpoint or f"checkpoints/Setting_{args.setting}/td3_best_actor"

        if not os.path.exists(ckpt):
            print(
                f"\n[!] Checkpoint not found: '{ckpt}'\n"
                "    Train the TD3 agent first, or use --no-rl for PID-only analysis."
            )
        else:
            print(f"\n[2/2] PID+TD3 step simulation  (checkpoint: {ckpt}) ...")
            max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
            actor = TD3Actor(42, 6, max_action).to(device)
            actor.load_state_dict(torch.load(ckpt, map_location=device))
            actor.eval()

            time_rl, q_rl = run_step_simulation(
                step_rad, args.setting, args.duration, actor=actor
            )
            metrics_rl = [
                analyze_step_response(time_rl, q_rl[i], float(step_rad[i]), i)
                for i in range(6)
            ]
            print(format_metrics_table(metrics_rl, f"PID+TD3   (Setting {args.setting})"))

    # ── 3. Delta summary ─────────────────────────────────────────────────────
    if metrics_rl is not None:
        _print_comparison(metrics_pid, metrics_rl)

    # ── 4. Plots ──────────────────────────────────────────────────────────────
    print("\nGenerating figures ...")

    fig1 = plot_step_responses(
        time_pid, q_pid, metrics_pid,
        step_rad,
        time_rl, q_rl, metrics_rl,
        args.setting,
    )
    fig1_path = f"rise_time_step_response_Setting_{args.setting}.png"
    fig1.savefig(fig1_path, dpi=130, bbox_inches="tight")
    print(f"  Saved -> {fig1_path}")

    if metrics_rl is not None:
        fig2 = plot_metric_barchart(metrics_pid, metrics_rl, args.setting)
        fig2_path = f"rise_time_comparison_Setting_{args.setting}.png"
        fig2.savefig(fig2_path, dpi=130, bbox_inches="tight")
        print(f"  Saved -> {fig2_path}")

    # ── 5. CSV ────────────────────────────────────────────────────────────────
    csv_path = f"rise_time_results_Setting_{args.setting}.csv"
    save_csv(metrics_pid, metrics_rl, csv_path)
    print(f"  Saved -> {csv_path}")

    print("\nDone.")
    try:
        plt.show()
    except Exception:
        pass


if __name__ == "__main__":
    main()
