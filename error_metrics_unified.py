"""
Unified error metrics: PID-Only and PID+TD3 on the same physics plant.

Both controllers are evaluated through Puma560EnvTD3 so the dynamics,
friction model, motor model, and integration scheme are identical.
PID-only is simulated by supplying zero residual torques — the environment's
internal FastPIDController still runs at 1 ms as normal.

Metrics computed (per joint and overall)
-----------------------------------------
  MSE : Mean Square Error        -- (1/N) * sum(e^2)         [rad^2]
  ISE : Integral Square Error    -- sum(e^2) * dt             [rad^2 * s]
  SSE : Steady-State Error       -- mean(|e|) over last 10 %  [rad]

Usage
-----
  # PID-only, static 25-deg step, Setting A, 5 s
  python error_metrics_unified.py --setting A --duration 5

  # PID + TD3 comparison, sinusoidal trajectory
  python error_metrics_unified.py --setting A --td3 --trajectory sine

  # Both settings, step to 45 deg on J2 only
  python error_metrics_unified.py --setting both --td3 --setpoints "0,45,0,0,0,0"

  # Full 10 s run matching the training distribution
  python error_metrics_unified.py --setting A --td3 --trajectory sine --duration 10
"""

from __future__ import annotations

import argparse
import os
from typing import Optional

import numpy as np
import torch

from rl_env import Puma560EnvTD3
from td3_lstm_models import TD3Actor, device
from trajectory import StaticTrajectory, SineTrajectory

# Puma560EnvTD3 forces rl_decimation=20 -> one outer step = 20 ms.
RL_STEP_DT = 0.001 * 20


# ─────────────────────────────────────────────────────────────────────────────
# Simulation (shared plant)
# ─────────────────────────────────────────────────────────────────────────────

def run_unified_simulation(
    trajectory,
    baseline_setting: str,
    duration_s: float,
    actor: Optional[TD3Actor] = None,
) -> tuple[np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """
    Simulate on Puma560EnvTD3 with or without a TD3 actor.

    PID-only mode: actor=None -> zero residual torques every step.
    PID+TD3 mode:  actor loaded -> actor outputs residual torques.

    Returns
    -------
    time        : 1-D array of timestamps (s).
    error_rad   : List of 6 arrays, per-joint tracking error e = q_ref - q (rad).
    q_ref_rad   : List of 6 arrays, per-joint reference angles (rad).
    """
    env = Puma560EnvTD3(
        dt=0.001,
        window_size=20,
        baseline_setting=baseline_setting,
        trajectory=trajectory,
    )

    max_steps = int(duration_s / RL_STEP_DT)
    state = env.reset(episode=300)   # episode=300 -> full safety-cage alpha (0.30)

    time_log: list[float] = []
    err_log:  list[list[float]] = [[] for _ in range(6)]
    ref_log:  list[list[float]] = [[] for _ in range(6)]

    for k in range(max_steps):
        if actor is not None:
            with torch.no_grad():
                t_in = torch.FloatTensor(state).unsqueeze(0).to(device)
                action: np.ndarray = actor(t_in).cpu().numpy().flatten()
        else:
            action = np.zeros(6, dtype=np.float32)

        state, _, terminated, _truncated, _ = env.step(action, episode=300)
        # Ignore truncation: we want the complete error trace regardless of
        # transient magnitude so the metrics cover the full requested duration.

        # State layout (Puma560EnvTD3._get_state):
        #   [q(6), qd(6), e(6), ed(6), e_int(6), q_ref(6), qd_ref(6)]
        #    0:6  6:12  12:18 18:24   24:30     30:36     36:42
        latest = env.history[-1]
        e_now   = latest[12:18]   # e = q_ref - q  (rad)
        ref_now = latest[30:36]   # q_ref           (rad)

        time_log.append((k + 1) * RL_STEP_DT)
        for i in range(6):
            err_log[i].append(float(e_now[i]))
            ref_log[i].append(float(ref_now[i]))

        if terminated:
            break

    time_arr = np.array(time_log)
    err_arrs = [np.array(err_log[i]) for i in range(6)]
    ref_arrs = [np.array(ref_log[i]) for i in range(6)]
    return time_arr, err_arrs, ref_arrs


# ─────────────────────────────────────────────────────────────────────────────
# Metric computation (matches error_metrics.py definitions)
# ─────────────────────────────────────────────────────────────────────────────

def compute_mse(errors: np.ndarray) -> float:
    """(1/N) * sum(e^2)  [rad^2]"""
    return float(np.mean(errors ** 2))


def compute_ise(errors: np.ndarray, dt: float) -> float:
    """sum(e^2) * dt  [rad^2 * s]"""
    return float(np.sum(errors ** 2) * dt)


def compute_sse(errors: np.ndarray, last_fraction: float = 0.10) -> float:
    """mean(|e|) over last `last_fraction` of the trace  [rad]"""
    window = max(1, int(len(errors) * last_fraction))
    return float(np.mean(np.abs(errors[-window:])))


def compute_metrics(
    error_arrays: list[np.ndarray],
    dt: float,
    sse_window: float = 0.10,
) -> list[dict]:
    return [
        {
            "joint": i + 1,
            "mse": compute_mse(error_arrays[i]),
            "ise": compute_ise(error_arrays[i], dt),
            "sse": compute_sse(error_arrays[i], sse_window),
        }
        for i in range(6)
    ]


# ─────────────────────────────────────────────────────────────────────────────
# Printing (same style as error_metrics.py)
# ─────────────────────────────────────────────────────────────────────────────

def print_table(results: list[dict], label: str) -> tuple[float, float, float]:
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")
    print(f"  {'Joint':<8} {'MSE (rad^2)':<22} {'ISE (rad^2*s)':<22} {'SSE (rad)':<18}")
    print(f"  {'-' * (width - 2)}")
    for r in results:
        print(f"  J{r['joint']:<7} {r['mse']:<22.8f} {r['ise']:<22.8f} {r['sse']:<18.8f}")
    print(f"  {'-' * (width - 2)}")
    overall_mse = float(np.mean([r["mse"] for r in results]))
    overall_ise = float(np.sum([r["ise"] for r in results]))
    overall_sse = float(np.mean([r["sse"] for r in results]))
    print(f"  {'Overall':<8} {overall_mse:<22.8f} {overall_ise:<22.8f} {overall_sse:<18.8f}")
    print(f"{'=' * width}")
    return overall_mse, overall_ise, overall_sse


def _fmt_pct(value: float, col: int = 22) -> str:
    return f"{value:>+10.2f}%{'':{col - 11}}"


def print_improvement_table(
    pid_results: list[dict],
    td3_results: list[dict],
    label: str,
) -> tuple[float, float, float]:
    """Per-joint improvement: positive means TD3 is better."""
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")
    print(f"  {'Joint':<8} {'MSE Improv.':<22} {'ISE Improv.':<22} {'SSE Improv.':<18}")
    print(f"  {'-' * (width - 2)}")

    for pid, td3 in zip(pid_results, td3_results):
        mse_imp = (pid["mse"] - td3["mse"]) / pid["mse"] * 100 if pid["mse"] > 0 else 0.0
        ise_imp = (pid["ise"] - td3["ise"]) / pid["ise"] * 100 if pid["ise"] > 0 else 0.0
        sse_imp = (pid["sse"] - td3["sse"]) / pid["sse"] * 100 if pid["sse"] > 0 else 0.0
        print(f"  J{pid['joint']:<7} {_fmt_pct(mse_imp)} {_fmt_pct(ise_imp)} {_fmt_pct(sse_imp, col=18)}")

    print(f"  {'-' * (width - 2)}")
    overall_pid_mse = float(np.mean([r["mse"] for r in pid_results]))
    overall_pid_ise = float(np.sum([r["ise"] for r in pid_results]))
    overall_pid_sse = float(np.mean([r["sse"] for r in pid_results]))
    overall_td3_mse = float(np.mean([r["mse"] for r in td3_results]))
    overall_td3_ise = float(np.sum([r["ise"] for r in td3_results]))
    overall_td3_sse = float(np.mean([r["sse"] for r in td3_results]))

    ov_mse = (overall_pid_mse - overall_td3_mse) / overall_pid_mse * 100 if overall_pid_mse > 0 else 0.0
    ov_ise = (overall_pid_ise - overall_td3_ise) / overall_pid_ise * 100 if overall_pid_ise > 0 else 0.0
    ov_sse = (overall_pid_sse - overall_td3_sse) / overall_pid_sse * 100 if overall_pid_sse > 0 else 0.0

    print(f"  {'Overall':<8} {_fmt_pct(ov_mse)} {_fmt_pct(ov_ise)} {_fmt_pct(ov_sse, col=18)}")
    print(f"{'=' * width}")
    return ov_mse, ov_ise, ov_sse


# ─────────────────────────────────────────────────────────────────────────────
# Trajectory builders
# ─────────────────────────────────────────────────────────────────────────────

def build_trajectory(
    traj_type: str,
    setpoints_rad: np.ndarray,
    sine_amp_rad: float,
    sine_freq: float,
):
    """
    Build the trajectory passed to Puma560EnvTD3.

    'static' -> StaticTrajectory (step input held constant)
    'sine'   -> SineTrajectory matching the training distribution.

    Note on frequency convention:
      The env's internal random trajectory uses  amp * sin(pi * freq * t).
      SineTrajectory uses                         amp * sin(2*pi * freq * t).
      To match a 1 Hz training signal (pi*1*t), pass sine_freq = 0.5 Hz.
      The default sine_freq=0.5 therefore reproduces the dominant training tone.
    """
    if traj_type == "sine":
        params = [
            (float(np.rad2deg(sine_amp_rad)), float(sine_freq), 0.0, 0.0)
            for _ in range(6)
        ]
        return SineTrajectory(params)
    return StaticTrajectory(setpoints_rad)


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Unified error metrics: PID-Only vs PID+TD3 on the same Puma560EnvTD3 plant. "
            "Both modes run the same simplified dynamics, friction, and motor models."
        ),
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "--setting", default="A", choices=["A", "B", "both"],
        help="PID gain setting.",
    )
    parser.add_argument(
        "--duration", default=10.0, type=float,
        help="Simulation duration in seconds (max effective: 10 s per episode).",
    )
    parser.add_argument(
        "--setpoints", default="0,45,0,0,0,0", type=str,
        help="Comma-separated target angles in degrees (static trajectory only).",
    )
    parser.add_argument(
        "--td3", action="store_true",
        help="Also evaluate the trained TD3 agent (requires checkpoint).",
    )
    parser.add_argument(
        "--sse_window", default=0.1, type=float,
        help="Fraction of final steps used for SSE computation.",
    )
    parser.add_argument(
        "--trajectory", default="static", choices=["static", "sine"],
        help="'static' for step input; 'sine' for sinusoidal (matches training).",
    )
    parser.add_argument(
        "--sine_amp", default=28.6, type=float,
        help="Sine amplitude in degrees (default 28.6 deg = 0.5 rad).",
    )
    parser.add_argument(
        "--sine_freq", default=0.5, type=float,
        help="Sine frequency in Hz for SineTrajectory (0.5 Hz matches training primary tone).",
    )
    parser.add_argument(
        "--checkpoint", default=None, metavar="PATH",
        help="Override TD3 actor checkpoint path.",
    )
    args = parser.parse_args()

    setpoints_deg = np.array([float(v) for v in args.setpoints.split(",")])
    if len(setpoints_deg) != 6:
        raise ValueError("--setpoints must have exactly 6 comma-separated values.")
    setpoints_rad = np.deg2rad(setpoints_deg)
    sine_amp_rad  = np.deg2rad(args.sine_amp)

    trajectory = build_trajectory(
        args.trajectory, setpoints_rad, sine_amp_rad, args.sine_freq
    )

    settings = ["A", "B"] if args.setting == "both" else [args.setting]
    n_steps  = int(args.duration / RL_STEP_DT)
    sse_steps = max(1, int(n_steps * args.sse_window))

    print(f"\nPlant        : Puma560EnvTD3  (same for PID and TD3)")
    print(f"Trajectory   : {args.trajectory.upper()}", end="")
    if args.trajectory == "sine":
        print(
            f" | Amplitude: {args.sine_amp:.1f} deg ({sine_amp_rad:.4f} rad)"
            f" | Frequency: {args.sine_freq} Hz"
        )
    else:
        print(f" | Setpoints: {setpoints_deg} deg")
    print(
        f"Duration     : {args.duration} s  |  RL step dt: {RL_STEP_DT*1000:.0f} ms"
        f"  |  Steps: {n_steps}"
    )
    print(
        f"SSE window   : last {args.sse_window*100:.0f}% of steps"
        f" ({sse_steps} steps = {sse_steps * RL_STEP_DT:.2f} s)"
    )

    summary: list[tuple[str, float, float, float]] = []

    for s in settings:
        # ── PID-Only ──────────────────────────────────────────────────────────
        print(f"\nRunning PID-Only (Setting {s}) on Puma560EnvTD3 ...")
        time_arr, err_pid, _ = run_unified_simulation(
            trajectory, s, args.duration, actor=None
        )
        results_pid = compute_metrics(err_pid, RL_STEP_DT, args.sse_window)
        mse, ise, sse = print_table(results_pid, f"PID-Only (Setting {s})  [Puma560EnvTD3]")
        summary.append((f"PID Setting {s}", mse, ise, sse))

        # ── PID + TD3 ─────────────────────────────────────────────────────────
        if args.td3:
            ckpt = args.checkpoint or f"checkpoints/Setting_{s}/td3_best_actor"
            if not os.path.exists(ckpt):
                print(f"\n  [!] Checkpoint not found: '{ckpt}' — skipping TD3 for Setting {s}.")
                continue

            print(f"\nRunning PID+TD3 (Setting {s}) on Puma560EnvTD3  [checkpoint: {ckpt}] ...")
            max_action = np.array([15.0, 20.0, 15.0, 5.0, 5.0, 3.0])
            actor = TD3Actor(42, 6, max_action).to(device)
            actor.load_state_dict(torch.load(ckpt, map_location=device), strict=False)
            actor.eval()

            _, err_td3, _ = run_unified_simulation(
                trajectory, s, args.duration, actor=actor
            )
            results_td3 = compute_metrics(err_td3, RL_STEP_DT, args.sse_window)
            mse_td3, ise_td3, sse_td3 = print_table(
                results_td3, f"PID+TD3 (Setting {s})  [Puma560EnvTD3]"
            )
            summary.append((f"TD3 Setting {s}", mse_td3, ise_td3, sse_td3))

            print_improvement_table(
                results_pid, results_td3,
                f"Improvement: TD3 vs PID - Setting {s}  (positive = TD3 better)"
            )

    # ── Summary ───────────────────────────────────────────────────────────────
    if len(summary) > 1:
        width = 80
        print(f"\n{'=' * width}")
        print("  SUMMARY")
        print(f"{'=' * width}")
        print(f"  {'Run':<25} {'Overall MSE':<20} {'Overall ISE':<20} {'Overall SSE'}")
        print(f"  {'-' * (width - 2)}")
        for label, mse, ise, sse in summary:
            print(f"  {label:<25} {mse:<20.8f} {ise:<20.8f} {sse:.8f}")
        print(f"{'=' * width}")


if __name__ == "__main__":
    main()
