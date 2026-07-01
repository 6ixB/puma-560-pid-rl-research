import argparse
import numpy as np

from generate_offline_data import get_baseline_gains
from pid_controller import PIDValue, run_pid_controller
from trajectory import StaticTrajectory, SineTrajectory


def compute_mse(errors):
    """Mean Square Error: (1/N) * sum(e^2)"""
    e = np.array(errors)
    return float(np.mean(e ** 2))


def compute_ise(errors, dt):
    """Integral Square Error: integral of e^2 dt ≈ sum(e^2) * dt"""
    e = np.array(errors)
    return float(np.sum(e ** 2) * dt)


def compute_sse(errors, last_fraction=0.1):
    """Steady-State Error: mean absolute error over the last `last_fraction` of steps."""
    e = np.array(errors)
    window = max(1, int(len(e) * last_fraction))
    return float(np.mean(np.abs(e[-window:])))


def compute_metrics(error_values, dt, sse_window=0.1):
    results = []
    for j in range(6):
        results.append({
            'joint': j + 1,
            'mse': compute_mse(error_values[j]),
            'ise': compute_ise(error_values[j], dt),
            'sse': compute_sse(error_values[j], sse_window),
        })
    return results


def print_table(results, label):
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")
    print(f"  {'Joint':<8} {'MSE (rad^2)':<22} {'ISE (rad^2*s)':<22} {'SSE (rad)':<18}")
    print(f"  {'-' * (width - 2)}")
    for r in results:
        print(f"  J{r['joint']:<7} {r['mse']:<22.8f} {r['ise']:<22.8f} {r['sse']:<18.8f}")
    print(f"  {'-' * (width - 2)}")
    overall_mse = float(np.mean([r['mse'] for r in results]))
    overall_ise = float(np.sum([r['ise'] for r in results]))
    overall_sse = float(np.mean([r['sse'] for r in results]))
    print(f"  {'Overall':<8} {overall_mse:<22.8f} {overall_ise:<22.8f} {overall_sse:<18.8f}")
    print(f"{'=' * width}")
    return overall_mse, overall_ise, overall_sse


def _fmt_pct(value, col=22):
    """Format a percentage value into a fixed-width column string."""
    return f"{value:>+10.2f}%{'':{col - 11}}"


def print_improvement_table(pid_results, td3_results, label):
    """Print a per-joint improvement table comparing PID vs TD3."""
    width = 72
    print(f"\n{'=' * width}")
    print(f"  {label}")
    print(f"{'=' * width}")
    print(f"  {'Joint':<8} {'MSE Improv.':<22} {'ISE Improv.':<22} {'SSE Improv.':<18}")
    print(f"  {'-' * (width - 2)}")

    for pid, td3 in zip(pid_results, td3_results):
        mse_imp = (pid['mse'] - td3['mse']) / pid['mse'] * 100 if pid['mse'] > 0 else 0.0
        ise_imp = (pid['ise'] - td3['ise']) / pid['ise'] * 100 if pid['ise'] > 0 else 0.0
        sse_imp = (pid['sse'] - td3['sse']) / pid['sse'] * 100 if pid['sse'] > 0 else 0.0
        print(f"  J{pid['joint']:<7} {_fmt_pct(mse_imp)} {_fmt_pct(ise_imp)} {_fmt_pct(sse_imp, col=18)}")

    print(f"  {'-' * (width - 2)}")

    overall_pid_mse = float(np.mean([r['mse'] for r in pid_results]))
    overall_pid_ise = float(np.sum([r['ise'] for r in pid_results]))
    overall_pid_sse = float(np.mean([r['sse'] for r in pid_results]))
    overall_td3_mse = float(np.mean([r['mse'] for r in td3_results]))
    overall_td3_ise = float(np.sum([r['ise'] for r in td3_results]))
    overall_td3_sse = float(np.mean([r['sse'] for r in td3_results]))

    ov_mse_imp = (overall_pid_mse - overall_td3_mse) / overall_pid_mse * 100 if overall_pid_mse > 0 else 0.0
    ov_ise_imp = (overall_pid_ise - overall_td3_ise) / overall_pid_ise * 100 if overall_pid_ise > 0 else 0.0
    ov_sse_imp = (overall_pid_sse - overall_td3_sse) / overall_pid_sse * 100 if overall_pid_sse > 0 else 0.0

    print(f"  {'Overall':<8} {_fmt_pct(ov_mse_imp)} {_fmt_pct(ov_ise_imp)} {_fmt_pct(ov_sse_imp, col=18)}")
    print(f"{'=' * width}")
    return ov_mse_imp, ov_ise_imp, ov_sse_imp


def build_trajectory(traj_type, setpoints_rad, sine_amp_rad, sine_freq):
    """Build the trajectory object based on the selected type."""
    if traj_type == 'sine':
        # Single-tone sinusoidal approximation of the training reference in
        # rl_env.py (q_ref = amp*sin(pi*freq*t) + 0.2*sin(0.6*pi*freq*t)).
        # SineTrajectory yields offset + amp*sin(2*pi*freq*t + phase); with the
        # default sine_freq=0.5 the primary tone (amp at 0.5 Hz) matches the
        # training signal's dominant component. The secondary 0.2*amp tone is
        # omitted. Applied equally to all 6 joints.
        params = []
        for _ in range(6):
            params.append((
                float(np.rad2deg(sine_amp_rad)),        # amp in degrees
                float(sine_freq),                        # primary frequency Hz
                0.0,                                     # offset
                0.0                                      # phase
            ))
        return SineTrajectory(params)
    else:
        return StaticTrajectory(setpoints_rad)


def run_pid_metrics(setting, trajectory, duration, dt, sse_window=0.1):
    gains = get_baseline_gains(setting)
    pid_values = [
        PIDValue(Kp=np.float64(g.Kp), Ki=np.float64(g.Ki), Kd=np.float64(g.Kd))
        for g in gains
    ]
    # Use first setpoint value as a dummy for the legacy static fallback
    setpoints_rad = np.zeros(6)
    _, _, error_values, *_ = run_pid_controller(
        setpoints=setpoints_rad,
        pid_values=pid_values,
        duration=duration,
        dt=dt,
        trajectory=trajectory,
    )
    return compute_metrics(error_values, dt, sse_window)


def run_td3_metrics(setting, trajectory, duration, dt, sse_window=0.1):
    from simulate_td3 import run_td3_simulation
    from rl_env import Puma560EnvTD3

    # run_td3_simulation drives Puma560EnvTD3, whose outer loop advances
    # env.dt * env.rl_decimation seconds per step (20 ms) regardless of the PID
    # `dt`. That function sizes its loop and labels its t_steps as if each step
    # were 10 ms, so (a) its returned t_steps are mislabeled and (b) it actually
    # simulates ~2x `duration`. Since we must not modify that file, we correct
    # both issues here: derive the true step time from the env, and keep only the
    # first `duration` seconds of samples so the TD3 metrics cover the same
    # real-time horizon (and identical ISE/SSE windows) as the PID baseline.
    probe = Puma560EnvTD3(dt=0.001, baseline_setting=setting)
    td3_dt = float(probe.dt * probe.rl_decimation)

    _, _, error_values_deg, *_ = run_td3_simulation(
        baseline_setting=setting,
        duration=duration,
        dt=dt,
        q0_rad=np.zeros(6),
        trajectory=trajectory,
    )

    # Trim to the first `duration` seconds and integrate ISE with the true TD3 dt.
    n_keep = max(1, int(round(duration / td3_dt)))
    # run_td3_simulation returns error in degrees; convert to radians.
    error_values_rad = [np.deg2rad(np.array(e[:n_keep])) for e in error_values_deg]
    return compute_metrics(error_values_rad, td3_dt, sse_window)


def main():
    parser = argparse.ArgumentParser(description="Compute MSE, ISE, and SSE tracking error metrics.")
    parser.add_argument("--setting", default="A", choices=["A", "B", "both"],
                        help="PID baseline setting (default: A)")
    parser.add_argument("--duration", default=10.0, type=float,
                        help="Simulation duration in seconds (default: 10.0)")
    parser.add_argument("--dt", default=0.01, type=float,
                        help="Time step in seconds (default: 0.01)")
    parser.add_argument("--setpoints", default="0,45,0,0,0,0", type=str,
                        help="Setpoints in degrees for static trajectory only (default: 0,45,0,0,0,0)")
    parser.add_argument("--td3", action="store_true",
                        help="Also evaluate the trained TD3 agent (requires checkpoint)")
    parser.add_argument("--sse_window", default=0.1, type=float,
                        help="Fraction of final steps used for SSE (default: 0.1 = last 10%%)")
    parser.add_argument("--trajectory", default="static", choices=["static", "sine"],
                        help="Trajectory type: 'static' (step input) or 'sine' (sinusoidal, matches training distribution) (default: static)")
    parser.add_argument("--sine_amp", default=28.6, type=float,
                        help="Sine trajectory amplitude in degrees (default: 28.6 = 0.5 rad, matches training)")
    parser.add_argument("--sine_freq", default=0.5, type=float,
                        help="Sine trajectory frequency in Hz (default: 0.5, matches training)")
    args = parser.parse_args()

    setpoints_deg = np.array([float(v) for v in args.setpoints.split(",")])
    if len(setpoints_deg) != 6:
        raise ValueError("--setpoints must have exactly 6 comma-separated values.")
    setpoints_rad = np.deg2rad(setpoints_deg)
    sine_amp_rad = np.deg2rad(args.sine_amp)

    settings = ["A", "B"] if args.setting == "both" else [args.setting]
    sse_steps = max(1, int((args.duration / args.dt) * args.sse_window))

    trajectory = build_trajectory(args.trajectory, setpoints_rad, sine_amp_rad, args.sine_freq)

    print(f"\nTrajectory:  {args.trajectory.upper()}", end="")
    if args.trajectory == "sine":
        print(f" | Amplitude: {args.sine_amp:.1f} deg ({sine_amp_rad:.4f} rad) | Frequency: {args.sine_freq} Hz")
    else:
        print(f" | Setpoints: {setpoints_deg} deg")
    print(f"Duration:    {args.duration}s  |  dt: {args.dt}s  |  Steps: {int(args.duration / args.dt)}")
    print(f"SSE window:  last {args.sse_window * 100:.0f}% of steps ({sse_steps} steps = {sse_steps * args.dt:.2f}s)")

    summary = []

    for s in settings:
        print(f"\nRunning PID baseline Setting {s}...")
        pid_results = run_pid_metrics(s, trajectory, args.duration, args.dt, args.sse_window)
        mse, ise, sse = print_table(pid_results, f"PID Baseline - Setting {s}")
        summary.append((f"PID Setting {s}", mse, ise, sse))

        if args.td3:
            print(f"\nRunning TD3 RL - Setting {s}...")
            try:
                td3_results = run_td3_metrics(s, trajectory, args.duration, args.dt, args.sse_window)
                mse_td3, ise_td3, sse_td3 = print_table(td3_results, f"TD3 RL - Setting {s}")
                summary.append((f"TD3 Setting {s}", mse_td3, ise_td3, sse_td3))

                print_improvement_table(pid_results, td3_results,
                                        f"Improvement: TD3 vs PID - Setting {s} (positive = better)")
            except FileNotFoundError as e:
                print(f"  Skipped TD3 - {e}")

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
