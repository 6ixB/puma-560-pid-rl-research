# Error Metrics — Research Evaluation Guide

**Date:** 2026-06-10

This document explains the correct way to run `error_metrics.py` to evaluate the TD3 + LSTM agent as the research intended, why a sinusoidal trajectory must be used instead of a static step, and what the results actually mean.

---

## Why Trajectory Choice Matters

### The Problem with Static Evaluation

Running `error_metrics.py` with default settings uses a **static step input** (e.g. Joint 2 → 45°). This produces modest-looking improvements (~18%) that underrepresent the agent's actual capability.

The reason: the TD3 agent was trained **exclusively on sinusoidal trajectories** inside `rl_env.py`:

```python
# From rl_env.py — _update_reference()
q_ref_val = amp * np.sin(np.pi * freq * t + phase) + 0.2 * np.sin(0.6 * np.pi * freq * t + phase)
```

With randomised parameters per episode: `amp ∈ [0.3, 0.7] rad`, `freq ∈ [0.8, 1.2]`. The agent has never seen a static step during training. Evaluating it on one is an out-of-distribution test — the agent was never optimised for it.

### The Correct Evaluation

To evaluate as the research intended, both the PID and TD3 simulations must use a **sinusoidal reference trajectory** that matches the training distribution. The default sine parameters are:

- **Amplitude:** 0.5 rad (28.6°) — centre of the training range [0.3, 0.7]
- **Frequency:** 0.5 Hz — approximately matches the training distribution

---

## Commands

### Minimum command to see research-intended results

```bash
uv run python error_metrics.py --trajectory sine --setting A --td3
```

### Full recommended evaluation (both settings, side by side)

```bash
uv run python error_metrics.py --trajectory sine --setting both --td3
```

### Customise sine parameters to sweep the training distribution

```bash
# Lower amplitude end of training range
uv run python error_metrics.py --trajectory sine --td3 --sine_amp 17.2 --sine_freq 0.5

# Upper amplitude end of training range
uv run python error_metrics.py --trajectory sine --td3 --sine_amp 40.1 --sine_freq 0.5

# Higher frequency
uv run python error_metrics.py --trajectory sine --td3 --sine_amp 28.6 --sine_freq 0.6
```

### Static step for reference (original behaviour)

```bash
uv run python error_metrics.py --trajectory static --setting both --td3
```

---

## Results (2026-06-10)

### Command used

```bash
uv run python error_metrics.py --trajectory sine --setting both --td3
```

### Setting A — PID Baseline

```
  Joint    MSE (rad^2)            ISE (rad^2*s)          SSE (rad)
  J1       0.00035800             0.00358005             0.01693249
  J2       0.00089923             0.00899226             0.03292995
  J3       0.00016469             0.00164689             0.01264071
  J4       0.00001232             0.00012322             0.00297412
  J5       0.00001071             0.00010713             0.00280347
  J6       0.00004993             0.00049928             0.00601326
  Overall  0.00024915             0.01494883             0.01238233
```

### Setting A — TD3 + LSTM

```
  Joint    MSE (rad^2)            ISE (rad^2*s)          SSE (rad)
  J1       0.00034982             0.00349816             0.01605948
  J2       0.00004327             0.00043272             0.00293190
  J3       0.00007303             0.00073033             0.00554587
  J4       0.00076972             0.00769716             0.02424120
  J5       0.00032719             0.00327188             0.01551217
  J6       0.00083238             0.00832376             0.02528383
  Overall  0.00039923             0.02395401             0.01492907
```

### Setting A — Per-Joint Improvement

```
  Joint    MSE Improv.            ISE Improv.            SSE Improv.
  J1       +2.29%                 +2.29%                 +5.16%
  J2       +95.19%                +95.19%                +91.10%
  J3       +55.65%                +55.65%                +56.13%
  J4       -6146.44%              -6146.44%              -715.07%
  J5       -2954.27%              -2954.27%              -453.32%
  J6       -1567.14%              -1567.14%              -320.47%
  Overall  -60.24%                -60.24%                -20.57%
```

### Setting B — Per-Joint Improvement

```
  Joint    MSE Improv.            ISE Improv.            SSE Improv.
  J1       +2.04%                 +2.04%                 +5.28%
  J2       +95.60%                +95.60%                +91.75%
  J3       +52.64%                +52.64%                +54.43%
  J4       -6445.81%              -6445.81%              -745.57%
  J5       -2991.19%              -2991.19%              -474.11%
  J6       -1648.35%              -1648.35%              -345.82%
  Overall  -76.46%                -76.46%                -24.85%
```

---

## Interpretation

### What went well

**Joint 2 (Shoulder) — ~95% improvement in both settings.**
This is the most dramatic result. The TD3 + LSTM agent reduces J2 tracking error by over 95% on sinusoidal trajectories. J2 is the gravity-dominant joint (G_coeff = 50 Nm/rad) and the one most affected by nonlinear dynamics. The LSTM's temporal memory allows it to anticipate the sinusoidal phase and pre-emptively apply corrective torque before gravity pulls the joint off-track. This is exactly what the research was designed to demonstrate.

**Joint 3 (Elbow) — ~55% improvement.**
J3 has a significant gravity coefficient (G_coeff = 30 Nm/rad) and benefits similarly. The agent learns the cross-joint coupling between J2 and J3 and corrects for both simultaneously.

**Joint 1 (Waist) — small but consistent +2% improvement.**
J1 has no gravity loading (G_coeff = 0) so the improvement is marginal, but the agent at least does not hurt it.

### What went wrong — Joints 4, 5, 6

**Joints 4, 5, 6 (Wrist) — catastrophic degradation (-715% to -6146%).**
The wrist joints have the smallest torque limits (5, 5, 3 Nm) and the lowest inertia. On a sinusoidal trajectory, the TD3 agent is applying residual torques to these joints that are far larger than the error warrants — the agent is actually fighting the PID rather than helping it.

This is a sign of **insufficient training** specific to the wrist joints. The agent has learned to aggressively correct J2 and J3, but it has not learned to leave J4–J6 alone when they are already tracking well. This manifests as the overall metric going negative despite J2 and J3 both improving substantially.

### Root cause summary

| Joint | G_coeff | Inertia | Agent behaviour |
|---|---|---|---|
| J1 | 0.0 | 4.0 kg·m² | Negligible effect — agent learns to leave alone |
| J2 | 50.0 | 6.0 kg·m² | **Strong improvement** — agent compensates gravity |
| J3 | 30.0 | 4.5 kg·m² | **Strong improvement** — agent compensates gravity |
| J4 | 0.0 | 1.5 kg·m² | Degraded — agent over-corrects |
| J5 | 0.0 | 1.0 kg·m² | Degraded — agent over-corrects |
| J6 | 0.0 | 0.8 kg·m² | Degraded — agent over-corrects |

The pattern is clear: the RL agent has learned to compensate **gravity** (the dominant nonlinearity in J2 and J3) but has not yet learned to be conservative with the zero-gravity wrist joints. More training episodes are needed for the agent to refine its policy on J4–J6.

---

## What Needs to Happen

### Train for more episodes

The current checkpoint was saved after a limited number of training episodes. The wrist joint degradation is a sign that the agent is still learning. Run:

```bash
uv run python train_td3_rl.py --baseline_setting A --max_episodes 3000
```

Then re-evaluate. A fully converged agent should show improvement on all joints, not just J2 and J3.

### Check TensorBoard to see training status

```bash
uv run tensorboard --logdir runs
```

Look at the `Eval/Error` curve. If it was still decreasing when training stopped, the agent had not converged. If it had plateaued, more training episodes may not help and hyperparameter tuning would be needed instead.

### Compare static vs sine results side by side

To see the full picture of where the agent succeeds and struggles:

```bash
# Static step evaluation
uv run python error_metrics.py --trajectory static --setting A --td3

# Sinusoidal evaluation (research-intended)
uv run python error_metrics.py --trajectory sine --setting A --td3
```

---

## Quick Reference — All Commands

| Purpose | Command |
|---|---|
| Research-intended evaluation (sine, both settings) | `uv run python error_metrics.py --trajectory sine --setting both --td3` |
| Research-intended evaluation (sine, Setting A only) | `uv run python error_metrics.py --trajectory sine --setting A --td3` |
| Static step evaluation (original) | `uv run python error_metrics.py --trajectory static --setting both --td3` |
| PID-only baseline, static | `uv run python error_metrics.py --trajectory static --setting both` |
| PID-only baseline, sine | `uv run python error_metrics.py --trajectory sine --setting both` |
| Custom sine amplitude and frequency | `uv run python error_metrics.py --trajectory sine --td3 --sine_amp 28.6 --sine_freq 0.5` |
