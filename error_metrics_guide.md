# Error Metrics — Research Evaluation Guide

**Date:** 2026-06-24
**Script:** `error_metrics_unified.py` *(replaces the original `error_metrics.py`)*

This document explains the correct way to run error metrics to evaluate the TD3 + LSTM
agent, why both controllers must run on the same physics plant, why a sinusoidal trajectory
must be used for research-intended results, and what the numbers actually mean.

---

## Why `error_metrics_unified.py` Instead of `error_metrics.py`

The original `error_metrics.py` ran the **PID baseline on roboticstoolbox** (full
rigid-body dynamics) while the **TD3 agent ran on Puma560EnvTD3** (simplified diagonal
inertia model). This plant mismatch made the comparison unfair:

- The TD3 agent was trained entirely on `Puma560EnvTD3`
- When evaluated against roboticstoolbox physics it had never seen, its residual torques
  caused drift instead of improvement
- The result was a misleading **-195% overall MSE** — the agent appeared catastrophic
  when it was simply being judged on a different world than it was trained on

`error_metrics_unified.py` fixes this by running **both PID and TD3 through
Puma560EnvTD3**. PID-only mode passes zero residual torques; the environment's internal
FastPIDController still runs at 1 ms as normal.

| | `error_metrics.py` (old) | `error_metrics_unified.py` (current) |
|---|---|---|
| PID plant | roboticstoolbox | Puma560EnvTD3 |
| TD3 plant | Puma560EnvTD3 | Puma560EnvTD3 |
| Comparison fair? | No | Yes |
| Overall MSE improvement (sine) | -195% (artefact) | +3.44% (real) |

---

## Why Trajectory Choice Matters

### The Problem with Static Evaluation

Running with `--trajectory static` uses a constant step input (e.g. all joints → 25°).
This produces near-zero improvements because the TD3 agent was trained **exclusively on
sinusoidal trajectories** inside `rl_env.py`:

```python
# From rl_env.py — _update_reference()
q_ref_val = amp * np.sin(np.pi * freq * t + phase) + 0.2 * np.sin(0.6 * np.pi * freq * t + phase)
```

With randomised parameters per episode: `amp ∈ [0.3, 0.7] rad`, `freq ∈ [0.8, 1.2]`.
The agent has never seen a static step during training — evaluating it on one is an
out-of-distribution test.

### The Correct Evaluation

To evaluate as the research intended, use a **sinusoidal reference trajectory** that
matches the training distribution:

- **Amplitude:** 0.5 rad (28.6°) — centre of the training range [0.3, 0.7]
- **Frequency:** 0.5 Hz — matches the dominant training tone

---

## Commands

### Minimum command — research-intended results

```bash
uv run python error_metrics_unified.py --td3 --trajectory sine
```

### Full recommended evaluation (both settings, side by side)

```bash
uv run python error_metrics_unified.py --td3 --trajectory sine --setting both
```

### Static step for reference

```bash
uv run python error_metrics_unified.py --td3 --setpoints "25,25,25,25,25,25"
```

### Isolate a specific joint

```bash
# Only J2 (shoulder) — hardest joint
uv run python error_metrics_unified.py --td3 --setpoints "0,45,0,0,0,0"

# Only J3 (elbow) — best-performing joint
uv run python error_metrics_unified.py --td3 --setpoints "0,0,30,0,0,0"
```

### Sweep the training distribution

```bash
# Lower amplitude end of training range
uv run python error_metrics_unified.py --td3 --trajectory sine --sine_amp 17.2

# Upper amplitude end of training range
uv run python error_metrics_unified.py --td3 --trajectory sine --sine_amp 40.1

# Higher frequency
uv run python error_metrics_unified.py --td3 --trajectory sine --sine_freq 0.6
```

---

## Results (2026-06-24, Setting A)

### Sinusoidal trajectory — training distribution (10 s)

#### PID-Only

```
  Joint    MSE (rad^2)            ISE (rad^2*s)          SSE (rad)
  J1       0.00038540             0.00385403             0.01616608
  J2       0.00006724             0.00067236             0.00251834
  J3       0.00011975             0.00119749             0.00578486
  J4       0.00083170             0.00831703             0.02448955
  J5       0.00035029             0.00350294             0.01583825
  J6       0.00094639             0.00946392             0.02618712
  Overall  0.00045013             0.02700777             0.01516403
```

#### PID+TD3

```
  Joint    MSE (rad^2)            ISE (rad^2*s)          SSE (rad)
  J1       0.00038101             0.00381009             0.01613565
  J2       0.00007494             0.00074939             0.00286079
  J3       0.00010635             0.00106346             0.00495530
  J4       0.00081763             0.00817626             0.02486498
  J5       0.00034745             0.00347449             0.01567410
  J6       0.00088041             0.00880410             0.02555489
  Overall  0.00043463             0.02607779             0.01500762
```

#### Per-Joint Improvement

```
  Joint    MSE Improv.            ISE Improv.            SSE Improv.
  J1            +1.14%                 +1.14%                 +0.19%
  J2           -11.46%                -11.46%                -13.60%
  J3           +11.19%                +11.19%                +14.34%
  J4            +1.69%                 +1.69%                 -1.53%
  J5            +0.81%                 +0.81%                 +1.04%
  J6            +6.97%                 +6.97%                 +2.41%
  Overall       +3.44%                 +3.44%                 +1.03%
```

### Static 25° step on all joints (5 s)

#### Per-Joint Improvement

```
  Joint    MSE Improv.            ISE Improv.            SSE Improv.
  J1            -1.56%                 -1.56%                 +7.51%
  J2            -1.52%                 -1.52%                 -0.42%
  J3            +3.94%                 +3.94%                 +0.33%
  J4            +1.55%                 +1.55%                -13.98%
  J5            -0.88%                 -0.88%                +17.53%
  J6            +1.50%                 +1.50%                -16.76%
  Overall       +0.63%                 +0.63%                 -0.45%
```

Near-zero improvement on step inputs, as expected — the agent was not trained on them.

---

## Interpretation

### What the sinusoidal results show

**J3 Elbow — strongest improvement (+11.2% MSE, +14.3% SSE).**
J3 carries a 30 Nm/rad gravity coefficient. On a sinusoidal trajectory, gravity
continuously disturbs the joint as its angle changes. The TD3 agent learns to anticipate
this pattern and adds corrective torque that the PID alone cannot provide.

**J6 Wrist-Yaw — second best (+7.0% MSE).**
Despite having no gravity loading, the agent finds useful small corrections on J6 during
sustained sinusoidal tracking.

**J1, J4, J5 — marginal (+0.8–1.7% MSE).**
Improvements exist but are small. The agent contributes minor corrections that do not
meaningfully move the needle.

**J2 Shoulder — regression (-11.5% MSE, -13.6% SSE).**
J2 has the heaviest gravity load at 50 Nm/rad. The agent over-corrects — it applies
residual torques larger than the tracking error warrants, pushing J2 past the reference
instead of toward it. The Safety Cage allows up to 30% of the PID magnitude, which on
J2's large torques is still a significant perturbation. More training is needed for the
agent to learn to be conservative here.

### Root cause summary

| Joint | G_coeff | Agent behaviour | Result |
|---|---|---|---|
| J1 | 0.0 Nm/rad | Adds negligible corrections | Marginal improvement |
| J2 | 50.0 Nm/rad | Over-corrects on heavy gravity joint | Regression |
| J3 | 30.0 Nm/rad | Learns appropriate gravity compensation | Best improvement |
| J4 | 0.0 Nm/rad | Adds small useful corrections | Marginal improvement |
| J5 | 0.0 Nm/rad | Adds small useful corrections | Marginal improvement |
| J6 | 0.0 Nm/rad | Finds useful pattern corrections | Second-best improvement |

The overall picture: TD3 provides **genuine but modest improvement (+3.44% MSE)** on the
training distribution. The main bottleneck is J2 — fixing that regression would push the
overall improvement substantially higher.

---

## What Needs to Happen

### Train for more episodes

The current checkpoint was saved after a limited number of episodes. J2's regression is a
sign the agent is still learning. Run:

```bash
uv run python train_td3_rl.py --baseline_setting A --max_episodes 3000
```

Then re-evaluate. A fully converged agent should show improvement on all joints, not just
J3 and J6.

### Check TensorBoard to see training status

```bash
uv run tensorboard --logdir runs
```

Look at the `Eval/Error` curve. If it was still decreasing when training stopped, the
agent had not converged. If it had plateaued, hyperparameter tuning would be needed
instead of simply running more episodes.

---

## Quick Reference — All Commands

| Purpose | Command |
|---|---|
| Research-intended evaluation (sine, Setting A) | `uv run python error_metrics_unified.py --td3 --trajectory sine` |
| Research-intended evaluation (sine, both settings) | `uv run python error_metrics_unified.py --td3 --trajectory sine --setting both` |
| Static step, all joints 25° | `uv run python error_metrics_unified.py --td3 --setpoints "25,25,25,25,25,25"` |
| Static step, J2 only 45° | `uv run python error_metrics_unified.py --td3 --setpoints "0,45,0,0,0,0"` |
| PID-only baseline, sine | `uv run python error_metrics_unified.py --trajectory sine` |
| PID-only baseline, static | `uv run python error_metrics_unified.py` |
| Custom sine amplitude and frequency | `uv run python error_metrics_unified.py --td3 --trajectory sine --sine_amp 28.6 --sine_freq 0.5` |
| Full 10-second run | `uv run python error_metrics_unified.py --td3 --trajectory sine --duration 10` |
