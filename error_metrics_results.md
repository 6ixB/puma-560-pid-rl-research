# Error Metrics Results — PID vs TD3 + LSTM

**Date:** 2026-06-24
**Script:** `error_metrics_unified.py`
**Plant:** `Puma560EnvTD3` — used for both PID-Only and PID+TD3 (same physics)

---

## Metrics Explained

| Metric | Formula | Unit | What it measures |
|---|---|---|---|
| **MSE** | `(1/N) * sum(e²)` | rad² | Average squared tracking error over the full simulation |
| **ISE** | `sum(e²) * dt` | rad²·s | Total accumulated squared error — penalises both magnitude and duration |
| **SSE** | `mean(|e|)` over last 10% of steps | rad | Residual error after the system has settled (steady-state) |

---

## Run 1 — Sinusoidal Trajectory (Training Distribution)

**Command:**
```bash
uv run python error_metrics_unified.py --td3 --trajectory sine --duration 10
```

**Configuration:** Setting A | Sine amplitude: 28.6° (0.4992 rad) | Frequency: 0.5 Hz |
Duration: 10 s | RL step dt: 20 ms | Steps: 500 | SSE window: last 10% (1.00 s)

### PID-Only — Setting A

```
========================================================================
  PID-Only (Setting A)  [Puma560EnvTD3]
========================================================================
  Joint    MSE (rad^2)            ISE (rad^2*s)          SSE (rad)
  ----------------------------------------------------------------------
  J1       0.00038540             0.00385403             0.01616608
  J2       0.00006724             0.00067236             0.00251834
  J3       0.00011975             0.00119749             0.00578486
  J4       0.00083170             0.00831703             0.02448955
  J5       0.00035029             0.00350294             0.01583825
  J6       0.00094639             0.00946392             0.02618712
  ----------------------------------------------------------------------
  Overall  0.00045013             0.02700777             0.01516403
========================================================================
```

### PID+TD3 — Setting A

```
========================================================================
  PID+TD3 (Setting A)  [Puma560EnvTD3]
========================================================================
  Joint    MSE (rad^2)            ISE (rad^2*s)          SSE (rad)
  ----------------------------------------------------------------------
  J1       0.00038101             0.00381009             0.01613565
  J2       0.00007494             0.00074939             0.00286079
  J3       0.00010635             0.00106346             0.00495530
  J4       0.00081763             0.00817626             0.02486498
  J5       0.00034745             0.00347449             0.01567410
  J6       0.00088041             0.00880410             0.02555489
  ----------------------------------------------------------------------
  Overall  0.00043463             0.02607779             0.01500762
========================================================================
```

### Summary Comparison

| Run | Overall MSE (rad²) | Overall ISE (rad²·s) | Overall SSE (rad) |
|---|---|---|---|
| PID-Only | 0.00045013 | 0.02700777 | 0.01516403 |
| PID+TD3 | 0.00043463 | 0.02607779 | 0.01500762 |
| **Improvement** | **+3.44%** | **+3.44%** | **+1.03%** |

### Per-Joint Improvement

```
========================================================================
  Improvement: TD3 vs PID - Setting A  (positive = TD3 better)
========================================================================
  Joint    MSE Improv.            ISE Improv.            SSE Improv.
  ----------------------------------------------------------------------
  J1            +1.14%                 +1.14%                 +0.19%
  J2           -11.46%                -11.46%                -13.60%
  J3           +11.19%                +11.19%                +14.34%
  J4            +1.69%                 +1.69%                 -1.53%
  J5            +0.81%                 +0.81%                 +1.04%
  J6            +6.97%                 +6.97%                 +2.41%
  ----------------------------------------------------------------------
  Overall       +3.44%                 +3.44%                 +1.03%
========================================================================
```

---

## Run 2 — Static Step Input (Out-of-Distribution Reference)

**Command:**
```bash
uv run python error_metrics_unified.py --td3 --setpoints "25,25,25,25,25,25" --duration 5
```

**Configuration:** Setting A | All joints → 25° step | Duration: 5 s | RL step dt: 20 ms |
Steps: 250 | SSE window: last 10% (0.50 s)

### Summary Comparison

| Run | Overall MSE (rad²) | Overall ISE (rad²·s) | Overall SSE (rad) |
|---|---|---|---|
| PID-Only | 0.00265553 | 0.07966577 | 0.00451454 |
| PID+TD3 | 0.00263868 | 0.07916044 | 0.00453492 |
| **Improvement** | **+0.63%** | **+0.63%** | **-0.45%** |

Near-zero improvement confirms the agent was not trained on step inputs. Use sinusoidal
results for research-grade evaluation.

---

## Per-Joint Analysis

### Joint 3 (Elbow) — Strongest Improvement

J3 benefits most from the TD3 agent on sinusoidal trajectories, with **+11.2% MSE** and
**+14.3% SSE** improvement. The elbow carries a 30 Nm/rad gravity coefficient — large
enough that PID alone accumulates meaningful tracking error on an oscillating reference,
leaving room for the agent to add useful corrections.

| Metric | PID | PID+TD3 | Improvement |
|---|---|---|---|
| MSE | 0.00011975 rad² | 0.00010635 rad² | +11.19% |
| ISE | 0.00119749 rad²·s | 0.00106346 rad²·s | +11.19% |
| SSE | 0.00578486 rad (0.331°) | 0.00495530 rad (0.284°) | +14.34% |

### Joint 6 (Wrist-Yaw) — Second Best

J6 has no gravity loading but the agent finds a consistent tracking pattern that reduces
error by +7.0% MSE. This is smaller but reliable across the full 10-second run.

| Metric | PID | PID+TD3 | Improvement |
|---|---|---|---|
| MSE | 0.00094639 rad² | 0.00088041 rad² | +6.97% |
| ISE | 0.00946392 rad²·s | 0.00880410 rad²·s | +6.97% |
| SSE | 0.02618712 rad (1.500°) | 0.02555489 rad (1.464°) | +2.41% |

### Joint 2 (Shoulder) — Regression

J2 carries the heaviest gravity load at 50 Nm/rad. The agent over-corrects, applying
residual torques that push J2 past the reference instead of toward it. This is the single
largest source of degradation in the results.

| Metric | PID | PID+TD3 | Improvement |
|---|---|---|---|
| MSE | 0.00006724 rad² | 0.00007494 rad² | -11.46% |
| ISE | 0.00067236 rad²·s | 0.00074939 rad²·s | -11.46% |
| SSE | 0.00251834 rad (0.144°) | 0.00286079 rad (0.164°) | -13.60% |

### Joints 1, 4, 5 — Marginal

Small positive improvements (+0.8–1.7% MSE). The agent contributes minor corrections
that do not meaningfully change the outcome.

---

## Interpretation

### Why the results differ from the old error_metrics.py

The original script reported **-195% overall MSE improvement** on a static step. This
was entirely due to a plant mismatch — PID used roboticstoolbox dynamics while TD3 used
the simplified Puma560EnvTD3 plant. On the unified plant, the same agent shows
**+3.44% improvement** on sinusoidal trajectories. The difference is not the agent
getting better or worse — it is the comparison becoming fair.

### What the +3.44% overall improvement means

Modest but real. The agent provides genuine benefit on J3 and J6, is neutral on J1/J4/J5,
and hurts J2. The overall positive number means the improvements outweigh the regression
in aggregate.

### Why J2 regresses

The Safety Cage allows up to 30% of the PID torque magnitude as residual. On J2 where
PID torques are already very large (fighting 50 Nm/rad gravity), 30% is still a large
absolute perturbation. The agent has not yet learned to be conservative enough on this
joint. More training is the remedy.

### Root cause table

| Joint | G_coeff | Inertia | Agent behaviour | Result |
|---|---|---|---|---|
| J1 | 0.0 Nm/rad | 4.0 kg·m² | Negligible corrections | Marginal improvement |
| J2 | 50.0 Nm/rad | 6.0 kg·m² | Over-corrects on heavy gravity | Regression |
| J3 | 30.0 Nm/rad | 4.5 kg·m² | Appropriate gravity compensation | Best improvement |
| J4 | 0.0 Nm/rad | 1.5 kg·m² | Small useful corrections | Marginal improvement |
| J5 | 0.0 Nm/rad | 1.0 kg·m² | Small useful corrections | Marginal improvement |
| J6 | 0.0 Nm/rad | 0.8 kg·m² | Finds consistent tracking pattern | Second-best improvement |

---

## Recommended Next Steps

1. **Train for more episodes** — the J2 regression is a sign of insufficient training:
   ```bash
   uv run python train_td3_rl.py --baseline_setting A --max_episodes 3000
   ```

2. **Check TensorBoard** — inspect whether `Eval/Error` had converged or was still falling:
   ```bash
   uv run tensorboard --logdir runs
   ```

3. **Compare Setting B** — see if the weaker gains produce larger relative improvement:
   ```bash
   uv run python error_metrics_unified.py --td3 --trajectory sine --setting both
   ```

4. **Vary the sine parameters** — stress-test across the training distribution:
   ```bash
   uv run python error_metrics_unified.py --td3 --trajectory sine --sine_amp 40.1
   uv run python error_metrics_unified.py --td3 --trajectory sine --sine_freq 0.6
   ```

---

## Notes

- All error values are in **radians**. To convert SSE to degrees: multiply by `180/pi`
  (e.g. 0.00578 rad = 0.331°).
- The SSE window is the **last 1.0 second** of the 10-second run (last 10% = 50 steps at
  20 ms/step). Joints that have not fully settled by 9 seconds will show inflated SSE.
- Both PID and TD3 use `Puma560EnvTD3` with `rl_decimation=20` (20 ms outer step,
  1 ms PID inner step). The RL step dt is therefore 20 ms for both modes.
