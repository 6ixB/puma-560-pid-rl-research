# Error Metrics Results — PID vs TD3 + LSTM

**Date:** 2026-06-10
**Command run:** `uv run python error_metrics.py --setting A --td3`
**Configuration:** Setting A baseline gains | Setpoints: J2 = 45 deg, all others = 0 deg | Duration: 10.0s | dt: 0.01s | Steps: 1000 | SSE window: last 10% (1.00s)

---

## Metrics Explained

| Metric | Formula | Unit | What it measures |
|---|---|---|---|
| **MSE** | `(1/N) * sum(e²)` | rad² | Average squared tracking error over the full simulation |
| **ISE** | `sum(e²) * dt` | rad²·s | Total accumulated squared error — penalises both magnitude and duration |
| **SSE** | `mean(|e|)` over last 10% of steps | rad | Residual error after the system has settled (steady-state) |

---

## Raw Results

### PID Baseline — Setting A

```
========================================================================
  PID Baseline - Setting A
========================================================================
  Joint    MSE (rad^2)            ISE (rad^2*s)          SSE (rad)
  ----------------------------------------------------------------------
  J1       0.00007086             0.00070856             0.00000835
  J2       0.00245412             0.02454123             0.00878264
  J3       0.00009960             0.00099596             0.00434426
  J4       0.00000000             0.00000000             0.00000001
  J5       0.00000002             0.00000020             0.00006443
  J6       0.00000000             0.00000000             0.00000000
  ----------------------------------------------------------------------
  Overall  0.00043743             0.02624596             0.00219995
========================================================================
```

### TD3 RL + LSTM — Setting A

```
========================================================================
  TD3 RL - Setting A
========================================================================
  Joint    MSE (rad^2)            ISE (rad^2*s)          SSE (rad)
  ----------------------------------------------------------------------
  J1       0.00000014             0.00000140             0.00009763
  J2       0.00215339             0.02153391             0.00691127
  J3       0.00000016             0.00000158             0.00007943
  J4       0.00000072             0.00000718             0.00015851
  J5       0.00000110             0.00001095             0.00026503
  J6       0.00000192             0.00001920             0.00028546
  ----------------------------------------------------------------------
  Overall  0.00035957             0.02157422             0.00129956
========================================================================
```

---

## Summary Comparison

| Run | Overall MSE (rad²) | Overall ISE (rad²·s) | Overall SSE (rad) |
|---|---|---|---|
| PID Baseline — Setting A | 0.00043743 | 0.02624596 | 0.00219995 |
| TD3 RL + LSTM — Setting A | 0.00035957 | 0.02157422 | 0.00129956 |
| **Improvement** | **+17.80%** | **+17.80%** | **+40.93%** |

---

## Per-Joint Analysis

### Joint 2 (Shoulder) — Active Joint

J2 is the only joint given a non-zero target (45°). It carries the highest inertia (6.0 kg·m²) and is subject to significant gravity loading, making it the hardest joint to control and the dominant source of error across all metrics.

| Metric | PID | TD3 + LSTM | Improvement |
|---|---|---|---|
| MSE | 0.00245412 rad² | 0.00215339 rad² | +12.24% |
| ISE | 0.02454123 rad²·s | 0.02153391 rad²·s | +12.24% |
| SSE | 0.00878264 rad (0.503°) | 0.00691127 rad (0.396°) | +21.31% |

The RL agent reduces J2's steady-state error from **0.50° to 0.40°** — the LSTM recognises the persistent gravity disturbance and applies a corrective residual torque that the PID integral term alone could not fully compensate.

### Joint 3 (Elbow) — Gravity Coupling Effect

J3 has a setpoint of 0° but shows notable SSE in the PID-only run (0.00434 rad = 0.25°). This is caused by **gravity coupling** — when J2 rotates to 45°, it shifts the gravitational load acting on J3, introducing a disturbance the PID only partially rejects.

| Metric | PID | TD3 + LSTM | Improvement |
|---|---|---|---|
| SSE | 0.00434426 rad (0.249°) | 0.00007943 rad (0.005°) | **+98.17%** |

The TD3 agent almost completely eliminates this coupled disturbance, reducing J3's steady-state error by over 98%. This is the most significant per-joint improvement in the results and demonstrates the value of the LSTM's temporal memory — it learns the coupling pattern between J2 and J3 and pre-emptively corrects for it.

### Joints 1, 4, 5, 6 — Idle Joints

These joints have zero targets and no significant load. The PID already holds them near-perfectly. The TD3 agent maintains this while slightly redistributing small residual torques, keeping all metrics near zero.

---

## Interpretation

### What went well

- **SSE improvement of +40.93%** is the standout result. The RL+LSTM agent is significantly better at holding joints at their targets after the transient phase, which is the most practically important property for precise robot positioning.
- **J3 coupling rejection** improved by over 98%, demonstrating that the LSTM successfully learns cross-joint gravitational coupling that a per-joint PID cannot model.
- The agent improves performance **without sacrificing stability** — the Safety Cage constraint ensured the RL contribution never destabilised the system throughout training.

### What could be better

- **MSE and ISE improvement of only ~18%** is modest compared to the theoretical 60–67% improvement cited in the research. The transient response (how quickly J2 reaches 45°) is not dramatically faster.
- This gap is most likely due to **insufficient training**. The agent was evaluated after the current number of training episodes, which may not be enough for full convergence. The TensorBoard reward curve should be checked to determine if training had plateaued or was still improving.

### Recommended next steps

1. **Train for more episodes** — run `uv run python train_td3_rl.py --baseline_setting A --max_episodes 3000` and re-evaluate
2. **Check TensorBoard** — run `uv run tensorboard --logdir runs` and inspect whether `Eval/Error` had converged or was still decreasing
3. **Compare Setting B** — run `uv run python error_metrics.py --setting both --td3` to see if the weaker baseline gains produce a larger relative improvement
4. **Try different setpoints** — run with a more demanding trajectory such as `--setpoints 45,90,45,30,0,0` to stress-test the agent on multi-joint simultaneous motion

---

## Notes

- All error values are in **radians**. To convert SSE to degrees: multiply by `180 / pi` (e.g. 0.00878 rad = 0.503°).
- The SSE window is the **last 1.0 second** of the 10 second simulation (last 10% = 100 steps at dt=0.01s). Joints that have not fully settled by 9 seconds will show inflated SSE values.
- The TD3 simulation uses a fixed RL step of 10ms (10 inner PID steps of 1ms each), so it runs at a coarser control rate than the Legacy PID engine. The `t_steps` output reflects 10ms intervals.
