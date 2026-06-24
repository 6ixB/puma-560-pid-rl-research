# Rise Time Analysis — Guide

**Script:** `run_rise_time_analysis.py`
**Date:** 2026-06-24

Evaluates step response performance for all 6 PUMA 560 joints by simulating a sudden
move-to-target command and measuring how quickly and cleanly each joint responds.
Both PID-Only and PID+TD3 run on the same `Puma560EnvTD3` plant for a fair comparison.

---

## Metrics Computed

| Metric | Abbreviation | Unit | What it measures |
|---|---|---|---|
| Rise Time (10–90%) | RT10-90 | seconds | Time from 10% to 90% of the target angle |
| Rise Time (0–100%) | RT0-100 | seconds | Time from first leaving zero to first reaching target |
| Settling Time (±2%) | ST ±2% | seconds | Last moment outside the ±2% tolerance band |
| Settling Time (±5%) | ST ±5% | seconds | Same for a looser ±5% band |
| Percent Overshoot | OS | % | How far past the target the joint swings |
| Steady-State Error | SS-Err | degrees | Average distance from target over the last 10% of the run |
| Delay Time | — | seconds | When the joint first crosses 50% of the target |
| Peak Time | — | seconds | When the joint reaches its absolute maximum |

---

## How to Run

```bash
uv run python run_rise_time_analysis.py [options]
```

### Options

| Flag | Default | Description |
|---|---|---|
| `--step-deg` | `25.0` | Step size applied to all 6 joints (degrees) |
| `--setting` | `A` | PID gain set: `A` or `B` |
| `--duration` | `5.0` | Simulation length in seconds |
| `--no-rl` | off | PID-only mode — no checkpoint needed |
| `--checkpoint` | auto | Override the TD3 actor checkpoint path |

---

## Common Recipes

**Default — PID vs TD3, 25° step, Setting A, 5 s:**
```bash
uv run python run_rise_time_analysis.py
```

**PID-only, no checkpoint needed:**
```bash
uv run python run_rise_time_analysis.py --no-rl
```

**Larger step, longer window:**
```bash
uv run python run_rise_time_analysis.py --step-deg 45 --duration 8
```

**Setting B:**
```bash
uv run python run_rise_time_analysis.py --setting B
```

**Both settings (run separately):**
```bash
uv run python run_rise_time_analysis.py --setting A
uv run python run_rise_time_analysis.py --setting B
```

---

## Outputs

Every run saves three files to the project directory:

| File | Contents |
|---|---|
| `rise_time_step_response_Setting_A.png` | 6-joint step response curves with PID vs TD3 overlaid, 10%/90% guides, ±2% settling band, and metric annotations |
| `rise_time_comparison_Setting_A.png` | 4-panel bar chart comparing rise time, settling time, overshoot, and SS-error across all joints |
| `rise_time_results_Setting_A.csv` | All metrics in machine-readable CSV format |

---

## How the Metrics Are Calculated

### Rise Time (10–90%)

1. Define thresholds: `10% × target` and `90% × target`
2. Scan the angle trace sample by sample
3. Use **linear interpolation** between the two samples that straddle each threshold to find the precise crossing time
4. Subtract: `RT10-90 = t_90 - t_10`

```
angle
 22.5° · · · · · ·╔══════════════  90% threshold
                   ↑ t_90
  2.5° · · ·╔══════              10% threshold
             ↑ t_10
  0°  ───────┘
             |←  RT10-90  →|
```

### Settling Time (±2%)

1. Draw a ±2% tolerance band around the target
2. Scan the trace **backwards from the end**
3. Find the last sample outside the band
4. Settling time = the very next sample after that

Returns `N/A` if the joint is still outside the band at the end of the simulation.

### Percent Overshoot

```
OS = (peak_angle - target) / target × 100
```

The algorithm finds the maximum angle reached during the entire run.

### Steady-State Error

Takes the mean of the **last 10% of samples** and measures how far it sits from the target:

```
SS-Err = |mean(last 10% of trace) - target|
```

---

## Results (2026-06-24, Setting A, 25° step, 5 s)

### PID-Only

```
  Joint   SP(deg)  Final(deg)   RT10-90(s)   RT0-100(s)  ST 2pct(s)   OS(pct)  SS-Err(deg)
  J1       25.0      25.062       0.0999       0.1258      0.8000      30.6      0.06160
  J2       25.0      24.351       0.1012       0.1279         N/A      26.2      0.64890
  J3       25.0      24.405       0.1046       0.1336         N/A      29.0      0.59461
  J4       25.0      25.077       0.1325       0.1748      0.7200      21.6      0.07675
  J5       25.0      25.083       0.1224       0.1589      0.4200      13.4      0.08275
  J6       25.0      25.087       0.1400       0.1858      0.7200      19.8      0.08738
```

### PID+TD3

```
  Joint   SP(deg)  Final(deg)   RT10-90(s)   RT0-100(s)  ST 2pct(s)   OS(pct)  SS-Err(deg)
  J1       25.0      25.057       0.0993       0.1248      0.7800      32.3      0.05697
  J2       25.0      24.348       0.1004       0.1266         N/A      28.1      0.65162
  J3       25.0      24.407       0.1059       0.1358         N/A      25.8      0.59265
  J4       25.0      25.087       0.1360       0.1805      0.5200      18.4      0.08749
  J5       25.0      25.068       0.1203       0.1549      0.4600      16.1      0.06824
  J6       25.0      25.102       0.1451       0.1940      0.5200      16.0      0.10202
```

### Improvement Summary

```
  Joint             D Rise-Time    D Settle-2%   D Overshoot    D SS-Error
  J1 Waist              +0.6 %         +2.5 %        +1.7 %      +0.005 deg
  J2 Shoulder           +0.8 %            N/A        +1.8 %      -0.003 deg
  J3 Elbow              -1.3 %            N/A        -3.2 %      +0.002 deg
  J4 Wrist-Roll         -2.6 %        +27.8 %        -3.3 %      -0.011 deg
  J5 Wrist-Pitch        +1.7 %         -9.5 %        +2.7 %      +0.015 deg
  J6 Wrist-Yaw          -3.7 %        +27.8 %        -3.8 %      -0.015 deg
```

---

## Interpretation

### Rise times are nearly identical across all joints (~0.10–0.14 s)

The TD3 agent changes rise time by less than 4% on every joint. This is expected — the Safety Cage limits the agent's residual torque to 30% of the PID torque magnitude, which is not enough to significantly alter the fast initial dynamics of a step response.

### J2 and J3 never fully settle (N/A settling time)

Both joints are gravity-loaded (50 and 30 Nm/rad respectively). On a static step, gravity continuously pulls them away from the target and the PID integral term alone cannot fully reject the disturbance within 5 seconds. The agent does not fix this on step inputs because it was trained on sinusoidal trajectories.

### J4 and J6 show the clearest TD3 benefit (+27.8% settling time improvement)

The wrist joints settle faster with TD3 active. The agent applies small damping corrections in the tail of the response, reducing wobble around the target.

### Use error_metrics_unified.py for sinusoidal performance

Rise time analysis targets **step response** behaviour. To evaluate the agent on the trajectory it was trained on (sinusoidal), use:

```bash
uv run python error_metrics_unified.py --td3 --trajectory sine
```

---

## Relationship to Error Metrics

| | Rise Time Analysis | Error Metrics (Unified) |
|---|---|---|
| **Trajectory** | Step input | Static step or sinusoidal |
| **Measures** | Speed and shape of initial response | Accumulated error over entire run |
| **Best for** | How fast the arm reacts | How accurate the arm is overall |
| **Same plant?** | Yes — both use Puma560EnvTD3 | Yes — both use Puma560EnvTD3 |
| **TD3 effect** | < 4% change (safety cage limits it) | +3.44% MSE on sinusoidal trajectory |

The two tools are complementary: rise time tells you about **speed**, error metrics tell you about **accuracy**.
