# PUMA 560 PID + Residual RL Research

A research implementation of **Residual Reinforcement Learning for PUMA 560 robot arm joint control**. Rather than replacing a classical PID controller, a TD3 (Twin Delayed DDPG) or SAC RL agent with an LSTM temporal memory learns to inject small residual torque corrections on top of the PID output across all 6 joints simultaneously. A Lyapunov-based **Safety Cage** hard-constrains how much the RL agent can deviate from the PID baseline, guaranteeing stability throughout training.

---

## Table of Contents

- [Project Overview](#project-overview)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [GPU Setup](#gpu-setup)
- [Project Structure](#project-structure)
- [Running the Project](#running-the-project)
  - [1. GUI Simulator](#1-gui-simulator)
  - [2. Train TD3 RL Agent](#2-train-td3-rl-agent)
  - [3. Train SAC RL Agent](#3-train-sac-rl-agent)
  - [4. Monitor Training with TensorBoard](#4-monitor-training-with-tensorboard)
  - [5. Evaluate a Trained Agent](#5-evaluate-a-trained-agent)
  - [6. Error Metrics (MSE, ISE, SSE)](#6-error-metrics-mse-ise-sse)
  - [7. Generate Offline Supervised Dataset](#7-generate-offline-supervised-dataset)
  - [8. Optimize PID Gains](#8-optimize-pid-gains)
  - [9. Generate Research Figures](#9-generate-research-figures)
  - [10. Utility Scripts](#10-utility-scripts)
  - [11. Build Standalone Executable](#11-build-standalone-executable)
- [Recommended First-Time Run Order](#recommended-first-time-run-order)
- [PID Baseline Settings](#pid-baseline-settings)
- [Architecture Overview](#architecture-overview)

---

## Project Overview

The control system works as follows:

1. A classical PID controller computes a baseline torque for each of the 6 joints
2. An LSTM-based RL agent observes a sliding window of the last 20 state snapshots (42-dimensional: positions, velocities, errors, integrals, setpoints) and outputs a residual torque correction
3. The Safety Cage clamps the residual torque so it never exceeds a fraction `alpha` of the PID torque — starting at `alpha = 0.05` and ramping up to `0.30` over 300 training episodes
4. The total torque applied is: `tau_total = tau_pid + tau_residual`

This approach improves tracking accuracy while keeping stability guarantees that a pure RL approach cannot provide.

---

## Prerequisites

- **Python 3.12**
- **uv** package manager — install from [https://docs.astral.sh/uv/](https://docs.astral.sh/uv/)
- **NVIDIA GPU** (optional but strongly recommended for training)
- **NVIDIA driver 520+** for CUDA 12.x support

---

## Installation

Clone the repository and install all dependencies using `uv`:

```bash
git clone <repo-url>
cd puma-560-pid-rl-research
uv sync
```

`uv sync` reads `pyproject.toml` and `uv.lock` and installs all dependencies into a local `.venv` automatically.

All `uv run python <script>` commands below automatically use this virtual environment — no manual activation needed.

---

## GPU Setup

By default `uv sync` installs the CPU-only build of PyTorch. To enable GPU training:

**1. Check your NVIDIA driver version:**
```bash
nvidia-smi
```

**2. Reinstall PyTorch with the matching CUDA wheel:**
```bash
# For CUDA 12.8 (driver 520+, recommended)
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu128

# For CUDA 12.6 fallback
uv pip install --reinstall torch --index-url https://download.pytorch.org/whl/cu126
```

**3. Verify CUDA is detected:**
```bash
uv run python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

Expected output: `True  NVIDIA GeForce RTX XXXX`

Once installed, all training scripts use the GPU automatically — no code changes needed. The device is selected via:
```python
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
```

---

## Project Structure

```
puma-560-pid-rl-research/
│
├── main.py                   # PySide6 GUI simulator (main entry point)
├── dynamics.py               # Forward & inverse dynamics via roboticstoolbox
├── pid_controller.py         # PID controller class and simulation runner
├── trajectory.py             # Trajectory generators: Static, Waypoint, Sine Wave
├── canvas.py                 # Matplotlib canvas for GUI plots (animated + static)
│
├── rl_env.py                 # RL environment: Puma560Env and Puma560EnvTD3
│                             #   - 1ms PID inner loop, 10-20ms RL outer step
│                             #   - Motor electrical dynamics (L di/dt = V - Ri - Kb*w)
│                             #   - Coulomb + Stribeck + viscous friction model
│                             #   - Lyapunov Safety Cage constraint
│
├── td3_lstm_models.py        # TD3 actor, critic, replay buffer (LSTM-based)
├── sac_lstm_models.py        # SAC actor, critic, replay buffer (LSTM-based)
├── train_td3_rl.py           # TD3 training loop with TensorBoard logging
├── train_sac_rl.py           # SAC training loop with TensorBoard logging
├── simulate_td3.py           # Load & evaluate trained TD3 checkpoint
│
├── generate_offline_data.py  # Offline dataset via inverse dynamics + L-BFGS-B
├── optimize_pid.py           # PID gain optimization via scipy L-BFGS-B
├── error_metrics.py          # MSE, ISE, SSE error metric calculator
│
├── show_params.py            # Print PUMA 560 dynamic parameters
├── interactive.py            # Interactive 3D robot teach pendant
│
├── offline_pid_dataset.csv   # Pre-generated supervised training dataset
├── simulation_results.png    # Last saved evaluation plot
│
├── pyproject.toml            # Project dependencies (Python 3.12)
├── uv.lock                   # Locked dependency versions
├── main.spec                 # PyInstaller spec -> dist/main.exe
├── PumaSimulator.spec        # PyInstaller spec -> dist/PumaSimulator.exe
│
├── checkpoints/              # Saved RL agent weights (created during training)
│   ├── Setting_A/
│   │   ├── td3_best_actor
│   │   ├── td3_best_critic
│   │   └── ...
│   └── Setting_B/
│
├── runs/                     # TensorBoard event logs (created during training)
│
└── research/
    ├── Six Joint Simulation with Analysis _python_20260519_419c6a.py
    │                         # Standalone script generating Figures 3-11
    └── *.png                 # Architecture diagrams
```

---

## Running the Project

### 1. GUI Simulator

The primary interactive tool. Opens a PySide6 window with a tabbed control panel on the left, 2D plots in the top-right, and a split 3D robot view (start pose / end pose) at the bottom.

```bash
uv run python main.py
```

**Tab: Forward Dynamics**
- Set initial joint angles (deg), initial velocities (deg/s), and applied torques (Nm)
- Set simulation duration and time step
- Click **Run Forward Dynamics** to simulate how the robot moves under the applied torques
- Toggle robot animation and plot animation independently

**Tab: Inverse Dynamics**
- Set start and target joint angles (deg)
- Click **Run Inverse Dynamics** to compute the torques required to follow a quintic polynomial trajectory between the two poses
- Select which variables to monitor: tau, q, qd, qdd

**Tab: PID Controller**
- Select simulation engine:
  - `Legacy (Robotics Toolbox)` — always available, uses roboticstoolbox full rigid body dynamics
  - `TD3 RL (Setting A)` — requires trained checkpoint at `checkpoints/Setting_A/td3_best_actor`
  - `TD3 RL (Setting B)` — requires trained checkpoint at `checkpoints/Setting_B/td3_best_actor`
- Load preset gains via the **Gain Presets** dropdown (Baseline Setting A or B)
- Select trajectory mode:
  - **Static** — fixed setpoint per joint
  - **Waypoints** — piecewise-linear interpolation between (time, angle) pairs
  - **Sine Wave** — sinusoidal setpoint with configurable amplitude, frequency, offset, phase
- Select which joints and variables to plot (Angle, Setpoint, Error, U(k), Torque, P, I, D)
- Click **Run PID Controller**

**Tab: PID Tuning Demo**
- Demonstrates the effect of changing Kp, Ki, or Kd on Joint 2
- Runs 3 experiments with different gain values and overlays the responses
- Select which gain to vary, then click **Run Tuning Demo**

---

### 2. Train TD3 RL Agent

Trains the LSTM-based TD3 agent. Checkpoints are saved to `checkpoints/Setting_A/` or `checkpoints/Setting_B/` and are required before using the TD3 engine in the GUI or running evaluations.

```bash
# Basic run — Setting A (stronger baseline gains)
uv run python train_td3_rl.py --baseline_setting A

# Setting B (weaker baseline, ~80% of Setting A gains)
uv run python train_td3_rl.py --baseline_setting B

# Full options
uv run python train_td3_rl.py \
  --baseline_setting A \
  --max_episodes 1000 \
  --batch_size 64 \
  --start_timesteps 1000 \
  --max_timesteps 500 \
  --eval_freq 10 \
  --save_freq 100 \
  --early_stopping_patience 200
```

**Key arguments:**

| Argument | Default | Description |
|---|---|---|
| `--baseline_setting` | `A` | PID baseline to train on (`A` or `B`) |
| `--max_episodes` | `1000` | Total training episodes |
| `--batch_size` | `64` | Replay buffer sample size per update |
| `--start_timesteps` | `1000` | Steps of random exploration before policy updates |
| `--max_timesteps` | `500` | Max steps per episode |
| `--eval_freq` | `10` | Evaluate and conditionally save best model every N episodes |
| `--save_freq` | `0` | Save a periodic checkpoint every N episodes (0 = disabled) |
| `--early_stopping_patience` | `0` | Stop if no improvement for N episodes (0 = disabled) |
| `--load_model` | `""` | Path to existing checkpoint to resume from |

Training output per episode:
```
Ep 001 | Steps: 500 | Total: 500 | Tr. Rwd: -1823.4 | Tr. Err: 0.312 | C_Loss: 0.00
Ep 010 | Steps: 500 | Total: 5000 | Tr. Rwd: -243.1 | Tr. Err: 0.028 | C_Loss: 1.23 (Saved Best Eval: 0.028)
```

---

### 3. Train SAC RL Agent

Soft Actor-Critic alternative. Uses the `Puma560Env` environment (which includes feedforward torque and critic-aware safety cage where critic uncertainty reduces alpha).

```bash
uv run python train_sac_rl.py --baseline_setting A

# Full options (same flags as TD3)
uv run python train_sac_rl.py \
  --baseline_setting A \
  --max_episodes 1000 \
  --batch_size 64
```

SAC checkpoints are saved as `checkpoints/Setting_A/sac_best_*`.

---

### 4. Monitor Training with TensorBoard

A training run already exists under `runs/`. View live training curves or historical runs:

```bash
uv run tensorboard --logdir runs
```

Then open [http://localhost:6006](http://localhost:6006) in a browser.

**Logged metrics:**
- `Train/Reward` — episode reward
- `Train/Error` — final tracking error (rad) per episode
- `Train/Critic_Loss` / `Train/Actor_Loss`
- `Train/Alpha` — safety cage alpha value
- `Eval/Reward` / `Eval/Error` — evaluation metrics every `--eval_freq` episodes

---

### 5. Evaluate a Trained Agent

Loads a trained TD3 checkpoint, runs a 90° step input for all 6 joints, plots angle tracking / error / residual torque for each joint, and saves `simulation_results.png`.

```bash
# Evaluate Setting A (default)
uv run python simulate_td3.py --baseline_setting A

# Specify a custom checkpoint
uv run python simulate_td3.py --checkpoint checkpoints/Setting_A/td3_ep_500_actor

# Compare two checkpoints side by side on the same plot
uv run python simulate_td3.py --compare \
  A:checkpoints/Setting_A/td3_best_actor:SettingA \
  B:checkpoints/Setting_B/td3_best_actor:SettingB
```

---

### 6. Error Metrics (MSE, ISE, SSE)

Runs the PID simulation (and optionally the TD3 agent) and computes three error metrics per joint:

| Metric | Formula | Units | Meaning |
|---|---|---|---|
| **MSE** | `(1/N) * sum(e^2)` | rad^2 | Average squared tracking error |
| **ISE** | `sum(e^2) * dt` | rad^2*s | Total accumulated squared error over time |
| **SSE** | `mean(|e|)` over last 10% of steps | rad | Residual error after system settles |

```bash
# Default: Setting A, Joint 2 target 45 deg, 10s simulation
uv run python error_metrics.py

# Both PID baseline settings
uv run python error_metrics.py --setting both

# Custom setpoints (all 6 joints in degrees)
uv run python error_metrics.py --setpoints 0,90,45,0,0,0

# Include trained TD3 agent for comparison (prints % improvement over PID)
uv run python error_metrics.py --setting A --td3

# Change SSE window (default 10% = last 1s of a 10s run)
uv run python error_metrics.py --sse_window 0.2

# Full example
uv run python error_metrics.py \
  --setting both \
  --td3 \
  --duration 10.0 \
  --dt 0.01 \
  --setpoints 0,45,0,0,0,0 \
  --sse_window 0.1
```

**Example output:**
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

---

### 7. Generate Offline Supervised Dataset

Runs scipy L-BFGS-B optimization at every timestep to compute the optimal PID gain deltas using the PUMA 560 rigid body dynamics. The resulting dataset can be used to pre-train the LSTM via supervised learning before RL fine-tuning.

```bash
uv run python generate_offline_data.py
```

Output: `offline_pid_dataset.csv` with columns `time, q_0..5, qd_0..5, sp_0..5, err_0..5, integral_0..5, derivative_0..5, delta_Kp_0..5, delta_Ki_0..5, delta_Kd_0..5`

This will take several minutes to run since it calls `scipy.optimize.minimize` at each of the 1000 timesteps.

---

### 8. Optimize PID Gains

Finds optimal static Kp/Ki/Kd gains for all 6 joints to hold the robot at 0° using L-BFGS-B minimization. Minimizes a weighted sum of MSE + control effort across all joints.

```bash
uv run python optimize_pid.py
```

Prints optimized gain values per joint when complete.

---

### 9. Generate Research Figures

Standalone script that simulates the full PID + RL + LSTM system and saves all publication-quality figures (Figures 3–11 from the paper).

```bash
uv run python "research/Six Joint Simulation with Analysis _python_20260519_419c6a.py"
```

**Generated files:**

| File | Contents |
|---|---|
| `fig3_training_convergence.png` | Reward, tracking error, alpha ramp, loss curves over 1000 episodes |
| `fig4_tracking_comparison.png` | Bar chart: RMS error for PID-only vs PID+RL vs PID+RL+LSTM vs Proposed |
| `fig5_trajectory_tracking.png` | Example joint trajectory tracking — reference vs PID vs proposed |
| `fig6_control_effort.png` | Pie chart of PID vs RL torque contribution + per-joint breakdown |
| `fig7_lyapunov.png` | Lyapunov function V(t) evolution — stable vs unstable vs proposed |
| `fig8_ablation_study.png` | Ablation: full model vs w/o LSTM vs w/o safety cage vs PID only |
| `fig9_friction_model.png` | Friction torque vs velocity showing Stribeck + Coulomb + viscous |
| `fig10_motor_dynamics.png` | Motor current and torque response to voltage step at different speeds |
| `fig11_step_response.png` | Step response for all 6 joints with Ziegler-Nichols / Pole Placement / Conservative gains |

---

### 10. Utility Scripts

```bash
# Print PUMA 560 dynamic parameters: mass, inertia, gear ratio, friction per link
uv run python show_params.py

# Open an interactive 3D teach pendant (drag joints manually)
uv run python interactive.py
```

---

### 11. Build Standalone Executable

Two PyInstaller specs package `main.py` into a self-contained `.exe` that runs without Python installed:

```bash
# Recommended — named PumaSimulator.exe
uv run pyinstaller PumaSimulator.spec

# Alternative — named main.exe
uv run pyinstaller main.spec
```

Output is placed in `dist/PumaSimulator.exe` or `dist/main.exe`. Note: the build bundles roboticstoolbox and spatialmath data files and will be several hundred MB.

---

## Recommended First-Time Run Order

```bash
# 1. Explore the GUI with the built-in Legacy engine (no training needed)
uv run python main.py

# 2. Train the TD3 agent on Setting A
uv run python train_td3_rl.py --baseline_setting A

# 3. Monitor training progress in another terminal
uv run tensorboard --logdir runs

# 4. Once training finishes, evaluate the agent
uv run python simulate_td3.py --baseline_setting A

# 5. Launch the GUI again and switch the engine to "TD3 RL (Setting A)"
uv run python main.py

# 6. Compute and compare error metrics
uv run python error_metrics.py --setting A --td3

# 7. Generate all research figures
uv run python "research/Six Joint Simulation with Analysis _python_20260519_419c6a.py"
```

---

## PID Baseline Settings

Two pre-tuned baseline PID gain sets are available. Setting A uses stronger gains; Setting B uses approximately 80% of Setting A.

| Joint | Setting A Kp | Setting A Ki | Setting A Kd | Setting B Kp | Setting B Ki | Setting B Kd |
|---|---|---|---|---|---|---|
| J1 (Waist) | 800 | 50 | 40 | 652.4 | 41.2 | 32.6 |
| J2 (Shoulder) | 1200 | 80 | 60 | 978.6 | 61.8 | 48.9 |
| J3 (Elbow) | 800 | 50 | 40 | 652.4 | 41.2 | 32.6 |
| J4 (Wrist Roll) | 200 | 10 | 15 | 163.1 | 10.3 | 8.2 |
| J5 (Wrist Bend) | 200 | 10 | 15 | 163.1 | 10.3 | 8.2 |
| J6 (Wrist Swivel) | 100 | 5 | 8 | 81.15 | 5.1 | 4.1 |

---

## Architecture Overview

```
Observation (20 x 42 history window)
        |
        v
  LSTM Feature Extractor
  (2-layer, hidden=256)
        |
        v
  Current State (42-dim) + Context Vector (6-dim)
        |
        v
  MLP Actor: 304 -> 256 -> 128 -> 6
        |
        v
  Residual Torque (6 joints, bounded by tau_max)
        |
        v
  Safety Cage (Lyapunov constraint)
  ||tau_rl|| <= alpha * ||tau_pid||
        |
        v
  tau_total = tau_pid + tau_safe_residual
        |
        v
  PUMA 560 Plant
  (motor dynamics + friction + gravity + inertia)
```

The Safety Cage enforces three constraints simultaneously:
1. **Magnitude bound** — residual norm cannot exceed `alpha * ||tau_pid||`
2. **Direction agreement** — residual cannot point strongly against the PID direction
3. **Power bound** — instantaneous power of residual cannot exceed 50W

`alpha` starts at 0.05 and ramps linearly to 0.30 over the first 300 training episodes, progressively granting the agent more authority as it becomes more competent.
