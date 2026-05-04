# Deep Residual PID Gain Tuning: Technical Architecture

This document serves as the formal technical specification for the unified DCNN-LSTM Control Architecture applied to the PUMA 560 Robot Arm. It breaks down the system dynamics, the neural network structure, and the Reinforcement / Supervised learning algorithms mathematically and programmatically.

---

## 1. The Control Algorithm: Residual Policy Tuning

To ensure absolute safety bounds while allowing for highly non-linear adaptation, the system employs **Residual Gain Tuning**. Rather than replacing the classical controller, the Neural Network acts as a parameterized additive policy on top of a highly-stable foundational baseline.

### 1.1 The Baseline Control Law
The robot operates fundamentally on an independent-joint Proportional-Integral-Derivative (PID) controller. For any joint $i$, the baseline torque applied is:
$$ \tau_{base,i}(t) = K_{P,i}^{base} e_i(t) + K_{I,i}^{base} \int e_i(t)dt + K_{D,i}^{base} \frac{de_i(t)}{dt} $$
Where $e_i(t)$ is the angular error $(q_{target} - q(t))$.

### 1.2 The Residual Nudge
The Neural Network policy $\pi_\theta(s_t)$ computes an action $a_t \in \mathbb{R}^{18}$ consisting of dynamic offsets $\Delta K_P, \Delta K_I, \Delta K_D$ for all 6 joints. The total combined gain used for the control tick $t$ becomes:
$$ K_{TOTAL} = K_{base} + Clamp(\Delta K) $$

**Safety Clamp Guarantee:**
To prevent explosive positive feedback, the control constraints enforce mathematically that physical gains never drop below zero:
```python
K_active = max(0.0, K_base + scaled_delta)
```

---

## 2. Neural Network Architecture (DCNN + LSTM)

The agent utilizes a hybrid spatial-temporal neural network designed to capture local velocity/momentum dynamics (DCNN) and long-horizon trajectory phases (LSTM).

### 2.1 The State Space (Input)
The observation space $S$ fed to the network is a temporal sliding window $W$ of length $T=10$ (representing the last 100ms of state). At each timestep, a 24-dimensional feature vector is polled:
*   $q_{1..6}$ (Positions)
*   $\dot{q}_{1..6}$ (Velocities)
*   $e_{1..6}$ (Positional Errors)
*   $\int e_{1..6}$ (Leaky Error Integrals)

Input Tensor Shape: `(Batch, Sequence_Length=10, Features=24)`

### 2.2 Deep 1D-CNN (Feature Extraction)
Prior to recurrent processing, the sequence passes through two `Conv1D` temporal feature extractors. This builds localized representations of acceleration and momentum without manual finite-difference calculations.
*   **Conv1:** `in_channels=24, out_channels=32, kernel=3, padding=1`
*   **Conv2:** `in_channels=32, out_channels=64, kernel=3, padding=1`

### 2.3 Long Short-Term Memory (LSTM)
The 64-dimensional feature maps act as sequence inputs for a 2-layer LSTM. The LSTM hidden states persist time-series understandings of gravity shifts and positional phases.
*   **Input:** `(Batch, 10, 64)`
*   **Output:** The final hidden state projection `out[:, -1, :]` of size 64.

### 2.4 The Action Heads (Actor-Critic)
*   **Actor (Policy):** An MLP transforming the 64-dim LSTM context into the 18 Continuous Actions. The final layer utilizes a `Tanh` bounding activation `[-1, 1]`, which is then heavily scaled via hyperparameters (e.g. `Max_Kp_Delta = ±50`).
*   **Critic (Value):** A separate MLP mapping the context to a single scalar $V(s)$. This determines the expected future cumulative reward of the state, used primarily internally by the PPO gradient updates.

---

## 3. The PPO Reinforcement Learning Algorithm

If deployed for self-guided exploration, the system is backed by **Proximal Policy Optimization (PPO)**. 

### 3.1 Reward Function $\mathcal{R}_t$
The agent is motivated via a dense reward function penalizing trajectory spread and absolute energy expenditure (to prevent the AI from adopting erratic, highly-oscillatory gain-switching).

$$ \mathcal{R}_t = - \alpha \sum_{i} (\Delta q_i)^2 - \beta \sum_{i} (\tau_i)^2 $$
Where $\alpha = 1.0$ (Tracking Penalty) block and $\beta = 0.001$ (Energy Penalty).

### 3.2 Optimization Flow
1.  **Rollout Phase:** The agent gathers batches of trajectories transitioning through states $s_t$, making predictions $a_t$, observing rewards $r_t$, and determining advantages $A_t = R_t - V_\theta(s_t)$.
2.  **Surrogate Loss Clipping:** PPO updates the actor to maximize expected rewards, but tightly clamps the gradient step $\epsilon=0.2$ to ensure the network isn't aggressively destroyed by one anomalously optimal or terrible batch.

---

## 4. Supervised Pre-Training (The Inverse Dynamics Solver)

To accelerate learning and avoid the "Random Flailing" problem native to initial RL, the system utilizes an offline Data Generator that relies on explicit Inverse Kinematics.

### 4.1 System Dynamics Equation
For a PUMA 560, the rigid body dynamics are formalized as:
$$ \tau = M(q)\ddot{q} + C(q,\dot{q})\dot{q} + G(q) $$

### 4.2 Calculating "The Perfect Delta"
If the goal is to shift from $q_t$ to $q_{target}$ in the next control step $dt$, we calculate $\ddot{q}_{ideal}$. Substituting this into the System Dynamics equation gives $\tau_{ideal}$ — the exact necessary torque.
Instead of feeding $\tau_{ideal}$ to the motors directly, an optimization loop `scipy.optimize.minimize` solves the underdetermined system of establishing the minimum bounded offsets $\delta K$:
$$ Minimize: \lambda( \delta K_{P}^2 + \delta K_{I}^2 + \delta K_{D}^2 ) $$
*Subject to generating $\tau_{ideal}$.*

This creates highly condensed, purely mathematical dataset records that the LSTM is then supervised across using `MSELoss`, bootstrapping its understanding of physical kinematics inherently into its weights prior to RL fine-tuning.
