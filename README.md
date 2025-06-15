# Networked Control System Stability Simulation

This project simulates the stability and control strategies within a networked control system, expanding upon concepts introduced in the paper **"Semantics of Instability in Networked Control."** It explores different control policies (based on G-function, F-function, and Lyapunov drift) to manage the system's **Age of System Instability (AOIS).**

---

## 📄 Project Overview

The code models a system that can be in either a **stable** (AOIS = 0) or **unstable** (AOIS > 0) state. A controller attempts to stabilize the system by sending messages, which can be **compressed** or **uncompressed**, each incurring a different energy cost.

The core objective is to **minimize a cost function** that considers both the energy expended on control actions and the "cost" of being in an unstable state (represented by the AOIS value).

---

## ✨ Features

- **Multiple Control Policies**: Simulate system behavior under three distinct control strategies:  
  - **G-function based**: Decision-making minimizing a cost-to-go function (`determin_next_G`).  
  - **F-function based**: Decision-making minimizing a cost related to system state transitions (`determin_next_F`).  
  - **Lyapunov Drift based**: Control decisions driven by minimizing Lyapunov drift to achieve stability (`pick_lyapunov`).  

- **Dynamic AOIS Simulation**: Track the Age of System Instability over time, observing how control actions influence its evolution.

- **Parameter Customization**: Easily adjust key system and control parameters to explore various scenarios.

- **Optimal Parameter Search**: Mode to find optimal parameters for the Lyapunov function.

- **Visualizations**: Plots showing:  
  - Evolution of AOIS over time with control actions highlighted  
  - Distribution of AOIS values during simulation  
  - Lyapunov drift over time and convergence

---

## ⚙️ Parameters

| Parameter         | Description                                                                                      |
|-------------------|--------------------------------------------------------------------------------------------------|
| `r1`              | Probability system remains unstable independently of control actions                             |
| `r0`              | Probability system remains stable independently of control actions                               |
| `rho`             | Probability a message is successfully decoded                                                    |
| `p`               | Probability system stabilizes after receiving compressed message                                 |
| `q`               | Probability system stabilizes after receiving uncompressed message                               |
| `lambda1`         | Energy cost for sending a compressed message                                                    |
| `lambda2`         | Energy cost for sending an uncompressed message                                                 |
| `V_val`           | Constant multiplier used internally for Lyapunov function scaling (not final cost)              |
| `stabilityMargine`| Penalty weight in Lyapunov drift calculation influencing stability preference                    |
| `RANDOM_SEED`     | Seed for RNG. Set to `None` to disable fixed seeding                                            |
| `h`               | Exponent used in Lyapunov function variant                                                     |
| `mode`            | Simulation mode selector (see below)                                                           |

---

### Mode Selector

| Mode | Description                                                  |
|-------|-------------------------------------------------------------|
| `-1`  | Find optimal Lyapunov parameters `a, b, c`                  |
| `0`   | Debugging mode running `run_sim`                            |
| `1`   | Run G-function based simulation (`run_sim_1`)                |
| `2`   | Run F-function based simulation (`run_sim_2`)                |
| `3`   | Run Lyapunov drift based simulation (`run_sim_3`)            |

---

## 💻 Code Structure and Key Functions

- **`get_transition_probs(st)`**  
  Calculates transition probabilities from current state `st` for all actions (Idle, Compressed, Uncompressed).

- **`stable_unstable(st, dt, prs)`**  
  Determines the next system state (stable or unstable) based on current state `st`, action `dt`, and transition probabilities `prs`.

- **`determin_next_F(st)`**  
  Implements the F-function based control policy selecting the action minimizing expected future cost.

- **`determin_next_G(st)`**  
  Implements the G-function based control policy selecting the action minimizing immediate expected cost.

- **`pick_lyapunov(st, a, b, c)`**  
  Lyapunov drift-based control policy selecting action that minimizes expected Lyapunov drift plus cost.

- **`V(x, a, b, c, h)`**  
  Lyapunov function defined as \( V(x) = a \cdot x^b + h \cdot c \), quantifying instability "badness."

- **`run_sim(numruns)` (mode 0)**  
  Debugging simulation to test early policy logic.

- **`run_sim_1(numruns)` (mode 1)**  
  Simulates system using G-function control policy.

- **`run_sim_2(numruns)` (mode 2)**  
  Simulates system using F-function control policy.

- **`run_sim_3(num_runs, a, b, c)` (mode 3)**  
  Simulates system using Lyapunov drift control policy with custom parameters.

- **`find_optimal_V()` (mode -1)**  
  Grid search for optimal Lyapunov parameters minimizing average AOIS.

- **`simulate_lyapunov(max_steps)`**  
  Simulates Lyapunov drift over time, checking convergence.

- **`plot_lyapunov_drift(max_range, n1_range, n2_range)`**  
  Visualizes Lyapunov drift and its convergence.

- **`plot_state_distribution(S_states)`**  
  Plots histogram and KDE of AOIS values observed in simulation.

---

## 🚀 Getting Started

### Prerequisites

- Python installed
- Required packages:

```bash
pip install numpy matplotlib seaborn
```

### Running the Simulation
- to run the results from the paper:
```bash
python plotter.py
```

-to run the new results
```bash
python thesis.py
```