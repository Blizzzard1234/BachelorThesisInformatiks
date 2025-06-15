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

- **`get_transition_probs(st: int) -> np.ndarray`**  
  **Variables**:  
  - `st` (int): Current system state  
  **Returns**:  
  - `np.ndarray`: Array `[Pr[Idle], Pr[Compressed], Pr[Uncompressed]]` representing transition probabilities for all actions from state `st`.

---

- **`stable_unstable(st: int, dt: int, prs: np.ndarray) -> int`**  
  **Variables**:  
  - `st` (int): Current system state  
  - `dt` (int): Selected action (0=Idle, 1=Compressed, 2=Uncompressed)  
  - `prs` (np.ndarray): Transition probabilities from `get_transition_probs()`  
  **Returns**:  
  - `int`: Next system state (0 = stable, 1 = unstable)

---

- **`determin_next_F(st: int) -> tuple[int, np.ndarray]`**  
  **Variables**:  
  - `st` (int): Current system state  
  **Returns**:  
  - `int`: Optimal action chosen based on minimizing expected future cost  
  - `np.ndarray`: Transition probability array for chosen action

---

- **`determin_next_G(st: int) -> tuple[int, np.ndarray]`**  
  **Variables**:  
  - `st` (int): Current system state  
  **Returns**:  
  - `int`: Optimal action minimizing immediate expected cost  
  - `np.ndarray`: Transition probability array for chosen action

---

- **`pick_lyapunov(st: int, a: float, b: float, c: float) -> tuple[int, np.ndarray]`**  
  **Variables**:  
  - `st` (int): Current system state  
  - `a`, `b`, `c` (float): Lyapunov parameters  
  **Returns**:  
  - `int`: Optimal action minimizing Lyapunov drift + cost  
  - `np.ndarray`: Transition probabilities under chosen action

---

- **`V(x: int, a: float, b: float, c: float, h: float) -> float`**  
  **Variables**:  
  - `x` (int): Current AOIS state  
  - `a`, `b`, `c`, `h` (float): Parameters of Lyapunov function  
  **Returns**:  
  - `float`: Computed value of Lyapunov function \( V(x) = a \cdot x^b + h \cdot c \)

---

- **`run_sim(numruns: int) -> None`** *(mode 0)*  
  **Variables**:  
  - `numruns` (int): Number of simulation steps  
  **Returns**:  
  - `None`: Runs a debug simulation for logic testing

---

- **`run_sim_1(numruns: int) -> tuple[np.ndarray, np.ndarray]`** *(mode 1)*  
  **Variables**:  
  - `numruns` (int): Number of simulation steps  
  **Returns**:  
  - `np.ndarray`: AOIS state history  
  - `np.ndarray`: Action history using G-function policy

---

- **`run_sim_2(numruns: int) -> tuple[np.ndarray, np.ndarray]`** *(mode 2)*  
  **Variables**:  
  - `numruns` (int): Number of simulation steps  
  **Returns**:  
  - `np.ndarray`: AOIS state history  
  - `np.ndarray`: Action history using F-function policy

---

- **`run_sim_3(numruns: int, a: float, b: float, c: float) -> tuple[np.ndarray, np.ndarray]`** *(mode 3)*  
  **Variables**:  
  - `numruns` (int): Number of simulation steps  
  - `a`, `b`, `c` (float): Lyapunov parameters  
  **Returns**:  
  - `np.ndarray`: AOIS state history  
  - `np.ndarray`: Action history using Lyapunov drift policy

---

- **`find_optimal_V() -> np.ndarray`** *(mode -1)*  
  **Returns**:  
  - `np.ndarray`: Array of parameter combinations and their performance, sorted by AOIS. Top 10 printed to console.

---

- **`simulate_lyapunov(max_steps: int) -> tuple[list[float], list[float]]`**  
  **Variables**:  
  - `max_steps` (int): Simulation duration  
  **Returns**:  
  - `list[float]`: Lyapunov drift values per step  
  - `list[float]`: Convergence values per step

---

- **`plot_lyapunov_drift(max_range: int, n1_range: int, n2_range: int) -> None`**  
  **Variables**:  
  - `max_range`, `n1_range`, `n2_range` (int): Configuration for simulation length and drift analysis  
  **Returns**:  
  - `None`: Plots Lyapunov drift and convergence results

---

- **`plot_state_distribution(S_states: np.ndarray) -> None`**  
  **Variables**:  
  - `S_states` (np.ndarray): AOIS state history array  
  **Returns**:  
  - `None`: Displays histogram and KDE of AOIS distribution



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
