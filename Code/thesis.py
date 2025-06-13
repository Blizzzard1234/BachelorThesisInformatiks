import numpy as np
import matplotlib.pyplot as plt
import argparse
import random

###################### Parameters
r1 = 0.9
r0 = 0.1
rho = 0.1 # pr that a package was successfully decoded
p = 0.5 # pr that we become stable when controler recives a compressed message
q = 0.9  #pr that we become stable when controler recives a uncompressed message
lambda1 = 2  # Energy cost for compressed
lambda2 = 4  # Energy cost for uncompressed
V_val = 1

debuggind_mode = 0



# Derived constants (stable === AOIS = 0; unstable === AOIS > 0)
a = r1                          #unstable & IDLE
b = r1 * (1 - rho * p)          #unstable & Compressed
c = r1 * (1 - rho * q)          #unstable & Uncompressed
d = 1 - r0                      #stable & IDLE
e = (1 - r0) * (1 - rho * p)    #stable & Compressed
f = (1 - r0) * ((1 - rho) + rho * (1 - q) )      #stable &  Uncompressed  this is not the same as the paper, since we also have the propability that we get something
# and deconde it, but it doesnt lead to a stable state ((1-r0) * rho * (1 - q))


######################### current stuff

# this function calculates the next step. it uses the F function
def calculate_next(st, dt):
    if st == 0:
        if dt == 0:
            prob_increment = 1 - r0
        elif dt == 1:
            prob_increment = (1 - r0) * (1 - rho * p)
        elif dt == 2:
            prob_increment = (1 - r0) * (1 - rho * q)
        else:
            raise ValueError(f"Something went wrong. we are in an invalid state {dt}")
    elif st > 0:
        if dt == 0:
            prob_increment = r1
        elif dt == 1:
            prob_increment = r1 * (1 - p * rho)
        elif dt == 2:
            prob_increment = r1 * (1 - q * rho)
        else:
            raise ValueError(f"Something went wrong. we are in an invalid state {dt}")

    # Determine the next state based on the calculated probability
    #TODO seed for determinismus.
    # TODO save the propabiliry for each type, as well as the overall propability we currently have for stability vs instability in an array
    if random.random() < prob_increment:
        return st + 1
    else:
        return 0

def determin_next_F(st):
    g_values = {}  # Dictionary to store G(S, action) values for each action

    if st == 0:
        g_values[0] = 1 - r0  # G_idle(0) = 0
        g_values[1] = (1 - r0) * (1 - rho * p) + V_val * lambda1  # G_sparse(0)
        g_values[2] = (1 - r0) * (1 - rho * q) + V_val * lambda2  # G_dense(0)
    else:
        g_values[0] = r1 * (1 + st)  # G_idle(S_t) = 0
        g_values[1] = r1 * (1 - p * rho) * (1 + st) + V_val * lambda1  # G_sparse(S_t)
        g_values[2] = r1 * (1 - q * rho) * (1 + st) + V_val * lambda2  # G_dense(S_t)

    # Find the action with the minimum G value
    optimal_action = min(g_values, key=g_values.get)
    K = (lambda2 - lambda1) / (q - p)
    if optimal_action > 0:
        if st > 0:
            if K > rho * r1 * (1 + st):
                optimal_action = 1
            else:
                optimal_action = 2
        else:
            if K > rho*(1-r0):
                optimal_action = 1
            else:
                optimal_action = 2

    return optimal_action, g_values

# this function calculates the next step. it uses the G function
def determin_next_G(st):
    g_values = {}  # Dictionary to store G(S, action) values for each action

    if st == 0:   #the 1 and 2 are different if the lambda is different
        g_values[0] = 0.0  # G_idle(0) = 0
        g_values[1] = V_val * lambda1 - rho * p * (1 - r0)  # G_sparse(0)
        g_values[2] = V_val * lambda2 - rho * q * (1 - r0)  # G_dense(0)
    else:
        g_values[0] = 0.0  # G_idle(S_t) = 0
        g_values[1] = V_val * lambda1 - rho * p * r1 * (1 + st)  # G_sparse(S_t)
        g_values[2] = V_val * lambda2 - rho * q * r1 * (1 + st)  # G_dense(S_t)

    # Find the action with the minimum G value
    optimal_action = min(g_values, key=g_values.get)

    K = (lambda2 - lambda1) / (q - p)
    if optimal_action > 0:
        if st > 0:
            if K > rho * r1 * (1 + st):
                optimal_action = 1
            else:
                optimal_action = 2
        else:
            if K > rho*(1-r0):
                optimal_action = 1
            else:
                optimal_action = 2
    return optimal_action, g_values


def pick_lyapunov(st, history, t):
    values = {}
    avg_val = {}

    values[0] = calculate_next(st, 0)
    values[1] = calculate_next(st, 1)
    values[2] = calculate_next(st, 2)

    ly = V(st)
    history[0][t] = (V(values[0]) - ly)
    history[1][t] = (V(values[1]) - ly) + lambda1 * V_val #TODO maybe try finding a variable V value, instead of a constant and see how that works out
    history[2][t] = (V(values[2]) - ly) + lambda2 * V_val

#TODO history is the same for each, exccept for the last value. so only that last value matters.
    avg_val[0] = history[0].mean()
    avg_val[1] = history[1].mean()
    avg_val[2] = history[2].mean()

    optimal_action = min(avg_val, key=avg_val.get)

    return optimal_action, history





def run_sim_1 (numruns = 1000):
    st = 0
    dt = 0

    n1 = (lambda1 / (r1 * rho * p)) - 1
    n2 = ((lambda2 - lambda1) / (r1 * rho * (q - p))) - 1

    for run in range(numruns):

        nextmin = determin_next_G(st)


        #detirmin if the next interaction stabalizes the system
        st = calculate_next(st, nextmin)


        #tests
        if st > 0:
            if n1 > st and n2 < st:  #this might be the wrong way around. I switched them here since n1 = 1999 and n2 = -1
                print(f"failure with AoIS: {st}; current action: {dt}")
        else:
            K = (lambda2 - lambda1) / (q - p)
            if rho * (1 - r0) < K:
                if dt != 1:
                    print(f"failure with AoIS: {st}; current action: {dt}, should be 1")
            else:
                if dt != 2:
                    print(f"failure with AoIS: {st}; current action: {dt}, should be 2")

def run_sim (numruns = 1000):
    S_history = [0]  # Initialize history with the starting state
    action_history = []
    st = 0

    for t in range(numruns):
        # 1. Determine the optimal action for the current state S(t)
        action, action_g_values = determin_next_F(st)
        action_history.append(action)

        # 2. Calculate the next state S(t+1) based on the chosen action
        next_S = calculate_next(st, action)
        S_history.append(next_S)
        st = next_S  # Update current_S for the next iteration

    return S_history, action_history


def run_sim_3(num_runs):

    S_history = [0]  # Initialize history with the starting state
    action_history = []
    st = 0
    history = np.zeros((3, num_runs)) #FIX ME the history is both wrong, and not necesery.
    for t in range(num_runs):
        # 1. Determine the optimal action for the current state S(t)
        action, history= pick_lyapunov(st, history, t)
        action_history.append(action)

        # 2. Calculate the next state S(t+1) based on the chosen action
        next_S = calculate_next(st, action)
        S_history.append(next_S)
        st = next_S  # Update current_S for the next iteration

    return S_history, action_history







#TODO try different variations, including a nested loop part where other values are tries
# stuff like a * x ** b + h * c with a,b,c as variable loops and h as a constant
def V(x):  # x = timestep
    return 0.5 * x ** 2


def has_converged(arr):
    isNeg = False
    conc = 0
    pnt = -1
    for i in np.arange(0, len(arr)):
        if arr[i] < 0:
            conc += 1
        else:
            conc = 0
            isNeg = False

        if not isNeg:
            isNeg = True
            pnt = i

    return pnt

def simulate_lyapunov(max_steps):


    st = 0  # 0 = stable; 1 = unstable
    drift_vals = np.zeros_like(range(0,max_steps), dtype=float)
    avg_val = np.zeros_like(range(0, max_steps), dtype=float)
    convergence = -1
    for x in range(max_steps):
        nextmin, valu = determin_next_G(st)

        # detirmin if the next interaction stabalizes the system
        st1 = calculate_next(st, nextmin)
        drift = V(st1) - V (st)
        drift_vals[x] = drift
        avg_val[x] = drift_vals.mean()#
        convergence = has_converged(avg_val)
    return convergence, drift_vals, avg_val




def plot_lyapunov_drift(max_range, n1_range=15, n2_range=25):
    convergence, drift_vals, avg_val = simulate_lyapunov(max_range)

    avg_values_range = 100
    stability_vals = np.zeros_like(range(0, avg_values_range), dtype=float)

    for i in range(avg_values_range):
        stability_vals[i], arr1, arr2 = simulate_lyapunov(500)

    avg_stability = np.mean(stability_vals)

    #bellman = bellman_stability(max_range)

    #scrapped this becauzse its crazy inaccurate
    #coeffs  = np.polyfit(range(0,max_range), avg_val, deg=10)  #hopefully gets a function that approximates lyap
    #f = np.poly1d(coeffs)
    #print(f)

    plt.figure(figsize=(10, 6))
    plt.plot(range(0,max_range), drift_vals, label='Lyapunov Drift at time t')
    plt.plot(range(0, max_range), avg_val, label='Mean drift at time t')
    #plt.plot(range(0, max_range), bellman, label='Bellman stability')
    if not convergence == -1 and not convergence == max_range:
        plt.axvline(convergence, color='red', linestyle='--', label=f'Lyapunov stability at {convergence}')
    plt.axhline(0, color='red', linestyle='--', label='Zero Drift Line')
    plt.yscale('symlog')
    plt.title("Lyapunov Drift at time t")
    plt.suptitle(f'r₀ = {r0}; r₁ = {r1}; ρ = {rho}. Average stability over 100 tries: = {avg_stability}')
    plt.xlabel("Time Step")
    plt.ylabel("Drift ΔV(x)")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    num_steps = 10000

    if debuggind_mode == 1:
        run_sim_1(num_steps)
    elif debuggind_mode == 2:
        plot_lyapunov_drift(num_steps)
    else:
        n = 2
        avgs = np.zeros_like(range(0,n), dtype=float)

        for i in range(n):
            S_states, actions = run_sim(num_steps)
            avgs[i] = (sum(S_states) / len(S_states))

        idle_actions = actions.count(0)
        sparse_actions = actions.count(1)
        dense_actions = actions.count(2)

        print("\n--- Simulation Summary ---")
        print(f"Total simulation steps: {num_steps}")
        print(f"Average Age Of System Instability (ADSI): {avgs[1]:.4f}")
        print(f"Average Age Of System Instability (ADSI) over {n}: {avgs.mean():.4f}")
        print(f"Breakdown of actions taken:")
        print(f"  Idle (0): {idle_actions} ({idle_actions / num_steps:.2%})")
        print(f"  Sparse Update (1): {sparse_actions} ({sparse_actions / num_steps:.2%})")
        print(f"  Dense Update (2): {dense_actions} ({dense_actions / num_steps:.2%})")
        print(f"Cost: {sparse_actions*lambda1 + dense_actions*lambda2}")  #TODO it should also include the cost of it beeing in a bad state, not just communicvation

        #TODO for lyapunov, in addition to this, also calculate the cost (st + ld1 + ld2) with st = V(st)
        # --- Visualize S(t) over Time ---

        #TODO add the distribution of AOSI with the avg as well as the number of times we have each IOSI
        try:
            plt.figure(figsize=(14, 7))
            plt.plot(range(len(S_states)), S_states, label='S(t) - Age of System Instability', color='royalblue')
            plt.title('Simulated Age of System Instability (S(t)) Over Time', fontsize=16)
            plt.xlabel('Time Step', fontsize=12)
            plt.ylabel('S(t) Value', fontsize=12)
            plt.grid(True, linestyle='--', alpha=0.7)
            plt.axhline(y=0, color='red', linestyle=':', linewidth=1, label='S(t) = 0 (Good State)')
            plt.legend(fontsize=10)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError plotting with Matplotlib: {e}")
            print("Please ensure matplotlib is installed (`pip install matplotlib`) if you want to see the plot.")