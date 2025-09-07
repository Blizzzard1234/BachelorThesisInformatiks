import math
import sys
from collections import deque

import matplotlib
import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import seaborn as sns
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor, as_completed
from tqdm import tqdm
import itertools
import os
from datetime import datetime
import time
###################### Parameters
r1 = 0.7 #pr that the state will remain unstable, independently of everything else
r0 = 0.2 #pr that the state will remain stable, independently of everything else
rho = 0.7 # pr that a package was successfully decoded
p = 0.5 # pr that we become stable when controler recives a compressed message
q = 0.9  #pr that we become stable when controler recives a uncompressed message
lambda1 = 2  # Energy cost for compressed
lambda2 = 6  # Energy cost for uncompressed
V_val = 1 # a constant multiplied with all lambda, except for cost calculation. this will only
# effect the next step calculation, but not the calculation of the final cost, so it is only a
#temporary factor to help calculations
stabilityMargine = 0
RANDOM_SEED = 1585645 # Change or set to None to disable fixed seeding
h = 0 # variant exponent
MAX_I = 100000  # Should be large enough to approximate infinite sums
#should it be computed numerically (sum from min -> min + MAX_I) => True; should be calculated using the closed_form version like given in the paper => False
numeric_calculation = False
show_all_plots = False
should_save = True
dynamic = False

#dont edit this
savedynamic = False
matplotlib.use("Agg")

# Derived constants (stable === AOIS = 0; unstable === AOIS > 0)
a = r1                          #unstable & IDLE
b = r1 * (1 - rho * p)          #unstable & Compressed
c = r1 * (1 - rho * q)          #unstable & Uncompressed
d = 1 - r0                      #stable & IDLE
e = (1 - r0) * (1 - rho * p)    #stable & Compressed
f = (1 - r0) * ((1 - rho) + rho * (1 - q) )      #stable &  Uncompressed  this is not the same as the paper, since we also have the propability that we get something
# and deconde it, but it doesnt lead to a stable state ((1-r0) * rho * (1 - q))
queue0 = deque(maxlen=100)
queue1 = deque(maxlen=100)
queue2 = deque(maxlen=100)
rng = random.Random()
######################### current stuff
def reset_random():
    global rng
    rng = random.Random(RANDOM_SEED)
    np.random.seed(RANDOM_SEED)

history_r1, history_r0, history_rho, history_p, history_q = [], [], [], [], []

def change_prob():
    global r1,r0,rho,p,q
    r1 = min(1, max(0, r1 + np.random.uniform(-0.005, 0.005)))
    r0 = min(1, max(0, r0 + np.random.uniform(-0.005, 0.005)))
    rho = min(1, max(0, rho + np.random.uniform(-0.005, 0.005)))
    p = min(1, max(0, p + np.random.uniform(-0.005, 0.005)))
    q = min(1, max(p, min(1, q + np.random.uniform(-0.005, 0.005))))
    if savedynamic:
        history_r1.append(r1)
        history_r0.append(r0)
        history_rho.append(rho)
        history_p.append(p)
        history_q.append(q)
    # q >= p
# this function calculates the next step. it uses the F function
def get_transition_probs(st,t=0):
    if not dynamic:
        if st == 0:
            return {
                0: 1 - r0,
                1: (1 - r0) * (1 - rho * p),
                2: (1 - r0) * (1 - rho * q)
            }
        else:
            return {
                0: r1,
                1: r1 * (1 - p * rho),
                2: r1 * (1 - q * rho)
            }
    else:
        return {
            0: sum(queue0) / len(queue0),
            1: sum(queue1) / len(queue1),
            2: sum(queue2) / len(queue2)
        }

def stable_unstable(st, dt, prs):
    stable = False
    if dynamic:
        change_prob()
    # Random change of stability
    if stable and rng.random() > r0:
        stable = False
    elif not stable and rng.random() > r1:
        stable = True

    #has the package been recived
    if rng.random() < rho:
        if dt == 1 and rng.random() < p:  # compressed packet
            stable = True
        elif dt == 2 and rng.random() < q:  # uncompressed packet
            stable = True

    if not stable:
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


    return optimal_action, g_values


def pick_lyapunov(st,a,b,c):


    vals = {}

    prs = get_transition_probs(st)


    ly = V(st,a,b,c)

    #This should give the expected |E value
    vals[0] = (( V(st+1,a,b,c) - ly) +(st +1) * stabilityMargine) * prs[0] + (V(0,a,b,c) - ly) * (1 - prs[0])
    vals[1] = ((V(st+1,a,b,c) - ly) + lambda1 * V_val + (st +1)* stabilityMargine) * prs[1] + (V(0,a,b,c) - ly + lambda1 * V_val) * (1 - prs[1])
    vals[2] = ((V(st+1,a,b,c) - ly) + lambda2 * V_val + (st +1)* stabilityMargine) * prs[2] + (V(0,a,b,c) - ly + lambda2 * V_val) * (1 - prs[2])



    optimal_action = min(vals, key=vals.get)

    return optimal_action, prs.copy()





def run_sim (numruns = 1000):
    st = 0
    dt = 0
    queue0.append(1 - r0)
    queue1.append((1 - r0) * (1 - rho * p))
    queue2.append((1 - r0) * (1 - rho * q))
    n1 = (lambda1 / (r1 * rho * p)) - 1
    n2 = ((lambda2 - lambda1) / (r1 * rho * (q - p))) - 1

    for run in range(numruns):

        nextmin = determin_next_G(st)


        #detirmin if the next interaction stabalizes the system
        prs = get_transition_probs(st)

        st = stable_unstable(st, nextmin, prs)


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

def run_sim_1 (numruns = 1000):
    S_history = [0]  # Initialize history with the starting state
    action_history = []
    st = 0
    queue0.append(1 - r0)
    queue1.append((1 - r0) * (1 - rho * p))
    queue2.append((1 - r0) * (1 - rho * q))
    for t in range(numruns):
        # 1. Determine the optimal action for the current state S(t)
        action, _ = determin_next_G(st)
        action_history.append(action)

        # 2. Calculate the next state S(t+1) based on the chosen action
        prs = get_transition_probs(st)

        next_S = stable_unstable(st, action, prs)

        S_history.append(next_S)
        st = next_S  # Update current_S for the next iteration

    return S_history, action_history

def run_sim_2 (numruns = 1000):
    S_history = [0]  # Initialize history with the starting state
    action_history = []
    st = 0
    queue0.append(1 - r0)
    queue1.append((1 - r0) * (1 - rho * p))
    queue2.append((1 - r0) * (1 - rho * q))
    for t in range(numruns):
        # 1. Determine the optimal action for the current state S(t)
        action, _ = determin_next_F(st)
        action_history.append(action)

        # 2. Calculate the next state S(t+1) based on the chosen action
        prs = get_transition_probs(st)

        next_S = stable_unstable(st, action, prs)

        S_history.append(next_S)
        st = next_S  # Update current_S for the next iteration

    return S_history, action_history


def run_sim_3(num_runs, a = 0.5,  b = 2, c = 0):
    S_history = [0]  # Initialize history with the starting state
    action_history = []
    st = 0

    queue0.append(1 - r0)
    queue1.append((1 - r0) * (1 - rho * p))
    queue2.append((1 - r0) * (1 - rho * q))
    for t in range(num_runs):
        # 1. Determine the optimal action for the current state S(t)
        action, _= pick_lyapunov(st,a,b,c)
        action_history.append(action)

        # 2. Calculate the next state S(t+1) based on the chosen action
        probs = get_transition_probs(st)
        next_S = stable_unstable(st, action, probs)
        if dynamic:
            if action == 0:
                queue0.append(1 if next_S > 1 else 0)
            elif action == 1:
                queue1.append(1 if next_S > 1 else 0)
            else:
                queue2.append(1 if next_S > 1 else 0)

        S_history.append(next_S)
        st = next_S  # Update current_S for the next iteration

    return S_history, action_history


def find_most_common(arr):
    values, counts = np.unique(arr, return_counts=True)
    most_common = values[np.argmax(counts)]

    return most_common


def find_optimal_V():
    results = []
    for a in np.arange(0, 5, 0.1):  # a value
        for b in range(0, 15, 1):  # integer range is fine
            st, actions = run_sim_3(100, a, b, 0)
            actions = np.array(actions)
            st = np.array(st)

            #num_idle = np.count_nonzero(actions == 0)
            num_comp = np.count_nonzero(actions == 1)
            num_uncomp = np.count_nonzero(actions == 2)
            num_instable = 0

            for i in st:
                if not i == 0:
                    num_instable = num_instable + i

            cost = lambda1 * num_comp + lambda2 * num_uncomp + num_instable
            mean_val = st.mean()
            most_common_val = find_most_common(st)
            results.append({
                'a': a,
                'b': b,
                'c': 0,
                'mean': mean_val,
                'most_common': most_common_val,
                'cost' : cost
            })

    sorted_results_AoSI = sorted(results, key=lambda r: r['mean'])
    sorted_results_cost = sorted(results, key=lambda r: r['cost'])
    print("*Best values by AoSI:")
    for r in sorted_results_AoSI[:10]:
        print(f"a, b, c = {r['a']:.1f}, {r['b']}, {r['c']} → mean={r['mean']:.3f}, most_common={r['most_common']}, cost = {r['cost']}")
    print("\n*Best values by Cost:")
    for r in sorted_results_cost[:10]:
        print(f"a, b, c = {r['a']:.1f}, {r['b']}, {r['c']} → mean={r['mean']:.3f}, most_common={r['most_common']}, cost = {r['cost']}")

    return sorted_results_AoSI, sorted_results_cost





#DONE try different variations, including a nested loop part where other values are tries
# stuff like a * x ** b + h * c with a,b,c as variable loops and h as a constant
def V(x, a=0.5, b=2, c=0, h=1):
    try:
        # Force float conversion to avoid issues like string inputs
        c = 0
        x = float(x)
        a = float(a)
        b = float(b)
        c = float(c)
        h = float(h)
        return a * x ** b + h * c
    except (TypeError, ValueError, OverflowError, ZeroDivisionError) as e:
        print(f"Error in V() = {a} * x ** {b} + {h} * {c}")
        return float('nan')  # or return None if you prefer


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

def plot_state_distribution(S_states, name):
    mean_val = sum(S_states) / len(S_states)

    plt.figure(figsize=(10, 5))
    sns.histplot(S_states, kde=False, bins=range(min(S_states), max(S_states) + 2),
                 color='skyblue', edgecolor='black')

    # Add average line
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.2f}')

    # Plot styling
    plt.title(f"Distribution of Age of System Instability (S(t)) for {name}", fontsize=14)
    plt.xlabel("S(t) Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.6)
    plt.legend()
    plt.tight_layout()
    plt.show()

def stabilization_time(S_states, threshold=1, window=20):
    for t in range(window, len(S_states)):
        if np.mean(S_states[t-window:t]) < threshold:
            return t
    return len(S_states)

ap = r1                          #unstable & IDLE
bp = r1 * (1 - rho * p)          #unstable & Compressed
cp = r1 * (1 - rho * q)          #unstable & Uncompressed
dp = 1 - r0                      #stable & IDLE
ep = (1 - r0) * (1 - rho * p)    #stable & Compressed
fp = (1 - r0) * ((1 - rho) + rho * (1 - q) )      #stable &  Uncompressed  this is not the same as the paper, since we also have the propability that we get something

# Function to compute stationary distribution u_n(i)
def compute_un_i(i, n1, n2, un0):
    if n1 > 0 and n2 > 0:
        if i == 0:
            return un0
        elif 1 <= i <= n1:
            return dp * ap ** (i - 1) * un0
        elif n1 < i <= n2:
            return dp * ap ** (n1 - 1) * bp ** (i - n1) * un0
        else:
            return dp * ap ** (n1 - 1) * bp ** (n2 - n1) * cp ** (i - n2) * un0
    elif n1 == 0 and n2 > 0:
        if i == 0:
            return un0
        elif 1 <= i <= n2:
            return ep * bp ** (i - 1) * un0
        else:
            return ep * bp ** (n2 - 1) * c ** (i - n2) * un0
    elif n1 == n2 == 0:
        if i == 0:
            return un0
        else:
            return fp * cp ** (i - 1) * un0
    else:
        raise ValueError("Invalid threshold combination")


########################## compute u_n(0)
def compute_un0(n1, n2):
    if n1 > 0 and n2 > 0:
        num = (1 - ap) * (1 - bp) * (1 - cp)
        den = ((1 - ap + dp) * (1 - cp) * (1 - bp) +
                       (1 - cp) * (bp - ap) * dp * ap ** (n1 - 1) +
                       dp * (1 - ap) * (cp - bp) * bp ** (n2 - n1) * ap ** (n1 - 1))
        return num / den
    elif n1 == 0 and n2 > 0:
        num = (1 - bp) * (1 - cp)
        den = (1 - bp + ep) * (1 - cp) + (cp - bp) * ep * bp ** (n2 - 1)
        return num / den
    elif n1 == n2 == 0:
        return (1 - cp) / (1 - cp + fp)
    else:
        raise ValueError("Invalid threshold combination")


############### computing sn, once numeric (until MAX_I) and once with the case form from the paper
def numeric_sn(n1,n2, un0):

    # Compute un(i) for i in range up to MAX_I
    un = np.array([compute_un_i(i, n1, n2, un0) for i in range(MAX_I)])

    # Cost: sn = sum(i * un(i))
    sn = sum(i * un[i] for i in range(MAX_I))

    return sn


def sn_like_the_paper(n1,n2, un0):
    if n1 > 0 and n2 > 0:
        term1 = un0*((dp*(ap**(n1+1)*n1 - n1*ap**(n1) - ap**(n1) + 1)) / ((1-ap)**2))
        term2 = un0*( (dp*bp*ap**(n1-1) * (bp**(n2-n1) * (bp*n2 - n2 - 1) - bp*n1 + n1 + 1)) / ((1-bp)**2) )
        term3 = un0*( (dp*cp*bp**(n2-n1) * ap**(n1-1) * (-cp*n2 + n2 +1)) / ((1-cp)**2) )
        return term1 + term2 + term3
    elif n1 == 0 and n2 > 0:
        term1 = un0 * (ep * (bp**(n2 + 1) * n2 - n2 * bp**n2 - bp**n2 + 1)) / ((1 - bp) ** 2)
        term2 = un0*(ep*cp*bp**(n2-1) * (-cp*n2 + n2 + 1)) / ((1-cp)**2)
        return term1 + term2
    elif n1 == n2 == 0:
        return un0 * fp / ((1-cp)**2)
    else:
        raise ValueError("Invalid threshold combination")


######################### computing the delta values
def numeric_delta1(n1,n2, un0):
    un = np.array([compute_un_i(i, n1, n2, un0) for i in range(n1, n2)])
    delta1 = sum(un)
    return delta1

def numeric_delta2(n1,n2, un0):
    un = np.array([compute_un_i(i, n1, n2, un0) for i in range(n2, n2+MAX_I)])
    delta1 = sum(un)
    return delta1

def delta1_like_the_paper(n1,n2, un0):
    if n1 > 0 and n2 > 0:
        return un0*dp*ap**(n1 - 1) * ((1 - bp ** (n2 - n1)) / (1 - bp))
    elif n1 == 0 and n2 > 0:
        return un0 * (1 + ep * ((1 - bp ** (n2 - 1)) / (1 - bp)))
    elif n1 == n2 == 0:
        return 0
    else:
        raise ValueError("Invalid threshold combination")


def delta2_like_the_paper(n1,n2, un0):
    if n1 > 0 and n2 > 0:
        return un0 * dp * ap ** (n1-1) * bp**(n2-n1) / (1-cp)
    elif n1 == 0 and n2 > 0:
        return un0 * ep * bp **(n2-1) / (1-cp)
    elif n1 == n2 == 0:
        return 1
    else:
        raise ValueError("Invalid threshold combination")



##################################   F(n1,n2)
def F_function(n1, n2, un0):

    if(numeric_calculation):
        sn = numeric_sn(n1,n2,un0)
        delta1 = numeric_delta1(n1,n2,un0)
        delta2 = numeric_delta2(n1,n2,un0)
    else:
        sn = sn_like_the_paper(n1,n2,un0)
        delta1 = delta1_like_the_paper(n1,n2,un0)
        delta2 = delta2_like_the_paper(n1,n2,un0)


    # Total cost
    return sn + lambda1 * delta1 + lambda2 * delta2


############################### finding the n1 and n2 values. lambdas have to be given for these since we need either the lambda or the n values
def find_optimal_thresholds(lambda1_val, lambda2_val, max_n1=100, max_n2=200):
    global lambda1, lambda2
    lambda1 = lambda1_val
    lambda2 = lambda2_val

    best_cost = float('inf')
    best_n1, best_n2 = 0, 0

    #min_ {n1,n2} F(n1,n2)
    for n1 in range(max_n1):
        for n2 in range(n1, max_n2):  # must ensure n2 ≥ n1 or it will rise an error
            un0 = compute_un0(n1,n2)
            cost = F_function(n1, n2, un0)
            if cost < best_cost:
                best_cost = cost
                best_n1, best_n2 = n1, n2

    return best_n1, best_n2, best_cost


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
    if rng.random()  < prob_increment:
        return st + 1
    else:
        return 0



def simulate_paper(n):
    n1, n2, cost = find_optimal_thresholds(lambda1, lambda2, 50, 50)
    st = 0
    S_history = [0]  # Initialize history with the starting state
    action_history = []

    for i in range(n):
        next = 0
        if st <n1 and st < n2:
            action = 0
        elif st >= n1 and st < n2:
            action = 1
        else:
            action = 2

        action_history.append(action)

        # 2. Calculate the next state S(t+1) based on the chosen action
        probs = get_transition_probs(st)
        next_S = stable_unstable(st, action, probs)

        S_history.append(next_S)
        st = next_S  # Update current_S for the next iteration

    return S_history, action_history


def sim_paper(num_steps):
    reset_random()
    n = 1
    queue0.append(1 - r0)
    queue1.append((1 - r0) * (1 - rho * p))
    queue2.append((1 - r0) * (1 - rho * q))
    avgs = np.zeros_like(range(0, n), dtype=float)
    for i in range(n):
        S_states, actions = simulate_paper(num_steps)

        avgs[i] = (sum(S_states) / len(S_states))

    num_instable = 0

    for i in S_states:
        if not i == 0:
            num_instable = num_instable + i

    idle_actions = actions.count(0)
    sparse_actions = actions.count(1)
    dense_actions = actions.count(2)
    stab_time = stabilization_time(S_states)
    cost = sparse_actions * lambda1 + dense_actions * lambda2 + num_instable
    true_cost = sparse_actions * lambda1 + dense_actions * lambda2
    AoSI = avgs.mean()

    if show_all_plots:
        print("\n--- Simulation Summary Paper Function ---")
        print(f"Total simulation steps: {num_steps}")
        print(f"Average Age Of System Instability (ADSI): {avgs[-1]:.4f}")
        print(f"Average Age Of System Instability (ADSI) over {n}: {AoSI:.4f}")
        print(f"Breakdown of actions taken:")
        print(f"  Idle (0): {idle_actions} ({idle_actions / num_steps:.2%})")
        print(f"  Sparse Update (1): {sparse_actions} ({sparse_actions / num_steps:.2%})")
        print(f"  Dense Update (2): {dense_actions} ({dense_actions / num_steps:.2%})")
        print(
            f"Cost: {cost}")  # DONE it should also include the cost of it beeing in a bad state, not just communicvation
        print(f"Bellman policy (G): stabilizes at t = {stab_time}")

        # DONE for lyapunov, in addition to this, also calculate the cost (st + ld1 + ld2) with st = V(st)
        # --- Visualize S(t) over Time ---

        try:
            action_colors = {
                0: 'green',  # Idle
                1: 'orange',  # Compressed
                2: 'red'  # Uncompressed
            }

            # Build segments for LineCollection
            points = np.array([range(len(S_states)), S_states]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Map each segment to its corresponding action
            colors = [action_colors[act] for act in actions[:-1]]

            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 7))

            # Add colored lines per segment
            lc = LineCollection(segments, colors=colors, linewidth=2.5)
            ax.add_collection(lc)

            # Overlay the full S(t) curve as a thin line for readability
            ax.plot(range(len(S_states)), S_states, color='black', linewidth=0.5, label='S(t) Trace')

            # Add horizontal baseline
            ax.axhline(y=0, color='red', linestyle=':', linewidth=1, label='S(t) = 0 (Good State)')

            # Styling
            ax.set_title(f'Simulated Age of System Instability (S(t)) according to the paper Simulation',
                         fontsize=16)
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('S(t) Value', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            custom_lines = [
                Line2D([0], [0], color='green', lw=2, label='Idle (0)'),
                Line2D([0], [0], color='red', lw=2, label='Compressed (1)'),
                Line2D([0], [0], color='orange', lw=2, label='Uncompressed (2)')
            ]

            # Add to the existing legend
            ax.legend(handles=[*custom_lines,
                               Line2D([0], [0], color='black', lw=0.5, label='S(t) Trace'),
                               Line2D([0], [0], color='red', lw=1, linestyle=':', label='S(t) = 0 (Good State)')],
                      fontsize=10)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError plotting with Matplotlib: {e}")
            print("Please ensure matplotlib is installed (`pip install matplotlib`) if you want to see the plot.")

        plot_state_distribution(S_states, "Paper Function")


    return AoSI, cost, true_cost, S_states


def find_most_common(arr):
    values, counts = np.unique(arr, return_counts=True)
    most_common = values[np.argmax(counts)]

    return most_common



#G-Function
def sim_GFun(num_steps):
    reset_random()
    n = 1
    avgs = np.zeros_like(range(0, n), dtype=float)
    for i in range(n):
        S_states, actions = run_sim_1(num_steps)

        avgs[i] = (sum(S_states) / len(S_states))

    num_instable = 0

    for i in S_states:
        if not i == 0:
            num_instable = num_instable + i

    idle_actions = actions.count(0)
    sparse_actions = actions.count(1)
    dense_actions = actions.count(2)
    stab_time = stabilization_time(S_states)
    cost = sparse_actions * lambda1 + dense_actions * lambda2 + num_instable
    true_cost = sparse_actions * lambda1 + dense_actions * lambda2
    AoSI = avgs.mean()
    if show_all_plots:
        print("\n\n--- Simulation Summary G Function---")
        print(f"Total simulation steps: {num_steps}")
        print(f"Average Age Of System Instability (ADSI): {avgs[-1]:.4f}")
        print(f"Average Age Of System Instability (ADSI) over {n}: {AoSI:.4f}")
        print(f"Breakdown of actions taken:")
        print(f"  Idle (0): {idle_actions} ({idle_actions / num_steps:.2%})")
        print(f"  Sparse Update (1): {sparse_actions} ({sparse_actions / num_steps:.2%})")
        print(f"  Dense Update (2): {dense_actions} ({dense_actions / num_steps:.2%})")
        print(
            f"Cost: {cost}")  # DONE it should also include the cost of it beeing in a bad state, not just communicvation
        print(f"Bellman policy (G): stabilizes at t = {stab_time}")

        # DONE for lyapunov, in addition to this, also calculate the cost (st + ld1 + ld2) with st = V(st)
        # --- Visualize S(t) over Time ---

        try:
            action_colors = {
                0: 'green',  # Idle
                1: 'orange',  # Compressed
                2: 'red'  # Uncompressed
            }

            # Build segments for LineCollection
            points = np.array([range(len(S_states)), S_states]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Map each segment to its corresponding action
            colors = [action_colors[act] for act in actions[:-1]]

            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 7))

            # Add colored lines per segment
            lc = LineCollection(segments, colors=colors, linewidth=2.5)
            ax.add_collection(lc)

            # Overlay the full S(t) curve as a thin line for readability
            ax.plot(range(len(S_states)), S_states, color='black', linewidth=0.5, label='S(t) Trace')

            # Add horizontal baseline
            ax.axhline(y=0, color='red', linestyle=':', linewidth=1, label='S(t) = 0 (Good State)')

            # Styling
            ax.set_title(f'Simulated Age of System Instability (S(t)) Over Time according to the G-Function',
                         fontsize=16)
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('S(t) Value', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            custom_lines = [
                Line2D([0], [0], color='green', lw=2, label='Idle (0)'),
                Line2D([0], [0], color='red', lw=2, label='Compressed (1)'),
                Line2D([0], [0], color='orange', lw=2, label='Uncompressed (2)')
            ]

            # Add to the existing legend
            ax.legend(handles=[*custom_lines,
                               Line2D([0], [0], color='black', lw=0.5, label='S(t) Trace'),
                               Line2D([0], [0], color='red', lw=1, linestyle=':', label='S(t) = 0 (Good State)')],
                      fontsize=10)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError plotting with Matplotlib: {e}")
            print("Please ensure matplotlib is installed (`pip install matplotlib`) if you want to see the plot.")

        plot_state_distribution(S_states, "G-Function")

    return AoSI, cost, true_cost, S_states


#G-Function
def sim_FFun(num_steps):
    reset_random()
    n = 1
    avgs = np.zeros_like(range(0, n), dtype=float)
    for i in range(n):
        S_states, actions = run_sim_2(num_steps)

        avgs[i] = (sum(S_states) / len(S_states))

    num_instable = 0

    for i in S_states:
        if not i == 0:
            num_instable = num_instable + i

    idle_actions = actions.count(0)
    sparse_actions = actions.count(1)
    dense_actions = actions.count(2)
    stab_time = stabilization_time(S_states)
    cost = sparse_actions * lambda1 + dense_actions * lambda2 + num_instable
    true_cost = sparse_actions * lambda1 + dense_actions * lambda2
    AoSI = avgs.mean()
    if show_all_plots:
        print("\n\n--- Simulation Summary F Function---")
        print(f"Total simulation steps: {num_steps}")
        print(f"Average Age Of System Instability (ADSI): {avgs[-1]:.4f}")
        print(f"Average Age Of System Instability (ADSI) over {n}: {AoSI:.4f}")
        print(f"Breakdown of actions taken:")
        print(f"  Idle (0): {idle_actions} ({idle_actions / num_steps:.2%})")
        print(f"  Sparse Update (1): {sparse_actions} ({sparse_actions / num_steps:.2%})")
        print(f"  Dense Update (2): {dense_actions} ({dense_actions / num_steps:.2%})")
        print(
            f"Cost: {cost}")  # DONE it should also include the cost of it beeing in a bad state, not just communicvation
        print(f"Bellman policy (G): stabilizes at t = {stab_time}")

        # DONE for lyapunov, in addition to this, also calculate the cost (st + ld1 + ld2) with st = V(st)
        # --- Visualize S(t) over Time ---

        try:
            action_colors = {
                0: 'green',  # Idle
                1: 'orange',  # Compressed
                2: 'red'  # Uncompressed
            }

            # Build segments for LineCollection
            points = np.array([range(len(S_states)), S_states]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Map each segment to its corresponding action
            colors = [action_colors[act] for act in actions[:-1]]

            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 7))

            # Add colored lines per segment
            lc = LineCollection(segments, colors=colors, linewidth=2.5)
            ax.add_collection(lc)

            # Overlay the full S(t) curve as a thin line for readability
            ax.plot(range(len(S_states)), S_states, color='black', linewidth=0.5, label='S(t) Trace')

            # Add horizontal baseline
            ax.axhline(y=0, color='red', linestyle=':', linewidth=1, label='S(t) = 0 (Good State)')

            # Styling
            ax.set_title(f'Simulated Age of System Instability (S(t)) Over Time according to the F-Function',
                         fontsize=16)
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('S(t) Value', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            custom_lines = [
                Line2D([0], [0], color='green', lw=2, label='Idle (0)'),
                Line2D([0], [0], color='red', lw=2, label='Compressed (1)'),
                Line2D([0], [0], color='orange', lw=2, label='Uncompressed (2)')
            ]

            # Add to the existing legend
            ax.legend(handles=[*custom_lines,
                               Line2D([0], [0], color='black', lw=0.5, label='S(t) Trace'),
                               Line2D([0], [0], color='red', lw=1, linestyle=':', label='S(t) = 0 (Good State)')],
                      fontsize=10)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError plotting with Matplotlib: {e}")
            print("Please ensure matplotlib is installed (`pip install matplotlib`) if you want to see the plot.")

        plot_state_distribution(S_states, "F-Function")

    return AoSI, cost, true_cost,S_states


def sim_lya(num_steps,a,b,c):
    reset_random()
    n = 1
    avgs = np.zeros_like(range(0, n), dtype=float)
    for i in range(n):
        S_states, actions = run_sim_3(num_steps,a,b,c)

        avgs[i] = (sum(S_states) / len(S_states))

    num_instable = 0

    for i in S_states:
        if not i == 0:
            num_instable = num_instable + i

    idle_actions = actions.count(0)
    sparse_actions = actions.count(1)
    dense_actions = actions.count(2)
    stab_time = stabilization_time(S_states)
    cost = sparse_actions * lambda1 + dense_actions * lambda2 + num_instable
    true_cost = sparse_actions * lambda1 + dense_actions * lambda2
    AoSI = avgs.mean()
    if show_all_plots:
        print("\n\n--- Simulation Summary Lyapunov Function---")
        print(f"Total simulation steps: {num_steps}")
        print(f"Average Age Of System Instability (ADSI): {avgs[-1]:.4f}")
        print(f"Average Age Of System Instability (ADSI) over {n}: {AoSI:.4f}")
        print(f"Breakdown of actions taken:")
        print(f"  Idle (0): {idle_actions} ({idle_actions / num_steps:.2%})")
        print(f"  Sparse Update (1): {sparse_actions} ({sparse_actions / num_steps:.2%})")
        print(f"  Dense Update (2): {dense_actions} ({dense_actions / num_steps:.2%})")
        print(
            f"Cost: {cost}")  # DONE it should also include the cost of it beeing in a bad state, not just communicvation
        print(f"Bellman policy (G): stabilizes at t = {stab_time}")

        # DONE for lyapunov, in addition to this, also calculate the cost (st + ld1 + ld2) with st = V(st)
        # --- Visualize S(t) over Time ---

        try:
            action_colors = {
                0: 'green',  # Idle
                1: 'orange',  # Compressed
                2: 'red'  # Uncompressed
            }

            # Build segments for LineCollection
            points = np.array([range(len(S_states)), S_states]).T.reshape(-1, 1, 2)
            segments = np.concatenate([points[:-1], points[1:]], axis=1)

            # Map each segment to its corresponding action
            colors = [action_colors[act] for act in actions[:-1]]

            # Create the plot
            fig, ax = plt.subplots(figsize=(14, 7))

            # Add colored lines per segment
            lc = LineCollection(segments, colors=colors, linewidth=2.5)
            ax.add_collection(lc)

            # Overlay the full S(t) curve as a thin line for readability
            ax.plot(range(len(S_states)), S_states, color='black', linewidth=0.5, label='S(t) Trace')

            # Add horizontal baseline
            ax.axhline(y=0, color='red', linestyle=':', linewidth=1, label='S(t) = 0 (Good State)')

            # Styling
            ax.set_title(f'Simulated Age of System Instability (S(t)) Over Time with V(s(t)) = {a}x^{b} + x*{c} according to the Lyapunov Function',
                         fontsize=16)
            ax.set_xlabel('Time Step', fontsize=12)
            ax.set_ylabel('S(t) Value', fontsize=12)
            ax.grid(True, linestyle='--', alpha=0.7)
            ax.legend(fontsize=10)
            custom_lines = [
                Line2D([0], [0], color='green', lw=2, label='Idle (0)'),
                Line2D([0], [0], color='red', lw=2, label='Compressed (1)'),
                Line2D([0], [0], color='orange', lw=2, label='Uncompressed (2)')
            ]

            # Add to the existing legend
            ax.legend(handles=[*custom_lines,
                               Line2D([0], [0], color='black', lw=0.5, label='S(t) Trace'),
                               Line2D([0], [0], color='red', lw=1, linestyle=':', label='S(t) = 0 (Good State)')],
                      fontsize=10)
            plt.tight_layout()
            plt.show()
        except Exception as e:
            print(f"\nError plotting with Matplotlib: {e}")
            print("Please ensure matplotlib is installed (`pip install matplotlib`) if you want to see the plot.")

        plot_state_distribution(S_states, f"Lyapunov Function with V(s(t)) = {a}x^{b} + x*{c}")

    return AoSI, cost, true_cost, S_states

def format_sigfigs(val):
    if val == 0:
        return "0"
    elif val >= 100:
        return str(round(val))  # just round to int
    elif val >= 10:
        return str(round(val, 1))  # one decimal place
    elif val >= 1:
        return str(round(val, 2))  # up to two decimal places
    else:
        # For values < 1, show up to 2 significant digits without scientific notation
        return f"{val:.2f}".rstrip("0").rstrip(".")


def worker(params):
    a, b, c, num_steps, lambda1, lambda2 = params
    st, actions = run_sim_3(num_steps, a, b, c)
    actions = np.array(actions)
    st = np.array(st)

    num_comp = np.count_nonzero(actions == 1)
    num_uncomp = np.count_nonzero(actions == 2)

    cost_total = np.sum(st[st != 0])
    cost = lambda1 * num_comp + lambda2 * num_uncomp + cost_total

    mean_val = st.mean()
    most_common_val = find_most_common(st)

    return {
        'a': a,
        'b': b,
        'c': c,
        'mean': mean_val,
        'most_common': most_common_val,
        'cost': cost,
        'msg cost': lambda1 * num_comp + lambda2 * num_uncomp
    }

def find_optimal_V(num_steps, cpu_fraction=0.8):
    param_grid = list(itertools.product(
        np.arange(0.1, 20, 0.1),
        np.arange(0.1, 10, 0.1),
        [0]
    ))
    total = len(param_grid)
    max_procs = max(1, math.floor(os.cpu_count() * cpu_fraction))
    results = []

    # Attach extra args for worker
    params_with_args = [(a, b, c, num_steps, lambda1, lambda2) for (a, b, c) in param_grid]

    with ProcessPoolExecutor(max_workers=max_procs) as executor:
        futures = {executor.submit(worker, p): p for p in params_with_args}
        for future in tqdm(as_completed(futures), total=total, desc="Parameter sweep", unit="combos"):
            results.append(future.result())

    sorted_results_AoSI = sorted(results, key=lambda r: r['mean'])
    sorted_results_cost = sorted(results, key=lambda r: r['cost'])
    sorted_msg_cost = sorted(results, key=lambda r: r['msg cost'])
    battery = 0
    if not battery == 0:
        sorted_results_AoSI = [item for item in sorted_results_AoSI if item['msg cost'] < battery]
        sorted_results_cost = [item for item in sorted_results_cost if item['msg cost'] < battery]
        sorted_msg_cost = [item for item in sorted_msg_cost if item['msg cost'] < battery]

    print("#* Best values by AoSI:")
    for r in sorted_results_AoSI[:10]:
        print(f"# a, b, c = {r['a']:.1f}, {r['b']}, {r['c']} "
              f"# → mean={r['mean']:.3f}, most_common={r['most_common']}, cost={r['cost']}")

    print("\n#* Best values by Cost:")
    for r in sorted_results_cost[:10]:
        print(f"# a, b, c = {r['a']:.1f}, {r['b']}, {r['c']} "
              f"# → mean={r['mean']:.3f}, most_common={r['most_common']}, cost={r['cost']}")

    print("\n#* Top 10 for message cost (mean and cost):")
    for r in sorted_msg_cost[:10]:
        print(f"# a, b, c = {r['a']:.1f}, {r['b']}, {r['c']} "
              f"# → mean={r['mean']:.3f}, cost={r['cost']}")

    return (
        sorted_results_AoSI[0]['a'], sorted_results_AoSI[0]['b'], sorted_results_AoSI[0]['c'],
        sorted_results_cost[0]['a'], sorted_results_cost[0]['b'], sorted_results_cost[0]['c'],
        sorted_msg_cost[0]['a'], sorted_msg_cost[0]['b'], sorted_msg_cost[0]['c']
    )

def plot_state_comparison(states_list, labels, name="Comparison"):
    assert len(states_list) == len(labels), "Each dataset must have a label."

    num_datasets = len(states_list)
    colors = ['skyblue', 'salmon', 'lightgreen', 'plum', 'orange', 'lightgrey']
    assert num_datasets <= len(colors), "Too many datasets for default colors."

    # Determine common bin range
    min_val = min(map(min, states_list))
    max_val = max(map(max, states_list))
    bins = range(min_val, max_val + 2)
    bin_centers = np.array(bins[:-1]) + 0.5

    # Compute histogram counts
    counts = [np.histogram(data, bins=bins)[0] for data in states_list]

    # Adjust bar width and offsets
    total_width = 0.8  # total width occupied per bin
    bar_width = total_width / num_datasets
    offsets = np.linspace(-total_width / 2 + bar_width / 2, total_width / 2 - bar_width / 2, num_datasets)

    # Plot
    plt.figure(figsize=(14, 6))
    for i, (hist, label, color) in enumerate(zip(counts, labels, colors)):
        plt.bar(bin_centers + offsets[i], hist, width=bar_width, label=label, color=color, edgecolor='black')

        # Mean line
        mean_val = sum(states_list[i]) / len(states_list[i])
        plt.axvline(mean_val, linestyle='--', color=color, linewidth=1.5, label=f'{label} Mean = {mean_val:.2f}')

    plt.title(f"Comparison of State Distributions: {name}", fontsize=16)
    plt.xlabel("S(t) Value", fontsize=12)
    plt.ylabel("Frequency", fontsize=12)
    plt.grid(True, linestyle='--', alpha=0.5)
    plt.legend(loc='upper right', bbox_to_anchor=(1.25, 1.0))
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    if len(sys.argv) == 1:
        num_steps = 10000
    else:
        num_steps = int(sys.argv[1])
        if len(sys.argv) == 3:
            RANDOM_SEED = int(sys.argv[2])
    if should_save:
        save_folder_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        SAVE_DIR = f'figures/{save_folder_timestamp}_timesteps_{num_steps}'
        os.makedirs(SAVE_DIR, exist_ok=True)

    ############################################################################################################################################
    ############################################################################################################################################
    compute_new_V = False  #If you set this to true, it will take a long time to compile!!!! Be warned
    ############################################################################################################################################
    ############################################################################################################################################

    if compute_new_V:
        a1,b1,c1,a2,b2,c2,a3,b3,c3 = find_optimal_V(num_steps)

    start_p = time.perf_counter()
    AoSI1, cost1, true_cost1, states1 = sim_paper(num_steps)
    end_p = time.perf_counter()
    p_duration = start_p - end_p
    print(f"paper simulation done. Time: {p_duration:.6f} sec")
    start_g = time.perf_counter()
    AoSI2, cost2, true_cost2, states2 = sim_GFun(num_steps)
    end_g = time.perf_counter()
    g_duration = start_g - end_g
    print(f"g function simulation done. Time: {g_duration:.6f} sec")
    #AoSI3, cost3, true_cost3, idle_actions3, sparse_actions3, dense_actions3 = sim_FFun(num_steps)
    avg_cost1 = cost1/num_steps
    avg_msg_cost1 = true_cost1/num_steps
    cost_through_aosi1 = cost1 - true_cost1
    avg_cost2 = cost2 / num_steps
    avg_msg_cost2 = true_cost2 / num_steps
    cost_through_aosi2 = cost2 - true_cost2
################################### Replace the V values here (As many as you like)
    a,b,c = 0.5, 2, 0
    # * Best values by AoSI:
    # a, b, c = 18.8, 2.0, 0 # → mean=0.340, most_common=0, cost=633989
    # a, b, c = 17.5, 5.2, 0 # → mean=0.341, most_common=0, cost=634100
    # a, b, c = 19.9, 4.6, 0 # → mean=0.341, most_common=0, cost=634108
    # a, b, c = 19.2, 6.7, 0 # → mean=0.341, most_common=0, cost=634118
    # a, b, c = 18.7, 9.6, 0 # → mean=0.341, most_common=0, cost=634136
    # a, b, c = 19.5, 0.9, 0 # → mean=0.341, most_common=0, cost=634138
    # a, b, c = 16.9, 7.6, 0 # → mean=0.341, most_common=0, cost=634147
    # a, b, c = 17.9, 6.2, 0 # → mean=0.341, most_common=0, cost=634149
    # a, b, c = 17.9, 1.0, 0 # → mean=0.342, most_common=0, cost=634159
    # a, b, c = 17.3, 3.2, 0 # → mean=0.342, most_common=0, cost=634162

    # * Best values by Cost:
    # a, b, c = 1.2, 0.9, 0 # → mean=1.428, most_common=0, cost=186754
    # a, b, c = 1.6, 0.8, 0 # → mean=1.435, most_common=0, cost=187749
    # a, b, c = 1.8, 0.7000000000000001, 0 # → mean=1.436, most_common=0, cost=187969
    # a, b, c = 4.1, 0.2, 0 # → mean=1.439, most_common=0, cost=188118
    # a, b, c = 2.5, 0.6, 0 # → mean=1.438, most_common=0, cost=188177
    # a, b, c = 0.7, 1.4000000000000001, 0 # → mean=1.419, most_common=0, cost=188240
    # a, b, c = 3.4, 0.2, 0 # → mean=1.442, most_common=0, cost=188376
    # a, b, c = 1.1, 1.1, 0 # → mean=1.431, most_common=0, cost=188401
    # a, b, c = 1.1, 1.2000000000000002, 0 # → mean=1.421, most_common=0, cost=188414
    # a, b, c = 2.0, 0.7000000000000001, 0 # → mean=1.439, most_common=0, cost=188499

    # * Top 10 for message cost (mean and cost):
    # a, b, c = 0.1, 0.2, 0 # → mean=2.068, cost=213369
    # a, b, c = 0.1, 0.1, 0 # → mean=2.081, cost=214777
    # a, b, c = 0.2, 0.30000000000000004, 0 # → mean=1.985, cost=207546
    # a, b, c = 0.3, 0.5, 0 # → mean=1.978, cost=206899
    # a, b, c = 0.2, 0.5, 0 # → mean=1.981, cost=207269
    # a, b, c = 0.1, 0.4, 0 # → mean=1.984, cost=207643
    # a, b, c = 0.2, 0.2, 0 # → mean=1.983, cost=207531
    # a, b, c = 0.1, 0.8, 0 # → mean=1.988, cost=208104
    # a, b, c = 0.2, 0.7000000000000001, 0 # → mean=1.986, cost=207912
    # a, b, c = 0.4, 0.1, 0 # → mean=1.984, cost=207726

    start_l = time.perf_counter()
    AoSI4, cost4, true_cost4, states4 = sim_lya(num_steps,rho * p,1,0)
    end_l = time.perf_counter()
    l_duration = start_l - end_l
    print(f"lyapunov calculation with 1/2x^2 done. Time: {l_duration:.6f} sec")
    avg_cost4 = cost4 / num_steps
    avg_msg_cost4 = true_cost4 / num_steps
    cost_through_aosi4 = cost4 - true_cost4
    # Actual data (replace these with your real values)
    paper = [AoSI1, cost1,true_cost1, cost_through_aosi1, avg_cost1, avg_msg_cost1]
    g_func = [AoSI2, cost2,true_cost2, cost_through_aosi2, avg_cost2, avg_msg_cost2]
    #f_func = [AoSI3, cost3,true_cost3, cost_through_aosi3, avg_cost3, avg_msg_cost3]
    lyapunov = [AoSI4, cost4,true_cost4, cost_through_aosi4, avg_cost4, avg_msg_cost4]

    labels = ['AoSI', 'Cost','Message Cost', 'Cost through AoSI', 'Average Cost', 'Average Message Cost']
    methods = ['Paper', 'G-Func', 'Lyapunov']
    colors = ['#4C72B0', '#55A868', '#8172B3']

    data = np.array([paper, g_func, lyapunov])

    # Normalize each column (metric) to [0, 1] for comparability
    normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-9)

    x = np.arange(len(labels))  # group positions
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(3):
        bars = ax.bar(x + (i - 1.5) * width, normalized[i], width, label=methods[i], color=colors[i])

        for j, bar in enumerate(bars):
            val = data[i][j]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Metric Comparison (Normalized Per Metric)')
    ax.legend()
    ax.set_ylim(0, np.max(data)/num_steps+0.5)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if should_save:
        filename = os.path.join(SAVE_DIR, f"Metric Comparison (Normalized Per Metric) most important; {num_steps} steps.png")
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.close()

    if show_all_plots:
        ######### INDIVIDUAL PLOTS PER METRIC
        for i, metric in enumerate(labels):
            values = [paper[i], g_func[i], lyapunov[i]]

            fig, ax = plt.subplots(figsize=(6, 4))
            bars = ax.bar(methods, values, color=colors)

            for j, bar in enumerate(bars):
                ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                        format_sigfigs(values[j]), ha='center', va='bottom', fontsize=8,rotation=30)

            ax.set_ylabel(metric)
            ax.set_title(f'{metric} Comparison. V(s) = {a:.1f} x ^ {b:.1f} + x * {c:.1f}')
            ax.grid(axis='y', linestyle='--', alpha=0.5)
            plt.tight_layout()
            plt.show()

    #plot_state_comparison([states1,states2,states4],  labels=["Paper", "G-Function", "Lyapunov Function"], name = "Paper, G-func, Lyapunov")
    print("Comparing 4 different values")

    AoSIp, costp,true_costp = AoSI1, cost1,true_cost1
    AoSIg, costg, true_costg = AoSI2, cost2,true_cost2
    avg_costg = costg / num_steps
    avg_msg_costg = true_costg / num_steps
    cost_through_aosig = costg - true_costg
    avg_costp = costp / num_steps
    avg_msg_costp = true_costp / num_steps
    cost_through_aosip = costp - true_costp

    a, b, c = 0.5, 2, 0
    fun1 = f'{a:.1f}x^{b}'
    AoSI1, cost1, true_cost1, states4 = sim_lya(num_steps, a, b, c)
    print(f"optimisation for {fun1} done")
    avg_cost1 = cost1 / num_steps
    avg_msg_cost1 = true_cost1 / num_steps
    cost_through_aosi1 = cost1 - true_cost1

    a, b, c = 1, 1, 0
    fun0 = f'x'
    AoSI0, cost0, true_cost0, states0 = sim_lya(num_steps, a, b, c)
    print(f"optimisation for V(x)=x done")
    avg_cost0 = cost0 / num_steps
    avg_msg_cost0 = true_cost0 / num_steps
    cost_through_aosi0 = cost0 - true_cost0


    ################################### Replace the V values here (Minimal AoSI)
    a, b, c = 18.8, 0.6, 0 # → mean=0.318, most_common=0, cost=63177

    if compute_new_V:
        a = a1
        b = b1
        c = c1
    fun2 = f'{a:.1f}x^{b:.1f}'
    #a, b, c = 1, 1, 0
    AoSI2, cost2, true_cost2, states5 = sim_lya(num_steps, a, b, c)
    print(f"optimisation for {fun2} done")
    avg_cost2 = cost2 / num_steps
    avg_msg_cost2 = true_cost2 / num_steps
    cost_through_aosi2 = cost2 - true_cost2
    ################################### Replace the V values here (Minimal Cost)
    savedynamic = True
    a, b, c = 2.2, 1.0, 0 # → mean=1.400, most_common=0, cost=18303

    if compute_new_V:
        a = a2
        b = b2
        c = c2
    fun3 = f'{a:.1f}x^{b:.1f}'
    AoSI3, cost3, true_cost3, states6 = sim_lya(num_steps, a, b, c)
    print(f"optimisation for {fun3} done")
    avg_cost3 = cost3 / num_steps
    avg_msg_cost3 = true_cost3 / num_steps
    cost_through_aosi3 = cost3 - true_cost3

    savedynamic = False
    ################################### Replace the V values here (Balanced)
    a, b, c = 0.1, 0.1, 0 # → mean=2.423, cost=24229

    if compute_new_V:
        a = a3
        b = b3
        c = c3
    fun4 = f'{a:.1f}x^{b:.1f}'
    AoSI4, cost4, true_cost4, states7 = sim_lya(num_steps, a, b, c)
    print(f"optimisation for {fun4} done")
    avg_cost4 = cost4 / num_steps
    avg_msg_cost4 = true_cost4 / num_steps
    cost_through_aosi4 = cost4 - true_cost4
    #stand = [AoSI1, cost1,true_cost1, cost_through_aosi1, avg_cost1, avg_msg_cost1]
    stand = [AoSI1, avg_cost1, avg_msg_cost1]
    linear = [AoSI0, avg_cost0, avg_msg_cost0]
    minaosi = [AoSI2, avg_cost2, avg_msg_cost2]
    mincost = [AoSI3, avg_cost3, avg_msg_cost3]
    avgaosi = [AoSI4, avg_cost4, avg_msg_cost4]
    gfun = [AoSIg, avg_costg, avg_msg_costg]
    pfun = [AoSIp, avg_costp, avg_msg_costp]

    methods = [
        rf'Quadratic: V(x) = {fun1}',
        rf'Linear: V(x) = x',
        rf'Minimize AoSI: $V(x) = {fun2}',
        rf'Minimize cost: $V(x) = {fun3}',
        rf'Minimize message cost: $V(x) = {fun4}',
        'G-Function',
        'Original paper method'
    ]

    #labels = ['AoSI', 'Cost','Message Cost', 'Cost through AoSI', 'Average Cost', 'Average Message Cost']
    labels = ['AoSI', 'Average Cost', 'Average Message Cost']

    colors = [
        '#4C72B0',  # blue
        '#55A868',  # green
        '#C44E52',  # red
        '#8172B3',  # purple
        '#CCB974',  # yellow
        '#64B5CD',  # cyan
        '#c25a00',  # orange
    ]


    data = np.array([stand, linear ,minaosi, mincost, avgaosi, gfun, pfun])  # shape: (6 methods, 5 metrics)
    #data = np.array([stand, minaosi, mincost, avgaosi, gfun, pfun])
    data = data.T  # shape becomes (5 metrics, 6 methods) so we have 6 bars per metric

    # Normalize each column (i.e., each metric) for fair comparison
    #normalized = (data - data.min(axis=1, keepdims=True)) / (data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True) + 1e-9)
    normalized = data
    x = np.arange(len(labels))  # 5 metric labels
    width = 0.13  # smaller width to fit 6 bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot 6 bars per metric
    for i in range(7):
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, normalized[:, i], width, label=methods[i], color=colors[i])
        for j, bar in enumerate(bars):
            raw_val = data[j][i]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    format_sigfigs(raw_val), ha='center', va='bottom', fontsize=8, rotation=30)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel(f"Average over {num_steps} time steps")
    ax.set_title("Comparison of Metrics Across Methods")
    ax.set_ylim(0, 12)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    if should_save:
        filename = os.path.join(SAVE_DIR, f"Comparison of Metrics Across Methods with different Vs; {num_steps} steps.png")
        plt.savefig(filename)
        print(f"Saved: {filename}")
    plt.close()

    #plot_state_comparison([states1, states2, states4,states5, states6, states7], labels=["Paper", "G-Function", "Lyapunov Function", "Lyapunov Minimize AoSI", "Lyapunov Minimize Cost", "Lyapunov Minimize msg. cost"],name="All in one")

    if dynamic:
        plt.plot(history_r1, label="r1")
        plt.plot(history_r0, label="r0")
        plt.plot(history_rho, label="rho")
        plt.plot(history_p, label="p")
        plt.plot(history_q, label="q")

        plt.xlabel("Timestep")
        plt.ylabel("Probability")
        plt.title("Evolution of Probabilities")
        plt.legend()
        if should_save:
            filename = os.path.join(SAVE_DIR,
                                    f"Evolution of probabilities; {num_steps} steps.png")
            plt.savefig(filename)
            print(f"Saved: {filename}")
        plt.close()
