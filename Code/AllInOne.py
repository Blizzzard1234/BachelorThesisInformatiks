import numpy as np
import matplotlib.pyplot as plt
import random
from matplotlib.collections import LineCollection
from matplotlib.lines import Line2D
import seaborn as sns
###################### Parameters
r1 = 0.9 #pr that the state will remain unstable, independently of everything else
r0 = 0.1 #pr that the state will remain stable, independently of everything else
rho = 0.9 # pr that a package was successfully decoded
p = 0.5 # pr that we become stable when controler recives a compressed message
q = 0.9  #pr that we become stable when controler recives a uncompressed message
lambda1 = 1  # Energy cost for compressed
lambda2 = 3  # Energy cost for uncompressed
V_val = 1 # a constant multiplied with all lambda, except for cost calculation. this will only
# effect the next step calculation, but not the calculation of the final cost, so it is only a
#temporary factor to help calculations
stabilityMargine = 1
RANDOM_SEED = 123543 # Change or set to None to disable fixed seeding
h = 1 # variant exponent
MAX_I = 100  # Should be large enough to approximate infinite sums

#should it be computed numerically (sum from min -> min + MAX_I) => True; should be calculated using the closed_form version like given in the paper => False
numeric_calculation = False

mode = 3 #-1 = find the optimal V value, 0 = debugging, 1 = with G function, 2 = with F function, 3 = with Lyapunov drift



# Derived constants (stable === AOIS = 0; unstable === AOIS > 0)
a = r1                          #unstable & IDLE
b = r1 * (1 - rho * p)          #unstable & Compressed
c = r1 * (1 - rho * q)          #unstable & Uncompressed
d = 1 - r0                      #stable & IDLE
e = (1 - r0) * (1 - rho * p)    #stable & Compressed
f = (1 - r0) * ((1 - rho) + rho * (1 - q) )      #stable &  Uncompressed  this is not the same as the paper, since we also have the propability that we get something
# and deconde it, but it doesnt lead to a stable state ((1-r0) * rho * (1 - q))

rng = random.Random()
######################### current stuff
def reset_random():
    global rng
    rng = random.Random(RANDOM_SEED)

# this function calculates the next step. it uses the F function
def get_transition_probs(st):
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


def stable_unstable(st, dt, prs):
    # Determine the next state based on the calculated probability
    if rng.random() < prs[dt]:
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


def pick_lyapunov(st,a,b,c):


    vals = {}

    prs = get_transition_probs(st)


    ly = V(st,a,b,c)

    #This should give the expected |E value
    vals[0] = ((V(st+1,a,b,c) - ly) + (st +1) * stabilityMargine) * prs[0] + (V(0,a,b,c) - ly) * (1 - prs[0])
    vals[1] = ((V(st+1,a,b,c) - ly) + lambda1 * V_val + (st +1)* stabilityMargine) * prs[1] + (V(0,a,b,c) - ly + lambda1 * V_val) * (1 - prs[1])
    vals[2] = ((V(st+1,a,b,c) - ly) + lambda2 * V_val + (st +1)* stabilityMargine) * prs[2] + (V(0,a,b,c) - ly + lambda2 * V_val) * (1 - prs[2])



    optimal_action = min(vals, key=vals.get)

    return optimal_action, prs.copy()





def run_sim (numruns = 1000):
    st = 0
    dt = 0

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
    for t in range(num_runs):
        # 1. Determine the optimal action for the current state S(t)
        action, _= pick_lyapunov(st,a,b,c)
        action_history.append(action)

        # 2. Calculate the next state S(t+1) based on the chosen action
        probs = get_transition_probs(st)
        next_S = stable_unstable(st, action, probs)

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
            for c in range(-10, 10, 1):
                st, actions = run_sim_3(100, a, b, c)
                actions = np.array(actions)
                st = np.array(st)

                #num_idle = np.count_nonzero(actions == 0)
                num_comp = np.count_nonzero(actions == 1)
                num_uncomp = np.count_nonzero(actions == 2)
                num_instable = 0

                for i in st:
                    if not i == 0:
                        num_instable = num_instable + 1

                cost = lambda1 * num_comp + lambda2 * num_uncomp + num_instable
                mean_val = st.mean()
                most_common_val = find_most_common(st)
                results.append({
                    'a': a,
                    'b': b,
                    'c': c,
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
    sns.histplot(S_states, kde=True, bins=range(min(S_states), max(S_states) + 2),
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

# Derived constants
a = r1
b = r1 * (1 - rho * p)
c = r1 * (1 - rho * q)
d = 1 - r0
e = (1 - r0) * (1 - rho * p)
f = (1 - r0) * (1 - rho)

# Function to compute stationary distribution u_n(i)
def compute_un_i(i, n1, n2, un0):
    if n1 > 0 and n2 > 0:
        if i == 0:
            return un0
        elif 1 <= i <= n1:
            return d * a ** (i - 1) * un0
        elif n1 < i <= n2:
            return d * a ** (n1 - 1) * b ** (i - n1) * un0
        else:
            return d * a ** (n1 - 1) * b ** (n2 - n1) * c ** (i - n2) * un0
    elif n1 == 0 and n2 > 0:
        if i == 0:
            return un0
        elif 1 <= i <= n2:
            return e * b ** (i - 1) * un0
        else:
            return e * b ** (n2 - 1) * c ** (i - n2) * un0
    elif n1 == n2 == 0:
        if i == 0:
            return un0
        else:
            return f * c ** (i - 1) * un0
    else:
        raise ValueError("Invalid threshold combination")




########################## compute u_n(0)
def compute_un0(n1, n2):
    if n1 > 0 and n2 > 0:
        num = (1 - a) * (1 - b) * (1 - c)
        den = ((1 - a + d) * (1 - c) * (1 - b) +
                       (1 - c) * (b - a) * d * a ** (n1 - 1) +
                       d * (1 - a) * (c - b) * b ** (n2 - n1) * a ** (n1 - 1))
        return num / den
    elif n1 == 0 and n2 > 0:
        num = (1 - b) * (1 - c)
        den = (1 - b + e) * (1 - c) + (c - b) * e * b ** (n2 - 1)
        return num / den
    elif n1 == n2 == 0:
        return (1 - c) / (1 - c + f)
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
        term1 = un0*((d*(a**(n1+1)*n1 - n1*a**(n1) - a**(n1) + 1))/((1-a)**2))
        term2 = un0*( (d*b*a**(n1-1) * (b**(n2-n1) * (b*n2 - n2 - 1) - b*n1 + n1 + 1)) / ((1-b)**2) )
        term3 = un0*( (d*c*b**(n2-n1) * a**(n1-1) * (-c*n2 + n2 +1)) / ((1-c)**2) )
        return term1 + term2 + term3
    elif n1 == 0 and n2 > 0:
        term1 = un0 * (e * (b**(n2 + 1) * n2 - n2 * b**n2 - b**n2 + 1)) / ((1 - b) ** 2)
        term2 = un0*(e*c*b**(n2-1) * (-c*n2 + n2 + 1)) / ((1-c)**2)
        return term1 + term2
    elif n1 == n2 == 0:
        return un0 * f / ((1-c)**2)
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
        return un0*d*a**(n1 - 1) * ((1 - b ** (n2 - n1)) / (1 - b))
    elif n1 == 0 and n2 > 0:
        return un0 * (1 + e * ((1 - b ** (n2 - 1)) / (1 - b)))
    elif n1 == n2 == 0:
        return 0
    else:
        raise ValueError("Invalid threshold combination")


def delta2_like_the_paper(n1,n2, un0):
    if n1 > 0 and n2 > 0:
        return un0 * d * a ** (n1-1) * b**(n2-n1) / (1-c)
    elif n1 == 0 and n2 > 0:
        return un0 * e * b **(n2-1) / (1-c)
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
def find_optimal_thresholds(lambda1_val, lambda2_val, max_n1=15, max_n2=25):
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
    avgs = np.zeros_like(range(0, n), dtype=float)
    for i in range(n):
        S_states, actions = simulate_paper(num_steps)

        avgs[i] = (sum(S_states) / len(S_states))

    num_instable = 0

    for i in S_states:
        if not i == 0:
            num_instable = num_instable + 1

    idle_actions = actions.count(0)
    sparse_actions = actions.count(1)
    dense_actions = actions.count(2)
    stab_time = stabilization_time(S_states)
    cost = sparse_actions * lambda1 + dense_actions * lambda2 + num_instable
    AoSI = avgs.mean()
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
        ax.set_title(f'Simulated Age of System Instability (S(t)) Over Time with V(st) = {a}x^{b} + x*{c} Paper Simulation',
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


    return AoSI, cost, actions, S_states, idle_actions, sparse_actions, dense_actions


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
            num_instable = num_instable + 1

    idle_actions = actions.count(0)
    sparse_actions = actions.count(1)
    dense_actions = actions.count(2)
    stab_time = stabilization_time(S_states)
    cost = sparse_actions * lambda1 + dense_actions * lambda2 + num_instable
    AoSI = avgs.mean()
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
        ax.set_title(f'Simulated Age of System Instability (S(t)) Over Time with V(st) = {a}x^{b} + x*{c}. G-Function',
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

    return AoSI, cost, actions, S_states, idle_actions, sparse_actions, dense_actions


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
            num_instable = num_instable + 1

    idle_actions = actions.count(0)
    sparse_actions = actions.count(1)
    dense_actions = actions.count(2)
    stab_time = stabilization_time(S_states)
    cost = sparse_actions * lambda1 + dense_actions * lambda2 + num_instable
    AoSI = avgs.mean()
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
        ax.set_title(f'Simulated Age of System Instability (S(t)) Over Time with V(st) = {a}x^{b} + x*{c}. F-Function',
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

    return AoSI, cost, actions, S_states, idle_actions, sparse_actions, dense_actions


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
            num_instable = num_instable + 1

    idle_actions = actions.count(0)
    sparse_actions = actions.count(1)
    dense_actions = actions.count(2)
    stab_time = stabilization_time(S_states)
    cost = sparse_actions * lambda1 + dense_actions * lambda2 + num_instable
    AoSI = avgs.mean()
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
        ax.set_title(f'Simulated Age of System Instability (S(t)) Over Time with V(st) = {a}x^{b} + x*{c}. Lyapunov Function',
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

    plot_state_distribution(S_states, "Lyapunov Function")

    return AoSI, cost, actions, S_states, idle_actions, sparse_actions, dense_actions

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

if __name__ == "__main__":
    num_steps = 10000



    AoSI1, cost1, _, _, idle_actions1, sparse_actions1, dense_actions1 = sim_paper(num_steps)
    AoSI2, cost2, _, _, idle_actions2, sparse_actions2, dense_actions2 = sim_GFun(num_steps)
    AoSI3, cost3, _, _, idle_actions3, sparse_actions3, dense_actions3 = sim_FFun(num_steps)


    #a,b,c = 0.5, 2, 0
    # AoSI
    #a, b, c = 2.9, 13, 1 #→ mean = 0.257, most_common = 0, cost = 204
    #a, b, c = 2.7, 4, -6 #→ mean = 0.267, most_common = 0, cost = 204
    #a, b, c = 3.5, 7, 0 #→ mean = 0.267, most_common = 0, cost = 201
    #a, b, c = 1.6, 7, -9# → mean = 0.277, most_common = 0, cost = 212
    # Cost
    #a, b, c = 1.2, 0, 5  # → mean = 1.020, most_common = 0, cost = 94
    #a, b, c = 0.0, 4, -2 #→ mean = 1.040, most_common = 0, cost = 95
    #a, b, c = 2.9, 0, -10 #→ mean = 1.010, most_common = 0, cost = 95
    #a, b, c = 2.7, 0, 5 #→ mean = 1.040, most_common = 0, cost = 97

    #*Top 10 most balanced values(mean and cost):
    a, b, c = 0.5, 1, 4  # → mean=0.782, cost=113, combined_score=0.329
    #a, b, c = 1.2, 1, 5  # → mean=0.782, cost=115, combined_score=0.339
    #a, b, c = 1.2, 0, 5  # → mean=1.020, cost=94, combined_score=0.345
    #a, b, c = 2.9, 0, -10  # → mean=1.010, cost=95, combined_score=0.346


    AoSI4, cost4, _, _, idle_actions4, sparse_actions4, dense_actions4 = sim_lya(num_steps,a,b,c)

    # Actual data (replace these with your real values)
    paper = [AoSI1, cost1, idle_actions1, sparse_actions1, dense_actions1]
    g_func = [AoSI2, cost2, idle_actions2, sparse_actions2, dense_actions2]
    f_func = [AoSI3, cost3, idle_actions3, sparse_actions3, dense_actions3]
    lyapunov = [AoSI4, cost4, idle_actions4, sparse_actions4, dense_actions4]

    labels = ['AoSI', 'Cost', 'Idle', 'Sparse', 'Dense']
    methods = ['Paper', 'G-Func', 'F-Func', 'Lyapunov']
    colors = ['#4C72B0', '#55A868', '#C44E52', '#8172B3']

    data = np.array([paper, g_func, f_func, lyapunov])

    # Normalize each column (metric) to [0, 1] for comparability
    normalized = (data - data.min(axis=0)) / (data.max(axis=0) - data.min(axis=0) + 1e-9)

    x = np.arange(len(labels))  # group positions
    width = 0.2

    fig, ax = plt.subplots(figsize=(10, 6))

    for i in range(4):
        bars = ax.bar(x + (i - 1.5) * width, normalized[i], width, label=methods[i], color=colors[i])

        for j, bar in enumerate(bars):
            val = data[i][j]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    f'{val:.2f}', ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel('Normalized Value')
    ax.set_title(f'Metric Comparison (Normalized Per Metric). V(s) = {a:.1f} x ^ {b:.1f} + x * {c:.1f}')
    ax.legend()
    ax.set_ylim(0, 1.2)
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()

    ######### INDIVIDUAL PLOTS PER METRIC
    for i, metric in enumerate(labels):
        values = [paper[i], g_func[i], f_func[i], lyapunov[i]]

        fig, ax = plt.subplots(figsize=(6, 4))
        bars = ax.bar(methods, values, color=colors)

        for j, bar in enumerate(bars):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + max(values) * 0.02,
                    format_sigfigs(values[j]), ha='center', va='bottom', fontsize=8)

        ax.set_ylabel(metric)
        ax.set_title(f'{metric} Comparison. V(s) = {a:.1f} x ^ {b:.1f} + x * {c:.1f}')
        ax.grid(axis='y', linestyle='--', alpha=0.5)
        plt.tight_layout()
        plt.show()


    print("Comparing 4 different values")

    AoSIp, costp, idle_actionsp, sparse_actionsp, dense_actionsp = AoSI1, cost1, idle_actions1, sparse_actions1, dense_actions1
    AoSIg, costg, idle_actionsg, sparse_actionsg, dense_actionsg = AoSI2, cost2, idle_actions2, sparse_actions2, dense_actions2

    a, b, c = 0.5, 2, 0
    AoSI1, cost1, _, _, idle_actions1, sparse_actions1, dense_actions1 = sim_lya(num_steps, a, b, c)
    a, b, c = 2.9, 13, 1  # → mean = 0.257, most_common = 0, cost = 204
    AoSI2, cost2, _, _, idle_actions2, sparse_actions2, dense_actions2 = sim_lya(num_steps, a, b, c)
    a, b, c = 1.2, 0, 5  # → mean = 1.020, most_common = 0, cost = 94
    AoSI3, cost3, _, _, idle_actions3, sparse_actions3, dense_actions3 = sim_lya(num_steps, a, b, c)
    a, b, c = 0.5, 1, 4  # → mean=0.782, cost=113, combined_score=0.329
    AoSI4, cost4, _, _, idle_actions4, sparse_actions4, dense_actions4 = sim_lya(num_steps, a, b, c)

    stand = [AoSI1, cost1, idle_actions1, sparse_actions1, dense_actions1]
    minaosi = [AoSI2, cost2, idle_actions2, sparse_actions2, dense_actions2]
    mincost = [AoSI3, cost3, idle_actions3, sparse_actions3, dense_actions3]
    avgaosi = [AoSI4, cost4, idle_actions4, sparse_actions4, dense_actions4]
    gfun = [AoSIg, costg, idle_actionsg, sparse_actionsg, dense_actionsg]
    pfun = [AoSIp, costp, idle_actionsp, sparse_actionsp, dense_actionsp]

    methods = [
        r'Standard: $V(x) = 0.5 \cdot x^2$',
        r'Minimize AoSI: $V(x) = 2.9 \cdot x^{13} + x$',
        r'Minimize cost: $V(x) = 1.2 + 5 \cdot x$',
        r'Average both: $V(x) = 0.5 \cdot x + 4 \cdot x$',
        'G-Function',
        'Original paper method'
    ]

    labels = ['AoSI', 'Cost', 'Idle', 'Sparse', 'Dense']

    colors = [
        '#4C72B0',  # blue
        '#55A868',  # green
        '#C44E52',  # red
        '#8172B3',  # purple
        '#CCB974',  # yellow
        '#64B5CD',  # cyan
    ]

    data = np.array([stand, minaosi, mincost, avgaosi, gfun, pfun])  # shape: (6 methods, 5 metrics)
    data = data.T  # shape becomes (5 metrics, 6 methods) so we have 6 bars per metric

    # Normalize each column (i.e., each metric) for fair comparison
    normalized = (data - data.min(axis=1, keepdims=True)) / (
                data.max(axis=1, keepdims=True) - data.min(axis=1, keepdims=True) + 1e-9)

    x = np.arange(len(labels))  # 5 metric labels
    width = 0.13  # smaller width to fit 6 bars

    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot 6 bars per metric
    for i in range(6):
        offset = (i - 2.5) * width
        bars = ax.bar(x + offset, normalized[:, i], width, label=methods[i], color=colors[i])
        for j, bar in enumerate(bars):
            raw_val = data[j][i]
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.02,
                    format_sigfigs(raw_val), ha='center', va='bottom', fontsize=8)

    ax.set_xticks(x)
    ax.set_xticklabels(labels)
    ax.set_ylabel("Normalized Value")
    ax.set_title("Comparison of Metrics Across Methods")
    ax.set_ylim(0, 1.2)
    ax.legend()
    plt.grid(axis='y', linestyle='--', alpha=0.5)
    plt.tight_layout()
    plt.show()
