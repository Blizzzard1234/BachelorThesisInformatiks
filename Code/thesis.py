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
lambda2 = 4  # Energy cost for uncompressed
V_val = 1 # a constant multiplied with all lambda, except for cost calculation. this will only
# effect the next step calculation, but not the calculation of the final cost, so it is only a
#temporary factor to help calculations
stabilityMargine = 1
RANDOM_SEED = 123543 # Change or set to None to disable fixed seeding
h = 1 # variant exponent


mode = 3 #-1 = find the optimal V value, 0 = debugging, 1 = with G function, 2 = with F function, 3 = with Lyapunov drift



# Derived constants (stable === AOIS = 0; unstable === AOIS > 0)
a = r1                          #unstable & IDLE
b = r1 * (1 - rho * p)          #unstable & Compressed
c = r1 * (1 - rho * q)          #unstable & Uncompressed
d = 1 - r0                      #stable & IDLE
e = (1 - r0) * (1 - rho * p)    #stable & Compressed
f = (1 - r0) * ((1 - rho) + rho * (1 - q) )      #stable &  Uncompressed  this is not the same as the paper, since we also have the propability that we get something
# and deconde it, but it doesnt lead to a stable state ((1-r0) * rho * (1 - q))


######################### current stuff
if RANDOM_SEED is not None:
    rng = random.Random(RANDOM_SEED)
else:
    rng = random.Random()



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
    #TODO split the 3 sections (package recived, stabalizes, spontanious stabalisation with r_i) into 3 parts, each in row, with seperate probabilities,
    stable = False
    #The chance that it is stable
    # External stabilization attempt with probability rho
    if stable and rng.random() > r0:
        stable = False

    # Spontaneous stabilization from unstable state
    elif not stable and rng.random() > r1:
        stable = True

        
    if rng.random() < rho:
        if dt == 1 and rng.random() < p:  # compressed packet
            stable = True
        elif dt == 2 and rng.random() < q:  # uncompressed packet
            stable = True

    # Destabilize even if just stabilized


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
            for c in range(0, 20, 1):
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

    # Normalize mean and cost to [0, 1] range for balanced scoring
    means = np.array([r['mean'] for r in results])
    costs = np.array([r['cost'] for r in results])
    mean_norm = (means - means.min()) / (means.max() - means.min())
    cost_norm = (costs - costs.min()) / (costs.max() - costs.min())
    combined_scores = mean_norm + cost_norm

    for i, r in enumerate(results):
        r['combined_score'] = combined_scores[i]
    sorted_results_combined = sorted(results, key=lambda r: r['combined_score'])

    print("*Best values by AoSI:")
    for r in sorted_results_AoSI[:10]:
        print(f"a, b, c = {r['a']:.1f}, {r['b']}, {r['c']} #→ mean={r['mean']:.3f}, most_common={r['most_common']}, cost = {r['cost']}")
    print("\n*Best values by Cost:")
    for r in sorted_results_cost[:10]:
        print(f"a, b, c = {r['a']:.1f}, {r['b']}, {r['c']} #→ mean={r['mean']:.3f}, most_common={r['most_common']}, cost = {r['cost']}")
    print("\n*Top 10 most balanced values (mean and cost):")
    for r in sorted_results_combined[:10]:
        print(f"a, b, c = {r['a']:.1f}, {r['b']}, {r['c']}  #→ mean={r['mean']:.3f}, cost={r['cost']}, combined_score={r['combined_score']:.3f}")


    return sorted_results_AoSI, sorted_results_cost, sorted_results_combined





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
    sns.histplot(S_states, kde=False, bins=range(min(S_states), max(S_states) + 2), color='skyblue', edgecolor='black')

    # Add average line
    plt.axvline(mean_val, color='red', linestyle='--', linewidth=1.5, label=f'Mean = {mean_val:.2f}')

    # Plot styling

    title = f"Distribution of Age of System Instability (S(t)) over {name}"
    if mode == 3:
        title += f' Over Time with V(st) = {a}x^{b} + x*{c}'

    plt.suptitle(title, fontsize=16)
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


def plot_cost_with_convergence(cost_array, num_steps, fun, epsilon=1e-2, window=5):
    time = np.arange(num_steps)

    # Find convergence point: where the cost stops changing significantly
    # We'll check when the absolute difference between consecutive costs falls below epsilon
    diffs = np.abs(np.diff(cost_array))
    convergence_indices = 0
    for i in range(len(diffs) - window):
        if np.all(diffs[i:i + window] < epsilon):
            convergence_indices = i + window  # +window to include that region
            break

    if convergence_indices > 0:
        convergence_idx = convergence_indices + 1  # +1 because diff shifts index by 1
        convergence_value = cost_array[convergence_idx]
    else:
        convergence_idx = None
        convergence_value = None

    plt.figure(figsize=(8, 4))
    plt.plot(time, cost_array, marker='o', linestyle='-', label='Cost')

    if convergence_idx is not None:
        plt.axhline(y=convergence_value, color='r', linestyle='--',
                    label=f'Converged at step {convergence_idx}')
        plt.axvline(x=convergence_idx, color='r', linestyle=':', alpha=0.7)

    plt.xlabel('Time step')
    plt.ylabel('Cost')
    plt.title(f'average cost over 50 iterations over time for function {fun} ')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_aosi(S_states, actions, fun):
    action_colors = {
        0: 'green',  # Idle
        1: 'orange',  # Compressed
        2: 'red',  # Uncompressed
        3: 'black' #for avg run
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
    title = f'Simulated Age of System Instability (S(t)) with {fun}'

    if mode == 3:
        title += f' Over Time with V(st) = {a}x^{b} + x*{c}'

    ax.set_title(title, fontsize=16)
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




if __name__ == "__main__":

#TODO also do the same for AoSI


    num_steps = 100

    if mode < 1:
        if mode == 0:
            run_sim(num_steps)
        if mode == -1:
            find_optimal_V()



    else:

        n = 50
        avgs = np.zeros_like(range(0,n), dtype=float)
        #you can run V optimizer and then just copy paste the exact thing here
        a,b,c = 0.5, 2, 1
        #AoSI
        #a, b, c = 2.9, 13, 1 #→ mean = 0.257, most_common = 0, cost = 204
        #a, b, c = 2.7, 4, -6 #→ mean = 0.267, most_common = 0, cost = 204
        #a, b, c = 3.5, 7, 0 #→ mean = 0.267, most_common = 0, cost = 201
        #a, b, c = 1.6, 7, -9# → mean = 0.277, most_common = 0, cost = 212
        #Cost
        #a, b, c = 1.2, 0, 5 #→ mean = 1.020, most_common = 0, cost = 94
        #a, b, c = 0.0, 4, -2 #→ mean = 1.040, most_common = 0, cost = 95
        #a, b, c = 2.9, 0, -10 #→ mean = 1.010, most_common = 0, cost = 95
        #a, b, c = 2.7, 0, 5 #→ mean = 1.040, most_common = 0, cost = 97



        if mode == 1:
            fun = "F-Function"
        elif mode == 2:
            fun = "G-Function"
        elif mode == 3:
            fun = "Lyapunov drivt method"
        else:
            fun = "error"

        arr_cost = np.zeros((num_steps, 0), dtype=np.float32)
        arr_aosi = np.zeros((num_steps, 0), dtype=np.float32)
        for i in range(n):

            if mode == 1:
                S_states, actions = run_sim_1(num_steps)
            elif mode == 2:
                S_states, actions = run_sim_2(num_steps)
            elif mode == 3:
                S_states, actions = run_sim_3(num_steps, a = a, b = b, c = c)
            else:
                raise ValueError("Wrong mode selected.")
            avgs[i] = (sum(S_states) / len(S_states))
            #addint the actions to the big array


            #rewriting states and actions as np arrays with float 32, to save space later
            aosi = np.array(S_states[:num_steps], dtype=np.float32)
            acts = np.array(actions, dtype=np.int32)
            arr_aosi = np.insert(arr_aosi, i, aosi, axis=1)


            # Look up transition cost at each point
            cost_per_state = np.array([0, lambda1, lambda2], dtype=np.float32)
            state_costs = cost_per_state[acts]

            costs = aosi + state_costs
            #getting the mean up to each point

            current_cost = np.zeros(num_steps, dtype=np.float32)
            for j in range(num_steps):
                current_cost[j] = costs[:j].mean()

            arr_cost = np.insert(arr_cost, i, current_cost, axis=1)

        #making the avg cost array, then deleting the big 2d array to save space
        avg_cost = arr_cost.mean(axis=1)
        avg_aosi = arr_aosi.mean(axis = 1)
        #print(arr_cost)
        del arr_cost



        plot_cost_with_convergence(avg_cost, num_steps, fun)
        num_instable = 0

        #del arr_action

        for i in S_states:
            if not i == 0:
                num_instable = num_instable + 1

        idle_actions = actions.count(0)
        sparse_actions = actions.count(1)
        dense_actions = actions.count(2)
        stab_time = stabilization_time(S_states)

        print("\n--- Simulation Summary ---")
        print(f"Total simulation steps: {num_steps}")
        print(f"Average Age Of System Instability (ADSI): {avgs[1]:.4f}")
        print(f"Average Age Of System Instability (ADSI) over {n}: {avgs.mean():.4f}")
        print(f"Breakdown of actions taken:")
        print(f"  Idle (0): {idle_actions} ({idle_actions / num_steps:.2%})")
        print(f"  Sparse Update (1): {sparse_actions} ({sparse_actions / num_steps:.2%})")
        print(f"  Dense Update (2): {dense_actions} ({dense_actions / num_steps:.2%})")
        print(f"Cost: {sparse_actions*lambda1 + dense_actions*lambda2 + num_instable}")  #DONE it should also include the cost of it beeing in a bad state, not just communicvation
        print(f"Bellman policy (G): stabilizes at t = {stab_time}")

        if mode == 3:
            print(f"Lyapunov Cost: {sparse_actions * lambda1 + dense_actions * lambda2 + V(num_instable)}")

        #DONE for lyapunov, in addition to this, also calculate the cost (st + ld1 + ld2) with st = V(st)
        # --- Visualize S(t) over Time ---

        fun1 = fun + f"with one run"
        fun2 = fun + f"as avg over {n} runs"
        try:
            plot_aosi(S_states, actions, fun1)
            plot_aosi(avg_aosi, np.full(num_steps, 3), fun2)
        except Exception as e:
            print(f"\nError plotting with Matplotlib: {e}")
            print("Please ensure matplotlib is installed (`pip install matplotlib`) if you want to see the plot.")

        plot_state_distribution(S_states, "one run")
        plot_state_distribution(avg_aosi, f"{n} runs")