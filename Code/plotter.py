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

# max for the sum
MAX_I = 100  # Should be large enough to approximate infinite sums

#should it be computed numerically (sum from min -> min + MAX_I) => True; should be calculated using the closed_form version like given in the paper => False
numeric_calculation = False
RANDOM_SEED = 123543 # Change or set to None to disable fixed seeding

if RANDOM_SEED is not None:
    rng = random.Random(RANDOM_SEED)
else:
    rng = random.Random()


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


def stable_unstable(st, prs):
    # Determine the next state based on the calculated probability
    if rng.random() < prs:
        return st + 1
    else:
        return 0


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
    return stable_unstable(st, prob_increment)



def simulate_AoSI(num_steps):
    st = 0
    n1, n2, cost = find_optimal_thresholds(lambda1,lambda1,50,50)
    S_history = [0]  # Initialize history with the starting state
    action_history = []
    for i in range(num_steps):
        if st < n1:
            dt = 0
        elif n1 <= st <= n2:
            dt = 1
        else:
            dt = 2
        action_history.append(dt)
        st = calculate_next(st, dt)
        S_history.append(st)

    average_ADSI = sum(S_history) / len(S_history)
    idle_actions = action_history.count(0)
    sparse_actions = action_history.count(1)
    dense_actions = action_history.count(2)

    print("\n--- Simulation Summary ---")
    print(f"Total simulation steps: {num_steps}")
    print(f"Average Age Of System Instability (ADSI): {average_ADSI:.4f}")
    print(f"Breakdown of actions taken:")
    print(f"  Idle (0): {idle_actions} ({idle_actions / num_steps:.2%})")
    print(f"  Sparse Update (1): {sparse_actions} ({sparse_actions / num_steps:.2%})")
    print(f"  Dense Update (2): {dense_actions} ({dense_actions / num_steps:.2%})")
    print(f"Cost: {sparse_actions * lambda1 + dense_actions * lambda2}")



####################################################################################################################################################################################
############################################################################### Lyapunov starts here ###############################################################################
####################################################################################################################################################################################


def V(x, a): #x = timestep, a = uncompresse (0), compresed (1)
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


    current_state = 0  # 0 = stable; 1 = unstable
    drift_vals = np.zeros_like(range(0,max_steps), dtype=float)
    avg_val = np.zeros_like(range(0, max_steps), dtype=float)
    convergence = -1
    for x in range(max_steps):
        probability = random.random()
        if current_state == 0: #stable
            if r0 > probability: #stays stable
                drift = V(0,probability) - V(x,probability)
            else: #becomes unstable
                current_state = 1
                drift = V(x+1,probability) - V(x,probability)
        else: #unstable
            if r1 > probability:
                drift = V(x + 1,probability) - V(x,probability)
            else:
                current_state = 0
                drift = V(0,probability) - V(x,probability)

        drift_vals[x] = drift
        avg_val[x] = drift_vals.mean()#
        convergence = has_converged(avg_val)
    return convergence, drift_vals, avg_val





####################################################################################################################################################################################
############################################################################### Plotting starts here ###############################################################################
####################################################################################################################################################################################


#################### Fig 5 for individual lambda 1 values; fast without individual n minimisation
def plot_cost_vs_lambda2(n1, n2, lambda1_val, lambda2_range):

    
    costs = []
    lambdas = list(lambda2_range)
    un0 = compute_un0(n1, n2)

    for lam2 in lambdas:
        cost = F_function(n1, n2, un0)
        costs.append(cost)

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, costs, marker='o')
    plt.xlabel(r'$\lambda_2$ (energy cost for uncompressed)')
    plt.ylabel('Total Cost F(n1, n2)')
    plt.title(f'Cost vs λ₂ for fixed n1={n1}, n2={n2}, λ₁={lambda1_val} calculated {"Numerically" if numeric_calculation else "Using closed form"}')
    plt.suptitle(f'r₀ = {r0}; r₁ = {r1}; ρ = {rho}')
    plt.grid(True)
    plt.show()

################### individual Fig 5 calculation, with percise n calculation
def plot_cost_vs_lambda2_with_individual_ns(lambda1_val, lambda2_range, n1_range = 15, n2_range = 25):

    
    costs = []
    lambdas = list(lambda2_range)
    n1_list = []
    n2_list = []

    for lam2 in lambdas:
        n1, n2, cost = find_optimal_thresholds(lambda1_val,lam2, n1_range, n2_range)
        costs.append(cost)
        n1_list.append(n1)
        n2_list.append(n2)

    plt.figure(figsize=(10, 6))
    plt.plot(lambdas, costs, marker='o')
    for x, y, n1, n2 in zip(lambdas, costs, n1_list, n2_list):
        plt.text(x, y + 0.05, f'({y:.2f})', ha='center', fontsize=8)

    plt.xlabel(r'$\lambda_2$ (energy cost for uncompressed)')
    plt.ylabel('Total Cost F(n1, n2)')
    plt.title(f'Cost vs λ₂ for fixed n1={n1}, n2={n2}, λ₁={lambda1_val} calculated {"Numerically" if numeric_calculation else "Using closed form"}')
    plt.suptitle(f'r₀ = {r0}; r₁ = {r1}; ρ = {rho}')
    plt.grid(True)
    filename = f"Optimal_Cost_2D_lambda1_{lambda1_val}_r0_{r0:.2f}_r1_{r1:.2f}_numeric_{numeric_calculation}.png"
    plt.savefig(filename)
    plt.show()

############### fancy 3D plot for Fig 5
def plot_3d_cost_surface(lambda1_range, lambda2_range, n1_range=15, n2_range=25):

    lambda1_values = list(lambda1_range)
    lambda2_values = list(lambda2_range)

    X, Y, Z = [], [], []

    for lam1 in lambda1_values:
        for lam2 in lambda2_values:
            _, _, cost = find_optimal_thresholds(lam1, lam2, n1_range, n2_range)
            X.append(lam2)
            Y.append(lam1)
            Z.append(cost)

    fig = plt.figure(figsize=(12, 8))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_trisurf(X, Y, Z, cmap='viridis', edgecolor='none')

    ax.set_xlabel(r'$\lambda_2$ (uncompressed)')
    ax.set_ylabel(r'$\lambda_1$ (compressed)')
    ax.set_zlabel('Optimal Cost $F(n_1^*, n_2^*)$')
    ax.set_title(f'3D Cost Surface: Optimal Cost vs. λ₁ and λ₂ calculated {"Numerically" if numeric_calculation else "Using closed form"}')
    plt.suptitle(f'r₀ = {r0}; r₁ = {r1}; ρ = {rho}')

    # Set the view angle here
    #ax.view_init(elev=30, azim=45)

    plt.tight_layout()
    filename = f"Optimal_Cost_3D_r0_{r0:.2f}_r1_{r1:.2f}_numeric_{numeric_calculation}.png"
    plt.savefig(filename)
    plt.show()


############################## Figure 6
def plot_time_vs_lambda2_with_individual_ns(lambda1_val_high, lambda1_val_low, lambda2_range, n1_range = 15, n2_range = 25):

    
    costs1 = []
    costs2 = []
    costs3 = []
    costs4 = []
    lambdas = list(lambda2_range)
    n1_list = []
    n2_list = []

    for lam2 in lambdas:
        lambda1 = lambda1_val_high
        n1, n2, cost = find_optimal_thresholds(lambda1_val_high,lam2, n1_range, n2_range)
        n1_list.append(n1)
        n2_list.append(n2)
        un0 = compute_un0(n1, n2)
        cost1 = delta1_like_the_paper(n1, n2, un0)
        cost3 = delta2_like_the_paper(n1, n2, un0)
        costs1.append(cost1)
        costs3.append(cost3)

    for lam2 in lambdas:
        lambda1 = lambda1_val_low
        n1, n2, cost = find_optimal_thresholds(lambda1_val_low,lam2, n1_range, n2_range)
        n1_list.append(n1)
        n2_list.append(n2)
        un0 = compute_un0(n1, n2)
        cost2 = delta1_like_the_paper(n1, n2, un0)
        cost4 = delta2_like_the_paper(n1, n2, un0)
        costs2.append(cost2)
        costs4.append(cost4)

    plt.figure(figsize=(10, 6))
    x = range(len(costs1))
    plt.plot(x, costs1, label='Low Quality: λ₁=1')
    plt.plot(x, costs2, label='Low Quality: λ₁=6')
    plt.plot(x, costs3, label='High Quality: λ₁=1')
    plt.plot(x, costs4, label='High Quality: λ₁=6')
    plt.legend()
    plt.xlabel(r'$\lambda_2$ (energy cost for uncompressed)')
    plt.ylabel('Optimal Transmission Time')
    plt.title(f' The evolution of the average transmission time in function of λ₁ and λ₂. Calculated {"Numerically" if numeric_calculation else "Using closed form"}')
    plt.suptitle(f'r₀ = {r0}; r₁ = {r1}; ρ = {rho}')
    plt.grid(True)
    filename = f"Time_vs_Lambda_r0_{r0:.2f}_r1_{r1:.2f}_numeric_{numeric_calculation}.png"
    plt.savefig(filename)
    plt.show()


def plot_lyapunov_drift(lambda1_range, lambda2_range, max_range, n1_range=15, n2_range=25):
    
    convergence, drift_vals, avg_val = simulate_lyapunov(max_range)

    avg_stability = 0
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
    filename = f"Lyapunov_drift_r0_{r0:.2f}_r1_{r1:.2f}_numeric_{numeric_calculation}.png"
    plt.savefig(filename)
    plt.show()







if __name__ == "__main__":
    ### parameters because why not


    ## execute the program
    plot_3d_cost_surface(range(0,10), range(0,10), 100, 100)
    # plot_cost_vs_lambda2_with_individual_ns(0,range(0,10),100, 100)
    #plot_cost_vs_lambda2_with_individual_ns(10,range(0,10), 100, 100)
    plot_time_vs_lambda2_with_individual_ns(1,6, range(0,10), 100, 100)

    plot_lyapunov_drift(1,6, 1000, 100, 100)

    simulate_AoSI(10000)


    print("Done")