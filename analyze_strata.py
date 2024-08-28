import pickle
import numpy as np
import matplotlib.pyplot as plt
import os
from itertools import combinations

def get_max_value(nested_dict):
    max_value = 0
    for _, v in nested_dict.items():
        if isinstance(v, dict):
            v = get_max_value(v)
        if v > max_value:
            max_value = v
    return max_value

def get_sum_value(nested_dict):
    sum_value = 0
    for _, v in nested_dict.items():
        if isinstance(v, dict):
            v = get_sum_value(v)
        sum_value += v
    return sum_value

################################################################################
# Folktables
probs_file = os.path.join("data", "folktables", "all_nks.pkl")
ts = 5916565

print("="*5, "Computing for Folktables", "="*5)
with open(probs_file, 'rb') as f:
    ns = pickle.load(f)

n = 5

max_ns = [[] for _ in range(1, n+1)]

for k in range(1, n+1):
    possible_collaborators = list(range(n))
    agent_combinations_list = list(combinations(possible_collaborators, k))

    for agent_combination in agent_combinations_list:
        agent_comb_str = ''.join([str(elem) for elem in agent_combination])
        
        base_agent = agent_combination[0]
        rem_agents_str = agent_comb_str[1:]

        ns_dict = ns[base_agent][k-1][rem_agents_str]
        max_ns[k-1].append(get_max_value(ns_dict))

max_ps = [[x/ts for x in max_ns[k]] for k in range(n)]


# create data points from max ps
xs1 = []; ys1 = []
for k in range(n):
    # sample a small number between 0 and 1
    # to add some noise to the data points
    noise = np.random.uniform(0, 0.2, len(max_ps[k]))
    xs1.extend([k+1+noise[i] for i in range(len(max_ps[k]))])
    ys1.extend([max_ps[k][i] for i in range(len(max_ps[k]))])

################################################################################
# German Credit
probs_file = os.path.join("data", "german_credit", "all_nks.pkl")
ts = 1000

print("="*5, "Computing for German Credit", "="*5)
with open(probs_file, 'rb') as f:
    ns = pickle.load(f)

n = 5

max_ns = [[] for _ in range(1, n+1)]

for k in range(1, n+1):
    possible_collaborators = list(range(n))
    agent_combinations_list = list(combinations(possible_collaborators, k))

    for agent_combination in agent_combinations_list:
        agent_comb_str = ''.join([str(elem) for elem in agent_combination])
        
        base_agent = agent_combination[0]
        rem_agents_str = agent_comb_str[1:]

        ns_dict = ns[base_agent][k-1][rem_agents_str]
        max_ns[k-1].append(get_max_value(ns_dict))

max_ps = [[x/ts for x in max_ns[k]] for k in range(n)]

# create data points from max ps
xs2 = []; ys2 = []
for k in range(n):
    # sample a small number between 0 and 1
    # to add some noise to the data points
    noise = np.random.uniform(0, 0.2, len(max_ps[k]))
    xs2.extend([k+1+noise[i] for i in range(len(max_ps[k]))])
    ys2.extend([max_ps[k][i] for i in range(len(max_ps[k]))])

################################################################################
# propublica
probs_file = os.path.join("data", "propublica", "all_nks.pkl")
ts = 6172

print("="*5, "Computing for Propublica", "="*5)
with open(probs_file, 'rb') as f:
    ns = pickle.load(f)

n = 5

max_ns = [[] for _ in range(1, n+1)]

for k in range(1, n+1):
    possible_collaborators = list(range(n))
    agent_combinations_list = list(combinations(possible_collaborators, k))

    for agent_combination in agent_combinations_list:
        agent_comb_str = ''.join([str(elem) for elem in agent_combination])
        
        base_agent = agent_combination[0]
        rem_agents_str = agent_comb_str[1:]

        ns_dict = ns[base_agent][k-1][rem_agents_str]
        max_ns[k-1].append(get_max_value(ns_dict))

max_ps = [[x/ts for x in max_ns[k]] for k in range(n)]

# create data points from max ps
xs3 = []; ys3 = []
for k in range(n):
    # sample a small number between 0 and 1
    # to add some noise to the data points
    noise = np.random.uniform(0, 0.2, len(max_ps[k]))
    xs3.extend([k+1+noise[i] for i in range(len(max_ps[k]))])
    ys3.extend([max_ps[k][i] for i in range(len(max_ps[k]))])

# regression line y = 1/2x
reg_xs = np.linspace(1, 5, 100)
reg_ys = [1/(2*x) for x in reg_xs]

################################################################################
print("="*5, "Generating plot", "="*5)

# Set the params
s = 6
params = {
   'legend.fontsize': s,
   'legend.title_fontsize': s,
   'xtick.labelsize': 6,
   'ytick.labelsize': 6,
   'axes.labelsize': s+1,
   'text.usetex': False,
   'figure.figsize': [3.5, 1.4],
   'lines.linewidth': 1,
   'lines.markersize': 3,
   'axes.titlesize': s,
   }

# set rc params
plt.rcParams.update(params)
# no background for the legend
plt.rc('legend', frameon=False) # no background for the legend

# make 1 x 3 subplots
fig, ax = plt.subplots(nrows=1, ncols=3)

# add grid to the subplots
for i in range(3):
    ax[i].grid(True, which='both', linestyle='--', linewidth=0.5)

# scatter each dataset across the subplots
ax[0].scatter(xs1, ys1)
ax[1].scatter(xs2, ys2)
ax[2].scatter(xs3, ys3)

# plot the regression line y = 1/2x
for i in range(3):
    ax[i].plot(reg_xs, reg_ys, color='red')

# set the x and y labels
ax[0].set_xlabel('No. of agents')
ax[0].set_ylabel('Relative size of \n largest stratum')

ax[1].set_xlabel('No. of agents')
ax[2].set_xlabel('No. of agents')

# set the title for each subplot
ax[0].set_title('Folktables')
ax[1].set_title('German Credit')
ax[2].set_title('Propublica')

# show all xticks
for i in range(3):
    ax[i].set_xticks(range(1, 6))

plt.tight_layout()

plot_name = 'largest_stratum.pdf'
save_file_path = os.path.join("results", "plots")
if not os.path.exists(save_file_path):
    os.makedirs(save_file_path)
plt.savefig(os.path.join(save_file_path, plot_name), bbox_inches='tight', dpi=300)