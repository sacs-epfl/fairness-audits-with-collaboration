from german_credit_SS import load_dataset, protected_attributes, \
      demographic_parity, error_DP
from itertools import combinations
import numpy as np
from german_credit_CS import CS
import pandas as pd
import matplotlib.pyplot as plt

# ground truth probabilities and dps
def get_probs_and_ys(X, y):

    n = len(protected_attributes)

    all_probs = dict()
    all_ys  = dict()

    for k in range(1,n+1):

        print(f'Working on k={k}')
        all_probs[k] = dict()
        all_ys[k] = dict()

        agent_combinations_list = list(combinations(range(n), k))

        for agent_combination in agent_combinations_list:
            agent_comb_str = ''.join([str(elem) for elem in agent_combination])
            
            all_probs[k][agent_comb_str] = dict()
            all_ys[k][agent_comb_str] = dict()

            total_strings = 2**(k)
            binary_strings = [format(i, f'0{k}b') for i in range(total_strings)]

            attrs = [protected_attributes[i] for i in agent_combination]
            print(f'Working on {attrs}')
            for binary_string in binary_strings:
                
                pairs = [(attrs[i], int(binary_string[i])) for i in range(k)]

                # Restore X_transformed that satisfies the binary string
                X_temp = X.copy()
                for attr, val in pairs:
                    X_temp = X_temp[X_temp[attr] == val]
                y_tmp = y.loc[X_temp.index]

                all_probs[k][agent_comb_str][binary_string] = len(X_temp) / len(X)
                all_ys[k][agent_comb_str][binary_string] = y_tmp.mean().item()

    return all_probs, all_ys

def demographic_parity_unbiased(X, y, attr, all_probs, all_ys, other_attrs):
    
    n_attrs = len(other_attrs)
    n_subspaces = 2**n_attrs

    X_1 = X[X[attr] == 1]
    y_1 = y.loc[X_1.index]

    X_0 = X[X[attr] == 0]
    y_0 = y.loc[X_0.index]

    subspaces_1 = []
    subspaces_0 = []

    binary_strings = [format(i, f'0{n_attrs}b') for i in range(n_subspaces)]

    for binary_string in binary_strings:
            
        pairs = [(other_attrs[i], int(binary_string[i])) for i in range(n_attrs)]

        X_temp = X_1.copy()
        for a, val in pairs:
            X_temp = X_temp[X_temp[a] == val]

        y_tmp = y_1.loc[X_temp.index]
        subspaces_1.append((X_temp, y_tmp))

        X_temp = X_0.copy()
        for a, val in pairs:
            X_temp = X_temp[X_temp[a] == val]
        y_tmp = y_0.loc[X_temp.index]
        subspaces_0.append((X_temp, y_tmp))
    
    agent_ids = [protected_attributes.index(a) for a in other_attrs]
    agent_ids.sort()
    agent_ids = ''.join([str(elem) for elem in agent_ids])

    dp_mean = 0
    dp_subspaces = []
    for i, binary_string in enumerate(binary_strings):
        # print(f'Agent string {agent_ids}, Binary string: {binary_string}')
        prob_subspace = all_probs[n_attrs][agent_ids][binary_string]
        dp_subspace = np.abs(subspaces_1[i][1].mean() - subspaces_0[i][1].mean()).item()
        dp_subspaces.append(dp_subspace)
        dp_mean += dp_subspace * prob_subspace

    dp_std_squared = 0
    extended_agent_ids = [protected_attributes.index(a) for a in other_attrs] + [protected_attributes.index(attr)]
    extended_agent_ids.sort()
    extended_agent_ids = ''.join([str(elem) for elem in extended_agent_ids])

    for i, binary_string in enumerate(binary_strings):
        prob_subspace = all_probs[n_attrs][agent_ids][binary_string]
        original_subspace_size = int(prob_subspace * 1000)  # there are 1000 data points in total
        sampled_subspace_size = len(subspaces_1[i][0])

        extended_binary_string_1 = '1' + binary_string
        extended_binary_string_0 = '0' + binary_string

        # print(f'Ex agent string {extended_agent_ids}, Ex binary string: {extended_binary_string_1}')
        
        ground_truth_y_given_1 = all_ys[n_attrs+1][extended_agent_ids][extended_binary_string_1]
        ground_truth_y_given_0 = all_ys[n_attrs+1][extended_agent_ids][extended_binary_string_0]
        
        dp_subspace_ground_truth = np.abs(ground_truth_y_given_1 - ground_truth_y_given_0).item()

        dp_std_squared += ((dp_subspaces[i] - dp_subspace_ground_truth)**2) * prob_subspace * (1 - sampled_subspace_size/original_subspace_size)
    
    dp_std = np.sqrt(dp_std_squared).item()

    return dp_mean, dp_std

def error_DP_unbiased(X, y, attr, all_probs, all_ys, other_attrs, ground_truth_dp):
    if other_attrs == []:
        return error_DP(X, y, attr, ground_truth_dp), None
    else:    
        dp_mean, dp_std = demographic_parity_unbiased(X, y, attr, all_probs, all_ys, other_attrs)
        return np.abs(dp_mean - ground_truth_dp[attr]).item(), dp_std 

if __name__ == '__main__':
    
    random_seed = 101
    rng = np.random.default_rng(random_seed)
    
    X, y = load_dataset()

    ground_truth_dp = dict()
    for attr in protected_attributes:
        dp_o = demographic_parity(X, y, attr)
        print("Demographic parity of the original dataset on {}: {:.3f}".format(attr, dp_o))
        ground_truth_dp[attr] = dp_o

    all_probs, all_ys = get_probs_and_ys(X, y)

    b = 100
    n_repeat = 5
    n_times = 5
    n = len(protected_attributes)
    
    agentwise_mean = []
    agentwise_std = []

    for i in range(n):

        print(f'Agent {i} - {protected_attributes[i]}')

        total_results = []

        for t in range(n_times):
            
            print(f'===> Global sampling {t+1}/{n_times}')
        

            joint_results = dict()

            for k in range(n):
                print(f'k = {k+1}/{n}')

                joint_results[k] = [list(), list()]

                # choose k agents excluding i
                possible_js = [j for j in range(n) if j != i]
                js = rng.choice(possible_js, size=k, replace=False)
                other_attrs = [protected_attributes[m] for m in js]

                js_and_i = [i] + list(js)

                for r in range(n_repeat):
                    print(f'Iteration {r+1}/{n_repeat}')

                    X_i, y_i = CS(X, y, b, protected_attributes[i],
                                random_seed=random_seed+100*(r+1), collaborators=[])
                    e_i = error_DP(X_i, y_i, protected_attributes[i], ground_truth_dp)            
                    
   
                    X_tot = pd.DataFrame()
                    y_tot = pd.DataFrame()
                    for j in js_and_i:

                        collaborators = [protected_attributes[m] for m in js_and_i if m != j]
                        X_j, y_j = CS(X, y, b, protected_attributes[j],
                                    random_seed=random_seed+100*(r+1), collaborators=collaborators)

                        X_tot = pd.concat([X_tot, X_j])
                        y_tot = pd.concat([y_tot, y_j])
                    
                    X_tot = X_tot[~X_tot.index.duplicated(keep='first')]
                    y_tot = y_tot[~y_tot.index.duplicated(keep='first')]
                    
                    assert len(X_tot) == len(y_tot)
                    
                    e_i_aposteriori, _ = error_DP_unbiased(X_tot, y_tot, protected_attributes[i], 
                                                        all_probs, all_ys, other_attrs, ground_truth_dp)

                    if k == 0:
                        assert e_i == e_i_aposteriori

                    joint_results[k][0].append(e_i)
                    joint_results[k][1].append(e_i_aposteriori)

                joint_results[k] = np.mean(joint_results[k][0])/np.mean(joint_results[k][1])

            total_results.append(list(joint_results.values()))
        
        agentwise_mean.append(np.mean(total_results, axis=0))
        agentwise_std.append(np.std(total_results, axis=0))

    agentwise_mean = np.array(agentwise_mean)
    agentwise_std = np.array(agentwise_std)

    # create a 5 x 2 subplot for 10 agents
    _, axs = plt.subplots(n, 1, figsize=(10, 20))
    axs = axs.flatten()
    for i in range(n):
        axs[i].plot(range(1,n+1), agentwise_mean[i], marker='o', color='blue')
        axs[i].fill_between(range(1,n+1), agentwise_mean[i]-agentwise_std[i], agentwise_mean[i]+agentwise_std[i], color='blue', alpha=0.2)
        axs[i].set_xlabel(r'Number of agents $(K)$')
        axs[i].set_ylabel(r'$e(CS^1)/e(CS^K)$')
        axs[i].set_title(f'Agent {i} - {protected_attributes[i]}')
    plt.tight_layout()
    plt.savefig(f'results/GC/ub_thm2_CS_all_seed{random_seed}_budget{b}.png')
            

    