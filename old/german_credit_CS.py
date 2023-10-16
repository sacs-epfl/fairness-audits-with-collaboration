from german_credit_SS import load_dataset, error_DP, protected_attributes, demographic_parity
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def CS(X, y, n, attribute, random_seed=42, collaborators=[]):

    # Get index of attribute in protected_attributes
    index = protected_attributes.index(attribute)
    random_state = random_seed + index
    
    all_attrs = collaborators + [attribute]
    n_attrs = len(collaborators) + 1

    n_subspaces = 2**n_attrs
    binary_strings = [format(i, f'0{n_attrs}b') for i in range(n_subspaces)]

    subspaces = []
    for binary_string in binary_strings:
            
        pairs = [(all_attrs[i], int(binary_string[i])) for i in range(n_attrs)]

        X_temp = X.copy()
        for attr, val in pairs:
            X_temp = X_temp[X_temp[attr] == val]

        y_tmp = y.loc[X_temp.index]
        subspaces.append((X_temp, y_tmp))
    
    # sample sub_n from each subspace
    sub_n = n//n_subspaces
    subset = pd.DataFrame()
    subset_y = pd.DataFrame()
    for X_i, y_i in subspaces:
        
        # check if subspace has sufficient samples
        if len(X_i) < sub_n:
            raise ValueError('Subspace has insufficient samples')

        subset_i = X_i.sample(n=sub_n, random_state=random_state)
        subset_i_y = y_i.loc[subset_i.index]
        subset = pd.concat([subset, subset_i])
        subset_y = pd.concat([subset_y, subset_i_y])

    return subset, subset_y

if __name__ == '__main__':
    random_seed = 101
    rng = np.random.default_rng(random_seed)
    
    X, y = load_dataset()

    ground_truth_dp = dict()
    for attr in protected_attributes:
        dp_o = demographic_parity(X, y, attr)
        print("Demographic parity of the original dataset on {}: {:.3f}".format(attr, dp_o))
        ground_truth_dp[attr] = dp_o


    b = 100
    n_repeat = 3
    n_times = 10
    n = len(protected_attributes)
    
    agentwise_mean = []
    agentwise_std = []

    for i in range(n):

        print(f'Agent {i} - {protected_attributes[i]}')

        total_results = []

        for _ in range(n_times):

            joint_results = dict()

            for k in range(n):
                print(f'k = {k+1}/{n}')

                joint_results[k] = [list(), list()]

                for r in range(n_repeat):
                    print(f'Iteration {r+1}/{n_repeat}')

                    X_i, y_i = CS(X, y, b, protected_attributes[i],
                                random_seed=random_seed+100*(r+1), collaborators=[])
                    e_i = error_DP(X_i, y_i, protected_attributes[i], ground_truth_dp)            
                    
                    # choose k agents excluding i
                    possible_js = [j for j in range(n) if j != i]
                    js = rng.choice(possible_js, size=k, replace=False)

                    js_and_i = [i] + list(js)

                    X_tot = pd.DataFrame()
                    y_tot = pd.DataFrame()
                    for j in js_and_i:

                        collaborators = [protected_attributes[m] for m in js_and_i if m != j]
                        X_j, y_j = CS(X, y, b, protected_attributes[j],
                                    random_seed=random_seed+100*(r+1), collaborators=collaborators)

                        X_tot = pd.concat([X_tot, X_j], ignore_index=True)
                        y_tot = pd.concat([y_tot, y_j], ignore_index=True)
                    
                    e_i_aposteriori = error_DP(X_tot, y_tot, protected_attributes[i], ground_truth_dp)

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
    plt.savefig(f'results/GC/thm2_CS_all_seed{random_seed}_budget{b}.png')
