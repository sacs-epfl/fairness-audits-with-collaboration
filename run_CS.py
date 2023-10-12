from folk_tables import protected_attributes as pa, \
    load_dataset, class_mappings
import numpy as np
from run_CS_heatmap import error_DP_biased, CS
import pickle
import pandas as pd
import matplotlib.pyplot as plt

if __name__ == '__main__':
    random_seed = 46
    rng = np.random.default_rng(random_seed)
    
    X, y = load_dataset()

    # transform X
    X_transformed = X.copy()
    for attr in pa:
        if attr == 'AGEP':
            X_transformed[attr] = X_transformed[attr].apply(lambda x: 1 if x > 25 else 0)
        else:
            class_mapping = class_mappings(attr)
            C1 = class_mapping[1] # list of values that are mapped to 1
            C0 = class_mapping[0] # list of values that are mapped to 0
            X_transformed[attr] = X_transformed[attr].apply(lambda x: 1 if x in C1 else 0)

    groud_truth_dp = pickle.load(open('./my_data/ground_truth_dp.pkl', 'rb')) 

    protected_attributes = ['SEX', 'MAR', 'AGEP'] #, 'NATIVITY', 'DREM']
    budget = 100
    n_repeat = 3
    n = len(protected_attributes)
    
    agentwise_mean = []
    agentwise_std = []

    for i in range(n):

        print(f'Agent {i} - {protected_attributes[i]}')
        results_r = []

        for r in range(n_repeat):
            print(f'Iteration {r+1}/{n_repeat}')

            results_k = []

            for k in range(n):
                print(f'k = {k+1}/{n}')

                # choose k agents excluding i
                possible_js = [j for j in range(n) if j != i]
                js = rng.choice(possible_js, size=k, replace=False)

                js_and_i = [i] + list(js)

                X_tot = pd.DataFrame()
                y_tot = pd.DataFrame()
                for j in js_and_i:

                    collaborators = [protected_attributes[m] for m in js_and_i if m != j]
                    X_j, y_j = CS(X_transformed, y, budget, protected_attributes[j],
                                random_seed=random_seed, collaborators=collaborators)

                    X_tot = pd.concat([X_tot, X_j], ignore_index=True)
                    y_tot = pd.concat([y_tot, y_j], ignore_index=True)
                
                e_i_aposteriori = error_DP_biased(X_tot, y_tot, protected_attributes[i], groud_truth_dp)

                results_k.append(e_i_aposteriori)

            results_r.append(results_k)

        results_r = np.array(results_r)
        results_r_mean = np.mean(results_r, axis=0)
        results_r_std = np.std(results_r, axis=0)

        agentwise_mean.append(results_r_mean)
        agentwise_std.append(results_r_std)
    

    agentwise_mean = np.array(agentwise_mean)
    agentwise_std = np.array(agentwise_std)

    # create a 5 x 2 subplot for 10 agents
    _, axs = plt.subplots(5, 2, figsize=(10, 20))
    axs = axs.flatten()
    for i in range(n):
        axs[i].plot(range(1,n+1), agentwise_mean[i], marker='o', color='blue')
        axs[i].fill_between(range(1,n+1), agentwise_mean[i]-agentwise_std[i], agentwise_mean[i]+agentwise_std[i], color='blue', alpha=0.2)
        axs[i].set_xlabel(r'Number of agents $(K)$')
        axs[i].set_ylabel(r'$e(SU^1)/e(SU^K)$')
        axs[i].set_title(f'Agent {i} - {protected_attributes[i]}')
    plt.tight_layout()
    plt.savefig(f'results/plot_thm2_CS_all_seed{random_seed}_budget{budget}.png')
