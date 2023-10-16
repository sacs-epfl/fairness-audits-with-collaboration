from folk_tables import protected_attributes as pa, \
    load_dataset, class_mappings
import pickle
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def demographic_parity_unbiased(X, y, attr1, attr2, all_probs):

    all_attrs = [attr1, attr2]
    n_attrs = 2
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
    
    a1, a2 = pa.index(attr1), pa.index(attr2)
    agent_ids = f'{a1}{a2}'
    if agent_ids not in all_probs[2].keys():
        agent_ids = f'{a2}{a1}'

    prob_y_given_attribute_1 = 0
    prob_y_given_attribute_0 = 0
    for i, binary_string in enumerate(binary_strings):
        print(f'Agent string {agent_ids}, Binary string: {binary_string}')
        if binary_string[0] == '1':
            prob_y_given_attribute_1 += all_probs[2][agent_ids][binary_string] * subspaces[i][1].mean()
        elif binary_string[0] == '0':
            prob_y_given_attribute_0 += all_probs[2][agent_ids][binary_string] * subspaces[i][1].mean()

    dp_attr = np.abs(prob_y_given_attribute_1 - prob_y_given_attribute_0)
    return dp_attr.item()

def error_DP_unbiased(X, y, attr1, attr2, all_probs, ground_truth_dp):
    return np.abs(demographic_parity_unbiased(X, y, attr1, attr2, all_probs) - ground_truth_dp[attr1]).item()

def demographic_parity_biased(X, y, attribute):
    prob_y_given_attribute_1 = y[X[attribute] == 1].mean().item()
    prob_y_given_attribute_0 = y[X[attribute] == 0].mean().item()
    dp_attr = np.abs(prob_y_given_attribute_1 - prob_y_given_attribute_0)
    return dp_attr

def error_DP_biased(X, y, attribute, ground_truth_dp):
    return np.abs(demographic_parity_biased(X, y, attribute) - ground_truth_dp[attribute])

def CS(X, y, n, attribute, random_seed=42, collaborators=[]):

    all_attrs = collaborators + [attribute]
    n_attrs = len(collaborators) + 1
    # print(f'All attributes: {all_attrs}')

    n_subspaces = 2**n_attrs
    sub_n = n//n_subspaces
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
    subset = pd.DataFrame()
    subset_y = pd.DataFrame()
    for X_i, y_i in subspaces:
        
        # check if subspace has sufficient samples
        if len(X_i) < sub_n:
            raise ValueError('Subspace has insufficient samples')

        subset_i = X_i.sample(n=sub_n, random_state=random_seed)
        subset_i_y = y_i.loc[subset_i.index]
        subset = pd.concat([subset, subset_i], ignore_index=True)
        subset_y = pd.concat([subset_y, subset_i_y], ignore_index=True)

    return subset, subset_y

if __name__ == '__main__':

    random_seed = 55

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
    # all_probs = pickle.load(open('./my_data/all_probs_2.pkl', 'rb'))

    # Results for 2-way collaboration, n x n grid
    # protected_attributes = ['SEX', 'MIL', 'MIG', 'MAR', 'DREM']
    protected_attributes = ['SEX', 'MAR', 'AGEP', 'NATIVITY', 'DREM']
    # protected_attributes = ['SEX', 'MIL']
    # protected_attributes = ['SEX','DIS','NATIVITY','DEAR','DEYE']
    n = len(protected_attributes)
    budget = 500
    random_seed = 5
    n_repeat = 1

    # initialize n x n grid
    results = [[[] for _ in range(n)] for _ in range(n)]

    for r in range(n_repeat):
        
        print(f'Running repetition {r+1}/{n_repeat}')
        random_seed_i = random_seed + r
        random_seed_j = 10*random_seed + r

        for i in range(n):
            
            print(f'Running agent {i+1}/{n}')
            # SS for self
            X_i, y_i = CS(X_transformed, y, budget, protected_attributes[i], 
                        random_seed=random_seed_i, collaborators=[])
            e_i = error_DP_biased(X_i, y_i, protected_attributes[i], groud_truth_dp)

            results[i][i].append(e_i)

            for j in range(i+1, n):

                print(f'-- Running subagent {j+1}/{n}')

                X_i, y_i = CS(X_transformed, y, budget, protected_attributes[i], 
                            random_seed=random_seed_i, collaborators=[protected_attributes[j]])

                X_j, y_j = CS(X_transformed, y, budget, protected_attributes[j], 
                            random_seed=random_seed_j, collaborators=[protected_attributes[i]])
                
                X_tot = pd.concat([X_i, X_j], ignore_index=True)
                y_tot = pd.concat([y_i, y_j], ignore_index=True)

                e_i_aposteriori = error_DP_biased(X_tot, y_tot, protected_attributes[i], groud_truth_dp)
                e_j_aposteriori = error_DP_biased(X_tot, y_tot, protected_attributes[j], groud_truth_dp)
                # e_i_aposteriori = error_DP_unbiased(X_tot, y_tot, protected_attributes[i], protected_attributes[j], all_probs, groud_truth_dp)
                # e_j_aposteriori = error_DP_unbiased(X_tot, y_tot, protected_attributes[j], protected_attributes[i], all_probs, groud_truth_dp)

                results[i][j].append(e_i_aposteriori)
                results[j][i].append(e_j_aposteriori)

    # take average
    results = [[sum(r)/len(r) for r in row] for row in results]

    pickle.dump(results, open(f'results/new_matrices/errors_CS_seed{random_seed}_budget{budget}_repeat{n_repeat}.pkl', 'wb'))

    # divide by self
    gain_matrix = [[-1 for _ in range(n)] for _ in range(n)]
    for i in range(n):
        for j in range(n):
            gain_matrix[i][j] = results[i][i]/results[i][j]
    
    # save results
    pickle.dump(gain_matrix, open(f'results/new_matrices/gains_CS_seed{random_seed}_budget{budget}_repeat{n_repeat}.pkl', 'wb'))

    # plot the matrix
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the heatmap
    cax = ax.imshow(gain_matrix, cmap='viridis', interpolation='nearest')

    # Add a colorbar
    cbar = fig.colorbar(cax)

    # Set axis labels and title
    ax.set_xticks(np.arange(n))
    ax.set_yticks(np.arange(n))
    ax.set_xticklabels(protected_attributes)
    ax.set_yticklabels(protected_attributes)
    ax.set_xlabel('Attribute 1')
    ax.set_ylabel('Attribute 2')
    ax.set_title('Error for Attribute Combinations')

    plt.xticks(np.arange(n), protected_attributes, rotation=45, ha='right')

    # Show the plot
    plt.savefig(f'results/new_heatmap_CS_seed{random_seed}_budget{budget}_repeat{n_repeat}.png')

