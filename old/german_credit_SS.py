from aif360.sklearn.datasets import fetch_german
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import matplotlib.pyplot as plt

# protected_attributes = ['age', 'sex', 'marital_status', 'own_telephone', 'employment']
protected_attributes = ['age', 'sex', 'own_telephone', 'employment']

# load the dataset
def load_dataset():
    # load the dataset as a numpy array
    dataset = fetch_german()
    # split into inputs and outputs
    X, y = dataset.X, dataset.y

    # transform the age column into zero and one depending on the age being greater than 25
    X['age'] = X['age'].apply(lambda x: 0 if x <= 25 else 1)

    # transform the sex column into 0 or 1
    X['sex'] = X['sex'].apply(lambda x: 0 if x == 'female' else 1).astype(int)

    # transform the marital_status column into 0 or 1
    X['marital_status'] = X['marital_status'].apply(lambda x: 0 if x == 'single' else 1).astype(int)

    # transform the own_telephone column into 0 or 1
    X['own_telephone'] = X['own_telephone'].apply(lambda x: 0 if x == 'none' else 1).astype(int)

    # transform the employment column into 0 or 1
    X['employment'] = X['employment'].apply(lambda x: 1 if x == '4<=X<7' or x == '>=7' else 0).astype(int)    
    
    X = X.reset_index(drop=True)

    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    y = pd.Series(y)

    return X, y

def SS(X, y, n, attribute, random_seed=42):
    # X is dataframe and y is numpy array
    assert attribute in protected_attributes, "Attribute not supported"

    # Get index of attribute in protected_attributes
    index = protected_attributes.index(attribute)
    random_state = random_seed + index
    
    X_1 = X[X[attribute] == 1]
    y_1 = y.loc[X_1.index]

    X_0 = X[X[attribute] == 0]
    y_0 = y.loc[X_0.index]

    sub_n = n//2
    subset_1 = X_1.sample(n=sub_n, random_state=random_state)
    subset_y_1 = y_1[subset_1.index]
    subset_0 = X_0.sample(n=sub_n, random_state=random_state)
    subset_y_0 = y_0[subset_0.index]

    subset = pd.concat([subset_1, subset_0], ignore_index=True)
    subset_y = pd.concat([subset_y_1, subset_y_0], ignore_index=True)

    return subset, subset_y

def demographic_parity(X, y, attribute):

    prob_y_given_attribute_1 = y[X[attribute] == 1].mean().item()
    prob_y_given_attribute_0 = y[X[attribute] == 0].mean().item()

    demographic_parity_attribute = abs(prob_y_given_attribute_1 - prob_y_given_attribute_0)
    return demographic_parity_attribute

def error_DP(X, y, attribute, ground_truth_dp):
    return np.abs(demographic_parity(X, y, attribute) - ground_truth_dp[attribute])

if __name__ == "__main__":

    random_seed = 46

    X, y = load_dataset()

    ground_truth_dp = dict()
    for attr in protected_attributes:
        dp_o = demographic_parity(X, y, attr)
        print("Demographic parity of the original dataset on {}: {:.3f}".format(attr, dp_o))
        ground_truth_dp[attr] = dp_o

    # Audit the model
    b = 100
    n_repeat = 20
    strat = SS
    results = dict()

    for k, attr in enumerate(protected_attributes):
        results[attr] = []
    
    print('================== Running single strategies ==================')
    # i is strategy index
    
    for k, attr in enumerate(protected_attributes):
        print(f'Running attribute {attr} {k+1}/{len(protected_attributes)}') 

        for r in range(n_repeat):
            # Get a subset of the data
            X_sampled, y_sampled = strat(X, y, b, attr, random_seed=random_seed+100*(r+1))

            # Calculate error
            error_dp = error_DP(X_sampled, y_sampled, attr, ground_truth_dp)

            # Store error
            results[attr].append([X_sampled, y_sampled, error_dp])

            print(f'Error for {attr}, budget {b} is {error_dp}')

    print('================== Running aposteriori ==================')

    # Repeat 5 times to get randomness
    n_times = 5
    total_agents = len(protected_attributes)
    rng = np.random.default_rng(random_seed)

    agentwise_mean = []
    agentwise_std = []

    for a_i in range(total_agents):
        attr_i = protected_attributes[a_i] 
        print(f'Running agent {attr_i} {a_i+1}/{total_agents}')
        
        total_results = []

        for _ in range(n_times):
            
            print(f'Running repetition {_+1}/{n_times}')

            joint_results = dict()
            for k in range(0, total_agents):  # Choose the number of agents to collaborate

                print(f'Running {k+1}/{total_agents} agents')

                # select the agents to collaborate except the a_i
                possible_agents = list(range(total_agents))
                possible_agents.remove(a_i)
                selected_agents = rng.choice(possible_agents, size=k, replace=False)
                
                joint_results[k] = [list(), list()]
                
                for r in range(n_repeat):
                    X_i, y_i, e_i = results[attr_i][r]
                    X_tot = X_i
                    y_tot = y_i

                    for a_j in selected_agents:

                        attr_j = protected_attributes[a_j]
                        X_j, y_j, _ = results[attr_j][r]
                        
                        X_tot = pd.concat([X_tot, X_j], ignore_index=True)
                        y_tot = pd.concat([y_tot, y_j], ignore_index=True)

                    e_aposteriori = error_DP(X_tot, y_tot, attr_i, ground_truth_dp)
                    joint_results[k][0].append(e_i)
                    joint_results[k][1].append(e_aposteriori)
                    
                joint_results[k] = np.mean(joint_results[k][0])/np.mean(joint_results[k][1])

            total_results.append(list(joint_results.values()))

        agentwise_mean.append(np.mean(total_results, axis=0))
        agentwise_std.append(np.std(total_results, axis=0))
    
    agentwise_mean = np.array(agentwise_mean)
    agentwise_std = np.array(agentwise_std)

    # create a 5 x 2 subplot for 10 agents
    _, axs = plt.subplots(5, 1, figsize=(10, 20))
    axs = axs.flatten()
    for i in range(total_agents):
        axs[i].plot(range(1,total_agents+1), agentwise_mean[i], marker='o', color='blue')
        axs[i].fill_between(range(1,total_agents+1), agentwise_mean[i]-agentwise_std[i], agentwise_mean[i]+agentwise_std[i], color='blue', alpha=0.2)
        axs[i].set_xlabel(r'Number of agents $(K)$')
        axs[i].set_ylabel(r'$e(SS^1)/e(SS^K)$')
        axs[i].set_title(f'Agent {i} - {protected_attributes[i]}')
    plt.tight_layout()
    plt.savefig(f'results/GC/thm1_SS_all_seed{random_seed}_budget{b}.png')