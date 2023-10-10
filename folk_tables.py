import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from folktables import ACSDataSource, ACSPublicCoverage
import matplotlib.pyplot as plt

protected_attributes = ['SEX','DIS','NATIVITY','DEAR',
            'DEYE','MIG','MIL','AGEP','DREM','MAR']
    
def load_dataset():
    # load the pandas data frames
    features = pd.read_csv('./my_data/features.csv')
    label = pd.read_csv('./my_data/label.csv')
    return features, label

def demographic_parity(samples, y, attribute):
    # Calculate demographic parity for 'attribute'

    binary_attributes = ['SEX','DIS','NATIVITY','DEAR','DEYE']

    n = len(samples)

    if attribute in binary_attributes:
        # if the auditor doesn't test all subpopulations, we set that the demographic parity is null
        if not (0 < y[samples[attribute] == 1].sum().item() < n) or not (0 < y[samples[attribute] == 2].sum().item() < n):
            return 0

        prob_y_given_attribute_1 = y[samples[attribute] == 1].mean().item()  # P(y=1|attribute=1)
        prob_y_given_attribute_0 = y[samples[attribute] == 2].mean().item()  # P(y=1|attribute=0)

    elif attribute == 'MIL':
        prob_y_given_attribute_1 = y[samples[attribute].isin([1,4])].mean().item()
        prob_y_given_attribute_0 = y[samples[attribute].isin([0,2,3])].mean().item()

    elif attribute == 'MIG':
        prob_y_given_attribute_1 = y[samples[attribute] == 1].mean().item()
        prob_y_given_attribute_0 = y[samples[attribute].isin([0,2,3])].mean().item()
    
    elif attribute == 'AGEP':
        prob_y_given_attribute_1 = y[samples[attribute] > 25].mean().item()
        prob_y_given_attribute_0 = y[samples[attribute] <= 25].mean().item()
    
    elif attribute == 'MAR':
        prob_y_given_attribute_1 = y[samples[attribute] == 1].mean().item()
        prob_y_given_attribute_0 = y[samples[attribute].isin([2,3,4,5])].mean().item()

    elif attribute == 'DREM':
        prob_y_given_attribute_1 = y[samples[attribute] == 1].mean().item()
        prob_y_given_attribute_0 = y[samples[attribute].isin([0,2])].mean().item()

    else:
        raise ValueError('Attribute not supported')

    demographic_parity_attribute = abs(prob_y_given_attribute_1 - prob_y_given_attribute_0)
    return demographic_parity_attribute

def SU(X, y, n, attribute, random_seed=42):

    assert attribute in protected_attributes, "Attribute not supported"

    # Get index of attribute in protected_attributes
    index = protected_attributes.index(attribute)
    random_state = random_seed + index
    
    subset = X.sample(n=n, random_state=random_state)
    subset_y = y.loc[subset.index]

    return subset, subset_y

def error_DP(X, y, attribute, ground_truth_dp):
    return np.abs(demographic_parity(X, y, attribute) - ground_truth_dp[attribute])

if __name__ == "__main__":
    random_seed = 46

    #########################################
    # Calculate ground truth demographic parity
    X, y = load_dataset()

    ground_truth_dp = dict()
    for attr in protected_attributes:
        dp_o = demographic_parity(X, y, attr)
        print("Demographic parity of the original dataset on {}: {:.3f}".format(attr, dp_o))
        ground_truth_dp[attr] = dp_o
    
    #########################################
    # Audit the model

    strategies = [[0, SU]]
    budgets = list(range(10000, 100001, 30000))
    budgets = [100000*5]
    Nrepeat = 10
    results = dict()
    for k, attr in enumerate(protected_attributes):
        results[attr] = dict()
        for i, strat in strategies:
            results[attr][i] = {b:[] for b in budgets}
    
    print('================== Running single strategies ==================')
    # i is strategy index
    for i, strat in strategies:
        # k is agent index (i.e. also attr index)
        print(f'Running strategy {strat.__name__} {i+1}/{len(strategies)}')

        for k, attr in enumerate(protected_attributes):
            print(f'Running attribute {attr} {k+1}/{len(protected_attributes)}') 

            for b in budgets:
                print(f'Running budget {b}')

                for r in range(Nrepeat):
                    # Get a subset of the data
                    X_sampled, y_sampled = strat(X, y, b, attr, random_seed=random_seed*10*(r+1))

                    # Calculate error
                    error_dp = error_DP(X_sampled, y_sampled, attr, ground_truth_dp)

                    # Store error
                    results[attr][i][b].append([X_sampled, y_sampled, error_dp])

                    print(f'Error for {attr}, budget {b} is {error_dp}')

    print('================== Running joint strategies ==================')

    joint_results = dict()
    # initialize joint results
    for i, strat in strategies:
        joint_results[i] = dict()
        for j in range(len(protected_attributes)):
            for k in range(j, len(protected_attributes)):
                attr1 = protected_attributes[j]
                attr2 = protected_attributes[k]
                joint_results[i][(attr1, attr2)] = {b:[] for b in budgets}
                joint_results[i][(attr2, attr1)] = {b:[] for b in budgets}

    # evaluate results for the joint strategy
    for i, strat in strategies:
        print(f'Running strategy {strat.__name__} {i+1}/{len(strategies)}')

        for j in range(len(protected_attributes)):
            for k in range(j, len(protected_attributes)):
                
                print(f'Running attributes {j} and {k}')

                attr1 = protected_attributes[j]
                attr2 = protected_attributes[k]

                for b in budgets:
                    print(f'Running budget {b}')
                
                    for r in range(Nrepeat):
                        # print(f'Running repetition {r+1}/{Nrepet}')
                        X1, y1, e1 = results[attr1][i][b][r]
                        X2, y2, e2 = results[attr2][i][b][r]
                        
                        X_tot = pd.concat([X1, X2], ignore_index=True)
                        y_tot = pd.concat([y1, y2], ignore_index=True)
                        
                        e1_aposteriori = error_DP(X_tot, y_tot, attr1, ground_truth_dp)
                        e2_aposteriori = error_DP(X_tot, y_tot, attr2, ground_truth_dp)

                        joint_results[i][(attr1, attr2)][b].append(e1/e1_aposteriori)
                        joint_results[i][(attr2, attr1)][b].append(e2/e2_aposteriori)

    

    # plot the results

    # Create an empty 2D array to store the error values
    num_attributes = len(protected_attributes)
    error_matrix = np.zeros((num_attributes, num_attributes))
    budget = budgets[-1]

    # Fill the error_matrix with error values from your joint_results dictionary
    for i, strat in strategies:
        for j in range(num_attributes):
            for k in range(num_attributes):
                attr1 = protected_attributes[j]
                attr2 = protected_attributes[k]
                # Calculate the average error for the combination (attr1, attr2)
                avg_error = np.mean(joint_results[i][(attr1, attr2)][budget])
                error_matrix[j, k] = avg_error
            
    # Create a figure and axis
    fig, ax = plt.subplots()

    # Create the heatmap
    cax = ax.imshow(error_matrix, cmap='viridis', interpolation='nearest')

    # Add a colorbar
    cbar = fig.colorbar(cax)

    # Set axis labels and title
    ax.set_xticks(np.arange(num_attributes))
    ax.set_yticks(np.arange(num_attributes))
    ax.set_xticklabels(protected_attributes)
    ax.set_yticklabels(protected_attributes)
    ax.set_xlabel('Attribute 1')
    ax.set_ylabel('Attribute 2')
    ax.set_title('Error for Attribute Combinations')

    plt.xticks(np.arange(num_attributes), protected_attributes, rotation=45, ha='right')

    # Show the plot
    plt.savefig(f'results/heatmap_{random_seed}_{budgets[0]}.png')
