import numpy as np
import pandas as pd
import pickle

LOAD_DEMOGRAPHIC_PARITY = True
protected_attributes = ['SEX','DIS','NATIVITY','DEAR',
            'DEYE','MIG','MIL','AGEP','DREM','MAR']
    
def load_dataset():
    # load the pandas data frames
    features = pd.read_csv('./my_data/features.csv')
    label = pd.read_csv('./my_data/label.csv')
    return features, label

def demographic_parity(samples, y, attribute):
    # Calculate demographic parity for 'attribute'

    if attribute == 'AGEP':
        prob_y_given_attribute_1 = y[samples[attribute] > 25].mean().item()
        prob_y_given_attribute_0 = y[samples[attribute] <= 25].mean().item()
    
    elif attribute in protected_attributes:
        class_mapping = class_mappings(attribute)
        C_1 = class_mapping[1]
        C_0 = class_mapping[0]
        prob_y_given_attribute_1 = y[samples[attribute].isin(C_1)].mean().item()
        prob_y_given_attribute_0 = y[samples[attribute].isin(C_0)].mean().item()
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

def SS(X, y, n, attribute, random_seed=42):

    assert attribute in protected_attributes, "Attribute not supported"

    # Get index of attribute in protected_attributes
    index = protected_attributes.index(attribute)
    random_state = random_seed + index
    
    if attribute == 'AGEP':
        X_1 = X[X[attribute] > 25]
        X_0 = X[X[attribute] <= 25]
        y_1 = y.loc[X_1.index]
        y_0 = y.loc[X_0.index]
    else:
        class_mapping = class_mappings(attribute)
        C_1 = class_mapping[1]
        C_0 = class_mapping[0]
        X_1 = X[X[attribute].isin(C_1)]
        X_0 = X[X[attribute].isin(C_0)]
        y_1 = y.loc[X_1.index]
        y_0 = y.loc[X_0.index]
    
    sub_n = n//2
    # print(f'Len of X_1: {len(X_1)}')
    # print(f'Len of X_0: {len(X_0)}')
    subset_1 = X_1.sample(n=sub_n, random_state=random_state)
    subset_1_y = y_1.loc[subset_1.index]
    subset_0 = X_0.sample(n=sub_n, random_state=random_state)
    subset_0_y = y_0.loc[subset_0.index]
    
    subset = pd.concat([subset_1, subset_0], ignore_index=True)
    subset_y = pd.concat([subset_1_y, subset_0_y], ignore_index=True)

    return subset, subset_y

def class_mappings(attribute):
    binary_attributes = ['SEX','DIS','NATIVITY','DEAR','DEYE']
    class_mappings = dict()
    
    if attribute in binary_attributes:
        class_mappings[1] = [1]
        class_mappings[0] = [2]
    
    elif attribute == 'MIL':
        class_mappings[1] = [1,4]
        class_mappings[0] = [0,2,3]
    
    elif attribute == 'MIG':
        class_mappings[1] = [1]
        class_mappings[0] = [0,2,3]
    
    elif attribute == 'MAR':
        class_mappings[1] = [1]
        class_mappings[0] = [2,3,4,5]
    
    elif attribute == 'DREM':
        class_mappings[1] = [1]
        class_mappings[0] = [0,2]
        
    else:
        raise ValueError('Attribute not supported')

    return class_mappings

def error_DP(X, y, attribute, ground_truth_dp):
    return np.abs(demographic_parity(X, y, attribute) - ground_truth_dp[attribute])

if __name__ == "__main__":
    random_seed = 46

    #########################################
    # Calculate ground truth demographic parity
    X, y = load_dataset()

    if LOAD_DEMOGRAPHIC_PARITY:
        ground_truth_dp = pickle.load(open('./my_data/ground_truth_dp.pkl', 'rb'))
    else:
        ground_truth_dp = dict()
        for attr in protected_attributes:
            dp_o = demographic_parity(X, y, attr)
            print("Demographic parity of the original dataset on {}: {:.3f}".format(attr, dp_o))
            ground_truth_dp[attr] = dp_o
        
        pickle.dump(ground_truth_dp, open('./my_data/ground_truth_dp.pkl', 'wb'))
    

    #########################################
    # Audit the model

    # strategies = [[0, SU], [1, SS]]
    # strategies = [[1, SS]]
    strategies = [[0, SU]]
    # budgets = [100, 500, 1000, 2000]
    budgets = [100]
    n_repeat = 5
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

                for r in range(n_repeat):
                    # Get a subset of the data
                    X_sampled, y_sampled = strat(X, y, b, attr, random_seed=random_seed+100*(r+1))

                    # Calculate error
                    error_dp = error_DP(X_sampled, y_sampled, attr, ground_truth_dp)

                    # Store error
                    results[attr][i][b].append([X_sampled, y_sampled, error_dp])

                    print(f'Error for {attr}, budget {b} is {error_dp}')


    # Save results
    pickle.dump(results, open(f'./results/results_{strategies[0][0]}_{budgets[0]}.pkl', 'wb'))
    # pickle.dump(results, open(f'./results/results_{strategies[0][0]}_many_0.pkl', 'wb'))    
