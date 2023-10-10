import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from folktables import ACSDataSource, ACSPublicCoverage

from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
import pickle

LOAD_SAVED_MODEL = False
SAVE_MODEL = False

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

if __name__ == "__main__":
    random_seed = 42

    #########################################
    # Model training and evaluation
    X, y = load_dataset()

    # split X and y into train, test and audit splits of 45%, 5% and 50% respectively
    X_train, X_audit, y_train, y_audit = train_test_split(X, y, test_size=0.5, random_state=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)

    # print the number of samples in each split
    print("Number of samples in training split: {}".format(len(X_train)))
    print("Number of samples in test split: {}".format(len(X_test)))
    print("Number of samples in audit split: {}".format(len(X_audit)))

    if LOAD_SAVED_MODEL:
        model = pickle.load(open(f'./my_data/model_{random_seed}.pkl', 'rb'))
    else:
        model = GradientBoostingClassifier(loss='exponential', n_estimators=5, max_depth=5)
        model.fit(X_train, y_train.values.ravel())
    
    y_pred = model.predict(X_test)
    model_perf = accuracy_score(y_test, y_pred)
    print("Model accuracy: {}".format(model_perf))

    # Save the model
    if not LOAD_SAVED_MODEL and SAVE_MODEL:
        pickle.dump(model, open(f'./my_data/model_{random_seed}.pkl', 'wb'))

    protected_attributes = ['SEX','DIS','NATIVITY','DEAR',
            'DEYE','MIG','MIL','AGEP','DREM','MAR']
    
    # Calculate ground truth demographic parity for the model
    ground_truth_dp = dict()
    y_pred = model.predict(X)
    for attr in protected_attributes:
        dp_o = demographic_parity(X, y, attr)
        print("Demographic parity of the original dataset on {}: {:.3f}".format(attr, dp_o))
        
        dp = demographic_parity(X, y_pred, attr)
        ground_truth_dp[attr] = dp
        print("Demographic parity of the model on {}: {:.3f}".format(attr, dp), "\n---\n")

