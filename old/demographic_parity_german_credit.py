from aif360.sklearn.datasets import fetch_german

import numpy as np
import pandas as pd
import os
from sympy import *

from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, LabelEncoder, MinMaxScaler
from sklearn.metrics import accuracy_score
from sklearn.linear_model import LogisticRegression
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.naive_bayes import GaussianNB
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split

from matplotlib import pyplot as plt
import matplotlib as mpl

# load the dataset
def load_dataset():
    # load the dataset as a dataframe
    dataset = fetch_german()
    # split into inputs and outputs
    X, y = dataset.X, dataset.y

    # transform the age column into zero and one depending on the age being greater than 25
    X['age'] = X['age'].apply(lambda x: 0 if x <= 25 else 1)

    # transform the sex column into 0 or 1
    X['sex'] = X['sex'].apply(lambda x: 0 if x == 'female' else 1).astype(int)

    # select categorical features
    cat_ix = X.select_dtypes(include=['category']).columns
    num_ix = X.select_dtypes(include=['int64', 'float64']).columns
    # one hot encode cat features only
    # label encode the target variable to have the classes 0 and 1
    y = LabelEncoder().fit_transform(y)
    return X, y, cat_ix, num_ix

def demographic_parity(samples, y, attribute):
    # Calculate demographic parity for 'attribute'

    n = len(samples)
    # if the auditor doesn't test all subpopulations, we set that the demographic parity is null
    if not (0 < y[samples[attribute] == 1].sum() < n) or not (0 < y[samples[attribute] == 0].sum() < n):
        return 0

    prob_y_given_attribute_1 = y[samples[attribute] == 1].mean()  # P(y=1|attribute=1)
    prob_y_given_attribute_0 = y[samples[attribute] == 0].mean()  # P(y=1|attribute=0)

    demographic_parity_attribute = abs(prob_y_given_attribute_1 - prob_y_given_attribute_0)

    return demographic_parity_attribute

# define models to test
def get_models(model_name=None):
    """
    Parameters
    ----------
    model_name : str
        The name of the model to be used. If None, all models are returned.

    Returns
    -------
    models : list
        A list of sklearn models.
    names : list
        A list of the names of the models.
    """
    if model_name is not None:
        if model_name == 'LR':
            return [LogisticRegression(solver='liblinear')], [model_name]
        elif model_name == 'LDA':
            return [LinearDiscriminantAnalysis()], [model_name]
        elif model_name == 'NB':
            return [GaussianNB()], [model_name]
        elif model_name == 'GPC':
            return [GaussianProcessClassifier()], [model_name]
        elif model_name == 'SVM':
            return [SVC(gamma='scale')], [model_name]
        else:
            raise Exception("Unknown model name")
    else:

        models, names = list(), list()
        # LR
        models.append(LogisticRegression(solver='liblinear'))
        names.append('LR')
        # LDA
        models.append(LinearDiscriminantAnalysis())
        names.append('LDA')
        # NB
        models.append(GaussianNB())
        names.append('NB')
        # GPC
        models.append(GaussianProcessClassifier())
        names.append('GPC')
        # SVM
        models.append(SVC(gamma='scale'))
        names.append('SVM') 
        return models, names

def error_DP(samples, y, attribute, groud_truth_dp):
    # Calculate the error in demographic parity for 'attribute'
    return np.abs(groud_truth_dp - demographic_parity(samples, y, attribute))

def BlackBox(samples, ct, best_model):
    '''
    The black-box algorithm to audit
    samples : pandas.DataFrame
        The dataset.
    ct : sklearn.compose.ColumnTransformer
        The ColumnTransformer object used to transform the data.
    best_model : sklearn model
        The best model trained earlier.
    '''
    # transform using the same ColumnTransformer object used for the training data
    transformed_samples = ct.transform(samples)
    # predict
    yhat = best_model.predict(transformed_samples)

    # return yhat along with samples
    return samples, yhat

def RS(samples, n, black_box, attribute, random_seed=42):
    '''
    Random Sampling.

    Parameters
    ----------
    samples : pandas.DataFrame
        The dataset.
    n : int
        The number of samples to be generated.
    black_box : function
        The black-box algorithm to audit.
    attribute : str
        The attribute to be sampled.
    random_seed : int
        Random seed to be used for reproducibility.

    Attribute is not used in the method.
    '''

    random_state = random_seed if attribute == 'age' else random_seed*100
    subset = samples.sample(n=n, random_state=random_state)

    return black_box(subset)

def SSNC(samples, n, black_box, attribute, random_seed=42):
    '''
    Stratified Sampling Non-Collaborative.

    Parameters
    ----------
    samples : pandas.DataFrame
        The dataset.
    n : int
        The number of samples to be generated.
    black_box : function
        The black-box algorithm to audit.
    attribute : str
        The attribute to be sampled.
    random_seed : int
        Random seed to be used for reproducibility.
    '''
    random_state = random_seed if attribute == 'age' else random_seed*100

    if n % 2 != 0:
        print("WARNING: uneven n, one mising query")
    
    sub_n = n//2
    samples_0 = samples[samples[attribute] == 0]
    samples_1 = samples[samples[attribute] == 1]
    
    # Check if there are enough samples for each attribute
    if len(samples_0) < sub_n or len(samples_1) < sub_n:
        raise Exception("Not enough samples for each attribute")
    
    subset_0 = samples_0.sample(n=sub_n, random_state=random_state)
    subset_1 = samples_1.sample(n=sub_n, random_state=random_state)

    subset = pd.concat([subset_0, subset_1], ignore_index=True)

    return black_box(subset)

def SSC(samples, n, black_box, attribute, random_seed=42):
    '''
    Stratified Sampling Collaborative.

    Parameters
    ----------
    samples : pandas.DataFrame
        The dataset.
    n : int
        The number of samples to be generated.
    black_box : function
        The black-box algorithm to audit.
    attribute : str
        The attribute to be sampled. 
    '''
    random_state = random_seed if attribute == 'age' else random_seed*100
    a1 = 'age'; a2 = 'sex'

    if n % 2 != 0:
        print("WARNING: uneven n, one mising query")
    
    sub_n = n//4
    samples_0_0 = samples[(samples[a1] == 0) & (samples[a2] ==0)]
    samples_0_1 = samples[(samples[a1] == 0) & (samples[a2] ==1)]
    samples_1_0 = samples[(samples[a1] == 1) & (samples[a2] ==0)]
    samples_1_1 = samples[(samples[a1] == 1) & (samples[a2] ==1)]
    
    # Check if there are enough samples for each attribute
    if len(samples_0_0) < sub_n or len(samples_0_1) < sub_n or len(samples_1_0) < sub_n or len(samples_1_1) < sub_n:
        raise Exception("Not enough samples for each attribute")    
    
    subset_0_0 = samples_0_0.sample(n=sub_n, random_state=random_state)
    subset_0_1 = samples_0_1.sample(n=sub_n, random_state=random_state)
    subset_1_0 = samples_1_0.sample(n=sub_n, random_state=random_state)
    subset_1_1 = samples_1_1.sample(n=sub_n, random_state=random_state)

    subset = pd.concat([subset_0_0, subset_0_1, subset_1_0, subset_1_1], ignore_index=True)

    return black_box(subset)

def OSNC(samples, n, black_box, attribute, prob_dict, random_seed=42):
    """
    Optimal Sampling Non-Collaborative.

    Parameters
    ----------
    samples : pandas.DataFrame
        The dataset.
    n : int
        The number of samples to be generated.
    black_box : function
        The black-box algorithm to audit.
    attribute : str
        The attribute to be sampled.
    prob_dict : dict
        A dictionary containing the true input distribution of the attributes.
        Has the following structure:
        {
            'age': {
                'prob_y_given_age_0': prob_y_given_age_0,
                'prob_y_given_age_1': prob_y_given_age_1
            },
            .. and similarly for other attribute
        }
    random_seed : int
        Random seed to be used for reproducibility.

    
    Returns
    -------
    Output of the black-box algorithm on the optimal samples.

    """
    random_state = random_seed if attribute == 'age' else random_seed*100

    # P0, P1 = pYC00, pYC01 for C0
    # P0, P1 = pYC10, pYC11 for C1
    P0 = prob_dict[attribute][f'prob_y_given_{attribute}_0']
    P1 = prob_dict[attribute][f'prob_y_given_{attribute}_1']
    
    # Finding the optimal strategy by solving the Equation (?) using sympy. Is it always equivalent to proportionate stratified sampling ??
    p1, p0, m, a = symbols("p1 p0 n a")
    espilon = 2*sqrt((p1*(1-p1))/(a*m)) + 2*sqrt((p0*(1-p0))/((1-a)*m))
    evaluation = espilon.subs(m, n)
    evaluation = evaluation.subs(p0, P0)
    evaluation = evaluation.subs(p1, P1)
    derivative = evaluation.diff(a)
    s = solve(derivative, a)
    if s != []:
        alpha = float(s[0])
        # print(f"Optimal alpha for attribute {attribute}: {s[0]}")
        
        # Sample alpha samples from C0 and n-alpha samples from C1
        samples_0 = samples[samples[attribute] == 0]
        samples_1 = samples[samples[attribute] == 1]
        subset_0 = samples_0.sample(n=int(n*(1.0-alpha)), random_state=random_state)
        subset_1 = samples_1.sample(n=int(n*alpha), random_state=random_state)

        subset = pd.concat([subset_0, subset_1], ignore_index=True)

        return black_box(subset)
    else:
        raise Exception("No solution found for optimal sampling")

def OSC(samples, n, black_box, attribute, prob_dict, random_seed=42):
    """
    Optimal Sampling Collaborative.

    Parameters
    ----------
    samples : pandas.DataFrame
        The dataset.
    n : int
        The number of samples to be generated.
    black_box : function
        The black-box algorithm to audit.
    attribute : str
        The attribute to be sampled.
    prob_dict : dict
        A dictionary containing the true input distribution of the attributes.
        Has the following structure:
        {
            'age': {
                'prob_y_given_age_0': prob_y_given_age_0,
                'prob_y_given_age_1': prob_y_given_age_1
            },
            .. and similarly for other attribute
        }
    random_seed : int
        Random seed to be used for reproducibility.

    
    Returns
    -------
    Output of the black-box algorithm on the optimal samples.

    """
    assert(attribute in ['age', 'sex'])
    random_state = random_seed if attribute == 'age' else random_seed*100

    # P0, P1 = pYC00, pYC01 for C0
    # P2, P3 = pYC10, pYC11 for C1
    P0 = prob_dict[attribute][f'prob_y_given_{attribute}_0']
    P1 = prob_dict[attribute][f'prob_y_given_{attribute}_1']

    attribute2 = 'age' if attribute == 'sex' else 'sex'
    P2 = prob_dict[attribute2][f'prob_y_given_{attribute2}_0']
    P3 = prob_dict[attribute2][f'prob_y_given_{attribute2}_1']

    # Finding the optimal strategy by solving the Equation (?) using sympy. Is it always equivalent to proportionate stratified sampling ??
    p1, p0, m, a = symbols("p1 p0 m a")
    espilon = 2*sqrt((p1*(1-p1))/(a*m)) + 2*sqrt((p0*(1-p0))/((1-a)*m))
    evaluation = espilon.subs(m, n)
    evaluation = evaluation.subs(p0, P0)
    evaluation = evaluation.subs(p1, P1)
    derivative = evaluation.diff(a)
    s = solve(derivative, a)
    if s != []:
        alpha = float(s[0])
        # print(f"Optimal alpha for attribute {attribute}: {s[0]}")

        # Solve for the other attribute
        p1, p0, m, a = symbols("p1 p0 m a")
        espilon = 2*sqrt((p1*(1-p1))/(a*m)) + 2*sqrt((p0*(1-p0))/((1-a)*m))
        evaluation = espilon.subs(m, n)
        evaluation = evaluation.subs(p0, P2)
        evaluation = evaluation.subs(p1, P3)
        derivative = evaluation.diff(a)
        s = solve(derivative, a)
        if s != []:
            beta = float(s[0])
            # print(f"Optimal alpha for attribute {attribute2}: {s[0]}")
        
            samples_1_1 = samples[(samples[attribute] == 1) & (samples[attribute2] == 1)]
            samples_1_0 = samples[(samples[attribute] == 1) & (samples[attribute2] == 0)]
            samples_0_1 = samples[(samples[attribute] == 0) & (samples[attribute2] == 1)]
            samples_0_0 = samples[(samples[attribute] == 0) & (samples[attribute2] == 0)]
            
            # Check if there are enough samples for each attribute
            if len(samples_0_0) < int((1-beta)*(1-alpha)*n):
                raise Exception(f"Not enough samples for the combination: required {int((1-beta)*(1-alpha)*n)}, having {len(samples_0_0)}")
            elif len(samples_0_1) < int(beta*(1-alpha)*n):
                raise Exception(f"Not enough samples for the combination: required {int(beta*(1-alpha)*n)}, having {len(samples_0_1)}")
            elif len(samples_1_0) < int((1-beta)*alpha*n):
                raise Exception(f"Not enough samples for the combination: required {int((1-beta)*alpha*n)}, having {len(samples_1_0)}")
            elif len(samples_1_1) < int(beta*alpha*n):
                raise Exception(f"Not enough samples for the combination: required {int(beta*alpha*n)}, having {len(samples_1_1)}")
            
            subset_1_1 = samples_1_1.sample(n=int(beta*alpha*n), random_state=random_state)
            subset_1_0 = samples_1_0.sample(n=int((1-beta)*alpha*n), random_state=random_state)
            subset_0_1 = samples_0_1.sample(n=int(beta*(1-alpha)*n), random_state=random_state)
            subset_0_0 = samples_0_0.sample(n=int((1-beta)*(1-alpha)*n), random_state=random_state)

            subset = pd.concat([subset_0_0, subset_0_1, subset_1_0, subset_1_1], ignore_index=True)

            return black_box(subset)
        
        else:
            raise Exception("No solution found for optimal sampling beta")
        
    else:
        raise Exception("No solution found for optimal sampling alpha")

if __name__ == "__main__":
    random_seed = 90
    exp_id = np.random.randint(100000)
    print("exp_id:", exp_id)
    np.random.seed(random_seed)

    X, y, cat_ix, num_ix = load_dataset() # X is pandas.DataFrame, y is numpy.ndarray

    #########################################
    # Calculate the probabilities
    prob_age_1 = X['age'].mean()  # P(age = 1)
    prob_age_0 = 1 - prob_age_1   # P(age = 0)

    prob_sex_1 = X['sex'].mean()  # P(sex = 1)
    prob_sex_0 = 1 - prob_sex_1   # P(sex = 0)

    print("P(age = 1):", prob_age_1)
    print("P(age = 0):", prob_age_0)
    print("P(sex = 1):", prob_sex_1)
    print("P(sex = 0):", prob_sex_0)

    # Calculate the conditional probabilities
    prob_y_given_age_0 = y[X['age'] == 0].mean()  # P(y=1|age=0)
    prob_y_given_age_1 = y[X['age'] == 1].mean()  # P(y=1|age=1)

    prob_y_given_sex_0 = y[X['sex'] == 0].mean()  # P(y=1|sex=0)
    prob_y_given_sex_1 = y[X['sex'] == 1].mean()  # P(y=1|sex=1)

    print("P(y=1|age=0):", prob_y_given_age_0)
    print("P(y=1|age=1):", prob_y_given_age_1)
    print("P(y=1|sex=0):", prob_y_given_sex_0)
    print("P(y=1|sex=1):", prob_y_given_sex_1)

    prob_dict = {
        'age': {
            'prob_y_given_age_0': prob_y_given_age_0,
            'prob_y_given_age_1': prob_y_given_age_1
        },
        'sex': {
            'prob_y_given_sex_0': prob_y_given_sex_0,
            'prob_y_given_sex_1': prob_y_given_sex_1
        }
    }

    # Calculate the joint probabilities
    prob_y_given_sex_1_and_age_1 = y[(X['sex'] == 1) & (X['age'] == 1)].mean()  # P(y=1|sex=1, age=1)
    prob_y_given_sex_1_and_age_0 = y[(X['sex'] == 1) & (X['age'] == 0)].mean()  # P(y=1|sex=1, age=0)
    prob_y_given_sex_0_and_age_1 = y[(X['sex'] == 0) & (X['age'] == 1)].mean()  # P(y=1|sex=0, age=1)
    prob_y_given_sex_0_and_age_0 = y[(X['sex'] == 0) & (X['age'] == 0)].mean()  # P(y=1|sex=0, age=0)

    print("P(y=1|sex=1, age=1):", prob_y_given_sex_1_and_age_1)
    print("P(y=1|sex=1, age=0):", prob_y_given_sex_1_and_age_0)
    print("P(y=1|sex=0, age=1):", prob_y_given_sex_0_and_age_1)
    print("P(y=1|sex=0, age=0):", prob_y_given_sex_0_and_age_0)

    # Demographic parity in the original dataset
    demographic_parity_age = demographic_parity(X, y, 'age')
    demographic_parity_sex = demographic_parity(X, y, 'sex')

    print("Demographic Parity for 'sex' in the dataset:", demographic_parity_sex)
    print("Demographic Parity for 'age' in the dataset:", demographic_parity_age)

    #########################################
    # split X and y into train, test and audit splits of 45%, 5% and 50% respectively
    X_train, X_audit, y_train, y_audit = train_test_split(X, y, test_size=0.5, random_state=random_seed)
    X_train, X_test, y_train, y_test = train_test_split(X_train, y_train, test_size=0.1, random_state=random_seed)

    # define the preprocessing
    ct = ColumnTransformer([('c',OneHotEncoder(),cat_ix), ('n',MinMaxScaler(),num_ix)])
    X_train = ct.fit_transform(X_train)
    X_test = ct.transform(X_test)

    # train different models and keep the best
    models, names = get_models('LR')
    best_acc = 0.0; best_model = None; best_model_name = None
    for i in range(len(models)):

        model = models[i]
        model.fit(X_train, y_train)

        # evaluate the model
        y_pred = model.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        print(f"{names[i]} Accuracy: {accuracy}")

        if accuracy > best_acc:
            best_acc = accuracy
            best_model = model
            best_model_name = names[i]

    print(f"Best model: {best_model_name} with accuracy: {best_acc}")

    # Calculate the ground truth demographic parity of the best model
    # This should be done on all data (train + test+ audit)
    X_temp = ct.transform(X)
    y_pred = best_model.predict(X_temp)

    demographic_parity_age_of_model = demographic_parity(X, y_pred, 'age')
    print("Demographic Parity for 'age' of the model:", demographic_parity_age_of_model)

    demographic_parity_sex_of_model = demographic_parity(X, y_pred, 'sex')
    print("Demographic Parity for 'sex' of the model:", demographic_parity_sex_of_model)

    #########################################
    # Audit the model

    B = [r*10 for r in range(3, 16, 3)]
    Nrepet= 100 ## number of repetitions per sample level

    # create a black-box function that runs the best model on the audit data
    black_box = lambda samples: BlackBox(samples, ct, best_model)

    strategies = [[0, RS], [1, SSNC], [2, SSC], [3, OSNC], [4, OSC]]

    res0 = [ [ [] for _ in B] for _ in strategies]
    res1 = [ [ [] for _ in B] for _ in strategies]

    # Running single strategies
    for i, strat in strategies:
        for b in range(len(B)):
            print(f'Running strategy {i} {strategies[i][1].__name__} for budget {B[b]}')
            budget = int(B[b])
            cur_res0 = []
            cur_res1 = []
            for k in range(Nrepet):
                # print(f'Running repetition {k+1}/{Nrepet}')
                if strat.__name__ == 'OSNC' or strat.__name__ == 'OSC':
                    # OSNC needs the prob_dict
                    s0, y0 = strat(X_audit, budget, black_box, 'age', prob_dict, random_seed=random_seed+k)
                    s1, y1 = strat(X_audit, budget, black_box, 'sex', prob_dict, random_seed=random_seed+k)
                else:
                    s0, y0 = strat(X_audit, budget, black_box, 'age', random_seed=random_seed+k)
                    s1, y1 = strat(X_audit, budget, black_box, 'sex', random_seed=random_seed+k)
                cur_res0.append([s0, y0, error_DP(s0, y0, 'age', demographic_parity_age_of_model)])
                cur_res1.append([s1, y1, error_DP(s1, y1, 'sex', demographic_parity_sex_of_model)])

            res0[i][b] = cur_res0
            res1[i][b] = cur_res1

    res = [ [ [ [] for _ in B] for _ in strategies] for _ in strategies]
    print('Running joint strategy')
    for i, strati in strategies:
        for j, stratj in strategies:
            # print(f'Running strategy {i} and {j}')
            for b in range(len(B)):
                # print(f'Running budget {B[b]}')
                budget = int(B[b])
                for r in range(Nrepet):
                    # print(f'Running repetition {r+1}/{Nrepet}')
                    s0, y0, e0 = res0[i][b][r]
                    s1, y1, e1 = res1[j][b][r]
                    s_tot = pd.concat([s0, s1], ignore_index=True)
                    y_tot = np.concatenate([y0, y1], axis=0)
                    res[i][j][b].append([
                        e0, # old error for age 
                        e1, # old error for sex
                        error_DP(s_tot, y_tot, 'age', demographic_parity_age_of_model), # new error for age
                        error_DP(s_tot, y_tot, 'sex', demographic_parity_sex_of_model) # new error for sex
                        ])
    
    # plot the results
    
    plt.gcf().clear()
    mpl.rcParams["axes.spines.right"] = False
    mpl.rcParams["axes.spines.top"] = False
    res= np.array(res)

    m = len(strategies)

    fig, axs = plt.subplots(m+1, m+1, sharex=True, sharey=True, figsize=(20, 20))
    labels = [r'$\epsilon(DP_0)$ w/o $A_1$', r'$\epsilon(DP_1)$ w/o $A_0$', r'$\epsilon(DP_0)$ w $A_1$', r'$\epsilon(DP_1)$ w $A_0$']
    strat_names = ['no sampling']+[ s.__name__ for _, s in strategies]
    lines = ['--', '-']
    colors = ['red', 'blue']
    alphas = [0.3, 1]

    for i in range(m):
        axs[i+1][0].set_ylabel(strat_names[i+1])
        axs[-1][i+1].set_xlabel(strat_names[i+1])

        # Without sharing data
        k = 0
        to_plot0 = np.mean(res[i][i].T, axis = 1)
        axs[i+1][0].plot(B, to_plot0[k], linestyle = lines[k//2], color = colors[k%2], label=labels[k])

        k = 1
        to_plot1 = np.mean(res[i][i].T, axis = 1)
        axs[0][i+1].plot(B, to_plot1[k], linestyle = lines[k//2], color = colors[k%2], label=labels[k])


        # Sharing data for each (strat, start')
        for j in range(m):
            to_plot = np.mean(res[i][j].T, axis = 1)
            
            for k in range(4):
                axs[i+1][j+1].plot(B, to_plot[k], linestyle = lines[k//2], alpha = alphas[k//2], color = colors[k%2], label=labels[k])

    axs[0][0].set_ylabel(strat_names[0])
    axs[-1][0].set_xlabel(strat_names[0])


    handles, labels = axs[-1][-1].get_legend_handles_labels()
    lgd = axs[-1][0].legend(handles, labels, loc='center', bbox_to_anchor=(0.2,-0.4),  ncol=5)
    
    # Check if results directory exists or else create one
    if not os.path.exists('./results'):
        os.makedirs('./results')

    plt.savefig(f'./results/results_{random_seed}.pdf', bbox_extra_artists=(lgd,), bbox_inches='tight')