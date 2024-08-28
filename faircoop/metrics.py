import numpy as np
import logging


def demographic_parity(features, labels, attribute) -> float:
    # Calculate demographic parity for 'attribute'
    X_0 = features.copy()
    X_0 = X_0[X_0[attribute] == 0]
    prob_y_given_attribute_0 = labels.loc[X_0.index].mean().item()
    
    X_1 = features.copy()
    X_1 = X_1[X_1[attribute] == 1]
    prob_y_given_attribute_1 = labels.loc[X_1.index].mean().item()
    
    demographic_parity_attribute = abs(prob_y_given_attribute_1 - prob_y_given_attribute_0)
    return demographic_parity_attribute

def demographic_parity_error(sampled_features, sampled_labels, attribute, ground_truth_dp):
    return np.abs(demographic_parity(sampled_features, sampled_labels, attribute) - ground_truth_dp)

# with independence assumption i.e. P(C1/C0) = P(C1); must use compute_subspace_probabilities_independent to generate meta files
def demographic_parity_unbiased_independent(features, labels, attr, all_probs, all_ys, other_attrs, protected_attributes, dataset_size: int):
    n_attrs = len(other_attrs)
    n_subspaces = 2 ** n_attrs

    X_1 = features[features[attr] == 1]
    y_1 = labels.loc[X_1.index]

    X_0 = features[features[attr] == 0]
    y_0 = labels.loc[X_0.index]

    subspaces_1 = []
    subspaces_0 = []

    agent_ids = [(protected_attributes.index(a), a) for a in other_attrs]
    agent_ids.sort()
    agent_id_str = ''.join([str(elem) for elem, _ in agent_ids])
    other_attrs = [a for _, a in agent_ids] # sorted accordings to ids

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

    prob_y_given_1 = 0; prob_y_given_0 = 0
    for i, binary_string in enumerate(binary_strings):
        
        prob_subspace = all_probs[n_attrs][agent_id_str][binary_string]

        prob_y_given_1 += subspaces_1[i][1].mean().item() * prob_subspace
        prob_y_given_0 += subspaces_0[i][1].mean().item() * prob_subspace

    dp_final = np.abs(prob_y_given_1 - prob_y_given_0).item()

    return dp_final

# without independence assumption i.e. P(C1/C0) != P(C1); uses compute_subspace_probabilities to generate meta files
def demographic_parity_unbiased(
        features, labels, attr, all_probs, all_ys, \
        other_attrs, protected_attributes, dataset_size: int):
    
    base_agent = protected_attributes.index(attr)
    n_attrs = len(other_attrs)
    n_subspaces = 2 ** n_attrs

    X_1 = features[features[attr] == 1]
    y_1 = labels.loc[X_1.index]

    X_0 = features[features[attr] == 0]
    y_0 = labels.loc[X_0.index]

    subspaces_1 = []
    subspaces_0 = []

    agent_ids = [(protected_attributes.index(a), a) for a in other_attrs]
    agent_ids.sort()
    agent_id_str = ''.join([str(elem) for elem, _ in agent_ids])
    other_attrs = [a for _, a in agent_ids] # sorted accordings to ids

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

    prob_y_given_1 = 0; prob_y_given_0 = 0
    for i, binary_string in enumerate(binary_strings):
        
        prob_subspace_0 = all_probs[base_agent][n_attrs][agent_id_str][binary_string][0]
        prob_subspace_1 = all_probs[base_agent][n_attrs][agent_id_str][binary_string][1]

        # check if subspace is empty
        if len(subspaces_1[i][1]) == 0:
            prob_y_given_1 += 0
            logging.debug(f'Empty subspace for {attr}=1, {other_attrs} = {binary_string}')
        else:
            prob_y_given_1 += subspaces_1[i][1].mean().item() * prob_subspace_1
        
        if len(subspaces_0[i][1]) == 0:
            prob_y_given_0 += 0
            logging.debug(f'Empty subspace for {attr}=0, {other_attrs} = {binary_string}')
        else:
            prob_y_given_0 += subspaces_0[i][1].mean().item() * prob_subspace_0

    dp_final = np.abs(prob_y_given_1 - prob_y_given_0).item()

    return dp_final

def demographic_parity_error_unbiased(features, labels, attr, all_probs, all_ys, other_attrs, ground_truth_dp,
                                      all_attributes, dataset_size: int):
    if not other_attrs:
        raise RuntimeError("Cannot unbias with empty other_attrs!")
    else:
        dp_mean = demographic_parity_unbiased(features, labels, attr, all_probs, all_ys, other_attrs,
                                                      all_attributes, dataset_size)
        return np.abs(dp_mean - ground_truth_dp).item()

