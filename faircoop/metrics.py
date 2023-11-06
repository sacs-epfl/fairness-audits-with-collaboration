import numpy as np


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

def demographic_parity_unbiased(features, labels, attr, all_probs, all_ys, other_attrs, protected_attributes, dataset_size: int):
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

def demographic_parity_error_unbiased(features, labels, attr, all_probs, all_ys, other_attrs, ground_truth_dp,
                                      all_attributes, dataset_size: int):
    if not other_attrs:
        raise RuntimeError("Cannot unbias with empty other_attrs!")
    else:
        dp_mean = demographic_parity_unbiased(features, labels, attr, all_probs, all_ys, other_attrs,
                                                      all_attributes, dataset_size)
        return np.abs(dp_mean - ground_truth_dp).item()

#### OLD VERSIONS ####
def demographic_parity_unbiased_old_v2(features, labels, attr, all_probs, all_ys, other_attrs, protected_attributes, dataset_size: int):
    all_attrs = other_attrs + [attr]
    n_attrs = len(all_attrs)
    n_subspaces = 2 ** n_attrs

    agent_ids = [(protected_attributes.index(a), a) for a in all_attrs]
    agent_ids.sort()
    agent_id_str = ''.join([str(elem) for elem, _ in agent_ids])
    own_index = agent_ids.index((protected_attributes.index(attr), attr))
    all_attrs = [a for _, a in agent_ids] # sorted accordings to ids

    binary_strings = [format(i, f'0{n_attrs}b') for i in range(n_subspaces)]

    prob_y_given_1 = 0; prob_y_given_0 = 0
    count_1 = 0; count_0 = 0
    p_subspaces = 0

    for binary_string in binary_strings:

        pairs = [(all_attrs[i], int(binary_string[i])) for i in range(n_attrs)]

        X_temp = features.copy()
        for a, val in pairs:
            X_temp = X_temp[X_temp[a] == val]
        y_tmp = labels.loc[X_temp.index]

        assert len(X_temp) == len(y_tmp), f'Length mismatch ==> X: {len(X_temp)}, y: {len(y_tmp)}'

        if binary_string[own_index] == '1':
            count_1 += 1
            prob_y_given_1 += y_tmp.mean().item() * all_probs[n_attrs][agent_id_str][binary_string]
            p_subspaces += all_probs[n_attrs][agent_id_str][binary_string]
        else:
            count_0 += 1
            prob_y_given_0 += y_tmp.mean().item() * all_probs[n_attrs][agent_id_str][binary_string]
            p_subspaces += all_probs[n_attrs][agent_id_str][binary_string]
        
    assert np.isclose(p_subspaces, 1), f'Probability over all subspaces is not 1: {p_subspaces}'
    assert count_1 == count_0, f'Mismatch ==> Count 1: {count_1}, Count 0: {count_0}'
    
    return np.abs(prob_y_given_1 - prob_y_given_0).item(), None
        
def demographic_parity_unbiased_old_v1(features, labels, attr, all_probs, all_ys, other_attrs, protected_attributes, dataset_size: int):
    n_attrs = len(other_attrs)
    n_subspaces = 2 ** n_attrs

    X_1 = features[features[attr] == 1]
    y_1 = labels.loc[X_1.index]

    X_0 = features[features[attr] == 0]
    y_0 = labels.loc[X_0.index]

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
        original_subspace_size = int(prob_subspace * dataset_size)  # there are 1000 data points in total
        sampled_subspace_size = len(subspaces_1[i][0])

        extended_binary_string_1 = '1' + binary_string
        extended_binary_string_0 = '0' + binary_string

        # print(f'Ex agent string {extended_agent_ids}, Ex binary string: {extended_binary_string_1}')

        ground_truth_y_given_1 = all_ys[n_attrs + 1][extended_agent_ids][extended_binary_string_1]
        ground_truth_y_given_0 = all_ys[n_attrs + 1][extended_agent_ids][extended_binary_string_0]

        dp_subspace_ground_truth = np.abs(ground_truth_y_given_1 - ground_truth_y_given_0).item()

        dp_std_squared += ((dp_subspaces[i] - dp_subspace_ground_truth) ** 2) * prob_subspace * (
                    1 - sampled_subspace_size / original_subspace_size)

    dp_std = np.sqrt(dp_std_squared).item()

    return dp_mean, dp_std

