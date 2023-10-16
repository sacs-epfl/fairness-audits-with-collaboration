import numpy as np


def demographic_parity(features, labels, attribute) -> float:
    # Calculate demographic parity for 'attribute'
    prob_y_given_attribute_0 = labels[features[attribute].isin([0])].mean().item()
    prob_y_given_attribute_1 = labels[features[attribute].isin([1])].mean().item()
    demographic_parity_attribute = abs(prob_y_given_attribute_1 - prob_y_given_attribute_0)
    return demographic_parity_attribute


def demographic_parity_error(sampled_features, sampled_labels, attribute, ground_truth_dp):
    return np.abs(demographic_parity(sampled_features, sampled_labels, attribute) - ground_truth_dp)


def demographic_parity_unbiased(features, labels, attr, all_probs, all_ys, other_attrs, protected_attributes):
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
        original_subspace_size = int(prob_subspace * 100000)  # there are 1000 data points in total
        # TODO make sure the above number matches the number of rows in the dataset
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


def demographic_parity_error_unbiased(features, labels, attr, all_probs, all_ys, other_attrs, ground_truth_dp,
                                      all_attributes):
    if not other_attrs:
        return demographic_parity_error(features, labels, attr, ground_truth_dp), None
    else:
        dp_mean, dp_std = demographic_parity_unbiased(features, labels, attr, all_probs, all_ys, other_attrs,
                                                      all_attributes)
        return np.abs(dp_mean - ground_truth_dp).item()
