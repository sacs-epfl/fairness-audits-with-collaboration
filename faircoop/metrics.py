import numpy as np


def demographic_parity(features, labels, attribute) -> float:
    # Calculate demographic parity for 'attribute'
    prob_y_given_attribute_0 = labels[features[attribute].isin([0])].mean().item()
    prob_y_given_attribute_1 = labels[features[attribute].isin([1])].mean().item()
    demographic_parity_attribute = abs(prob_y_given_attribute_1 - prob_y_given_attribute_0)
    return demographic_parity_attribute


def demographic_parity_error(sampled_features, sampled_labels, attribute, ground_truth_dp):
    return np.abs(demographic_parity(sampled_features, sampled_labels, attribute) - ground_truth_dp)
