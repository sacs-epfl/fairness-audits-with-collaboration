import logging
from abc import abstractmethod, ABC
from itertools import combinations
from typing import List, Dict, Optional

import pandas as pd

from faircoop.metrics import demographic_parity


class Dataset(ABC):

    def __init__(self):
        self.features: pd.array = None
        self.labels: pd.array = None
        self.protected_attributes: List[str] = []
        self.ground_truth_dps: Dict[str, float] = {}
        self.logger = logging.getLogger(self.__class__.__name__)
        self.subspace_features_probabilities = None
        self.subspace_labels_probabilities = None

    def compute_subspace_probabilities(self):
        n = len(self.protected_attributes)

        all_probs = dict()
        all_ys = dict()

        for k in range(1, n + 1):

            self.logger.debug(f'Working on k={k}')
            all_probs[k] = dict()
            all_ys[k] = dict()

            agent_combinations_list = list(combinations(range(n), k))

            for agent_combination in agent_combinations_list:
                agent_comb_str = ''.join([str(elem) for elem in agent_combination])

                all_probs[k][agent_comb_str] = dict()
                all_ys[k][agent_comb_str] = dict()

                total_strings = 2 ** (k)
                binary_strings = [format(i, f'0{k}b') for i in range(total_strings)]

                attrs = [self.protected_attributes[i] for i in agent_combination]
                self.logger.debug(f'Working on {attrs}')
                for binary_string in binary_strings:

                    pairs = [(attrs[i], int(binary_string[i])) for i in range(k)]

                    # Restore X_transformed that satisfies the binary string
                    X_temp = self.features.copy()
                    for attr, val in pairs:
                        X_temp = X_temp[X_temp[attr] == val]
                    y_tmp = self.labels.loc[X_temp.index]
                    assert len(X_temp) == len(y_tmp), f'Length mismatch ==> X: {len(X_temp)}, y: {len(y_tmp)}'

                    all_probs[k][agent_comb_str][binary_string] = len(X_temp) / len(self.features)
                    all_ys[k][agent_comb_str][binary_string] = y_tmp.mean().item()

        self.subspace_features_probabilities = all_probs
        self.subspace_labels_probabilities = all_ys

    def sample_selfish_uniform(self, budget: int, attribute: str, random_seed: Optional[int] = None):
        assert attribute in self.protected_attributes, "Attribute is not protected!"

        if random_seed is None:
            subset = self.features.sample(n=budget)
        else:
            subset = self.features.sample(n=budget, random_state=random_seed)
        subset_y = self.labels.loc[subset.index]

        return subset, subset_y

    def sample_selfish_stratified(self, budget: int, attribute: str, random_seed: Optional[int] = None):
        assert attribute in self.protected_attributes, "Attribute is not protected!"

        X_0 = self.features[self.features[attribute].isin([0])]
        X_1 = self.features[self.features[attribute].isin([1])]
        y_0 = self.labels.loc[X_0.index]
        y_1 = self.labels.loc[X_1.index]

        sub_n = budget // 2
        if random_seed is None:
            subset_1 = X_1.sample(n=sub_n)
            subset_0 = X_0.sample(n=sub_n)
        else:
            subset_1 = X_1.sample(n=sub_n, random_state=random_seed)
            subset_0 = X_0.sample(n=sub_n, random_state=random_seed)
        subset_1_y = y_1.loc[subset_1.index]
        subset_0_y = y_0.loc[subset_0.index]

        subset = pd.concat([subset_1, subset_0], ignore_index=True)
        subset_y = pd.concat([subset_1_y, subset_0_y], ignore_index=True)

        return subset, subset_y

    def sample_coordinated_stratified(self, attributes: List[str], budget: int, random_seed: Optional[int] = None):
        n_attrs = len(attributes)

        n_subspaces = 2 ** n_attrs
        sub_n = budget // n_subspaces
        binary_strings = [format(i, f'0{n_attrs}b') for i in range(n_subspaces)]

        subspaces = []
        for binary_string in binary_strings:
            pairs = [(attributes[i], int(binary_string[i])) for i in range(n_attrs)]

            X_temp = self.features.copy()
            for attr, val in pairs:
                X_temp = X_temp[X_temp[attr] == val]

            y_tmp = self.labels.loc[X_temp.index]
            subspaces.append((X_temp, y_tmp))

        # sample sub_n from each subspace
        subset = pd.DataFrame()
        subset_y = pd.DataFrame()
        for X_i, y_i in subspaces:
            # check if subspace has sufficient samples
            if len(X_i) < sub_n:
                raise ValueError('Subspace has insufficient samples')

            if random_seed is not None:
                subset_i = X_i.sample(n=sub_n, random_state=random_seed)
            else:
                subset_i = X_i.sample(n=sub_n)
            subset_i_y = y_i.loc[subset_i.index]
            
            subset = pd.concat([subset, subset_i], ignore_index=True)
            subset_y = pd.concat([subset_y, subset_i_y], ignore_index=True)

        return subset, subset_y

    def compute_ground_truth_dp(self):
        for protected_attribute in self.protected_attributes:
            dp: float = demographic_parity(self.features, self.labels, protected_attribute)
            self.ground_truth_dps[protected_attribute] = dp
            self.logger.debug("Ground truth DP of %s: %f" % (protected_attribute, dp))

    @abstractmethod
    def load_dataset(self):
        pass

    def get_name(self) -> str:
        return "unknown"
