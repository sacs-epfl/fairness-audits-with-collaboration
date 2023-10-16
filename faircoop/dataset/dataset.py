import logging
from abc import abstractmethod, ABC
from typing import List, Dict

import pandas as pd

from faircoop.metrics import demographic_parity


class Dataset(ABC):

    def __init__(self):
        self.features: pd.array = None
        self.labels: pd.array = None
        self.protected_attributes: List[str] = []
        self.ground_truth_dps: Dict[str, float] = {}
        self.logger = logging.getLogger(self.__class__.__name__)

    def sample_selfish_uniform(self, budget: int, attribute: str, random_seed: int = 42):
        assert attribute in self.protected_attributes, "Attribute is not protected!"

        index = self.protected_attributes.index(attribute)
        random_state = random_seed + index

        subset = self.features.sample(n=budget, random_state=random_state)
        subset_y = self.labels.loc[subset.index]

        return subset, subset_y

    def sample_selfish_stratified(self, budget: int, attribute: str, random_seed: int = 42):
        assert attribute in self.protected_attributes, "Attribute is not protected!"

        index = self.protected_attributes.index(attribute)
        random_state = random_seed + index
        X_0 = self.features[self.features[attribute].isin([0])]
        X_1 = self.features[self.features[attribute].isin([1])]
        y_0 = self.labels.loc[X_0.index]
        y_1 = self.labels.loc[X_1.index]

        sub_n = budget // 2
        subset_1 = X_1.sample(n=sub_n, random_state=random_state)
        subset_1_y = y_1.loc[subset_1.index]
        subset_0 = X_0.sample(n=sub_n, random_state=random_state)
        subset_0_y = y_0.loc[subset_0.index]

        subset = pd.concat([subset_1, subset_0], ignore_index=True)
        subset_y = pd.concat([subset_1_y, subset_0_y], ignore_index=True)

        return subset, subset_y

    def sample_coordinated_stratified(self, collaborators: List, budget: int, attribute: str, random_seed: int = 42):
        all_attrs = collaborators + [attribute]
        n_attrs = len(collaborators) + 1

        n_subspaces = 2 ** n_attrs
        sub_n = budget // n_subspaces
        binary_strings = [format(i, f'0{n_attrs}b') for i in range(n_subspaces)]

        subspaces = []
        for binary_string in binary_strings:
            pairs = [(all_attrs[i], int(binary_string[i])) for i in range(n_attrs)]

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

            subset_i = X_i.sample(n=sub_n, random_state=random_seed)
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
