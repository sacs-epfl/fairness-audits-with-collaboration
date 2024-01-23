import logging
from abc import abstractmethod, ABC
from itertools import combinations
from typing import List, Dict, Optional
import os
import pickle
import heapq
import cvxpy as cp

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

    # with independence assumption
    def compute_subspace_probabilities_independent(self):

        # Check if pickle file already exists
        all_probs_file = os.path.join("data", self.get_name(), "all_probs.pkl")
        all_ys_file = os.path.join("data", self.get_name(), "all_ys.pkl")

        if os.path.exists(all_probs_file) and os.path.exists(all_ys_file):
            self.logger.info("Loading subspace probabilities from pickle file...")
            self.subspace_features_probabilities = pickle.load(open(all_probs_file, "rb"))
            self.subspace_labels_probabilities = pickle.load(open(all_ys_file, "rb"))
            return

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

        # Save to pickle file
        self.logger.info("Saving subspace probabilities to pickle file...")
        pickle.dump(all_probs, open(all_probs_file, "wb"))
        pickle.dump(all_ys, open(all_ys_file, "wb"))

    # no independence assumption
    def compute_subspace_probabilities(self):

        # Check if pickle file already exists
        all_probs_file = os.path.join("data", self.get_name(), "all_probs.pkl")
        all_ys_file = os.path.join("data", self.get_name(), "all_ys.pkl")

        if os.path.exists(all_probs_file) and os.path.exists(all_ys_file):
            self.logger.info("Loading subspace probabilities from pickle file...")
            self.subspace_features_probabilities = pickle.load(open(all_probs_file, "rb"))
            self.subspace_labels_probabilities = pickle.load(open(all_ys_file, "rb"))
            return
        
        n = n = len(self.protected_attributes)

        all_probs = dict()
        all_ys = dict()

        for base_agent in range(n): # Base agent
            print(f'Working on base agent {base_agent}')
            base_attr = self.protected_attributes[base_agent]
            possible_collaborators = [i for i in range(n) if i != base_agent]
            all_probs[base_agent] = dict()
            all_ys[base_agent] = dict()

            # initialize the dict for k = 0
            all_probs[base_agent] = {0:{'':{'':{}}}}
            all_ys[base_agent] = {0:{'':{'':{}}}}

            X_0 = self.features.copy()
            X_0 = X_0[X_0[base_attr] == 0]
            y_0 = self.labels.loc[X_0.index]

            all_probs[base_agent][0][''][''][0] = 1
            all_ys[base_agent][0][''][''][0] = y_0.mean().item()

            X_1 = self.features.copy()
            X_1 = X_1[X_1[base_attr] == 1]
            y_1 = self.labels.loc[X_1.index]

            all_probs[base_agent][0][''][''][1] = 1
            all_ys[base_agent][0][''][''][1] = y_1.mean().item()

            for k in range(1, n): # Number of collaborators, 1 to n-1

                print(f'Working on k={k}')
                all_probs[base_agent][k] = dict()
                all_ys[base_agent][k] = dict()

                agent_combinations_list = list(combinations(possible_collaborators, k))

                for agent_combination in agent_combinations_list:
                    agent_comb_str = ''.join([str(elem) for elem in agent_combination])

                    all_probs[base_agent][k][agent_comb_str] = dict()
                    all_ys[base_agent][k][agent_comb_str] = dict()

                    total_strings = 2 ** (k)
                    binary_strings = [format(i, f'0{k}b') for i in range(total_strings)]

                    attrs = [self.protected_attributes[i] for i in agent_combination]
                    print(f'Working on {attrs}')
                    for binary_string in binary_strings:

                        all_probs[base_agent][k][agent_comb_str][binary_string] = dict()
                        all_ys[base_agent][k][agent_comb_str][binary_string] = dict()

                        pairs = [(attrs[i], int(binary_string[i])) for i in range(k)]

                        # Restore X_transformed that satisfies the binary string
                        X_temp = X_0.copy()
                        for attr, val in pairs:
                            X_temp = X_temp[X_temp[attr] == val]
                        y_tmp = y_0.loc[X_temp.index]
                        assert len(X_temp) == len(y_tmp), f'Length mismatch ==> X: {len(X_temp)}, y: {len(y_tmp)}'
                        
                        all_probs[base_agent][k][agent_comb_str][binary_string][0] = len(X_temp) / len(X_0)
                        all_ys[base_agent][k][agent_comb_str][binary_string][0] = y_tmp.mean().item()
                        
                        X_temp = X_1.copy()
                        for attr, val in pairs:
                            X_temp = X_temp[X_temp[attr] == val]
                        y_tmp = y_1.loc[X_temp.index]
                        assert len(X_temp) == len(y_tmp), f'Length mismatch ==> X: {len(X_temp)}, y: {len(y_tmp)}'
                        
                        all_probs[base_agent][k][agent_comb_str][binary_string][1] = len(X_temp) / len(X_1)
                        all_ys[base_agent][k][agent_comb_str][binary_string][1] = y_tmp.mean().item()
        
        self.subspace_features_probabilities = all_probs
        self.subspace_labels_probabilities = all_ys

        # Save to pickle file
        self.logger.info("Saving subspace probabilities to pickle file...")
        pickle.dump(all_probs, open(all_probs_file, "wb"))
        pickle.dump(all_ys, open(all_ys_file, "wb"))

    def sample_selfish_uniform(self, budget: int, attribute: str, random_seed: Optional[int] = None, oversample: bool = False):
        assert attribute in self.protected_attributes, "Attribute is not protected!"

        # checks if there are enough samples in the dataset
        budget = self._sample_subspace_with_limit([len(self.features)], [budget])[0]

        if random_seed is None:
            subset = self.features.sample(n=budget)
        else:
            subset = self.features.sample(n=budget, random_state=random_seed)
        subset_y = self.labels.loc[subset.index]

        return subset, subset_y

    def sample_selfish_stratified(self, budget: int, attribute: str, random_seed: Optional[int] = None, oversample: bool = False):
        assert attribute in self.protected_attributes, f"Attribute is not protected {self.protected_attributes}!"

        X_0 = self.features.copy()
        X_0 = X_0[X_0[attribute] == 0]
        
        X_1 = self.features.copy()
        X_1 = X_1[X_1[attribute] == 1]

        y_0 = self.labels.loc[X_0.index]
        y_1 = self.labels.loc[X_1.index]

        sub_n = budget // 2

        if oversample:
            sub_n1, sub_n0 = self._sample_subspace_with_limit([len(X_1), len(X_0)], [sub_n]*2)
        else:
            sub_n1 = sub_n0 = sub_n

        if random_seed is None:
            subset_1 = X_1.sample(n=sub_n1)
            subset_0 = X_0.sample(n=sub_n0)
        else:
            subset_1 = X_1.sample(n=sub_n1, random_state=random_seed)
            subset_0 = X_0.sample(n=sub_n0, random_state=random_seed)
        
        subset_1_y = y_1.loc[subset_1.index]
        subset_0_y = y_0.loc[subset_0.index]

        subset = pd.concat([subset_1, subset_0], ignore_index=True)
        subset_y = pd.concat([subset_1_y, subset_0_y], ignore_index=True)

        return subset, subset_y

    def sample_selfish_neyman(self, budget: int, attribute: str, random_seed: Optional[int] = None, oversample: bool = False):
        assert attribute in self.protected_attributes, f"Attribute is not protected {self.protected_attributes}!"

        X_0 = self.features.copy()
        X_0 = X_0[X_0[attribute] == 0]
        
        X_1 = self.features.copy()
        X_1 = X_1[X_1[attribute] == 1]

        y_0 = self.labels.loc[X_0.index]
        y_1 = self.labels.loc[X_1.index]

        base_agent = self.protected_attributes.index(attribute)
        p_positive_knowing_minority = self.subspace_labels_probabilities[base_agent][0][''][''][0]
        p_positive_knowing_majority = self.subspace_labels_probabilities[base_agent][0][''][''][1]
        sub_n0, sub_n1 = self._solve_no_collab(budget, p_positive_knowing_minority, p_positive_knowing_majority)

        if oversample:
            sub_n0, sub_n1 = self._sample_subspace_with_limit([len(X_0), len(X_1)], [sub_n0, sub_n1])

        if random_seed is None:
            subset_1 = X_1.sample(n=sub_n1)
            subset_0 = X_0.sample(n=sub_n0)
        else:
            subset_1 = X_1.sample(n=sub_n1, random_state=random_seed)
            subset_0 = X_0.sample(n=sub_n0, random_state=random_seed)
        
        subset_1_y = y_1.loc[subset_1.index]
        subset_0_y = y_0.loc[subset_0.index]

        subset = pd.concat([subset_1, subset_0], ignore_index=True)
        subset_y = pd.concat([subset_1_y, subset_0_y], ignore_index=True)

        return subset, subset_y

    def sample_coordinated_stratified(self, attributes: List[str], budget: int, random_seed: Optional[int] = None, oversample: bool = False):
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

        len_subspaces = [len(X_i) for X_i, _ in subspaces]  
        if oversample:
            sub_ns = self._sample_subspace_with_limit(len_subspaces, [sub_n]*n_subspaces)
        else:
            sub_ns = [sub_n] * n_subspaces

        # sample sub_n from each subspace
        subset = pd.DataFrame()
        subset_y = pd.DataFrame()
        for i, (X_i, y_i) in enumerate(subspaces):
            # check if subspace has sufficient samples
            if not oversample and len(X_i) < sub_ns[i]:
                raise ValueError('Subspace has insufficient samples')

            if random_seed is not None:
                subset_i = X_i.sample(n=sub_ns[i], random_state=random_seed)
            else:
                subset_i = X_i.sample(n=sub_ns[i])
            subset_i_y = y_i.loc[subset_i.index]
            
            subset = pd.concat([subset, subset_i], ignore_index=True)
            subset_y = pd.concat([subset_y, subset_i_y], ignore_index=True)

        return subset, subset_y

    def compute_ground_truth_dp(self):
        for protected_attribute in self.protected_attributes:
            dp: float = demographic_parity(self.features, self.labels, protected_attribute)
            self.ground_truth_dps[protected_attribute] = dp
            self.logger.debug("Ground truth DP of %s: %f" % (protected_attribute, dp))

    def _sample_subspace_with_limit(self, subspaces_sizes_avail: List[int], subspace_sizes_req: int):
        """
        PARAMETERS
            subspaces_sizes_avail: List[int]
                list of sizes of subspaces that are available for sampling
            n_each: int
                number of samples to be sampled from each subspace

        RETURNS
            list of sizes of subspaces that were sampled by oversampling equally from all subspaces
        """
        assert len(subspaces_sizes_avail) == len(subspace_sizes_req), 'Incorrect lists passed'
        
        logging.debug(f'subspaces_sizes_avail: {subspaces_sizes_avail}')
        # total number of subspaces
        n_total = len(subspaces_sizes_avail)
        # total number of samples to be sampled across all subspaces
        budget_total = sum(subspace_sizes_req)
        if budget_total > sum(subspaces_sizes_avail):
            raise ValueError('Not enough samples in dataset')
        
        n_rem = n_total
        logging.debug(f'subspace_sizes_req: {subspace_sizes_req}')

        subspaces_size_sampled = [-1] * n_total
        priority_queue = []
        for i in range(n_total):
            heapq.heappush(priority_queue, (subspaces_sizes_avail[i], i))
        
        yet_unsampled = set(range(n_total))

        while n_rem > 0:
            _, idx = heapq.heappop(priority_queue)
            n_rem -= 1
            if subspace_sizes_req[idx] > subspaces_sizes_avail[idx]:
                n_left = subspace_sizes_req[idx] - subspaces_sizes_avail[idx]
                n_left_each = n_left // n_rem
                extra = n_left % n_rem
                # set how much was sampled for idx
                subspaces_size_sampled[idx] = subspaces_sizes_avail[idx]
                # set how much is available for idx to 0
                subspaces_sizes_avail[idx] = 0
                # remove idx from yet_unsampled
                yet_unsampled.remove(idx)
                # distribute the remaining samples to other subspaces
                for i in yet_unsampled:
                    subspace_sizes_req[i] += n_left_each
                    if extra > 0:
                        subspace_sizes_req[i] += 1
                        extra -= 1
                assert extra == 0, 'Mistake in the algorithm'
            else:
                subspaces_size_sampled[idx] = subspace_sizes_req[idx]
                subspaces_sizes_avail[idx] -= subspace_sizes_req[idx]
                subspace_sizes_req[idx] = 0
                yet_unsampled.remove(idx)
        
        assert len(yet_unsampled) == 0, 'Mistake in the algorithm'
        assert -1 not in subspaces_size_sampled, 'Mistake in the algorithm'
        assert sum(subspaces_size_sampled) == budget_total, 'Mistake in the algorithm'
        
        logging.debug(f'subspaces_size_sampled: {subspaces_size_sampled}')
        return subspaces_size_sampled

    def _solve_no_collab(self, n, p_positive_knowing_minority, p_positive_knowing_majority):
        n_majority = cp.Variable(integer=True)
        n_minority = cp.Variable(integer=True)
        objective = (cp.sqrt((p_positive_knowing_majority * (1-p_positive_knowing_majority)))*cp.inv_pos(cp.sqrt(n_majority))
                        + cp.sqrt((p_positive_knowing_minority * (1-p_positive_knowing_minority)))*cp.inv_pos(cp.sqrt(n_minority))
                    )
        constraints = [n_minority >= 1,
                        n_majority >= 1,
                        n_minority+ n_majority == n
                    ]
        prob = cp.Problem(cp.Minimize(objective), constraints)
        # Résolution du problème
        prob.solve(solver=cp.GUROBI)  # ou tout autre solveur pris en charge par CVXPY
        return [int(n_minority.value), int(n_majority.value)]

    @abstractmethod
    def load_dataset(self):
        pass

    def get_name(self) -> str:
        return "unknown"
