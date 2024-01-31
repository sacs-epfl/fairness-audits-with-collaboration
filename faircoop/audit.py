import logging
from typing import List, Tuple

import pandas as pd

from faircoop.metrics import demographic_parity_error, demographic_parity_error_unbiased


class Audit:

    def __init__(self, args, dataset):
        self.args = args
        self.agentwise_used_seeds = []  # to track the seed that is finally used, varies agentwise
        self.dataset = dataset
        self.results: List = []
        self.logger = logging.getLogger(self.__class__.__name__)

        if not self.args.attributes_to_audit:
            self.args.attributes_to_audit = self.dataset.protected_attributes

        self.num_agents: int = len(self.args.attributes_to_audit)

        if self.args.unbias_mean and self.args.collaboration != "apriori":
            raise RuntimeError("Unbiasing only supported for apriori collaboration!")
        
        if self.args.unbias_mean and self.args.collaboration == "apriori" and self.args.sample == "uniform":
            raise RuntimeError("Unbiasing not supported for apriori collaboration with uniform sampling!")

    def _distribute_values(self, input_list: List, num_agents: int) -> List[List]:
        # Distribute the values to the agents
        distributed_values = [[0 for _ in range(len(input_list))] for _ in range(num_agents)]
        for i, value in enumerate(input_list):
            equal_share = value // num_agents
            
            for k in range(num_agents):
                distributed_values[k][i] += equal_share
            
            remainder = value % num_agents
            for k in range(remainder):
                distributed_values[k][i] += 1

        assert sum([sum(d) for d in distributed_values]) == sum(input_list)

        return distributed_values

    def run(self):
        self.logger.info("Running audit with %s sampling and collaboration mode %s and unbiasing %s: (seed: %s)",
                         self.args.sample, self.args.collaboration, self.args.unbias_mean, self.args.seed)

        queries_per_agent: List[Tuple[List, List]] = []
        collab_attributes = self.args.attributes_to_audit

        if self.args.sample == "neyman" and self.args.collaboration == "apriori":
            if not self.dataset.is_solved():
                self.dataset.solve_collab(self.args.attributes_to_audit, self.args.budget*self.num_agents)
            
            subspace_wise_budgets = self.dataset.subspace_wise_budgets
            self.logger.debug("Subspace-wise budgets: %s %d", subspace_wise_budgets, sum(subspace_wise_budgets))
            agentwise_subspace_budgets = self._distribute_values(subspace_wise_budgets, self.num_agents)

        for agent, attribute in enumerate(self.args.attributes_to_audit):
            self.logger.info("Agent %d auditing attribute %s of black box...", agent, attribute)
            
            # Set the seed depending on the agent
            random_seed = self.args.seed + (agent+1) * 10000 if self.args.seed is not None else self.args.seed
            self.agentwise_used_seeds.append(random_seed)
            
            if self.args.sample == "uniform":
                if self.args.collaboration in ["none", "aposteriori", "apriori"]:
                    x_sampled, y_sampled = self.dataset.sample_selfish_uniform(
                        self.args.budget, attribute, random_seed=random_seed, oversample=self.args.oversample)
                else:
                    raise RuntimeError(f"Sample strategy {self.args.sample} with {self.args.collaboration} not supported!")
                
                queries_per_agent.append((x_sampled, y_sampled))
            
            elif self.args.sample == "neyman":
                if self.args.collaboration in ["none", "aposteriori"]:
                    x_sampled, y_sampled = self.dataset.sample_selfish_neyman(
                        self.args.budget, attribute, random_seed=random_seed, oversample=self.args.oversample)
                elif self.args.collaboration == "apriori":
                    x_sampled, y_sampled = self.dataset.sample_coordinated_neyman(
                        collab_attributes, agentwise_subspace_budgets[agent], random_seed=random_seed)
                
                queries_per_agent.append((x_sampled, y_sampled))
            
            elif self.args.sample == "stratified":
                if self.args.collaboration in ["none", "aposteriori"]:
                    x_sampled, y_sampled = self.dataset.sample_selfish_stratified(
                        self.args.budget, attribute, random_seed, oversample=self.args.oversample)
                elif self.args.collaboration == "apriori":
                    x_sampled, y_sampled = self.dataset.sample_coordinated_stratified(
                        collab_attributes, self.args.budget, random_seed, oversample=self.args.oversample)
                                
                queries_per_agent.append((x_sampled, y_sampled))

        # Combine the queries of collaborating agents and compute the DPs
        self.logger.info("Sampling done. Computing DP error...")

        if self.args.collaboration == "none":
            # Compute the DP error per agent
            for agent, sampled in enumerate(queries_per_agent):
                x_sampled, y_sampled = sampled
                used_seed = self.agentwise_used_seeds[agent]
                attribute = self.args.attributes_to_audit[agent]
                dp_error = demographic_parity_error(
                    x_sampled, y_sampled, attribute, self.dataset.ground_truth_dps[attribute])
                self.results.append((used_seed if used_seed is not None else -1, self.args.budget, agent,
                                     attribute, dp_error))
        elif self.args.collaboration in ["aposteriori", "apriori"]:
            total_strata_allocations = {}
            for i in range(2 ** len(self.dataset.protected_attributes)):
                total_strata_allocations[i] = 0

            for agent_index in range(len(queries_per_agent)):
                # print("Agent %d allocation:" % agent_index)
                strata_allocations = {}
                for i in range(2 ** len(self.dataset.protected_attributes)):
                    strata_allocations[i] = 0

                for idx, query in queries_per_agent[agent_index][0].iterrows():
                    bitset = []
                    for attribute in self.dataset.protected_attributes:
                        bitset.append(getattr(query, attribute))
                    bit_string = ''.join(str(bit) for bit in bitset)
                    strata_allocations[int(bit_string, 2)] += 1
                    total_strata_allocations[int(bit_string, 2)] += 1

                # for i in range(2 ** len(self.dataset.protected_attributes)):
                #     print("Strata %d: %d" % (i, strata_allocations[i]))

            print("Total allocation:")
            for i in range(2 ** len(self.dataset.protected_attributes)):
                print("%s,%s,%d,%d" % (self.args.sample, self.args.collaboration, i, total_strata_allocations[i]))

            # Combine all queries and then compute the DP error per agent
            x_all = pd.concat([x_agent for x_agent, _ in queries_per_agent], ignore_index=True)
            y_all = pd.concat([y_agent for _, y_agent in queries_per_agent], ignore_index=True)
            for agent, sampled in enumerate(queries_per_agent):
                used_seed = self.agentwise_used_seeds[agent]
                attribute = self.args.attributes_to_audit[agent]

                if self.args.unbias_mean:
                    other_attributes = [attr for attr in self.args.attributes_to_audit if attr != attribute]
                    dp_error = demographic_parity_error_unbiased(
                        x_all, y_all, attribute, self.dataset.subspace_features_probabilities,
                        self.dataset.subspace_labels_probabilities, other_attributes,
                        self.dataset.ground_truth_dps[attribute], self.dataset.protected_attributes,
                        len(self.dataset.features))
                else:
                    dp_error = demographic_parity_error(x_all, y_all, attribute, self.dataset.ground_truth_dps[attribute])
                self.results.append((used_seed if used_seed is not None else -1, self.args.budget, agent,
                                     attribute, dp_error))
        else:
            raise RuntimeError("Collaboration strategy %s not implemented!", self.args.collaboration)
