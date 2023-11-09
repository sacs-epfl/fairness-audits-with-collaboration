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

        if args.collaboration == "apriori" and args.sample != "stratified":
            self.logger.info("Setting sampling method to stratified for a posteriori collaboration")
            self.args.sample = "stratified"

    def run(self):
        self.logger.info("Running audit with %s sampling and collaboration mode %s and unbiasing %s: (seed: %s)",
                         self.args.sample, self.args.collaboration, self.args.unbias_mean, self.args.seed)

        queries_per_agent: List[Tuple[List, List]] = []
        collab_attributes = self.args.attributes_to_audit
        for agent, attribute in enumerate(self.args.attributes_to_audit):
            self.logger.info("Agent %d auditing attribute %s of black box...", agent, attribute)
            
            # Set the seed depending on the agent
            random_seed = self.args.seed + (agent+1) * 100000 if self.args.seed is not None else self.args.seed
            self.agentwise_used_seeds.append(random_seed)
            
            if self.args.sample == "uniform":
                if self.args.collaboration in ["none", "aposteriori"]:
                    x_sampled, y_sampled = self.dataset.sample_selfish_uniform(
                        self.args.budget, attribute, random_seed=random_seed, oversample=self.args.oversample)
                else:
                    raise RuntimeError(f"Sample strategy {self.args.sample} with {self.args.collaboration} not supported!")
                
                queries_per_agent.append((x_sampled, y_sampled))
            elif self.args.sample == "stratified":
                if self.args.collaboration in ["none", "aposteriori"]:
                    x_sampled, y_sampled = self.dataset.sample_selfish_stratified(
                        self.args.budget, attribute, random_seed, oversample=self.args.oversample)
                elif self.args.collaboration == "apriori":
                    x_sampled, y_sampled = self.dataset.sample_coordinated_stratified(
                        collab_attributes, self.args.budget, random_seed, oversample=self.args.oversample)
                else:
                    raise RuntimeError("Sample strategy not supported!")
                                
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
