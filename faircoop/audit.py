import logging
from typing import List, Tuple

import pandas as pd

from faircoop.metrics import demographic_parity_error, demographic_parity_error_unbiased


class Audit:

    def __init__(self, args, dataset):
        self.args = args
        self.dataset = dataset
        self.results: List = []
        self.logger = logging.getLogger(self.__class__.__name__)

        if args.collaboration == "aposteriori" and args.sample != "stratified":
            self.logger.info("Setting sampling method to stratified for a posteriori collaboration")
            self.args.sample = "stratified"

        if self.args.agents > len(self.dataset.protected_attributes):
            raise RuntimeError("There cannot be more agents (%d) than protected attributes (%d)!",
                               self.args.agents, len(self.dataset.protected_attributes))

    def run(self):
        self.logger.info("Running audit with %s sampling and collaboration mode: %s (seed: %d)",
                         self.args.sample, self.args.collaboration, self.args.seed)

        queries_per_agent: List[Tuple[List, List]] = []
        for agent in range(self.args.agents):
            attribute = self.dataset.protected_attributes[agent]
            self.logger.info("Agent %d auditing attribute %s of black box...", agent, attribute)
            if self.args.sample == "uniform":
                if self.args.collaboration == "none":
                    x_sampled, y_sampled = self.dataset.sample_selfish_uniform(
                        self.args.budget, attribute, self.args.seed)
                else:
                    raise RuntimeError("Sample strategy not supported!")
                queries_per_agent.append((x_sampled, y_sampled))
            elif self.args.sample == "stratified":
                if self.args.collaboration in ["none", "aposteriori"]:
                    x_sampled, y_sampled = self.dataset.sample_selfish_stratified(
                        self.args.budget, attribute, self.args.seed)
                elif self.args.collaboration == "apriori":
                    # Determine the collaborating agents
                    collab_agents = [self.dataset.protected_attributes[agent_index] for agent_index in range(self.args.agents) if agent_index != agent]
                    x_sampled, y_sampled = self.dataset.sample_coordinated_stratified(
                        collab_agents, self.args.budget, attribute, self.args.seed + agent * 10000000)
                else:
                    raise RuntimeError("Sample strategy not supported!")
                queries_per_agent.append((x_sampled, y_sampled))

        if self.args.collaboration == "none":
            # Compute the DP error per agent
            for agent, sampled in enumerate(queries_per_agent):
                x_sampled, y_sampled = sampled
                attribute = self.dataset.protected_attributes[agent]
                dp_error = demographic_parity_error(
                    x_sampled, y_sampled, attribute, self.dataset.ground_truth_dps[attribute])
                self.results.append((self.args.seed, self.args.budget, agent, dp_error))
        elif self.args.collaboration in ["aposteriori", "apriori"]:
            # Combine all queries and then compute the DP error per agent
            # TODO assume all agents work together
            x_all = pd.concat([x_agent for x_agent, _ in queries_per_agent])
            y_all = pd.concat([y_agent for _, y_agent in queries_per_agent])
            for agent, sampled in enumerate(queries_per_agent):
                attribute = self.dataset.protected_attributes[agent]
                other_attributes = [attr for attr in self.dataset.protected_attributes if attr != attribute]
                if self.args.unbias_mean:
                    dp_error = demographic_parity_error_unbiased(
                        x_all, y_all, attribute, self.dataset.subspace_features_probabilities,
                        self.dataset.subspace_labels_probabilities, other_attributes,
                        self.dataset.ground_truth_dps[attribute], self.dataset.protected_attributes,
                        len(self.dataset.features))
                else:
                    dp_error = demographic_parity_error(x_all, y_all, attribute, self.dataset.ground_truth_dps[attribute])
                self.results.append((self.args.seed, self.args.budget, agent, dp_error))
        else:
            raise RuntimeError("Collaboration strategy %s not implemented!", self.args.collaboration)
