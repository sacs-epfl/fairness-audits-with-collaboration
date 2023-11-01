import logging
from typing import List, Tuple, Dict

import pandas as pd

from faircoop.metrics import demographic_parity_error, demographic_parity_error_unbiased


class Audit:

    def __init__(self, args, dataset):
        self.args = args
        self.agentwise_used_seeds = []  # to track the seed that is finally used, varies agentwise
        self.dataset = dataset
        self.results: List = []
        self.logger = logging.getLogger(self.__class__.__name__)

        if self.args.agents > len(self.dataset.protected_attributes):
            raise RuntimeError("There cannot be more agents (%d) than protected attributes (%d)!",
                               self.args.agents, len(self.dataset.protected_attributes))

        # Assign agents to attributes
        self.agent_to_attribute: Dict[int, int] = {}
        if self.args.agent_to_attribute:
            pairs = args.agent_to_attribute.split(',')
            assert len(pairs) == args.agents, "Each agent must be assigned to an attribute!"
            for pair in pairs:
                parts = pair.split("=")
                agent_index = int(parts[0])
                assert agent_index < args.agents, "Agent index %d out of range!" % agent_index
                attribute_name = parts[1]
                if attribute_name not in self.dataset.protected_attributes:
                    raise RuntimeError("Attribute %s not protected!" % attribute_name)
                self.agent_to_attribute[agent_index] = self.dataset.protected_attributes.index(attribute_name)
        else:
            self.agent_to_attribute = {i: i for i in range(self.args.agents)}

        if args.collaboration == "aposteriori" and args.sample != "stratified":
            self.logger.info("Setting sampling method to stratified for a posteriori collaboration")
            self.args.sample = "stratified"

    def run(self):
        self.logger.info("Running audit with %s sampling and collaboration mode: %s (seed: %s)",
                         self.args.sample, self.args.collaboration, self.args.seed)

        queries_per_agent: List[Tuple[List, List]] = []
        collab_agents = ["%d" % agent_index for agent_index in self.args.collaborating_agents]
        collab_attributes = [self.dataset.protected_attributes[self.agent_to_attribute[agent_index]] for agent_index in
                             self.args.collaborating_agents]
        for agent in range(self.args.agents):
            attribute = self.dataset.protected_attributes[self.agent_to_attribute[agent]]
            self.logger.info("Agent %d auditing attribute %s of black box...", agent, attribute)
            
            # Set the seed depending on the agent
            random_seed = self.args.seed + (agent+1) * 100000 if self.args.seed is not None else self.args.seed
            self.agentwise_used_seeds.append(random_seed)
            
            if self.args.sample == "uniform":
                if self.args.collaboration == "none":
                    x_sampled, y_sampled = self.dataset.sample_selfish_uniform(
                        self.args.budget, attribute, random_seed=random_seed)
                else:
                    raise RuntimeError("Sample strategy not supported!")
                queries_per_agent.append((x_sampled, y_sampled))
            elif self.args.sample == "stratified":
                if self.args.collaboration in ["none", "aposteriori"]:
                    x_sampled, y_sampled = self.dataset.sample_selfish_stratified(
                        self.args.budget, attribute, random_seed)
                elif self.args.collaboration == "apriori":
                    x_sampled, y_sampled = self.dataset.sample_coordinated_stratified(
                        collab_attributes, self.args.budget, random_seed)
                else:
                    raise RuntimeError("Sample strategy not supported!")
                                
                queries_per_agent.append((x_sampled, y_sampled))

        # Combine the queries of collaborating agents and compute the DPs

        if self.args.collaboration == "none":
            # Compute the DP error per agent
            for agent, sampled in enumerate(queries_per_agent):
                x_sampled, y_sampled = sampled
                used_seed = self.agentwise_used_seeds[agent]
                attribute = self.dataset.protected_attributes[self.agent_to_attribute[agent]]
                dp_error = demographic_parity_error(
                    x_sampled, y_sampled, attribute, self.dataset.ground_truth_dps[attribute])
                self.results.append((used_seed if used_seed is not None else -1, self.args.budget, agent,
                                     self.dataset.protected_attributes[self.agent_to_attribute[agent]],
                                     "-".join(collab_agents), dp_error))
        elif self.args.collaboration in ["aposteriori", "apriori"]:
            # Combine all queries and then compute the DP error per agent
            x_collab = pd.concat([x_agent for agent_index, (x_agent, _) in enumerate(queries_per_agent) if agent_index in self.args.collaborating_agents], ignore_index=True)
            y_collab = pd.concat([y_agent for agent_index, (_, y_agent) in enumerate(queries_per_agent) if agent_index in self.args.collaborating_agents], ignore_index=True)
            for agent, sampled in enumerate(queries_per_agent):
                used_seed = self.agentwise_used_seeds[agent]
                attribute = self.dataset.protected_attributes[self.agent_to_attribute[agent]]
                x_agent = x_collab if agent in self.args.collaborating_agents else queries_per_agent[agent][0]
                y_agent = y_collab if agent in self.args.collaborating_agents else queries_per_agent[agent][1]

                if self.args.unbias_mean:
                    other_attributes = [attr for attr in self.dataset.protected_attributes if attr != attribute]
                    dp_error = demographic_parity_error_unbiased(
                        x_agent, y_agent, attribute, self.dataset.subspace_features_probabilities,
                        self.dataset.subspace_labels_probabilities, other_attributes,
                        self.dataset.ground_truth_dps[attribute], self.dataset.protected_attributes,
                        len(self.dataset.features))
                else:
                    dp_error = demographic_parity_error(x_agent, y_agent, attribute, self.dataset.ground_truth_dps[attribute])
                self.results.append((used_seed if used_seed is not None else -1, self.args.budget, agent,
                                     self.dataset.protected_attributes[self.agent_to_attribute[agent]],
                                     "-".join(collab_agents), dp_error))
        else:
            raise RuntimeError("Collaboration strategy %s not implemented!", self.args.collaboration)
