import os

import pandas as pd

from faircoop.dataset.dataset import Dataset


class ProPublicaDataset(Dataset):

    def load_dataset(self):
        self.logger.info("Loading ProPublica dataset...")
        self.features = pd.read_csv(os.path.join("data", self.get_name(), "features.csv"))
        self.labels = pd.read_csv(os.path.join("data", self.get_name(), "labels.csv"))
        self.logger.info("ProPublica dataset loaded (rows: %d)", len(self.features))

        self.protected_attributes = ["Female", "African_American", "Age_Below_TwentyFive", "Misdemeanor", "Number_of_Priors"]
        self.compute_ground_truth_dp()
        self.compute_subspace_probabilities()

    def get_name(self) -> str:
        return "propublica"
