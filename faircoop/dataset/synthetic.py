import os

import pandas as pd

from faircoop.dataset.dataset import Dataset


class SyntheticDataset(Dataset):

    def load_dataset(self):
        self.logger.info("Loading synthetic dataset...")
        self.features = pd.read_csv(os.path.join("data", self.get_name(), "features.csv"))
        self.labels = pd.read_csv(os.path.join("data", self.get_name(), "labels.csv"))
        self.logger.info("Synthetic dataset loaded (rows: %d)", len(self.features))

        self.protected_attributes = list(self.features.columns.values)
        self.compute_ground_truth_dp()
        #self.compute_subspace_probabilities()

    def get_name(self) -> str:
        return "synthetic"
