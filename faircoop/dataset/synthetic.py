import os

import pandas as pd

from faircoop.dataset.dataset import Dataset


class SyntheticDataset(Dataset):

    def load_dataset(self):
        self.logger.info("Loading synthetic dataset...")
        self.features = pd.read_csv(os.path.join("data", "synthetic", "features.csv"))
        self.labels = pd.read_csv(os.path.join("data", "synthetic", "labels.csv"))
        self.logger.info("Synthetic dataset loaded (rows: %d)", len(self.features))

        self.protected_attributes = ["a0", "a1"]  # list(self.features.columns.values)
        self.compute_ground_truth_dp()
        self.compute_subspace_probabilities()
