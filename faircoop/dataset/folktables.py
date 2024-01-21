import os

import pandas as pd

from faircoop.dataset.dataset import Dataset


class FolktablesDataset(Dataset):
    def load_dataset(self):
        self.logger.info("Loading Folktables dataset...")
        self.features = pd.read_csv(os.path.join("data", self.get_name(), "features_bin.csv"))
        self.labels = pd.read_csv(os.path.join("data", self.get_name(), "labels_bin.csv"))
        self.logger.info("Folktables dataset loaded (rows: %d)", len(self.features))

        self.protected_attributes = ["SEX", "MAR", "AGEP", "NATIVITY", "MIG"]
        self.compute_ground_truth_dp()
        self.compute_subspace_probabilities()

    def get_name(self) -> str:
        return "folktables"
