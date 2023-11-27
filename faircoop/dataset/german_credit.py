import os

import pandas as pd

from faircoop.dataset.dataset import Dataset


class GermanCreditDataset(Dataset):
    def load_dataset(self):
        self.logger.info("Loading German Credit dataset...")
        self.features = pd.read_csv(os.path.join("data", self.get_name(), "features.csv"))
        self.labels = pd.read_csv(os.path.join("data", self.get_name(), "labels.csv"))
        self.logger.info("German Credit dataset loaded (rows: %d)", len(self.features))

        # self.protected_attributes = ["sex", "age"]
        self.protected_attributes = ['age', 'sex', 'marital_status', 'own_telephone', 'employment']
        self.compute_ground_truth_dp()
        self.compute_subspace_probabilities()

    def get_name(self) -> str:
        return "german_credit"
