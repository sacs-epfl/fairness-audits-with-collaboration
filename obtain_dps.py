"""
Obtain and log the demographic parities for each available dataset.
"""
import os

from faircoop.dataset import SyntheticDataset

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")

    with open(os.path.join("results", "dps.csv"), "w") as out_file:
        out_file.write("dataset,attribute,dp\n")
        dataset = SyntheticDataset()
        dataset.load_dataset()
        for attribute, dp in dataset.ground_truth_dps.items():
            out_file.write("%s,%s,%f\n" % (dataset.get_name(), attribute, dp))
