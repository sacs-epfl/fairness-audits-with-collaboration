"""
Obtain and log the demographic parities for each available dataset.
"""
import os

from faircoop.dataset import get_dataset

DATASETS = ["synthetic"]

if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")

    for dataset_name in DATASETS:
        dataset = get_dataset(dataset_name)
        dataset.load_dataset()

        print("=== Probabilities (%s) ===" % dataset.get_name())
        for attribute in dataset.features.columns.values:
            a_is_0 = len(dataset.features[dataset.features[attribute] == 0]) / len(dataset.features)
            a_is_1 = len(dataset.features[dataset.features[attribute] == 1]) / len(dataset.features)
            print("P(%s = 0) = %f, P(%s = 1) = %f" % (attribute, a_is_0, attribute, a_is_1))

        print("")
        print("=== Demographic Parity (%s) ===" % dataset.get_name())

        with open(os.path.join("results", "dps.csv"), "w") as out_file:
            out_file.write("dataset,attribute,dp\n")

            for attribute, dp in dataset.ground_truth_dps.items():
                out_file.write("%s,%s,%f\n" % (dataset.get_name(), attribute, dp))
                print("%s => %f" % (attribute, dp))

        print("")
