import os
from itertools import combinations

from faircoop.dataset import get_dataset

DATASETS = ["german_credit", "propublica"]


if __name__ == "__main__":
    if not os.path.exists("results"):
        os.mkdir("results")

    results_path = os.path.join("results", "strata.csv")
    with open(results_path, "w") as out_file:
        out_file.write("dataset,k,strata,freq\n")

    for dataset_name in DATASETS:
        dataset = get_dataset(dataset_name)
        dataset.load_dataset()

        print("Analyzing strata of dataset %s (rows: %d, features: %d)" % (dataset_name, len(dataset.features), len(dataset.protected_attributes)))

        for k in range(2, len(dataset.protected_attributes) + 1):
            print("Considering K = %d" % k)

            # Generate possible agent configurations
            lst = list(range(0, len(dataset.protected_attributes)))
            combs = list(combinations(lst, k))

            print("Combinations: %d (%s)" % (len(combs), combs))
            for comb in combs:
                strata_size = {}
                for i in range(2 ** k):
                    strata_size[i] = 0

                print("Considering combination: %s" % str(comb))
                for idx, row in dataset.features.iterrows():
                    if idx > 0 and idx % 100000 == 0:
                        print("Row %d..." % idx)

                    strata = []
                    for agent_index, agent_attribute in enumerate(comb):
                        bitval = int(getattr(row, dataset.protected_attributes[agent_attribute]))
                        strata.append(bitval)
                    bit_num = int(''.join(str(bit) for bit in strata), 2)
                    strata_size[bit_num] += 1

                with open(results_path, "a") as out_file:
                    for strata, freq in strata_size.items():
                        out_file.write("%s,%d,%d,%d\n" % (dataset_name, k, strata, freq))
                print(strata_size)
