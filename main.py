import logging
import os
from typing import List

from args import get_args

from faircoop.audit import Audit
from faircoop.dataset import get_dataset

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    args = get_args()
    orig_seed = args.seed

    results: List = []
    for budget in range(100, 1100, 100):
        args.budget = budget
        args.seed = orig_seed
        for exp_num in range(args.repetitions):
            audit = Audit(args, get_dataset(args.dataset))
            audit.run()
            results += audit.results
            args.seed += 1

    if not os.path.exists("results"):
        os.mkdir("results")

    results_file_name = "%s_%s_n%d.csv" % (args.collaboration, args.sample, args.agents)
    results_file_path = os.path.join("results", results_file_name)
    with open(results_file_path, "w") as results_file:
        results_file.write("collaboration,sample,agents,seed,budget,agent,dp_error\n")
        for result in results:
            results_file.write("%s,%s,%d,%d,%d,%d,%f\n" % (args.collaboration, args.sample, args.agents, *result))
