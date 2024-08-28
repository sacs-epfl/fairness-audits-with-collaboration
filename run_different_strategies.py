import logging
import os
from copy import deepcopy
from multiprocessing.context import Process
from typing import List, Tuple

from args import get_args

from faircoop.audit import Audit
from faircoop.dataset import get_dataset
from write_results import write_results, merge_csv_files

logging.basicConfig(level=logging.INFO)


def run(info: Tuple[str, bool, List[int], str], args):
    dataset = get_dataset(args.dataset)

    if args.sample == "neyman" and args.collaboration == "apriori":
        dataset.solve_collab(args.attributes_to_audit, args.budget * len(args.attributes_to_audit))

    results: List = []
    collaboration, should_unbias, budget, results_file_name = info
    if should_unbias:
        args.unbias_mean = True

    args.budget = budget
    args.collaboration = collaboration

    for _ in range(args.repetitions):
        audit = Audit(args, dataset)
        audit.run()
        results += audit.results
        if args.seed is not None:
            args.seed += 1

    write_dir = os.path.join("results", args.dataset)
    write_results(args, results, results_file_name, write_dir)


if __name__ == "__main__":
    args = get_args()
    agents = len(args.attributes_to_audit)

    if args.dataset == "german_credit":
        budgets = [50, 100, 150, 200, 250]
    elif args.dataset == "propublica":
        budgets = [100, 250, 500, 750, 1000]
    elif args.dataset == "folktables":
        budgets = [100, 250, 500, 750, 1000]
    else:
        raise RuntimeError("Unknown dataset %s" % args.dataset)

    result_csv_files = []
    processes = []

    # set when to unbias the mean
    # applicable for aprirori - neyman and stratified
    if args.sample == "uniform":
        to_run = [("none", False), ("aposteriori", False), ("apriori", False)]
    else: # stratified or neyman
        to_run = [("none", False), ("aposteriori", False), ("apriori", True)]

    for info in to_run:
        for budget in budgets:
            if info[1]:
                out_file_name = "%s_%s_n%d_b%d_unbias.csv" % (info[0], args.sample, agents, budget)
            else:
                out_file_name = "%s_%s_n%d_b%d.csv" % (info[0], args.sample, agents, budget)

            p = Process(target=run, args=((info[0], info[1], budget, out_file_name), deepcopy(args)))
            p.start()
            processes.append(p)

            result_csv_files.append(os.path.join("results", args.dataset, out_file_name))

    print("Running %d processes (budgets: %s)..." % (len(processes), budgets))

    for p in processes:
        p.join()

    print("Processes done - combining results")

    output_dir = os.path.join("results", args.dataset)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    output_file_name = os.path.join(output_dir, "merged_%s_%s_n%d.csv" % (args.dataset, args.sample, agents))
    merge_csv_files(result_csv_files, output_file_name)
