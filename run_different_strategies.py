import logging
from copy import deepcopy
from multiprocessing.context import Process
from typing import List, Tuple

from args import get_args

from faircoop.audit import Audit
from faircoop.dataset import get_dataset
from write_results import write_results, merge_csv_files

logging.basicConfig(level=logging.INFO)


def run(info: Tuple[str, bool, List[int]], args):
    dataset = get_dataset(args.dataset)
    results: List = []
    collaboration, should_unbias, budgets = info
    if should_unbias:
        args.unbias_mean = True
    for budget in budgets:
        args.budget = budget
        args.seed += budget * 10000
        args.collaboration = collaboration
        for exp_num in range(args.repetitions):
            audit = Audit(args, dataset)
            audit.run()
            results += audit.results
            args.seed += 1

    write_results(args, results)


if __name__ == "__main__":
    args = get_args()
    orig_seed = args.seed

    processes = []

    if args.dataset == "synthetic":
        budgets = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    elif args.dataset == "german_credit":
        budgets = [50, 100, 150, 200, 250]
    elif args.dataset == "folktables":
        budgets = [100, 200, 300, 400, 500, 600, 700, 800, 900, 1000]
    else:
        raise RuntimeError("Unknown dataset %s" % args.dataset)

    result_csv_files = []
    for info in [("none", False), ("aposteriori", False), ("apriori", False)]:
        for budget in budgets:
            p = Process(target=run, args=((info[0], info[1], budget), deepcopy(args)))
            p.start()
            processes.append(p)
            result_csv_files.append("%s_%s_n%d_b%d.csv" % (info[0], args.sample, args.agents, budget))

    print("Running %d processes (budgets: %s)..." % (len(processes), budgets))

    for p in processes:
        p.join()

    print("Processes done - combining results")
    output_file_name = "%s_%s_n%d.csv" % (args.dataset, args.sample, args.agents)
    merge_csv_files(result_csv_files, output_file_name)
