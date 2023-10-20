import logging
from copy import deepcopy
from multiprocessing.context import Process
from typing import List, Tuple

from args import get_args

from faircoop.audit import Audit
from faircoop.dataset import get_dataset
from write_results import write_results

logging.basicConfig(level=logging.INFO)

def run(info: Tuple[str, bool], args):
    dataset = get_dataset(args.dataset)
    results: List = []
    collaboration, should_unbias = info
    if should_unbias:
        args.unbias_mean = True
    for budget in range(100, 1100, 100):
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
    for info in [("none", False), ("aposteriori", False), ("apriori", False), ("apriori", True)]:
        p = Process(target=run, args=(info, deepcopy(args)))
        p.start()
        processes.append(p)

    print("Running %d processes..." % len(processes))

    for p in processes:
        p.join()
