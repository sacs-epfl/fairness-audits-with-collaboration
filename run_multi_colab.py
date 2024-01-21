import logging
import os
from copy import deepcopy
from multiprocessing.context import Process
from typing import List, Tuple
from itertools import combinations

from args import get_args

from faircoop.audit import Audit
from faircoop.dataset import get_dataset
from write_results import write_results

logging.basicConfig(level=logging.INFO)


def run(info: Tuple[int, str], args):
    dataset = get_dataset(args.dataset)
    results: List = []
    budget, results_file_name = info

    for _ in range(args.repetitions):
        audit = Audit(args, dataset)
        audit.run()
        results += audit.results
        if args.seed is not None:
            args.seed += 1

    write_results(args, results, results_file_name, write_dir="results/multicolab_b%d" % budget)


if __name__ == "__main__":
    args = get_args()
    
    assert args.attributes_to_audit is None

    if args.collaboration == "apriori":
        args.unbias_mean = True
    else:
        args.unbias_mean = False

    if args.dataset not in ["synthetic", "german_credit", "propublica", "folktables"]:
        raise RuntimeError("Unknown dataset %s" % args.dataset)


    if args.dataset == "german_credit":
        protected_attributes = ['age', 'sex', 'marital_status', 'own_telephone', 'employment']
    else:
        raise RuntimeError("Unknown dataset %s" % args.dataset)

    n = len(protected_attributes)
    processes = []

    if args.collaboration == "none":
        comb = protected_attributes # comb stands for combination, see below
        args.attributes_to_audit = comb
        out_file_name = "multicolab_%s_%s_n%d_b%d_%s.csv" % (args.dataset, args.collaboration, len(comb), args.budget, ",".join(comb))
        p = Process(target=run, args=((args.budget, out_file_name), deepcopy(args)))
        p.start()
        processes.append(p)
    else:
        for k in range(2, n + 1):
            for comb in combinations(protected_attributes, k):
                args.attributes_to_audit = comb
                out_file_name = "multicolab_%s_%s_n%d_b%d_%s.csv" % (args.dataset, args.collaboration, k, args.budget, ",".join(comb))
                p = Process(target=run, args=((args.budget, out_file_name), deepcopy(args)))
                p.start()
                processes.append(p)

    for p in processes:
        p.join()
