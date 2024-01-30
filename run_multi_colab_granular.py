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

    write_results(args, results, results_file_name, write_dir=f"results/{args.dataset}/multicolab_b{budget}")


if __name__ == "__main__":
    args = get_args()
    
    assert args.attributes_to_audit is None

    if args.collaboration == "apriori" and args.sample != "uniform":
        args.unbias_mean = True
    else:
        args.unbias_mean = False

    if args.dataset not in ["synthetic", "german_credit", "propublica", "folktables"]:
        raise RuntimeError("Unknown dataset %s" % args.dataset)


    if args.dataset == "german_credit":
        protected_attributes = ['age', 'sex', 'marital_status', 'own_telephone', 'employment']
    elif args.dataset == "folktables":
        protected_attributes = ["SEX", "MAR", "AGEP", "NATIVITY", "MIG"]
    elif args.dataset == "propublica":
        protected_attributes = ["Female", "African_American", "Age_Below_TwentyFive", "Misdemeanor", "Number_of_Priors"]

    n = len(protected_attributes)
    processes = []

    if args.collaboration == "none":
        comb = protected_attributes # comb stands for combination, see below
        args.attributes_to_audit = comb
        out_file_name = "multicolab_%s_%s_%s_n%d_b%d_%s.csv" % (args.dataset, args.collaboration, args.sample, len(comb), args.budget, ",".join(comb))
        p = Process(target=run, args=((args.budget, out_file_name), deepcopy(args)))
        p.start()
        processes.append(p)
    else:
        k = args.n_collab
        assert k <= n and k >= 2, "k must be between 2 and n"
        for comb in combinations(protected_attributes, k):
            args.attributes_to_audit = comb
            out_file_name = "multicolab_%s_%s_%s_n%d_b%d_%s.csv" % (args.dataset, args.collaboration, args.sample, k, args.budget, ",".join(comb))
            p = Process(target=run, args=((args.budget, out_file_name), deepcopy(args)))
            p.start()
            processes.append(p)

    for p in processes:
        p.join()
