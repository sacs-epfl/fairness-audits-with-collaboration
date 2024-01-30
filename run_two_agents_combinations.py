import logging
import os
from copy import deepcopy
from multiprocessing.context import Process
from typing import List

from args import get_args

from faircoop.audit import Audit
from faircoop.dataset import get_dataset
from write_results import merge_csv_files

logging.basicConfig(level=logging.INFO)


def write_results(args, results, results_file_name: str):
    if not os.path.exists("results"):
        os.mkdir("results")

    agents: int = len(args.attributes_to_audit)
    results_file_path = os.path.join("results", results_file_name)

    logging.info("Writing results to %s", results_file_path)
    with open(results_file_path, "w") as results_file:
        results_file.write("dataset,collaboration,sample,agents,a0,a1,seed,budget,agent,attribute,dp_error\n")

        for result in results:
            if result[-2] == args.attributes_to_audit[0]:
                results_file.write("%s,%s,%s,%d,%s,%s,%d,%d,%d,%s,%f\n" % (args.dataset, args.collaboration, args.sample, agents, args.attributes_to_audit[0], args.attributes_to_audit[1] if len(args.attributes_to_audit) == 2 else args.attributes_to_audit[0], *result))


def run(attribute_0: str, attribute_1: str, results_file_name: str, args):
    dataset = get_dataset(args.dataset)
    results: List = []

    if attribute_0 == attribute_1:
        args.attributes_to_audit = [attribute_0]
    else:
        args.attributes_to_audit = [attribute_0, attribute_1]

    if args.collaboration == "apriori" and attribute_0 != attribute_1:
        args.unbias_mean = True

    for _ in range(args.repetitions):
        audit = Audit(args, dataset)
        audit.run()
        results += audit.results
        if args.seed is not None:
            args.seed += 1

    write_results(args, results, results_file_name)


if __name__ == "__main__":
    args = get_args()

    if args.dataset in ["synthetic", "german_credit"]:
        args.budget = 100
    elif args.dataset == "folktables":
        args.budget = 500
    else:
        raise RuntimeError("Unsupported dataset %s" % args.dataset)

    result_csv_files = []
    dataset = get_dataset(args.dataset)
    dataset.load_dataset()

    for collaboration in ["apriori"]:
        args.collaboration = collaboration
        processes = []
        for attribute_0 in list(dataset.features.columns.values)[:3]:
            for attribute_1 in list(dataset.features.columns.values)[:3]:
                out_file_name = "combinations_n2_%s_%s_%s_%s.csv" % (args.dataset, collaboration, attribute_0, attribute_1)
                p = Process(target=run, args=(attribute_0, attribute_1, out_file_name, deepcopy(args)))
                p.start()
                processes.append(p)
                result_csv_files.append(os.path.join("results", out_file_name))

        print("Running %d processes (budget: %s)..." % (len(processes), args.budget))

        for p in processes:
            p.join()

    print("Processes done - combining results")
    output_file_name = os.path.join("results", "combinations_n2_%s.csv" % args.dataset)
    merge_csv_files(result_csv_files, output_file_name)
