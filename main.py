import logging
from typing import List

from args import get_args

from faircoop.audit import Audit
from faircoop.dataset import get_dataset
from write_results import write_results

logging.basicConfig(level=logging.INFO)

if __name__ == "__main__":
    args = get_args()
    orig_seed = args.seed

    results: List = []
    for exp_num in range(args.repetitions):
        audit = Audit(args, get_dataset(args.dataset))
        audit.run()
        results += audit.results
        if args.seed is not None:
            args.seed += 1

    results_filename = "results_%s_%s_%s_%d_%d.csv" % (args.dataset, args.collaboration, args.sample, len(args.attributes_to_audit), orig_seed)
    write_results(args, results, results_filename)
