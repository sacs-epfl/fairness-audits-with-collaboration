import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="stratified", choices=["uniform", "stratified", "neyman"])
    parser.add_argument("--collaboration", type=str, default="none", choices=["none", "aposteriori", "apriori"])
    parser.add_argument("--attributes-to-audit", type=str, default=None, help="The attributes to audit - pick all attributes if not specified.")
    parser.add_argument("--dataset", type=str, default="german_credit", choices=["german_credit", "folktables", "propublica"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument('--unbias-mean', action='store_true')
    parser.add_argument('--oversample', action='store_true')
    parser.add_argument('--n_collab', type=int, default=-1, help='To be used with run_multi_colab_granular.py, specifies the number of collaborating agents.')

    args = parser.parse_args()

    if args.attributes_to_audit:
        args.attributes_to_audit = args.attributes_to_audit.split(",")

    return args
