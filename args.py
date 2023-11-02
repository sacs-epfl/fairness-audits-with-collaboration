import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="stratified", choices=["uniform", "stratified"])
    parser.add_argument("--collaboration", type=str, default="none", choices=["none", "aposteriori", "apriori"])
    parser.add_argument("--attributes-to-audit", type=str, default=None, help="The attributes to audit - pick all attributes if not specified.")
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic", "german_credit", "folktables", "propublica"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument('--unbias-mean', action='store_true')

    args = parser.parse_args()

    return args
