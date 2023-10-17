import argparse


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample", type=str, default="stratified", choices=["uniform", "stratified"])
    parser.add_argument("--collaboration", type=str, default="none", choices=["none", "aposteriori", "apriori"])
    parser.add_argument("--dataset", type=str, default="synthetic", choices=["synthetic"])
    parser.add_argument("--budget", type=int, default=100)
    parser.add_argument("--agents", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--repetitions", type=int, default=10)
    parser.add_argument('--unbias-mean', action='store_true')

    return parser.parse_args()
