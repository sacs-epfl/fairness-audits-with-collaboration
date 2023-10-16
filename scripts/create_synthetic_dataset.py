"""
Create a synthetic dataset with some bias on the first weight.
"""
import argparse
import os
import random

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--attributes", type=int, default=20)
parser.add_argument("--rows", type=int, default=None)
parser.add_argument("--bias", type=float, help="The bias on the first weights.", default=5)
args = parser.parse_args()

rand = random.Random(args.seed)

weights = []
for i in range(args.attributes):
    weights.append(1)  # No bias
weights[0] = args.bias

bs = sum(weights)
WEIGHTS = [i / bs for i in weights]  # Normalize

features = []
labels = []
num_rows = 2 ** args.attributes if args.rows is None else args.rows
print("Will create a synthetic dataset with %d rows" % num_rows)
rows_iter = range(2 ** args.attributes) if args.rows is None else rand.sample(range(2 ** args.attributes), args.rows)
for row in rows_iter:
    if row % 10000 == 0:
        print("Created %d rows..." % row)

    values = [int(c) for c in format(row, '#0%db' % (args.attributes + 2))[2:]]
    features.append(values)

    # Determine the output label
    tot = 0
    for i in range(args.attributes):
        tot += WEIGHTS[i] * values[i]

    y = 1 if rand.random() < tot else 0
    labels.append(y)

if not os.path.exists("data"):
    os.mkdir("data")

if not os.path.exists(os.path.join("data", "synthetic")):
    os.mkdir(os.path.join("data", "synthetic"))

with open(os.path.join("data", "synthetic", "features.csv"), "w") as out_file:
    header_str = ",".join(["a%d" % i for i in range(args.attributes)])
    out_file.write(header_str + "\n")
    for row in range(num_rows):
        values_str = ",".join([("%d" % i) for i in features[row]])
        out_file.write(values_str + "\n")

with open(os.path.join("data", "synthetic", "labels.csv"), "w") as out_file:
    out_file.write("Y\n")
    for row in range(num_rows):
        out_file.write("%d\n" % labels[row])
