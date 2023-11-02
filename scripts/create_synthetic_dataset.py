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
parser.add_argument("--attribute-imbalance", type=str, default=None, help="The imbalance of o particular attributes, provided as: '0=0.3,1=0.4 if we want P(a0=1) = 0.3 and P(a1=1) = 0.4")
parser.add_argument("--bias", type=str, default=None, help="The bias on particular weights, provided as: '0=3,1=4'.")
args = parser.parse_args()

rand = random.Random(args.seed)
generated_rows = set()

biases = {int(k): int(v) for k, v in (pair.split('=') for pair in args.bias.split(','))} if args.bias else {}
weights = []
for i in range(args.attributes):
    if i in biases:
        weights.append(biases[i])
    else:
        weights.append(1)  # No bias

bs = sum(weights)
WEIGHTS = [i / bs for i in weights]  # Normalize

features = []
labels = []
num_rows = 2 ** args.attributes if args.rows is None else args.rows
print("Will create a synthetic dataset with %d rows" % num_rows)
rows_iter = range(2 ** args.attributes) if args.rows is None else rand.sample(range(2 ** args.attributes), args.rows)
row_count = 0
imbalances = {int(k): float(v) for k, v in (pair.split('=') for pair in args.attribute_imbalance.split(','))} if args.bias else {}
for row in rows_iter:
    if row_count % 10000 == 0:
        print("Created %d rows..." % row_count)

    if args.rows is not None:
        sample_row = rand.sample(range(2 ** args.attributes), 1)[0]
        values = [int(c) for c in format(sample_row, '#0%db' % (args.attributes + 2))[2:]]

        # Modify the sample values according ot attribute imbalance
        for attribute_index in range(args.attributes):
            if attribute_index in imbalances:
                values[attribute_index] = 0 if rand.random() < imbalances[attribute_index] else 1
    else:
        values = [int(c) for c in format(row, '#0%db' % (args.attributes + 2))[2:]]

    features.append(values)

    # Determine the output label
    tot = 0
    for i in range(args.attributes):
        tot += WEIGHTS[i] * values[i]

    y = 1 if rand.random() < tot else 0
    labels.append(y)
    row_count += 1

if not os.path.exists(os.path.join("..", "data")):
    os.mkdir(os.path.join("..", "data"))

if not os.path.exists(os.path.join("..", "data", "synthetic")):
    os.mkdir(os.path.join("..", "data", "synthetic"))

with open(os.path.join("..", "data", "synthetic", "features.csv"), "w") as out_file:
    header_str = ",".join(["a%d" % i for i in range(args.attributes)])
    out_file.write(header_str + "\n")
    for row in range(num_rows):
        values_str = ",".join([("%d" % i) for i in features[row]])
        out_file.write(values_str + "\n")

with open(os.path.join("..", "data", "synthetic", "labels.csv"), "w") as out_file:
    out_file.write("Y\n")
    for row in range(num_rows):
        out_file.write("%d\n" % labels[row])
