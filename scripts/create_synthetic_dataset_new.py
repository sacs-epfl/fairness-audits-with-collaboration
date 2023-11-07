"""
Create a synthetic dataset with some bias on the first weight.
"""
import argparse
import os
import numpy as np
from scipy.optimize import linprog

parser = argparse.ArgumentParser()
parser.add_argument("--seed", type=int, default=42)
parser.add_argument("--attributes", type=int, default=20)
parser.add_argument("--rows", type=int, default=100000)
parser.add_argument("--pc0", type=float, default=0.9, help="The probability of a0=1")
parser.add_argument("--pc1", type=float, default=0.7, help="The probability of a1=1")
parser.add_argument("--dp0", type=float, default=0.19, help="Demographic parity on a0")
parser.add_argument("--dp1", type=float, default=0.11, help="Demographic parity on a1")
parser.add_argument("--pyc01", type=float, default=0.6, help="The probability of y=1 given a0=1")
args = parser.parse_args()

rng = np.random.default_rng(args.seed)

pc0 = args.pc0; pc1 = args.pc1
dp0 = args.dp0; dp1 = args.dp1
pyc01 = args.pyc01
pyc00 = abs(dp0 - pyc01)
py = pyc01 * pc0 + pyc00 * (1 - pc0)
pyc11 = py + dp1 * (1 - pc1)
pyc10 = abs(pyc11 - dp1)

##############################################
q1 = pc0
r1 = pc1

obj = [1, 1, 1, 1]

lhs_eq =  [[r1, 0, 1-r1, 0],
[0, r1, 0, 1-r1],
[q1, 1-q1, 0, 0],
[0, 0, q1, 1-q1]]

rhs_eq = [pyc01, pyc00, pyc11, pyc10]

bnd = [(5e-2, 1), (5e-2, 1), (5e-2, 1), (5e-2, 1)]

opt = linprog(c=obj, A_eq=lhs_eq, b_eq=rhs_eq, bounds=bnd, method="interior-point")

pc01c11, pc00c11, pc01c10, pc00c10 = opt.x

print("=== Probabilities ===")
print("P(a0 = 0) = %f, P(a0 = 1) = %f" % (1 - pc0, pc0))
print("P(a1 = 0) = %f, P(a1 = 1) = %f" % (1 - pc1, pc1))
print("P(y = 0) = %f, P(y = 1) = %f" % (1 - py, py))
print("P(y = 1 | a0 = 0) = %f, P(y = 1 | a0 = 1) = %f" % (pyc00, pyc01))
print("P(y = 1 | a1 = 0) = %f, P(y = 1 | a1 = 1) = %f" % (pyc10, pyc11))
print("dp0 = %f, dp1 = %f" % (dp0, dp1))
print("P(y = 1 | a0 = 0, a1 = 0) = %f" % pc00c10)
print("P(y = 1 | a0 = 0, a1 = 1) = %f" % pc01c10)
print("P(y = 1 | a0 = 1, a1 = 0) = %f" % pc00c11)
print("P(y = 1 | a0 = 1, a1 = 1) = %f" % pc01c11)
print("")

##############################################

features = []
labels = []
num_rows = args.rows
print("Will create a synthetic dataset with %d rows" % num_rows)

xs = []
for i in range(args.attributes):

    if i == 0: # a0
        x = rng.binomial(1, pc0, size=num_rows)
    elif i == 1: # a1
        x = rng.binomial(1, pc1, size=num_rows)
    else: # a2, a3, ...
        x = rng.binomial(1, 0.5, size=num_rows)
    xs.append(x)

# transpose xs
xs = np.array(xs)
xs = xs.T
features = xs

# compute y based on pyc01 and pyc00
for x in xs:
    if x[0] == 1 and x[1] == 1:
        y = rng.binomial(1, pc01c11)
    elif x[0] == 1 and x[1] == 0:
        y = rng.binomial(1, pc01c10)
    elif x[0] == 0 and x[1] == 1:
        y = rng.binomial(1, pc00c11)
    else:
        y = rng.binomial(1, pc00c10)    
    labels.append(y)    

print(f'Data created. Saving now..')

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
