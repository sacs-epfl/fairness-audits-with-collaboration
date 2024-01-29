"""
Binarize the ProPublica dataset.
"""
import os

import pandas as pd

dataset_path = os.path.join("..", "data", "propublica")
if not os.path.exists(os.path.join(dataset_path, "propublica_raw.csv")):
    raise RuntimeError("propublica original data not found!")

df = pd.read_csv(os.path.join(dataset_path, "propublica_raw.csv"))
print("Loaded ProPublica dataset (rows: %d)" % len(df))

features = df.iloc[:, 1:].copy()
labels = df.iloc[:, 0]
labels.name = "Y"

# Number of priors > 0 -> 1 else 0
features["Number_of_Priors"] = features["Number_of_Priors"].apply(lambda x: 1 if x > 0 else 0)

features.to_csv(os.path.join(dataset_path, 'features.csv'), index=False)
labels.to_csv(os.path.join(dataset_path, 'labels.csv'), index=False)
