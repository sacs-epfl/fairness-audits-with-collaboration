"""
Binarize the folk tables dataset.
"""
import os
import pandas as pd

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths relative to the current script
data_dir = os.path.join(base_dir, "..", "data")
folktables_dir = os.path.join(data_dir, "folktables")

if not os.path.exists(os.path.join(folktables_dir, "features.csv")) or \
    not os.path.exists(os.path.join(folktables_dir, "labels.csv")):
    raise RuntimeError("Folktables original data not found!")

features = pd.read_csv(os.path.join(folktables_dir, "features.csv"))
labels = pd.read_csv(os.path.join(folktables_dir, "labels.csv"))
print("="*5, "Loaded folk tables dataset (rows: %d)" % len(features), "="*5)

# Binarize features
features['SEX'] = features['SEX'].apply(lambda x: 0 if x == 2 else 1)
features['MAR'] = features['MAR'].apply(lambda x: 0 if x in [2, 3, 4, 5] else 1)
features['AGEP'] = features['AGEP'].apply(lambda x: 0 if x <= 25 else 1)
features['NATIVITY'] = features['NATIVITY'].apply(lambda x: 0 if x == 2 else 1)
features['MIG'] = features['MIG'].apply(lambda x: 0 if x in [0, 2, 3] else 1)

print("="*5, "Binarized folk tables dataset!", "="*5)

features = features.reset_index(drop=True)
features.to_csv(os.path.join(folktables_dir, "features_bin.csv"), index=False)

print("="*5, "Saved features!", "="*5)

labels['PUBCOV'] = labels['PUBCOV'].apply(lambda x: 0 if not x else 1)
labels = labels.rename(columns={"PUBCOV": "Y"})
labels.to_csv(os.path.join(folktables_dir, "labels_bin.csv"), index=False)

print("="*5, "Saved labels!", "="*5)