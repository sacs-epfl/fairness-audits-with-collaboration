"""
Binarize the folk tables dataset.
"""
import os

import pandas as pd

ft_path = os.path.join("..", "data", "folktables")
if not os.path.exists(os.path.join(ft_path, "features.csv")) or not os.path.exists(os.path.join(ft_path, "labels.csv")):
    raise RuntimeError("Folktables original data not found!")

features = pd.read_csv(os.path.join(ft_path, "features.csv"))
labels = pd.read_csv(os.path.join(ft_path, "labels.csv"))
print("Loaded folk tables dataset (rows: %d)" % len(features))

# Binarize features
features['SEX'] = features['SEX'].apply(lambda x: 0 if x == 2 else 1)
features['MAR'] = features['MAR'].apply(lambda x: 0 if x in [2, 3, 4, 5] else 1)
features['AGEP'] = features['AGEP'].apply(lambda x: 0 if x <= 25 else 1)
features['NATIVITY'] = features['NATIVITY'].apply(lambda x: 0 if x == 2 else 1)
features['MIG'] = features['MIG'].apply(lambda x: 0 if x in [0, 2, 3] else 1)

features = features.reset_index(drop=True)
features.to_csv(os.path.join("..", "data", "folktables", "features_bin.csv"), index=False)

labels['PUBCOV'] = labels['PUBCOV'].apply(lambda x: 0 if not x else 1)
labels = labels.rename(columns={"PUBCOV": "Y"})
labels.to_csv(os.path.join("..", "data", "folktables", "labels_bin.csv"), index=False)
