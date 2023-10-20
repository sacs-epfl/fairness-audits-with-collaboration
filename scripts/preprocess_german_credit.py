"""
Binarize the German Credit dataset.
"""
import os

import pandas as pd
from aif360.sklearn.datasets import fetch_german
from sklearn.preprocessing import LabelEncoder

if not os.path.exists(os.path.join("..", "data")):
    os.mkdir(os.path.join("..", "data"))

if not os.path.exists(os.path.join("..", "data", "german_credit")):
    os.mkdir(os.path.join("..", "data", "german_credit"))

print("Fetching German Credit dataset...")
dataset = fetch_german()
# split into inputs and outputs

X, y = dataset.X, dataset.y

# transform the age column into zero and one depending on the age being greater than 25
X['age'] = X['age'].apply(lambda x: 0 if x <= 25 else 1)

# transform the sex column into 0 or 1
X['sex'] = X['sex'].apply(lambda x: 0 if x == 'female' else 1).astype(int)

# transform the marital_status column into 0 or 1
X['marital_status'] = X['marital_status'].apply(lambda x: 0 if x == 'single' else 1).astype(int)

# transform the own_telephone column into 0 or 1
X['own_telephone'] = X['own_telephone'].apply(lambda x: 0 if x == 'none' else 1).astype(int)

# transform the employment column into 0 or 1
X['employment'] = X['employment'].apply(lambda x: 1 if x == '4<=X<7' or x == '>=7' else 0).astype(int)

X = X.reset_index(drop=True)
X.to_csv(os.path.join("..", "data", "german_credit", "features.csv"), index=False)

# label encode the target variable to have the classes 0 and 1
y = LabelEncoder().fit_transform(y)
y = pd.DataFrame(y)
y = y.rename(columns={0: "Y"})
y.to_csv(os.path.join("..", "data", "german_credit", "labels.csv"), index=False)

print("Processed German Credit dataset!")