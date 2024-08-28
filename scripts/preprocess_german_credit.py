"""
Binarize the German Credit dataset.
"""
import os

import pandas as pd
from aif360.sklearn.datasets import fetch_german
from sklearn.preprocessing import LabelEncoder

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths relative to the current script
data_dir = os.path.join(base_dir, "..", "data")
german_credit_dir = os.path.join(data_dir, "german_credit")

# Create the directories if they don't exist
if not os.path.exists(data_dir):
    os.mkdir(data_dir)

if not os.path.exists(german_credit_dir):
    os.mkdir(german_credit_dir)

print("="*5, "Fetching German Credit dataset...", "="*5)
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
X.to_csv(os.path.join(german_credit_dir, "features.csv"), index=False)

# label encode the target variable to have the classes 0 and 1
y = LabelEncoder().fit_transform(y)
y = pd.DataFrame(y)
y = y.rename(columns={0: "Y"})
y.to_csv(os.path.join(german_credit_dir, "labels.csv"), index=False)

print("="*5, "Processed German Credit dataset!", "="*5)