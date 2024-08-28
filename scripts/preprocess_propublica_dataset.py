"""
Binarize the ProPublica dataset.
"""
import os
import pandas as pd
import numpy as np

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths relative to the current script
data_dir = os.path.join(base_dir, "..", "data")
propublica_dir = os.path.join(data_dir, "propublica")

# Create the directories if they don't exist
if not os.path.exists(data_dir):
    raise RuntimeError("data directory not found!")

if not os.path.exists(propublica_dir):
    raise RuntimeError("proPublica dir does not exist!")

if not os.path.exists(os.path.join(propublica_dir, "compas-scores-two-years.csv")):
    raise RuntimeError("propublica original data file not found: compas-scores-two-years.csv")

df_o = pd.read_csv(os.path.join(propublica_dir, "compas-scores-two-years.csv"))
print("="*5, "Loaded Raw ProPublica dataset (rows: %d)" % len(df_o), "="*5)


df = df_o[['age', 'c_charge_degree', 'race', 'age_cat', 'score_text', 'sex', 'priors_count',
               'days_b_screening_arrest', 'decile_score', 'is_recid', 'two_year_recid',
               'c_jail_in', 'c_jail_out']]

# Applying the filters as in Compas Analysis R notebook
# URL: https://github.com/propublica/compas-analysis/blob/master/Compas%20Analysis.ipynb
df = df[df['days_b_screening_arrest'] <= 30]
df = df[df['days_b_screening_arrest'] >= -30]
df = df[df['is_recid'] != -1]
df = df[df['c_charge_degree'] != "O"]
df = df[df['score_text'] != 'N/A']

# Getting the number of rows after filtering
num_rows = df.shape[0]
print("="*5, "Filtered ProPublica dataset (rows: %d)" % num_rows, "="*5)

# keep rows in df_o whose index is in df
df_o = df_o.loc[df.index]

# create a new final df
new_df = pd.DataFrame()

# column 1: African_American
new_df['African_American'] = np.where(df_o['race'] == 'African-American', 1, 0)

# column 2: Gender
new_df['Female'] = np.where(df_o['sex'] == 'Female', 1, 0)

# column 3: Age_Below_TwentyFive
new_df['Age_Below_TwentyFive'] = np.where(df_o['age'] < 25, 1, 0)

# column 4: Two_yr_Recidivism
new_df['Two_yr_Recidivism'] = np.where(df_o['two_year_recid'] == 1, 1, 0)

# column 5: Number_of_Priors
new_df['Number_of_Priors'] = df_o['priors_count'].values

# column 6: Misdemeanor
new_df['Misdemeanor'] = np.where(df_o['c_charge_degree'] == 'M', 1, 0)

# labels is the two year recidivism
labels = new_df['Two_yr_Recidivism']
labels.name = "Y"

# features are the rest of the columns
features = new_df.drop(columns=["Two_yr_Recidivism"])

# Number of priors > 0 -> 1 else 0
features["Number_of_Priors"] = features["Number_of_Priors"].apply(lambda x: 1 if x > 0 else 0)

features.to_csv(os.path.join(propublica_dir, 'features.csv'), index=False)
labels.to_csv(os.path.join(propublica_dir, 'labels.csv'), index=False)