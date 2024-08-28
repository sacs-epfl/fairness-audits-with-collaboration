import os
from folktables import ACSDataSource, ACSPublicCoverage

# Get the directory of the current script
base_dir = os.path.dirname(os.path.abspath(__file__))

# Define the paths relative to the current script
data_dir = os.path.join(base_dir, "..", "data")
folktables_dir = os.path.join(data_dir, "folktables")

data_source = ACSDataSource(survey_year='2018', horizon='5-Year', survey='person')

print("="*5, "Downloading ACS Public Coverage data...", "="*5)

# This will download into data/2018/5-Year directory
# The download takes a few minutes (less than an hour)
# Requires a machine with at least 150GB of RAM
acs_data = data_source.get_data(join_household=True, download=True)
nrows = len(acs_data)

print("="*5, "Downloaded ACS Public Coverage data (rows: %d)!" % nrows, "="*5)

features, label, group = ACSPublicCoverage.df_to_pandas(acs_data)

features.to_csv(os.path.join(folktables_dir, "features.csv"), index=False)
label.to_csv(os.path.join(folktables_dir, "labels.csv"), index=False)

print("="*5, "Saved ACS Public Coverage data!", "="*5)