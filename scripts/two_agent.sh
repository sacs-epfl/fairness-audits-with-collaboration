#!/bin/bash

dataset=german_credit # german_credit, propublica, folktables

# attrs_to_audit="Female,African_American" # Propublica
attrs_to_audit="age,sex" # German Credit
# attrs_to_audit="SEX,MAR" # Folktables

# repetitions=500 # Propublica
repetitions=1000 # German Credit
# repetitions=300 # Folktables

# bugdet, sample, collaboration will be set in run_different_strategies.py
# seed starts at specified value and increments by 1 for each repetition
python run_different_strategies.py \
    --sample stratified \
    --dataset $dataset \
    --seed 90 \
    --attributes-to-audit $attrs_to_audit \
    --repetitions $repetitions \
    --oversample

# result will be saved as results/<dataset>/merged_<dataset>_stratified_n2.csv