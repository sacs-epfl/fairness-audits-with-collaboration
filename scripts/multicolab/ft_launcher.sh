#!/bin/bash

# take n_collab as input
sample=$1
collaboration=$2
n_collab=$3

# if collaboration is apriori, then --oversample
if [ $collaboration == "apriori" ]; then
    oversample="--oversample"
else
    oversample=""
fi

python /home/dhasade/audits/ml-audits/run_multi_colab_granular.py \
    --dataset folktables \
    --sample $sample \
    --collaboration $collaboration \
    --n_collab $n_collab \
    --budget 500 \
    --seed 12 \
    --repetitions 300 $oversample
