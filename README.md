# ml-audits

This repository hosts the code for our work on multi-agent collaborative auditing.

# Steps-to-run

1. Ensure that you have dataset files in the `data` folder. These should go as `data/<dataset_name>/features.csv` and `data/<dataset_name>/labels.csv` where `<dataset_name>` could be one of `folktables`, `german_credit`, `propublica` or `synthetic`. 
2. Run `analyze_dataset.py` by updating the `DATASETS = [...]` list to include the dataset names you want to analyze. This step is important as it will generate the required probabilities for debiasing, saved as `data/<dataset_name>/all_probs.pkl` and `data/<dataset_name>/all_ys.pkl`.
3. To generate gain plots, you will have to launch `run_multi_colab.py` as follows:

```
python run_multi_colab.py --repetitions 500 --dataset <dataset_name>  --seed 112 --oversample --sample stratified --collaboration <collab_type> --budget 200
```

where `<collab_type>` could be one of `none`, `aposteriori` or `apriori`. The result files will be generated in `results/<dataset_name>/multicolab_b<budget>`.