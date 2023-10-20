from faircoop.dataset.german_credit import GermanCreditDataset
from faircoop.dataset.synthetic import SyntheticDataset


def get_dataset(dataset_name: str):
    # Prepare the dataset
    if dataset_name == "synthetic":
        dataset = SyntheticDataset()
    elif dataset_name == "german_credit":
        dataset = GermanCreditDataset()
    else:
        raise RuntimeError("Unknown dataset!")

    dataset.load_dataset()
    return dataset
