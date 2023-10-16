from faircoop.dataset.synthetic import SyntheticDataset


def get_dataset(dataset_name: str):
    # Prepare the dataset
    if dataset_name == "synthetic":
        dataset = SyntheticDataset()
        dataset.load_dataset()
        return dataset
    else:
        raise RuntimeError("Unknown dataset!")
