from faircoop.dataset.folktables import FolktablesDataset
from faircoop.dataset.german_credit import GermanCreditDataset
from faircoop.dataset.propublica import ProPublicaDataset

def get_dataset(dataset_name: str):
    # Prepare the dataset
    if dataset_name == "german_credit":
        dataset = GermanCreditDataset()
    elif dataset_name == "folktables":
        dataset = FolktablesDataset()
    elif dataset_name == "propublica":
        dataset = ProPublicaDataset()
    else:
        raise RuntimeError("Unknown dataset!")

    dataset.load_dataset()
    return dataset
