"""
Scripts to test the sampling of different strategies.
"""
from faircoop.dataset import SyntheticDataset

TRIALS = 50
BUDGET = 100

dataset = SyntheticDataset()
dataset.load_dataset()


def test_aposteriori():
    frequencies = [0] * len(dataset.features)
    for trial in range(TRIALS):
        query_indices_0 = dataset.sample_selfish_stratified(100, dataset.protected_attributes[0], random_seed=trial+1)
        query_indices_1 = dataset.sample_selfish_stratified(100, dataset.protected_attributes[1], random_seed=trial + 1)
        assert len(set(query_indices_0)) == BUDGET
        assert len(set(query_indices_1)) == BUDGET
        for query in query_indices_0 + query_indices_1:
            frequencies[query] += 1

    print(min(frequencies))
    print(max(frequencies))
    print(frequencies)


def test_apriori():
    frequencies = [0] * len(dataset.features)
    for trial in range(TRIALS):
        query_indices_0 = dataset.sample_coordinated_stratified([dataset.protected_attributes[1]], 100, dataset.protected_attributes[0], random_seed=trial + 1)
        query_indices_1 = dataset.sample_coordinated_stratified([dataset.protected_attributes[0]], 100, dataset.protected_attributes[1], random_seed=trial + 1 + 100000)
        assert len(set(query_indices_0)) == BUDGET
        assert len(set(query_indices_1)) == BUDGET
        for query in query_indices_0 + query_indices_1:
            frequencies[query] += 1

    print(min(frequencies))
    print(max(frequencies))
    print(frequencies)


test_aposteriori()
print("=====")
test_apriori()
