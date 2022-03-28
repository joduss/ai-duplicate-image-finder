import pandas as pd

from src.data.dataset.dataset import Dataset


def load_train_v2() -> Dataset:
    return _load(f"./data/dataset_v2/train-dataset/dataset")


def load_validation_v2() -> Dataset:
    return _load(f"./data/dataset_v2/validation-dataset/dataset")


def load_test_v2() -> Dataset:
    return _load(f"./data/dataset_v2/test-dataset/dataset")

def _load(path: str) -> Dataset:
    dataframe = pd.read_csv(filepath_or_buffer=path)
    return Dataset.from_dataframe(dataframe)