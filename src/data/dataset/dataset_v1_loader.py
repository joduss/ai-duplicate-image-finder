import pandas as pd
from src.data.dataset.dataset import Dataset


def dataset_name() -> str:
    return "v1"


def load_train_v1() -> Dataset:
    return _load(f"./data/dataset_v1/train-dataset/dataset")


def load_validation_v1() -> Dataset:
    return _load(f"./data/dataset_v1/validation-dataset/dataset")


def load_test_v1() -> Dataset:
    return _load(f"./data/dataset_v1/test-dataset/dataset")


def _load(path: str) -> Dataset:
    dataframe = pd.read_csv(filepath_or_buffer=path)
    return Dataset.from_dataframe(dataframe)
