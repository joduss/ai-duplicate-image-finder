import pickle

import pandas as pd
from src.data.dataset.dataset import Dataset


def load_train_v1() -> Dataset:
    return _load(f"{Dataset.BASE_PATH}/train-dataset/dataset")


def load_validation_v1() -> Dataset:
    return _load(f"{Dataset.BASE_PATH}/validation-dataset/dataset")


def load_test_v1() -> Dataset:
    return _load(f"{Dataset.BASE_PATH}/test-dataset/dataset")



def _load(path: str) -> Dataset:
    dataframe = pd.read_csv(filepath_or_buffer=path)
    return Dataset.from_dataframe(dataframe)