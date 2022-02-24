import pickle

from data.dataset.image_pair import ImagePair


class Dataset:

    BASE_PATH = "data/dataset_v1"

    def __init__(self, images: list[str], image_pairs: list[ImagePair]):
        self.all_images = images
        self.image_pairs = image_pairs


def load_train() -> Dataset:
    with open(f"{Dataset.BASE_PATH}/train-dataset/dataset", "rb") as file:
        return pickle.load(file)


def load_validation() -> Dataset:
    with open(f"{Dataset.BASE_PATH}/validation-dataset/dataset", "rb") as file:
        return pickle.load(file)


def load_test() -> Dataset:
    with open(f"{Dataset.BASE_PATH}/test-dataset/dataset", "rb") as file:
        return pickle.load(file)