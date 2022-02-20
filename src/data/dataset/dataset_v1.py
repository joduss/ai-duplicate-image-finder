import pickle

from data.dataset.image_pair import ImagePair


class Dataset:

    def __init__(self, images: list[str], image_pairs: list[ImagePair]):
        self.all_images = images
        self.image_pairs = image_pairs


def load_train() -> Dataset:
    with open("data/train-dataset/dataset", "rb") as file:
        return pickle.load(file)


def load_validation() -> Dataset:
    with open("data/validation-dataset/dataset", "rb") as file:
        return pickle.load(file)


def load_test() -> Dataset:
    with open("data/test-dataset/dataset", "rb") as file:
        return pickle.load(file)