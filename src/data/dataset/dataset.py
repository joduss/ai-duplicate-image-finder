from __future__ import annotations

import random

from src.data.dataset.image_pair import ImagePair
import pandas as pd


class Dataset:

    def __init__(self, images: list[str], image_pairs: list[ImagePair]):
        self.all_images = images
        self.image_pairs = image_pairs


    def shuffle_pairs(self):
        random.shuffle(self.image_pairs)


    def to_dataframe(self):
        array = []

        for pair in self.image_pairs:
            array.append({
                "image_a": pair.image_a,
                "image_b": pair.image_b,
                "augmented": pair.augmented,
                "similar": pair.similar
            })

        return pd.DataFrame(array)


    def similarities(self):
        return list(map(lambda pair: pair.similar, self.image_pairs))


    @staticmethod
    def from_dataframe(dataframe: pd.DataFrame) -> Dataset:
        all_images = set()
        pairs: list[ImagePair] = []

        for index, row in dataframe.iterrows():
            all_images.add(row["image_a"])
            all_images.add(row["image_b"])

            pairs.append(ImagePair(
                image_a=row["image_a"],
                image_b=row["image_b"],
                similar=bool(row["similar"]),
                augmented=bool(row["augmented"])
            ))

        return Dataset(images=list(all_images), image_pairs=pairs)
