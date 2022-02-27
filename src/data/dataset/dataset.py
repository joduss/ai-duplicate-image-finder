from __future__ import annotations

from data.dataset.image_pair import ImagePair
import pandas as pd


class Dataset:
    BASE_PATH = "data/dataset_v1"


    def __init__(self, images: list[str], image_pairs: list[ImagePair]):
        self.all_images = images
        self.image_pairs = image_pairs


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
                similar=row["similar"],
                augmented=row["augmented"]
            ))

        return Dataset(images=list(all_images), image_pairs=pairs)
