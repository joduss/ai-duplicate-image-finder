import argparse
import os
import random
from math import sqrt
from PIL import Image
from PIL import ImageDraw
import cv2
import numpy as np

# PARSING ARGUMENTS
from src.data.dataset.dataset_builder import DatasetBuilder
from src.data.dataset.image_pair import ImagePair

parser = argparse.ArgumentParser(description='Build dataset from local all_images.')
parser.add_argument('input', type=str, help='Directory to scan for pictures')
parser.add_argument('output', type=str, help='Where to create the dataset')

args = parser.parse_args()

INPUT = args.input
OUTPUT = args.output


# PROGRAM

def run():
    builder = AugmentedDatasetBuilder(source_path=args.input, dest_path=args.output)
    builder.build()
    pass


class AugmentedDatasetBuilder(DatasetBuilder):

    AUGMENTED_IMAGE_DIR_NAME = "augmented"


    def __init__(self, source_path: str, dest_path: str, max_img_dim: int = 384,
                 augmentation_per_img: int = 5, area_ratio: float = 0.7, angle: int = 20):
        """
        :param source_path:
        :param dest_path:
        :param augmentation_per_img:
        :param area_ratio: Do some cropping with area that is at least that much compared to the original.
        :param angle: Angles to generate.
        """
        super().__init__(source_path, dest_path, max_img_dim, pairs_per_img=augmentation_per_img)
        self.augmentation_per_img = augmentation_per_img
        self.area_ratio = area_ratio
        self.angle = angle
        self.augmented_pairs: list[ImagePair] = []



    def augment_image(self, image: np.ndarray, image_path: str, target_dir: str) -> list[ImagePair]:
        augmented_image_dir = f"{target_dir}/{self.AUGMENTED_IMAGE_DIR_NAME}"
        os.makedirs(augmented_image_dir, exist_ok=True)

        for i in range(self.augmentation_per_img):
            if random.random() > 0.5:
                # Create single image and create pair with it and the original
                augmented_image, angle = self.rotate(image)
                augmented_image, ratio = self.crop(augmented_image)

                augmented_image_path = self.generate_image_path(original_image_path=image_path,
                                                                to_append=str(i),
                                                                target_dir=augmented_image_dir)

                augmented_image = self.resize_image(augmented_image)
                self.save_image(augmented_image, augmented_image_path)

                self.augmented_pairs.append(
                    ImagePair(image_a=image_path,
                              image_b=augmented_image_path,
                              similar=True,
                              augmented=True)
                )

            else:
                # Generate 2 images and create a pair from them
                augmented_image_1, angle = self.rotate(image)
                augmented_image_1, ratio = self.crop(augmented_image_1)

                augmented_image_path_1 = self.generate_image_path(original_image_path=image_path,
                                                                to_append=f"{str(i)}-a",
                                                                target_dir=augmented_image_dir)

                augmented_image_1 = self.resize_image(augmented_image_1)
                self.save_image(augmented_image_1, augmented_image_path_1)

                augmented_image_2, angle = self.rotate(image)
                augmented_image_2, ratio = self.crop(augmented_image_2)

                augmented_image_path_2 = self.generate_image_path(original_image_path=image_path,
                                                                to_append=f"{str(i)}-b",
                                                                target_dir=augmented_image_dir)

                augmented_image_2 = self.resize_image(augmented_image_2)
                self.save_image(augmented_image_2, augmented_image_path_2)

                self.augmented_pairs.append(
                    ImagePair(image_a=augmented_image_path_1,
                              image_b=augmented_image_path_2,
                              similar=True,
                              augmented=True)
                )

        return self.augmented_pairs


    def crop(self, image: np.ndarray) -> (np.ndarray, int):

        # 0.25 =>
        # 0.5 => 0.7
        ratio_delta = 0.3
        dim_ratio = sqrt(self.area_ratio)
        ratio_max = min(1.0, dim_ratio + ratio_delta)
        ratio_delta = min(ratio_delta, ratio_max - dim_ratio)
        ratio_min = dim_ratio - ratio_delta

        height_ratio = random.uniform(ratio_min, ratio_max)
        width_ratio = random.uniform(ratio_min, ratio_max)

        start_row = int(random.uniform(0, 1-height_ratio) * image.shape[0])
        start_col = int(random.uniform(0, 1-width_ratio) * image.shape[1])

        new_height = int(height_ratio * image.shape[0])
        new_width = int(width_ratio * image.shape[1])

        return image[start_row:start_row + new_height, start_col:start_col + new_width], height_ratio * width_ratio


    def rotate(self, image: np.ndarray) -> (np.ndarray, float):
        angle = random.uniform(-self.angle, self.angle)

        img_center = (image.shape[1] // 2, image.shape[0] // 2)
        rotation_mtx = cv2.getRotationMatrix2D(img_center, angle, 1)
        rotated_image = cv2.warpAffine(image, rotation_mtx, (image.shape[1], image.shape[0]), flags=cv2.INTER_LINEAR)
        return rotated_image, angle



if __name__ == '__main__':
    run()
