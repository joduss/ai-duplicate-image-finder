import argparse
import pickle
from functools import partial
from pathlib import Path
import os
from multiprocessing import freeze_support, Pool
from random import shuffle

import cv2
import numpy as np
from tqdm import tqdm
import pyheif
from PIL import Image

# PARSING ARGUMENTS
from src.data.dataset.loader import Dataset
from src.data.dataset.image_pair import ImagePair

parser = argparse.ArgumentParser(description='Build dataset from local all_images.')
parser.add_argument('input', type=str, help='Directory to scan for pictures')
parser.add_argument('output', type=str, help='Where to create the dataset')

args = parser.parse_args()

INPUT = args.input
OUTPUT = args.output


# PROGRAM

def run():
    builder = DatasetBuilder(source_path=args.input, dest_path=args.output, max_img_dim=512)
    builder.build()
    pass


class DatasetBuilder:
    """
    Build print_loaded_image_name_tf dataset from print_loaded_image_name_tf source by creating resized copy.
    Can be subclasses to support augmentation.
    """

    RESIZED_ORIG_IMG_DIR_NAME = "resized_originals"

    def __init__(self, source_path: str, dest_path: str, max_img_dim: int, pairs_per_img: int = 1):
        self.source_path = os.path.abspath(source_path)
        self.dest_path = dest_path
        self.max_img_dim = max_img_dim
        self.pairs_per_img = pairs_per_img

        self.test_ratio = 0.2
        self.val_ratio = 0.15


    def build(self):
        print(f"Building dataset from all_images in {INPUT} and saves it in {OUTPUT}")

        original_images = self._list_images()

        test_split_pos = int(self.test_ratio * len(original_images))
        val_split_pos = int((self.test_ratio + self.val_ratio) * len(original_images))

        test_images = original_images[:test_split_pos]
        val_images = original_images[test_split_pos:val_split_pos]
        train_images = original_images[val_split_pos:]

        self._create_dataset_with(test_images, f"{self.dest_path}/test-dataset")
        self._create_dataset_with(val_images, f"{self.dest_path}/validation-dataset")
        self._create_dataset_with(train_images, f"{self.dest_path}/train-dataset")


    def _create_dataset_with(self, images: list[str], dataset_dir: str):
        os.makedirs(dataset_dir, exist_ok=True)

        resized_imgs, augmented_pairs = self.process_images(images, dataset_dir)
        all_images = set(resized_imgs)

        for augmented_pair in augmented_pairs:
            all_images.add(augmented_pair.image_a)
            all_images.add(augmented_pair.image_b)

        dataset = Dataset(images=list(all_images), image_pairs=self._create_image_pairs(resized_imgs) + augmented_pairs)
        dataset.shuffle_pairs()

        dataset.to_dataframe().to_csv(f"{dataset_dir}/dataset")


    def process_images(self, images: list[str], target_dir: str) -> tuple[list[str], list[ImagePair]]:
        all_augmented_pairs: list[ImagePair] = []
        processed_img: list[str] = []
        
        with Pool(processes=12) as p:
            with tqdm(total=len(images)) as pbar:
                f = partial(self._process_image_at_path, target_dir=target_dir)
                for i, new_path_augmented_pairs in enumerate(p.imap_unordered(f, images)):
                    pbar.update()
                    processed_img_path, augmented_pairs = new_path_augmented_pairs
                    all_augmented_pairs += augmented_pairs

                    if processed_img_path is not None:
                        processed_img.append(processed_img_path)
        # for img in images:
        #     all_augmented_pairs += self._process_image_at_path(img)

        return processed_img, all_augmented_pairs


    def _process_image_at_path(self, image_path: str, target_dir: str) -> (str, list[ImagePair]):
        """
        Returns print_loaded_image_name_tf tuple with:
            - resize image path
            - list of image pair resulting from image augmentation.
        """
        try:
            resized_img_dir = f"{target_dir}/{self.RESIZED_ORIG_IMG_DIR_NAME}"
            os.makedirs(resized_img_dir, exist_ok=True)

            image: np.ndarray = self._read_image(image_path)
            resized_img = self.resize_image(image)
            resized_image_path = self.generate_image_path(image_path, target_dir=resized_img_dir)
            self.save_image(resized_img, resized_image_path)

            return resized_image_path, self.augment_image(image=image, image_path=resized_image_path, target_dir=target_dir)

        except Exception as e:
            print(f"An exception {e} was caught when processing image {image_path}")
            return None, []


    def augment_image(self, image: np.ndarray, image_path: str, target_dir: str) -> list[ImagePair]:
        """
        Augment the image, and returns print_loaded_image_name_tf list of ImagePair resulting from the augmentation.
        By default, this method does not do anything.
        When overriding this method, you are responsible to save the image to print_loaded_image_name_tf directory
        which is not the same as 'self.dataset_image_dir'.
        :param image: The original image
        :param image_path: path of the resized image, which is added to the dataset
        :param: target_dir: directory where to save augmented images.
        """
        return []


    def resize_image(self, image) -> np.ndarray:
        resized_img: np.ndarray = image

        height_resize_factor = self.max_img_dim / resized_img.shape[0]
        width_resize_factor = self.max_img_dim / resized_img.shape[1]
        resize_factor = min(height_resize_factor, width_resize_factor)

        new_size = (round(resized_img.shape[1] * resize_factor), round(resized_img.shape[0] * resize_factor))
        return cv2.resize(resized_img, new_size, interpolation=cv2.INTER_AREA)


    def _list_images(self) -> list[str]:
        images = []
        for root, dirs, files in os.walk(self.source_path):
            images += [f"{root}/{file}" for file in files if self.is_image(file)]
            print(dir, end='\r')

        return images


    def _create_image_pairs(self, images: list[str]) -> list[ImagePair]:
        images_copy = images.copy()
        shuffle(images_copy)

        similar_pairs = [ImagePair(image, image, True) for image in images]

        all_pairs: set[ImagePair] = set(similar_pairs)

        for i in range(self.pairs_per_img):
            shuffle(images_copy)
            new_pairs = list(map(lambda imgs: ImagePair(imgs[0], imgs[1], False), zip(images, images_copy)))
            all_pairs = all_pairs.union(new_pairs)

        return list(all_pairs)


    @staticmethod
    def _read_image(image_path: str) -> np.ndarray:
        if image_path.lower().endswith(".heic"):
            heif_image = pyheif.read(image_path, apply_transformations=False, convert_hdr_to_8bit=False)
            pil_image = Image.frombytes(
                heif_image.mode,
                heif_image.size,
                heif_image.data,
                "raw",
                heif_image.mode,
                heif_image.stride,
            )
            return cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)

        return cv2.imread(image_path)


    @staticmethod
    def save_image(image: np.ndarray, image_path: str):
        cv2.imwrite(filename=image_path, img=image, params=[cv2.IMWRITE_JPEG_QUALITY, 80])


    @staticmethod
    def generate_image_path(original_image_path: str, target_dir: str, to_append: str or None = None):
        path = Path(original_image_path)
        image_path_parts = path.parts

        if to_append is None:
            return f"{os.path.expandvars(target_dir)}/{image_path_parts[-2]}_{path.stem}.jpg"
        else:
            return f"{os.path.expandvars(target_dir)}/{image_path_parts[-2]}_{path.stem}_{to_append}.jpg"


    @staticmethod
    def is_image(filename: str):
        filename_lower = filename.lower()
        return filename_lower.endswith(".jpg") \
               or filename_lower.endswith(".heic") \
               or filename_lower.endswith(".jpeg") \




if __name__ == '__main__':
    run()
