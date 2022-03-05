from __future__ import annotations

import PIL.Image as pilim
from matplotlib import pyplot as plt
from PIL.Image import Image
from PIL.ImageOps import exif_transpose

from src.data.dataset.image_pair import ImagePair


def resize(image: Image, max_length: int) -> Image:
    # note: size is (width, height)
    if image.size[0] > image.size[1]:
        width = max_length
        height = image.size[1] * max_length / image.size[0]
    else:
        height = max_length
        width = image.size[0] * max_length / image.size[1]

    return exif_transpose(image.resize((int(width), int(height)), pilim.BILINEAR))


def plot(image_pairs: list[ImagePair], predicted_similarities: list[bool] = None, fig_size_multiplier: float = 1, titles: list[str] = None):
    assert predicted_similarities is None or len(predicted_similarities) == len(image_pairs)

    plot_images = []
    plot_titles = []

    if titles is not None:
        for title in titles:
            plot_titles += [title, title]

    for idx in range(len(image_pairs)):
        pair = image_pairs[idx]
        image_a = pair.image_a
        image_b = pair.image_b
        is_similar = pair.similar

        if titles is None:
            if predicted_similarities is not None:
                if predicted_similarities[idx] == is_similar:
                    image_title = f"Ok: Similarity is {is_similar}"
                else:
                    image_title = f"Wrong: Expected similarity {is_similar}, was {predicted_similarities[idx]}"
            else:
                image_title = "Similar" if is_similar else "Different"

            plot_titles += [image_title, image_title]

        image_a = resize(pilim.open(image_a), int(200 * fig_size_multiplier))
        image_b = resize(pilim.open(image_b), int(200 * fig_size_multiplier))

        plot_images += [image_a, image_b]

    plot_grid(titles=plot_titles, images=plot_images,
         rows=len(image_pairs), cols=2, fig_size_multiplier=fig_size_multiplier)



def plot_grid(rows: int, cols: int, images: list[Image], titles: list[str], fig_size_multiplier: float = 1):
    """
    Plots images with the given titles in print_loaded_image_name_tf grid.
    :param rows:
    :param cols:
    :param images:
    :param titles:
    :param fig_size_multiplier: Multiplier to reduce or increase the size of each image.
    """
    fig = plt.figure(figsize=(3 * cols * fig_size_multiplier, 3 * rows * fig_size_multiplier))
    count = rows * cols

    for idx in range(count):
        ax = plt.subplot(rows, cols, idx + 1)
        plt.imshow(images[idx])
        plt.title(titles[idx])
        plt.axis("off")

    fig.tight_layout()
