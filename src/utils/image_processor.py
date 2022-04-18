import numpy as np
import cv2
import PIL.Image as pilim

class ImageProcessor:
    """
    Offers image preprocessing functions.
    """

    @property
    def target_height(self):
        return self.target_size[1]

    @property
    def target_width(self):
        return self.target_size[0]


    def __init__(self, image_target_size: tuple):
        """
        :param image_target_size: (height, width)
        """
        self.target_size = image_target_size


    def fit_resize_and_rescale(self, image: np.ndarray):
        """
        Resizes (fit) the image and scale pixel values in the range 0-1.
        """
        return_pil_image = isinstance(image, pilim.Image)
        image = np.array(image) if return_pil_image else image

        image = self.fit_resize(image)
        image = self.rescale(image)

        return pilim.fromarray(image) if return_pil_image else image


    def fill_resize_and_rescale(self, image: np.ndarray):
        """
        Resizes (fill) the image and scale pixel values in the range 0-1.
        """
        return_pil_image = isinstance(image, pilim.Image)
        image = np.array(image) if return_pil_image else image

        image = self.fill_resize(image)
        image = self.rescale(image)

        return pilim.fromarray(image) if return_pil_image else image


    def fit_resize(self, image: np.ndarray or pilim.Image):
        """
        Resizes (fit) the image to the target size.
        """
        return_pil_image = isinstance(image, pilim.Image)
        image = np.array(image) if return_pil_image else image

        # Compute new size
        width = image.shape[1]
        height = image.shape[0]

        height_resize_factor = self.target_height / height
        width_resize_factor= self.target_width / width

        # if height_resize_factor < 1 or width_resize_factor < 1:
        resize_factor = min(height_resize_factor, width_resize_factor)
        # else:

        new_size = tuple([round(x * resize_factor) for x in (height, width)])
        new_height, new_width = (min(new_size[0], self.target_height), min(new_size[1], self.target_width))

        # Convert numpy to PIL.
        # Note: In OpenCV size is (width, height)
        image = cv2.resize(image, (new_width, new_height))

        delta_w = self.target_width - new_width
        delta_h = self.target_height - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        color = [0, 0, 0]
        new_im = cv2.copyMakeBorder(image, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                    value=color)
        new_im = np.array(new_im)

        return pilim.fromarray(new_im) if return_pil_image else new_im



    def fill_resize(self, image: np.ndarray):
        """
        Resizes (fill) the image to the target size.
        """
        return_pil_image = isinstance(image, pilim.Image)
        image = np.array(image) if return_pil_image else image

        # Compute new size
        width = image.shape[1]
        height = image.shape[0]

        height_resize_factor = self.target_height / height
        width_resize_factor = self.target_width / width

        # if height_resize_factor < 1 or width_resize_factor < 1:
        resize_factor = max(height_resize_factor, width_resize_factor)
        # else:

        new_size = tuple([round(x * resize_factor) for x in (height, width)])
        new_height, new_width = (new_size[0], new_size[1])

        # Convert numpy to PIL.
        # Note: In OpenCV size is (width, height)
        image = cv2.resize(image, (new_width, new_height))

        delta_w = self.target_width - new_width
        delta_h = self.target_height - new_height
        top, bottom = delta_h // 2, delta_h - (delta_h // 2)
        left, right = delta_w // 2, delta_w - (delta_w // 2)

        new_im = image[-bottom:-bottom + self.target_height, -right:-right + self.target_width]
        new_im = np.array(new_im)

        return pilim.fromarray(new_im) if return_pil_image else new_im



    def rescale(self, image: np.ndarray):
        return image / 255