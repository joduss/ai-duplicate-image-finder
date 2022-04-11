

class ImageShape:

    def __init__(self, height: int, width: int, depth: int):
        self.height = height
        self.width = width
        self.depth = depth

    def to_shape(self) -> tuple:
        """
        :return:(Height, width, depth)
        """
        return (self.height, self.width, self.depth)


    def as_tuple(self) -> tuple:
        """
        :return:(Height, width, depth)
        """
        return (self.height, self.width, self.depth)


    @property
    def size(self) -> tuple:
        """
        :return: (self.height, self.width)
        """
        return (self.height, self.width)