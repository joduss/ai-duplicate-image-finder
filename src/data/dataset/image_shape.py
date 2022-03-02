

class ImageShape:

    def __init__(self, height: int, width: int, depth: int):
        self.height = height
        self.width = width
        self.depth = depth

    def to_shape(self) -> tuple:
        return (self.height, self.width, self.depth)
