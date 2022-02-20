

class ImagePair:

    def __init__(self, image_a: str, image_b: str, similar: bool):

        if image_a <= image_b:
            self.image_a = image_a
            self.image_b = image_b
        else:
            self.image_a = image_b
            self.image_b = image_a

        self.similar = similar


    def __eq__(self, other):
        if not isinstance(object, ImagePair):
            return False

        return self.image_a == other.image_a \
               and self.image_b == other.image_b \
               and self.similar == other.similar


    def __hash__(self):

        return hash((self.image_a, self.image_b, self.similar))
