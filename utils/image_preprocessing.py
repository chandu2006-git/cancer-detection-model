from PIL import Image
import numpy as np


def preprocess_image(image_file):
    """
    Convert uploaded image to model input format.
    """

    image = Image.open(image_file)

    image = image.convert("RGB")

    image = image.resize((224, 224))

    image = np.array(image).astype("float32")

    image = np.expand_dims(image, axis=0)

    return image