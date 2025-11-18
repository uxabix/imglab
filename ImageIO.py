from os import path

import numpy as np
from PIL import Image


def load_image(filename: str, folder_path: str = "input") -> np.ndarray:
    image = Image.open(path.join(folder_path, filename)).convert("RGB")
    image_array = np.array(image)

    return image_array


def save_image(filename: str, array: np.ndarray, folder_path: str = "output"):
    Image.fromarray(array).save(path.join(folder_path, filename))
