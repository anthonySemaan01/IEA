import os
from enum import Enum


class FileStructure(Enum):
    IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "images-ubyte"])
    GRAY_SCALE_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "gray_scale_images"])
    CROPPED_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "cropped_images"])
    RESIZED_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "resized_images"])
    TRAINING_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "training", "images","content"])
    TRAINING_UBYTE_IMAGES_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "images-ubyte", "emnist-letters-train-images-idx3-ubyte"])
    TESTING_IMAGES_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "testing", "samples", "sample.png"])
