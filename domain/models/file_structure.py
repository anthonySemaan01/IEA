import os
from enum import Enum


class FileStructure(Enum):
    IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "images"])
    GRAY_SCALE_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "gray_scale_images"])
    CROPPED_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "cropped_images"])
    RESIZED_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "resized_images"])
    ERODED_DILATED_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "eroded_dilated"])
    PROCESSED_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "processed"])
    TRAINING_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "training", "images"])
    TRAINING_UBYTE_IMAGES_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "images-ubyte", "emnist-letters-train-images-idx3-ubyte"])
    TESTING_IMAGES_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "testing"])
    VECTOR_DIGIT_LETTER_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_digit_letter.csv"])
    VECTOR_EVEN_ODD__PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_even_odd.csv"])
    VECTOR_EVEN_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_even_values.csv"])
    VECTOR_ODD_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_odd_values.csv"])
    VECTOR_UPPER_LOWER_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_lower.csv"])
    VECTOR_UPPER_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_values.csv"])
    VECTOR_LOWER_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_lower_values.csv"])

