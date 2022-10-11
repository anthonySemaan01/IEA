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
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "digit_letter.csv"])
    VECTOR_EVEN_ODD__PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "even_odd.csv"])
    VECTOR_CONSONANT_VOWELS_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "consonant_vowel.csv"])
    VECTOR_ODD_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "odd.csv"])
    VECTOR_EVEN_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "even.csv"])
    VECTOR_VOWELS_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "vowels.csv"])
    VECTOR_CONSONNANT_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "consonnant.csv"])
