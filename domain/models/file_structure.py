import os
from enum import Enum


class FileStructure(Enum):
    IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "images"])
    GRAY_SCALE_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "gray_scale_images"])
    CROPPED_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "cropped_images"])

