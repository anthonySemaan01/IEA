from abc import ABC, abstractmethod
from fastapi import UploadFile, File
import numpy as np


class AbstractLetterFineTuner(ABC):

    @abstractmethod
    def find_non_white_pixels(self, image_arr: np.ndarray) -> list:
        pass

    @abstractmethod
    def gray_scale_converter(self, image_arr: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def letter_finder(self, file: UploadFile = File(...)):
        pass

    @abstractmethod
    def image_cropper(self, image_arr: np.ndarray, x: int, y: int, width: int, height: int):
        pass

    @abstractmethod
    def image_resizer(self, path: str):
        pass
