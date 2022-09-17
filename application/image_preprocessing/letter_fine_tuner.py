import os

import cv2
import numpy as np
from fastapi import UploadFile

from domain.contracts.abstract_letter_fine_tuner import AbstractLetterFineTuner
from domain.models.file_structure import FileStructure
from shared.helper.load_file import image_loader
from shared.helper.save_file import save_file
from PIL import Image as im


class LetterFineTuner(AbstractLetterFineTuner):

    def find_non_white_pixels(self, file: np.ndarray):
        gray_scale = self.gray_scale_converter(file)




    def gray_scale_converter(self, file: np.ndarray) -> np.ndarray:
        gray_scale = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)
        image = im.fromarray(gray_scale)
        print((str(FileStructure.GRAY_SCALE_IMAGES_PATH.value)))
        image.save(str(FileStructure.GRAY_SCALE_IMAGES_PATH.value) + "\\img{}.png".format(
            len(os.listdir(str(FileStructure.GRAY_SCALE_IMAGES_PATH.value)))))
        return gray_scale

    def image_cropper(self, file: UploadFile, x: int, y: int, width: int, height: int):
        path = save_file(upload_file=file, destination=str(FileStructure.IMAGES_PATH.value))
        image = image_loader(path=path)
        cropped_image = image[x:width, y:height]
        return cropped_image

    def image_resizer(self, path: str):
        return "hello world"

    def letter_finder(self, path: str):
        return "hello world"
