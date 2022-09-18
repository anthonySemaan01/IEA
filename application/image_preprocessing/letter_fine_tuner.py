import os
from typing import Type

import cv2
import numpy as np
from fastapi import UploadFile, File

from domain.contracts.abstract_letter_fine_tuner import AbstractLetterFineTuner
from domain.models.file_structure import FileStructure
from shared.helper.load_file import image_loader
from shared.helper.save_file import save_file
from PIL import Image as im


class LetterFineTuner(AbstractLetterFineTuner):

    def find_non_white_pixels(self, image_arr: np.ndarray) -> list:
        gray_scale = 200
        rows = image_arr.shape[0]
        columns = image_arr.shape[1]
        point_a = {"x": image_arr.shape[0], "y": image_arr.shape[1]}
        point_b = {"x": 0, "y": 0}

        for row in range(0, rows - 1):
            for column in range(0, columns - 1):
                if image_arr[row][column] < gray_scale and point_a["x"] > row:
                    point_a["x"] = row

                if image_arr[row][column] < gray_scale and point_a["y"] > column:
                    point_a["y"] = column

                if image_arr[row][column] < gray_scale and point_b["x"] < row:
                    point_b["x"] = row

                if image_arr[row][column] < gray_scale and point_b["y"] < column:
                    point_b["y"] = column

        out = list()
        out.append(point_a)
        out.append(point_b)

        return out

    def gray_scale_converter(self, image_arr: np.ndarray) -> np.ndarray:
        gray_scale = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
        image = im.fromarray(gray_scale)
        image.save(str(FileStructure.GRAY_SCALE_IMAGES_PATH.value) + "\\img{}.png".format(
            len(os.listdir(str(FileStructure.GRAY_SCALE_IMAGES_PATH.value)))))
        return gray_scale

    def image_cropper(self, image_arr: np.ndarray, x: int, y: int, width: int, height: int):
        cropped_image = image_arr[x:width, y:height]
        image = im.fromarray(cropped_image)
        image.save(str(FileStructure.CROPPED_IMAGES_PATH.value) + "\\img{}.png".format(
            len(os.listdir(str(FileStructure.CROPPED_IMAGES_PATH.value)))))
        return cropped_image

    def image_resizer(self, image_arr: np.ndarray):
        resized_image = cv2.resize(image_arr, (28, 28), cv2.INTER_AREA)
        image = im.fromarray(resized_image)
        image.save(str(FileStructure.RESIZED_IMAGES_PATH.value) + "\\img{}.png".format(
            len(os.listdir(str(FileStructure.RESIZED_IMAGES_PATH.value)))))
        return resized_image

    def letter_finder(self, file: UploadFile = File(...)):
        path = save_file(upload_file=file, destination=str(FileStructure.IMAGES_PATH.value))
        image_arr = image_loader(path=path)
        gray_scale = self.gray_scale_converter(image_arr=image_arr)
        coordinates = self.find_non_white_pixels(image_arr=gray_scale)
        cropped_image = self.image_cropper(image_arr=gray_scale, x=coordinates[0]["x"], y=coordinates[0]["y"],
                                           width=coordinates[1]["x"], height=coordinates[1]["y"])
        resized_image = self.image_resizer(cropped_image)
