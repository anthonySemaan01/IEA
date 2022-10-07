import os
from typing import Type, Tuple

import cv2
import numpy as np
from fastapi import UploadFile, File
from numpy import ndarray

from domain.contracts.abstract_letter_fine_tuner import AbstractLetterFineTuner
from domain.models.file_structure import FileStructure
from shared.helper.load_file import image_loader
from domain.exceptions.image_preprocessing_exception import ImagePreprocessingException
from shared.helper.save_file import save_file
from PIL import Image as im


class LetterFineTuner(AbstractLetterFineTuner):

    def find_non_white_pixels(self, image_arr: np.ndarray) -> list:
        try:
            gray_scale = 200
            rows = image_arr.shape[0]
            columns = image_arr.shape[1]
            point_a = {"x": image_arr.shape[0], "y": image_arr.shape[1]}
            point_b = {"x": 0, "y": 0}

            for row in range(0, rows - 1):
                for column in range(0, columns - 1):
                    if image_arr[row][column] <= gray_scale and point_a["x"] > row:
                        point_a["x"] = row

                    if image_arr[row][column] <= gray_scale and point_a["y"] > column:
                        point_a["y"] = column

                    if image_arr[row][column] <= gray_scale and point_b["x"] < row:
                        point_b["x"] = row

                    if image_arr[row][column] <= gray_scale and point_b["y"] < column:
                        point_b["y"] = column

            out = list()
            out.append(point_a)
            out.append(point_b)
        except Exception as e:
            print(e.__str__())
            raise ImagePreprocessingException(additional_message="error while finding letter bbox")

        return out

    def gray_scale_converter(self, image_arr: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            gray_scale = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
            image = im.fromarray(gray_scale)
            gray_scale_path = str(FileStructure.GRAY_SCALE_IMAGES_PATH.value) + "\\img{}.png".format(
                len(os.listdir(str(FileStructure.GRAY_SCALE_IMAGES_PATH.value))))
            image.save(gray_scale_path)
        except Exception as e:
            print(e.__str__())
            raise ImagePreprocessingException(additional_message="error while converting to gray-scale")
        print("gray scale: ", gray_scale)
        return gray_scale, gray_scale_path

    def image_cropper(self, image_arr: np.ndarray, x: int, y: int, width: int, height: int) -> tuple[np.ndarray, str]:
        try:
            cropped_image = image_arr[x:width, y:height]
            image = im.fromarray(cropped_image)
            cropped_image_path = str(FileStructure.CROPPED_IMAGES_PATH.value) + "\\img{}.png".format(
                len(os.listdir(str(FileStructure.CROPPED_IMAGES_PATH.value))))
            image.save(cropped_image_path)
        except Exception as e:
            print(e.__str__())
            raise ImagePreprocessingException(additional_message="error while cropping image")
        return cropped_image, cropped_image_path

    def image_resizer(self, image_arr: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            resized_image = cv2.resize(image_arr, (28, 28), cv2.INTER_AREA)
            image = im.fromarray(resized_image)
            resized_image_path = str(FileStructure.RESIZED_IMAGES_PATH.value) + "\\img{}.png".format(
                len(os.listdir(str(FileStructure.RESIZED_IMAGES_PATH.value))))
            image.save(resized_image_path)
        except Exception as e:
            print(e.__str__())
            raise ImagePreprocessingException(additional_message="error while resizing image")
        return resized_image, resized_image_path

    def letter_finder(self, file: UploadFile = File(...)) -> tuple[ndarray, str]:
        path = save_file(upload_file=file, destination=str(FileStructure.IMAGES_PATH.value))
        image_arr = image_loader(path=path)
        gray_scale, gray_scale_path = self.gray_scale_converter(image_arr=image_arr)
        coordinates = self.find_non_white_pixels(image_arr=gray_scale)
        cropped_image, cropped_image_path = self.image_cropper(image_arr=gray_scale, x=coordinates[0]["x"],
                                                               y=coordinates[0]["y"],
                                           width=coordinates[1]["x"], height=coordinates[1]["y"])
        resized_image, resized_image_path = self.image_resizer(cropped_image)
        image_resized = im.fromarray(resized_image)
        image_resized.save(str(FileStructure.TESTING_IMAGES_PATH.value) + "\\tester.png")
        return resized_image, resized_image_path
