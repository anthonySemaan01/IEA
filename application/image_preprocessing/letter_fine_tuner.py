import os

import cv2
import numpy as np
from PIL import Image as im
from fastapi import UploadFile, File
from numpy import ndarray

from domain.contracts.abstract_letter_fine_tuner import AbstractLetterFineTuner
from domain.exceptions.image_preprocessing_exception import ImagePreprocessingException
from domain.models.file_structure import FileStructure
from shared.helper.load_file import image_loader
from shared.helper.save_file import save_file


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

    def black_white_converter(self, image_arr: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            gray_scale = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
            (thresh, im_bw) = cv2.threshold(gray_scale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
            gray_scale_path = str(FileStructure.GRAY_SCALE_IMAGES_PATH.value) + "\\img{}.png".format(
                len(os.listdir(str(FileStructure.GRAY_SCALE_IMAGES_PATH.value))))
            # image.save(gray_scale_path)
            cv2.imwrite(gray_scale_path, im_bw)
        except Exception as e:
            print(e.__str__())
            raise ImagePreprocessingException(additional_message="error while converting to gray-scale")
        return im_bw, gray_scale_path

    def bbox_drawer(self, image_arr: np.ndarray, x1: int, y1: int, width: int, height: int):
        bbox_image = cv2.rectangle(image_arr, (y1, x1), (height, width), (255, 0, 0), 3)
        image_out = im.fromarray(bbox_image)
        image_out.save(str(FileStructure.BBOX_IMAGE_TESTING.value) + "\\tester.png")

    def image_cropper(self, image_arr: np.ndarray, x: int, y: int, width: int, height: int) -> tuple[np.ndarray, str]:
        try:
            cropped_image = image_arr[x:width, y:height]
            cropped_image_path = str(FileStructure.CROPPED_IMAGES_PATH.value) + "\\img{}.png".format(
                len(os.listdir(str(FileStructure.CROPPED_IMAGES_PATH.value))))
            cv2.imwrite(cropped_image_path, cropped_image)
        except Exception as e:
            print(e.__str__())
            raise ImagePreprocessingException(additional_message="error while cropping image")
        return cropped_image, cropped_image_path

    def image_resizer(self, image_arr: np.ndarray) -> tuple[np.ndarray, str]:
        try:
            resized_image = cv2.resize(image_arr, (32, 32), cv2.INTER_AREA)
            resized_image_path = str(FileStructure.RESIZED_IMAGES_PATH.value) + "\\img{}.png".format(
                len(os.listdir(str(FileStructure.RESIZED_IMAGES_PATH.value))))
            cv2.imwrite(resized_image_path, resized_image)
        except Exception as e:
            print(e.__str__())
            raise ImagePreprocessingException(additional_message="error while resizing image")
        return resized_image, resized_image_path

    def erosion_dilation(self, image_arr: np.ndarray) -> tuple[np.ndarray, str]:
        kernel = np.ones((5, 5), np.uint8)
        try:
            eroded = cv2.erode(image_arr, kernel, iterations=1)
            dilated = cv2.dilate(eroded, kernel, iterations=1)
            eroded_dilated_path = str(FileStructure.ERODED_DILATED_PATH.value) + "\\img{}.png".format(
                len(os.listdir(str(FileStructure.ERODED_DILATED_PATH.value))))
            cv2.imwrite(eroded_dilated_path, dilated)
        except Exception as e:
            print(e)
            print(e.__str__())
            raise ImagePreprocessingException(additional_message="error while eroding/dilating image")
        return dilated, eroded_dilated_path

    def letter_finder(self, file: UploadFile = File(...)) -> tuple[ndarray, str]:
        path = save_file(upload_file=file, destination=str(FileStructure.IMAGES_PATH.value))
        image_arr = image_loader(path=path)
        gray_scale, gray_scale_path = self.black_white_converter(image_arr=image_arr)
        coordinates = self.find_non_white_pixels(image_arr=gray_scale)
        self.bbox_drawer(image_arr=image_arr, x1=coordinates[0]["x"], y1=coordinates[0]["y"], width=coordinates[1]["x"],
                         height=coordinates[1]["y"])
        cropped_image, cropped_image_path = self.image_cropper(image_arr=gray_scale, x=coordinates[0]["x"],
                                                               y=coordinates[0]["y"],
                                                               width=coordinates[1]["x"], height=coordinates[1]["y"])
        dilated_arr, eroded_dilated_path = self.erosion_dilation(image_arr=cropped_image)
        resized_image, resized_image_path = self.image_resizer(image_arr=dilated_arr)

        image_out = im.fromarray(resized_image)
        image_out.save(str(FileStructure.TESTING_IMAGES_PATH.value) + "\\tester.png")
        output_path = str(FileStructure.PROCESSED_PATH.value) + "\\img{}.png".format(
            len(os.listdir(str(FileStructure.PROCESSED_PATH.value))))
        image_out.save(output_path)
        return resized_image, output_path
