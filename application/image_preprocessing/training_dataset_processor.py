import os
import cv2
import numpy as np
from PIL import Image as im
from domain.models.file_structure import FileStructure
from shared.helper.load_file import image_loader


def find_non_white_pixels(image_arr: np.ndarray) -> list:
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
    return out


def black_white_converter(image_arr: np.ndarray) -> np.ndarray:
    gray_scale = cv2.cvtColor(image_arr, cv2.COLOR_BGR2GRAY)
    (thresh, im_bw) = cv2.threshold(gray_scale, 128, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)
    return im_bw


def image_cropper(image_arr: np.ndarray, x: int, y: int, width: int, height: int) -> np.ndarray:
    cropped_image = image_arr[x:width, y:height]
    return cropped_image


def image_resizer(image_arr: np.ndarray) -> np.ndarray:
    resized_image = cv2.resize(image_arr, (48, 48), cv2.INTER_AREA)
    return resized_image


def erosion_dilation(image_arr: np.ndarray) -> np.ndarray:
    invert: np.ndarray = 255 - image_arr
    kernel = np.ones((5, 5), np.uint8)
    eroded = cv2.erode(invert, kernel, iterations=1)
    dilated = cv2.dilate(eroded, kernel, iterations=1)
    dilated = 255 - dilated
    return dilated


def letter_finder(path: str):
    for image in os.listdir(path=path):
        image_path = "\\".join([path, image])
        image_arr = image_loader(path=image_path)
        gray_scale = black_white_converter(image_arr=image_arr)
        coordinates = find_non_white_pixels(image_arr=gray_scale)
        cropped_image = image_cropper(image_arr=gray_scale, x=coordinates[0]["x"],
                                      y=coordinates[0]["y"],
                                      width=coordinates[1]["x"], height=coordinates[1]["y"])
        dilated_arr = erosion_dilation(image_arr=cropped_image)
        resized_image = image_resizer(image_arr=dilated_arr)

        image_out = im.fromarray(resized_image)
        output_path = "\\".join([FileStructure.TRAINING_IMAGES_PATH.value, image])
        image_out.save(output_path)
        print(image + " done")
