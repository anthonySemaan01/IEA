import typing

import numpy as np
import cv2
import os

from domain.contracts.abstract_feature_extractor import AbstractFeatureExtractor
from domain.exceptions.feature_generation_exception import FeatureGeneration
from domain.exceptions.feature_extraction_exception import FeatureExtraction


class FeatureExtractor(AbstractFeatureExtractor):
    threshold = 200

    def __init__(self, path):
        self.path = path

    def get_total_black_pixels(self, path):
        img = cv2.imread(path)
        number_of_black_pix = np.sum(img < 255 - self.threshold)  # extracting only black pixels
        return number_of_black_pix

    def get_total_white_pixels(self, path):
        img = cv2.imread(path)
        number_of_white_pix = np.sum(img >= self.threshold)  # extracting only white pixels
        return number_of_white_pix

    def get_total_left_pixels(self, path):
        img = cv2.imread(path)
        total_left = img[:, :14]
        total_left = np.sum(total_left >= self.threshold)
        return total_left

    def get_total_right_pixels(self, path):
        img = cv2.imread(path)
        total_right = img[:, 14:]
        total_right = np.sum(total_right >= self.threshold)
        return total_right

    def get_total_up_pixels(self, path):
        img = cv2.imread(path)
        total_up = img[:14, :]
        total_up = np.sum(total_up >= self.threshold)
        return total_up

    def get_total_down_pixels(self, path):
        img = cv2.imread(path)
        total_down = img[14:, :]
        total_down = np.sum(total_down >= self.threshold)
        return total_down

    def get_sub_pixels1(self, path):
        img = cv2.imread(path)
        sub_area = img[:4, :4]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels2(self, path):
        img = cv2.imread(path)
        sub_area = img[4:8, :4]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels3(self, path):
        img = cv2.imread(path)
        sub_area = img[8:12, :4]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels4(self, path):
        img = cv2.imread(path)
        sub_area = img[12:16, :4]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels5(self, path):
        img = cv2.imread(path)
        sub_area = img[16:20, :4]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels6(self, path):
        img = cv2.imread(path)
        sub_area = img[20:24, :4]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels7(self, path):
        img = cv2.imread(path)
        sub_area = img[24:28, :4]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels8(self, path):
        img = cv2.imread(path)
        sub_area = img[:4, 4:8]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels9(self, path):
        img = cv2.imread(path)
        sub_area = img[4:8, 4:8]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels10(self, path):
        img = cv2.imread(path)
        sub_area = img[8:12, 4:8]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels11(self, path):
        img = cv2.imread(path)
        sub_area = img[12:16, 4:8]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels12(self, path):
        img = cv2.imread(path)
        sub_area = img[16:20, 4:8]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels13(self, path):
        img = cv2.imread(path)
        sub_area = img[20:24, 4:8]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels14(self, path):
        img = cv2.imread(path)
        sub_area = img[24:28, 4:8]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels15(self, path):
        img = cv2.imread(path)
        sub_area = img[:4, 8:12]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels16(self, path):
        img = cv2.imread(path)
        sub_area = img[4:8, 8:12]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels17(self, path):
        img = cv2.imread(path)
        sub_area = img[8:12, 8:12]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels18(self, path):
        img = cv2.imread(path)
        sub_area = img[12:16, 8:12]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels19(self, path):
        img = cv2.imread(path)
        sub_area = img[16:20, 8:12]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels20(self, path):
        img = cv2.imread(path)
        sub_area = img[20:24, 8:12]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels21(self, path):
        img = cv2.imread(path)
        sub_area = img[24:28, 8:12]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels22(self, path):
        img = cv2.imread(path)
        sub_area = img[:4, 12:16]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels23(self, path):
        img = cv2.imread(path)
        sub_area = img[4:8, 12:16]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels24(self, path):
        img = cv2.imread(path)
        sub_area = img[8:12, 12:16]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels25(self, path):
        img = cv2.imread(path)
        sub_area = img[12:16, 12:16]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels26(self, path):
        img = cv2.imread(path)
        sub_area = img[16:20, 12:16]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels27(self, path, ):
        img = cv2.imread(path)
        sub_area = img[20:24, 12:16]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels28(self, path):
        img = cv2.imread(path)
        sub_area = img[24:28, 12:16]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels29(self, path):
        img = cv2.imread(path)
        sub_area = img[:4, 16:20]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels30(self, path):
        img = cv2.imread(path)
        sub_area = img[4:8, 16:20]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels31(self, path):
        img = cv2.imread(path)
        sub_area = img[8:12, 16:20]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels32(self, path):
        img = cv2.imread(path)
        sub_area = img[12:16, 16:20]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels33(self, path):
        img = cv2.imread(path)
        sub_area = img[16:20, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels34(self, path):
        img = cv2.imread(path)
        sub_area = img[20:24, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels35(self, path):
        img = cv2.imread(path)
        sub_area = img[24:28, 16:20]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels36(self, path):
        img = cv2.imread(path)
        sub_area = img[:4, 20:24]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels37(self, path):
        img = cv2.imread(path)
        sub_area = img[4:8, 20:24]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels38(self, path):
        img = cv2.imread(path)
        sub_area = img[8:12, 20:24]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels39(self, path):
        img = cv2.imread(path)
        sub_area = img[12:16, 20:24]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels40(self, path):
        img = cv2.imread(path)
        sub_area = img[16:20, 20:24]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels41(self, path):
        img = cv2.imread(path)
        sub_area = img[20:24, 20:24]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels42(self, path):
        img = cv2.imread(path)
        sub_area = img[24:28, 20:24]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels43(self, path):
        img = cv2.imread(path)
        sub_area = img[:4, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels44(self, path):
        img = cv2.imread(path)
        sub_area = img[4:8, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels45(self, path):
        img = cv2.imread(path)
        sub_area = img[8:12, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels46(self, path):
        img = cv2.imread(path)
        sub_area = img[12:16, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels47(self, path):
        img = cv2.imread(path)
        sub_area = img[16:20, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels48(self, path):
        img = cv2.imread(path)
        sub_area = img[20:24, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def get_sub_pixels49(self, path):
        img = cv2.imread(path)
        sub_area = img[24:28, 24:28]
        sub_area = np.sum(sub_area >= self.threshold)
        return sub_area

    def extract_features(self, path_to_directory: str):
        output_vector: list = []
        vector: list = []

        for image in os.listdir(path_to_directory):
            try:
                path = str(os.path.join(path_to_directory, image))
                vector.append(self.get_total_black_pixels(path))
                vector.append(self.get_total_white_pixels(path))
                vector.append(self.get_total_left_pixels(path))
                vector.append(self.get_total_right_pixels(path))
                vector.append(self.get_total_up_pixels(path))
                vector.append(self.get_total_down_pixels(path))
                vector.append(self.get_sub_pixels1(path))
                vector.append(self.get_sub_pixels2(path))
                vector.append(self.get_sub_pixels3(path))
                vector.append(self.get_sub_pixels4(path))
                vector.append(self.get_sub_pixels5(path))
                vector.append(self.get_sub_pixels6(path))
                vector.append(self.get_sub_pixels7(path))
                vector.append(self.get_sub_pixels8(path))
                vector.append(self.get_sub_pixels9(path))
                vector.append(self.get_sub_pixels10(path))
                vector.append(self.get_sub_pixels11(path))
                vector.append(self.get_sub_pixels12(path))
                vector.append(self.get_sub_pixels13(path))
                vector.append(self.get_sub_pixels14(path))
                vector.append(self.get_sub_pixels15(path))
                vector.append(self.get_sub_pixels16(path))
                vector.append(self.get_sub_pixels17(path))
                vector.append(self.get_sub_pixels18(path))
                vector.append(self.get_sub_pixels19(path))
                vector.append(self.get_sub_pixels20(path))
                vector.append(self.get_sub_pixels21(path))
                vector.append(self.get_sub_pixels22(path))
                vector.append(self.get_sub_pixels23(path))
                vector.append(self.get_sub_pixels24(path))
                vector.append(self.get_sub_pixels25(path))
                vector.append(self.get_sub_pixels26(path))
                vector.append(self.get_sub_pixels27(path))
                vector.append(self.get_sub_pixels28(path))
                vector.append(self.get_sub_pixels29(path))
                vector.append(self.get_sub_pixels30(path))
                vector.append(self.get_sub_pixels31(path))
                vector.append(self.get_sub_pixels32(path))
                vector.append(self.get_sub_pixels33(path))
                vector.append(self.get_sub_pixels34(path))
                vector.append(self.get_sub_pixels35(path))
                vector.append(self.get_sub_pixels36(path))
                vector.append(self.get_sub_pixels37(path))
                vector.append(self.get_sub_pixels38(path))
                vector.append(self.get_sub_pixels39(path))
                vector.append(self.get_sub_pixels40(path))
                vector.append(self.get_sub_pixels41(path))
                vector.append(self.get_sub_pixels42(path))
                vector.append(self.get_sub_pixels43(path))
                vector.append(self.get_sub_pixels44(path))
                vector.append(self.get_sub_pixels45(path))
                vector.append(self.get_sub_pixels46(path))
                vector.append(self.get_sub_pixels47(path))
                vector.append(self.get_sub_pixels48(path))
                vector.append(self.get_sub_pixels49(path))
                output_vector = vector
            except Exception as e:
                raise FeatureExtraction(additional_message=e.__str__())

        return output_vector
