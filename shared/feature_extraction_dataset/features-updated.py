from typing import List

import numpy as np
import cv2
import os
import sys

import pandas as pd


class FeatureExtractor:
    threshold = 127  # Below is black, above is white since values are already Grayscale

    def __init__(self, path):
        self.path = path

    # 1 White/Black
    def get_total_black_pixels(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        number_of_black_pix = np.sum(img <= self.threshold)  # extracting only black pixels
        return number_of_black_pix

    def get_total_white_pixels(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        number_of_white_pix = np.sum(img > self.threshold)  # extracting only white pixels
        return number_of_white_pix

    # 2 Left / Right
    def get_total_left_pixels(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total_left = img[:, :16]
        total_left = np.sum(total_left <= self.threshold)  # return total black pixels in that zone # white
        # background, black writing line
        return total_left

    def get_total_right_pixels(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total_right = img[:, 16:]
        total_right = np.sum(total_right <= self.threshold)
        return total_right

    # 3 Up / Down
    def get_total_up_pixels(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total_up = img[:16, :]
        total_up = np.sum(total_up <= self.threshold)
        return total_up

    def get_total_down_pixels(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        total_down = img[16:, :]
        total_down = np.sum(total_down <= self.threshold)
        return total_down

    # 4 Horizontal four zones
    def get_total_up_left_pixels_horizontal(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_min = 0
        col_max = 32
        row_min = 0
        row_max = 8
        black_pixels_in_vector = 0
        while col_max > 0:
            vector = img[col_min:col_max, row_min:row_max]
            black_pixels_in_vector = black_pixels_in_vector + np.sum(vector <= self.threshold)
            col_max = col_max - 8
            row_min = row_min + 8
            row_max = row_max + 8
        return black_pixels_in_vector

    def get_total_down_right_pixels_horizontal(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_min = 24
        col_max = 32
        row_min = 0
        row_max = 8
        black_pixels_in_vector = 0
        while col_min >= 0:
            vector = img[col_min:col_max, row_min:row_max]
            black_pixels_in_vector = black_pixels_in_vector + np.sum(vector <= self.threshold)
            col_min = col_min - 8
            row_min = row_min + 8
            row_max = row_max + 8
        return black_pixels_in_vector

    def get_total_up_right_pixels_horizontal(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_min = 0
        col_max = 8
        row_min = 0
        row_max = 8
        black_pixels_in_vector = 0
        while col_max <= 32:
            vector = img[col_min:col_max, row_min:row_max]
            black_pixels_in_vector = black_pixels_in_vector + np.sum(vector <= self.threshold)
            col_max = col_max + 8
            row_min = row_min + 8
            row_max = row_max + 8
        return black_pixels_in_vector

    def get_total_down_left_pixels_horizontal(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col_min = 0
        col_max = 8
        row_min = 24
        row_max = 32
        black_pixels_in_vector = 0
        while row_max > 0:
            vector = img[col_min:col_max, row_min:row_max]
            black_pixels_in_vector = black_pixels_in_vector + np.sum(vector <= self.threshold)
            col_max = col_max + 8
            row_min = row_min - 8
            row_max = row_max - 8
        return black_pixels_in_vector

    # 5 cols traverse
    def get_nb_of_black_pixels_col1(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col1 = img[0:32, 0:8]
        nb_black_pixels = np.sum(col1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_col2(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col1 = img[0:32, 8:16]
        nb_black_pixels = np.sum(col1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_col3(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col1 = img[0:32, 16:24]
        nb_black_pixels = np.sum(col1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_col4(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        col1 = img[0:32, 24:32]
        nb_black_pixels = np.sum(col1 <= self.threshold)
        return nb_black_pixels

    # 6 rows traverse
    def get_nb_of_black_pixels_row1(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row1 = img[0:8, 0:32]
        nb_black_pixels = np.sum(row1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_row2(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row2 = img[8:16, 0:32]
        nb_black_pixels = np.sum(row2 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_row3(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row3 = img[16:24, 0:32]
        nb_black_pixels = np.sum(row3 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_row4(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        row4 = img[24:32, 0:32]
        nb_black_pixels = np.sum(row4 <= self.threshold)
        return nb_black_pixels

    # 7 Horizontal traverse
    def get_nb_of_black_pixels_diag1(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diag1 = np.diag(img, -1)
        nb_black_pixels = np.sum(diag1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_diag2(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diag1 = np.diag(img, -2)
        nb_black_pixels = np.sum(diag1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_diag3(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diag1 = np.diag(img, -3)
        nb_black_pixels = np.sum(diag1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_diag4(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diag1 = np.diag(img)
        nb_black_pixels = np.sum(diag1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_diag5(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diag1 = np.diag(img, 1)
        nb_black_pixels = np.sum(diag1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_diag6(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diag1 = np.diag(img, 2)
        nb_black_pixels = np.sum(diag1 <= self.threshold)
        return nb_black_pixels

    def get_nb_of_black_pixels_diag7(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        diag1 = np.diag(img, 3)
        nb_black_pixels = np.sum(diag1 <= self.threshold)
        return nb_black_pixels

    #  8 Nb of Black pixels in each sub area
    def get_sub_pixels1(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, :8]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels2(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, :8]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels3(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, :8]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels4(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, :8]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels5(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 8:16]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels6(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 8:16]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels7(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 8:16]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels8(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 8:16]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels9(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 16:24]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels10(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 16:24]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels11(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 16:24]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels12(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 16:24]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels13(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 24:32]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels14(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 24:32]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels15(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 24:32]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    def get_sub_pixels16(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 24:32]
        sub_area = np.sum(sub_area <= self.threshold)
        return sub_area

    # 9 Number of black pixels in each ROW OF EACH SUBAREA
    def get_sub_pixels1_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, :8]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels2_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, :8]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels3_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, :8]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels4_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, :8]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels5_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 8:16]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels6_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 8:16]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels7_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 8:16]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels8_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 8:16]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels9_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 16:24]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels10_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 16:24]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels11_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 16:24]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels12_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 16:24]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels13_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[0:8, 24:32]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels14_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 24:32]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels15_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 24:32]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

    def get_sub_pixels16_rows(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 24:32]
        rows_black_pixels = []
        for row_idx in range(0, 8):
            row = sub_area[row_idx]
            nb_of_black_pixels = np.sum(row <= self.threshold)
            rows_black_pixels.append(nb_of_black_pixels)
        return rows_black_pixels

        # 10 Number of black pixels in each COLUMN OF EACH SUBAREA

    def get_sub_pixels1_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, :8]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels2_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, :8]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels3_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, :8]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels4_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, :8]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels5_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 8:16]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels6_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 8:16]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels7_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 8:16]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels8_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 8:16]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels9_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 16:24]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels10_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 16:24]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels11_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 16:24]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels12_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 16:24]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels13_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[0:8, 24:32]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels14_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 24:32]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels15_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 24:32]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    def get_sub_pixels16_cols(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 24:32]
        cols_black_pixels = []
        for col_idx in range(0, 8):
            col = sub_area[:, col_idx]
            nb_of_black_pixels = np.sum(col <= self.threshold)
            cols_black_pixels.append(nb_of_black_pixels)
        return cols_black_pixels

    # 11 Number of black pixels in each HORIZONTAL TRAVERSE OF EACH SUBAREA
    def get_sub_pixels1_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, :8]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels2_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, :8]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels3_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, :8]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels4_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, :8]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels5_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 8:16]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels6_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 8:16]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels7_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 8:16]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels8_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 8:16]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels9_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 16:24]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels10_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 16:24]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels11_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 16:24]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels12_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 16:24]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels13_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[:8, 24:32]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels14_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[8:16, 24:32]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels15_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[16:24, 24:32]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    def get_sub_pixels16_horiz(self, path):
        img = cv2.imread(path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        sub_area = img[24:32, 24:32]
        horiz_black_pixels = []
        for horiz_idx in range(-7, 8):
            horiz = np.diag(sub_area, horiz_idx)
            nb_of_black_pixels = np.sum(horiz <= self.threshold)
            horiz_black_pixels.append(nb_of_black_pixels)
        return horiz_black_pixels

    # 12 directional distribution
    def compute_directional_distribution(self, path) -> list:
        arr = cv2.imread(path)
        arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
        arr = 255 - arr
        # print(arr)
        features = []
        zones: List[np.ndarray] = []
        zone_1 = arr[0:8, 0:8]
        zone_2 = arr[0:8, 8:16]
        zone_3 = arr[0:8, 16:24]
        zone_4 = arr[0:8, 24:32]
        zone_5 = arr[8:16, 0:8]
        zone_6 = arr[8:16, 8:16]
        zone_7 = arr[8:16, 16:24]
        zone_8 = arr[8:16, 24:32]
        zone_9 = arr[16:24, 0:8]
        zone_10 = arr[16:24, 8:16]
        zone_11 = arr[16:24, 16:24]
        zone_12 = arr[16:24, 24:32]
        zone_13 = arr[24:32, 0:8]
        zone_14 = arr[24:32, 8:16]
        zone_15 = arr[24:32, 16:24]
        zone_16 = arr[24:32, 24:32]

        zones.append(zone_1)
        zones.append(zone_2)
        zones.append(zone_3)
        zones.append(zone_4)
        zones.append(zone_5)
        zones.append(zone_6)
        zones.append(zone_7)
        zones.append(zone_8)
        zones.append(zone_9)
        zones.append(zone_10)
        zones.append(zone_11)
        zones.append(zone_12)
        zones.append(zone_13)
        zones.append(zone_14)
        zones.append(zone_15)
        zones.append(zone_16)

        for zone in zones:
            rows, columns = zone.shape
            d1 = 0
            d2 = 0
            d3 = 0
            d4 = 0
            d5 = 0
            d6 = 0
            d7 = 0
            d8 = 0
            for m in range(1, rows - 1):
                for n in range(1, columns - 1):
                    if zone[m][n] != 0:
                        d1 = d1 + 2 * zone[m][n + 1] + zone[m - 1][n + 1] + zone[m + 1][n + 1]
                        d2 = d2 + 2 * zone[m - 1][n + 1] + zone[m - 1][n] + zone[m][n + 1]
                        d3 = d3 + 2 * zone[m - 1][n] + zone[m - 1][n - 1] + zone[m - 1][n + 1]
                        d4 = d4 + 2 * zone[m - 1][n - 1] + zone[m][n - 1] + zone[m - 1][n]
                        d5 = d5 + 2 * zone[m][n - 1] + zone[m - 1][n - 1] + zone[m + 1][n - 1]
                        d6 = d6 + 2 * zone[m + 1][n - 1] + zone[m][n - 1] + zone[m + 1][n]
                        d7 = d7 + 2 * zone[m + 1][n] + zone[m + 1][n - 1] + zone[m + 1][n + 1]
                        d8 = d8 + 2 * zone[m + 1][n + 1] + zone[m + 1][n] + zone[m][n + 1]
            features.append(d1)
            features.append(d2)
            features.append(d3)
            features.append(d4)
            features.append(d5)
            features.append(d6)
            features.append(d7)
            features.append(d8)
        return features

    def extract_features(self, path_to_directory: str):
        output_vector: list = []
        vector: list = []
        dataset = list()

        for image in os.listdir(path_to_directory):
            try:
                path = str(os.path.join(path_to_directory, image))

                print(path)


                if path[-11:-8] == "001":
                    vector.append("0")

                if path[-11:-8] == "002":
                    vector.append("1")

                if path[-11:-8] == "003":
                    vector.append("2")

                if path[-11:-8] == "004":
                    vector.append("3")

                if path[-11:-8] == "005":
                    vector.append("4")

                if path[-11:-8] == "006":
                    vector.append("5")

                if path[-11:-8] == "007":
                    vector.append("6")

                if path[-11:-8] == "008":
                    vector.append("7")

                if path[-11:-8] == "009":
                    vector.append("8")

                if path[-11:-8] == "010":
                    vector.append("9")

                if path[-11:-8] == "011":
                    vector.append("A")

                if path[-11:-8] == "012":
                    vector.append("B")

                if path[-11:-8] == "013":
                    vector.append("C")

                if path[-11:-8] == "014":
                    vector.append("D")

                if path[-11:-8] == "015":
                    vector.append("E")

                if path[-11:-8] == "016":
                    vector.append("F")

                if path[-11:-8] == "017":
                    vector.append("G")

                if path[-11:-8] == "018":
                    vector.append("H")

                if path[-11:-8] == "019":
                    vector.append("I")

                if path[-11:-8] == "020":
                    vector.append("J")

                if path[-11:-8] == "021":
                    vector.append("K")

                if path[-11:-8] == "022":
                    vector.append("L")

                if path[-11:-8] == "023":
                    vector.append("M")

                if path[-11:-8] == "024":
                    vector.append("N")

                if path[-11:-8] == "025":
                    vector.append("O")

                if path[-11:-8] == "026":
                    vector.append("P")

                if path[-11:-8] == "027":
                    vector.append("Q")

                if path[-11:-8] == "028":
                    vector.append("R")

                if path[-11:-8] == "029":
                    vector.append("S")

                if path[-11:-8] == "030":
                    vector.append("T")

                if path[-11:-8] == "031":
                    vector.append("U")

                if path[-11:-8] == "032":
                    vector.append("V")

                if path[-11:-8] == "033":
                    vector.append("W")

                if path[-11:-8] == "034":
                    vector.append("X")

                if path[-11:-8] == "035":
                    vector.append("Y")

                if path[-11:-8] == "036":
                    vector.append("Z")

                if path[-11:-8] == "037":
                    vector.append("a")

                if path[-11:-8] == "038":
                    vector.append("b")

                if path[-11:-8] == "039":
                    vector.append("c")

                if path[-11:-8] == "040":
                    vector.append("d")

                if path[-11:-8] == "041":
                    vector.append("e")

                if path[-11:-8] == "042":
                    vector.append("f")

                if path[-11:-8] == "043":
                    vector.append("g")

                if path[-11:-8] == "044":
                    vector.append("h")

                if path[-11:-8] == "045":
                    vector.append("i")

                if path[-11:-8] == "046":
                    vector.append("j")

                if path[-11:-8] == "047":
                    vector.append("k")

                if path[-11:-8] == "048":
                    vector.append("l")

                if path[-11:-8] == "049":
                    vector.append("m")

                if path[-11:-8] == "050":
                    vector.append("n")

                if path[-11:-8] == "051":
                    vector.append("o")

                if path[-11:-8] == "052":
                    vector.append("p")

                if path[-11:-8] == "053":
                    vector.append("q")

                if path[-11:-8] == "054":
                    vector.append("r")

                if path[-11:-8] == "055":
                    vector.append("s")

                if path[-11:-8] == "056":
                    vector.append("t")

                if path[-11:-8] == "057":
                    vector.append("u")

                if path[-11:-8] == "058":
                    vector.append("v")

                if path[-11:-8] == "059":
                    vector.append("w")

                if path[-11:-8] == "060":
                    vector.append("x")

                if path[-11:-8] == "061":
                    vector.append("y")

                if path[-11:-8] == "062":
                    vector.append("z")

                print(vector)


                vector.append(self.get_total_black_pixels(path))
                vector.append(self.get_total_white_pixels(path))

                vector.append(self.get_total_left_pixels(path))
                vector.append(self.get_total_right_pixels(path))

                vector.append(self.get_total_up_pixels(path))
                vector.append(self.get_total_down_pixels(path))

                vector.append(self.get_total_up_left_pixels_horizontal(path))
                vector.append(self.get_total_down_right_pixels_horizontal(path))
                vector.append(self.get_total_up_right_pixels_horizontal(path))
                vector.append(self.get_total_down_left_pixels_horizontal(path))

                vector.append(self.get_nb_of_black_pixels_col1(path))
                vector.append(self.get_nb_of_black_pixels_col2(path))
                vector.append(self.get_nb_of_black_pixels_col3(path))
                vector.append(self.get_nb_of_black_pixels_col4(path))

                vector.append(self.get_nb_of_black_pixels_row1(path))
                vector.append(self.get_nb_of_black_pixels_row2(path))
                vector.append(self.get_nb_of_black_pixels_row3(path))
                vector.append(self.get_nb_of_black_pixels_row4(path))

                vector.append(self.get_nb_of_black_pixels_diag1(path))
                vector.append(self.get_nb_of_black_pixels_diag2(path))
                vector.append(self.get_nb_of_black_pixels_diag3(path))
                vector.append(self.get_nb_of_black_pixels_diag4(path))
                vector.append(self.get_nb_of_black_pixels_diag5(path))
                vector.append(self.get_nb_of_black_pixels_diag6(path))
                vector.append(self.get_nb_of_black_pixels_diag7(path))

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

                for x in range(0, 8):
                    res = self.get_sub_pixels1_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels2_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels3_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels4_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels5_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels6_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels7_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels8_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels9_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels10_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels11_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels12_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels13_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels14_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels15_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels16_rows(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels1_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels2_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels3_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels4_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels5_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels6_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels7_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels8_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels9_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels10_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels11_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels12_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels13_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels14_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels15_cols(path)[x]
                    vector.append(res)

                for x in range(0, 8):
                    res = self.get_sub_pixels16_cols(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels1_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels2_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels3_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels4_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels5_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels6_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels7_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels8_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels9_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels10_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels11_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels12_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels13_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels14_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels15_horiz(path)[x]
                    vector.append(res)

                for x in range(-7, 8):
                    res = self.get_sub_pixels16_horiz(path)[x]
                    vector.append(res)

                features = self.compute_directional_distribution(path)
                for feature in features:
                    vector.append(feature)

                print(len(vector))

                output_vector = vector
                dataset.append(vector)
                vector= list()

            except Exception as e:
                raise

        return dataset


extractor = FeatureExtractor('images')
test = extractor.extract_features('images')
df= pd.DataFrame(test)
df.to_csv('dataset-NEW.csv')