import os
from typing import List

import cv2
import numpy as np
import pandas as pd

# np.set_printoptions(threshold=sys.maxsize)


threshold = 127  # Below is black, above is white since values are already Grayscale


# 1 White/Black
def get_total_black_pixels(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    number_of_black_pix = np.sum(img <= threshold)  # extracting only black pixels
    return number_of_black_pix


def get_total_white_pixels(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    number_of_white_pix = np.sum(img > threshold)  # extracting only white pixels
    return number_of_white_pix

    # 2 Left / Right


def get_total_left_pixels(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_left = img[:, :16]
    total_left = np.sum(total_left <= threshold)  # return total black pixels in that zone # white
    # background, black writing line
    return total_left


def get_total_right_pixels(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_right = img[:, 16:]
    total_right = np.sum(total_right <= threshold)
    return total_right

    # 3 Up / Down


def get_total_up_pixels(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_up = img[:16, :]
    total_up = np.sum(total_up <= threshold)
    return total_up


def get_total_down_pixels(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    total_down = img[16:, :]
    total_down = np.sum(total_down <= threshold)
    return total_down

    # 4 Horizontal four zones


def get_total_up_left_pixels_horizontal(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_min = 0
    col_max = 32
    row_min = 0
    row_max = 8
    black_pixels_in_vector = 0
    while col_max > 0:
        vector = img[col_min:col_max, row_min:row_max]
        black_pixels_in_vector = black_pixels_in_vector + np.sum(vector <= threshold)
        col_max = col_max - 8
        row_min = row_min + 8
        row_max = row_max + 8
    return black_pixels_in_vector


def get_total_down_right_pixels_horizontal(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_min = 24
    col_max = 32
    row_min = 0
    row_max = 8
    black_pixels_in_vector = 0
    while col_min >= 0:
        vector = img[col_min:col_max, row_min:row_max]
        black_pixels_in_vector = black_pixels_in_vector + np.sum(vector <= threshold)
        col_min = col_min - 8
        row_min = row_min + 8
        row_max = row_max + 8
    return black_pixels_in_vector


def get_total_up_right_pixels_horizontal(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_min = 0
    col_max = 8
    row_min = 0
    row_max = 8
    black_pixels_in_vector = 0
    while col_max <= 32:
        vector = img[col_min:col_max, row_min:row_max]
        black_pixels_in_vector = black_pixels_in_vector + np.sum(vector <= threshold)
        col_max = col_max + 8
        row_min = row_min + 8
        row_max = row_max + 8
    return black_pixels_in_vector


def get_total_down_left_pixels_horizontal(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col_min = 0
    col_max = 8
    row_min = 24
    row_max = 32
    black_pixels_in_vector = 0
    while row_max > 0:
        vector = img[col_min:col_max, row_min:row_max]
        black_pixels_in_vector = black_pixels_in_vector + np.sum(vector <= threshold)
        col_max = col_max + 8
        row_min = row_min - 8
        row_max = row_max - 8
    return black_pixels_in_vector

    # 5 cols traverse


def get_nb_of_black_pixels_col1(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col1 = img[0:32, 0:8]
    nb_black_pixels = np.sum(col1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_col2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col1 = img[0:32, 8:16]
    nb_black_pixels = np.sum(col1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_col3(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col1 = img[0:32, 16:24]
    nb_black_pixels = np.sum(col1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_col4(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    col1 = img[0:32, 24:32]
    nb_black_pixels = np.sum(col1 <= threshold)
    return nb_black_pixels

    # 6 rows traverse


def get_nb_of_black_pixels_row1(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row1 = img[0:8, 0:32]
    nb_black_pixels = np.sum(row1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_row2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row2 = img[8:16, 0:32]
    nb_black_pixels = np.sum(row2 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_row3(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row3 = img[16:24, 0:32]
    nb_black_pixels = np.sum(row3 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_row4(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    row4 = img[24:32, 0:32]
    nb_black_pixels = np.sum(row4 <= threshold)
    return nb_black_pixels

    # 7 Horizontal traverse


def get_nb_of_black_pixels_diag1(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diag1 = np.diag(img, -1)
    nb_black_pixels = np.sum(diag1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_diag2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diag1 = np.diag(img, -2)
    nb_black_pixels = np.sum(diag1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_diag3(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diag1 = np.diag(img, -3)
    nb_black_pixels = np.sum(diag1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_diag4(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diag1 = np.diag(img)
    nb_black_pixels = np.sum(diag1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_diag5(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diag1 = np.diag(img, 1)
    nb_black_pixels = np.sum(diag1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_diag6(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diag1 = np.diag(img, 2)
    nb_black_pixels = np.sum(diag1 <= threshold)
    return nb_black_pixels


def get_nb_of_black_pixels_diag7(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    diag1 = np.diag(img, 3)
    nb_black_pixels = np.sum(diag1 <= threshold)
    return nb_black_pixels

    #  8 Nb of Black pixels in each sub area


def get_sub_pixels1(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, :8]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels2(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, :8]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels3(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, :8]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels4(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, :8]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels5(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, 8:16]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels6(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 8:16]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels7(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 8:16]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels8(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 8:16]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels9(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, 16:24]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels10(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 16:24]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels11(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 16:24]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels12(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 16:24]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels13(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[0:8, 24:32]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels14(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 24:32]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels15(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 24:32]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area


def get_sub_pixels16(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 24:32]
    sub_area = np.sum(sub_area <= threshold)
    return sub_area

    # 9 Number of black pixels in each ROW OF EACH SUBAREA


def get_sub_pixels1_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, :8]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels2_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, :8]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels3_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, :8]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels4_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, :8]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels5_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, 8:16]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels6_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 8:16]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels7_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 8:16]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels8_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 8:16]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels9_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, 16:24]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels10_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 16:24]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels11_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 16:24]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels12_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 16:24]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels13_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[0:8, 24:32]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels14_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 24:32]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels15_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 24:32]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels


def get_sub_pixels16_rows(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 24:32]
    rows_black_pixels = []
    for row_idx in range(0, 8):
        row = sub_area[row_idx]
        nb_of_black_pixels = np.sum(row <= threshold)
        rows_black_pixels.append(nb_of_black_pixels)
    return rows_black_pixels

    # 10 Number of black pixels in each COLUMN OF EACH SUBAREA


def get_sub_pixels1_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, :8]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels2_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, :8]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels3_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, :8]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels4_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, :8]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels5_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, 8:16]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels6_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 8:16]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels7_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 8:16]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels8_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 8:16]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels9_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, 16:24]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels10_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 16:24]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels11_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 16:24]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels12_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 16:24]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels13_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[0:8, 24:32]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels14_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 24:32]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels15_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 24:32]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels


def get_sub_pixels16_cols(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 24:32]
    cols_black_pixels = []
    for col_idx in range(0, 8):
        col = sub_area[:, col_idx]
        nb_of_black_pixels = np.sum(col <= threshold)
        cols_black_pixels.append(nb_of_black_pixels)
    return cols_black_pixels

    # 11 Number of black pixels in each HORIZONTAL TRAVERSE OF EACH SUBAREA


def get_sub_pixels1_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, :8]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels2_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, :8]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels3_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, :8]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels4_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, :8]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels5_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, 8:16]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels6_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 8:16]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels7_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 8:16]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels8_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 8:16]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels9_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[:8, 16:24]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels10_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 16:24]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels11_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 16:24]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels12_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 16:24]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels13_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[0:8, 24:32]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels14_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[8:16, 24:32]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels15_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[16:24, 24:32]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_sub_pixels16_horiz(path):
    img = cv2.imread(path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    sub_area = img[24:32, 24:32]
    horiz_black_pixels = []
    for horiz_idx in range(-7, 8):
        horiz = np.diag(sub_area, horiz_idx)
        nb_of_black_pixels = np.sum(horiz <= threshold)
        horiz_black_pixels.append(nb_of_black_pixels)
    return horiz_black_pixels


def get_edges(path) -> list:
    arr = cv2.imread(path)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(arr, 100, 200)
    return edges


def compute_FFT(path) -> list:
    arr = cv2.imread(path)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    f = np.fft.fft2(arr, [10, 10])  # THE SECOND PARAM IS FOR N*N size of returned 2D FFT vector
    fshift = np.fft.fftshift(f)
    magnitude_spectrum = 20 * np.log(np.abs(fshift))
    return magnitude_spectrum

    # 15 Keypoint detector (black/white blobs orientation)


def compute_nb_SIFT_keypoints(path) -> list:
    arr = cv2.imread(path)
    arr = cv2.cvtColor(arr, cv2.COLOR_BGR2GRAY)
    sift = cv2.SIFT_create(400)
    kp, des = sift.detectAndCompute(arr, None)
    # numberOfKeyPoints = len(kp)

    if des is not None:
        first_KP_desc = des[0]
    else:
        first_KP_desc = [0]
    first_KP_desc_arr = []
    count = 0
    for desc in first_KP_desc:
        if (count == 119):
            break
        first_KP_desc_arr.append(int(desc))
        count = count + 1
    if len(first_KP_desc_arr) == 0:
        return []
    return first_KP_desc_arr

    # 12 directional distribution


def compute_directional_distribution(path) -> list:
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


def extract_features(path_to_directory: str):
    output_vector: list = []
    vector: list = []

    for image in os.listdir(path_to_directory):
        path = str(os.path.join(path_to_directory, image))
        print("path:", path)
        print(image[3:6])
        if image[3:6] == "001":
            vector.append("0")

        if image[3:6] == "002":
            vector.append("1")

        if image[3:6] == "003":
            vector.append("2")

        if image[3:6] == "004":
            vector.append("3")

        if image[3:6] == "005":
            vector.append("4")

        if image[3:6] == "006":
            vector.append("5")

        if image[3:6] == "007":
            vector.append("6")

        if image[3:6] == "008":
            vector.append("7")

        if image[3:6] == "009":
            vector.append("8")

        if image[3:6] == "010":
            vector.append("9")

        if image[3:6] == "011":
            vector.append("A")

        if image[3:6] == "012":
            vector.append("B")

        if image[3:6] == "013":
            vector.append("C")

        if image[3:6] == "014":
            vector.append("D")

        if image[3:6] == "015":
            vector.append("E")

        if image[3:6] == "016":
            vector.append("F")

        if image[3:6] == "017":
            vector.append("G")

        if image[3:6] == "018":
            vector.append("H")

        if image[3:6] == "019":
            vector.append("I")

        if image[3:6] == "020":
            vector.append("J")

        if image[3:6] == "021":
            vector.append("K")

        if image[3:6] == "022":
            vector.append("L")

        if image[3:6] == "023":
            vector.append("M")

        if image[3:6] == "024":
            vector.append("N")

        if image[3:6] == "025":
            vector.append("O")

        if image[3:6] == "026":
            vector.append("P")

        if image[3:6] == "027":
            vector.append("Q")

        if image[3:6] == "028":
            vector.append("R")

        if image[3:6] == "029":
            vector.append("S")

        if image[3:6] == "030":
            vector.append("T")

        if image[3:6] == "031":
            vector.append("U")

        if image[3:6] == "032":
            vector.append("V")

        if image[3:6] == "033":
            vector.append("W")

        if image[3:6] == "034":
            vector.append("X")

        if image[3:6] == "035":
            vector.append("Y")

        if image[3:6] == "036":
            vector.append("Z")

        if image[3:6] == "037":
            vector.append("a")

        if image[3:6] == "038":
            vector.append("b")

        if image[3:6] == "039":
            vector.append("c")

        if image[3:6] == "040":
            vector.append("d")

        if image[3:6] == "041":
            vector.append("e")

        if image[3:6] == "042":
            vector.append("f")

        if image[3:6] == "043":
            vector.append("g")

        if image[3:6] == "044":
            vector.append("h")

        if image[3:6] == "045":
            vector.append("i")

        if image[3:6] == "046":
            vector.append("j")

        if image[3:6] == "047":
            vector.append("k")

        if image[3:6] == "048":
            vector.append("l")

        if image[3:6] == "049":
            vector.append("m")

        if image[3:6] == "050":
            vector.append("n")

        if image[3:6] == "051":
            vector.append("o")

        if image[3:6] == "052":
            vector.append("p")

        if image[3:6] == "053":
            vector.append("q")

        if image[3:6] == "054":
            vector.append("r")

        if image[3:6] == "055":
            vector.append("s")

        if image[3:6] == "056":
            vector.append("t")

        if image[3:6] == "057":
            vector.append("u")

        if image[3:6] == "058":
            vector.append("v")

        if image[3:6] == "059":
            vector.append("w")

        if image[3:6] == "060":
            vector.append("x")

        if image[3:6] == "061":
            vector.append("y")

        if image[3:6] == "062":
            vector.append("z")

        vector.append(get_total_black_pixels(path))
        vector.append(get_total_white_pixels(path))

        vector.append(get_total_left_pixels(path))
        vector.append(get_total_right_pixels(path))

        vector.append(get_total_up_pixels(path))
        vector.append(get_total_down_pixels(path))

        vector.append(get_total_up_left_pixels_horizontal(path))
        vector.append(get_total_down_right_pixels_horizontal(path))
        vector.append(get_total_up_right_pixels_horizontal(path))
        vector.append(get_total_down_left_pixels_horizontal(path))

        vector.append(get_nb_of_black_pixels_col1(path))
        vector.append(get_nb_of_black_pixels_col2(path))
        vector.append(get_nb_of_black_pixels_col3(path))
        vector.append(get_nb_of_black_pixels_col4(path))

        vector.append(get_nb_of_black_pixels_row1(path))
        vector.append(get_nb_of_black_pixels_row2(path))
        vector.append(get_nb_of_black_pixels_row3(path))
        vector.append(get_nb_of_black_pixels_row4(path))

        vector.append(get_nb_of_black_pixels_diag1(path))
        vector.append(get_nb_of_black_pixels_diag2(path))
        vector.append(get_nb_of_black_pixels_diag3(path))
        vector.append(get_nb_of_black_pixels_diag4(path))
        vector.append(get_nb_of_black_pixels_diag5(path))
        vector.append(get_nb_of_black_pixels_diag6(path))
        vector.append(get_nb_of_black_pixels_diag7(path))

        # 4) First KP desc
        #  to increase the weight of this feature we could append it multiple times
        # first_KP_desc = compute_nb_SIFT_keypoints(path)
        # for desc in first_KP_desc:
        #     vector.append(desc)

        # 5) FFT magnitude vector
        # twoD_FFT_magn_vector = compute_FFT(path)
        # for row in twoD_FFT_magn_vector:
        #     for val in row:
        #         vector.append(int(val))

        # 6) edges detection
        twoD_edges_vector = get_edges(path)
        for row in twoD_edges_vector:
            for val in row:
                vector.append(int(val))

        print(len(vector))
        output_vector.append(vector)
        vector = []

    return output_vector


pathing = "datasets/training/images"
vec = extract_features(pathing)
df = pd.DataFrame(vec)
df.to_csv('datasets/training/ann2.csv')
