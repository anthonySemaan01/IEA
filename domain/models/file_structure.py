import os
from enum import Enum


class FileStructure(Enum):
    IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "images"])
    GRAY_SCALE_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "gray_scale_images"])
    CROPPED_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "cropped_images"])
    RESIZED_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "resized_images"])
    ERODED_DILATED_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "eroded_dilated"])
    PROCESSED_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "processed"])
    TRAINING_IMAGES_PATH = "\\".join([os.path.abspath(os.curdir), "datasets", "training", "images"])
    TRAINING_UBYTE_IMAGES_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "images-ubyte", "emnist-letters-train-images-idx3-ubyte"])
    TESTING_IMAGES_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "testing"])
    BBOX_IMAGE_TESTING = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "bbox_image_testing"])
    VECTOR_DIGIT_LETTER_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_digit_letter.csv"])
    VECTOR_EVEN_ODD__PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_even_odd.csv"])
    VECTOR_EVEN_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_even_values.csv"])
    VECTOR_ODD_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_odd_values.csv"])
    VECTOR_UPPER_LOWER_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_lower.csv"])
    VECTOR_UPPER_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_values.csv"])

    VECTOR_LOWER_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_lower_values.csv"])

    VECTOR_LOWER_CLASSES_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_lower_classes.csv"])
    VECTOR_LOWER_CLASS_ONE = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_lower_class_one_values.csv"])
    VECTOR_LOWER_CLASS_TWO = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_lower_class_two_values.csv"])
    VECTOR_LOWER_CLASS_THREE = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_lower_class_three_values.csv"])
    VECTOR_LOWER_CLASS_FOUR = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_lower_class_four_values.csv"])

    VECTOR_UPPER_CLASSES_PATH = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_classes.csv"])
    VECTOR_UPPER_CLASS_ONE = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_class_one_values.csv"])
    VECTOR_UPPER_CLASS_TWO = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_class_two_values.csv"])
    VECTOR_UPPER_CLASS_THREE = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_class_three_values.csv"])
    VECTOR_UPPER_CLASS_FOUR = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector", "training_set_upper_class_four_values.csv"])

    SAV_DT_DIGIT_LETTER = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "DT_digit_letter.sav"])
    SAV_DT_EVEN_ODD = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "DT_even_odd.sav"])
    SAV_DT_EVEN_VALUES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "DT_even_values.sav"])
    SAV_DT_LOWER_VALUES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "DT_lower_values.sav"])
    SAV_DT_ODD_VALUES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "DT_odd_values.sav"])
    SAV_DT_UPPER_LOWER = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "DT_upper_lower.sav"])
    SAV_DT_UPPER_VALUES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "DT_upper_values.sav"])

    SAV_KNN_DIGIT_LETTER = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "KNN_digit_letter.sav"])
    SAV_KNN_EVEN_ODD = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "KNN_even_odd.sav"])
    SAV_KNN_EVEN_VALUES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "KNN_even_values.sav"])
    SAV_KNN_LOWER_VALUES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "KNN_lower_values.sav"])
    SAV_KNN_ODD_VALUES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "KNN_odd_values.sav"])
    SAV_KNN_UPPER_LOWER = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "KNN_upper_lower.sav"])
    SAV_KNN_UPPER_VALUES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "KNN_upper_values.sav"])

    SAV_SVM_DIGIT_LETTER = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "SVM_digit_letter.sav"])
    SAV_SVM_EVEN_ODD = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "SVM_even_odd.sav"])
    SAV_SVM_UPPER_LOWER = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "sav_files", "SVM_upper_lower.sav"])

    VECTOR_CLASSES = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "classes.csv"])

    VECTOR_CLASSE1_MODELS = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class1_models.csv"])
    VECTOR_CLASSE1_MODEL1 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class1_model1.csv"])
    VECTOR_CLASSE1_MODEL2 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class1_model2.csv"])
    VECTOR_CLASSE1_MODEL3 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class1_model3.csv"])

    VECTOR_CLASSE2_MODELS = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class2_models.csv"])
    VECTOR_CLASSE2_MODEL1 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class2_model1.csv"])
    VECTOR_CLASSE2_MODEL2 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class2_model2.csv"])
    VECTOR_CLASSE2_MODEL11 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class2_model11.csv"])
    VECTOR_CLASSE2_MODEL12 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class2_model12.csv"])

    VECTOR_CLASSE3_MODELS = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class3_models.csv"])
    VECTOR_CLASSE3_MODEL1 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class3_model1.csv"])
    VECTOR_CLASSE3_MODEL2 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class3_model2.csv"])

    VECTOR_CLASSE4_MODEL1 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class4_model1.csv"])

    VECTOR_CLASSE5_MODELS = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class5_models.csv"])
    VECTOR_CLASSE5_MODEL1 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class5_model1.csv"])
    VECTOR_CLASSE5_MODEL2 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class5_model2.csv"])
    VECTOR_CLASSE5_MODEL11 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class5_model11.csv"])
    VECTOR_CLASSE5_MODEL12 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class5_model12.csv"])

    VECTOR_CLASSE6_MODELS = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class6_models.csv"])
    VECTOR_CLASSE6_MODEL1 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class6_model1.csv"])
    VECTOR_CLASSE6_MODEL2 = "\\".join(
        [os.path.abspath(os.curdir), "datasets", "training", "vector2", "class6_model2.csv"])

