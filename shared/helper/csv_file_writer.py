import pandas as pd

from domain.models.file_structure import FileStructure


def write_training_set_digit_letter(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_DIGIT_LETTER_PATH.value)
    dataframe.to_csv(path)


def write_training_set_even_odd(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_EVEN_ODD__PATH.value)
    dataframe.to_csv(path)


def write_training_even_values(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_EVEN_PATH.value)
    dataframe.to_csv(path)


def write_training_odd_values(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_ODD_PATH.value)
    dataframe.to_csv(path)


def write_training_upper_lower(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_UPPER_LOWER_PATH.value)
    dataframe.to_csv(path)


def write_training_upper_values(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_UPPER_PATH.value)
    dataframe.to_csv(path)


def write_training_lower_values(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_LOWER_PATH.value)
    dataframe.to_csv(path)


def write_training_lower_classes(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_LOWER_CLASSES_PATH.value)
    dataframe.to_csv(path)


def write_training_lower_class_one(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_LOWER_CLASS_ONE.value)
    dataframe.to_csv(path)


def write_training_lower_class_two(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_LOWER_CLASS_TWO.value)
    dataframe.to_csv(path)


def write_training_lower_class_three(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_LOWER_CLASS_THREE.value)
    dataframe.to_csv(path)


def write_training_lower_class_four(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_LOWER_CLASS_FOUR.value)
    dataframe.to_csv(path)


def write_training_upper_classes(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_UPPER_CLASSES_PATH.value)
    dataframe.to_csv(path)


def write_training_upper_class_one(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_UPPER_CLASS_ONE.value)
    dataframe.to_csv(path)


def write_training_upper_class_two(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_UPPER_CLASS_TWO.value)
    dataframe.to_csv(path)


def write_training_upper_class_three(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_UPPER_CLASS_THREE.value)
    dataframe.to_csv(path)


def write_training_upper_class_four(dataframe: pd.DataFrame):
    path: str = str(FileStructure.VECTOR_UPPER_CLASS_FOUR.value)
    dataframe.to_csv(path)

def write_to_csv_file(path_to_csv_file: str, dataframe: pd.DataFrame):
    dataframe.to_csv(path_to_csv_file)
