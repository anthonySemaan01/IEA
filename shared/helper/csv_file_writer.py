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
