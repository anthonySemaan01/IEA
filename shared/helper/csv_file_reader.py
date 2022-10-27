import pandas as pd

from domain.models.file_structure import FileStructure


def read_training_set_digit_letter() -> pd.DataFrame:
    path: str = str(FileStructure.VECTOR_DIGIT_LETTER_PATH.value)
    csv_file = pd.read_csv(path, index_col=0)
    dataframe = pd.DataFrame(csv_file)
    return dataframe


def read_training_set_even_odd() -> pd.DataFrame:
    path: str = str(FileStructure.VECTOR_EVEN_ODD__PATH.value)
    csv_file = pd.read_csv(path, index_col=0)
    dataframe = pd.DataFrame(csv_file)
    return dataframe


def read_training_even_values() -> pd.DataFrame:
    path: str = str(FileStructure.VECTOR_EVEN_PATH.value)
    csv_file = pd.read_csv(path, index_col=0)
    dataframe = pd.DataFrame(csv_file)
    return dataframe


def read_training_odd_values() -> pd.DataFrame:
    path: str = str(FileStructure.VECTOR_ODD_PATH.value)
    csv_file = pd.read_csv(path, index_col=0)
    dataframe = pd.DataFrame(csv_file)
    return dataframe


def read_training_upper_lower() -> pd.DataFrame:
    path: str = str(FileStructure.VECTOR_UPPER_LOWER_PATH.value)
    csv_file = pd.read_csv(path, index_col=0)
    dataframe = pd.DataFrame(csv_file)
    return dataframe


def read_training_upper_values() -> pd.DataFrame:
    path: str = str(FileStructure.VECTOR_UPPER_PATH.value)
    csv_file = pd.read_csv(path, index_col=0)
    dataframe = pd.DataFrame(csv_file)
    return dataframe


def read_training_lower_values() -> pd.DataFrame:
    path: str = str(FileStructure.VECTOR_LOWER_PATH.value)
    csv_file = pd.read_csv(path, index_col=0)
    dataframe = pd.DataFrame(csv_file)
    return dataframe
