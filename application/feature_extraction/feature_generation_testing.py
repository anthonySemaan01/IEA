import os
import pandas as pd
from application.feature_extraction.feature_extractor import FeatureExtractor
from domain.models.file_structure import FileStructure
import os
import typing
from domain.exceptions.feature_generation_exception import FeatureGeneration
from domain.exceptions.feature_extraction_exception import FeatureExtraction
from containers import Services

extractor = Services.feature_generation(FileStructure.TESTING_IMAGES_PATH.value)


def feature_generation_test(retrain: bool = False):
    try:
        x_test: list = extractor.extract_features(path_to_directory=extractor.path)
        if retrain:
            saved_x_train_csv = pd.read_csv(str(FileStructure.VECTOR_PATH.value))
            x_dataframe = pd.DataFrame(saved_x_train_csv)
            x_dataframe.loc[len(x_dataframe.index)] = x_test
            x_dataframe.to_csv(str(FileStructure.VECTOR_PATH.value))
        return x_test
    except Exception as e:
        raise FeatureGeneration(additional_message=e.__str__())

