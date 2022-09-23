import os
import pandas as pd
from application.feature_extraction.feature_extractor import FeatureExtractor
from domain.models.file_structure import FileStructure
import os
import typing

extractor = FeatureExtractor(FileStructure.TRAINING_IMAGES_PATH.value)


def feature_generation_training():
    x_train: list = extractor.extract_features(path_to_directory=extractor.path)
    saved_x_train = pd.DataFrame(x_train)
    saved_x_train.to_csv(str(FileStructure.VECTOR_PATH.value))
    return x_train
