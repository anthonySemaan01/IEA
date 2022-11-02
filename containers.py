from dependency_injector import containers, providers
from domain.contracts.abstract_letter_fine_tuner import AbstractLetterFineTuner
from domain.contracts.abstract_feature_extractor import AbstractFeatureExtractor
from application.image_preprocessing.letter_fine_tuner import LetterFineTuner
from application.feature_extraction.feature_extractor1 import FeatureExtractor1
from application.feature_extraction.feature_extractor2 import FeatureExtractor2


class Services(containers.DeclarativeContainer):

    # extractor : preprocessing stage
    letter_fine_tuner = providers.Factory(
        AbstractLetterFineTuner.register(LetterFineTuner))

    # extracts features vectors
    feature_generation1 = providers.Factory(
        AbstractFeatureExtractor.register(FeatureExtractor1))
    feature_generation2 = providers.Factory(
        AbstractFeatureExtractor.register(FeatureExtractor2))
