from dependency_injector import containers, providers
from domain.contracts.abstract_letter_fine_tuner import AbstractLetterFineTuner
from domain.contracts.abstract_feature_extractor import AbstractFeatureExtractor
from application.image_preprocessing.letter_fine_tuner import LetterFineTuner
from application.feature_extraction.feature_extractor import FeatureExtractor


class Services(containers.DeclarativeContainer):

    # extractor
    letter_fine_tuner = providers.Factory(
        AbstractLetterFineTuner.register(LetterFineTuner))

    feature_generation = providers.Factory(
        AbstractFeatureExtractor.register(FeatureExtractor))
