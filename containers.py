from dependency_injector import containers, providers
from domain.contracts.abstract_letter_fine_tuner import AbstractLetterFineTuner
from application.image_preprocessing.letter_fine_tuner import LetterFineTuner


class Services(containers.DeclarativeContainer):

    # extractor
    letter_fine_tuner = providers.Factory(
        AbstractLetterFineTuner.register(LetterFineTuner))
