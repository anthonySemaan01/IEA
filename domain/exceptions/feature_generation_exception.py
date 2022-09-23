from domain.exceptions.application_error import ApplicationError


class FeatureGeneration(ApplicationError):
    def __init__(self, additional_message:str):
        """"
        raised when an error occur during saving/reading files"""
        super().__init__(default_message="Feature Generation error", additional_message=additional_message)