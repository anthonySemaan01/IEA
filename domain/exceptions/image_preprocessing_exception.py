from domain.exceptions.application_error import ApplicationError


class ImagePreprocessingException(ApplicationError):
    def __init__(self, additional_message: str):
        super().__init__(default_message="An error occurred while preprocessing the Image",
                         additional_message=additional_message)
