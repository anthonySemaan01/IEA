from abc import ABC, abstractmethod


class AbstractLetterFineTuner(ABC):

    @abstractmethod
    def letter_finder(self, path: str):
        pass

    @abstractmethod
    def image_cropper(self, path: str, x: int, y: int, width: int, height: int):
        pass

    @abstractmethod
    def image_resizer(self, path: str):
        pass
