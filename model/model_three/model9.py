from domain.models.file_structure import FileStructure
import tensorflow as tf
from tensorflow import keras

class ModelNine:
    def __init__(self):
        path = FileStructure.WEIGHTS_MODEL_9.value
        self.model = tf.keras.models.load_model(path)
    def classification_nine(self, x_test) -> str:
        predication = self.model.predict(x_test)
        return predication
