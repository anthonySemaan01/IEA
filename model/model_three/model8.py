from domain.models.file_structure import FileStructure
import tensorflow as tf
from tensorflow import keras


class ModelEight:
    def __init__(self):
        path = FileStructure.WEIGHTS_MODEL_8.value
        self.model = tf.keras.models.load_model(path)

    def classification_eight(self, x_test):
        predication = self.model.predict(x_test)
        return predication
