from domain.models.file_structure import FileStructure
import tensorflow as tf


class ModelFive:
    def __int__(self):
        path = FileStructure.WEIGHTS_MODEL_5.value
        self.model = tf.keras.models.load_model(path)

    def classification_five(self, x_test):
        predication = self.model.predict(x_test)
        return predication
