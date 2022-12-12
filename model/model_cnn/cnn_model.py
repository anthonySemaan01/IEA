from domain.models.file_structure import FileStructure
import tensorflow as tf


class ModelCNNDigits:
    def __int__(self):
        path = FileStructure.WEIGHTS_MODEL_CNN.value
        self.model = tf.keras.models.load_model(path)

    def classification_cnn(self, image):
        predication = self.model.predict(image)
        return predication
