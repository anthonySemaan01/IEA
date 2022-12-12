import numpy as np
import tensorflow as tf

from domain.models.file_structure import FileStructure

class_matcher = ["0", "1", "2", "3", "4", "5", "6", "7", "8", "9"]


class ModelCNNDigits:

    @staticmethod
    def classification_cnn(image):
        path = FileStructure.WEIGHTS_MODEL_CNN.value
        model = tf.keras.models.load_model(path)
        image = image.reshape(1, 28, 28, 1)
        prediction = model.predict(image)
        classes = np.argmax(prediction, axis=1)
        return class_matcher[classes[0]]
