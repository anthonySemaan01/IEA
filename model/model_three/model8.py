import tensorflow as tf

from domain.models.file_structure import FileStructure
import numpy as np

class_matcher = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C',
                 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P',
                 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y', 'Z', 'a', 'b', 'c',
                 'd', 'e', 'f', 'g', 'h', 'i', 'j', 'k', 'l', 'm', 'n', 'o', 'p',
                 'q', 'r', 's', 't', 'u', 'v', 'w', 'x', 'y', 'z']
class ModelEight:

    @staticmethod
    def classification_eight(x_test):
        path = FileStructure.WEIGHTS_MODEL_8.value
        model = tf.keras.models.load_model(path)
        x_test.insert(0, 0)
        prediction = model.predict(np.array(x_test).reshape((1, 666)))
        x_test.pop(0)
        classes = np.argmax(prediction, axis=1)
        print(class_matcher[classes[0]])
        return class_matcher[classes[0]]
