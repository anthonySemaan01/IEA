import pickle

import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from domain.models.file_structure import FileStructure


class DigitLetterClassifier:
    @staticmethod
    def classify_digit_letter(x_test) -> str:
        try:

            out = ""
            knn = KNeighborsClassifier(n_neighbors=5)
            dt = pickle.load(open(str(FileStructure.SAV_DT_DIGIT_LETTER.value), "rb"))
            svm = pickle.load(open(str(FileStructure.SAV_SVM_DIGIT_LETTER.value), "rb"))

            x_test_np = np.array(x_test)

            x_test_np = x_test_np.reshape(1, -1)
            df = pd.read_csv(str(FileStructure.VECTOR_DIGIT_LETTER_PATH.value), index_col=0)
            y_train = df.iloc[:, 0].to_numpy()
            x_train = df.drop(df.columns[0], axis=1)
            x_train = x_train.to_numpy()

            knn.fit(x_train, y_train)

            y_predict_knn = knn.predict(x_test_np)
            y_predict_dt = dt.predict(x_test_np)
            y_predict_svm = svm.predict(x_test_np)

            outputs: list = [y_predict_knn, y_predict_dt, y_predict_svm]
            print(outputs)

            if outputs.count("Digit") > outputs.count("Letter"):
                out = "Digit"
            else:
                out = "Letter"

            print(out)
            
            return y_predict_knn

        except Exception as e:
            print(e)
