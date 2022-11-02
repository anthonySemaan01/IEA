import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier

from domain.models.file_structure import FileStructure


class ClassFiveModelTwoClassifier:
    @staticmethod
    def classify_class_five_model_two(x_test) -> str:
        try:
            # TODO
            # find k
            knn = KNeighborsClassifier(n_neighbors=5)

            x_test_np = np.array(x_test)

            x_test_np = x_test_np.reshape(1, -1)
            df = pd.read_csv(str(FileStructure.VECTOR_CLASSE5_MODEL2.value), index_col=0)
            y_train = df.iloc[:, 0].to_numpy()
            x_train = df.drop(df.columns[0], axis=1)
            x_train = x_train.to_numpy()

            knn.fit(x_train, y_train)

            y_predict_knn = knn.predict(x_test_np)

            outputs: list = [y_predict_knn]
            print(outputs)
            return y_predict_knn

        except Exception as e:
            print(e)