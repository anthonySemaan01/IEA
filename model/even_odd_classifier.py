import numpy as np
import pandas as pd
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier

from domain.models.file_structure import FileStructure


class EvenOddClassifier:
    @staticmethod
    def classify_even_odd(x_test) -> str:
        knn = KNeighborsClassifier(n_neighbors=3)
        # clf = svm.SVC(kernel='linear')
        dt = DecisionTreeClassifier()
        x_test_np = np.array(x_test)

        x_test_np = x_test_np.reshape(1, -1)
        df = pd.read_csv(str(FileStructure.VECTOR_EVEN_ODD__PATH.value), index_col=0)
        y_train = df.iloc[:, 0].to_numpy()
        x_train = df.drop(df.columns[0], axis=1)
        x_train = x_train.to_numpy()

        knn.fit(x_train, y_train)
        # clf.fit(x_train, y_train)
        dt.fit(x_train, y_train)
        y_predict_knn = knn.predict(x_test_np)
        # y_predict_svm = clf.predict(x_test_np)
        y_predict_dt = dt.predict(x_test_np)

        outputs: list = [y_predict_knn, y_predict_dt]

        if outputs.count("even") > outputs.count("odd"):
            return "even"
        else:
            return "odd"
