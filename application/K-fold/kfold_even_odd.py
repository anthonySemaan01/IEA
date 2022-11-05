import csv

from domain.contracts.abstract_feature_extractor import AbstractFeatureExtractor
from domain.models.file_structure import FileStructure


# np.set_printoptions(threshold=sys.maxsize)

class Kfold_even_odd(AbstractFeatureExtractor):

    def __init__(self, path):
        self.path = path

    # 1 White/Black
    def kfold(self, path):
        k = 4
        training_samples_size: int = 440
        start_idx: int = 0
        end_idx = int(training_samples_size / k)

        # path of ./training_set_digit_letter.csv
        VECTOR_EVEN_ODD_PATH: str = str(FileStructure.VECTOR_EVEN_ODD__PATH.value)

        with open(VECTOR_EVEN_ODD_PATH, newline='') as csvfile:
            data = list(csv.reader(csvfile))

        data.pop(0)

        while (start_idx <= (k - 1) * (
                training_samples_size / k)):  # for k = 4: for start < 7500 // for k=10: for start
            # < 9000

            X_train = []
            y_train = []
            X_validation = []
            y_validation = []

            print("For validation set: ", start_idx, end_idx)
            print()
            # Training Set
            for i in (range(0, start_idx)):
                X_train.append(data[i])
                y_train.append(data[i][1])  # now we have the respective labels for the piece of training we took
                # y_train.append([i,data[i][1]]) # for Micheal: Replace it by this one to visualize it as well as in
                # 2 & 3       # 1

            for i in range(end_idx, training_samples_size):
                X_train.append(data[i])
                y_train.append(data[i][1])  # now we have the respective labels for the piece of training we took
                # y_train.append([i,data[i][1]]) # for Micheal: Replace it by this one to visualize it as well as in 3
                # 2

            print("Training Set ", y_train)
            print()

            # Validation Set
            for i in (range(start_idx, end_idx)):
                X_validation.append(data[i])
                y_validation.append(data[i][1])  # now we have the respective labels for the piece of validation we took
                # y_validation.append([i, data[i][1]]) # for Micheal: Replace it by this one to visualize it
                # 3

            print("Validation Set ", y_validation)
            print()

            # Training added here
            # Result : y_pred

            # # Accuracy
            # accuracy = sum([
            #     int( y_pred_i == y_test_i)
            #     for y_pred_i, y_test_i in zip(y_pred, y_validation)
            #     ])/len(y_validation)
            # print("accuracy: ", accuracy)

            start_idx = start_idx + int((training_samples_size / k))
            end_idx = end_idx + int((training_samples_size / k))
