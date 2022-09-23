from application.feature_extraction.feature_generation_testing import FeatureExtractor
from model.knn.knn import KNN


class Inference:
    def __init__(self, model_name: KNN, x_test: list):
        self.model_name = model_name
        self.x_test = x_test

    def start_inference(self):
        letter = self.model_name.infer(self.x_test)
        return letter
