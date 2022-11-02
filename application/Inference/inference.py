from model.model_one.ensemble import Ensemble


class Inference:
    def __init__(self, model_name, x_test: list):
        self.model_name = model_name
        self.x_test = x_test

    def start_inference(self):
        letter = self.model_name.infer(self.x_test)
        return letter
