from model.model_three.model5 import ModelFive
from model.model_three.model8 import ModelEight
from model.model_three.model9 import ModelNine

model_weights: dict = {
    "model_8": 0.4,
    "model_5": 0.3,
    "model_9": 0.3
}


class EnsembleThree:
    def infer(self, x_test: list) -> str:
        result_model_five = ModelFive.classification_five(x_test=x_test)
        result_model_eight = ModelEight.classification_eight(x_test=x_test)
        result_model_nine = ModelNine.classification_nine(x_test=x_test)

        print("result_model_5:", result_model_five)
        print("result_model_8:", result_model_eight)
        print("result_model_9:", result_model_nine)

        if result_model_eight == result_model_nine:
            return result_model_eight

        else:
            return result_model_five
