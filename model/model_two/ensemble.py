from model.model_two import classes, class_one_models, class_one_model_one, class_one_model_two, class_one_model_three, \
    class_two_models, class_two_model_one, class_two_model_two, class_two_model_11, class_two_model_12, \
    class_three_model_one, class_five_model_11, class_five_model_one, class_four_model_one, \
    class_five_model_two, class_six_model_one, class_six_model_two, class_three_model_two, class_three_models, \
    class_five_model_12, class_five_models, class_six_models


class EnsembleTwo:

    def __init__(self):
        self.classes = classes.ClassesClassifier()

        self.class_one_classifier = class_one_models.ClassOneClassifier()
        self.class_one_classifier_model_one = class_one_model_one.ClassOneModelOneClassifier()
        self.class_one_classifier_model_two = class_one_model_two.ClassOneModelTwoClassifier()
        self.class_one_classifier_model_three = class_one_model_three.ClassOneModelThreeClassifier()

        self.class_two_classifier = class_two_models.ClassTwoClassifier()
        self.class_two_classifier_model_one = class_two_model_one.ClassTwoModelOneClassifier()
        self.class_two_classifier_model_two = class_two_model_two.ClassTwoModelTwoClassifier()
        self.class_two_classifier_model_11 = class_two_model_11.ClassTwoModel11Classifier()
        self.class_two_classifier_model_12 = class_two_model_12.ClassTwoModel12Classifier()

        self.class_three_classifier = class_three_models.ClassThreeClassifier()
        self.class_three_classifier_model_one = class_three_model_one.ClassThreeModelOneClassifier()
        self.class_three_classifier_model_two = class_three_model_two.ClassThreeModelTwoClassifier()

        self.class_four_classifier_model_one = class_four_model_one.ClassFourModelOneClassifier()

        self.class_five_classifier = class_five_models.ClassFiveClassifier()
        self.class_five_classifier_model_one = class_five_model_one.ClassFiveModelOneClassifier()
        self.class_five_classifier_model_two = class_five_model_two.ClassFiveModelTwoClassifier()
        self.class_five_classifier_model_11 = class_five_model_11.ClassFiveModel11Classifier()
        self.class_five_classifier_model_12 = class_five_model_12.ClassFiveModel12Classifier()

        self.class_six_classifier = class_six_models.ClassSixClassifier()
        self.class_six_classifier_model_one = class_six_model_one.ClassSixModelOneClassifier()
        self.class_six_classifier_model_two = class_six_model_two.ClassSixModelTwoClassifier()

    def infer(self, x_test):
        result_class = self.classes.classify_class(x_test)
        print("result first layer: ", result_class)

        if "class1" in result_class:
            result_class_one_models = self.class_one_classifier.classify_class_one(x_test)
            if "class1_model1" in result_class_one_models:
                result_class_one_model_one = self.class_one_classifier_model_one.classify_class_one_model_one(x_test)
                return result_class_one_model_one

            if "class1_model2" in result_class_one_models:
                result_class_one_model_two = self.class_one_classifier_model_two.classify_class_one_model_two(x_test)
                return result_class_one_model_two

            if "class1_model3" in result_class_one_models:
                result_class_one_model_three = self.class_one_classifier_model_three.classify_class_one_model_three(
                    x_test)
                return result_class_one_model_three

        if "class2" in result_class:
            result_class_two_models = self.class_two_classifier.classify_class_two(x_test)

            if "class2_model1" in result_class_two_models:
                result_class_two_model_one = self.class_two_classifier_model_one.classify_class_two_model_one(x_test)

                if "class2_model11" in result_class_two_model_one:
                    result_class_two_model11 = self.class_two_classifier_model_11.classify_class_two_model_11(x_test)
                    return result_class_two_model11

                if "class2_model12" in result_class_two_model_one:
                    result_class_two_model12 = self.class_two_classifier_model_12.classify_class_two_model_12(x_test)
                    return result_class_two_model12

            if "class2_model2" in result_class_two_models:
                result_class_two_model_two = self.class_two_classifier_model_two.classify_class_two_model_two(x_test)
                return result_class_two_model_two

        if "class3" in result_class:
            result_class_three_models = self.class_three_classifier.classify_class_three(x_test)
            if "class3_model1" in result_class_three_models:
                result_class_three_model_one = self.class_three_classifier_model_one.classify_class_three_model_one(
                    x_test)
                return result_class_three_model_one

            if "class3_model2" in result_class_three_models:
                result_class_three_model_two = self.class_three_classifier_model_two.classify_class_three_model_two(
                    x_test)
                return result_class_three_model_two

        if "class4" in result_class:
            return self.class_four_classifier_model_one.classify_class_four_model_one(x_test)

        if "class5" in result_class:
            result_class_five_models = self.class_five_classifier.classify_class_five(x_test)
            print("here ", result_class_five_models)
            if "class5_model1" in result_class_five_models:
                result_class_five_model_one = self.class_five_classifier_model_one.classify_class_Five_model_one(x_test)

                if "class5_model11" in result_class_five_model_one:
                    result_class_five_model11 = self.class_five_classifier_model_11.classify_class_Five_model_11(x_test)
                    return result_class_five_model11

                if "class5_model12" in result_class_five_model_one:
                    result_class_five_model12 = self.class_five_classifier_model_12.classify_class_Five_model_12(x_test)
                    return result_class_five_model12

            if "class5_model2" in result_class_five_models:
                result_class_five_model_two = self.class_five_classifier_model_two.classify_class_five_model_two(x_test)
                return result_class_five_model_two

        if "class6" in result_class:
            result_class_six_models = self.class_six_classifier.classify_class_six(x_test)

            if "class6_model1" in result_class_six_models:
                result_class_six_model_one = self.class_six_classifier_model_one.classify_class_six_model_one(x_test)
                return result_class_six_model_one

            if "class6_model2" in result_class_six_models:
                result_class_six_model_two = self.class_six_classifier_model_two.classify_class_six_model_two(x_test)
                return result_class_six_model_two
