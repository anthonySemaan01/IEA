from model.model_one import upper_class_two_classifier, odd_classifier, lower_classes_classifier, \
    upper_class_four_classifier, upper_class_three_classifier, digit_letter_classifier, lower_class_three_classifier, \
    lower_class_one_classifier, upper_class_one_classifier, lower_class_four_classifier, upper_lower_classifier, \
    even_classifier, upper_classifier, lower_class_two_classifier, even_odd_classifier, upper_classes_classifier


class Ensemble:
    def __init__(self):
        self.digit_letter = digit_letter_classifier.DigitLetterClassifier()
        self.upper_lower = upper_lower_classifier.UpperLowerClassifier()
        self.upper = upper_classifier.UpperClassifier()
        # self.lower = lower_classifier.LowerClassifier()

        self.lower_classes_classifier = lower_classes_classifier.LowerClassesClassifier()
        self.lower_class_one_classifier = lower_class_one_classifier.LowerClassOneClassifier()
        self.lower_class_two_classifier = lower_class_two_classifier.LowerClassTwoClassifier()
        self.lower_class_three_classifier = lower_class_three_classifier.LowerClassThreeClassifier()
        self.lower_class_four_classifier = lower_class_four_classifier.LowerClassFourClassifier()

        self.upper_classes_classifier = upper_classes_classifier.UpperClassesClassifier()
        self.upper_class_one_classifier = upper_class_one_classifier.UpperClassOneClassifier()
        self.upper_class_two_classifier = upper_class_two_classifier.UpperClassTwoClassifier()
        self.upper_class_three_classifier = upper_class_three_classifier.UpperClassThreeClassifier()
        self.upper_class_four_classifier = upper_class_four_classifier.UpperClassFourClassifier()

        self.even_odd = even_odd_classifier.EvenOddClassifier()
        self.odd = odd_classifier.OddClassifier()
        self.even = even_classifier.EvenClassifier()

    def infer(self, x_test):
        result_digit_letter = self.digit_letter.classify_digit_letter(x_test)
        print("it's a ", result_digit_letter)
        if result_digit_letter == "Letter":
            result_upper_lower = self.upper_lower.classify_upper_lower(x_test)
            print("it's a ", result_upper_lower)
            if result_upper_lower == "Upper":
                result_upper_class = self.upper_classes_classifier.classify_upper_classes(x_test)
                print(result_upper_class)
                if "class_one" in result_upper_class:
                    result_class_one = self.upper_class_one_classifier.classify_upper_class_one(x_test)
                    return result_class_one
                if "class_two" in result_upper_class:
                    result_class_two = self.upper_class_two_classifier.classify_upper_class_two(x_test)
                    return result_class_two
                if "class_three" in result_upper_class:
                    result_class_three = self.upper_class_three_classifier.classify_upper_class_three(x_test)
                    return result_class_three
                if "class_four" in result_upper_class:
                    result_class_four = self.upper_class_four_classifier.classify_upper_class_four(x_test)
                    return result_class_four
            else:
                result_lower_class = self.lower_classes_classifier.classify_lower_classes(x_test)
                print(result_lower_class)
                if "class_one" in result_lower_class:
                    result_class_one = self.lower_class_one_classifier.classify_lower_class_one(x_test)
                    return result_class_one
                if "class_two" in result_lower_class:
                    result_class_two = self.lower_class_two_classifier.classify_lower_class_two(x_test)
                    return result_class_two
                if "class_three" in result_lower_class:
                    result_class_three = self.lower_class_three_classifier.classify_lower_class_three(x_test)
                    return result_class_three
                if "class_four" in result_lower_class:
                    result_class_four = self.lower_class_four_classifier.classify_lower_class_four(x_test)
                    return result_class_four
        else:
            result_even_odd = self.even_odd.classify_even_odd(x_test)
            print("it's an ", result_even_odd)
            if result_even_odd == "Even":
                result_even = self.even.classify_even_digit(x_test)
                return result_even
            else:
                result_odd = self.odd.classify_odd_digit(x_test)
                return result_odd
