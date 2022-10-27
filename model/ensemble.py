from model import upper_classifier, upper_lower_classifier, digit_letter_classifier, even_classifier, \
    even_odd_classifier, odd_classifier, lower_classifier


class Ensemble:
    def __init__(self):
        self.digit_letter = digit_letter_classifier.DigitLetterClassifier()
        self.upper_lower = upper_lower_classifier.UpperLowerClassifier()
        self.upper = upper_classifier.UpperClassifier()
        self.lower = lower_classifier.LowerClassifier()
        self.even_odd = even_odd_classifier.EvenOddClassifier()
        self.odd = odd_classifier.OddClassifier()
        self.even = even_classifier.EvenClassifier()

    def infer(self, x_test):
        result_digit_letter = self.digit_letter.classify_digit_letter_knn(x_test)
        print("it's a ", result_digit_letter)
        if result_digit_letter == "Letter":
            result_upper_lower = self.upper_lower.classify_upper_lower(x_test)
            print("it's a ", result_upper_lower)
            if result_upper_lower == "Upper":
                result_upper_letter = self.upper.classify_upper(x_test)
                return result_upper_letter
            else:
                result_lower_letter = self.lower.classify_lower(x_test)
                return result_lower_letter
        else:
            result_even_odd = self.even_odd.classify_even_odd(x_test)
            print("it's an ", result_even_odd)
            if result_even_odd == "Even":
                result_even = self.even.classify_even_digit(x_test)
                return result_even
            else:
                result_odd = self.odd.classify_odd_digit(x_test)
                return result_odd
