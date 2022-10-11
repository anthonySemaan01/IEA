from model import consonant_classifier, consonant_vowel_classifier, digit_letter_classifier, even_classifier, \
    even_odd_classifier, odd_classifier, vowels_classifier


class Ensemble:
    def __init__(self):
        self.digit_letter = digit_letter_classifier.DigitLetterClassifier()
        self.consonant_vowel = consonant_vowel_classifier.ConsonantVowelClassifier()
        self.consonant = consonant_classifier.ConsonantClassifier()
        self.vowel = vowels_classifier.VowelClassifier()
        self.even_odd = even_odd_classifier.EvenOddClassifier()
        self.odd = odd_classifier.OddClassifier()
        self.even = even_classifier.EvenClassifier()

    def infer(self, x_test):
        result_digit_letter = self.digit_letter.classify_digit_letter_knn(x_test)
        print(result_digit_letter)
        # if result_digit_letter == "Letter":
        #     result_consonant_vowel = self.consonant_vowel.classify_consonant_vowel(x_test)
        #     if result_consonant_vowel == "Consonant":
        #         result_consonant = self.consonant.classify_consonant(x_test)
        #         return result_consonant
        #     else:
        #         result_vowel = self.vowel.classify_vowels(x_test)
        #         return result_vowel
        # else:
        result_even_odd = self.even_odd.classify_even_odd(x_test)

        if result_even_odd == "Even":
            result_even = self.even.classify_even_digit(x_test)
            return result_even
        else:
            result_odd = self.odd.classify_odd_digit(x_test)
            return result_odd
