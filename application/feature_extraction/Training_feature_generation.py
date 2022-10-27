from containers import Services
from domain.exceptions.feature_generation_exception import FeatureGeneration
from domain.models.file_structure import FileStructure
from shared.helper.csv_file_reader import read_training_set_digit_letter, read_training_set_even_odd, \
    read_training_odd_values, read_training_even_values, read_training_lower_values, read_training_upper_lower, \
    read_training_upper_values
from shared.helper.csv_file_writer import write_training_even_values, write_training_odd_values, \
    write_training_upper_lower, write_training_set_even_odd, \
    write_training_set_digit_letter, write_training_lower_values, write_training_upper_values

feature_extractor = Services.feature_generation(FileStructure.TESTING_IMAGES_PATH.value)


class TrainingFeatureGeneration:

    @staticmethod
    def feature_generation_test(output_label: str):
        try:
            vector: list = feature_extractor.extract_features(path_to_directory=feature_extractor.path)
        except Exception as e:
            raise FeatureGeneration(additional_message=e.__str__())

        digit_letter_df = read_training_set_digit_letter()
        even_odd_df = read_training_set_even_odd()
        even_df = read_training_even_values()
        odd_df = read_training_odd_values()
        upper_lower_df = read_training_upper_lower()
        upper_df = read_training_upper_values()
        lower_df = read_training_lower_values()

        # TODO
        # Add the code here, update the fetched dataframes
        digits: list = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        letters: list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                         "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                         "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]
        even: list = ["0", "2", "4", "6", "8"]
        upper: list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                       "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

        if output_label in digits:
            vector.insert(0, "Digit")
            digit_letter_df.loc[len(digit_letter_df)] = vector
            write_training_set_digit_letter(digit_letter_df)

            if output_label in even:
                vector.pop(0)
                vector.insert(0, "Even")
                even_odd_df.loc[len(even_odd_df)] = vector
                write_training_set_even_odd(even_odd_df)
                vector.pop(0)
                vector.insert(0, output_label)
                even_df.loc[len(even_df)] = vector
                write_training_even_values(even_df)

            else:
                vector.pop(0)
                vector.insert(0, "Odd")
                even_odd_df.loc[len(even_odd_df)] = vector
                write_training_set_even_odd(even_odd_df)
                vector.pop(0)
                vector.insert(0, output_label)
                odd_df.loc[len(odd_df)] = vector
                write_training_odd_values(odd_df)
        else:
            vector.insert(0, "Letter")
            digit_letter_df.loc[len(digit_letter_df)] = vector
            write_training_set_digit_letter(digit_letter_df)

            if output_label in upper:
                vector.pop(0)
                vector.insert(0, "Upper")
                upper_lower_df.loc[len(upper_lower_df)] = vector
                write_training_upper_lower(upper_lower_df)
                vector.pop(0)
                vector.insert(0, output_label)
                upper_df.loc[len(upper_df)] = vector
                write_training_upper_values(upper_df)
            else:
                vector.pop(0)
                vector.insert(0, "Lower")
                upper_lower_df.loc[len(upper_lower_df)] = vector
                write_training_upper_lower(upper_lower_df)
                vector.pop(0)
                vector.insert(0, output_label)
                lower_df.loc[len(lower_df)] = vector
                write_training_lower_values(lower_df)


        print("Digit_Letter_df: ")
        print(digit_letter_df)

        print("Odd_Even_df: ")
        print(even_odd_df)

        print("Odd_df: ")
        print(odd_df)

        print("Even_df: ")
        print(even_df)

        print("Upper_Lower_df: ")
        print(upper_lower_df)

        print("Upper_df: ")
        print(upper_df)

        print("Lower_df: ")
        print(lower_df)

        return vector
