from containers import Services
from domain.exceptions.feature_generation_exception import FeatureGeneration
from domain.models.file_structure import FileStructure
from shared.helper.csv_file_reader import read_training_set_digit_letter, read_training_set_even_odd, \
    read_training_odd_values, read_training_even_values, read_training_upper_lower, \
    read_training_lower_classes, read_training_lower_class_four, \
    read_training_lower_class_one, read_training_lower_class_three, read_training_lower_class_two, \
    read_training_upper_classes, read_training_upper_class_one, read_training_upper_class_two, \
    read_training_upper_class_three, read_training_upper_class_four
from shared.helper.csv_file_writer import write_training_even_values, write_training_odd_values, \
    write_training_upper_lower, write_training_set_even_odd, \
    write_training_set_digit_letter, write_training_lower_class_four, \
    write_training_lower_class_one, write_training_lower_class_three, write_training_lower_class_two, \
    write_training_lower_classes, write_training_upper_classes, write_training_upper_class_one, \
    write_training_upper_class_two, write_training_upper_class_three, write_training_upper_class_four

feature_extractor1 = Services.feature_generation1(FileStructure.TESTING_IMAGES_PATH.value)


class TrainingFeatureGeneration:

    @staticmethod
    def feature_generation_test(output_label: str):
        try:
            vector: list = feature_extractor1.extract_features(path_to_directory=feature_extractor1.path)

        except Exception as e:
            raise FeatureGeneration(additional_message=e.__str__())

        try:
            digit_letter_df = read_training_set_digit_letter()
            even_odd_df = read_training_set_even_odd()
            even_df = read_training_even_values()
            odd_df = read_training_odd_values()
            upper_lower_df = read_training_upper_lower()

            lower_classes_df = read_training_lower_classes()
            lower_class_one_df = read_training_lower_class_one()
            lower_class_two_df = read_training_lower_class_two()
            lower_class_three_df = read_training_lower_class_three()
            lower_class_four_df = read_training_lower_class_four()

            upper_classes_df = read_training_upper_classes()
            upper_class_one_df = read_training_upper_class_one()
            upper_class_two_df = read_training_upper_class_two()
            upper_class_three_df = read_training_upper_class_three()
            upper_class_four_df = read_training_upper_class_four()
        except Exception as e:
            print("error while reading files, ", e.__str__())

        # TODO
        # Add the code here, update the fetched dataframes
        digits: list = ["1", "2", "3", "4", "5", "6", "7", "8", "9"]
        letters: list = ["a", "b", "c", "d", "e", "f", "g", "h", "i", "j", "k", "l", "m", "n", "o", "p", "q", "r", "s",
                         "t", "u", "v", "w", "x", "y", "z", "A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                         "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

        lower_class_one: list = ["a", "b", "c", "d", "e", "h", "o"]
        lower_class_two: list = ["p", "q", "y", "g", "j", "i"]
        lower_class_three: list = ["m", "n", "u", "v", "w", "x"]
        lower_class_four: list = ["z", "f", "r", "s", "t", "l", "k"]

        upper_class_one: list = ["A", "B", "C", "D", "O", "G", "Q"]
        upper_class_two: list = ["I", "J", "L", "T", "Y", "P"]
        upper_class_three: list = ["E", "K", "M", "R", "N", "F", "H"]
        upper_class_four: list = ["V", "U", "W", "X", "Z", "S"]

        even: list = ["0", "2", "4", "6", "8"]
        upper: list = ["A", "B", "C", "D", "E", "F", "G", "H", "I", "J", "K", "L",
                       "M", "N", "O", "P", "Q", "R", "S", "T", "U", "V", "W", "X", "Y", "Z"]

        print(vector)
        if output_label in digits:
            print("digit")
            vector.insert(0, "Digit")
            digit_letter_df.loc[len(digit_letter_df)] = vector
            write_training_set_digit_letter(digit_letter_df)

            if output_label in even:
                print("even")
                vector.pop(0)
                vector.insert(0, "Even")
                even_odd_df.loc[len(even_odd_df)] = vector
                write_training_set_even_odd(even_odd_df)
                vector.pop(0)
                vector.insert(0, output_label)
                even_df.loc[len(even_df)] = vector
                write_training_even_values(even_df)

            else:
                print("odd")
                vector.pop(0)
                vector.insert(0, "Odd")
                even_odd_df.loc[len(even_odd_df)] = vector
                print("HERE")

                write_training_set_even_odd(even_odd_df)
                vector.pop(0)
                vector.insert(0, output_label)
                print(vector)
                odd_df.loc[len(odd_df)] = vector
                print(odd_df)
                write_training_odd_values(odd_df)

        else:
            print("letter")
            vector.insert(0, "Letter")
            digit_letter_df.loc[len(digit_letter_df)] = vector
            write_training_set_digit_letter(digit_letter_df)

            if output_label in upper:
                print("upper")
                vector.pop(0)
                vector.insert(0, "Upper")
                upper_lower_df.loc[len(upper_lower_df)] = vector
                write_training_upper_lower(upper_lower_df)
                vector.pop(0)

                if output_label in upper_class_one:
                    print("class_one")
                    vector.insert(0, "class_one")
                    upper_classes_df.loc[len(upper_classes_df)] = vector
                    write_training_upper_classes(upper_classes_df)
                    vector.pop(0)
                    vector.insert(0, output_label)
                    upper_class_one_df.loc[len(upper_class_one_df)] = vector
                    write_training_upper_class_one(upper_class_one_df)

                if output_label in upper_class_two:
                    print("class_two")
                    vector.insert(0, "class_two")
                    upper_classes_df.loc[len(upper_classes_df)] = vector
                    write_training_upper_classes(upper_classes_df)
                    vector.pop(0)
                    vector.insert(0, output_label)
                    upper_class_two_df.loc[len(upper_class_two_df)] = vector
                    write_training_upper_class_two(upper_class_two_df)

                if output_label in upper_class_three:
                    print("class_three")
                    vector.insert(0, "class_three")
                    upper_classes_df.loc[len(upper_classes_df)] = vector
                    write_training_upper_classes(upper_classes_df)
                    vector.pop(0)
                    vector.insert(0, output_label)
                    upper_class_three_df.loc[len(upper_class_three_df)] = vector
                    write_training_upper_class_three(upper_class_three_df)

                if output_label in upper_class_four:
                    print("class_four")
                    vector.insert(0, "class_four")
                    upper_classes_df.loc[len(upper_classes_df)] = vector
                    write_training_upper_classes(upper_classes_df)
                    vector.pop(0)
                    vector.insert(0, output_label)
                    upper_class_four_df.loc[len(upper_class_four_df)] = vector
                    write_training_upper_class_four(upper_class_four_df)
            else:
                print("lower")
                vector.pop(0)
                vector.insert(0, "Lower")
                upper_lower_df.loc[len(upper_lower_df)] = vector
                write_training_upper_lower(upper_lower_df)
                vector.pop(0)

                if output_label in lower_class_one:
                    print("class_one")
                    vector.insert(0, "class_one")
                    lower_classes_df.loc[len(lower_classes_df)] = vector
                    write_training_lower_classes(lower_classes_df)
                    vector.pop(0)
                    vector.insert(0, output_label)
                    lower_class_one_df.loc[len(lower_class_one_df)] = vector
                    write_training_lower_class_one(lower_class_one_df)

                if output_label in lower_class_two:
                    print("class_two")
                    vector.insert(0, "class_two")
                    lower_classes_df.loc[len(lower_classes_df)] = vector
                    write_training_lower_classes(lower_classes_df)
                    vector.pop(0)
                    vector.insert(0, output_label)
                    lower_class_two_df.loc[len(lower_class_two_df)] = vector
                    write_training_lower_class_two(lower_class_two_df)

                if output_label in lower_class_three:
                    print("class_three")
                    vector.insert(0, "class_three")
                    lower_classes_df.loc[len(lower_classes_df)] = vector
                    write_training_lower_classes(lower_classes_df)
                    vector.pop(0)
                    vector.insert(0, output_label)
                    lower_class_three_df.loc[len(lower_class_three_df)] = vector
                    write_training_lower_class_three(lower_class_three_df)

                if output_label in lower_class_four:
                    print("class_four")
                    vector.insert(0, "class_four")
                    lower_classes_df.loc[len(lower_classes_df)] = vector
                    write_training_lower_classes(lower_classes_df)
                    vector.pop(0)
                    vector.insert(0, output_label)
                    lower_class_four_df.loc[len(lower_class_four_df)] = vector
                    write_training_lower_class_four(lower_class_four_df)

        print(digit_letter_df)
        print(upper_lower_df)
        print(upper_classes_df)
        print(upper_class_one_df)
        print(upper_class_two_df)
        print(upper_class_three_df)
        print(upper_class_four_df)

        return vector
