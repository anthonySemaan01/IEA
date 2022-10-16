from containers import Services
from domain.exceptions.feature_generation_exception import FeatureGeneration
from domain.models.file_structure import FileStructure
from shared.helper.csv_file_reader import read_training_set_digit_letter, read_training_set_even_odd, \
    read_training_odd_values, read_training_even_values, read_training_lower_values, read_training_upper_lower, \
    read_training_upper_values
from shared.helper.csv_file_writer import write_training_even_values, write_training_lower_values, \
    write_training_odd_values, write_training_upper_values, write_training_upper_lower, write_training_set_even_odd, \
    write_training_set_digit_letter

extractor = Services.feature_generation(FileStructure.TESTING_IMAGES_PATH.value)


class TrainingFeatureGeneration:

    @staticmethod
    def feature_generation_test(output_label: str):
        # TODO
        # Add the output label to the dataframe
        try:
            x_test: list = extractor.extract_features(path_to_directory=extractor.path)
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

        write_training_set_digit_letter(digit_letter_df)
        write_training_set_even_odd(even_odd_df)
        write_training_even_values(even_df)
        write_training_odd_values(odd_df)
        write_training_upper_lower(upper_lower_df)
        write_training_upper_values(upper_df)
        write_training_lower_values(lower_df)

        return x_test
