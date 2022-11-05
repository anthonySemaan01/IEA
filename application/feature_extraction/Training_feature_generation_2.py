from containers import Services
from domain.exceptions.feature_generation_exception import FeatureGeneration
from domain.models.file_structure import FileStructure
from shared.helper.csv_file_reader import read_csv_file
from shared.helper.csv_file_writer import write_to_csv_file

feature_extractor2 = Services.feature_generation2(FileStructure.TESTING_IMAGES_PATH.value)


class TrainingFeatureGeneration2:

    @staticmethod
    def feature_generation_test(output_label: str):
        try:
            vector: list = feature_extractor2.extract_features(path_to_directory=feature_extractor2.path)
        except Exception as e:
            raise FeatureGeneration(additional_message=e.__str__())

        class1 = ["a", "e", "n", "h", "b", "6", "r", "G", "C", "D", "O", "Q", "0"]
        class2 = ["S", "3", "4", "5", "8", "d", "H", "B", "P", "F", "K", "R"]
        class3 = ["M", "N", "9", "q", "m", "g"]
        class4 = ["U", "V", "W"]
        class5 = ["I", "J", "L", "T", "1", "7", "l", "i", "j", "f", "t"]
        class6 = ["2", "Z", "E", "A", "X", "Y"]

        m1c1 = ["O", "a", "0", "D", "Q"]
        m2c1 = ["C", "G", "e", "r"]
        m3c1 = ["6", "b", "h", "n"]

        m1c2 = ["3", "8", "K", "B", "P", "R", "F"]
        m2c2 = ["H", "4", "5", "S", "d"]

        m1c3 = ["9", "q", "g"]
        m2c3 = ["M", "N", "m"]

        m1c5 = ["I", "J", "1", "l", "i", "j"]
        m2c5 = ["T", "7", "t", "F", "L"]

        m1c6 = ["2", "Z", "E"]
        m2c6 = ["A", "X", "Y"]

        m11c2 = ["3", "8", "B", "K", "R"]
        m12c2 = ["F", "P"]

        m11c5 = ["1", "l", "J", "I"]
        m12c5 = ["i", "j"]

        if output_label in class1:
            vector.insert(0, "class1")
            classes_df = read_csv_file(str(FileStructure.VECTOR_CLASSES.value))
            classes_df.loc[len(classes_df)] = vector
            write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSES.value), dataframe=classes_df)
            vector.pop(0)

            if output_label in m1c1:
                vector.insert(0, "class1_model1")
                class1_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE1_MODELS.value))
                class1_models_df.loc[len(class1_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE1_MODELS.value),
                                  dataframe=class1_models_df)
                vector.pop(0)
                class1_model1_df = read_csv_file(str(FileStructure.VECTOR_CLASSE1_MODEL1.value))
                vector.insert(0, output_label[0])
                class1_model1_df.loc[len(class1_model1_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE1_MODEL1.value),
                                  dataframe=class1_model1_df)

            elif output_label in m2c1:
                vector.insert(0, "class1_model2")
                class1_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE1_MODELS.value))
                class1_models_df.loc[len(class1_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE1_MODELS.value),
                                  dataframe=class1_models_df)
                vector.pop(0)
                class1_model2_df = read_csv_file(str(FileStructure.VECTOR_CLASSE1_MODEL2.value))
                vector.insert(0, output_label[0])
                class1_model2_df.loc[len(class1_model2_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE1_MODEL2.value),
                                  dataframe=class1_model2_df)

            else:
                vector.insert(0, "class1_model3")
                class1_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE1_MODELS.value))
                class1_models_df.loc[len(class1_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE1_MODELS.value),
                                  dataframe=class1_models_df)
                vector.pop(0)
                class1_model3_df = read_csv_file(str(FileStructure.VECTOR_CLASSE1_MODEL3.value))
                vector.insert(0, output_label[0])
                class1_model3_df.loc[len(class1_model3_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE1_MODEL3.value),
                                  dataframe=class1_model3_df)

        elif output_label in class2:
            vector.insert(0, "class2")
            classes_df = read_csv_file(str(FileStructure.VECTOR_CLASSES.value))
            classes_df.loc[len(classes_df)] = vector
            write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSES.value), dataframe=classes_df)
            vector.pop(0)
            if output_label in m1c2:
                vector.insert(0, "class2_model1")
                class2_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE2_MODELS.value))
                class2_models_df.loc[len(class2_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE2_MODELS.value),
                                  dataframe=class2_models_df)
                vector.pop(0)
                if output_label in m11c2:
                    class2_model1_df = read_csv_file(str(FileStructure.VECTOR_CLASSE2_MODEL1.value))
                    vector.insert(0, "class2_model11")
                    class2_model1_df.loc[len(class2_model1_df)] = vector
                    write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE2_MODEL1.value),
                                      dataframe=class2_model1_df)
                    vector.pop(0)
                    class2_model11_df = read_csv_file(str(FileStructure.VECTOR_CLASSE2_MODEL11.value))
                    vector.insert(0, output_label[0])
                    class2_model11_df.loc[len(class2_model11_df)] = vector
                    write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE2_MODEL11.value),
                                      dataframe=class2_model11_df)

                else:
                    class2_model1_df = read_csv_file(str(FileStructure.VECTOR_CLASSE2_MODEL1.value))
                    vector.insert(0, "class2_model12")
                    class2_model1_df.loc[len(class2_model1_df)] = vector
                    write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE2_MODEL1.value),
                                      dataframe=class2_model1_df)
                    vector.pop(0)
                    class2_model12_df = read_csv_file(str(FileStructure.VECTOR_CLASSE2_MODEL12.value))
                    vector.insert(0, output_label[0])
                    class2_model12_df.loc[len(class2_model12_df)] = vector
                    write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE2_MODEL12.value),
                                      dataframe=class2_model12_df)

            if output_label in m2c2:
                vector.insert(0, "class2_model2")
                class2_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE2_MODELS.value))
                class2_models_df.loc[len(class2_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE2_MODELS.value),
                                  dataframe=class2_models_df)
                vector.pop(0)
                class2_model2_df = read_csv_file(str(FileStructure.VECTOR_CLASSE2_MODEL2.value))
                vector.insert(0, output_label[0])
                class2_model2_df.loc[len(class2_model2_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE2_MODEL2.value),
                                  dataframe=class2_model2_df)

        elif output_label in class3:
            vector.insert(0, "class3")
            classes_df = read_csv_file(str(FileStructure.VECTOR_CLASSES.value))
            classes_df.loc[len(classes_df)] = vector
            write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSES.value), dataframe=classes_df)
            vector.pop(0)

            if output_label in m1c3:
                vector.insert(0, "class3_model1")
                class3_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE3_MODELS.value))
                class3_models_df.loc[len(class3_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE3_MODELS.value),
                                  dataframe=class3_models_df)
                vector.pop(0)
                class3_model1_df = read_csv_file(str(FileStructure.VECTOR_CLASSE3_MODEL1.value))
                vector.insert(0, output_label[0])
                class3_model1_df.loc[len(class3_model1_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE3_MODEL1.value),
                                  dataframe=class3_model1_df)

            else:
                vector.insert(0, "class3_model2")
                class3_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE3_MODELS.value))
                class3_models_df.loc[len(class3_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE3_MODELS.value),
                                  dataframe=class3_models_df)
                vector.pop(0)
                class3_model2_df = read_csv_file(str(FileStructure.VECTOR_CLASSE3_MODEL2.value))
                vector.insert(0, output_label[0])
                class3_model2_df.loc[len(class3_model2_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE3_MODEL2.value),
                                  dataframe=class3_model2_df)

        elif output_label in class4:
            vector.insert(0, "class4")
            classes_df = read_csv_file(str(FileStructure.VECTOR_CLASSES.value))
            classes_df.loc[len(classes_df)] = vector
            write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSES.value), dataframe=classes_df)
            vector.pop(0)
            vector.insert(0, output_label[0])
            class4_model1_df = read_csv_file(str(FileStructure.VECTOR_CLASSE4_MODEL1))
            class4_model1_df.loc[len(class4_model1_df)] = vector
            write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE4_MODEL1.value),
                              dataframe=class4_model1_df)

        elif output_label in class5:
            vector.insert(0, "class5")
            classes_df = read_csv_file(str(FileStructure.VECTOR_CLASSES.value))
            classes_df.loc[len(classes_df)] = vector
            write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSES.value), dataframe=classes_df)
            vector.pop(0)

            if output_label in m1c5:
                vector.insert(0, "class5_model1")
                class5_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE5_MODELS.value))
                class5_models_df.loc[len(class5_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE5_MODELS.value),
                                  dataframe=class5_models_df)
                vector.pop(0)
                if output_label in m11c5:
                    class5_model1_df = read_csv_file(str(FileStructure.VECTOR_CLASSE5_MODEL1.value))
                    vector.insert(0, "class5_model11")
                    class5_model1_df.loc[len(class5_model1_df)] = vector
                    write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE5_MODEL1.value),
                                      dataframe=class5_model1_df)
                    vector.pop(0)
                    class5_model11_df = read_csv_file(str(FileStructure.VECTOR_CLASSE5_MODEL11.value))
                    vector.insert(0, output_label[0])
                    class5_model11_df.loc[len(class5_model11_df)] = vector
                    write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE5_MODEL11.value),
                                      dataframe=class5_model11_df)

                else:
                    class5_model1_df = read_csv_file(str(FileStructure.VECTOR_CLASSE5_MODEL1.value))
                    vector.insert(0, "class5_model12")
                    class5_model1_df.loc[len(class5_model1_df)] = vector
                    write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE5_MODEL1.value),
                                      dataframe=class5_model1_df)
                    vector.pop(0)
                    class5_model12_df = read_csv_file(str(FileStructure.VECTOR_CLASSE5_MODEL12.value))
                    vector.insert(0, output_label[0])
                    class5_model12_df.loc[len(class5_model12_df)] = vector
                    write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE5_MODEL12.value),
                                      dataframe=class5_model12_df)

            if output_label in m2c2:
                vector.insert(0, "class5_model2")
                class5_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE5_MODELS.value))
                class5_models_df.loc[len(class5_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE5_MODELS.value),
                                  dataframe=class5_models_df)
                vector.pop(0)
                class5_model2_df = read_csv_file(str(FileStructure.VECTOR_CLASSE5_MODEL2.value))
                vector.insert(0, output_label[0])
                class5_model2_df.loc[len(class5_model2_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE5_MODEL2.value),
                                  dataframe=class5_model2_df)

        else:
            vector.insert(0, "class6")
            classes_df = read_csv_file(str(FileStructure.VECTOR_CLASSES.value))
            classes_df.loc[len(classes_df)] = vector
            write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSES.value), dataframe=classes_df)
            vector.pop(0)

            if output_label in m1c3:
                vector.insert(0, "class6_model1")
                class6_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE6_MODELS.value))
                class6_models_df.loc[len(class6_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE6_MODELS.value),
                                  dataframe=class6_models_df)
                vector.pop(0)
                class6_model1_df = read_csv_file(str(FileStructure.VECTOR_CLASSE6_MODEL1.value))
                vector.insert(0, output_label[0])
                class6_model1_df.loc[len(class6_model1_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE6_MODEL1.value),
                                  dataframe=class6_model1_df)

            else:
                vector.insert(0, "class6_model2")
                class6_models_df = read_csv_file(str(FileStructure.VECTOR_CLASSE6_MODELS.value))
                class6_models_df.loc[len(class6_models_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE6_MODELS.value),
                                  dataframe=class6_models_df)
                vector.pop(0)
                class6_model2_df = read_csv_file(str(FileStructure.VECTOR_CLASSE6_MODEL2.value))
                vector.insert(0, output_label[0])
                class6_model2_df.loc[len(class6_model2_df)] = vector
                write_to_csv_file(path_to_csv_file=str(FileStructure.VECTOR_CLASSE6_MODEL2.value),
                                  dataframe=class6_model2_df)

        return vector


