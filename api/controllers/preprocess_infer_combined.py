from fastapi import APIRouter, UploadFile, File

from application.Inference.inference import Inference
from containers import Services
from domain.exceptions.image_preprocessing_exception import ImagePreprocessingException
from domain.models.file_structure import FileStructure
from model.model_one.ensemble import Ensemble
from model.model_two.ensemble import EnsembleTwo
from persistence.repositories.api_response import ApiResponse

router = APIRouter()
letter_fine_tuner = Services.letter_fine_tuner()
feature_extractor1 = Services.feature_generation1(FileStructure.TESTING_IMAGES_PATH.value)
feature_extractor2 = Services.feature_generation2(FileStructure.TESTING_IMAGES_PATH.value)


@router.post('/preprocess_inference_model1')
async def predictions(file: UploadFile = File(...)):
    try:
        output_image, output_image_path = letter_fine_tuner.letter_finder(file=file)
        vector: list = feature_extractor1.extract_features(path_to_directory=feature_extractor1.path)
        my_model = Ensemble()
        inference_result = Inference(model_name=my_model, x_test=vector).start_inference()
        return ApiResponse(data=str(inference_result))

    except ImagePreprocessingException as e:
        return ApiResponse(success=False, error=e.__str__())
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())


@router.post('/preprocess_inference_model2')
async def predictions(file: UploadFile = File(...)):
    try:
        output_image, output_image_path = letter_fine_tuner.letter_finder(file=file)
        vector: list = feature_extractor2.extract_features(path_to_directory=feature_extractor2.path)
        my_model = EnsembleTwo()
        inference_result = Inference(model_name=my_model, x_test=vector).start_inference()
        return ApiResponse(data=str(inference_result))

    except ImagePreprocessingException as e:
        return ApiResponse(success=False, error=e.__str__())
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())
