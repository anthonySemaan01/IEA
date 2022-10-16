import io

from fastapi import APIRouter, UploadFile, File, Response

from application.Inference.inference import Inference
from application.feature_extraction.Training_feature_generation import feature_generation_test
from domain.exceptions.image_preprocessing_exception import ImagePreprocessingException
from model.ensemble import Ensemble
from persistence.repositories.api_response import ApiResponse
from containers import Services

router = APIRouter()
letter_fine_tuner = Services.letter_fine_tuner()


@router.post('/preprocess_inference')
async def predictions(file: UploadFile = File(...)):
    try:
        output_image, output_image_path = letter_fine_tuner.letter_finder(file=file)
        vector = feature_generation_test()
        my_model = Ensemble()
        inference_result = Inference(model_name=my_model, x_test=vector).start_inference()
        return ApiResponse(data=str(inference_result))

    except ImagePreprocessingException as e:
        return ApiResponse(success=False, error=e.__str__())
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())


