from fastapi import APIRouter, UploadFile, File

from application.feature_extraction.Training_feature_generation import TrainingFeatureGeneration
from application.feature_extraction.Training_feature_generation_2 import TrainingFeatureGeneration2
from containers import Services
from domain.exceptions.feature_extraction_exception import FeatureExtraction
from domain.exceptions.feature_generation_exception import FeatureGeneration
from persistence.repositories.api_response import ApiResponse

router = APIRouter()
letter_fine_tuner = Services.letter_fine_tuner()


@router.post('/re-train1')
async def predictions(image: UploadFile = File(...), output_label: str = ""):
    try:
        output_image, output_image_path = letter_fine_tuner.letter_finder(file=image)
        vector = TrainingFeatureGeneration.feature_generation_test(output_label=output_label)

    except FeatureExtraction as e:
        return ApiResponse(success=False, error=e.__str__())
    except FeatureGeneration as e:
        return ApiResponse(success=False, error=e.__str__())
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())

    return ApiResponse(data=str(vector))


@router.post('/re-train2')
async def predictions(image: UploadFile = File(...), output_label: str = ""):
    try:
        output_image, output_image_path = letter_fine_tuner.letter_finder(file=image)
        vector = TrainingFeatureGeneration2.feature_generation_test(output_label=output_label)

    except FeatureExtraction as e:
        return ApiResponse(success=False, error=e.__str__())
    except FeatureGeneration as e:
        return ApiResponse(success=False, error=e.__str__())
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())

    return ApiResponse(data=str(vector))
