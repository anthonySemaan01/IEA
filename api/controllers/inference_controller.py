from fastapi import APIRouter

from application.Inference.inference import Inference
from application.feature_extraction.feature_generation_testing import feature_generation_test
from domain.exceptions.feature_extraction_exception import FeatureExtraction
from domain.exceptions.feature_generation_exception import FeatureGeneration
from model.ensemble import Ensemble
from persistence.repositories.api_response import ApiResponse

router = APIRouter()


@router.post('/inference')
async def predictions(retrain: bool = False, output_label: str = ""):
    try:
        vector = feature_generation_test(retrain=retrain, output_label=output_label)
        my_model = Ensemble()
        inference_result = Inference(model_name=my_model, x_test=vector).start_inference()
    except FeatureExtraction as e:
        return ApiResponse(success=False, error=e.__str__())
    except FeatureGeneration as e:
        return ApiResponse(success=False, error=e.__str__())
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())

    return ApiResponse(data=str(inference_result))
