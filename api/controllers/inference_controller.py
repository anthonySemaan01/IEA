import io
import numpy as np
from fastapi import APIRouter, UploadFile, File, Response
from persistence.repositories.api_response import ApiResponse
from application.feature_extraction.feature_generation_testing import feature_generation_test
from domain.exceptions.feature_generation_exception import FeatureGeneration
from domain.exceptions.feature_extraction_exception import FeatureExtraction
from application.Inference.inference import Inference
from model.knn.knn import KNN
router = APIRouter()


@router.post('/inference')
async def predictions():
    try:
        vector = feature_generation_test()
        knn = KNN()
        inference_result = Inference(model_name=knn, x_test=vector).start_inference()
    except FeatureExtraction as e:
        return ApiResponse(success=False, error=e.__str__())
    except FeatureGeneration as e:
        return ApiResponse(success=False, error=e.__str__())
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())

    return ApiResponse(data=str(inference_result))