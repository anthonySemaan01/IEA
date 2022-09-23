import io
import numpy as np
from fastapi import APIRouter, UploadFile, File, Response
from persistence.repositories.api_response import ApiResponse
from application.feature_extraction.feature_generation_testing import feature_generation_test
from domain.exceptions.feature_generation_exception import FeatureGeneration
from domain.exceptions.feature_extraction_exception import FeatureExtraction

router = APIRouter()


@router.post('/inference')
async def predictions():
    try:
        vector = feature_generation_test()

    except FeatureExtraction as e:
        return ApiResponse(success=False, error=e.__str__())
    except FeatureGeneration as e:
        return ApiResponse(success=False, error=e.__str__())
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())

    return ApiResponse(data=str(vector))