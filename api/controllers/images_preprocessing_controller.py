import io

from fastapi import APIRouter, UploadFile, File, Response
from domain.exceptions.image_preprocessing_exception import ImagePreprocessingException
from persistence.repositories.api_response import ApiResponse
from containers import Services

router = APIRouter()
letter_fine_tuner = Services.letter_fine_tuner()


@router.post('/inference')
async def predictions(file: UploadFile = File(...)):
    try:
        letter_fine_tuner.letter_finder(file=file)
    except Exception as e:
        return ApiResponse(success=False, error=e.__str__())
    except ImagePreprocessingException as e:
        return ApiResponse(success=False, error=e.__str__())

    return ApiResponse(data="Done")