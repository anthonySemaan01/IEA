import io

from fastapi import APIRouter, UploadFile, File, Response

from persistence.repositories.api_response import ApiResponse
from containers import Services

router = APIRouter()
letter_fine_tuner = Services.letter_fine_tuner()


@router.post('/inference')
async def predictions(file: UploadFile = File(...)):
    letter_fine_tuner.letter_finder(file=file)
    return ApiResponse(data="Done")