import io

from fastapi import APIRouter, UploadFile, File, Response

from persistence.repositories.api_response import ApiResponse
from application.image_preprocessing.letter_fine_tuner import LetterFineTuner

router = APIRouter()
letter_fine_tuner = LetterFineTuner()


@router.post('/inference')
async def predictions(file: UploadFile = File(...), x_coord: int = 28, y_coord: int = 28, width: int = 28,
                      height: int = 28):
    image_cropped = letter_fine_tuner.image_cropper(file=file, x=x_coord, y=y_coord, width=width, height=height)
    print(type(image_cropped))
    letter_fine_tuner.find_non_white_pixels(image_cropped)
    return ApiResponse(data=image_cropped.shape)