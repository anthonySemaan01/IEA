from fastapi import APIRouter, UploadFile, File
from persistence.repositories.api_response import ApiResponse
from containers import Services
from model.model_cnn.cnn_model import ModelCNNDigits
import numpy as np

router = APIRouter()
letter_fine_tuner = Services.letter_fine_tuner()


@router.post('/infer_with_cnn')
async def predictions(file: UploadFile = File(...)):
    try:
        output_image, output_image_path = letter_fine_tuner.letter_finder(file=file, cnn=True)
        output_image_inverse = 255 - output_image
        x_test = np.expand_dims(output_image_inverse, -1)
        my_cnn_model = ModelCNNDigits
        prediction = ModelCNNDigits.classification_cnn(image=x_test)
        return ApiResponse(data=str(prediction))
    except Exception as e:
        print(e.__str__())
