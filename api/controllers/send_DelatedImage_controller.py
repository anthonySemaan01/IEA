from fastapi import APIRouter
from persistence.repositories.api_response import ApiResponse
import os
import base64

router = APIRouter()


@router.get('/send_Delated_Image')
async def send_Delated_Image():
    ImagePath = "datasets/eroded_dilated/tester.png" # Delated image path For Anthony
    
    if(os.path.exists(ImagePath)):
        with open(ImagePath, 'rb') as f:
            base64image = base64.b64encode(f.read())
        return {"imageName": f.name,"data": base64image}
    
    return {"error": "image not found"}

