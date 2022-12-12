from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware

from api.controllers import health_check_controller, image_preprocessing_controller, training_controller, \
    preprocess_infer_combined, translate_controller, send_BBImage_controller, send_CroppedImage_controller, send_DelatedImage_controller, send_ResizedImage_controller

app = FastAPI(version='1.0', title='IEA backend API',
              description="Provide a detailed image preprocessing and feature extraction pipeline")

# resolve CORS problems related to calling the API from another local server
app.add_middleware(CORSMiddleware,
                   allow_origins=["*"],
                   allow_credentials=True,
                   allow_methods=["*"],
                   allow_headers=["*"],
                   )

app.include_router(
    router=health_check_controller.router,
    prefix="/health",
    tags=["health"]
)

app.include_router(
    router=image_preprocessing_controller.router,
    prefix="",
    tags=["preprocessing"]
)

app.include_router(
    router=training_controller.router,
    prefix="",
    tags=["inference"]
)

app.include_router(
    router=preprocess_infer_combined.router,
    prefix="",
    tags=["All_In_All"]
)

app.include_router(
    router=translate_controller.router,
    prefix="",
    tags=["Translation"]
)

app.include_router(
    router=send_BBImage_controller.router,
    prefix="",
    tags=["Bounding Box"]
)

app.include_router(
    router=send_CroppedImage_controller.router,
    prefix="",
    tags=["Cropped Image"]
)

app.include_router(
    router=send_DelatedImage_controller.router,
    prefix="",
    tags=["Delated Image"]
)

app.include_router(
    router=send_ResizedImage_controller.router,
    prefix="",
    tags=["Resized Image"]
)