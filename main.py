from fastapi import FastAPI
from starlette.middleware.cors import CORSMiddleware
from api.controllers import health_check_controller, image_preprocessing_controller, inference_controller

app = FastAPI(version='1.0', title='IEA backend API',
              description="Provide different image preprocessing and images-ubyte extraction")

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
    router=inference_controller.router,
    prefix="",
    tags=["inference"]
)

