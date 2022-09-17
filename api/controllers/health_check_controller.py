from fastapi import APIRouter, Response
from persistence.repositories.api_response import ApiResponse
router = APIRouter()


@router.get('/')
async def check_health():
    return ApiResponse(success=True)

