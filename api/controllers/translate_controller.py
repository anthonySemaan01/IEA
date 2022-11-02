from fastapi import APIRouter
from persistence.repositories.api_response import ApiResponse
from translate import Translator

router = APIRouter()


@router.get('/translate')
async def translate(word: str):
    try:
        translator= Translator(to_lang="es")
        translation = translator.translate(word)
        return {"success": True, "data": translation}
    except:
        return {"success": False, "data": False}

