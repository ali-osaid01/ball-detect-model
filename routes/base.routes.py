from fastapi import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter()

@router.get("/")
async def root():
    return JSONResponse({"message": "Welcome to Toucher App -> AI Model YOLO"})