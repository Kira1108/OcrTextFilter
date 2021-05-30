from fastapi import APIRouter, HTTPException
from app.pkgs.parser import OCRTextParser


router = APIRouter(
    prefix="/image",
    tags=["image"],
    responses={404: {"description": "Not found"}},
)


ocr_parser = OCRTextParser()

@router.get("/")
async def parse_image(path:str):
    r = ocr_parser.parse(path)
    return {"data":
                {"is_contain":any(r.values()),
                 "detail":r
                }
           }



