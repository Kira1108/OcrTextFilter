from fastapi import APIRouter, HTTPException
from app.pkgs.parser import TextParser
from pydantic import BaseModel
from typing import List


router = APIRouter(
    prefix="/text",
    tags=["text"],
    responses={404: {"description": "Not found"}},
)


text_parser = TextParser()

class Text(BaseModel):
    text:List[str]

@router.post("/")
async def parse_text(text:Text):
    r = text_parser.parse(text.text)
    return {"data":
                {"is_contain":any(r.values()),
                 "detail":r
                }
           }
