from fastapi import FastAPI
from .routers import image, text



app = FastAPI()

app.include_router(image.router)
app.include_router(text.router)

@app.get("/")
async def root():
    return {"message":"success"}



