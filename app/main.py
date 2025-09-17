# app/main.py
import os
import requests
from io import BytesIO
from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

import model_loader

app = FastAPI(title="Crop Disease Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
def startup():
    try:
        model_loader.load_my_model()
    except Exception as e:
        print("Warning: model failed to load on startup. Server will still run.")
        print(str(e))

@app.get("/health")
async def health():
    status = model_loader.get_model_status()
    return JSONResponse(status)

@app.post("/predict")
async def predict_file(file: UploadFile = File(...), top_k: int = 1):
    """Endpoint for direct file upload (manual testing)."""
    if not file:
        raise HTTPException(status_code=400, detail="No file uploaded.")
    contents = await file.read()
    try:
        preds = model_loader.predict_from_bytes(contents, top_k=top_k)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
class PredictRequest(BaseModel):
    image_url: str
    top_k: int = 1

@app.post("/predict_url")
async def predict_url(req: PredictRequest):
    """Endpoint for Node service: send image URL, get prediction."""
    try:
        resp = requests.get(req.image_url)
        if resp.status_code != 200:
            raise HTTPException(status_code=400, detail="Failed to download image from URL")
        contents = resp.content
        preds = model_loader.predict_from_bytes(contents, top_k=req.top_k)
        return {"predictions": preds}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=int(os.getenv("PORT", "8000")), reload=True)