from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os

app = FastAPI()

# Setup templates and static folders
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your trained model
model = YOLO("runs/classify/train_dog_disease_v2/weights/best.pt")

@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(img)[0]  # <-- FIXED HERE

        pred_class = results.names[results.probs.top1]
        confidence = results.probs.top1conf.item()

        result = {
            "filename": file.filename,
            "prediction": pred_class,
            "confidence": f"{confidence:.2%}"
        }

        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": result
        })

    except Exception as e:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"error": str(e)}
        })
