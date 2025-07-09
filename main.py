from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from ultralytics import YOLO
from PIL import Image
import io
import os

# 🛡️ Patch PyTorch safe unpickling
from torch.serialization import add_safe_globals
from torch.nn.modules.container import Sequential
add_safe_globals([Sequential])

app = FastAPI()

# Set up Jinja2 template directory and static files
templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# Load your trained model (handle file not found for safety)
model_path = "runs/classify/train_dog_disease_v2/weights/best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    model = None
    print("⚠️ Model file not found:", model_path)

# Home page route
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})

# Prediction route
@app.post("/predict", response_class=HTMLResponse)
async def predict(request: Request, file: UploadFile = File(...)):
    if not model:
        return templates.TemplateResponse("index.html", {
            "request": request,
            "result": {"error": "Model not loaded. Please check the model path."}
        })

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(img)[0]  # Get first result
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

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Use PORT env variable if present
    uvicorn.run("main:app", host="0.0.0.0", port=port)
