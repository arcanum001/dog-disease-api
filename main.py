from fastapi import FastAPI, File, UploadFile, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from ultralytics import YOLO
from PIL import Image
import io
import os
import torch

# ✅ Patch PyTorch safe unpickling
from torch.serialization import add_safe_globals
from torch.nn.modules.container import Sequential
add_safe_globals([Sequential])

# ✅ Override torch.load to force weights_only=False
_original_torch_load = torch.load
def patched_torch_load(*args, **kwargs):
    kwargs['weights_only'] = False
    return _original_torch_load(*args, **kwargs)
torch.load = patched_torch_load

# ✅ App setup
app = FastAPI()

# ✅ Enable CORS (for your Next.js frontend)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # Replace * with frontend domain in production
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

templates = Jinja2Templates(directory="templates")
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Load model
model_path = "runs/classify/train_dog_disease_v2/weights/best.pt"
if os.path.exists(model_path):
    model = YOLO(model_path)
else:
    model = None
    print("⚠ Model file not found:", model_path)


# ✅ Root route (HTML form)
@app.get("/", response_class=HTMLResponse)
async def home(request: Request):
    return templates.TemplateResponse("index.html", {"request": request, "result": None})


# ✅ HTML upload handler
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

        results = model(img)[0]
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


# ✅ JSON prediction endpoint (for Next.js frontend)
@app.post("/predict-json")
async def predict_json(file: UploadFile = File(...)):
    if not model:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": "Model not loaded. Please check the model path."}
        )

    try:
        contents = await file.read()
        img = Image.open(io.BytesIO(contents)).convert("RGB")

        results = model(img)[0]
        pred_class = results.names[results.probs.top1]
        confidence = results.probs.top1conf.item()

        return {
            "success": True,
            "prediction": pred_class,
            "confidence": confidence  # e.g. 0.893
        }

    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={"success": False, "error": str(e)}
        )


# ✅ Run locally (optional)
if _name_ == "_main_": # type: ignore
    import uvicorn
    port = int(os.environ.get("PORT", 8080))
    uvicorn.run("main:app", host="0.0.0.0", port=port)