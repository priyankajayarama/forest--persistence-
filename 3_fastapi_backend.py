# =============================================================================
# WEEK 3 — FASTAPI BACKEND
# Forest Persistence Segmentation Capstone
# Prerequisite: Week 1 complete, models/best_model.pth exists
# =============================================================================


# =============================================================================
# CELL 1 — INSTALL WEEK 3 LIBRARIES
# =============================================================================
# !pip install fastapi uvicorn python-multipart prometheus-client pydantic -q


# =============================================================================
# CELL 2 — WRITE app.py TO DISK
# =============================================================================
APP_CODE = '''import io
import uuid
import logging
from datetime import datetime

import cv2
import numpy as np
import torch
import albumentations as A
from albumentations.pytorch import ToTensorV2
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from prometheus_client import Counter, Histogram, generate_latest
from starlette.responses import Response
import segmentation_models_pytorch as smp

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Forest Segmentation API",
    description="Predicts forest cover from satellite images using U-Net",
    version="1.0.0"
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# Prometheus metrics
PREDICTION_COUNT   = Counter("predictions_total",    "Total predictions served")
PREDICTION_LATENCY = Histogram("prediction_latency", "Prediction latency in seconds")

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    model = smp.Unet(
        encoder_name="resnet34",
        encoder_weights=None,
        in_channels=3,
        classes=1,
        activation=None
    )
    model.load_state_dict(torch.load("models/best_model.pth", map_location=DEVICE))
    model.eval()
    logger.info(f"Model loaded on {DEVICE}")
    return model.to(DEVICE)

MODEL = load_model()

transform = A.Compose([
    A.Resize(256, 256),
    A.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225)),
    ToTensorV2()
])


@app.get("/health")
def health():
    """Health check — confirms API and model are running."""
    return {
        "status":  "ok",
        "model":   "loaded",
        "device":  DEVICE,
        "version": "1.0.0"
    }


@app.post("/predict")
@PREDICTION_LATENCY.time()
async def predict(file: UploadFile = File(...)):
    """
    Accept a satellite image, return forest segmentation results.
    - Input:  PNG or JPG image (any size, auto-resized to 256x256)
    - Output: forest coverage %, prediction ID, timestamp
    """
    try:
        PREDICTION_COUNT.inc()

        # Read and decode image
        contents = await file.read()
        nparr    = np.frombuffer(contents, np.uint8)
        image    = cv2.imdecode(nparr, cv2.IMREAD_COLOR)

        if image is None:
            raise HTTPException(status_code=400, detail="Invalid image file")

        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        # Preprocess and run model
        aug    = transform(image=image)
        tensor = aug["image"].unsqueeze(0).to(DEVICE)

        with torch.no_grad():
            pred = torch.sigmoid(MODEL(tensor)).cpu().numpy()[0, 0]

        # Calculate forest coverage
        forest_pct = float((pred > 0.5).mean() * 100)

        return {
            "prediction_id":       str(uuid.uuid4()),
            "timestamp":           datetime.now().isoformat(),
            "forest_coverage_pct": round(forest_pct, 2),
            "mask_shape":          list(pred.shape),
            "model_version":       "1.0.0",
            "status":              "success"
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
def metrics():
    """Prometheus metrics endpoint for monitoring."""
    return Response(generate_latest(), media_type="text/plain")


@app.get("/")
def root():
    return {
        "message": "Forest Segmentation API",
        "docs":    "/docs",
        "health":  "/health",
        "predict": "POST /predict"
    }
'''

import os
os.makedirs("models", exist_ok=True)

with open("app.py", "w") as f:
    f.write(APP_CODE)

print("app.py written successfully!")


# =============================================================================
# CELL 3 — TEST THE API LOCALLY IN COLAB
# Run this cell, then test in a separate cell below
# =============================================================================

# First start the server in the background
import subprocess, time, os

server = subprocess.Popen(
    ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)
time.sleep(4)
print("API server started (PID:", server.pid, ")")
print("API docs: http://localhost:8000/docs")


# =============================================================================
# CELL 4 — TEST HEALTH ENDPOINT
# =============================================================================
import requests

response = requests.get("http://localhost:8000/health")
print("Status code:", response.status_code)
print("Response:   ", response.json())


# =============================================================================
# CELL 5 — TEST PREDICT ENDPOINT WITH A SYNTHETIC IMAGE
# =============================================================================
import numpy as np
from PIL import Image
import io
import requests

# Create a test image
test_img = Image.fromarray(
    np.random.randint(50, 180, (256, 256, 3), dtype=np.uint8)
)
buf = io.BytesIO()
test_img.save(buf, format="PNG")
buf.seek(0)

response = requests.post(
    "http://localhost:8000/predict",
    files={"file": ("test.png", buf, "image/png")}
)

print("Status code:", response.status_code)
print("Response:")
import json
print(json.dumps(response.json(), indent=2))


# =============================================================================
# CELL 6 — VIEW INTERACTIVE API DOCS URL
# (open this in your browser if running locally)
# =============================================================================
print("Interactive API docs:  http://localhost:8000/docs")
print("Alternative docs:      http://localhost:8000/redoc")
print("Health check:          http://localhost:8000/health")
print("Metrics:               http://localhost:8000/metrics")
print()
print("Week 3 complete! FastAPI backend is running.")
print("app.py saved — ready for Docker in Week 5.")
