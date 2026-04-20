# 🌲 Forest Persistence Segmentation

![Python](https://img.shields.io/badge/Python-3.10-blue?style=flat-square)
![PyTorch](https://img.shields.io/badge/PyTorch-2.0-orange?style=flat-square)
![FastAPI](https://img.shields.io/badge/FastAPI-0.104-green?style=flat-square)
![Docker](https://img.shields.io/badge/Docker-ready-blue?style=flat-square)
![License](https://img.shields.io/badge/License-MIT-yellow?style=flat-square)

A **production-ready deep learning system** that detects and segments persistent forest areas from satellite imagery. Built using a U-Net architecture with a pretrained ResNet34 encoder, served via a FastAPI backend,

---

## Overview

This project addresses a critical environmental challenge — **monitoring forest persistence** at scale using satellite imagery and deep learning. The system:

- Accepts satellite images and returns pixel-level forest segmentation masks
- Identifies which areas have maintained stable forest cover over 20 years
- Provides confidence scores and coverage statistics per image
- Tracks model performance over time with automated drift detection
- Scales to production with Docker, CI/CD, and cloud deployment

**Use cases:**
- Environmental monitoring organisations
- Climate researchers tracking deforestation
- Government agencies assessing land use change
- NGOs reporting on forest conservation progress

---

## Demo Screenshots

### Dashboard
```
[Upload satellite image] → [Run segmentation] → [View forest mask + coverage %]
```

### API Response
```json
{
  "prediction_id": "a3f2c1d4-...",
  "timestamp": "2024-01-15T10:23:45",
  "forest_coverage_pct": 67.3,
  "mask_shape": [256, 256],
  "model_version": "1.0.0"
}
```

---

## Architecture

```
                    ┌─────────────────────────────────┐
                    │        Streamlit Dashboard       │
                    │  Upload → Predict → Visualise   │
                    └──────────────┬──────────────────┘
                                   │ HTTP POST /predict
                    ┌──────────────▼──────────────────┐
                    │          FastAPI Backend         │
                    │   Authentication · Validation    │
                    │   Logging · Prometheus metrics   │
                    └──────────────┬──────────────────┘
                                   │
                    ┌──────────────▼──────────────────┐
                    │        U-Net Model               │
                    │   ResNet34 encoder (ImageNet)    │
                    │   Dice + BCE loss                │
                    │   256×256 input · 1 output class │
                    └──────────────┬──────────────────┘
                                   │
               ┌───────────────────▼──────────────────────┐
               │              MLflow Registry              │
               │  Experiment tracking · Model versioning  │
               └──────────────────────────────────────────┘
```

---

## Results

| Metric    | Score  | Description                        |
|-----------|--------|------------------------------------|
| IoU       | 0.72   | Intersection over Union            |
| Precision | 0.81   | Of predicted forest, % correct     |
| Recall    | 0.76   | Of actual forest, % detected       |
| Latency   | <200ms | API response time (single image)   |

**Training configuration:**
- Encoder: ResNet34 (pretrained on ImageNet)
- Loss: BCE + Dice combined
- Optimiser: AdamW (lr=1e-4, weight_decay=1e-4)
- Epochs: 50 with early stopping
- Augmentation: horizontal/vertical flip, rotation, brightness contrast

---

## Quick Start

```bash
# 1. Clone the repo
git clone https://github.com/YOUR_USERNAME/forest-segmentation.git
cd forest-segmentation

# 2. Run with Docker (easiest)
docker-compose up --build

# 3. Open in browser
# Dashboard:  http://localhost:8501
# API docs:   http://localhost:8000/docs
# Metrics:    http://localhost:9090
```

---

## Installation

### Requirements
- Python 3.10+
- Docker + Docker Compose (for containerised deployment)
- 4GB RAM minimum (8GB recommended for training)

### Local setup (without Docker)

```bash
# Clone
git clone https://github.com/YOUR_USERNAME/forest-segmentation.git
cd forest-segmentation

# Create virtual environment
python -m venv venv
source venv/bin/activate      # Linux/Mac
# venv\Scripts\activate       # Windows

# Install dependencies
pip install -r requirements.txt

# Run the API
uvicorn app:app --host 0.0.0.0 --port 8000 --reload

# In a new terminal, run the dashboard
streamlit run dashboard.py
```

### Google Colab setup

```python
!pip install segmentation-models-pytorch mlflow albumentations \
             fastapi uvicorn python-multipart streamlit folium \
             streamlit-folium prometheus-client pydantic \
             pillow opencv-python-headless -q
```

---

## Usage

### Via the dashboard

1. Open `http://localhost:8501`
2. Upload a satellite image (PNG or JPG, any size)
3. Click **Run forest segmentation**
4. View the predicted mask and forest coverage percentage

### Via the API

```python
import requests

with open("satellite_image.png", "rb") as f:
    response = requests.post(
        "http://localhost:8000/predict",
        files={"file": f}
    )

result = response.json()
print(f"Forest coverage: {result['forest_coverage_pct']}%")
```

### Via cURL

```bash
curl -X POST "http://localhost:8000/predict" \
     -H "accept: application/json" \
     -F "file=@satellite_image.png"
```

---

## Project Structure

```
forest-segmentation/
│
├── app.py                        # FastAPI backend
├── dashboard.py                  # Streamlit frontend
├── requirements.txt              # Python dependencies
├── docker-compose.yml            # Run all services together
├── Dockerfile                    # API container
├── Dockerfile.dashboard          # Dashboard container
├── README.md                     # This file
├── CHANGELOG.md                  # Version history
├── CONTRIBUTING.md               # How to contribute
├── LICENSE                       # MIT license
│
├── models/
│   └── best_model.pth            # Trained model weights
│
├── data/
│   ├── train/
│   │   ├── images/               # Training satellite tiles
│   │   └── masks/                # Training forest masks
│   ├── val/
│   │   ├── images/               # Validation tiles
│   │   └── masks/                # Validation masks
│   └── test/
│       ├── images/               # Test tiles
│       └── masks/                # Test masks
│
├── outputs/
│   ├── loss_curve.png            # Training loss plot
│   └── predictions.png           # Sample predictions
│
├── monitoring/
│   ├── prometheus.yml            # Metrics scrape config
│   └── logs/
│       └── predictions.jsonl     # Prediction audit log
│
├── scripts/
│   └── deploy_to_hf.py           # Hugging Face deployment
│
└── .github/
    └── workflows/
        └── ci_cd.yml             # GitHub Actions CI/CD
```

---

## Data

### Source
Forest persistence data from **Google Earth Engine**:
```
projects/forestdatapartnership/assets/community_forests/ForestPersistence_2020
```

This dataset shows how stable forest cover was across a 20-year period (2000–2020). Values close to 1.0 indicate persistent stable forest; values near 0 indicate non-forest or recently cleared areas.

### Study region
Amazon basin: `BBox(-60.0, -10.0, -59.5, -9.5)`

### Processing pipeline
1. Export `.tif` from GEE at 30m resolution
2. Convert to PNG tiles of 256×256 pixels
3. Filter tiles with <5% forest coverage (non-informative)
4. Split: 70% train / 15% val / 15% test (stratified, seed=42)

---

## Model

### Architecture: U-Net with ResNet34 encoder

U-Net is a fully convolutional network designed for biomedical image segmentation, extended here for satellite imagery. The encoder-decoder structure with skip connections preserves spatial detail while learning high-level features.

**Why ResNet34?**
- Pretrained on ImageNet — transfers visual features to satellite imagery
- Lightweight enough for fast inference (<200ms per image)
- Proven performance on segmentation benchmarks

### Loss function: BCE + Dice

```python
total_loss = BCE_loss + Dice_loss
```

Plain cross-entropy treats each pixel equally, which is problematic when forest and non-forest pixels are imbalanced. Dice loss directly optimises the overlap between predicted and true masks.

### Training details

| Parameter     | Value            |
|---------------|------------------|
| Input size    | 256 × 256        |
| Batch size    | 4                |
| Learning rate | 1e-4             |
| Optimiser     | AdamW            |
| Scheduler     | ReduceLROnPlateau|
| Epochs        | 50               |
| Early stopping| patience=5       |

---

## API Reference

### `GET /health`
Returns system health status.

**Response:**
```json
{"status": "ok", "model": "loaded", "device": "cuda"}
```

### `POST /predict`
Accepts a satellite image and returns segmentation results.

**Request:** `multipart/form-data` with field `file` (PNG or JPG)

**Response:**
```json
{
  "prediction_id": "uuid-string",
  "timestamp": "2024-01-15T10:23:45",
  "forest_coverage_pct": 67.3,
  "mask_shape": [256, 256],
  "model_version": "1.0.0"
}
```

**Error codes:**
- `400` — Invalid file format
- `500` — Model inference error

### `GET /metrics`
Returns Prometheus-format metrics for monitoring.

Full interactive docs available at `http://localhost:8000/docs`

---

## Deployment

### Docker (recommended)

```bash
docker-compose up --build
```

This starts:
- API on port 8000
- Dashboard on port 8501
- Prometheus on port 9090
- Grafana on port 3000

### Hugging Face Spaces (free cloud hosting)

```bash
# Set your token
export HF_TOKEN=your_token_here

# Deploy
python scripts/deploy_to_hf.py
```

Visit: `https://huggingface.co/spaces/YOUR_USERNAME/forest-segmentation`

### CI/CD

Every push to `main` automatically:
1. Runs the test suite
2. Builds the Docker image
3. Deploys to Hugging Face Spaces

---

## Monitoring

### Prometheus metrics
- `predictions_total` — total number of predictions served
- `prediction_latency` — histogram of response times

### Drift detection
The `DriftMonitor` class tracks average forest coverage over time. If recent predictions shift >15% from the baseline, an alert is triggered — a signal that input data distribution has changed and the model may need retraining.

```python
monitor = DriftMonitor(baseline_coverage=45.0)
monitor.log_prediction(id, forest_pct)
monitor.summary()
```

---

## Testing

```bash
# Run all tests
pytest tests/ -v

# Run with coverage report
pytest tests/ --cov=. --cov-report=html
```

Test coverage targets: >80% across all modules.

---

## Tech Stack

| Layer         | Technology                              |
|---------------|-----------------------------------------|
| Model         | PyTorch, segmentation-models-pytorch    |
| Training      | MLflow, Albumentations                  |
| Backend API   | FastAPI, Uvicorn, Pydantic              |
| Frontend      | Streamlit, Folium                       |
| Monitoring    | Prometheus, Grafana                     |
| Containerisation | Docker, Docker Compose               |
| CI/CD         | GitHub Actions                          |
| Hosting       | Hugging Face Spaces                     |
| Data          | Google Earth Engine, PIL, OpenCV        |

---

---

## Acknowledgements

- [Forest Data Partnership](https://www.forestdatapartnership.org/) for the ForestPersistence_2020 dataset
- [Google Earth Engine](https://earthengine.google.com/) for satellite data access
- [segmentation-models-pytorch](https://github.com/qubvel/segmentation_models.pytorch) for the U-Net implementation
- Anthropic's Claude for development assistance

---

*Built as a capstone project for the Data Science career programme.*
