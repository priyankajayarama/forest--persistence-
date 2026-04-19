# =============================================================================
# WEEK 5 — DOCKER + DEPLOYMENT
# Forest Persistence Segmentation Capstone
# Prerequisite: Weeks 1–4 complete (app.py, dashboard.py, models/ exist)
# =============================================================================


# =============================================================================
# CELL 1 — WRITE ALL DOCKER FILES
# =============================================================================
import os

os.makedirs("monitoring", exist_ok=True)
os.makedirs("scripts",    exist_ok=True)
os.makedirs(".github/workflows", exist_ok=True)

# ── Dockerfile (API) ──────────────────────────────────────────────────────────
DOCKERFILE_API = """FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY app.py .
COPY models/ models/

EXPOSE 8000

HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD curl -f http://localhost:8000/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000"]
"""

# ── Dockerfile (Dashboard) ────────────────────────────────────────────────────
DOCKERFILE_DASHBOARD = """FROM python:3.10-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY dashboard.py .

EXPOSE 8501

CMD ["streamlit", "run", "dashboard.py", \\
     "--server.port=8501", \\
     "--server.address=0.0.0.0", \\
     "--server.headless=true"]
"""

# ── docker-compose.yml ────────────────────────────────────────────────────────
DOCKER_COMPOSE = """version: "3.8"

services:
  api:
    build:
      context: .
      dockerfile: Dockerfile
    ports:
      - "8000:8000"
    volumes:
      - ./models:/app/models
      - ./outputs:/app/outputs
    environment:
      - PYTHONUNBUFFERED=1
    restart: unless-stopped

  dashboard:
    build:
      context: .
      dockerfile: Dockerfile.dashboard
    ports:
      - "8501:8501"
    depends_on:
      - api
    environment:
      - API_URL=http://api:8000
    restart: unless-stopped

  prometheus:
    image: prom/prometheus:latest
    ports:
      - "9090:9090"
    volumes:
      - ./monitoring/prometheus.yml:/etc/prometheus/prometheus.yml
    restart: unless-stopped

  grafana:
    image: grafana/grafana:latest
    ports:
      - "3000:3000"
    depends_on:
      - prometheus
    environment:
      - GF_SECURITY_ADMIN_PASSWORD=admin
    restart: unless-stopped
"""

# ── requirements.txt ──────────────────────────────────────────────────────────
REQUIREMENTS = """torch==2.0.1
segmentation-models-pytorch==0.3.3
albumentations==1.3.1
fastapi==0.104.1
uvicorn==0.24.0
streamlit==1.28.0
streamlit-folium==0.15.0
folium==0.14.0
mlflow==2.8.0
prometheus-client==0.18.0
opencv-python-headless==4.8.1.78
Pillow==10.1.0
numpy==1.24.3
pydantic==2.4.2
python-multipart==0.0.6
requests==2.31.0
"""

# ── Prometheus config ─────────────────────────────────────────────────────────
PROMETHEUS = """global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'forest-segmentation-api'
    static_configs:
      - targets: ['api:8000']
    metrics_path: '/metrics'
"""

# Write all files
files = {
    "Dockerfile":                  DOCKERFILE_API,
    "Dockerfile.dashboard":        DOCKERFILE_DASHBOARD,
    "docker-compose.yml":          DOCKER_COMPOSE,
    "requirements.txt":            REQUIREMENTS,
    "monitoring/prometheus.yml":   PROMETHEUS,
}

for path, content in files.items():
    with open(path, "w") as f:
        f.write(content)
    print(f"Written: {path}")

print("\nAll Docker files ready!")


# =============================================================================
# CELL 2 — GITHUB ACTIONS CI/CD PIPELINE
# =============================================================================
CICD = """name: Forest Segmentation CI/CD

on:
  push:
    branches: [main]
  pull_request:
    branches: [main]

jobs:
  test:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v3

      - name: Set up Python 3.10
        uses: actions/setup-python@v4
        with:
          python-version: "3.10"

      - name: Install dependencies
        run: pip install -r requirements.txt

      - name: Run tests
        run: python -m pytest tests/ -v --tb=short

  build:
    needs: test
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Build API Docker image
        run: docker build -t forest-segmentation-api:latest .

      - name: Build Dashboard Docker image
        run: docker build -f Dockerfile.dashboard -t forest-segmentation-dashboard:latest .

  deploy:
    needs: build
    runs-on: ubuntu-latest
    if: github.ref == 'refs/heads/main'
    steps:
      - uses: actions/checkout@v3

      - name: Deploy to Hugging Face Spaces
        env:
          HF_TOKEN: ${{ secrets.HF_TOKEN }}
        run: |
          pip install huggingface_hub
          python scripts/deploy_to_hf.py
"""

with open(".github/workflows/ci_cd.yml", "w") as f:
    f.write(CICD)

print("CI/CD pipeline written: .github/workflows/ci_cd.yml")


# =============================================================================
# CELL 3 — DEPLOY TO HUGGING FACE SPACES (free public hosting)
# =============================================================================
HF_DEPLOY = '''"""
Deploys the Streamlit dashboard to Hugging Face Spaces.

Before running:
1. Create a free account at huggingface.co
2. Get your token from: huggingface.co/settings/tokens
3. Replace YOUR_HF_USERNAME below
4. Run: HF_TOKEN=your_token python scripts/deploy_to_hf.py
"""
import os
from huggingface_hub import HfApi

HF_USERNAME = "YOUR_HF_USERNAME"
SPACE_NAME  = "forest-segmentation"

api = HfApi(token=os.environ.get("HF_TOKEN"))

# Create the Space if it doesn\'t exist
try:
    api.create_repo(
        repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
        repo_type="space",
        space_sdk="streamlit",
        exist_ok=True,
        private=False
    )
    print(f"Space ready: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
except Exception as e:
    print(f"Note: {e}")

# Upload files
files_to_upload = [
    ("dashboard.py",           "app.py"),
    ("requirements.txt",       "requirements.txt"),
    ("models/best_model.pth",  "models/best_model.pth"),
]

for local_path, remote_path in files_to_upload:
    if os.path.exists(local_path):
        api.upload_file(
            path_or_fileobj=local_path,
            path_in_repo=remote_path,
            repo_id=f"{HF_USERNAME}/{SPACE_NAME}",
            repo_type="space"
        )
        print(f"Uploaded: {local_path}")
    else:
        print(f"Skipped (not found): {local_path}")

print()
print(f"Deployed to: https://huggingface.co/spaces/{HF_USERNAME}/{SPACE_NAME}")
'''

with open("scripts/deploy_to_hf.py", "w") as f:
    f.write(HF_DEPLOY)

print("Deployment script written: scripts/deploy_to_hf.py")


# =============================================================================
# CELL 4 — DEPLOYMENT INSTRUCTIONS
# =============================================================================
print("""
=== HOW TO DEPLOY ===

Option A — Run locally with Docker:
  docker-compose up --build
  Then open: http://localhost:8501

Option B — Deploy to Hugging Face (free, public URL):
  1. Create account: huggingface.co
  2. Get token: huggingface.co/settings/tokens
  3. Edit scripts/deploy_to_hf.py — set YOUR_HF_USERNAME
  4. Run: HF_TOKEN=your_token python scripts/deploy_to_hf.py

Option C — Push to GitHub (CI/CD auto-deploys):
  1. Create GitHub repo
  2. Add HF_TOKEN as a GitHub Secret
  3. git push origin main
  4. GitHub Actions builds and deploys automatically

=== MONITORING ===
After docker-compose up:
  Grafana:    http://localhost:3000  (login: admin/admin)
  Prometheus: http://localhost:9090
  API docs:   http://localhost:8000/docs

Week 5 complete!
""")
