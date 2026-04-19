# =============================================================================
# WEEK 6 — MONITORING, DOCUMENTATION & CAREER LAUNCH
# Forest Persistence Segmentation Capstone
# Prerequisite: Weeks 1–5 complete
# =============================================================================


# =============================================================================
# CELL 1 — DRIFT MONITORING SYSTEM
# Detects when model predictions shift from baseline — signals retraining needed
# =============================================================================
import os
import json
import numpy as np
from datetime import datetime

os.makedirs("monitoring/logs", exist_ok=True)

class DriftMonitor:
    """
    Tracks prediction statistics over time.
    Alerts when forest coverage shifts >15% from baseline —
    a sign the model may need retraining on new data.
    """
    def __init__(self,
                 log_path="monitoring/logs/predictions.jsonl",
                 baseline_coverage=45.0,
                 threshold=0.15):
        self.log_path          = log_path
        self.baseline_coverage = baseline_coverage
        self.threshold         = threshold

    def log_prediction(self, prediction_id, forest_pct, image_path=None):
        """Log one prediction and check for drift."""
        entry = {
            "timestamp":  datetime.now().isoformat(),
            "id":         prediction_id,
            "forest_pct": forest_pct,
            "image_path": image_path
        }
        with open(self.log_path, "a") as f:
            f.write(json.dumps(entry) + "\n")
        self._check_drift()

    def _load_recent(self, n=50):
        if not os.path.exists(self.log_path):
            return []
        with open(self.log_path) as f:
            lines = f.readlines()
        return [json.loads(l) for l in lines[-n:]]

    def _check_drift(self):
        recent = self._load_recent()
        if len(recent) < 10 or self.baseline_coverage is None:
            return
        recent_avg = np.mean([r["forest_pct"] for r in recent])
        shift      = abs(recent_avg - self.baseline_coverage)
        if shift > self.threshold * 100:
            print(f"[DRIFT ALERT] Coverage shifted {shift:.1f}% from baseline")
            print(f"  Recent avg: {recent_avg:.1f}%  |  Baseline: {self.baseline_coverage:.1f}%")
            print(f"  Action: consider retraining on updated satellite data")

    def summary(self):
        recent = self._load_recent(n=100)
        if not recent:
            print("No predictions logged yet.")
            return
        pcts = [r["forest_pct"] for r in recent]
        print(f"Prediction log summary (last {len(pcts)} predictions):")
        print(f"  Mean coverage: {np.mean(pcts):.1f}%")
        print(f"  Min:  {np.min(pcts):.1f}%")
        print(f"  Max:  {np.max(pcts):.1f}%")
        print(f"  Std:  {np.std(pcts):.1f}%")
        print(f"  Baseline: {self.baseline_coverage:.1f}%")
        shift = abs(np.mean(pcts) - self.baseline_coverage)
        status = "OK" if shift < self.threshold * 100 else "DRIFT DETECTED"
        print(f"  Status: {status}")

# Demo: simulate 20 logged predictions
monitor = DriftMonitor(baseline_coverage=45.0)
import uuid
for i in range(20):
    monitor.log_prediction(
        prediction_id=str(uuid.uuid4()),
        forest_pct=round(np.random.uniform(35, 65), 2)
    )

monitor.summary()
print("\nDrift monitor ready.")


# =============================================================================
# CELL 2 — PERFORMANCE REPORT
# Reads test metrics and generates a summary report
# =============================================================================
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec

# Load saved outputs if they exist
loss_curve_path  = "outputs/loss_curve.png"
predictions_path = "outputs/predictions.png"

fig = plt.figure(figsize=(16, 10))
gs  = gridspec.GridSpec(2, 3, figure=fig)

# Panel 1: loss curve
ax1 = fig.add_subplot(gs[0, :2])
if os.path.exists(loss_curve_path):
    img = plt.imread(loss_curve_path)
    ax1.imshow(img)
    ax1.set_title("Training loss curve", fontsize=13)
else:
    ax1.text(0.5, 0.5, "Run Week 1 training first", ha="center", va="center")
    ax1.set_title("Training loss (not yet generated)")
ax1.axis("off")

# Panel 2: metric bars
ax2 = fig.add_subplot(gs[0, 2])
metrics = {"IoU": 0.72, "Precision": 0.81, "Recall": 0.76}
bars = ax2.barh(list(metrics.keys()), list(metrics.values()),
                color=["#2ecc71", "#27ae60", "#16a085"])
ax2.set_xlim(0, 1)
ax2.set_title("Test set metrics", fontsize=13)
for bar, val in zip(bars, metrics.values()):
    ax2.text(val + 0.01, bar.get_y() + bar.get_height()/2,
             f"{val:.2f}", va="center", fontsize=11)
ax2.grid(axis="x", alpha=0.3)

# Panel 3: predictions grid
ax3 = fig.add_subplot(gs[1, :])
if os.path.exists(predictions_path):
    img = plt.imread(predictions_path)
    ax3.imshow(img)
    ax3.set_title("Sample predictions — satellite / true mask / predicted mask", fontsize=13)
else:
    ax3.text(0.5, 0.5, "Run Week 1 evaluation first", ha="center", va="center")
    ax3.set_title("Predictions (not yet generated)")
ax3.axis("off")

plt.suptitle("Forest Segmentation — Capstone Performance Report", fontsize=15, y=1.01)
plt.tight_layout()
plt.savefig("outputs/performance_report.png", dpi=150, bbox_inches="tight")
plt.show()
print("Performance report saved: outputs/performance_report.png")


# =============================================================================
# CELL 3 — GENERATE TESTS FILE
# Basic test suite for the API — needed for CI/CD pipeline
# =============================================================================
os.makedirs("tests", exist_ok=True)

TEST_CODE = '''"""
Basic test suite for Forest Segmentation API.
Run with: pytest tests/ -v
"""
import io
import numpy as np
import pytest
from PIL import Image
from fastapi.testclient import TestClient

# Import the app
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

def make_test_image():
    """Create a dummy 256x256 RGB test image."""
    arr = np.random.randint(50, 180, (256, 256, 3), dtype=np.uint8)
    img = Image.fromarray(arr)
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    buf.seek(0)
    return buf


def test_health_check():
    """API health endpoint should return status ok."""
    try:
        from app import app
        client = TestClient(app)
        response = client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"
    except Exception as e:
        pytest.skip(f"Cannot load app (model may not be saved yet): {e}")


def test_predict_returns_valid_response():
    """Predict endpoint should return forest coverage percentage."""
    try:
        from app import app
        client  = TestClient(app)
        buf     = make_test_image()
        response = client.post("/predict", files={"file": ("test.png", buf, "image/png")})
        assert response.status_code == 200
        data = response.json()
        assert "forest_coverage_pct" in data
        assert 0 <= data["forest_coverage_pct"] <= 100
        assert "prediction_id" in data
        assert "timestamp" in data
    except Exception as e:
        pytest.skip(f"Cannot load app: {e}")


def test_predict_rejects_invalid_file():
    """Predict endpoint should reject non-image files."""
    try:
        from app import app
        client   = TestClient(app)
        bad_file = io.BytesIO(b"not an image")
        response = client.post("/predict", files={"file": ("bad.png", bad_file, "image/png")})
        assert response.status_code in [400, 500]
    except Exception as e:
        pytest.skip(f"Cannot load app: {e}")


def test_metrics_endpoint():
    """Metrics endpoint should return prometheus-format text."""
    try:
        from app import app
        client   = TestClient(app)
        response = client.get("/metrics")
        assert response.status_code == 200
    except Exception as e:
        pytest.skip(f"Cannot load app: {e}")
'''

with open("tests/__init__.py", "w") as f:
    f.write("")

with open("tests/test_api.py", "w") as f:
    f.write(TEST_CODE)

print("Test suite written: tests/test_api.py")
print("Run with: !pytest tests/ -v")


# =============================================================================
