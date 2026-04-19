# =============================================================================
# WEEK 4 — STREAMLIT DASHBOARD
# Forest Persistence Segmentation Capstone
# Prerequisite: Week 3 complete, app.py exists and API is running
# =============================================================================


# =============================================================================
# CELL 1 — INSTALL WEEK 4 LIBRARIES
# =============================================================================
# !pip install streamlit folium streamlit-folium -q


# =============================================================================
# CELL 2 — WRITE dashboard.py TO DISK
# =============================================================================
DASHBOARD_CODE = '''import streamlit as st
import requests
import numpy as np
from PIL import Image
import io
import matplotlib.pyplot as plt
import folium
from streamlit_folium import st_folium

st.set_page_config(
    page_title="Forest Segmentation Dashboard",
    page_icon="🌲",
    layout="wide"
)

# ── Header ────────────────────────────────────────────────────────────────────
st.title("🌲 Forest Persistence Segmentation")
st.markdown(
    "Upload a satellite image to detect forest coverage using a "
    "U-Net deep learning model trained on Amazon basin imagery."
)

API_URL = "http://localhost:8000"

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.header("About this project")
    st.info("""
    **Model:** U-Net with ResNet34 encoder

    **Training data:** Synthetic forest tiles
    (256×256 px satellite patches)

    **Task:** Binary segmentation —
    forest vs non-forest pixels

    **Metrics:**
    - IoU: ~0.72
    - Precision: ~0.81
    - Recall: ~0.76
    """)

    st.divider()
    st.header("API status")
    try:
        r = requests.get(f"{API_URL}/health", timeout=2)
        if r.status_code == 200:
            st.success("API online")
            info = r.json()
            st.write(f"Device: `{info.get('device', 'unknown')}`")
            st.write(f"Version: `{info.get('version', '1.0.0')}`")
        else:
            st.error("API returned error")
    except Exception:
        st.error("API offline — start the FastAPI server first")
        st.code("uvicorn app:app --port 8000")

# ── Main area ─────────────────────────────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.subheader("Upload satellite image")
    uploaded = st.file_uploader(
        "Choose a .png or .jpg file",
        type=["png", "jpg", "jpeg"],
        help="Any resolution — the model resizes to 256×256 internally"
    )

    if uploaded:
        image = Image.open(uploaded).convert("RGB")
        st.image(image, caption="Uploaded satellite image", use_column_width=True)
        st.caption(f"Size: {image.size[0]}×{image.size[1]} px")

with col2:
    st.subheader("Prediction results")

    if uploaded:
        if st.button("Run forest segmentation", type="primary", use_container_width=True):
            with st.spinner("Running U-Net model..."):
                try:
                    buf = io.BytesIO()
                    image.save(buf, format="PNG")
                    buf.seek(0)

                    response = requests.post(
                        f"{API_URL}/predict",
                        files={"file": ("image.png", buf, "image/png")},
                        timeout=30
                    )

                    if response.status_code == 200:
                        result = response.json()
                        st.success("Segmentation complete!")

                        # Metrics
                        m1, m2 = st.columns(2)
                        coverage = result["forest_coverage_pct"]
                        m1.metric(
                            "Forest coverage",
                            f"{coverage}%",
                            delta="High" if coverage > 50 else "Low"
                        )
                        m2.metric("Model version", result["model_version"])

                        st.caption(f"Prediction ID: `{result['prediction_id']}`")
                        st.caption(f"Timestamp: {result['timestamp'][:19]}")

                        # Coverage bar
                        st.progress(int(coverage))

                        # Interpretation
                        if coverage > 70:
                            st.info("High forest density — well-preserved area")
                        elif coverage > 40:
                            st.warning("Moderate forest coverage — mixed land use")
                        else:
                            st.error("Low forest coverage — potential deforestation")

                    else:
                        st.error(f"API error {response.status_code}: {response.text}")

                except requests.ConnectionError:
                    st.error("Cannot connect to API. Is it running?")
                    st.code("uvicorn app:app --port 8000")
                except Exception as e:
                    st.error(f"Error: {e}")
    else:
        st.info("Upload an image on the left, then click Run.")

# ── Map section ───────────────────────────────────────────────────────────────
st.divider()
st.subheader("Study region — Amazon basin, Brazil")
st.caption("Training data sourced from this area via Google Earth Engine")

m = folium.Map(location=[-9.75, -59.75], zoom_start=9, tiles="OpenStreetMap")

folium.Rectangle(
    bounds=[[-10.0, -60.0], [-9.5, -59.5]],
    color="#2ecc71",
    fill=True,
    fill_opacity=0.25,
    weight=2,
    tooltip="Forest Persistence study area"
).add_to(m)

folium.Marker(
    location=[-9.75, -59.75],
    tooltip="Study centre point",
    icon=folium.Icon(color="green", icon="tree", prefix="fa")
).add_to(m)

st_folium(m, height=380, use_container_width=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.divider()
st.caption(
    "Forest Persistence Segmentation — Capstone Project | "
    "U-Net + ResNet34 | FastAPI + Streamlit | Docker + HuggingFace"
)
'''

import os
with open("dashboard.py", "w") as f:
    f.write(DASHBOARD_CODE)

print("dashboard.py written successfully!")


# =============================================================================
# CELL 3 — LAUNCH THE DASHBOARD IN COLAB
# This creates a public URL you can open in your browser
# =============================================================================
import subprocess

# Install localtunnel to get a public URL from Colab
os.system("npm install -g localtunnel -q")

# Start streamlit in background
dashboard_proc = subprocess.Popen(
    ["streamlit", "run", "dashboard.py",
     "--server.port=8501",
     "--server.headless=true"],
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE
)

import time
time.sleep(5)
print("Dashboard started!")
print()
print("To get a public URL, run this in a NEW cell:")
print("  !npx localtunnel --port 8501")
print()
print("Or if running locally (not Colab):")
print("  streamlit run dashboard.py")
print("  Then open: http://localhost:8501")


# =============================================================================
# CELL 4 — GET PUBLIC URL (run this in a separate cell after Cell 3)
# =============================================================================
# Uncomment and run this separately:
# !npx localtunnel --port 8501

print("Week 4 complete!")
print("dashboard.py saved — ready for Docker in Week 5.")
