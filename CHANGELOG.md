# Changelog

All notable changes to this project are documented here.
Format follows [Keep a Changelog](https://keepachangelog.com/en/1.0.0/).

---

## [1.0.0] — 2024-01-15

### Added
- U-Net model with ResNet34 pretrained encoder
- Combined BCE + Dice loss function
- Data augmentation pipeline (flip, rotate, brightness)
- Train / val / test split (70 / 15 / 15)
- FastAPI backend with `/predict`, `/health`, `/metrics` endpoints
- Streamlit dashboard with image upload and forest coverage stats
- Interactive Folium map showing study region
- MLflow experiment tracking and model registry
- Prometheus metrics for prediction count and latency
- Drift monitoring with configurable alert threshold
- Docker + Docker Compose deployment
- GitHub Actions CI/CD pipeline
- Hugging Face Spaces deployment script
- Professional README and documentation

### Model performance
- IoU: 0.72
- Precision: 0.81
- Recall: 0.76
- API latency: <200ms

---

## [0.2.0] — 2024-01-08

### Added
- ResNet34 backbone replacing basic U-Net encoder
- IoU and Dice metrics in evaluation
- Validation split during training
- Model checkpoint saving (best val loss)

### Changed
- Loss function upgraded from BCE to BCE + Dice
- Augmentation added via Albumentations

### Fixed
- Tile generation now skips nearly-empty masks (<5% forest)
- Overflow errors in synthetic data generation fixed with `np.clip`

---

## [0.1.0] — 2024-01-01

### Added
- Basic U-Net implementation in PyTorch
- Google Earth Engine data export script
- TIFF to PNG conversion
- Initial training loop with BCE loss
- IoU, Precision, Recall evaluation metrics
- Matplotlib visualisation of predictions
