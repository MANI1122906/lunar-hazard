# Lunar-Hazard-Detective

An AI-powered detection system for identifying lunar hazards (boulders and landslides) using Chandrayaan satellite imagery.

## Project Overview

This project combines multiple deep learning models to detect and classify lunar surface hazards:
- **Boulder Detection**: YOLOv11-based object detection
- **Landslide Detection**: U-Net based semantic segmentation
- **Slope Analysis**: Data Terrain Model (DTM) processing

## Project Structure

```
Lunar-Hazard-Detective/
├── data/                   # Dataset storage
│   ├── raw/                # Original Chandrayaan .tif / .hdf5 files
│   ├── processed/          # Enhanced images (CLAHE applied)
│   └── masks/              # Ground truth labels for training
├── models/                 # Trained model weights
│   ├── boulder_yolo.pt     # YOLOv11 weights
│   └── landslide_unet.pth  # U-Net weights
├── src/                    # Source code
│   ├── preprocessing.py    # CLAHE & Shadow enhancement logic
│   ├── slope_engine.py     # DTM to Slope conversion math
│   ├── detection.py        # Inference logic for YOLO & U-Net
│   └── utils.py            # Helper functions (Coordinate conversion)
├── notebooks/              # Research and experimentation
│   ├── 01_data_exploration.ipynb
│   └── 02_model_training.ipynb
├── assets/                 # Non-code files for README/UI
│   ├── architecture_diag.png
│   ├── demo_video.mp4
│   └── background_audio.mp3
├── app.py                  # Streamlit Dashboard
├── requirements.txt        # Dependencies
└── .gitignore              # Git ignore rules
```

## Installation

1. Clone the repository
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

Run the Streamlit dashboard:
```bash
streamlit run app.py
```

## Features

- Real-time hazard detection on lunar imagery
- Interactive visualization of detected regions
- Slope analysis and terrain classification
- Model confidence metrics

## Data

The project uses satellite imagery from Chandrayaan missions. Raw data should be placed in `data/raw/`.

## Models

Download pre-trained weights and place them in the `models/` directory:
- `boulder_yolo.pt` - YOLOv11 trained on boulder detection
- `landslide_unet.pth` - U-Net trained on landslide segmentation

## Contributors

Your team members here

## License

MIT License
