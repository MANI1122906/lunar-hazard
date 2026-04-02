# 🌕 Lunar Hazard Detective
**AI-Powered Autonomous Hazard Detection for Lunar Landing Sites**  
Targeting safe landing zones using Chandrayaan Satellite Imagery

📁 **GitHub:** [MANI1122906/lunar-hazard](https://github.com/MANI1122906/lunar-hazard)

---

## 📊 Project Overview
Navigating the lunar south pole requires extreme precision. Lunar-Hazard-Detective is an end-to-end vision pipeline that processes high-resolution satellite data to identify landing risks:

- 🪨 **Boulders** — Object Detection (YOLOv11)
- 🌊 **Landslides** — Semantic Segmentation (U-Net)
- ⛰️ **Steep Slopes** — DTM Analysis

---

## 🛠 Technical Implementation

### 1. Boulder Detection (YOLOv11)
- **Model:** YOLOv11 (Small/Medium variants for speed-accuracy balance)
- **Function:** Real-time identification of discrete obstacles
- **Precision:** Optimized for low-light lunar conditions using CLAHE preprocessing

### 2. Landslide Segmentation (U-Net)
- **Model:** U-Net Architecture with ResNet-34 Backbone
- **Function:** Pixel-level classification of unstable terrain and regolith displacement
- **Output:** Binary masks indicating Safe vs Unsafe zones

### 3. Slope Analysis Engine
- **Input:** Digital Terrain Models (DTM)
- **Logic:** Calculates surface gradients using moving-window algorithms
- **Constraint:** Flags any region with inclination > 15° as high-hazard

---

## 📂 Repository Structure
```
Lunar-Hazard-Detective/
├── assets/                  # Architecture & demo visuals
├── data/
│   ├── raw/                 # .tif / .hdf5 Chandrayaan imagery
│   ├── processed/           # CLAHE & shadow-corrected images
│   └── masks/               # Ground truth segmentation labels
├── models/
│   ├── boulder_yolo.pt      # YOLOv11 weights (placeholder)
│   └── landslide_unet.pth   # U-Net weights (placeholder)
├── src/
│   ├── preprocessing.py     # CLAHE & image enhancement
│   ├── slope_engine.py      # DTM slope conversion
│   ├── detection.py         # YOLO & U-Net inference
│   └── utils.py             # Geospatial helpers
├── app.py                   # Streamlit Dashboard
└── requirements.txt         # Dependencies
```

---

## ⚠️ Note on Model Weights
The model files (`boulder_yolo.pt` and `landslide_unet.pth`) in this repo are **placeholders** — real trained weights are not included due to file size limits.

**But don't worry — the app runs perfectly without them! ✅**

The system runs in **MOCK mode** by default which:
- Simulates boulder detection with realistic confidence scores
- Generates landslide risk maps and hazard analysis
- Produces full GO / NO-GO verdicts
- Shows the complete dashboard with all features working

No model files needed to run or demo this project.

---

## 🚀 Getting Started

### Installation
```bash
# Clone the repo
git clone https://github.com/MANI1122906/lunar-hazard.git
cd lunar-hazard

# Setup environment
python -m venv .venv
.\.venv\Scripts\Activate     # Windows
source .venv/bin/activate    # Linux/Mac

# Install dependencies
pip install -r requirements.txt
```

### Run the App
```bash
streamlit run app.py
```

---

## 🌐 Accessing the App

| URL | When it works |
|-----|---------------|
| `http://localhost:8501` | ✅ Works **only while** `streamlit run app.py` is running in terminal |
| `http://192.168.1.102:8501` | ✅ Works on same WiFi network while terminal is running |

> ⚠️ **Important:** The URL will **NOT open** if the terminal is stopped or closed.  
> Always keep the terminal running while using the app.

### Simple Rule
```
Terminal running ✅  →  http://localhost:8501 works
Terminal stopped ❌  →  URL won't open
```

> ⏳ **Note:** First load takes 10-15 seconds — this is normal for Streamlit startup.

---

## 💡 Key Features

| Feature | Description |
|---------|-------------|
| **CLAHE Preprocessing** | Reveals details hidden in deep lunar shadows |
| **Multi-Modal Analysis** | Combines imagery + DTM for 3D terrain understanding |
| **Interactive Dashboard** | Upload `.tif` files, get instant hazard maps |
| **GO / NO-GO Verdict** | Clear binary safety decision with confidence score |
| **MOCK Mode** | Fully functional without real model weights |
| **Mission Ready** | Handles large-scale ISRO/NASA file formats |

---

## ⚠️ Developer Note
When handling segmentation masks, cast to `uint8` before using OpenCV:
```python
# Required for cv2.findContours compatibility
processed_mask = (prediction_mask > 0.5).astype(np.uint8) * 255
```

---

## 👥 Team
**Mani Mani Kumar** — Team Leader & Lead Developer

---

## 📄 License
This project is licensed under the **MIT License**.
