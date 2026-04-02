🌕 Lunar-Hazard-Detective
AI-Powered Autonomous Hazard Detection for Lunar Landing Sites
Targeting safe landing zones using Chandrayaan Satellite Imagery

📊 Project Overview
Navigating the lunar south pole requires extreme precision. Lunar-Hazard-Detective is an end-to-end vision pipeline that processes high-resolution satellite data to identify landing risks. It specifically targets Boulders (Object Detection), Landslides (Semantic Segmentation), and Steep Slopes (DTM Analysis) to ensure mission success.

🛠 Technical Implementation (Core Components)
1. Boulder Detection (YOLOv11)
Model: YOLOv11 (Small/Medium variants for speed-accuracy balance).

Function: Real-time identification of discrete obstacles.

Precision: Optimized for low-light lunar conditions using shadow-enhancement preprocessing.

2. Landslide Segmentation (U-Net)
Model: U-Net Architecture with ResNet-34 Backbone.

Function: Pixel-level classification of unstable terrain and regolith displacement.

Output: Binary masks indicating "Safe" vs "Unsafe" zones.

3. Slope Analysis Engine
Input: Digital Terrain Models (DTM).

Logic: Calculates surface gradients using moving-window algorithms.

Constraint: Automatically flags any region with an inclination > 15° as a high-hazard zone.

📂 Repository Structure
Plaintext
Lunar-Hazard-Detective/
├── assets/                  # System architecture & demo visuals
├── data/                    
│   ├── raw/                 # .tif / .hdf5 Chandrayaan imagery
│   ├── processed/           # CLAHE & Shadow-corrected images
│   └── masks/               # Ground truth segmentation labels
├── models/                  
│   ├── boulder_yolo.pt      # Weights for YOLOv11
│   └── landslide_unet.pth   # Weights for U-Net
├── src/                     
│   ├── preprocessing.py     # Image enhancement & CLAHE logic
│   ├── slope_engine.py      # DTM to Slope conversion math
│   ├── detection.py         # Inference logic (YOLO & U-Net)
│   └── utils.py             # Coordinate & Geospatial helpers
├── app.py                   # Streamlit Dashboard (GUI)
└── requirements.txt         # Dependencies
🚀 Getting Started
Installation
Clone & Navigate:

Bash
git clone https://github.com/your-username/Lunar-Hazard-Detective.git
cd Lunar-Hazard-Detective
Setup Environment:

Bash
python -m venv .venv
source .venv/bin/activate  # Linux/Mac
.\.venv\Scripts\Activate   # Windows
Install Requirements:

Bash
pip install -r requirements.txt
Running the App
Bash
streamlit run app.py
💡 Key Features for Evaluation
Advanced Preprocessing: Uses CLAHE (Contrast Limited Adaptive Histogram Equalization) to reveal details in deep lunar shadows.

Multi-Modal Analysis: Combines standard imagery with DTM (Digital Terrain Model) data for 3D terrain understanding.

Interactive Dashboard: Users can upload .tif files and get instant hazard maps with confidence scores.

Mission Ready: Logic is built to handle the large-scale file formats used by ISRO/NASA.

⚠️ Important Note for Developers
When handling masks for segmentation, ensure they are cast to uint8 before using OpenCV functions to avoid data type mismatches:

Python
# Vital for cv2.findContours compatibility
processed_mask = (prediction_mask > 0.5).astype(np.uint8) * 255
👥 Team
Mani Kumar – Lead Developer
📄 License
This project is licensed under the MIT License.
