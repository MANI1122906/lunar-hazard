🌕 Lunar Hazard Detective

AI-Powered Autonomous Hazard Detection for Lunar Landing Sites
Targeting safe landing zones using Chandrayaan Satellite Imagery

📁 GitHub: https://github.com/MANI1122906/lunar-hazard

⚡ One-Click Run (Windows) — RECOMMENDED

🚀 Fastest way to run the project (No setup needed)
✔ No manual installation required
✔ Fully automated execution
✔ Works out-of-the-box in MOCK mode

▶️ Steps
Download or extract the project ZIP
Double-click:
run_project.bat
✅ What happens automatically
Creates virtual environment
Installs dependencies
Launches Streamlit dashboard

👉 Open in browser:

http://localhost:8501
🌐 How to Use the Application

⏳ Note: The dashboard may take 15–20 seconds to load after opening. Please be patient.

▶️ Steps to Run Analysis
Open the app:
http://localhost:8501
Wait for the dashboard to load
Upload a sample image:
Use any .jpg image from the data/ folder
Click Analyze
View results:
Boulder detection
Landslide segmentation
Slope hazard analysis
✅ Final GO / NO-GO decision
🎯 Simple Flow
Run .bat → Open browser → Wait → Upload image → Click Analyze → View results
📊 Project Overview

Navigating the lunar south pole requires extreme precision.
Lunar Hazard Detective is an AI-powered pipeline that detects landing hazards using satellite imagery:

🪨 Boulders — Object Detection (YOLOv11)
🌊 Landslides — Semantic Segmentation (U-Net)
⛰️ Steep Slopes — DTM Analysis
🛠 Technical Implementation
🔹 Boulder Detection (YOLOv11)
Detects rocks and obstacles
Optimized for low-light lunar imagery (CLAHE)
🔹 Landslide Segmentation (U-Net)
Pixel-level terrain classification
Outputs Safe vs Unsafe zones
🔹 Slope Analysis Engine
Uses Digital Terrain Models (DTM)
Flags slopes greater than 15° as hazardous
📂 Project Structure
Lunar-Hazard-Detective/
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
├── run_project.bat          # One-click launcher for Windows
├── run_project.sh           # One-click launcher for Linux/Mac
└── requirements.txt         # Dependencies
💻 Manual Installation (All OS)

Use this if .bat doesn’t work or for Linux/Mac users

git clone https://github.com/MANI1122906/lunar-hazard.git
cd lunar-hazard

python -m venv .venv

# Windows
.\.venv\Scripts\activate

# Linux/Mac
source .venv/bin/activate

pip install -r requirements.txt

streamlit run app.py
🌐 Accessing the App
URL	Condition
http://localhost:8501
	Works while app is running
http://<your-ip>:8501	Works on same WiFi

⚠️ Keep terminal running or app will stop working

💡 Key Features
CLAHE image enhancement
Multi-model AI pipeline
Interactive Streamlit dashboard
GO / NO-GO landing decision
Works fully in MOCK mode
Supports large satellite image formats
⚠️ Model Weights Note

Model files are placeholders due to size limitations.
The app runs in MOCK MODE, which:

Simulates detection outputs
Generates realistic hazard maps
Provides full functionality without trained weights
👨‍💻 Developer

Mani Kumar
Team Lead & Developer

📄 License

MIT License

🔥 Why This Project Stands Out
✅ One-click execution (.bat)
✅ No setup required
✅ Fully working demo (even without weights)
✅ Clear instructions for evaluators
✅ Real-world space application
🚀 Next Improvements (Optional)
Add demo screenshots / GIF
Deploy on cloud (Streamlit Cloud / AWS)
Integrate real trained model weights
