Lunar Hazard Detective

AI-Powered Autonomous Hazard Detection for Lunar Landing Sites
Targeting safe landing zones using Chandrayaan Satellite Imagery

📁 GitHub: https://github.com/MANI1122906/lunar-hazard

⚡ One-Click Run (Windows) — RECOMMENDED

🚀 Fastest way to run the project (No setup needed)

▶️ Steps
Download or extract the project ZIP
Double-click:
run_project.bat
✅ What happens automatically
Virtual environment setup
Dependency installation
Streamlit app launch

👉 Open in browser:

http://localhost:8501
📊 Project Overview

Navigating the lunar south pole requires extreme precision.
Lunar Hazard Detective is an end-to-end AI pipeline that detects landing risks using satellite imagery:

🪨 Boulders — Object Detection (YOLOv11)
🌊 Landslides — Semantic Segmentation (U-Net)
⛰️ Steep Slopes — DTM Analysis
🛠 Technical Implementation
🔹 Boulder Detection (YOLOv11)
Detects rocks and obstacles
Optimized for lunar low-light using CLAHE
🔹 Landslide Segmentation (U-Net)
Pixel-level classification
Outputs Safe vs Unsafe terrain
🔹 Slope Analysis Engine
Uses Digital Terrain Models (DTM)
Flags slopes > 15° as hazardous
📂 Project Structure
Lunar-Hazard-Detective/
├── assets/
├── data/
├── models/
├── src/
├── app.py
├── run_project.bat
└── requirements.txt
💻 Manual Installation (All OS)

Use this if .bat doesn't work or for Linux/Mac users

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

⚠️ Keep terminal running or app will stop

💡 Key Features
CLAHE image enhancement
Multi-model AI pipeline
Interactive Streamlit dashboard
GO / NO-GO landing decision
Works in MOCK mode (no weights needed)
⚠️ Model Weights Note

Model files are placeholders due to size limits.
The app runs in MOCK MODE with:

Simulated detections
Realistic outputs
Fully working UI
👨‍💻 Developer

Mani Kumar
Team Lead & Developer

📄 License

MIT License
