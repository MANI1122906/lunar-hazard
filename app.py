"""
Lunar Hazard Detective - Mission Control Dashboard

This is the main Streamlit application serving as the frontend for the Lunar Hazard
Detection System. It provides real-time detection visualization, analysis reports,
and system control for boulder and landslide identification.

Features:
- Dark theme with cyan neon aesthetic
- Real-time hazard detection and visualization
- Interactive file uploads and analysis
- Hazard analysis report generation
- Audio integration for system ambient sounds
- Mock mode for frontend testing

Author: Lunar Hazard Detective Team
Version: 1.0.0
"""

import streamlit as st
import numpy as np
import cv2
from PIL import Image
import sys
from pathlib import Path
from io import BytesIO
import random
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from preprocessing import ShadowAwareEnhancer
from detection import HazardDetector, BoulderDetection
from slope_engine import TerrainAnalyzer

# ============================================================================
# MOCK MODE CONFIGURATION
# ============================================================================
MOCK_MODE = True  # Set to False when models are ready
MOCK_BOULDER_COUNT = 8
MOCK_LANDSLIDE_PERCENTAGE = 15.5
MOCK_AVG_SLOPE = 22.3

# ============================================================================
# PAGE CONFIGURATION
# ============================================================================
st.set_page_config(
    page_title="Lunar Hazard Detective",
    page_icon="🌙",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ============================================================================
# CUSTOM CSS - Dark Theme with Cyan Neon Borders
# ============================================================================
st.markdown("""
    <style>
        /* Main background */
        .stApp {
            background-color: #0a0e27;
            color: #e0e0e0;
        }
        
        /* Remove default Streamlit styling */
        body {
            background-color: #0a0e27;
            font-family: 'Courier New', monospace;
        }
        
        /* Cyan neon borders */
        .neon-border {
            border: 2px solid #00ffff;
            border-radius: 8px;
            padding: 15px;
            background-color: #0f1535;
            box-shadow: 0 0 10px rgba(0, 255, 255, 0.3);
            margin: 10px 0;
        }
        
        .neon-danger {
            border: 2px solid #ff00ff;
            box-shadow: 0 0 10px rgba(255, 0, 255, 0.3);
        }
        
        .status-bar {
            background: linear-gradient(90deg, #0a0e27 0%, #1a1f3a 50%, #0a0e27 100%);
            border-top: 2px solid #00ffff;
            border-bottom: 2px solid #00ffff;
            padding: 10px 20px;
            margin-bottom: 20px;
        }
        
        .metric-card {
            background-color: #0f1535;
            border: 1px solid #00ffff;
            border-radius: 5px;
            padding: 15px;
            margin: 5px;
            box-shadow: inset 0 0 5px rgba(0, 255, 255, 0.1);
        }
        
        /* Dataframe styling */
        .dataframe {
            background-color: #0f1535 !important;
            color: #00ffff !important;
        }
        
        /* Button styling */
        .stButton > button {
            background-color: #00ffff;
            color: #0a0e27;
            font-weight: bold;
            border: 2px solid #00ffff;
            border-radius: 5px;
            padding: 10px 20px;
            transition: all 0.3s ease;
        }
        
        .stButton > button:hover {
            background-color: #0a0e27;
            color: #00ffff;
            box-shadow: 0 0 15px rgba(0, 255, 255, 0.6);
        }
        
        /* File uploader */
        .stFileUploader section {
            border: 2px dashed #00ffff;
            border-radius: 5px;
            background-color: #0f1535;
        }
        
        /* Headers */
        h1, h2, h3 {
            color: #00ffff;
            text-shadow: 0 0 10px rgba(0, 255, 255, 0.5);
        }
        
        /* Sidebar */
        .stSidebar {
            background-color: #0f1535;
            border-right: 2px solid #00ffff;
        }
        
        .stSidebar h1, .stSidebar h2, .stSidebar h3 {
            color: #00ffff;
        }
    </style>
""", unsafe_allow_html=True)

# ============================================================================
# SESSION STATE INITIALIZATION
# ============================================================================
if 'detection_results' not in st.session_state:
    st.session_state.detection_results = None
if 'uploaded_image' not in st.session_state:
    st.session_state.uploaded_image = None
if 'processing_time' not in st.session_state:
    st.session_state.processing_time = 0


# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def generate_mock_boulders(image: np.ndarray, count: int = MOCK_BOULDER_COUNT) -> list:
    """
    Generate realistic simulated boulder detections for demo mode.
    
    Args:
        image (np.ndarray): Input image
        count (int): Number of boulders to generate
        
    Returns:
        list: List of BoulderDetection objects
    """
    height, width = image.shape[:2]
    boulders = []
    
    for _ in range(count):
        # Random position
        x_min = random.randint(0, max(1, width - 100))
        y_min = random.randint(0, max(1, height - 100))
        
        # Random size (realistic: 5-50 pixels = 1.25-12.5 meters at 0.25m/px)
        size = random.randint(5, 50)
        x_max = min(x_min + size, width)
        y_max = min(y_min + size, height)
        
        # Random confidence
        confidence = random.uniform(0.65, 0.99)
        
        # Calculate diameter
        diameter_px = np.sqrt((x_max - x_min) * (y_max - y_min))
        diameter_m = diameter_px * 0.25  # 0.25m/pixel
        
        boulder = BoulderDetection(
            x_min=x_min, y_min=y_min,
            x_max=x_max, y_max=y_max,
            confidence=confidence,
            diameter_meters=diameter_m
        )
        boulders.append(boulder)
    
    return boulders


def generate_mock_landslide(image: np.ndarray, percentage: float = MOCK_LANDSLIDE_PERCENTAGE) -> np.ndarray:
    """
    Generate realistic simulated landslide segmentation for demo mode.
    
    Args:
        image (np.ndarray): Input image
        percentage (float): Percentage of image to mark as landslide
        
    Returns:
        np.ndarray: Binary segmentation mask
    """
    height, width = image.shape[:2]
    mask = np.zeros((height, width), dtype=np.uint8)
    
    # Create a few irregular regions
    num_regions = random.randint(2, 4)
    for _ in range(num_regions):
        # Random center
        cy = random.randint(50, height - 50)
        cx = random.randint(50, width - 50)
        
        # Random size
        radius = random.randint(30, 80)
        
        # Draw circle
        cv2.circle(mask, (cx, cy), radius, 255, -1)
        
        # Apply morphological operations for irregular shape
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
    
    # Ensure percentage matches
    current_percentage = np.sum(mask > 0) / mask.size * 100
    if current_percentage > percentage * 1.5:
        mask = (mask > 0).astype(np.uint8) * 255
        # Erode to reduce size
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        mask = cv2.erode(mask, kernel, iterations=2)
    
    return mask


def overlay_detections(image: np.ndarray, boulders: list, 
                      landslide_mask: np.ndarray = None,
                      confirmed_mask: np.ndarray = None) -> np.ndarray:
    """
    Create visualization with boulders and landslides overlaid on original image.
    
    Args:
        image (np.ndarray): Base image
        boulders (list): List of BoulderDetection objects
        landslide_mask (np.ndarray): Landslide segmentation mask
        confirmed_mask (np.ndarray): Confirmed landslide mask
        
    Returns:
        np.ndarray: Image with detections overlaid
    """
    # Convert to RGB if grayscale
    if len(image.shape) == 2:
        result = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
    else:
        result = image.copy()
    
    # Draw landslide regions
    if landslide_mask is not None:
        # Potential landslides in magenta
        landslide_contours = cv2.findContours((landslide_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(result, landslide_contours, -1, (255, 0, 255), 2)
    
    # Draw confirmed landslides
    if confirmed_mask is not None:
        confirmed_contours = cv2.findContours((confirmed_mask.astype(np.uint8) * 255), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[0]
        cv2.drawContours(result, confirmed_contours, -1, (0, 255, 255), 3)
    
    # Draw boulders in cyan
    for boulder in boulders:
        x_min, y_min = int(boulder.x_min), int(boulder.y_min)
        x_max, y_max = int(boulder.x_max), int(boulder.y_max)
        
        # Draw bounding box
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (0, 255, 255), 2)
        
        # Draw center point
        cx = (x_min + x_max) // 2
        cy = (y_min + y_max) // 2
        cv2.circle(result, (cx, cy), 3, (0, 255, 255), -1)
        
        # Draw size label
        label = f"{boulder.diameter_meters:.1f}m"
        cv2.putText(result, label, (x_min, y_min - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 1)
    
    return result


def create_status_bar(mock_mode: bool = False, models_loaded: bool = False):
    """Render system status bar"""
    status_text = "🟢 Online" if models_loaded or mock_mode else "🔴 Offline"
    mode_text = "[MOCK MODE]" if mock_mode else "[LIVE MODE]"
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    st.markdown(f"""
    <div class='status-bar'>
        <div style='display: flex; justify-content: space-between; align-items: center;'>
            <div style='color: #00ffff; font-weight: bold;'>
                {mode_text} {status_text}
            </div>
            <div style='color: #00ffff; opacity: 0.7;'>
                System Time: {timestamp}
            </div>
            <div style='color: #00ffff; font-size: 0.9em;'>
                GPU: {'CUDA Available' if True else 'CPU Mode'}
            </div>
        </div>
    </div>
    """, unsafe_allow_html=True)


# ============================================================================
# MAIN APPLICATION
# ============================================================================

# Header
col1, col2 = st.columns([1, 4])
with col1:
    st.markdown("<h1 style='font-size: 2.5em; margin: 0;'>🌙</h1>", unsafe_allow_html=True)
with col2:
    st.markdown("<h1 style='margin: 0;'>LUNAR HAZARD DETECTIVE</h1>", unsafe_allow_html=True)
    st.markdown("<p style='margin: 0; color: #00ffff; font-size: 0.9em;'>Neural Nexus | IIT Jammu | Mission Control</p>", 
               unsafe_allow_html=True)

st.divider()

# System Status Bar
create_status_bar(mock_mode=MOCK_MODE, models_loaded=True)

# ============================================================================
# SIDEBAR CONFIGURATION
# ============================================================================
with st.sidebar:
    st.markdown("<h2>⚙️ MISSION CONTROL</h2>", unsafe_allow_html=True)
    
    # Operating Mode
    st.markdown("### 🎯 Operating Mode")
    operating_mode = st.radio(
        "Select Analysis Mode",
        ["Upload Image", "Camera Input", "Batch Processing"],
        label_visibility="collapsed"
    )
    
    # Detection Parameters
    st.markdown("### 🔧 Detection Parameters")
    col1, col2 = st.columns(2)
    with col1:
        boulder_confidence = st.slider(
            "Boulder Confidence",
            0.0, 1.0, 0.65,
            help="Min confidence for boulder detection"
        )
    with col2:
        landslide_threshold = st.slider(
            "Landslide Sensitivity",
            0.0, 1.0, 0.5,
            help="Landslide detection threshold"
        )
    
    # Audio Control
    st.markdown("### 🔊 System Audio")
    audio_enabled = st.checkbox("Enable System Ambient", value=False)
    
    if audio_enabled:
        audio_path = Path(__file__).parent / "assets" / "background_audio.mp3"
        if audio_path.exists():
            with open(audio_path, 'rb') as f:
                audio_data = f.read()
            st.audio(audio_data, format="audio/mp3")
        else:
            st.info("🔇 Background audio not available")
    
    # System Information
    st.markdown("### 📊 System Info")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Mode", "MOCK" if MOCK_MODE else "LIVE")
    with col2:
        st.metric("Compute", "CPU/GPU")
    
    st.markdown("---")
    st.markdown("<p style='font-size: 0.8em; color: #00ffff;'>v1.0.0 | Lunar Hazard Detective</p>", 
               unsafe_allow_html=True)


# ============================================================================
# MAIN CONTENT AREA
# ============================================================================

if operating_mode == "Upload Image":
    st.markdown("<h3>📤 IMAGE UPLOAD & ANALYSIS</h3>", unsafe_allow_html=True)
    
    col1, col2 = st.columns([1, 2])
    
    with col1:
        st.markdown("#### Input Image")
        uploaded_file = st.file_uploader(
            "Choose an image file",
            type=["jpg", "jpeg", "png", "tif", "tiff"],
            label_visibility="collapsed"
        )
    
    with col2:
        st.markdown("#### Analysis Status")
        if uploaded_file is None:
            st.info("📂 Awaiting image upload...")
        else:
            st.success("✓ Image loaded and ready for analysis")
    
    if uploaded_file is not None:
        # Load and display original image
        image = Image.open(uploaded_file)
        image_array = np.array(image)
        
        # Ensure grayscale
        if len(image_array.shape) == 3:
            if image_array.shape[2] == 4:  # RGBA
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGBA2GRAY)
            else:  # RGB
                image_array = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        
        # Resize to reasonable size for demo
        max_dim = 512
        if max(image_array.shape) > max_dim:
            ratio = max_dim / max(image_array.shape)
            new_size = (int(image_array.shape[1] * ratio), int(image_array.shape[0] * ratio))
            image_array = cv2.resize(image_array, new_size)
        
        st.session_state.uploaded_image = image_array
        
        # Analysis Button
        col1, col2, col3 = st.columns([1, 1, 2])
        
        with col1:
            analyze_button = st.button("🚀 ANALYZE", use_container_width=True)
        
        with col2:
            reset_button = st.button("🔄 RESET", use_container_width=True)
        
        with col3:
            st.markdown("")
        
        if reset_button:
            st.session_state.detection_results = None
            st.rerun()
        
        if analyze_button:
            with st.spinner("🔬 Analyzing lunar surface..."):
                # Generate mock results
                if MOCK_MODE:
                    boulders = generate_mock_boulders(image_array)
                    landslide_mask = generate_mock_landslide(image_array)
                    
                    # Create simulation of slope data
                    slope_mask = np.random.uniform(10, 35, image_array.shape)
                    
                    st.session_state.detection_results = {
                        'boulders': boulders,
                        'landslides': landslide_mask,
                        'confirmed_landslides': (landslide_mask > 0) & (slope_mask > 20),
                        'average_slope': MOCK_AVG_SLOPE,
                        'analysis': {
                            'boulder_count': len(boulders),
                            'max_boulder_diameter_m': max([b.diameter_meters for b in boulders], default=0),
                            'avg_boulder_confidence': np.mean([b.confidence for b in boulders]) if boulders else 0,
                            'landslide_pixel_count': np.sum(landslide_mask > 0),
                            'confirmed_landslide_pixel_count': np.sum(((landslide_mask > 0) & (slope_mask > 20)).astype(int)),
                            'landslide_risk_percentage': MOCK_LANDSLIDE_PERCENTAGE
                        }
                    }
                
                st.success("✓ Analysis Complete!")
        
        # Results Display
        if st.session_state.detection_results:
            st.markdown("---")
            st.markdown("<h3>📋 DETECTION RESULTS</h3>", unsafe_allow_html=True)
            
            # Side-by-side Images
            result_col1, result_col2 = st.columns(2)
            
            with result_col1:
                st.markdown("<div class='neon-border'><h4 style='margin: 0;'>Original Image</h4></div>", 
                           unsafe_allow_html=True)
                st.image(image_array, use_column_width=True, channels="GRAY")
            
            with result_col2:
                st.markdown("<div class='neon-border'><h4 style='margin: 0;'>Detected Hazards</h4></div>", 
                           unsafe_allow_html=True)
                
                # Create overlay
                detected_image = overlay_detections(
                    image_array,
                    st.session_state.detection_results['boulders'],
                    st.session_state.detection_results['landslides'],
                    st.session_state.detection_results.get('confirmed_landslides')
                )
                st.image(detected_image, use_column_width=True, channels="BGR")
            
            # Hazard Analysis Report
            st.markdown("<h3>📊 HAZARD ANALYSIS REPORT</h3>", unsafe_allow_html=True)
            
            analysis = st.session_state.detection_results['analysis']
            
            # Key Metrics
            metric_cols = st.columns(4)
            
            with metric_cols[0]:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='color: #00ffff; opacity: 0.7; font-size: 0.9em;'>Boulder Count</div>
                    <div style='color: #00ffff; font-size: 1.8em; font-weight: bold;'>{analysis['boulder_count']}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[1]:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='color: #00ffff; opacity: 0.7; font-size: 0.9em;'>Max Diameter</div>
                    <div style='color: #00ffff; font-size: 1.8em; font-weight: bold;'>{analysis['max_boulder_diameter_m']:.1f}m</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[2]:
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='color: #00ffff; opacity: 0.7; font-size: 0.9em;'>Avg Confidence</div>
                    <div style='color: #00ffff; font-size: 1.8em; font-weight: bold;'>{analysis['avg_boulder_confidence']:.2%}</div>
                </div>
                """, unsafe_allow_html=True)
            
            with metric_cols[3]:
                risk_color = "#ff00ff" if analysis['landslide_risk_percentage'] > 20 else "#00ffff"
                st.markdown(f"""
                <div class='metric-card'>
                    <div style='color: {risk_color}; opacity: 0.7; font-size: 0.9em;'>Landslide Risk</div>
                    <div style='color: {risk_color}; font-size: 1.8em; font-weight: bold;'>{analysis['landslide_risk_percentage']:.1f}%</div>
                </div>
                """, unsafe_allow_html=True)
            
            # Detailed Report Table
            st.markdown("#### Detailed Metrics")
            
            report_data = {
                "Metric": [
                    "🪨 Boulder Count",
                    "📏 Max Boulder Diameter",
                    "📊 Avg Boulder Confidence",
                    "🔴 Detected Landslide Pixels",
                    "🟢 Confirmed Landslide Pixels",
                    "⚠️ Landslide Risk %",
                    "🏔️ Average Slope"
                ],
                "Value": [
                    f"{analysis['boulder_count']}",
                    f"{analysis['max_boulder_diameter_m']:.2f} m",
                    f"{analysis['avg_boulder_confidence']:.2%}",
                    f"{analysis['landslide_pixel_count']}",
                    f"{analysis['confirmed_landslide_pixel_count']}",
                    f"{analysis['landslide_risk_percentage']:.1f}%",
                    f"{st.session_state.detection_results['average_slope']:.2f}°"
                ]
            }
            
            st.dataframe(report_data, use_container_width=True, hide_index=True)
            
            # Export Results
            st.markdown("#### 💾 Export Options")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                # Convert image to bytes for download
                img_pil = Image.fromarray(detected_image)
                buf = BytesIO()
                img_pil.save(buf, format="PNG")
                btn = st.download_button(
                    label="📥 Download Detection Image",
                    data=buf.getvalue(),
                    file_name="detection_result.png",
                    mime="image/png",
                    use_container_width=True
                )
            
            with col2:
                st.markdown("<p style='text-align: center; color: #00ffff; opacity: 0.5;'>Report Export</p>", 
                           unsafe_allow_html=True)
            
            with col3:
                st.markdown("<p style='text-align: center; color: #00ffff; opacity: 0.5;'>Data Sharing</p>", 
                           unsafe_allow_html=True)

elif operating_mode == "Camera Input":
    st.markdown("<h3>📷 CAMERA INPUT (Not Yet Implemented)</h3>", unsafe_allow_html=True)
    st.info("📸 Camera input functionality will be available in a future update")
    st.markdown("Expected features: Real-time video capture and streaming analysis")

else:  # Batch Processing
    st.markdown("<h3>📁 BATCH PROCESSING (Not Yet Implemented)</h3>", unsafe_allow_html=True)
    st.info("🗂️ Batch processing functionality will be available in a future update")
    st.markdown("Expected features: Process multiple images and generate comprehensive reports")

# ============================================================================
# FOOTER
# ============================================================================
st.divider()
st.markdown("""
<div style='text-align: center; color: #00ffff; opacity: 0.5; font-size: 0.8em; padding: 20px;'>
    🚀 Lunar Hazard Detective v1.0.0 | Neural Nexus Lab | IIT Jammu <br>
    Powered by YOLOv11 & U-Net | Real-time Lunar Surface Analysis
</div>
""", unsafe_allow_html=True)
