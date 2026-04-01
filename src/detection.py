"""
Detection Module: Hazard Detection and Fusion Logic

This module integrates YOLOv11 for boulder detection and U-Net for landslide
segmentation, with advanced fusion logic that cross-references slope data for
confirmed hazard identification.

Author: Lunar Hazard Detective Team
Version: 1.0.0
"""

import numpy as np
import torch
import cv2
from typing import Tuple, List, Dict, Optional, Union
from pathlib import Path
from dataclasses import dataclass
import logging

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


@dataclass
class BoulderDetection:
    """Data class for boulder detection results"""
    x_min: float
    y_min: float
    x_max: float
    y_max: float
    confidence: float
    diameter_meters: float
    
    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'x_min': self.x_min,
            'y_min': self.y_min,
            'x_max': self.x_max,
            'y_max': self.y_max,
            'confidence': self.confidence,
            'diameter_meters': self.diameter_meters
        }


class BoulderDetector:
    """
    YOLOv11-based boulder detection with size filtering.
    
    This detector identifies boulders on the lunar surface using deep learning
    and filters detections by physical size to reduce false positives.
    
    Attributes:
        model_path (Path): Path to YOLOv11 model weights
        confidence_threshold (float): Minimum detection confidence (0-1)
        pixel_size_m (float): Pixel size in meters for size conversion
    """
    
    def __init__(self, model_path: Union[str, Path], 
                 confidence_threshold: float = 0.5,
                 pixel_size_m: float = 0.25):
        """
        Initialize boulder detector.
        
        Args:
            model_path (str or Path): Path to YOLOv11 .pt model file
            confidence_threshold (float): Minimum confidence score (0.1-0.9 recommended)
            pixel_size_m (float): Conversion factor: 0.25m/pixel for Chandrayaan
        """
        self.model_path = Path(model_path) if model_path else None
        self.confidence_threshold = confidence_threshold
        self.pixel_size_m = pixel_size_m
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"BoulderDetector initialized on {self.device}")
    
    def load_model(self) -> bool:
        """
        Load YOLOv11 model from disk.
        
        Returns:
            bool: True if model loaded successfully
        """
        if not ULTRALYTICS_AVAILABLE:
            logger.warning("Ultralytics package not available")
            return False
        
        if not self.model_path:
            logger.warning("No model path provided")
            return False
        
        try:
            if not self.model_path.exists():
                logger.error(f"Model file not found: {self.model_path}")
                return False
            
            self.model = YOLO(str(self.model_path))
            logger.info(f"YOLOv11 model loaded successfully from {self.model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading YOLO model: {str(e)}")
            return False
    
    def _pixel_to_meters(self, pixel_distance: float) -> float:
        """
        Convert pixel distance to meters.
        
        Args:
            pixel_distance (float): Distance in pixels
            
        Returns:
            float: Distance in meters
        """
        return pixel_distance * self.pixel_size_m
    
    def _filter_by_size(self, bbox: Tuple[float, float, float, float],
                       min_size_m: float = 0.5,
                       max_size_m: float = 100.0) -> Tuple[bool, float]:
        """
        Filter bounding box by physical size to remove noise.
        
        Args:
            bbox: Bounding box (x_min, y_min, x_max, y_max)
            min_size_m: Minimum boulder size in meters
            max_size_m: Maximum boulder size in meters
            
        Returns:
            Tuple[bool, float]: (passes_filter, diameter_m)
        """
        x_min, y_min, x_max, y_max = bbox
        width_px = x_max - x_min
        height_px = y_max - y_min
        diameter_px = np.sqrt(width_px * height_px)  # Approximate diameter
        diameter_m = self._pixel_to_meters(diameter_px)
        
        passes_filter = min_size_m <= diameter_m <= max_size_m
        return passes_filter, diameter_m
    
    def detect(self, image: np.ndarray) -> List[BoulderDetection]:
        """
        Detect boulders in lunar image.
        
        Args:
            image (np.ndarray): Input image (8-bit grayscale preferred)
            
        Returns:
            List[BoulderDetection]: List of detected boulder objects
        """
        if self.model is None:
            if not self.load_model():
                logger.warning("Using empty detections")
                return []
        
        detections = []
        
        try:
            # Run YOLO inference
            results = self.model(image, conf=self.confidence_threshold, verbose=False)
            
            if len(results) > 0:
                result = results[0]  # First image
                boxes = result.boxes
                
                for box in boxes:
                    x_min, y_min, x_max, y_max = box.xyxy[0].cpu().numpy()
                    confidence = float(box.conf)
                    
                    # Apply size filter
                    passes_filter, diameter_m = self._filter_by_size(
                        (x_min, y_min, x_max, y_max)
                    )
                    
                    if passes_filter:
                        detection = BoulderDetection(
                            x_min=float(x_min),
                            y_min=float(y_min),
                            x_max=float(x_max),
                            y_max=float(y_max),
                            confidence=confidence,
                            diameter_meters=diameter_m
                        )
                        detections.append(detection)
            
            logger.info(f"Detected {len(detections)} boulders")
            return detections
        
        except Exception as e:
            logger.error(f"Error during boulder detection: {str(e)}")
            return []


class LandslideDetector:
    """
    U-Net based landslide segmentation.
    
    This detector identifies landslide-prone regions using semantic segmentation
    and cross-references results with terrain slope data.
    
    Attributes:
        model_path (Path): Path to trained U-Net weights
        device (torch.device): Computation device (CPU/GPU)
    """
    
    def __init__(self, model_path: Optional[Union[str, Path]] = None):
        """
        Initialize landslide detector.
        
        Args:
            model_path (str or Path): Path to U-Net model weights (.pth file)
        """
        self.model_path = Path(model_path) if model_path else None
        self.model = None
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        logger.info(f"LandslideDetector initialized on {self.device}")
    
    def load_model(self, model=None) -> bool:
        """
        Load trained U-Net model.
        
        Args:
            model: Pre-initialized model (optional for testing)
            
        Returns:
            bool: True if model loaded successfully
        """
        try:
            if model is not None:
                self.model = model.to(self.device)
                logger.info("Model provided directly")
                return True
            
            if not self.model_path or not self.model_path.exists():
                logger.warning(f"Model file not found: {self.model_path}")
                return False
            
            checkpoint = torch.load(self.model_path, map_location=self.device)
            # Model initialization depends on architecture definition
            logger.info(f"Model checkpoint loaded from {self.model_path}")
            return True
        
        except Exception as e:
            logger.error(f"Error loading U-Net model: {str(e)}")
            return False
    
    def segment(self, image: np.ndarray) -> np.ndarray:
        """
        Segment landslide regions in lunar image.
        
        Args:
            image (np.ndarray): Input image (H × W, 8-bit)
            
        Returns:
            np.ndarray: Binary segmentation mask (0-1 or 0-255)
        """
        if self.model is None:
            logger.warning("Model not loaded, returning empty mask")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)
        
        try:
            # Preprocess image for model input
            img_tensor = torch.from_numpy(image).float().unsqueeze(0).unsqueeze(0)
            img_tensor = img_tensor.to(self.device)
            
            # Run inference
            with torch.no_grad():
                output = self.model(img_tensor)
            
            # Post-process output
            segmentation = torch.sigmoid(output).squeeze().cpu().numpy()
            segmentation = (segmentation > 0.5).astype(np.uint8) * 255
            
            logger.info(f"Segmentation complete. Detected area: {np.sum(segmentation > 0)} pixels")
            return segmentation
        
        except Exception as e:
            logger.error(f"Error during landslide segmentation: {str(e)}")
            return np.zeros((image.shape[0], image.shape[1]), dtype=np.uint8)


class HazardDetector:
    """
    Integrated hazard detection system combining YOLO and U-Net with fusion logic.
    
    This is the core detection engine that:
    1. Detects boulders using YOLOv11
    2. Segments landslides using U-Net
    3. Validates landslides against slope data
    4. Generates combined hazard analysis report
    """
    
    def __init__(self, 
                 boulder_model_path: Optional[Union[str, Path]] = None,
                 landslide_model_path: Optional[Union[str, Path]] = None):
        """
        Initialize integrated hazard detector.
        
        Args:
            boulder_model_path: Path to YOLO model
            landslide_model_path: Path to U-Net model
        """
        self.boulder_detector = BoulderDetector(boulder_model_path) if boulder_model_path else None
        self.landslide_detector = LandslideDetector(landslide_model_path) if landslide_model_path else None
        logger.info("HazardDetector initialized")
    
    def detect_hazards(self, image: np.ndarray,
                      slope_mask: Optional[np.ndarray] = None) -> Dict:
        """
        Comprehensive hazard detection on lunar image.
        
        Args:
            image (np.ndarray): Input lunar image
            slope_mask (np.ndarray, optional): Pre-computed slope hazard mask
            
        Returns:
            Dict: Detection results with boulders, landslides, and combined analysis
        """
        results = {
            'boulders': [],
            'landslides': None,
            'confirmed_landslides': None,
            'average_slope': 0.0,
            'analysis': {}
        }
        
        try:
            # Detect boulders
            if self.boulder_detector:
                results['boulders'] = self.boulder_detector.detect(image)
            
            # Segment landslides
            if self.landslide_detector:
                results['landslides'] = self.landslide_detector.segment(image)
                
                # Cross-reference with slope if available
                if slope_mask is not None and results['landslides'] is not None:
                    # Calculate average slope in detected region
                    detected_idx = results['landslides'] > 0
                    if np.sum(detected_idx) > 0:
                        avg_slope = float(np.mean(slope_mask[detected_idx]))
                        results['average_slope'] = avg_slope
                        
                        # Confirm landslide only if slope > 20°
                        confirmed_idx = detected_idx & (slope_mask > 20)
                        results['confirmed_landslides'] = confirmed_idx.astype(np.uint8) * 255
            
            # Generate analysis report
            results['analysis'] = self._generate_analysis(results)
            
            logger.info(f"Hazard detection complete: {len(results['boulders'])} boulders, "
                       f"avg slope: {results['average_slope']:.2f}°")
            
            return results
        
        except Exception as e:
            logger.error(f"Error during hazard detection: {str(e)}")
            return results
    
    def _generate_analysis(self, results: Dict) -> Dict:
        """
        Generate analytical summary of detected hazards.
        
        Args:
            results (Dict): Detection results
            
        Returns:
            Dict: Analysis summary
        """
        analysis = {
            'boulder_count': len(results['boulders']),
            'max_boulder_diameter_m': 0.0,
            'avg_boulder_confidence': 0.0,
            'landslide_pixel_count': 0,
            'confirmed_landslide_pixel_count': 0,
            'landslide_risk_percentage': 0.0
        }
        
        # Boulder analysis
        if results['boulders']:
            diameters = [b.diameter_meters for b in results['boulders']]
            confidences = [b.confidence for b in results['boulders']]
            analysis['max_boulder_diameter_m'] = float(np.max(diameters))
            analysis['avg_boulder_confidence'] = float(np.mean(confidences))
        
        # Landslide analysis
        if results['landslides'] is not None:
            total_pixels = results['landslides'].size
            landslide_pixels = np.sum(results['landslides'] > 0)
            analysis['landslide_pixel_count'] = int(landslide_pixels)
            analysis['landslide_risk_percentage'] = 100 * landslide_pixels / total_pixels
        
        if results['confirmed_landslides'] is not None:
            confirmed_pixels = np.sum(results['confirmed_landslides'] > 0)
            analysis['confirmed_landslide_pixel_count'] = int(confirmed_pixels)
        
        return analysis
