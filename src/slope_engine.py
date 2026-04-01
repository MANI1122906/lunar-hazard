"""
Slope Engine: Terrain Analysis and Hazard Zone Identification

This module implements Digital Terrain Model (DTM) processing and slope analysis
to identify terrain characteristics critical for validating landslide detections.
The core logic filters detections to only 'Confirmed Landslide' status when slope > 20°.

Author: Lunar Hazard Detective Team
Version: 1.0.0
"""

import numpy as np
from typing import Tuple, Optional, Union
from scipy import ndimage
from pathlib import Path
import logging

try:
    import rasterio
    RASTERIO_AVAILABLE = True
except ImportError:
    RASTERIO_AVAILABLE = False

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TerrainAnalyzer:
    """
    Comprehensive terrain analysis engine for lunar DTM data.
    
    This class provides methods to:
    - Calculate terrain slopes from Digital Terrain Models
    - Identify hazardous terrain zones
    - Cross-reference slope data with detection results
    - Generate confirmation masks for landslide validation
    
    Attributes:
        slope_threshold (float): Minimum slope angle (degrees) to identify hazards
        pixel_size (float): Size of each pixel in meters (default: 0.25m for Chandrayaan)
    """
    
    def __init__(self, slope_threshold: float = 20.0, pixel_size: float = 0.25):
        """
        Initialize the Terrain Analyzer.
        
        Args:
            slope_threshold (float): Minimum slope (degrees) for hazard zone. 
                                    Landslides typically occur on slopes > 20°.
            pixel_size (float): Pixel size in meters. Chandrayaan imagery typically
                              has 0.25 m/pixel resolution.
        """
        self.slope_threshold = slope_threshold
        self.pixel_size = pixel_size
        logger.info(f"TerrainAnalyzer initialized with slope_threshold={slope_threshold}°, "
                   f"pixel_size={pixel_size}m/px")
    
    def calculate_slope(self, dtm: np.ndarray) -> np.ndarray:
        """
        Calculate terrain slope from Digital Terrain Model using Sobel derivatives.
        
        The slope is calculated as: slope = arctan(sqrt(dz/dx² + dz/dy²))
        
        Args:
            dtm (np.ndarray): Digital Terrain Model with elevation data (shape: H×W)
            
        Returns:
            np.ndarray: Slope map in degrees (same shape as DTM)
            
        Raises:
            ValueError: If DTM is empty or invalid
        """
        if dtm is None or dtm.size == 0:
            raise ValueError("DTM is empty or None")
        
        try:
            # Calculate gradients using Sobel operator
            gradient_x = ndimage.sobel(dtm.astype(float), axis=1) / (8 * self.pixel_size)
            gradient_y = ndimage.sobel(dtm.astype(float), axis=0) / (8 * self.pixel_size)
            
            # Calculate slope magnitude
            slope_magnitude = np.sqrt(gradient_x**2 + gradient_y**2)
            slope_radians = np.arctan(slope_magnitude)
            slope_degrees = np.degrees(slope_radians)
            
            logger.info(f"Slope calculated. Min: {slope_degrees.min():.2f}°, "
                       f"Max: {slope_degrees.max():.2f}°, Mean: {slope_degrees.mean():.2f}°")
            
            return slope_degrees
        
        except Exception as e:
            logger.error(f"Error calculating slope: {str(e)}")
            raise
    
    def calculate_aspect(self, dtm: np.ndarray) -> np.ndarray:
        """
        Calculate terrain aspect (direction of steepest slope).
        
        Aspect represents the compass direction of the slope face, useful for
        understanding solar exposure and potential hazard zones.
        
        Args:
            dtm (np.ndarray): Digital Terrain Model
            
        Returns:
            np.ndarray: Aspect map in degrees (0-360, where 0° = North)
        """
        try:
            gradient_x = ndimage.sobel(dtm.astype(float), axis=1)
            gradient_y = ndimage.sobel(dtm.astype(float), axis=0)
            
            # Calculate aspect using atan2
            aspect = np.degrees(np.arctan2(-gradient_y, gradient_x)) + 180
            aspect = aspect % 360
            
            logger.info(f"Aspect calculated. Ranging from 0-360°")
            return aspect
        
        except Exception as e:
            logger.error(f"Error calculating aspect: {str(e)}")
            raise
    
    def generate_hazard_mask(self, slope: np.ndarray, 
                           aspect: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Generate binary hazard mask where slope > 20 degrees.
        
        This mask identifies terrain regions susceptible to landslides based on
        geophysical principles (slopes > 20° are highly unstable on the Moon).
        
        Args:
            slope (np.ndarray): Slope map in degrees
            aspect (np.ndarray, optional): Aspect map for additional filtering
            
        Returns:
            np.ndarray: Binary mask (1 = hazardous, 0 = safe)
        """
        try:
            # Primary criterion: slope > threshold
            hazard_mask = (slope > self.slope_threshold).astype(np.uint8)
            
            # Optional: Add aspect-based filtering (south-facing slopes are more exposed)
            if aspect is not None:
                # Sun-facing slopes (aspect 90-270) are more prone to thermal instability
                south_facing = (aspect > 90) & (aspect < 270)
                hazard_mask = hazard_mask | south_facing.astype(np.uint8)
            
            hazard_pixels = np.sum(hazard_mask)
            total_pixels = hazard_mask.size
            hazard_percentage = 100 * hazard_pixels / total_pixels
            
            logger.info(f"Hazard mask generated: {hazard_pixels} pixels ({hazard_percentage:.1f}%) "
                       f"classified as hazardous (slope > {self.slope_threshold}°)")
            
            return hazard_mask
        
        except Exception as e:
            logger.error(f"Error generating hazard mask: {str(e)}")
            raise
    
    def validate_landslide_region(self, segmentation_mask: np.ndarray,
                                 slope: np.ndarray) -> Tuple[np.ndarray, float]:
        """
        Validate detected landslide region against slope criteria.
        
        Only regions with average slope > 20° are confirmed as landslides.
        This is the core 'Fusion Logic' that combines U-Net output with terrain analysis.
        
        Args:
            segmentation_mask (np.ndarray): Binary U-Net landslide segmentation
            slope (np.ndarray): Slope map in degrees
            
        Returns:
            Tuple[np.ndarray, float]: 
                - confirmed_mask: Binary mask of confirmed landslides
                - average_slope: Average slope in detected region
        """
        try:
            if np.sum(segmentation_mask) == 0:
                logger.warning("No landslide regions detected in segmentation mask")
                return segmentation_mask, 0.0
            
            # Calculate average slope in detected region
            detected_slopes = slope[segmentation_mask > 0]
            average_slope = float(np.mean(detected_slopes))
            
            # Create confirmed mask: only where slope > threshold
            hazard_mask = self.generate_hazard_mask(slope)
            confirmed_mask = (segmentation_mask > 0) & (hazard_mask > 0)
            confirmed_mask = confirmed_mask.astype(np.uint8)
            
            confirmation_rate = np.sum(confirmed_mask) / np.sum(segmentation_mask) * 100
            
            logger.info(f"Landslide validation complete. Average slope in region: {average_slope:.2f}°, "
                       f"Confirmation rate: {confirmation_rate:.1f}%")
            
            return confirmed_mask, average_slope
        
        except Exception as e:
            logger.error(f"Error validating landslide region: {str(e)}")
            raise
    
    def smooth_dtm(self, dtm: np.ndarray, kernel_size: int = 5) -> np.ndarray:
        """
        Apply Gaussian smoothing to DTM to reduce noise.
        
        Args:
            dtm (np.ndarray): Digital Terrain Model
            kernel_size (int): Size of smoothing kernel (must be odd)
            
        Returns:
            np.ndarray: Smoothed DTM
        """
        try:
            if kernel_size % 2 == 0:
                kernel_size += 1
            
            sigma = kernel_size / 6.0
            smoothed = ndimage.gaussian_filter(dtm.astype(float), sigma=sigma)
            
            logger.info(f"DTM smoothed with kernel_size={kernel_size}")
            return smoothed
        
        except Exception as e:
            logger.error(f"Error smoothing DTM: {str(e)}")
            raise
    
    def calculate_curvature(self, dtm: np.ndarray) -> np.ndarray:
        """
        Calculate terrain curvature for feature detection.
        
        Curvature highlights concave and convex features useful for identifying
        craters and valleys.
        
        Args:
            dtm (np.ndarray): Digital Terrain Model
            
        Returns:
            np.ndarray: Curvature map
        """
        try:
            # Calculate second derivatives
            dxx = ndimage.sobel(ndimage.sobel(dtm.astype(float), axis=1), axis=1)
            dyy = ndimage.sobel(ndimage.sobel(dtm.astype(float), axis=0), axis=0)
            
            curvature_map = dxx + dyy
            
            logger.info(f"Curvature calculated. Range: [{curvature_map.min():.2f}, "
                       f"{curvature_map.max():.2f}]")
            return curvature_map
        
        except Exception as e:
            logger.error(f"Error calculating curvature: {str(e)}")
            raise
    
    def load_dtm_from_geotiff(self, filepath: Union[str, Path]) -> Optional[np.ndarray]:
        """
        Load Digital Terrain Model from GeoTIFF file using rasterio.
        
        Args:
            filepath (str or Path): Path to GeoTIFF DTM file
            
        Returns:
            np.ndarray: Elevation data, or None if loading fails
        """
        if not RASTERIO_AVAILABLE:
            logger.warning("Rasterio not available. Cannot load GeoTIFF.")
            return None
        
        try:
            filepath = Path(filepath)
            with rasterio.open(filepath) as src:
                dtm = src.read(1)  # Read first band
                logger.info(f"Loaded DTM from {filepath}: shape={dtm.shape}, dtype={dtm.dtype}")
                return dtm
        
        except Exception as e:
            logger.error(f"Error loading GeoTIFF: {str(e)}")
            return None
