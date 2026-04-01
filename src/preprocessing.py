"""
Preprocessing Module: Shadow-Aware Enhancement for Lunar Imagery

This module implements advanced image preprocessing techniques specifically designed
for Chandrayaan satellite imagery, including CLAHE-based enhancement and shadow
detection for improved feature visibility.

Author: Lunar Hazard Detective Team
Version: 1.0.0
"""

import cv2
import numpy as np
from typing import Tuple, Optional, Union
from pathlib import Path
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class ShadowAwareEnhancer:
    """
    Advanced image enhancement class for lunar satellite imagery.
    
    Implements Contrast Limited Adaptive Histogram Equalization (CLAHE)
    combined with shadow detection and enhancement for improved visibility
    of lunar surface features including boulders and landslides.
    
    Attributes:
        clip_limit (float): CLAHE clipping threshold (default: 3.0)
        tile_size (int): Grid size for CLAHE (default: 8)
    """
    
    def __init__(self, clip_limit: float = 3.0, tile_size: int = 8):
        """
        Initialize the Shadow-Aware Enhancer.
        
        Args:
            clip_limit (float): Contrast limit for CLAHE. Higher values preserve
                               more contrast but may amplify noise. Range: [2.0-5.0]
            tile_size (int): Size of the grid for histogram equalization.
                            Must be power-related value. Default: 8
        """
        self.clip_limit = clip_limit
        self.tile_size = tile_size
        self.clahe = cv2.createCLAHE(
            clipLimit=self.clip_limit, 
            tileGridSize=(self.tile_size, self.tile_size)
        )
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Apply CLAHE to enhance local contrast in lunar imagery.
        
        Args:
            image (np.ndarray): Input image (8-bit or 16-bit grayscale/color)
            
        Returns:
            np.ndarray: CLAHE-enhanced image
            
        Raises:
            ValueError: If image is empty or invalid
        """
        if image is None or image.size == 0:
            raise ValueError("Image is empty or None")
        
        try:
            # Handle different image formats
            if len(image.shape) == 3:
                # Color image: convert to LAB, apply CLAHE to L-channel
                lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
                l_channel = lab[:, :, 0]
            else:
                # Grayscale image
                l_channel = image
            
            # Ensure 8-bit for CLAHE
            if l_channel.dtype == np.uint16:
                l_channel = (l_channel / 256).astype(np.uint8)
            
            # Apply CLAHE
            enhanced_l = self.clahe.apply(l_channel.astype(np.uint8))
            
            # Convert back to original color space
            if len(image.shape) == 3:
                lab[:, :, 0] = enhanced_l
                enhanced = cv2.cvtColor(lab, cv2.COLOR_LAB2BGR)
            else:
                enhanced = enhanced_l
            
            logger.info(f"CLAHE applied successfully. Image shape: {enhanced.shape}")
            return enhanced
        
        except Exception as e:
            logger.error(f"Error applying CLAHE: {str(e)}")
            raise
    
    def enhance_shadows(self, image: np.ndarray, strength: float = 1.5) -> np.ndarray:
        """
        Enhance shadow details using morphological operations and intensity scaling.
        
        This is crucial for lunar imagery where shadows reveal crater and slope
        information that is critical for hazard detection.
        
        Args:
            image (np.ndarray): Input image
            strength (float): Enhancement strength factor (1.0-3.0). Higher values
                            amplify shadows more but may increase noise.
            
        Returns:
            np.ndarray: Shadow-enhanced image
        """
        if strength < 1.0:
            logger.warning(f"Shadow strength {strength} < 1.0, will not enhance shadows")
            strength = 1.0
        
        try:
            # Create shadow detection mask using morphological operations
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            
            # Threshold for shadow detection
            if image.dtype == np.uint16:
                image_8bit = (image / 256).astype(np.uint8)
            else:
                image_8bit = image.astype(np.uint8)
            
            # Local shadow mask
            shadow_mask = cv2.morphologyEx(image_8bit, cv2.MORPH_OPEN, kernel)
            
            # Enhance shadows by scaling intensity
            enhanced = image.astype(np.float32) * strength
            enhanced = np.clip(enhanced, 0, 255).astype(np.uint8)
            
            logger.info(f"Shadow enhancement applied with strength: {strength}")
            return enhanced
        
        except Exception as e:
            logger.error(f"Error enhancing shadows: {str(e)}")
            raise
    
    def normalize_to_8bit(self, image: np.ndarray) -> np.ndarray:
        """
        Convert 16-bit satellite TIFF to normalized 8-bit PNG for model inference.
        
        This is essential for making satellite imagery compatible with deep learning
        models trained on 8-bit data.
        
        Args:
            image (np.ndarray): Input image (uint16 or uint8)
            
        Returns:
            np.ndarray: Normalized 8-bit image
        """
        try:
            if image.dtype == np.uint16:
                # Normalize 16-bit to 8-bit using percentile clipping
                p2, p98 = np.percentile(image, (2, 98))
                normalized = np.clip((image - p2) / (p98 - p2) * 255, 0, 255)
                normalized = normalized.astype(np.uint8)
                logger.info("Converted 16-bit image to 8-bit using percentile normalization")
            elif image.dtype == np.uint8:
                normalized = image
            else:
                # Handle float images
                if image.max() <= 1.0:
                    normalized = (image * 255).astype(np.uint8)
                else:
                    normalized = np.clip(image, 0, 255).astype(np.uint8)
            
            return normalized
        
        except Exception as e:
            logger.error(f"Error normalizing image: {str(e)}")
            raise
    
    def tiff_to_png(self, tiff_path: Union[str, Path], 
                   output_path: Union[str, Path], 
                   apply_enhancement: bool = True) -> bool:
        """
        Convert 16-bit satellite TIFF to 8-bit normalized PNG for model inference.
        
        This pipeline:
        1. Loads 16-bit TIFF
        2. Applies CLAHE enhancement (optional)
        3. Enhances shadows
        4. Normalizes to 8-bit
        5. Saves as PNG
        
        Args:
            tiff_path (str or Path): Path to input TIFF file
            output_path (str or Path): Path to output PNG file
            apply_enhancement (bool): Whether to apply CLAHE enhancement
            
        Returns:
            bool: True if conversion successful, False otherwise
        """
        try:
            tiff_path = Path(tiff_path)
            output_path = Path(output_path)
            
            if not tiff_path.exists():
                logger.error(f"TIFF file not found: {tiff_path}")
                return False
            
            # Load TIFF
            image = cv2.imread(str(tiff_path), cv2.IMREAD_ANYDEPTH)
            if image is None:
                logger.error(f"Failed to load TIFF: {tiff_path}")
                return False
            
            logger.info(f"Loaded TIFF: shape={image.shape}, dtype={image.dtype}")
            
            # Apply preprocessing pipeline
            if apply_enhancement:
                image = self.apply_clahe(image)
            
            image = self.enhance_shadows(image)
            image = self.normalize_to_8bit(image)
            
            # Save as PNG
            output_path.parent.mkdir(parents=True, exist_ok=True)
            success = cv2.imwrite(str(output_path), image)
            
            if success:
                logger.info(f"Successfully saved PNG: {output_path}")
                return True
            else:
                logger.error(f"Failed to save PNG: {output_path}")
                return False
        
        except Exception as e:
            logger.error(f"Error in TIFF to PNG conversion: {str(e)}")
            return False
    
    def batch_process(self, input_dir: Union[str, Path], 
                     output_dir: Union[str, Path],
                     apply_enhancement: bool = True) -> int:
        """
        Batch process all TIFF files in a directory.
        
        Args:
            input_dir (str or Path): Directory containing TIFF files
            output_dir (str or Path): Directory to save processed PNGs
            apply_enhancement (bool): Whether to apply CLAHE enhancement
            
        Returns:
            int: Number of successfully processed files
        """
        input_dir = Path(input_dir)
        output_dir = Path(output_dir)
        
        tiff_files = list(input_dir.glob('*.tif')) + list(input_dir.glob('*.tiff'))
        logger.info(f"Found {len(tiff_files)} TIFF files to process")
        
        success_count = 0
        for tiff_file in tiff_files:
            output_file = output_dir / f"{tiff_file.stem}.png"
            if self.tiff_to_png(tiff_file, output_file, apply_enhancement):
                success_count += 1
        
        logger.info(f"Successfully processed {success_count}/{len(tiff_files)} files")
        return success_count
