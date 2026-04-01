"""
Utility Functions: Helper Functions & Coordinate Conversion
Provides common utilities for the project.
"""

import numpy as np
from typing import Tuple, Optional


def lon_lat_to_pixel(longitude: float, latitude: float,
                     image_shape: Tuple[int, int],
                     bounds: Tuple[float, float, float, float]) -> Tuple[int, int]:
    """
    Convert geographic coordinates to pixel coordinates
    
    Args:
        longitude: Longitude coordinate
        latitude: Latitude coordinate
        image_shape: Shape of image (height, width)
        bounds: Geographic bounds (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        Pixel coordinates (x, y)
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    height, width = image_shape
    
    # Calculate pixel position
    x = int((longitude - min_lon) / (max_lon - min_lon) * width)
    y = int((max_lat - latitude) / (max_lat - min_lat) * height)
    
    return x, y


def pixel_to_lon_lat(x: int, y: int,
                    image_shape: Tuple[int, int],
                    bounds: Tuple[float, float, float, float]) -> Tuple[float, float]:
    """
    Convert pixel coordinates to geographic coordinates
    
    Args:
        x: X pixel coordinate
        y: Y pixel coordinate
        image_shape: Shape of image (height, width)
        bounds: Geographic bounds (min_lon, min_lat, max_lon, max_lat)
        
    Returns:
        Geographic coordinates (longitude, latitude)
    """
    min_lon, min_lat, max_lon, max_lat = bounds
    height, width = image_shape
    
    longitude = min_lon + (x / width) * (max_lon - min_lon)
    latitude = max_lat - (y / height) * (max_lat - min_lat)
    
    return longitude, latitude


def bbox_to_polygon(bbox: Tuple[float, float, float, float]) -> np.ndarray:
    """
    Convert bounding box to polygon coordinates
    
    Args:
        bbox: Bounding box (x_min, y_min, x_max, y_max)
        
    Returns:
        Polygon coordinates as array
    """
    x_min, y_min, x_max, y_max = bbox
    polygon = np.array([
        [x_min, y_min],
        [x_max, y_min],
        [x_max, y_max],
        [x_min, y_max]
    ])
    
    return polygon


def calculate_distance(point1: Tuple[float, float],
                      point2: Tuple[float, float]) -> float:
    """
    Calculate Euclidean distance between two points
    
    Args:
        point1: First point (x, y)
        point2: Second point (x, y)
        
    Returns:
        Distance
    """
    return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize image to [0, 1] range
    
    Args:
        image: Input image
        
    Returns:
        Normalized image
    """
    min_val = np.min(image)
    max_val = np.max(image)
    
    if max_val - min_val == 0:
        return np.zeros_like(image, dtype=np.float32)
    
    normalized = (image - min_val) / (max_val - min_val)
    
    return normalized


def resize_image(image: np.ndarray, size: Tuple[int, int]) -> np.ndarray:
    """
    Resize image to specified size
    
    Args:
        image: Input image
        size: Target size (height, width)
        
    Returns:
        Resized image
    """
    import cv2
    height, width = size
    resized = cv2.resize(image, (width, height), interpolation=cv2.INTER_LINEAR)
    
    return resized
