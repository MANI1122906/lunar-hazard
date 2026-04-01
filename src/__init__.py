"""
Lunar-Hazard-Detective Package

A complete system for detecting lunar surface hazards using deep learning.
"""

__version__ = "1.0.0"
__author__ = "Your Team"

from . import preprocessing
from . import slope_engine
from . import detection
from . import utils

__all__ = ['preprocessing', 'slope_engine', 'detection', 'utils']
