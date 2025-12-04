"""Shelf Occupancy Analyzer - Sistema de análisis de ocupación de anaqueles."""

__version__ = "0.1.0"

from shelf_occupancy.config import Config, load_config
from shelf_occupancy.preprocessing import ImagePreprocessor, preprocess_image
from shelf_occupancy.utils import (
    BoundingBox,
    load_image,
    save_image,
)

__all__ = [
    "__version__",
    "Config",
    "load_config",
    "ImagePreprocessor",
    "preprocess_image",
    "BoundingBox",
    "load_image",
    "save_image",
]
