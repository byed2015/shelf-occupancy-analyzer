"""Módulo de detección de estructuras."""

from shelf_occupancy.detection.edges import EdgeDetector, detect_edges
from shelf_occupancy.detection.lines import Line, LineDetector, detect_lines
from shelf_occupancy.detection.shelves import ShelfDetector, detect_shelves_from_lines

# Importar ObjectDetector solo si ultralytics está disponible
try:
    from shelf_occupancy.detection.objects import ObjectDetector, DetectedObject, detect_products
    OBJECTS_AVAILABLE = True
except ImportError:
    OBJECTS_AVAILABLE = False
    ObjectDetector = None
    DetectedObject = None
    detect_products = None

__all__ = [
    "EdgeDetector",
    "detect_edges",
    "Line",
    "LineDetector",
    "detect_lines",
    "ShelfDetector",
    "detect_shelves_from_lines",
    "ObjectDetector",
    "DetectedObject",
    "detect_products",
    "OBJECTS_AVAILABLE",
]
