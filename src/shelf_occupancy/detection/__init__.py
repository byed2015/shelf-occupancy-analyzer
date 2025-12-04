"""Módulo de detección de estructuras."""

from shelf_occupancy.detection.edges import EdgeDetector, detect_edges
from shelf_occupancy.detection.lines import Line, LineDetector, detect_lines
from shelf_occupancy.detection.shelves import ShelfDetector, detect_shelves_from_lines

__all__ = [
    "EdgeDetector",
    "detect_edges",
    "Line",
    "LineDetector",
    "detect_lines",
    "ShelfDetector",
    "detect_shelves_from_lines",
]
