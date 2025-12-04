"""Módulo de utilidades."""

from shelf_occupancy.utils.geometry import (
    BoundingBox,
    Quadrilateral,
    calculate_angle,
    create_grid,
    is_horizontal,
    is_vertical,
    line_distance,
    point_in_bbox,
)
from shelf_occupancy.utils.image_io import (
    ensure_rgb,
    get_image_dimensions,
    image_to_uint8,
    load_image,
    normalize_image,
    resize_image,
    save_image,
)

__all__ = [
    "BoundingBox",
    "Quadrilateral",
    "calculate_angle",
    "create_grid",
    "is_horizontal",
    "is_vertical",
    "line_distance",
    "point_in_bbox",
    "ensure_rgb",
    "get_image_dimensions",
    "image_to_uint8",
    "load_image",
    "normalize_image",
    "resize_image",
    "save_image",
]
