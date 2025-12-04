"""Utilidades geométricas."""

from typing import List, Optional, Tuple

import cv2
import numpy as np


class BoundingBox:
    """Clase para representar un bounding box."""
    
    def __init__(self, x: int, y: int, width: int, height: int):
        """
        Inicializa un bounding box.
        
        Args:
            x: Coordenada x de la esquina superior izquierda
            y: Coordenada y de la esquina superior izquierda
            width: Ancho del box
            height: Alto del box
        """
        self.x = x
        self.y = y
        self.width = width
        self.height = height
    
    @property
    def x1(self) -> int:
        """Coordenada x izquierda."""
        return self.x
    
    @property
    def y1(self) -> int:
        """Coordenada y superior."""
        return self.y
    
    @property
    def x2(self) -> int:
        """Coordenada x derecha."""
        return self.x + self.width
    
    @property
    def y2(self) -> int:
        """Coordenada y inferior."""
        return self.y + self.height
    
    @property
    def center(self) -> Tuple[float, float]:
        """Centro del bounding box."""
        return (self.x + self.width / 2, self.y + self.height / 2)
    
    @property
    def area(self) -> int:
        """Área del bounding box."""
        return self.width * self.height
    
    def to_xyxy(self) -> Tuple[int, int, int, int]:
        """Retorna coordenadas en formato (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def to_xywh(self) -> Tuple[int, int, int, int]:
        """Retorna coordenadas en formato (x, y, w, h)."""
        return (self.x, self.y, self.width, self.height)
    
    def intersects(self, other: "BoundingBox") -> bool:
        """Verifica si intersecta con otro bounding box."""
        return not (
            self.x2 < other.x1 or
            self.x1 > other.x2 or
            self.y2 < other.y1 or
            self.y1 > other.y2
        )
    
    def intersection(self, other: "BoundingBox") -> float:
        """Calcula el área de intersección con otro bounding box."""
        if not self.intersects(other):
            return 0.0
        
        x1 = max(self.x1, other.x1)
        y1 = max(self.y1, other.y1)
        x2 = min(self.x2, other.x2)
        y2 = min(self.y2, other.y2)
        
        return float((x2 - x1) * (y2 - y1))
    
    def iou(self, other: "BoundingBox") -> float:
        """Calcula Intersection over Union con otro bounding box."""
        intersection = self.intersection(other)
        union = self.area + other.area - intersection
        
        if union == 0:
            return 0.0
        
        return intersection / union
    
    def __repr__(self) -> str:
        return f"BoundingBox(x={self.x}, y={self.y}, w={self.width}, h={self.height})"


class Quadrilateral:
    """
    Clase para representar un cuadrilátero (anaquel con perspectiva).
    
    Permite anaqueles inclinados siguiendo las líneas naturales de la imagen,
    sin necesidad de corregir la perspectiva globalmente.
    """
    
    def __init__(
        self,
        top_left: Tuple[float, float],
        top_right: Tuple[float, float],
        bottom_right: Tuple[float, float],
        bottom_left: Tuple[float, float]
    ):
        """
        Inicializa un cuadrilátero con 4 puntos.
        
        Args:
            top_left: Punto superior izquierdo (x, y)
            top_right: Punto superior derecho (x, y)
            bottom_right: Punto inferior derecho (x, y)
            bottom_left: Punto inferior izquierdo (x, y)
        """
        self.top_left = top_left
        self.top_right = top_right
        self.bottom_right = bottom_right
        self.bottom_left = bottom_left
    
    @classmethod
    def from_bbox(cls, bbox: BoundingBox) -> "Quadrilateral":
        """Crea un cuadrilátero desde un BoundingBox rectangular."""
        return cls(
            top_left=(bbox.x, bbox.y),
            top_right=(bbox.x + bbox.width, bbox.y),
            bottom_right=(bbox.x + bbox.width, bbox.y + bbox.height),
            bottom_left=(bbox.x, bbox.y + bbox.height)
        )
    
    @classmethod
    def from_lines(
        cls,
        top_line: Optional[Tuple[float, float, float, float]],
        bottom_line: Optional[Tuple[float, float, float, float]],
        image_width: int,
        default_x_left: float = 0.0,
        default_x_right: Optional[float] = None
    ) -> "Quadrilateral":
        """
        Crea un cuadrilátero desde dos líneas (superior e inferior).
        Las líneas pueden estar inclinadas (perspectiva).
        
        Args:
            top_line: Línea superior (x1, y1, x2, y2) o None
            bottom_line: Línea inferior (x1, y1, x2, y2) o None
            image_width: Ancho de la imagen
            default_x_left: X por defecto para borde izquierdo
            default_x_right: X por defecto para borde derecho
        
        Returns:
            Cuadrilátero que sigue las líneas inclinadas
        """
        if default_x_right is None:
            default_x_right = image_width
        
        # Calcular puntos de las líneas en los bordes izquierdo/derecho
        if top_line:
            top_left = cls._line_y_at_x(top_line, default_x_left)
            top_right = cls._line_y_at_x(top_line, default_x_right)
        else:
            top_left = top_right = 0
        
        if bottom_line:
            bottom_left = cls._line_y_at_x(bottom_line, default_x_left)
            bottom_right = cls._line_y_at_x(bottom_line, default_x_right)
        else:
            bottom_left = bottom_right = 0
        
        return cls(
            top_left=(default_x_left, top_left),
            top_right=(default_x_right, top_right),
            bottom_right=(default_x_right, bottom_right),
            bottom_left=(default_x_left, bottom_left)
        )
    
    @staticmethod
    def _line_y_at_x(line: Tuple[float, float, float, float], x: float) -> float:
        """
        Calcula el valor Y de una línea en una coordenada X dada.
        
        Args:
            line: Línea (x1, y1, x2, y2)
            x: Coordenada X donde calcular Y
        
        Returns:
            Coordenada Y
        """
        x1, y1, x2, y2 = line
        
        if abs(x2 - x1) < 1e-6:  # Línea vertical
            return (y1 + y2) / 2
        
        # Interpolación lineal: y = y1 + (y2 - y1) * (x - x1) / (x2 - x1)
        t = (x - x1) / (x2 - x1)
        y = y1 + (y2 - y1) * t
        
        return y
    
    def to_bbox(self) -> BoundingBox:
        """
        Convierte a BoundingBox rectangular (bounding box mínimo).
        Útil para compatibilidad con código existente.
        """
        all_x = [self.top_left[0], self.top_right[0], self.bottom_right[0], self.bottom_left[0]]
        all_y = [self.top_left[1], self.top_right[1], self.bottom_right[1], self.bottom_left[1]]
        
        x_min = int(min(all_x))
        y_min = int(min(all_y))
        x_max = int(max(all_x))
        y_max = int(max(all_y))
        
        return BoundingBox(x_min, y_min, x_max - x_min, y_max - y_min)
    
    def get_corners(self) -> np.ndarray:
        """
        Obtiene los 4 vértices como array numpy para OpenCV.
        
        Returns:
            Array (4, 2) con coordenadas [[x,y], [x,y], ...]
        """
        return np.array([
            self.top_left,
            self.top_right,
            self.bottom_right,
            self.bottom_left
        ], dtype=np.float32)
    
    def get_perspective_transform(self, width: int, height: int) -> np.ndarray:
        """
        Calcula la matriz de transformación de perspectiva para "enderezar"
        este cuadrilátero a un rectángulo.
        
        Args:
            width: Ancho del rectángulo destino
            height: Alto del rectángulo destino
        
        Returns:
            Matriz 3x3 de transformación de perspectiva
        """
        src = self.get_corners()
        dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)
        
        M = cv2.getPerspectiveTransform(src, dst)
        return M
    
    def warp_to_rectangle(self, image: np.ndarray, width: int, height: int) -> np.ndarray:
        """
        Extrae y "endereza" la región de este cuadrilátero.
        
        Transforma el cuadrilátero inclinado en un rectángulo,
        permitiendo análisis uniforme sin distorsión.
        
        Args:
            image: Imagen completa
            width: Ancho del rectángulo destino
            height: Alto del rectángulo destino
        
        Returns:
            Región extraída y enderezada
        """
        M = self.get_perspective_transform(width, height)
        warped = cv2.warpPerspective(image, M, (width, height))
        return warped
    
    @property
    def center(self) -> Tuple[float, float]:
        """Centro del cuadrilátero."""
        all_x = [self.top_left[0], self.top_right[0], self.bottom_right[0], self.bottom_left[0]]
        all_y = [self.top_left[1], self.top_right[1], self.bottom_right[1], self.bottom_left[1]]
        return (sum(all_x) / 4, sum(all_y) / 4)
    
    @property
    def area(self) -> float:
        """Área aproximada del cuadrilátero (usando Shoelace formula)."""
        corners = self.get_corners()
        x = corners[:, 0]
        y = corners[:, 1]
        
        # Shoelace formula
        area = 0.5 * abs(sum(x[i] * y[i+1] - x[i+1] * y[i] for i in range(-1, 3)))
        return area
    
    def __repr__(self) -> str:
        return (f"Quadrilateral(TL={self.top_left}, TR={self.top_right}, "
                f"BR={self.bottom_right}, BL={self.bottom_left})")


def calculate_angle(line: Tuple[float, float, float, float]) -> float:
    """
    Calcula el ángulo de una línea en grados.
    
    Args:
        line: Línea en formato (x1, y1, x2, y2)
    
    Returns:
        Ángulo en grados [-90, 90]
    """
    x1, y1, x2, y2 = line
    angle = np.arctan2(y2 - y1, x2 - x1) * 180 / np.pi
    return angle


def is_horizontal(line: Tuple[float, float, float, float], tolerance: float = 10.0) -> bool:
    """
    Verifica si una línea es horizontal.
    
    Args:
        line: Línea en formato (x1, y1, x2, y2)
        tolerance: Tolerancia en grados
    
    Returns:
        True si es horizontal
    """
    angle = abs(calculate_angle(line))
    return angle <= tolerance or angle >= (180 - tolerance)


def is_vertical(line: Tuple[float, float, float, float], tolerance: float = 10.0) -> bool:
    """
    Verifica si una línea es vertical.
    
    Args:
        line: Línea en formato (x1, y1, x2, y2)
        tolerance: Tolerancia en grados
    
    Returns:
        True si es vertical
    """
    angle = abs(calculate_angle(line))
    return abs(angle - 90) <= tolerance or abs(angle + 90) <= tolerance


def line_distance(line1: Tuple[float, float, float, float], 
                  line2: Tuple[float, float, float, float]) -> float:
    """
    Calcula la distancia entre dos líneas paralelas.
    
    Args:
        line1: Primera línea (x1, y1, x2, y2)
        line2: Segunda línea (x1, y1, x2, y2)
    
    Returns:
        Distancia perpendicular entre las líneas
    """
    # Punto medio de cada línea
    x1_mid = (line1[0] + line1[2]) / 2
    y1_mid = (line1[1] + line1[3]) / 2
    x2_mid = (line2[0] + line2[2]) / 2
    y2_mid = (line2[1] + line2[3]) / 2
    
    # Distancia euclidiana entre puntos medios
    return np.sqrt((x2_mid - x1_mid)**2 + (y2_mid - y1_mid)**2)


def create_grid(
    bbox: BoundingBox,
    grid_size: Tuple[int, int]
) -> List[List[BoundingBox]]:
    """
    Crea una cuadrícula de bounding boxes dentro de un área.
    
    Args:
        bbox: Bounding box del área a dividir
        grid_size: Tamaño de la cuadrícula (cols, rows)
    
    Returns:
        Matriz de bounding boxes [rows][cols]
    """
    cols, rows = grid_size
    cell_width = bbox.width / cols
    cell_height = bbox.height / rows
    
    grid = []
    for row in range(rows):
        row_cells = []
        for col in range(cols):
            x = bbox.x + int(col * cell_width)
            y = bbox.y + int(row * cell_height)
            w = int(cell_width)
            h = int(cell_height)
            row_cells.append(BoundingBox(x, y, w, h))
        grid.append(row_cells)
    
    return grid


def point_in_bbox(point: Tuple[float, float], bbox: BoundingBox) -> bool:
    """
    Verifica si un punto está dentro de un bounding box.
    
    Args:
        point: Punto (x, y)
        bbox: Bounding box
    
    Returns:
        True si el punto está dentro
    """
    x, y = point
    return bbox.x1 <= x <= bbox.x2 and bbox.y1 <= y <= bbox.y2
