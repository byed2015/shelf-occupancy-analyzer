"""Módulo de detección de líneas."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from shelf_occupancy.config import HoughConfig
from shelf_occupancy.utils.geometry import calculate_angle, is_horizontal, is_vertical


@dataclass
class Line:
    """Representa una línea detectada."""
    x1: float
    y1: float
    x2: float
    y2: float
    angle: Optional[float] = None
    length: Optional[float] = None
    
    def __post_init__(self):
        """Calcula ángulo y longitud si no están definidos."""
        if self.angle is None:
            self.angle = calculate_angle((self.x1, self.y1, self.x2, self.y2))
        if self.length is None:
            self.length = np.sqrt((self.x2 - self.x1)**2 + (self.y2 - self.y1)**2)
    
    def to_tuple(self) -> Tuple[float, float, float, float]:
        """Retorna la línea como tupla (x1, y1, x2, y2)."""
        return (self.x1, self.y1, self.x2, self.y2)
    
    def is_horizontal(self, tolerance: float = 10.0) -> bool:
        """Verifica si la línea es horizontal."""
        return is_horizontal(self.to_tuple(), tolerance)
    
    def is_vertical(self, tolerance: float = 10.0) -> bool:
        """Verifica si la línea es vertical."""
        return is_vertical(self.to_tuple(), tolerance)


class LineDetector:
    """Detector de líneas usando Transformada de Hough."""
    
    def __init__(self, config: Optional[HoughConfig] = None):
        """
        Inicializa el detector de líneas.
        
        Args:
            config: Configuración de Hough. Si None, usa valores por defecto.
        """
        if config is None:
            config = HoughConfig()
        self.config = config
        logger.info("LineDetector inicializado")
    
    def detect(
        self,
        edges: np.ndarray,
        min_line_length: Optional[int] = None,
        max_line_gap: Optional[int] = None,
        use_polar: bool = False
    ) -> List[Line]:
        """
        Detecta líneas en una imagen de bordes usando HoughLinesP o HoughLines.
        
        Args:
            edges: Imagen de bordes (binaria)
            min_line_length: Longitud mínima de línea. Si None, usa config.
            max_line_gap: Gap máximo entre segmentos. Si None, usa config.
            use_polar: Si True, usa HoughLines (polar) que captura todos los ángulos
        
        Returns:
            Lista de líneas detectadas
        """
        if min_line_length is None:
            min_line_length = self.config.min_line_length
        if max_line_gap is None:
            max_line_gap = self.config.max_line_gap
        
        if use_polar:
            logger.debug(f"Detectando líneas con HoughLines (polar) - captura todos los ángulos")
            lines = self._detect_polar(edges)
        else:
            logger.debug(f"Detectando líneas con HoughLinesP")
            logger.debug(f"  - min_line_length: {min_line_length}")
            logger.debug(f"  - max_line_gap: {max_line_gap}")
            
            # Detectar líneas
            lines_raw = cv2.HoughLinesP(
                edges,
                rho=self.config.rho,
                theta=self.config.theta,
                threshold=self.config.threshold,
                minLineLength=min_line_length,
                maxLineGap=max_line_gap
            )
            
            if lines_raw is None:
                logger.warning("No se detectaron líneas")
                return []
            
            # Convertir a objetos Line
            lines = []
            for line in lines_raw:
                x1, y1, x2, y2 = line[0]
                lines.append(Line(float(x1), float(y1), float(x2), float(y2)))
        
        logger.info(f"Detectadas {len(lines)} líneas")
        return lines
    
    def _detect_polar(self, edges: np.ndarray) -> List[Line]:
        """
        Detecta líneas usando HoughLines (representación polar).
        Este método captura líneas en TODOS los ángulos.
        
        Args:
            edges: Imagen de bordes
        
        Returns:
            Lista de líneas en coordenadas cartesianas
        """
        h, w = edges.shape
        
        # Detectar líneas en espacio polar (rho, theta)
        lines_polar = cv2.HoughLines(
            edges,
            rho=1,  # Resolución de 1 pixel
            theta=np.pi/180,  # Resolución de 1 grado
            threshold=int(self.config.threshold * 0.7)  # Umbral más bajo para capturar más
        )
        
        if lines_polar is None:
            logger.warning("No se detectaron líneas en modo polar")
            return []
        
        # Convertir de polar a cartesiano
        lines_cartesian = []
        max_dim = max(h, w)
        
        for line_polar in lines_polar:
            rho, theta = line_polar[0]
            
            # Convertir a coordenadas cartesianas
            a = np.cos(theta)
            b = np.sin(theta)
            x0 = a * rho
            y0 = b * rho
            
            # Extender línea a los bordes de la imagen
            length = max_dim * 2
            x1 = int(x0 + length * (-b))
            y1 = int(y0 + length * (a))
            x2 = int(x0 - length * (-b))
            y2 = int(y0 - length * (a))
            
            # Recortar a límites de imagen
            x1, y1, x2, y2 = self._clip_line_to_image(x1, y1, x2, y2, w, h)
            
            if x1 is not None:  # Línea válida
                lines_cartesian.append(Line(float(x1), float(y1), float(x2), float(y2)))
        
        logger.debug(f"Convertidas {len(lines_cartesian)} líneas polares a cartesianas")
        return lines_cartesian
    
    def _clip_line_to_image(
        self,
        x1: int, y1: int, x2: int, y2: int,
        width: int, height: int
    ) -> Tuple[Optional[int], Optional[int], Optional[int], Optional[int]]:
        """
        Recorta una línea a los límites de la imagen usando Cohen-Sutherland.
        
        Returns:
            (x1, y1, x2, y2) recortados, o (None, None, None, None) si fuera de límites
        """
        # Algoritmo simplificado de recorte
        def clip_point(x, y):
            x = max(0, min(x, width - 1))
            y = max(0, min(y, height - 1))
            return x, y
        
        # Verificar si la línea interseca con la imagen
        if (x1 < 0 and x2 < 0) or (x1 >= width and x2 >= width):
            return None, None, None, None
        if (y1 < 0 and y2 < 0) or (y1 >= height and y2 >= height):
            return None, None, None, None
        
        # Recortar puntos
        x1, y1 = clip_point(x1, y1)
        x2, y2 = clip_point(x2, y2)
        
        # Verificar longitud mínima
        length = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
        if length < 10:  # Línea muy corta
            return None, None, None, None
        
        return x1, y1, x2, y2
    
    def detect_dominant_angle(self, lines: List[Line]) -> float:
        """
        Detecta el ángulo dominante en un conjunto de líneas.
        Útil para imágenes con perspectiva donde las líneas "horizontales"
        están inclinadas.
        
        Args:
            lines: Lista de líneas
        
        Returns:
            Ángulo dominante en grados
        """
        if not lines:
            return 0.0
        
        # Agrupar líneas por ángulo similar
        angles = [l.angle for l in lines]
        
        # Normalizar ángulos a rango [-90, 90]
        normalized_angles = []
        for angle in angles:
            if angle > 90:
                angle = angle - 180
            elif angle < -90:
                angle = angle + 180
            normalized_angles.append(angle)
        
        # Encontrar ángulo más frecuente usando histograma
        hist, bins = np.histogram(normalized_angles, bins=36)  # bins de 5 grados
        dominant_bin = np.argmax(hist)
        dominant_angle = (bins[dominant_bin] + bins[dominant_bin + 1]) / 2
        
        logger.info(f"Ángulo dominante detectado: {dominant_angle:.2f}°")
        return dominant_angle
    
    def filter_by_orientation(
        self,
        lines: List[Line],
        orientation: str = "horizontal",
        tolerance: float = 10.0,
        adaptive: bool = True
    ) -> List[Line]:
        """
        Filtra líneas por orientación con soporte de perspectiva adaptativa.
        
        En modo adaptativo para horizontal:
        1. Detecta ángulo dominante (puede ser inclinado)
        2. Filtra líneas cercanas a ese ángulo (no a 0°)
        
        Args:
            lines: Lista de líneas
            orientation: "horizontal", "vertical" o "both"
            tolerance: Tolerancia en grados
            adaptive: Si True, detecta ángulo dominante para imágenes con perspectiva
        
        Returns:
            Lista de líneas filtradas
        """
        if not lines:
            return []
        
        # Si es adaptativo, detectar ángulo dominante
        if adaptive and orientation == "horizontal":
            # Para horizontal, considerar líneas en rango "quasi-horizontal"
            # Ampliar rango para capturar perspectivas extremas
            quasi_horizontal = [l for l in lines if abs(l.angle) < 60 or abs(abs(l.angle) - 180) < 60]
            
            if quasi_horizontal:
                dominant_angle = self.detect_dominant_angle(quasi_horizontal)
                
                # Filtrar líneas cercanas al ángulo dominante
                filtered = []
                for line in lines:
                    angle_diff = self._angle_difference(line.angle, dominant_angle)
                    if angle_diff <= tolerance:
                        filtered.append(line)
                
                logger.info(f"Filtradas {len(filtered)} líneas {orientation} (adaptativo, ángulo={dominant_angle:.1f}°) de {len(lines)} totales")
                return filtered
        
        elif adaptive and orientation == "vertical":
            # Para vertical, buscar líneas perpendiculares a horizontal dominante
            all_angles = [l.angle for l in lines]
            
            # Detectar ángulo vertical dominante
            # Vertical ≈ ±90° respecto a horizontal
            quasi_vertical = [
                l for l in lines
                if 45 < abs(l.angle) < 135 or abs(abs(l.angle) - 180) > 45
            ]
            
            if quasi_vertical:
                dominant_angle = self.detect_dominant_angle(quasi_vertical)
                
                filtered = []
                for line in lines:
                    angle_diff = self._angle_difference(line.angle, dominant_angle)
                    if angle_diff <= tolerance:
                        filtered.append(line)
                
                logger.info(f"Filtradas {len(filtered)} líneas {orientation} (adaptativo, ángulo={dominant_angle:.1f}°) de {len(lines)} totales")
                return filtered
        
        # Modo no adaptativo (original)
        if orientation == "horizontal":
            filtered = [l for l in lines if l.is_horizontal(tolerance)]
        elif orientation == "vertical":
            filtered = [l for l in lines if l.is_vertical(tolerance)]
        elif orientation == "both":
            filtered = [l for l in lines if l.is_horizontal(tolerance) or l.is_vertical(tolerance)]
        else:
            raise ValueError(f"Orientación desconocida: {orientation}")
        
        logger.info(f"Filtradas {len(filtered)} líneas {orientation} de {len(lines)} totales")
        return filtered
    
    def _angle_difference(self, angle1: float, angle2: float) -> float:
        """
        Calcula la diferencia mínima entre dos ángulos.
        Maneja correctamente ángulos en rango [-180, 180].
        
        Returns:
            Diferencia en grados [0, 180]
        """
        diff = abs(angle1 - angle2)
        
        # Normalizar a rango [0, 180]
        while diff > 180:
            diff = 360 - diff
        
        return diff
    
    def filter_by_length(
        self,
        lines: List[Line],
        min_length: float,
        max_length: Optional[float] = None
    ) -> List[Line]:
        """
        Filtra líneas por longitud.
        
        Args:
            lines: Lista de líneas
            min_length: Longitud mínima
            max_length: Longitud máxima (opcional)
        
        Returns:
            Lista de líneas filtradas
        """
        filtered = [l for l in lines if l.length >= min_length]
        
        if max_length is not None:
            filtered = [l for l in filtered if l.length <= max_length]
        
        logger.info(f"Filtradas {len(filtered)} líneas por longitud de {len(lines)} totales")
        return filtered
    
    def merge_similar_lines(
        self,
        lines: List[Line],
        angle_threshold: float = 5.0,
        distance_threshold: float = 30.0,
        adaptive_angle: Optional[float] = None
    ) -> List[Line]:
        """
        Fusiona líneas similares (cercanas y paralelas) con soporte de perspectiva.
        
        Args:
            lines: Lista de líneas
            angle_threshold: Diferencia máxima de ángulo en grados
            distance_threshold: Distancia máxima entre líneas
            adaptive_angle: Ángulo dominante para comparación adaptativa
        
        Returns:
            Lista de líneas fusionadas
        """
        if not lines:
            return []
        
        # Agrupar líneas similares
        merged = []
        used = set()
        
        for i, line1 in enumerate(lines):
            if i in used:
                continue
            
            # Encontrar líneas similares
            similar = [line1]
            
            for j, line2 in enumerate(lines):
                if i == j or j in used:
                    continue
                
                # Verificar si son similares
                if adaptive_angle is not None:
                    # Modo adaptativo: comparar contra ángulo dominante
                    angle_diff1 = abs(line1.angle - adaptive_angle)
                    angle_diff2 = abs(line2.angle - adaptive_angle)
                    if angle_diff1 > 180:
                        angle_diff1 = 360 - angle_diff1
                    if angle_diff2 > 180:
                        angle_diff2 = 360 - angle_diff2
                    
                    # Ambas deben estar cerca del ángulo dominante
                    if angle_diff1 <= angle_threshold and angle_diff2 <= angle_threshold:
                        similar_angle = True
                    else:
                        similar_angle = False
                else:
                    # Modo normal: comparar entre sí
                    angle_diff = abs(line1.angle - line2.angle)
                    if angle_diff > 180:
                        angle_diff = 360 - angle_diff
                    similar_angle = angle_diff <= angle_threshold
                
                if similar_angle:
                    # Calcular distancia entre líneas
                    mid1_x = (line1.x1 + line1.x2) / 2
                    mid1_y = (line1.y1 + line1.y2) / 2
                    mid2_x = (line2.x1 + line2.x2) / 2
                    mid2_y = (line2.y1 + line2.y2) / 2
                    
                    distance = np.sqrt((mid2_x - mid1_x)**2 + (mid2_y - mid1_y)**2)
                    
                    if distance <= distance_threshold:
                        similar.append(line2)
                        used.add(j)
            
            # Promediar líneas similares
            if len(similar) > 1:
                x1_avg = np.mean([l.x1 for l in similar])
                y1_avg = np.mean([l.y1 for l in similar])
                x2_avg = np.mean([l.x2 for l in similar])
                y2_avg = np.mean([l.y2 for l in similar])
                merged.append(Line(x1_avg, y1_avg, x2_avg, y2_avg))
            else:
                merged.append(line1)
            
            used.add(i)
        
        logger.info(f"Fusionadas {len(lines)} líneas en {len(merged)} líneas")
        return merged


def detect_lines(
    edges: np.ndarray,
    min_line_length: int = 100,
    max_line_gap: int = 10,
    orientation: Optional[str] = None
) -> List[Line]:
    """
    Función de conveniencia para detectar líneas.
    
    Args:
        edges: Imagen de bordes
        min_line_length: Longitud mínima de línea
        max_line_gap: Gap máximo entre segmentos
        orientation: Filtrar por orientación ("horizontal", "vertical", "both")
    
    Returns:
        Lista de líneas detectadas
    """
    config = HoughConfig(min_line_length=min_line_length, max_line_gap=max_line_gap)
    detector = LineDetector(config)
    lines = detector.detect(edges)
    
    if orientation:
        lines = detector.filter_by_orientation(lines, orientation)
    
    return lines
