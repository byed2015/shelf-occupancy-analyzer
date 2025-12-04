"""Módulo para detección y corrección de perspectiva."""

from dataclasses import dataclass
from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger


@dataclass
class VanishingPoint:
    """Punto de fuga detectado."""
    x: float
    y: float
    confidence: float  # 0-1


class PerspectiveCorrector:
    """Corrector de perspectiva para imágenes con vista lateral."""
    
    def __init__(self, min_angle_for_correction: float = 10.0):
        """
        Inicializa el corrector.
        
        Args:
            min_angle_for_correction: Ángulo mínimo (grados) para aplicar corrección
        """
        self.min_angle = min_angle_for_correction
        logger.info(f"PerspectiveCorrector inicializado (umbral: {min_angle_for_correction}°)")
    
    def needs_correction(self, dominant_angle: float) -> bool:
        """
        Determina si una imagen necesita corrección de perspectiva.
        
        Args:
            dominant_angle: Ángulo dominante detectado en líneas horizontales
        
        Returns:
            True si el ángulo justifica corrección
        """
        abs_angle = abs(dominant_angle)
        # Normalizar ángulos cercanos a 180° o -180°
        if abs_angle > 90:
            abs_angle = 180 - abs_angle
        
        needs = abs_angle >= self.min_angle
        
        if needs:
            logger.info(f"Perspectiva significativa detectada: {dominant_angle:.1f}° (umbral: {self.min_angle}°)")
        else:
            logger.debug(f"Perspectiva insignificante: {dominant_angle:.1f}° < {self.min_angle}°")
        
        return needs
    
    def estimate_rotation_angle(self, dominant_angle: float) -> float:
        """
        Estima el ángulo de rotación necesario para enderezar la imagen.
        
        Para imágenes con perspectiva lateral, las líneas "horizontales"
        están inclinadas. Necesitamos rotar para alinearlas.
        
        Args:
            dominant_angle: Ángulo dominante de líneas horizontales
        
        Returns:
            Ángulo de rotación a aplicar
        """
        # Normalizar ángulo a rango [-45, 45]
        angle = dominant_angle
        
        if angle > 45:
            angle = angle - 90
        elif angle < -45:
            angle = angle + 90
        
        # El ángulo de rotación es el negativo (queremos rotar en dirección opuesta)
        rotation_angle = -angle
        
        logger.info(f"Ángulo de rotación calculado: {rotation_angle:.2f}° (dominante: {dominant_angle:.2f}°)")
        return rotation_angle
    
    def rotate_image(
        self,
        image: np.ndarray,
        angle: float,
        scale: float = 1.0,
        border_value: Tuple[int, int, int] = (0, 0, 0)
    ) -> np.ndarray:
        """
        Rota una imagen alrededor de su centro.
        
        Args:
            image: Imagen a rotar
            angle: Ángulo de rotación en grados (positivo = antihorario)
            scale: Factor de escala (1.0 = sin cambio)
            border_value: Color de relleno para áreas vacías
        
        Returns:
            Imagen rotada
        """
        h, w = image.shape[:2]
        center = (w // 2, h // 2)
        
        # Calcular matriz de rotación
        M = cv2.getRotationMatrix2D(center, angle, scale)
        
        # Calcular nuevo tamaño para evitar recorte
        cos = np.abs(M[0, 0])
        sin = np.abs(M[0, 1])
        
        new_w = int((h * sin) + (w * cos))
        new_h = int((h * cos) + (w * sin))
        
        # Ajustar matriz para centrar la rotación
        M[0, 2] += (new_w / 2) - center[0]
        M[1, 2] += (new_h / 2) - center[1]
        
        # Rotar
        rotated = cv2.warpAffine(
            image,
            M,
            (new_w, new_h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_CONSTANT,
            borderValue=border_value
        )
        
        logger.debug(f"Imagen rotada {angle:.2f}°: {image.shape} -> {rotated.shape}")
        return rotated
    
    def correct_simple_rotation(
        self,
        image: np.ndarray,
        dominant_angle: float
    ) -> Tuple[np.ndarray, float]:
        """
        Corrige perspectiva mediante rotación simple.
        Útil para perspectivas moderadas (< 30°).
        
        Args:
            image: Imagen original
            dominant_angle: Ángulo dominante de líneas horizontales
        
        Returns:
            (imagen_corregida, ángulo_aplicado)
        """
        if not self.needs_correction(dominant_angle):
            logger.info("Corrección no necesaria")
            return image, 0.0
        
        rotation_angle = self.estimate_rotation_angle(dominant_angle)
        
        # Aplicar rotación
        corrected = self.rotate_image(image, rotation_angle)
        
        logger.success(f"Perspectiva corregida: rotación de {rotation_angle:.2f}°")
        return corrected, rotation_angle
    
    def detect_vanishing_point(
        self,
        lines: List,  # List[Line] pero evitamos import circular
        image_shape: Tuple[int, int]
    ) -> Optional[VanishingPoint]:
        """
        Detecta punto de fuga analizando intersecciones de líneas paralelas.
        
        En perspectiva lateral, líneas paralelas convergen a un punto de fuga.
        
        Args:
            lines: Lista de líneas (objetos Line)
            image_shape: (altura, anchura) de la imagen
        
        Returns:
            VanishingPoint o None si no se puede detectar
        """
        if len(lines) < 3:
            logger.warning("Muy pocas líneas para detectar punto de fuga")
            return None
        
        # Calcular intersecciones entre pares de líneas
        intersections = []
        
        for i, line1 in enumerate(lines):
            for line2 in lines[i+1:]:
                intersection = self._line_intersection(
                    line1.x1, line1.y1, line1.x2, line1.y2,
                    line2.x1, line2.y1, line2.x2, line2.y2
                )
                
                if intersection:
                    x, y = intersection
                    # Filtrar intersecciones muy alejadas (outliers)
                    h, w = image_shape
                    if -w < x < 2*w and -h < y < 2*h:
                        intersections.append((x, y))
        
        if len(intersections) < 3:
            logger.warning("Muy pocas intersecciones válidas")
            return None
        
        # Usar RANSAC o mediana para punto robusto
        intersections_array = np.array(intersections)
        
        # Mediana robusta
        vp_x = float(np.median(intersections_array[:, 0]))
        vp_y = float(np.median(intersections_array[:, 1]))
        
        # Calcular confianza basada en densidad de intersecciones
        distances = np.sqrt((intersections_array[:, 0] - vp_x)**2 + 
                          (intersections_array[:, 1] - vp_y)**2)
        confidence = float(np.sum(distances < 100) / len(intersections))
        
        logger.info(f"Punto de fuga detectado: ({vp_x:.1f}, {vp_y:.1f}), confianza={confidence:.2f}")
        
        return VanishingPoint(vp_x, vp_y, confidence)
    
    def _line_intersection(
        self,
        x1: float, y1: float, x2: float, y2: float,
        x3: float, y3: float, x4: float, y4: float
    ) -> Optional[Tuple[float, float]]:
        """
        Calcula la intersección de dos líneas.
        
        Returns:
            (x, y) de intersección o None si son paralelas
        """
        # Usando determinantes
        denom = (x1 - x2) * (y3 - y4) - (y1 - y2) * (x3 - x4)
        
        if abs(denom) < 1e-6:  # Líneas paralelas
            return None
        
        t = ((x1 - x3) * (y3 - y4) - (y1 - y3) * (x3 - x4)) / denom
        
        x = x1 + t * (x2 - x1)
        y = y1 + t * (y2 - y1)
        
        return (x, y)
    
    def correct_perspective_transform(
        self,
        image: np.ndarray,
        vanishing_point: VanishingPoint,
        horizon_y: Optional[float] = None
    ) -> np.ndarray:
        """
        Aplica transformación de perspectiva completa.
        Más preciso que rotación simple pero más complejo.
        
        Args:
            image: Imagen original
            vanishing_point: Punto de fuga detectado
            horizon_y: Y de línea de horizonte (None = automático)
        
        Returns:
            Imagen corregida
        """
        h, w = image.shape[:2]
        
        if horizon_y is None:
            horizon_y = vanishing_point.y
        
        # Definir puntos fuente (trapecio perspectiva)
        # Asumiendo que el punto de fuga está en el horizonte
        margin = w * 0.1
        
        src_points = np.float32([
            [margin, horizon_y - h*0.3],  # Superior izquierda
            [w - margin, horizon_y - h*0.3],  # Superior derecha
            [w - margin*0.5, horizon_y + h*0.3],  # Inferior derecha
            [margin*0.5, horizon_y + h*0.3]  # Inferior izquierda
        ])
        
        # Definir puntos destino (rectángulo)
        dst_points = np.float32([
            [margin, 0],
            [w - margin, 0],
            [w - margin, h],
            [margin, h]
        ])
        
        # Calcular matriz de transformación
        M = cv2.getPerspectiveTransform(src_points, dst_points)
        
        # Aplicar transformación
        corrected = cv2.warpPerspective(image, M, (w, h))
        
        logger.success("Transformación de perspectiva aplicada")
        return corrected


def correct_perspective(
    image: np.ndarray,
    dominant_angle: float,
    method: str = "rotation"
) -> Tuple[np.ndarray, bool]:
    """
    Función de conveniencia para corregir perspectiva.
    
    Args:
        image: Imagen a corregir
        dominant_angle: Ángulo dominante de líneas horizontales
        method: "rotation" (simple) o "transform" (completo)
    
    Returns:
        (imagen_corregida, fue_corregida)
    """
    corrector = PerspectiveCorrector()
    
    if method == "rotation":
        corrected, angle = corrector.correct_simple_rotation(image, dominant_angle)
        was_corrected = abs(angle) > 0.1
        return corrected, was_corrected
    
    else:
        raise NotImplementedError(f"Método {method} no implementado aún")
