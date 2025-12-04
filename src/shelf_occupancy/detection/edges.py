"""Módulo de detección de bordes."""

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from shelf_occupancy.config import CannyConfig


class EdgeDetector:
    """Detector de bordes usando algoritmo de Canny."""
    
    def __init__(self, config: Optional[CannyConfig] = None):
        """
        Inicializa el detector de bordes.
        
        Args:
            config: Configuración de Canny. Si None, usa valores por defecto.
        """
        if config is None:
            config = CannyConfig()
        self.config = config
        logger.info("EdgeDetector inicializado")
    
    def detect(
        self,
        image: np.ndarray,
        low_threshold: Optional[int] = None,
        high_threshold: Optional[int] = None,
        auto_threshold: bool = False
    ) -> np.ndarray:
        """
        Detecta bordes en una imagen usando Canny.
        
        Args:
            image: Imagen de entrada (BGR o escala de grises)
            low_threshold: Umbral bajo. Si None, usa config.
            high_threshold: Umbral alto. Si None, usa config.
            auto_threshold: Si calcular umbrales automáticamente usando Otsu
        
        Returns:
            Imagen binaria con bordes detectados
        """
        # Convertir a escala de grises si es necesario
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
        
        # Calcular umbrales automáticos si se solicita
        if auto_threshold:
            # Método de Otsu para calcular umbral
            sigma = 0.33
            median = np.median(gray)
            low_threshold = int(max(0, (1.0 - sigma) * median))
            high_threshold = int(min(255, (1.0 + sigma) * median))
            logger.debug(f"Umbrales automáticos: low={low_threshold}, high={high_threshold}")
        else:
            if low_threshold is None:
                low_threshold = self.config.low_threshold
            if high_threshold is None:
                high_threshold = self.config.high_threshold
        
        logger.debug(f"Aplicando Canny con umbrales: {low_threshold}, {high_threshold}")
        
        # Aplicar detección de bordes
        edges = cv2.Canny(
            gray,
            low_threshold,
            high_threshold,
            apertureSize=self.config.aperture_size
        )
        
        return edges
    
    def detect_with_morphology(
        self,
        image: np.ndarray,
        kernel_size: Tuple[int, int] = (3, 3),
        iterations: int = 1
    ) -> np.ndarray:
        """
        Detecta bordes y aplica operaciones morfológicas para limpiar el resultado.
        
        Args:
            image: Imagen de entrada
            kernel_size: Tamaño del kernel morfológico
            iterations: Número de iteraciones de operaciones morfológicas
        
        Returns:
            Imagen de bordes procesada
        """
        edges = self.detect(image)
        
        # Aplicar cierre para conectar bordes cercanos
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, kernel_size)
        edges_closed = cv2.morphologyEx(edges, cv2.MORPH_CLOSE, kernel, iterations=iterations)
        
        return edges_closed
    
    def detect_multiscale(
        self,
        image: np.ndarray,
        scales: list = [0.5, 1.0, 1.5]
    ) -> np.ndarray:
        """
        Detecta bordes a múltiples escalas y combina los resultados.
        
        Args:
            image: Imagen de entrada
            scales: Lista de factores de escala
        
        Returns:
            Imagen combinada de bordes multi-escala
        """
        h, w = image.shape[:2]
        combined = np.zeros((h, w), dtype=np.uint8)
        
        for scale in scales:
            # Redimensionar imagen
            if scale != 1.0:
                new_w = int(w * scale)
                new_h = int(h * scale)
                scaled = cv2.resize(image, (new_w, new_h))
            else:
                scaled = image
            
            # Detectar bordes
            edges = self.detect(scaled)
            
            # Redimensionar de vuelta si es necesario
            if scale != 1.0:
                edges = cv2.resize(edges, (w, h))
            
            # Combinar
            combined = cv2.bitwise_or(combined, edges)
        
        return combined


def detect_edges(
    image: np.ndarray,
    low_threshold: int = 50,
    high_threshold: int = 150,
    aperture_size: int = 3
) -> np.ndarray:
    """
    Función de conveniencia para detectar bordes.
    
    Args:
        image: Imagen de entrada
        low_threshold: Umbral bajo de Canny
        high_threshold: Umbral alto de Canny
        aperture_size: Tamaño de apertura de Sobel
    
    Returns:
        Imagen de bordes
    """
    config = CannyConfig(
        low_threshold=low_threshold,
        high_threshold=high_threshold,
        aperture_size=aperture_size
    )
    detector = EdgeDetector(config)
    return detector.detect(image)
