"""Módulo de preprocesamiento de imágenes."""

from typing import Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from shelf_occupancy.config import PreprocessingConfig
from shelf_occupancy.utils import image_to_uint8, normalize_image, resize_image


class ImagePreprocessor:
    """Preprocesador de imágenes para análisis de anaqueles."""
    
    def __init__(self, config: PreprocessingConfig):
        """
        Inicializa el preprocesador.
        
        Args:
            config: Configuración de preprocesamiento
        """
        self.config = config
        logger.info("Preprocesador de imágenes inicializado")
    
    def apply_clahe(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica CLAHE (Contrast Limited Adaptive Histogram Equalization).
        
        Args:
            image: Imagen de entrada (BGR o escala de grises)
        
        Returns:
            Imagen con corrección de iluminación
        """
        logger.debug("Aplicando CLAHE")
        
        # Convertir a LAB para aplicar CLAHE solo en el canal L
        if len(image.shape) == 3:
            lab = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
            l, a, b = cv2.split(lab)
        else:
            l = image
        
        # Crear objeto CLAHE
        clahe = cv2.createCLAHE(
            clipLimit=self.config.clahe.clip_limit,
            tileGridSize=self.config.clahe.tile_grid_size
        )
        
        # Aplicar CLAHE al canal L
        l_clahe = clahe.apply(l)
        
        # Reconstruir imagen
        if len(image.shape) == 3:
            lab_clahe = cv2.merge([l_clahe, a, b])
            result = cv2.cvtColor(lab_clahe, cv2.COLOR_LAB2BGR)
        else:
            result = l_clahe
        
        return result
    
    def apply_bilateral_filter(self, image: np.ndarray) -> np.ndarray:
        """
        Aplica filtro bilateral para reducción de ruido.
        
        Args:
            image: Imagen de entrada
        
        Returns:
            Imagen filtrada
        """
        logger.debug("Aplicando filtro bilateral")
        
        filtered = cv2.bilateralFilter(
            image,
            d=self.config.bilateral_filter.d,
            sigmaColor=self.config.bilateral_filter.sigma_color,
            sigmaSpace=self.config.bilateral_filter.sigma_space
        )
        
        return filtered
    
    def resize(
        self,
        image: np.ndarray,
        target_size: Optional[Tuple[int, int]] = None
    ) -> np.ndarray:
        """
        Redimensiona la imagen.
        
        Args:
            image: Imagen de entrada
            target_size: Tamaño objetivo. Si None, usa el de config.
        
        Returns:
            Imagen redimensionada
        """
        if target_size is None:
            target_size = self.config.target_size
        
        if target_size is not None:
            logger.debug(f"Redimensionando imagen a {target_size}")
            return resize_image(image, target_size)
        
        return image
    
    def normalize(self, image: np.ndarray) -> np.ndarray:
        """
        Normaliza los valores de la imagen.
        
        Args:
            image: Imagen de entrada
        
        Returns:
            Imagen normalizada [0, 1]
        """
        if not self.config.normalize_values:
            return image
        
        logger.debug("Normalizando imagen")
        return normalize_image(image, method="minmax")
    
    def preprocess(
        self,
        image: np.ndarray,
        apply_resize: bool = True,
        apply_clahe: bool = True,
        apply_filter: bool = True,
        apply_normalize: bool = True
    ) -> np.ndarray:
        """
        Aplica el pipeline completo de preprocesamiento.
        
        Args:
            image: Imagen de entrada
            apply_resize: Si aplicar redimensionamiento
            apply_clahe: Si aplicar corrección de iluminación
            apply_filter: Si aplicar filtrado de ruido
            apply_normalize: Si normalizar valores
        
        Returns:
            Imagen preprocesada
        """
        logger.info("Iniciando preprocesamiento de imagen")
        
        # Asegurar que la imagen es uint8
        processed = image_to_uint8(image)
        
        # 1. Corrección de iluminación
        if apply_clahe:
            processed = self.apply_clahe(processed)
        
        # 2. Filtrado de ruido
        if apply_filter:
            processed = self.apply_bilateral_filter(processed)
        
        # 3. Redimensionamiento
        if apply_resize:
            processed = self.resize(processed)
        
        # 4. Normalización
        if apply_normalize and self.config.normalize_values:
            processed = self.normalize(processed)
            # Convertir de vuelta a uint8 para procesamiento posterior
            processed = image_to_uint8(processed)
        
        logger.info("Preprocesamiento completado")
        return processed


def preprocess_image(
    image: np.ndarray,
    config: Optional[PreprocessingConfig] = None
) -> np.ndarray:
    """
    Función de conveniencia para preprocesar una imagen.
    
    Args:
        image: Imagen de entrada
        config: Configuración de preprocesamiento. Si None, usa default.
    
    Returns:
        Imagen preprocesada
    """
    if config is None:
        config = PreprocessingConfig()
    
    preprocessor = ImagePreprocessor(config)
    return preprocessor.preprocess(image)
