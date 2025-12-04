"""Módulo de estimación de profundidad usando Depth-Anything-V2."""

from typing import Optional, Tuple

import cv2
import numpy as np
import torch
from loguru import logger
from PIL import Image
from transformers import pipeline

from shelf_occupancy.config import DepthEstimationConfig


class DepthEstimator:
    """Estimador de profundidad usando Depth-Anything-V2."""
    
    def __init__(self, config: Optional[DepthEstimationConfig] = None):
        """
        Inicializa el estimador de profundidad.
        
        Args:
            config: Configuración de estimación. Si None, usa valores por defecto.
        """
        if config is None:
            config = DepthEstimationConfig()
        self.config = config
        
        # Determinar device
        if self.config.device == "cuda" and not torch.cuda.is_available():
            logger.warning("CUDA no disponible, usando CPU")
            self.device = "cpu"
        else:
            self.device = self.config.device
        
        logger.info(f"Inicializando DepthEstimator en {self.device}")
        
        # Cargar modelo
        try:
            self.pipe = pipeline(
                task="depth-estimation",
                model=self.config.model_name,
                device=0 if self.device == "cuda" else -1
            )
            logger.info(f"Modelo cargado: {self.config.model_name}")
        except Exception as e:
            logger.error(f"Error cargando modelo: {e}")
            raise
    
    def estimate(
        self,
        image: np.ndarray,
        return_colored: bool = True
    ) -> Tuple[np.ndarray, Optional[np.ndarray]]:
        """
        Estima profundidad de una imagen.
        
        Args:
            image: Imagen de entrada (BGR o RGB)
            return_colored: Si retornar también mapa de color
        
        Returns:
            Tupla (depth_map, depth_colored) donde:
            - depth_map: Mapa de profundidad normalizado [0, 1]
            - depth_colored: Mapa de profundidad colorizado (opcional)
        """
        logger.debug(f"Estimando profundidad de imagen {image.shape}")
        
        # Convertir a RGB si es necesario
        if len(image.shape) == 3 and image.shape[2] == 3:
            # Asumimos BGR, convertir a RGB
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Convertir a PIL Image
        pil_image = Image.fromarray(rgb_image)
        original_size = pil_image.size
        
        # Estimar profundidad
        try:
            result = self.pipe(pil_image)
            depth_pil = result["depth"]
        except Exception as e:
            logger.error(f"Error en estimación de profundidad: {e}")
            raise
        
        # Convertir a numpy
        depth_map = np.array(depth_pil)
        
        # Redimensionar al tamaño original si es necesario
        if self.config.postprocessing.resize_to_original:
            if depth_map.shape[:2] != image.shape[:2]:
                depth_map = cv2.resize(
                    depth_map,
                    (image.shape[1], image.shape[0]),
                    interpolation=cv2.INTER_LINEAR
                )
        
        # Normalizar
        if self.config.normalize_depth:
            depth_map = self._normalize_depth(depth_map)
        
        # Aplicar post-procesamiento
        depth_map = self._postprocess(depth_map)
        
        # Crear versión coloreada si se solicita
        depth_colored = None
        if return_colored:
            depth_colored = self._colorize_depth(depth_map)
        
        logger.debug(f"Profundidad estimada: {depth_map.shape}, rango [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
        return depth_map, depth_colored
    
    def _normalize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Normaliza el mapa de profundidad a [0, 1].
        
        Args:
            depth_map: Mapa de profundidad crudo
        
        Returns:
            Mapa de profundidad normalizado
        """
        min_val = depth_map.min()
        max_val = depth_map.max()
        
        if max_val - min_val > 0:
            normalized = (depth_map - min_val) / (max_val - min_val)
        else:
            normalized = depth_map
        
        return normalized
    
    def _postprocess(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Aplica post-procesamiento al mapa de profundidad.
        
        Args:
            depth_map: Mapa de profundidad
        
        Returns:
            Mapa procesado
        """
        # Aplicar filtro bilateral para suavizar manteniendo bordes
        if depth_map.dtype != np.float32:
            depth_float = depth_map.astype(np.float32)
        else:
            depth_float = depth_map
        
        processed = cv2.bilateralFilter(
            depth_float,
            d=self.config.postprocessing.bilateral_filter.d,
            sigmaColor=self.config.postprocessing.bilateral_filter.sigma_color,
            sigmaSpace=self.config.postprocessing.bilateral_filter.sigma_space
        )
        
        return processed
    
    def _colorize_depth(self, depth_map: np.ndarray) -> np.ndarray:
        """
        Crea una visualización coloreada del mapa de profundidad.
        
        Args:
            depth_map: Mapa de profundidad normalizado [0, 1]
        
        Returns:
            Mapa coloreado (RGB)
        """
        # Convertir a uint8
        depth_uint8 = (depth_map * 255).astype(np.uint8)
        
        # Aplicar colormap
        colored = cv2.applyColorMap(depth_uint8, cv2.COLORMAP_VIRIDIS)
        
        # Convertir BGR a RGB
        colored_rgb = cv2.cvtColor(colored, cv2.COLOR_BGR2RGB)
        
        return colored_rgb
    
    def estimate_batch(
        self,
        images: list,
        return_colored: bool = False
    ) -> list:
        """
        Estima profundidad para un batch de imágenes.
        
        Args:
            images: Lista de imágenes
            return_colored: Si retornar mapas coloreados
        
        Returns:
            Lista de tuplas (depth_map, depth_colored)
        """
        results = []
        
        for i, image in enumerate(images):
            logger.info(f"Procesando imagen {i+1}/{len(images)}")
            depth_map, depth_colored = self.estimate(image, return_colored)
            results.append((depth_map, depth_colored))
        
        return results


def estimate_depth(
    image: np.ndarray,
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf",
    device: str = "cpu"
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Función de conveniencia para estimar profundidad.
    
    Args:
        image: Imagen de entrada
        model_name: Nombre del modelo
        device: Device a usar
    
    Returns:
        Tupla (depth_map, depth_colored)
    """
    config = DepthEstimationConfig(model_name=model_name, device=device)
    estimator = DepthEstimator(config)
    return estimator.estimate(image, return_colored=True)
