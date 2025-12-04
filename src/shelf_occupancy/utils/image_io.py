"""Utilidades para manejo de imágenes."""

from pathlib import Path
from typing import Optional, Tuple, Union

import cv2
import numpy as np
from PIL import Image


def load_image(path: Union[str, Path], color_mode: str = "BGR") -> np.ndarray:
    """
    Carga una imagen desde archivo.
    
    Args:
        path: Ruta al archivo de imagen
        color_mode: Modo de color ('BGR', 'RGB', 'GRAY')
    
    Returns:
        Imagen como array de numpy
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"No se encontró la imagen: {path}")
    
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {path}")
    
    if color_mode == "RGB":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    elif color_mode == "GRAY":
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img


def save_image(
    image: np.ndarray,
    path: Union[str, Path],
    color_mode: str = "BGR",
    create_dirs: bool = True
) -> None:
    """
    Guarda una imagen a archivo.
    
    Args:
        image: Imagen como array de numpy
        path: Ruta donde guardar la imagen
        color_mode: Modo de color de la imagen ('BGR', 'RGB', 'GRAY')
        create_dirs: Si crear directorios si no existen
    """
    path = Path(path)
    
    if create_dirs:
        path.parent.mkdir(parents=True, exist_ok=True)
    
    # Convertir a BGR si es necesario (OpenCV guarda en BGR)
    if color_mode == "RGB" and len(image.shape) == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    success = cv2.imwrite(str(path), image)
    if not success:
        raise ValueError(f"No se pudo guardar la imagen: {path}")


def resize_image(
    image: np.ndarray,
    target_size: Optional[Tuple[int, int]] = None,
    scale_factor: Optional[float] = None,
    interpolation: int = cv2.INTER_LINEAR
) -> np.ndarray:
    """
    Redimensiona una imagen.
    
    Args:
        image: Imagen a redimensionar
        target_size: Tamaño objetivo (width, height)
        scale_factor: Factor de escala (alternativa a target_size)
        interpolation: Método de interpolación
    
    Returns:
        Imagen redimensionada
    """
    if target_size is not None:
        return cv2.resize(image, target_size, interpolation=interpolation)
    elif scale_factor is not None:
        new_size = (
            int(image.shape[1] * scale_factor),
            int(image.shape[0] * scale_factor)
        )
        return cv2.resize(image, new_size, interpolation=interpolation)
    else:
        return image


def normalize_image(image: np.ndarray, method: str = "minmax") -> np.ndarray:
    """
    Normaliza una imagen.
    
    Args:
        image: Imagen a normalizar
        method: Método de normalización ('minmax', 'zscore')
    
    Returns:
        Imagen normalizada
    """
    if method == "minmax":
        min_val = image.min()
        max_val = image.max()
        if max_val - min_val > 0:
            return (image - min_val) / (max_val - min_val)
        else:
            return image
    elif method == "zscore":
        mean = image.mean()
        std = image.std()
        if std > 0:
            return (image - mean) / std
        else:
            return image - mean
    else:
        raise ValueError(f"Método de normalización desconocido: {method}")


def image_to_uint8(image: np.ndarray) -> np.ndarray:
    """
    Convierte imagen a uint8.
    
    Args:
        image: Imagen a convertir
    
    Returns:
        Imagen en formato uint8
    """
    if image.dtype == np.uint8:
        return image
    
    # Si está en [0, 1], escalar a [0, 255]
    if image.max() <= 1.0:
        image = image * 255
    
    return image.astype(np.uint8)


def get_image_dimensions(image: np.ndarray) -> Tuple[int, int, int]:
    """
    Obtiene las dimensiones de una imagen.
    
    Args:
        image: Imagen
    
    Returns:
        Tupla (height, width, channels). Si es escala de grises, channels=1
    """
    if len(image.shape) == 2:
        h, w = image.shape
        return h, w, 1
    else:
        h, w, c = image.shape
        return h, w, c


def ensure_rgb(image: np.ndarray) -> np.ndarray:
    """
    Asegura que la imagen esté en formato RGB.
    
    Args:
        image: Imagen en cualquier formato
    
    Returns:
        Imagen en formato RGB
    """
    if len(image.shape) == 2:
        # Escala de grises -> RGB
        return cv2.cvtColor(image, cv2.COLOR_GRAY2RGB)
    elif image.shape[2] == 4:
        # RGBA -> RGB
        return cv2.cvtColor(image, cv2.COLOR_RGBA2RGB)
    else:
        return image
