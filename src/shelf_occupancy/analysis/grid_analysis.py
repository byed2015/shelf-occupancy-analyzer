"""Módulo de análisis de ocupación por cuadrículas con refinamiento integrado."""

from typing import List, Optional, Tuple

import cv2
import numpy as np
from loguru import logger

from shelf_occupancy.config import OccupancyAnalysisConfig
from shelf_occupancy.utils.geometry import BoundingBox, create_grid


class GridAnalyzer:
    """Analiza ocupación de anaqueles usando cuadrículas con refinamiento automático."""
    
    def __init__(self, config: Optional[OccupancyAnalysisConfig] = None, enable_refinement: bool = True):
        """
        Inicializa el analizador de cuadrículas.
        
        Args:
            config: Configuración de análisis. Si None, usa valores por defecto.
            enable_refinement: Si aplicar refinamiento para reducir falsos positivos.
        """
        if config is None:
            config = OccupancyAnalysisConfig()
        self.config = config
        self.enable_refinement = enable_refinement
        logger.info(f"GridAnalyzer inicializado (refinamiento: {enable_refinement})")
    
    def analyze_shelf(
        self,
        depth_map: np.ndarray,
        shelf_bbox: BoundingBox,
        image: Optional[np.ndarray] = None,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> Tuple[np.ndarray, float, dict]:
        """
        Analiza ocupación de un anaquel usando cuadrículas.
        
        Args:
            depth_map: Mapa de profundidad [0, 1]
            shelf_bbox: Bounding box del anaquel
            grid_size: Tamaño de cuadrícula (cols, rows). Si None, usa config.
        
        Returns:
            Tupla (occupancy_grid, occupancy_percentage, stats) donde:
            - occupancy_grid: Matriz de ocupación [0, 1] por celda
            - occupancy_percentage: Porcentaje global de ocupación
            - stats: Diccionario con estadísticas
        """
        if grid_size is None:
            grid_size = self.config.grid_size
        
        logger.debug(f"Analizando anaquel {shelf_bbox} con grid {grid_size}")
        
        # Extraer región del anaquel del mapa de profundidad
        shelf_depth = depth_map[
            shelf_bbox.y:shelf_bbox.y + shelf_bbox.height,
            shelf_bbox.x:shelf_bbox.x + shelf_bbox.width
        ]
        
        # Crear cuadrícula
        grid = create_grid(
            BoundingBox(0, 0, shelf_depth.shape[1], shelf_depth.shape[0]),
            grid_size
        )
        
        # Calcular ocupación por celda
        rows, cols = grid_size[1], grid_size[0]
        occupancy_grid = np.zeros((rows, cols), dtype=np.float32)
        
        for i in range(rows):
            for j in range(cols):
                cell = grid[i][j]
                
                # Extraer profundidades de la celda
                cell_depth = shelf_depth[
                    cell.y:cell.y + cell.height,
                    cell.x:cell.x + cell.width
                ]
                
                # Calcular ocupación de la celda
                if cell_depth.size > 0:
                    occupancy = self._calculate_cell_occupancy(cell_depth)
                    occupancy_grid[i, j] = occupancy
        
        # Aplicar refinamiento si está habilitado
        if self.enable_refinement and image is not None:
            occupancy_grid = self._apply_refinement(
                occupancy_grid, depth_map, image, shelf_bbox
            )
        
        # Calcular porcentaje global
        occupancy_percentage = np.mean(occupancy_grid) * 100
        
        # Calcular estadísticas
        stats = {
            'mean_occupancy': float(np.mean(occupancy_grid)),
            'median_occupancy': float(np.median(occupancy_grid)),
            'std_occupancy': float(np.std(occupancy_grid)),
            'min_occupancy': float(np.min(occupancy_grid)),
            'max_occupancy': float(np.max(occupancy_grid)),
            'occupied_cells': int(np.sum(occupancy_grid > self.config.thresholds.min_occupancy)),
            'total_cells': int(rows * cols),
            'grid_shape': (rows, cols)
        }
        
        logger.debug(f"Ocupación del anaquel: {occupancy_percentage:.1f}%")
        
        return occupancy_grid, occupancy_percentage, stats
    
    def _calculate_cell_occupancy(self, cell_depth: np.ndarray) -> float:
        """
        Calcula ocupación de una celda basándose en su profundidad.
        
        LÓGICA MEJORADA v1.3.1:
        - Depth-Anything retorna: 0 = cerca (productos), 1 = lejos (vacío)
        - Calculamos mediana de profundidad del cuadrilátero
        - Ocupación = 1 - mediana (invertir para que cerca = ocupado)
        - Validamos con varianza (productos tienen más textura)
        
        Args:
            cell_depth: Valores de profundidad de la celda [0, 1]
        
        Returns:
            Valor de ocupación [0, 1]
        """
        if cell_depth.size == 0:
            return 0.0
        
        # Calcular mediana de profundidad (más robusto que media)
        median_depth = np.median(cell_depth)
        
        # Invertir: profundidad baja (0) = cerca = ocupado (1)
        #          profundidad alta (1) = lejos = vacío (0)
        occupancy_base = 1.0 - median_depth
        
        # Validar con varianza: productos tienen más textura que fondo vacío
        std_depth = np.std(cell_depth)
        
        # Si la varianza es muy baja, puede ser fondo uniforme
        # Penalizar si std < 0.05 (muy uniforme)
        if std_depth < 0.05:
            # Puede ser fondo vacío uniforme, reducir ocupación
            variance_penalty = std_depth / 0.05  # [0, 1]
            occupancy_base *= (0.5 + 0.5 * variance_penalty)
        
        # Clamp a [0, 1]
        occupancy = np.clip(occupancy_base, 0.0, 1.0)
        
        return float(occupancy)
    
    def _apply_refinement(
        self,
        occupancy_grid: np.ndarray,
        depth_map: np.ndarray,
        image: np.ndarray,
        shelf_bbox: BoundingBox
    ) -> np.ndarray:
        """
        Aplica refinamiento a la cuadrícula de ocupación.
        
        Reduce falsos positivos detectando:
        - Regiones de fondo (sin productos)
        - Áreas de baja textura (espacios vacíos)
        - Márgenes con ruido estructural
        
        Args:
            occupancy_grid: Cuadrícula de ocupación original
            depth_map: Mapa de profundidad completo
            image: Imagen original
            shelf_bbox: Región del anaquel
        
        Returns:
            Cuadrícula de ocupación refinada
        """
        # Extraer región del anaquel
        shelf_depth = depth_map[
            shelf_bbox.y:shelf_bbox.y + shelf_bbox.height,
            shelf_bbox.x:shelf_bbox.x + shelf_bbox.width
        ]
        
        shelf_img = image[
            shelf_bbox.y:shelf_bbox.y + shelf_bbox.height,
            shelf_bbox.x:shelf_bbox.x + shelf_bbox.width
        ]
        
        # 1. Detectar fondo (áreas con profundidad alta = vacías)
        depth_75 = np.percentile(shelf_depth, 75)
        depth_90 = np.percentile(shelf_depth, 90)
        threshold = depth_75 if (depth_90 - depth_75) > 0.2 else 0.75
        background_mask = shelf_depth > threshold
        
        # Limpiar con morfología
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
        background_mask = cv2.morphologyEx(
            background_mask.astype(np.uint8),
            cv2.MORPH_CLOSE,
            kernel
        ).astype(bool)
        
        # 2. Detectar baja textura
        if len(shelf_img.shape) == 3:
            gray = cv2.cvtColor(shelf_img, cv2.COLOR_BGR2GRAY)
        else:
            gray = shelf_img
        
        # Varianza local (indicador de textura)
        cell_size = 20
        variance_map = np.zeros_like(gray, dtype=np.float32)
        h, w = gray.shape
        for y in range(0, h, cell_size // 2):
            for x in range(0, w, cell_size // 2):
                y_end = min(y + cell_size, h)
                x_end = min(x + cell_size, w)
                cell = gray[y:y_end, x:x_end]
                variance = np.var(cell) / 255.0
                variance_map[y:y_end, x:x_end] = variance
        
        if variance_map.max() > 0:
            variance_map = variance_map / variance_map.max()
        
        low_texture_mask = variance_map < 0.01
        
        # 3. Máscara de región válida (excluir márgenes)
        margin = 10
        valid_mask = np.ones_like(background_mask, dtype=bool)
        valid_mask[:margin, :] = False
        valid_mask[-margin:, :] = False
        valid_mask[:, :margin] = False
        valid_mask[:, -margin:] = False
        
        # Redimensionar máscaras al tamaño del grid
        grid_h, grid_w = occupancy_grid.shape
        background_resized = cv2.resize(
            background_mask.astype(np.float32),
            (grid_w, grid_h),
            interpolation=cv2.INTER_NEAREST
        )
        low_texture_resized = cv2.resize(
            low_texture_mask.astype(np.float32),
            (grid_w, grid_h),
            interpolation=cv2.INTER_NEAREST
        )
        valid_resized = cv2.resize(
            valid_mask.astype(np.float32),
            (grid_w, grid_h),
            interpolation=cv2.INTER_NEAREST
        )
        
        # Aplicar refinamientos
        occupancy_refined = occupancy_grid.copy()
        
        # Penalizar fondo (30%)
        occupancy_refined = np.where(
            background_resized > 0.5,
            occupancy_refined * 0.7,
            occupancy_refined
        )
        
        # Penalizar baja textura (50%)
        occupancy_refined = np.where(
            low_texture_resized > 0.5,
            occupancy_refined * 0.5,
            occupancy_refined
        )
        
        # Invalidar márgenes
        occupancy_refined = np.where(
            valid_resized > 0.5,
            occupancy_refined,
            0.0
        )
        
        return occupancy_refined
    
    def analyze_multiple_shelves(
        self,
        depth_map: np.ndarray,
        shelves: List[BoundingBox],
        image: Optional[np.ndarray] = None,
        grid_size: Optional[Tuple[int, int]] = None
    ) -> List[Tuple[np.ndarray, float, dict]]:
        """
        Analiza ocupación de múltiples anaqueles.
        
        Args:
            depth_map: Mapa de profundidad completo
            shelves: Lista de bounding boxes de anaqueles
            image: Imagen original (necesaria para refinamiento)
            grid_size: Tamaño de cuadrícula
        
        Returns:
            Lista de tuplas (occupancy_grid, percentage, stats) por anaquel
        """
        results = []
        
        for i, shelf in enumerate(shelves):
            logger.info(f"Analizando anaquel {i+1}/{len(shelves)}")
            result = self.analyze_shelf(depth_map, shelf, image, grid_size)
            results.append(result)
        
        return results
    
    def classify_occupancy_level(self, occupancy: float) -> str:
        """
        Clasifica nivel de ocupación.
        
        Args:
            occupancy: Porcentaje de ocupación [0, 100]
        
        Returns:
            Clasificación: 'high', 'medium', 'low'
        """
        if occupancy >= 70:
            return 'high'
        elif occupancy >= 40:
            return 'medium'
        else:
            return 'low'


def analyze_occupancy(
    depth_map: np.ndarray,
    shelves: List[BoundingBox],
    grid_size: Tuple[int, int] = (10, 5)
) -> List[Tuple[np.ndarray, float, dict]]:
    """
    Función de conveniencia para analizar ocupación.
    
    Args:
        depth_map: Mapa de profundidad
        shelves: Lista de anaqueles
        grid_size: Tamaño de cuadrícula
    
    Returns:
        Resultados de análisis por anaquel
    """
    analyzer = GridAnalyzer()
    return analyzer.analyze_multiple_shelves(depth_map, shelves, grid_size)
