"""Módulo de detección de anaqueles."""

from typing import List, Optional, Tuple, Union

import numpy as np
from loguru import logger
from sklearn.cluster import DBSCAN

from shelf_occupancy.config import ClusteringConfig, ShelfDetectionConfig
from shelf_occupancy.detection.lines import Line
from shelf_occupancy.utils.geometry import BoundingBox, Quadrilateral


class ShelfDetector:
    """Detector de anaqueles basado en líneas."""
    
    def __init__(self, config: Optional[ShelfDetectionConfig] = None):
        """
        Inicializa el detector de anaqueles.
        
        Args:
            config: Configuración de detección. Si None, usa valores por defecto.
        """
        if config is None:
            config = ShelfDetectionConfig()
        self.config = config
        logger.info("ShelfDetector inicializado")
    
    def detect_from_lines(
        self,
        horizontal_lines: List[Line],
        vertical_lines: List[Line],
        image_shape: Tuple[int, int],
        use_quadrilaterals: bool = True
    ) -> Union[List[BoundingBox], List[Quadrilateral]]:
        """
        Detecta anaqueles a partir de líneas horizontales y verticales.
        
        Args:
            horizontal_lines: Lista de líneas horizontales (pueden estar inclinadas)
            vertical_lines: Lista de líneas verticales
            image_shape: Forma de la imagen (height, width)
            use_quadrilaterals: Si True, retorna Quadrilaterals (soporta perspectiva)
                              Si False, retorna BoundingBoxes rectangulares
        
        Returns:
            Lista de anaqueles (Quadrilateral o BoundingBox)
        """
        if not horizontal_lines:
            logger.warning("No hay líneas horizontales para detectar anaqueles")
            return []
        
        height, width = image_shape
        
        if use_quadrilaterals:
            # Detectar anaqueles como cuadriláteros inclinados
            shelves = self._cluster_lines_to_quadrilaterals(horizontal_lines, height, width)
        else:
            # Detectar anaqueles como bounding boxes rectangulares (modo antiguo)
            shelves = self._cluster_horizontal_lines(horizontal_lines, height, width)
        
        logger.info(f"Detectados {len(shelves)} anaqueles potenciales")
        
        # Filtrar anaqueles muy pequeños
        if use_quadrilaterals:
            min_area = 50 * width * 0.5  # área mínima basada en imagen
            shelves = [s for s in shelves if s.area >= min_area]
        else:
            min_height = 50
            shelves = [s for s in shelves if s.height >= min_height]
        
        logger.info(f"Filtrados {len(shelves)} anaqueles válidos")
        
        return shelves
    
    def _cluster_lines_to_quadrilaterals(
        self,
        lines: List[Line],
        image_height: int,
        image_width: int
    ) -> List[Quadrilateral]:
        """
        Agrupa líneas y crea cuadriláteros inclinados para anaqueles con perspectiva.
        
        Este método NO corrige la perspectiva. En su lugar, crea anaqueles que
        SIGUEN las líneas inclinadas naturales, permitiendo análisis correcto
        de imágenes con perspectiva lateral.
        
        Args:
            lines: Lista de líneas (pueden estar inclinadas)
            image_height: Alto de la imagen
            image_width: Ancho de la imagen
        
        Returns:
            Lista de anaqueles como Quadrilaterals
        """
        if not lines:
            return []
        
        # Ordenar líneas por posición Y promedio
        lines_with_y = [(line, (line.y1 + line.y2) / 2) for line in lines]
        lines_with_y.sort(key=lambda x: x[1])
        
        # Extraer solo las líneas ordenadas
        sorted_lines = [line for line, _ in lines_with_y]
        
        # Agrupar líneas cercanas usando clustering
        y_positions = np.array([y for _, y in lines_with_y]).reshape(-1, 1)
        
        clustering = DBSCAN(
            eps=self.config.clustering.distance_tolerance * 1.5,  # Más tolerante para perspectiva
            min_samples=1  # Permitir anaqueles con una sola línea
        )
        labels = clustering.fit_predict(y_positions)
        
        # Agrupar líneas por cluster
        clusters = {}
        for line, label in zip(sorted_lines, labels):
            if label == -1:  # Ruido
                continue
            if label not in clusters:
                clusters[label] = []
            clusters[label].append(line)
        
        # Obtener línea representativa de cada cluster (promedio)
        cluster_lines = []
        for label in sorted(clusters.keys()):
            cluster = clusters[label]
            
            # Promediar líneas del cluster
            avg_x1 = np.mean([l.x1 for l in cluster])
            avg_y1 = np.mean([l.y1 for l in cluster])
            avg_x2 = np.mean([l.x2 for l in cluster])
            avg_y2 = np.mean([l.y2 for l in cluster])
            
            cluster_lines.append((avg_x1, avg_y1, avg_x2, avg_y2))
        
        # Crear cuadriláteros entre pares de líneas consecutivas
        quadrilaterals = []
        
        if len(cluster_lines) == 0:
            # Sin líneas, usar toda la imagen como un anaquel
            quad = Quadrilateral.from_bbox(BoundingBox(0, 0, image_width, image_height))
            quadrilaterals.append(quad)
        
        elif len(cluster_lines) == 1:
            # Una sola línea, crear anaquel arriba y abajo
            line = cluster_lines[0]
            y_mid = (line[1] + line[3]) / 2
            
            if y_mid > image_height / 2:
                # Línea en parte inferior, anaquel arriba
                top_line = (0, 0, image_width, 0)
                quadrilaterals.append(Quadrilateral.from_lines(top_line, line, image_width))
            else:
                # Línea en parte superior, anaquel abajo
                bottom_line = (0, image_height, image_width, image_height)
                quadrilaterals.append(Quadrilateral.from_lines(line, bottom_line, image_width))
        
        else:
            # Múltiples líneas, crear anaqueles entre líneas consecutivas
            for i in range(len(cluster_lines) - 1):
                top_line = cluster_lines[i]
                bottom_line = cluster_lines[i + 1]
                
                quad = Quadrilateral.from_lines(top_line, bottom_line, image_width)
                quadrilaterals.append(quad)
            
            # Añadir anaquel superior si hay espacio
            first_line = cluster_lines[0]
            y_first = (first_line[1] + first_line[3]) / 2
            if y_first > 50:
                top_line = (0, 0, image_width, 0)
                quad = Quadrilateral.from_lines(top_line, first_line, image_width)
                quadrilaterals.insert(0, quad)
            
            # Añadir anaquel inferior si hay espacio
            last_line = cluster_lines[-1]
            y_last = (last_line[1] + last_line[3]) / 2
            if y_last < image_height - 50:
                bottom_line = (0, image_height, image_width, image_height)
                quad = Quadrilateral.from_lines(last_line, bottom_line, image_width)
                quadrilaterals.append(quad)
        
        logger.info(f"Creados {len(quadrilaterals)} cuadriláteros inclinados")
        return quadrilaterals
    
    def _cluster_horizontal_lines(
        self,
        lines: List[Line],
        image_height: int,
        image_width: int
    ) -> List[BoundingBox]:
        """
        Agrupa líneas horizontales en anaqueles con soporte de perspectiva.
        
        Para imágenes con perspectiva, las líneas pueden estar inclinadas.
        Este método agrupa líneas por su posición Y promedio, permitiendo
        variación en X debido a la perspectiva.
        
        Args:
            lines: Lista de líneas horizontales (pueden estar inclinadas)
            image_height: Alto de la imagen
            image_width: Ancho de la imagen
        
        Returns:
            Lista de anaqueles como BoundingBoxes
        """
        if not lines:
            return []
        
        # Para cada línea, calcular Y promedio y detectar inclinación
        line_data = []
        for line in lines:
            y_avg = (line.y1 + line.y2) / 2
            x_avg = (line.x1 + line.x2) / 2
            angle = line.angle if line.angle is not None else 0
            line_data.append({
                'y_avg': y_avg,
                'x_avg': x_avg,
                'angle': angle,
                'line': line
            })
        
        # Detectar si hay perspectiva significativa (líneas muy inclinadas)
        angles = [abs(d['angle']) for d in line_data]
        max_angle = max(angles) if angles else 0
        has_perspective = max_angle > 5  # Más de 5 grados = perspectiva
        
        if has_perspective:
            logger.info(f"Perspectiva detectada (ángulo máximo: {max_angle:.2f}°)")
        
        # Extraer posiciones Y  para clustering
        y_positions = np.array([d['y_avg'] for d in line_data]).reshape(-1, 1)
        
        # Clustering con DBSCAN
        clustering = DBSCAN(
            eps=self.config.clustering.distance_tolerance,
            min_samples=self.config.clustering.min_lines_per_shelf
        )
        labels = clustering.fit_predict(y_positions)
        
        # Crear anaqueles a partir de clusters
        shelves = []
        unique_labels = set(labels)
        
        for label in unique_labels:
            if label == -1:  # Ruido
                continue
            
            # Obtener líneas del cluster
            cluster_mask = labels == label
            cluster_y_positions = y_positions[cluster_mask]
            
            # Calcular límites del anaquel
            y_center = np.mean(cluster_y_positions)
            
            shelves.append(y_center)
        
        # Ordenar anaqueles de arriba a abajo
        shelves.sort()
        
        # Crear BoundingBoxes
        bboxes = []
        
        if len(shelves) == 0:
            # Si no se detectaron anaqueles, usar toda la imagen
            bboxes.append(BoundingBox(0, 0, image_width, image_height))
        elif len(shelves) == 1:
            # Un solo anaquel, usar toda la imagen
            bboxes.append(BoundingBox(0, 0, image_width, image_height))
        else:
            # Múltiples anaqueles, dividir por las líneas detectadas
            for i in range(len(shelves) - 1):
                y1 = int(shelves[i])
                y2 = int(shelves[i + 1])
                h = y2 - y1
                
                bboxes.append(BoundingBox(0, y1, image_width, h))
            
            # Agregar anaquel superior e inferior
            if shelves[0] > 50:  # Si hay espacio arriba
                bboxes.insert(0, BoundingBox(0, 0, image_width, int(shelves[0])))
            
            if shelves[-1] < image_height - 50:  # Si hay espacio abajo
                y_start = int(shelves[-1])
                h = image_height - y_start
                bboxes.append(BoundingBox(0, y_start, image_width, h))
        
        return bboxes
    
    def detect_simple_grid(
        self,
        image_shape: Tuple[int, int],
        n_rows: int = 4,
        n_cols: int = 1
    ) -> List[BoundingBox]:
        """
        Crea una cuadrícula simple de anaqueles cuando no se detectan líneas.
        
        Args:
            image_shape: Forma de la imagen (height, width)
            n_rows: Número de filas (anaqueles)
            n_cols: Número de columnas
        
        Returns:
            Lista de bounding boxes en cuadrícula
        """
        height, width = image_shape
        
        row_height = height // n_rows
        col_width = width // n_cols
        
        bboxes = []
        
        for row in range(n_rows):
            for col in range(n_cols):
                x = col * col_width
                y = row * row_height
                w = col_width
                h = row_height
                
                bboxes.append(BoundingBox(x, y, w, h))
        
        logger.info(f"Creada cuadrícula simple: {n_rows}x{n_cols} = {len(bboxes)} anaqueles")
        
        return bboxes


def detect_shelves_from_lines(
    horizontal_lines: List[Line],
    vertical_lines: List[Line],
    image_shape: Tuple[int, int],
    config: Optional[ShelfDetectionConfig] = None
) -> List[BoundingBox]:
    """
    Función de conveniencia para detectar anaqueles.
    
    Args:
        horizontal_lines: Líneas horizontales
        vertical_lines: Líneas verticales
        image_shape: Forma de la imagen
        config: Configuración opcional
    
    Returns:
        Lista de anaqueles detectados
    """
    detector = ShelfDetector(config)
    return detector.detect_from_lines(horizontal_lines, vertical_lines, image_shape)
