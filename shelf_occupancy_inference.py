"""
Interfaz de inferencia simplificada para integración con Streamlit.

Este módulo proporciona una API limpia para procesar imágenes de anaqueles
sin necesidad de conocer los detalles internos del pipeline.

Uso:
    from shelf_occupancy_inference import ShelfOccupancyAnalyzer
    
    analyzer = ShelfOccupancyAnalyzer()
    results = analyzer.process("imagen.jpg")
    
    print(f"Ocupación promedio: {results['avg_occupancy']:.1f}%")
"""

from pathlib import Path
from typing import Dict, List, Tuple, Union, Any
import numpy as np
import cv2
from loguru import logger

from shelf_occupancy.config import load_config, AppConfig
from shelf_occupancy.utils.image_io import load_image
from shelf_occupancy.preprocessing import ImageProcessor
from shelf_occupancy.detection import EdgeDetector, LineDetector, ShelfDetector
from shelf_occupancy.depth import DepthEstimator
from shelf_occupancy.analysis import GridAnalyzer
from shelf_occupancy.visualization import OccupancyVisualizer


class ShelfOccupancyAnalyzer:
    """
    Analizador de ocupación de anaqueles - Interfaz simplificada.
    
    Encapsula todo el pipeline de procesamiento en una API simple
    lista para integrar con Streamlit o cualquier otro frontend.
    """
    
    def __init__(self, config_path: Union[str, Path, None] = None):
        """
        Inicializa el analizador.
        
        Args:
            config_path: Ruta al archivo config.yaml. Si es None, usa configuración por defecto.
        """
        logger.info("Inicializando ShelfOccupancyAnalyzer")
        
        # Cargar configuración
        self.config = load_config(config_path) if config_path else load_config()
        
        # Inicializar componentes del pipeline
        self.preprocessor = ImageProcessor(self.config.preprocessing)
        self.edge_detector = EdgeDetector(self.config.shelf_detection.canny)
        self.line_detector = LineDetector(self.config.shelf_detection.hough)
        self.shelf_detector = ShelfDetector(self.config.shelf_detection)
        self.depth_estimator = DepthEstimator(self.config.depth_estimation)
        self.grid_analyzer = GridAnalyzer(self.config.occupancy_analysis)
        self.visualizer = OccupancyVisualizer(self.config.visualization)
        
        logger.success("ShelfOccupancyAnalyzer inicializado correctamente")
    
    def process(
        self, 
        image_input: Union[str, Path, np.ndarray],
        return_visualizations: bool = True,
        return_steps: bool = False
    ) -> Dict[str, Any]:
        """
        Procesa una imagen de anaquel y retorna resultados completos.
        
        Args:
            image_input: Ruta a imagen o array numpy (RGB)
            return_visualizations: Si True, incluye imágenes visualizadas en resultados
            return_steps: Si True, incluye pasos intermedios del pipeline
        
        Returns:
            Diccionario con:
                - avg_occupancy: float (porcentaje promedio)
                - num_shelves: int (número de anaqueles detectados)
                - shelves: List[Dict] (métricas por anaquel)
                - pipeline_image: np.ndarray (visualización completa, si return_visualizations=True)
                - overlay_image: np.ndarray (overlay con ocupación, si return_visualizations=True)
                - steps: Dict (pasos intermedios, si return_steps=True)
        """
        logger.info("Iniciando procesamiento de imagen")
        
        # Cargar imagen
        if isinstance(image_input, (str, Path)):
            image = load_image(str(image_input))
            image_name = Path(image_input).stem
        else:
            image = image_input
            image_name = "uploaded_image"
        
        logger.info(f"Imagen cargada: {image.shape}")
        
        # Pipeline paso a paso
        steps = {}
        
        # Paso 1: Conversión a escala de grises (preprocesamiento simplificado)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        gray_smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
        processed = cv2.cvtColor(gray_smooth, cv2.COLOR_GRAY2BGR)
        
        if return_steps:
            steps['preprocessed'] = processed.copy()
        
        # Paso 2: Detección de bordes (Canny con auto-threshold)
        median_val = np.median(gray_smooth)
        lower = int(max(0, 0.66 * median_val))
        upper = int(min(255, 1.33 * median_val))
        edges = cv2.Canny(gray_smooth, lower, upper, apertureSize=3)
        
        if return_steps:
            steps['edges'] = edges.copy()
        
        # Paso 3: Detección de líneas
        all_lines = self.line_detector.detect(edges, use_polar=False)
        
        # Filtrado ABSOLUTO (no adaptativo)
        h_lines = self.line_detector.filter_by_orientation(
            all_lines, "horizontal", tolerance=20, adaptive=False
        )
        v_lines = self.line_detector.filter_by_orientation(
            all_lines, "vertical", tolerance=20, adaptive=False
        )
        
        # Fusionar líneas similares
        h_lines = self.line_detector.merge_similar_lines(
            h_lines, angle_threshold=5, distance_threshold=30
        )
        v_lines = self.line_detector.merge_similar_lines(
            v_lines, angle_threshold=5, distance_threshold=30
        )
        
        logger.info(f"Líneas detectadas: {len(h_lines)} H, {len(v_lines)} V")
        
        if return_steps:
            lines_img = image.copy()  # Usar imagen original, no procesada
            for line in h_lines:
                cv2.line(lines_img, (int(line.x1), int(line.y1)), 
                        (int(line.x2), int(line.y2)), (0, 255, 0), 2)
            for line in v_lines:
                cv2.line(lines_img, (int(line.x1), int(line.y1)), 
                        (int(line.x2), int(line.y2)), (255, 0, 0), 2)
            steps['lines'] = lines_img
        
        # Paso 4: Detección de anaqueles (cuadriláteros)
        shelves = self.shelf_detector.detect_from_lines(
            h_lines, v_lines, image.shape[:2], use_quadrilaterals=True  # Usar shape de original
        )
        
        if not shelves:
            logger.warning("No se detectaron anaqueles, usando grid simple")
            shelves = self.shelf_detector.detect_simple_grid(image.shape[:2], n_rows=4)  # Usar original
        
        logger.info(f"Anaqueles detectados: {len(shelves)}")
        
        if return_steps:
            shelves_img = image.copy()  # Usar original
            for shelf in shelves:
                if hasattr(shelf, 'get_corners'):
                    corners = shelf.get_corners()
                    pts = np.array([
                        [int(corners[0][0]), int(corners[0][1])],
                        [int(corners[1][0]), int(corners[1][1])],
                        [int(corners[2][0]), int(corners[2][1])],
                        [int(corners[3][0]), int(corners[3][1])]
                    ], dtype=np.int32)
                    cv2.polylines(shelves_img, [pts], True, (0, 255, 255), 3)
            steps['shelves'] = shelves_img
        
        # Paso 5: Estimación de profundidad
        depth_map, depth_colored = self.depth_estimator.estimate(image)
        logger.info(f"Profundidad estimada: rango [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
        if return_steps:
            steps['depth'] = depth_colored
        
        # Paso 6: Análisis de ocupación
        occupancy_percentages = []
        shelf_stats = []
        
        for i, shelf in enumerate(shelves):
            if hasattr(shelf, 'warp_to_rectangle'):  # Es Quadrilateral
                # Extraer región enderezada
                bbox = shelf.to_bbox()
                shelf_width = max(100, bbox.width)
                shelf_height = max(50, bbox.height)
                shelf_depth_warped = shelf.warp_to_rectangle(depth_map, shelf_width, shelf_height)
                
                # Analizar
                grid, occupancy_pct, stats = self.grid_analyzer.analyze_shelf(
                    shelf_depth_warped, bbox
                )
            else:  # BoundingBox tradicional
                shelf_region = depth_map[bbox.y1:bbox.y2, bbox.x1:bbox.x2]
                grid, occupancy_pct, stats = self.grid_analyzer.analyze_shelf(
                    shelf_region, shelf
                )
            
            occupancy_percentages.append(occupancy_pct)
            shelf_stats.append(stats)
            
            logger.debug(f"Anaquel {i+1}: {occupancy_pct:.1f}% ocupado")
        
        avg_occupancy = np.mean(occupancy_percentages) if occupancy_percentages else 0.0
        logger.info(f"Ocupación promedio: {avg_occupancy:.1f}%")
        
        # Preparar resultados
        results = {
            'avg_occupancy': float(avg_occupancy),
            'num_shelves': len(shelves),
            'shelves': [
                {
                    'id': i + 1,
                    'occupancy': float(occ),
                    'stats': stats
                }
                for i, (occ, stats) in enumerate(zip(occupancy_percentages, shelf_stats))
            ]
        }
        
        # Visualizaciones (si se solicitan)
        if return_visualizations:
            # Crear overlay con ocupación USANDO CUADRILÁTEROS REALES
            overlay = image.copy()
            
            for shelf, occ_pct in zip(shelves, occupancy_percentages):
                if hasattr(shelf, 'get_corners'):
                    corners = shelf.get_corners().astype(np.int32)
                    
                    # Color según ocupación
                    if occ_pct < 30:
                        color = (0, 0, 255)  # Rojo
                    elif occ_pct < 70:
                        color = (0, 255, 255)  # Amarillo
                    else:
                        color = (0, 255, 0)  # Verde
                    
                    # Dibujar polígono con transparencia
                    overlay_temp = overlay.copy()
                    cv2.fillPoly(overlay_temp, [corners], color)
                    cv2.addWeighted(overlay_temp, 0.3, overlay, 0.7, 0, overlay)
                    
                    # Borde del cuadrilátero
                    cv2.polylines(overlay, [corners], True, color, 4)
                    
                    # Texto con ocupación
                    center = shelf.center
                    text = f"{occ_pct:.1f}%"
                    (text_width, text_height), _ = cv2.getTextSize(
                        text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3
                    )
                    
                    # Fondo negro para texto
                    cv2.rectangle(
                        overlay,
                        (int(center[0]) - text_width//2 - 10, int(center[1]) - text_height//2 - 10),
                        (int(center[0]) + text_width//2 + 10, int(center[1]) + text_height//2 + 10),
                        (0, 0, 0), -1
                    )
                    
                    cv2.putText(
                        overlay, text,
                        (int(center[0]) - text_width//2, int(center[1]) + text_height//2),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3
                    )
                else:
                    # Fallback para BoundingBox
                    if occ_pct < 30:
                        color = (0, 0, 255)
                    elif occ_pct < 70:
                        color = (0, 255, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.rectangle(overlay, (shelf.x1, shelf.y1), (shelf.x2, shelf.y2), color, 4)
            
            results['overlay_image'] = overlay
            
            # Crear visualización del pipeline completo
            pipeline_image = self._create_pipeline_visualization(
                image, processed, edges, steps.get('lines', processed),
                steps.get('shelves', processed), depth_colored, overlay
            )
            results['pipeline_image'] = pipeline_image
        
        if return_steps:
            results['steps'] = steps
        
        logger.success("Procesamiento completado")
        return results
    
    def _create_pipeline_visualization(
        self,
        original: np.ndarray,
        preprocessed: np.ndarray,
        edges: np.ndarray,
        lines: np.ndarray,
        shelves: np.ndarray,
        depth: np.ndarray,
        occupancy: np.ndarray,
        max_width: int = 2400
    ) -> np.ndarray:
        """
        Crea visualización concatenada del pipeline completo.
        
        Returns:
            Imagen con 7 pasos del pipeline concatenados
        """
        # Convertir edges a BGR
        edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        
        # Redimensionar todas las imágenes al mismo tamaño
        h, w = preprocessed.shape[:2]
        target_h = 400
        target_w = int(w * target_h / h)
        
        images = [original, preprocessed, edges_bgr, lines, shelves, depth, occupancy]
        resized = [cv2.resize(img, (target_w, target_h)) for img in images]
        
        # Concatenar horizontalmente (2 filas)
        row1 = np.hstack(resized[:4])
        row2 = np.hstack(resized[4:])
        
        # Ajustar ancho de row2 si es necesario
        if row2.shape[1] < row1.shape[1]:
            padding = np.zeros((target_h, row1.shape[1] - row2.shape[1], 3), dtype=np.uint8)
            row2 = np.hstack([row2, padding])
        
        # Concatenar verticalmente
        pipeline = np.vstack([row1, row2])
        
        # Redimensionar si excede max_width
        if pipeline.shape[1] > max_width:
            scale = max_width / pipeline.shape[1]
            new_h = int(pipeline.shape[0] * scale)
            pipeline = cv2.resize(pipeline, (max_width, new_h))
        
        return pipeline


def main():
    """Función de ejemplo de uso."""
    import sys
    
    # Configurar logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Inicializar analizador
    analyzer = ShelfOccupancyAnalyzer()
    
    # Procesar imagen de ejemplo
    image_path = "data/raw/SKU110K_fixed/images/test_117.jpg"
    
    if not Path(image_path).exists():
        logger.error(f"Imagen no encontrada: {image_path}")
        return
    
    # Procesar
    results = analyzer.process(image_path, return_visualizations=True)
    
    # Mostrar resultados
    print("\n" + "="*50)
    print("RESULTADOS DEL ANÁLISIS")
    print("="*50)
    print(f"Ocupación promedio: {results['avg_occupancy']:.1f}%")
    print(f"Anaqueles detectados: {results['num_shelves']}")
    print("\nDetalle por anaquel:")
    for shelf in results['shelves']:
        print(f"  Anaquel {shelf['id']}: {shelf['occupancy']:.1f}%")
    print("="*50)
    
    # Guardar visualizaciones
    output_dir = Path("data/results/inference_test")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    cv2.imwrite(str(output_dir / "pipeline.jpg"), results['pipeline_image'])
    cv2.imwrite(str(output_dir / "overlay.jpg"), results['overlay_image'])
    
    logger.success(f"Visualizaciones guardadas en {output_dir}")


if __name__ == "__main__":
    main()
