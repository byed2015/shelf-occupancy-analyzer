"""
Script para visualizar el pipeline completo paso a paso.

Este script procesa una imagen a través de todas las etapas del pipeline
y genera visualizaciones de cada paso, guardándolas individualmente y
concatenadas en una única imagen resumen.

Uso:
    # Procesar una imagen específica
    uv run python visualize_pipeline.py --image data/raw/sample/sku110k_sample_000.jpg
    
    # Usar la primera imagen disponible
    uv run python visualize_pipeline.py
    
    # Con configuración personalizada
    uv run python visualize_pipeline.py --config config/config.yaml
"""

import argparse
import sys
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from shelf_occupancy.analysis import GridAnalyzer
from shelf_occupancy.config import load_config
from shelf_occupancy.depth import DepthEstimator
from shelf_occupancy.detection import EdgeDetector, LineDetector, ShelfDetector
from shelf_occupancy.preprocessing import ImagePreprocessor
from shelf_occupancy.utils import load_image, save_image
from shelf_occupancy.visualization import OccupancyVisualizer


class PipelineVisualizer:
    """Visualizador del pipeline completo paso a paso."""
    
    def __init__(self, config_path: str = "config/config.yaml"):
        """
        Inicializa el visualizador.
        
        Args:
            config_path: Ruta al archivo de configuración
        """
        self.config = load_config(config_path)
        self.steps: Dict[str, np.ndarray] = {}
        self.step_info: Dict[str, str] = {}
        
        logger.info("PipelineVisualizer inicializado")
    
    def process_image(self, image_path: Path) -> bool:
        """
        Procesa una imagen a través del pipeline completo.
        
        Args:
            image_path: Ruta a la imagen
        
        Returns:
            True si el procesamiento fue exitoso
        """
        logger.info("=" * 80)
        logger.info(f"🚀 VISUALIZACIÓN DEL PIPELINE COMPLETO")
        logger.info("=" * 80)
        logger.info(f"📁 Imagen: {image_path.name}\n")
        
        try:
            # Paso 0: Cargar imagen original
            logger.info("📷 PASO 0: Carga de imagen")
            original = load_image(image_path, color_mode="BGR")
            self.steps['0_original'] = original.copy()
            self.step_info['0_original'] = f"Original\n{original.shape[1]}x{original.shape[0]}"
            logger.info(f"   ✓ Imagen cargada: {original.shape}\n")
            
            # Paso 1: Conversión a escala de grises (único preprocesamiento necesario)
            logger.info("🔧 PASO 1: Conversión a escala de grises")
            gray = cv2.cvtColor(original, cv2.COLOR_BGR2GRAY)
            
            # Aplicar suavizado gaussiano ligero para reducir ruido
            gray_smooth = cv2.GaussianBlur(gray, (5, 5), 1.0)
            
            gray_bgr = cv2.cvtColor(gray_smooth, cv2.COLOR_GRAY2BGR)
            self.steps['1_preprocessed'] = gray_bgr
            self.step_info['1_preprocessed'] = "Escala de Grises\n+ Gaussian Blur"
            logger.info(f"   ✓ Conversión a escala de grises")
            logger.info(f"   ✓ Suavizado gaussiano aplicado\n")
            
            # Paso 2: Detección de bordes (Canny optimizado)
            logger.info("🔍 PASO 2: Detección de bordes")
            # Canny con auto-threshold (más robusto)
            median_val = np.median(gray_smooth)
            lower = int(max(0, 0.66 * median_val))
            upper = int(min(255, 1.33 * median_val))
            edges = cv2.Canny(gray_smooth, lower, upper, apertureSize=3)
            
            edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
            self.steps['2_edges'] = edges_bgr
            self.step_info['2_edges'] = f"Bordes (Canny Auto)\nUmbrales: {lower}/{upper}"
            logger.info(f"   ✓ Bordes detectados (auto-threshold: {lower}/{upper})\n")
            
            # Paso 3: Detección de líneas con perspectiva adaptativa
            logger.info("📐 PASO 3: Detección de líneas")
            line_detector = LineDetector(self.config.shelf_detection.hough)
            
            # Usar HoughLinesP normal (más rápido)
            all_lines = line_detector.detect(edges, use_polar=False)
            logger.info(f"   ✓ {len(all_lines)} líneas detectadas")
            
            # Filtrado ABSOLUTO (no adaptativo): horizontal cerca de 0°, vertical cerca de ±90°
            # Tolerancia de 20° captura perspectivas moderadas sin confundir orientaciones
            h_lines = line_detector.filter_by_orientation(all_lines, "horizontal", tolerance=20, adaptive=False)
            v_lines = line_detector.filter_by_orientation(all_lines, "vertical", tolerance=20, adaptive=False)
            
            # Detectar ángulo dominante para visualización
            dominant_angle_h = line_detector.detect_dominant_angle(h_lines) if h_lines else 0.0
            dominant_angle_v = line_detector.detect_dominant_angle(v_lines) if v_lines else 90.0
            
            # Fusionar líneas similares
            h_lines = line_detector.merge_similar_lines(h_lines, angle_threshold=5, distance_threshold=30)
            v_lines = line_detector.merge_similar_lines(v_lines, angle_threshold=5, distance_threshold=30)
            
            # Visualizar líneas con colores según ángulo
            lines_img = original.copy()
            
            # Dibujar horizontales en verde con intensidad según cercanía a ángulo dominante
            for line in h_lines:
                angle_diff = line_detector._angle_difference(line.angle, dominant_angle_h)
                intensity = int(255 * (1 - angle_diff / 15))  # Más brillante = más cercano
                cv2.line(lines_img, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (0, max(100, intensity), 0), 2)
            
            # Dibujar verticales en azul
            for line in v_lines:
                angle_diff = line_detector._angle_difference(line.angle, dominant_angle_v)
                intensity = int(255 * (1 - angle_diff / 15))
                cv2.line(lines_img, (int(line.x1), int(line.y1)), (int(line.x2), int(line.y2)), (max(100, intensity), 0, 0), 2)
            
            # Agregar texto con ángulo dominante
            cv2.putText(
                lines_img,
                f"Horizontal: {dominant_angle_h:.1f}deg",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (0, 255, 0),
                2
            )
            cv2.putText(
                lines_img,
                f"Vertical: {dominant_angle_v:.1f}deg",
                (10, 70),
                cv2.FONT_HERSHEY_SIMPLEX,
                1.0,
                (255, 0, 0),
                2
            )
            
            self.steps['3_lines'] = lines_img
            self.step_info['3_lines'] = f"Líneas (Hough Polar)\nH:{len(h_lines)}@{dominant_angle_h:.1f}° V:{len(v_lines)}@{dominant_angle_v:.1f}°"
            logger.info(f"   ✓ {len(h_lines)} líneas horizontales @ {dominant_angle_h:.1f}°")
            logger.info(f"   ✓ {len(v_lines)} líneas verticales @ {dominant_angle_v:.1f}°\n")
            
            # Paso 4: Detección de anaqueles como cuadriláteros inclinados
            # NO corregimos perspectiva, segmentamos siguiendo las líneas naturales
            logger.info("📦 PASO 4: Detección de anaqueles (cuadriláteros inclinados)")
            shelf_detector = ShelfDetector(self.config.shelf_detection)
            
            # Detectar anaqueles sin objetos (método simplificado)
            shelves = shelf_detector.detect_from_lines(
                h_lines, 
                v_lines, 
                original.shape[:2], 
                use_quadrilaterals=True,
                detected_objects=None  # No usar objetos para refinamiento
            )
            
            if not shelves:
                logger.warning("   ⚠ No se detectaron anaqueles, usando cuadrícula simple")
                shelves = shelf_detector.detect_simple_grid(original.shape[:2], n_rows=4)
            
            # FILTRADO GEOMÉTRICO MEJORADO (sin YOLO)
            logger.info(f"🔍 Filtrando anaqueles por geometría y posición...")
            valid_shelves = []
            
            for i, shelf in enumerate(shelves):
                # Validación geométrica: no piso ni techo
                center_y = shelf.center[1]
                image_height = original.shape[0]
                is_floor = center_y > image_height * 0.85  # 15% inferior
                is_ceiling = center_y < image_height * 0.05  # 5% superior
                
                # Validar área mínima
                if hasattr(shelf, 'get_area'):
                    area = shelf.get_area()
                else:
                    area = shelf.width * shelf.height if hasattr(shelf, 'width') else 1000000
                
                min_area = image_height * 100  # Área mínima proporcional a imagen
                is_too_small = area < min_area
                
                # Validar aspect ratio (anaqueles son más anchos que altos)
                if hasattr(shelf, 'width') and hasattr(shelf, 'height'):
                    aspect_ratio = shelf.width / shelf.height if shelf.height > 0 else 0
                    is_valid_ratio = 1.5 < aspect_ratio < 50  # Anaqueles típicamente 2:1 a 20:1
                else:
                    is_valid_ratio = True  # Asumir válido si no podemos calcular
                
                # Decidir si es válido
                if not is_floor and not is_ceiling and not is_too_small and is_valid_ratio:
                    valid_shelves.append(shelf)
                    logger.info(f"   ✓ Anaquel {i+1}: área={area:.0f}px² - VÁLIDO")
                else:
                    reason = "piso" if is_floor else ("techo" if is_ceiling else ("muy pequeño" if is_too_small else "aspect ratio inválido"))
                    logger.warning(f"   ✗ Anaquel {i+1}: área={area:.0f}px² - DESCARTADO ({reason})")
            
            if valid_shelves:
                logger.info(f"   ✓ Anaqueles válidos: {len(valid_shelves)}/{len(shelves)}")
                shelves = valid_shelves
            else:
                logger.warning("   ⚠ No hay anaqueles válidos tras filtrado, usando todos")
            
            # Visualizar anaqueles (dibujar cuadriláteros inclinados)
            shelves_img = original.copy()
            for i, shelf in enumerate(shelves):
                # Obtener puntos del cuadrilátero
                if hasattr(shelf, 'get_corners'):  # Es Quadrilateral
                    corners = shelf.get_corners().astype(np.int32)
                    cv2.polylines(shelves_img, [corners], True, (0, 255, 255), 3)
                    
                    # Dibujar puntos de esquina para claridad visual
                    for corner in corners:
                        cv2.circle(shelves_img, tuple(corner), 8, (0, 0, 255), -1)
                    
                    # Etiqueta en centro
                    center = shelf.center
                    cv2.putText(
                        shelves_img,
                        f"S{i+1}",
                        (int(center[0]) - 20, int(center[1])),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.2,
                        (0, 255, 255),
                        3
                    )
                else:  # Es BoundingBox
                    cv2.rectangle(
                        shelves_img,
                        (shelf.x, shelf.y),
                        (shelf.x + shelf.width, shelf.y + shelf.height),
                        (0, 255, 255),
                        3
                    )
                    cv2.putText(
                        shelves_img,
                        f"S{i+1}",
                        (shelf.x + 10, shelf.y + 30),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 255),
                        2
                    )
            
            self.steps['4_shelves'] = shelves_img
            self.step_info['4_shelves'] = f"Anaqueles (Cuadriláteros)\n{len(shelves)} detectados"
            logger.info(f"   ✓ {len(shelves)} anaqueles detectados\n")
            
            # Paso 5: Estimación de profundidad (usar imagen original)
            logger.info("🌊 PASO 5: Estimación de profundidad")
            depth_estimator = DepthEstimator(self.config.depth_estimation)
            depth_map, depth_colored = depth_estimator.estimate(original, return_colored=True)
            
            # Convertir depth_colored de RGB a BGR para OpenCV
            depth_colored_bgr = cv2.cvtColor(depth_colored, cv2.COLOR_RGB2BGR)
            self.steps['5_depth'] = depth_colored_bgr
            self.step_info['5_depth'] = f"Profundidad\nRango: [{depth_map.min():.2f}, {depth_map.max():.2f}]"
            logger.info(f"   ✓ Mapa de profundidad generado")
            logger.info(f"   ✓ Rango: [{depth_map.min():.3f}, {depth_map.max():.3f}]\n")
            
            # Paso 6: Análisis de ocupación (NORMALIZACIÓN POR CUADRILÁTERO)
            logger.info("📊 PASO 6: Análisis de ocupación (normalización independiente por anaquel)")
            
            # Para cada anaquel (cuadrilátero), calcular ocupación con normalización local
            occupancy_percentages = []
            stats_list = []
            
            for i, shelf in enumerate(shelves):
                if hasattr(shelf, 'get_corners'):  # Es Quadrilateral
                    # Crear máscara del cuadrilátero en la imagen
                    mask = np.zeros(depth_map.shape[:2], dtype=np.uint8)
                    corners = shelf.get_corners().astype(np.int32)
                    cv2.fillPoly(mask, [corners], 1)
                    
                    # Extraer profundidades dentro del cuadrilátero
                    shelf_depth_values = depth_map[mask == 1]
                    
                    if shelf_depth_values.size > 0:
                        # 🔥 NORMALIZACIÓN LOCAL POR CUADRILÁTERO
                        # Medir min/max DENTRO del cuadrilátero (no de la imagen completa)
                        depth_min = np.min(shelf_depth_values)
                        depth_max = np.max(shelf_depth_values)
                        depth_range = depth_max - depth_min
                        
                        # Normalizar profundidades al rango [0, 1] LOCAL
                        if depth_range > 0.01:  # Evitar división por cero
                            normalized_depths = (shelf_depth_values - depth_min) / depth_range
                        else:
                            # Si el rango es muy pequeño, asumir uniforme
                            normalized_depths = np.ones_like(shelf_depth_values) * 0.5
                        
                        # Calcular mediana de profundidades normalizadas
                        median_normalized = np.median(normalized_depths)
                        mean_normalized = np.mean(normalized_depths)
                        
                        # Interpretación:
                        # - median_normalized cercano a 0 = mayoría de píxeles cerca del fondo (vacío)
                        # - median_normalized cercano a 1 = mayoría de píxeles cerca del frente (lleno)
                        # Por lo tanto: ocupación = median_normalized * 100
                        
                        occupancy = median_normalized * 100  # Convertir a porcentaje
                        
                        logger.info(f"   Anaquel {i+1}:")
                        logger.info(f"      → Rango profundidad: [{depth_min:.3f}, {depth_max:.3f}]")
                        logger.info(f"      → Mediana normalizada: {median_normalized:.3f}")
                        logger.info(f"      → Media normalizada: {mean_normalized:.3f}")
                        logger.info(f"      → Ocupación: {occupancy:.1f}%")
                        
                        occupancy_percentages.append(occupancy)
                        stats_list.append({
                            'mean_occupancy': float(mean_normalized),
                            'median_occupancy': float(median_normalized),
                            'std_occupancy': float(np.std(normalized_depths)),
                            'min_occupancy': float(np.min(normalized_depths)),
                            'max_occupancy': float(np.max(normalized_depths)),
                            'depth_min': float(depth_min),
                            'depth_max': float(depth_max),
                            'depth_range': float(depth_range),
                            'occupied_cells': int(np.sum(normalized_depths > 0.3)),
                            'total_cells': int(shelf_depth_values.size)
                        })
                    else:
                        logger.warning(f"   Anaquel {i+1}: Sin valores de profundidad válidos")
                        occupancy_percentages.append(0.0)
                        stats_list.append({})
                else:  # BoundingBox tradicional
                    shelf_region = depth_map[shelf.y1:shelf.y2, shelf.x1:shelf.x2]
                    if shelf_region.size > 0:
                        depth_min = np.min(shelf_region)
                        depth_max = np.max(shelf_region)
                        depth_range = depth_max - depth_min
                        
                        if depth_range > 0.01:
                            normalized = (shelf_region - depth_min) / depth_range
                            median_normalized = np.median(normalized)
                        else:
                            median_normalized = 0.5
                        
                        occupancy = median_normalized * 100
                        occupancy_percentages.append(occupancy)
                        stats_list.append({})
                    else:
                        occupancy_percentages.append(0.0)
                        stats_list.append({})
            
            # Paso 6.5: Visualización combinada (cuadriláteros + depth)
            logger.info("\n🔗 PASO 6.5: Visualización combinada (anaqueles + profundidad)")
            
            # Vista combinada simplificada - depth en escala de grises + anaqueles
            combined_view = original.copy()
            
            # 1. Aplicar mapa de profundidad en escala de grises (sutil)
            depth_norm = ((depth_map - depth_map.min()) / (depth_map.max() - depth_map.min()) * 255).astype(np.uint8)
            depth_gray = cv2.cvtColor(cv2.applyColorMap(depth_norm, cv2.COLORMAP_BONE), cv2.COLOR_BGR2GRAY)
            depth_gray_3ch = cv2.cvtColor(depth_gray, cv2.COLOR_GRAY2BGR)
            cv2.addWeighted(depth_gray_3ch, 0.25, combined_view, 0.75, 0, combined_view)
            
            # 2. Dibujar cuadriláteros de anaqueles
            for i, shelf in enumerate(shelves):
                if hasattr(shelf, 'get_corners'):
                    corners = shelf.get_corners().astype(np.int32)
                    # Líneas cian gruesas para anaqueles
                    cv2.polylines(combined_view, [corners], True, (255, 255, 0), 4)
                    
                    # Etiqueta simple
                    center = shelf.center
                    label = f"A{i+1}"
                    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
                    
                    # Fondo semi-transparente para la etiqueta
                    overlay = combined_view.copy()
                    cv2.rectangle(overlay, 
                                (int(center[0]) - text_w//2 - 5, int(center[1]) - text_h//2 - 5),
                                (int(center[0]) + text_w//2 + 5, int(center[1]) + text_h//2 + 5),
                                (0, 0, 0), -1)
                    cv2.addWeighted(overlay, 0.6, combined_view, 0.4, 0, combined_view)
                    
                    cv2.putText(
                        combined_view,
                        label,
                        (int(center[0]) - text_w//2, int(center[1]) + text_h//2),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        (255, 255, 0),
                        2
                    )
                else:
                    cv2.rectangle(
                        combined_view,
                        (shelf.x1, shelf.y1),
                        (shelf.x2, shelf.y2),
                        (255, 255, 0),
                        4
                    )
            
            # Agregar leyenda
            legend_y = 35
            legend_bg = combined_view.copy()
            cv2.rectangle(legend_bg, (5, 5), (550, 50), (0, 0, 0), -1)
            cv2.addWeighted(legend_bg, 0.7, combined_view, 0.3, 0, combined_view)
            
            cv2.putText(combined_view, "Amarillo: Anaqueles | Fondo: Profundidad (gris)", 
                       (10, legend_y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            self.steps['6.5_combined'] = combined_view
            self.step_info['6.5_combined'] = f"Vista Combinada\n{len(shelves)} anaqueles"
            logger.info(f"   ✓ Vista combinada creada")
            logger.info(f"   ✓ Anaqueles válidos: {len(shelves)}\n")
            
            # Paso 7: Visualización final
            logger.info("\n🎨 PASO 7: Visualización de resultados")
            visualizer = OccupancyVisualizer(self.config.visualization)
            
            # Crear overlay con ocupación USANDO CUADRILÁTEROS REALES
            # Dibujar directamente sobre imagen original
            overlay = original.copy()
            
            for i, (shelf, occ_pct) in enumerate(zip(shelves, occupancy_percentages)):
                if hasattr(shelf, 'get_corners'):
                    # Dibujar cuadrilátero con color según ocupación
                    corners = shelf.get_corners().astype(np.int32)
                    
                    # Color según ocupación: rojo (vacío) -> amarillo -> verde (lleno)
                    if occ_pct < 30:
                        color = (0, 0, 255)  # Rojo - bajo
                    elif occ_pct < 70:
                        color = (0, 255, 255)  # Amarillo - medio
                    else:
                        color = (0, 255, 0)  # Verde - alto
                    
                    # Dibujar polígono con transparencia
                    overlay_temp = overlay.copy()
                    cv2.fillPoly(overlay_temp, [corners], color)
                    cv2.addWeighted(overlay_temp, 0.3, overlay, 0.7, 0, overlay)
                    
                    # Dibujar borde del cuadrilátero
                    cv2.polylines(overlay, [corners], True, color, 4)
                    
                    # Texto con ocupación en el centro
                    center = shelf.center
                    text = f"{occ_pct:.1f}%"
                    (text_width, text_height), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 3)
                    
                    # Fondo negro para el texto
                    cv2.rectangle(overlay, 
                                (int(center[0]) - text_width//2 - 10, int(center[1]) - text_height//2 - 10),
                                (int(center[0]) + text_width//2 + 10, int(center[1]) + text_height//2 + 10),
                                (0, 0, 0), -1)
                    
                    cv2.putText(overlay, text, (int(center[0]) - text_width//2, int(center[1]) + text_height//2),
                               cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                else:
                    # Fallback para BoundingBox
                    if occ_pct < 30:
                        color = (0, 0, 255)
                    elif occ_pct < 70:
                        color = (0, 255, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.rectangle(overlay, (shelf.x1, shelf.y1), (shelf.x2, shelf.y2), color, 4)
            
            self.steps['6_occupancy'] = overlay
            avg_occ = np.mean(occupancy_percentages)
            self.step_info['6_occupancy'] = f"Ocupación Final\nPromedio: {avg_occ:.1f}%"
            logger.info(f"   ✓ Overlay de ocupación creado")
            logger.info(f"   ✓ Ocupación promedio: {avg_occ:.1f}%\n")
            
            # Guardar metadatos
            self.metadata = {
                'image_path': str(image_path),
                'num_shelves': len(shelves),
                'occupancy_percentages': occupancy_percentages,
                'average_occupancy': float(avg_occ),
                'stats': stats_list,
                'dominant_angle_h': float(dominant_angle_h),
                'dominant_angle_v': float(dominant_angle_v),
                'uses_quadrilaterals': hasattr(shelves[0], 'get_corners') if shelves else False
            }
            
            logger.success("✅ Pipeline completado exitosamente\n")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error en el pipeline: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def save_individual_steps(self, output_dir: Path, image_name: str):
        """
        Guarda cada paso como imagen individual.
        
        Args:
            output_dir: Directorio de salida
            image_name: Nombre base de la imagen
        """
        output_dir.mkdir(parents=True, exist_ok=True)
        
        logger.info("💾 Guardando pasos individuales...")
        
        for step_name, step_image in self.steps.items():
            output_path = output_dir / f"{image_name}_{step_name}.jpg"
            save_image(step_image, output_path)
            logger.info(f"   ✓ {output_path.name}")
        
        logger.info(f"\n📁 Pasos guardados en: {output_dir}\n")
    
    def create_concatenated_view(
        self,
        output_path: Path,
        title: str = "Pipeline de Análisis de Ocupación de Anaqueles"
    ):
        """
        Crea una visualización concatenada con todos los pasos.
        
        Args:
            output_path: Ruta donde guardar la imagen concatenada
            title: Título de la visualización
        """
        logger.info("🎨 Creando visualización concatenada...")
        
        # Configurar figura
        n_steps = len(self.steps)
        cols = 4
        rows = (n_steps + cols - 1) // cols
        
        fig, axes = plt.subplots(rows, cols, figsize=(20, rows * 5))
        fig.suptitle(title, fontsize=20, fontweight='bold', y=0.995)
        
        # Aplanar axes para fácil indexación
        if rows == 1:
            axes = axes.reshape(1, -1)
        axes_flat = axes.flatten()
        
        # Plotear cada paso
        for idx, (step_name, step_image) in enumerate(sorted(self.steps.items())):
            ax = axes_flat[idx]
            
            # Convertir BGR a RGB para matplotlib
            if len(step_image.shape) == 3:
                display_image = cv2.cvtColor(step_image, cv2.COLOR_BGR2RGB)
            else:
                display_image = step_image
            
            ax.imshow(display_image)
            ax.set_title(
                self.step_info.get(step_name, step_name),
                fontsize=12,
                fontweight='bold',
                pad=10
            )
            ax.axis('off')
        
        # Ocultar axes sobrantes
        for idx in range(n_steps, len(axes_flat)):
            axes_flat[idx].axis('off')
        
        # Añadir información de métricas
        if hasattr(self, 'metadata'):
            info_text = f"Imagen: {Path(self.metadata['image_path']).name}\n"
            info_text += f"Anaqueles: {self.metadata['num_shelves']}\n"
            info_text += f"Ocupación promedio: {self.metadata['average_occupancy']:.1f}%"
            
            fig.text(
                0.02, 0.02, info_text,
                fontsize=11,
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
                verticalalignment='bottom'
            )
        
        plt.tight_layout()
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        
        logger.success(f"✅ Visualización concatenada guardada: {output_path}\n")
    
    def generate_report(self, output_path: Path):
        """
        Genera un reporte en texto con las métricas.
        
        Args:
            output_path: Ruta del archivo de reporte
        """
        if not hasattr(self, 'metadata'):
            logger.warning("No hay metadatos para generar reporte")
            return
        
        logger.info("📝 Generando reporte...")
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write("=" * 80 + "\n")
            f.write("REPORTE DE ANÁLISIS DE OCUPACIÓN DE ANAQUELES\n")
            f.write("=" * 80 + "\n\n")
            
            f.write(f"Imagen analizada: {Path(self.metadata['image_path']).name}\n")
            f.write(f"Número de anaqueles detectados: {self.metadata['num_shelves']}\n")
            f.write(f"Ocupación promedio: {self.metadata['average_occupancy']:.2f}%\n\n")
            
            f.write("-" * 80 + "\n")
            f.write("MÉTRICAS POR ANAQUEL\n")
            f.write("-" * 80 + "\n\n")
            
            for i, (occ_pct, stats) in enumerate(zip(
                self.metadata['occupancy_percentages'],
                self.metadata['stats']
            )):
                f.write(f"Anaquel {i+1}:\n")
                f.write(f"  - Ocupación: {occ_pct:.2f}%\n")
                f.write(f"  - Celdas ocupadas: {stats['occupied_cells']}/{stats['total_cells']}\n")
                f.write(f"  - Ocupación mín/máx: {stats['min_occupancy']:.3f} / {stats['max_occupancy']:.3f}\n")
                f.write(f"  - Desviación estándar: {stats['std_occupancy']:.3f}\n")
                f.write("\n")
            
            f.write("=" * 80 + "\n")
        
        logger.success(f"✅ Reporte guardado: {output_path}\n")


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(
        description="Visualiza el pipeline completo de análisis de ocupación de anaqueles"
    )
    parser.add_argument(
        "--image",
        type=str,
        default=None,
        help="Ruta a la imagen a procesar. Si no se especifica, usa la primera disponible."
    )
    parser.add_argument(
        "--config",
        type=str,
        default="config/config.yaml",
        help="Ruta al archivo de configuración"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/results/pipeline_visualization",
        help="Directorio de salida para resultados"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    # Determinar imagen a procesar
    if args.image:
        image_path = Path(args.image)
        if not image_path.exists():
            logger.error(f"Imagen no encontrada: {image_path}")
            return 1
    else:
        # Buscar primera imagen disponible
        sample_dir = Path("data/raw/sample")
        image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
        
        if not image_files:
            logger.error("No se encontraron imágenes en data/raw/sample/")
            return 1
        
        image_path = image_files[0]
        logger.info(f"Usando imagen: {image_path.name}\n")
    
    # Crear visualizador
    visualizer = PipelineVisualizer(args.config)
    
    # Procesar imagen
    success = visualizer.process_image(image_path)
    
    if not success:
        return 1
    
    # Crear directorio de salida
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    image_name = image_path.stem
    
    # Guardar pasos individuales
    steps_dir = output_dir / "individual_steps"
    visualizer.save_individual_steps(steps_dir, image_name)
    
    # Crear visualización concatenada
    concat_path = output_dir / f"{image_name}_pipeline_complete.png"
    visualizer.create_concatenated_view(concat_path)
    
    # Generar reporte
    report_path = output_dir / f"{image_name}_report.txt"
    visualizer.generate_report(report_path)
    
    logger.info("=" * 80)
    logger.info("🎉 PROCESO COMPLETADO")
    logger.info("=" * 80)
    logger.info(f"\n📁 Resultados guardados en: {output_dir}")
    logger.info(f"   - Visualización completa: {concat_path.name}")
    logger.info(f"   - Pasos individuales: {steps_dir}/")
    logger.info(f"   - Reporte de métricas: {report_path.name}\n")
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
