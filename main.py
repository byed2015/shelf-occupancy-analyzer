"""Pipeline completo de análisis de ocupación de anaqueles."""

import sys
from pathlib import Path

import cv2
import matplotlib.pyplot as plt
from loguru import logger

from shelf_occupancy.analysis import GridAnalyzer
from shelf_occupancy.config import load_config
from shelf_occupancy.depth import DepthEstimator
from shelf_occupancy.detection import EdgeDetector, LineDetector, ShelfDetector
from shelf_occupancy.preprocessing import ImagePreprocessor
from shelf_occupancy.utils import load_image
from shelf_occupancy.visualization import OccupancyVisualizer


def main():
    """Pipeline completo de análisis."""
    # Configurar logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("=" * 80)
    logger.info("🚀 PIPELINE COMPLETO DE ANÁLISIS DE OCUPACIÓN DE ANAQUELES")
    logger.info("=" * 80)
    
    # Cargar configuración
    config = load_config()
    
    # Directorios
    data_dir = Path("data")
    sample_dir = data_dir / "raw" / "sample"
    results_dir = data_dir / "results" / "complete_pipeline"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener imagen
    image_files = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if not image_files:
        logger.error("No se encontraron imágenes")
        return False
    
    # Procesar primera imagen
    img_path = image_files[0]
    logger.info(f"\n📁 Procesando: {img_path.name}\n")
    
    # ====================
    # FASE 1: CARGA Y PREPROCESAMIENTO
    # ====================
    logger.info("📷 FASE 1: Carga y Preprocesamiento")
    logger.info("-" * 40)
    
    image = load_image(img_path, color_mode="BGR")
    logger.info(f"✅ Imagen cargada: {image.shape}")
    
    preprocessor = ImagePreprocessor(config.preprocessing)
    processed = preprocessor.preprocess(image, apply_resize=False, apply_normalize=False)
    logger.info(f"✅ Preprocesamiento completado")
    
    # ====================
    # FASE 2: DETECCIÓN DE ESTRUCTURA
    # ====================
    logger.info(f"\n🔍 FASE 2: Detección de Estructura de Anaqueles")
    logger.info("-" * 40)
    
    # Detectar bordes
    edge_detector = EdgeDetector(config.shelf_detection.canny)
    edges = edge_detector.detect(processed)
    logger.info(f"✅ Bordes detectados")
    
    # Detectar líneas
    line_detector = LineDetector(config.shelf_detection.hough)
    all_lines = line_detector.detect(edges)
    h_lines = line_detector.filter_by_orientation(all_lines, "horizontal", tolerance=15)
    v_lines = line_detector.filter_by_orientation(all_lines, "vertical", tolerance=15)
    h_lines = line_detector.merge_similar_lines(h_lines)
    logger.info(f"✅ Líneas detectadas: {len(h_lines)} horizontales, {len(v_lines)} verticales")
    
    # Detectar anaqueles
    shelf_detector = ShelfDetector(config.shelf_detection)
    shelves = shelf_detector.detect_from_lines(h_lines, v_lines, processed.shape[:2])
    
    if not shelves:
        logger.warning("No se detectaron anaqueles, usando cuadrícula simple")
        shelves = shelf_detector.detect_simple_grid(processed.shape[:2], n_rows=4)
    
    logger.info(f"✅ Anaqueles detectados: {len(shelves)}")
    for i, shelf in enumerate(shelves):
        logger.info(f"   Anaquel {i+1}: {shelf}")
    
    # ====================
    # FASE 3: ESTIMACIÓN DE PROFUNDIDAD
    # ====================
    logger.info(f"\n🌊 FASE 3: Estimación de Profundidad")
    logger.info("-" * 40)
    
    depth_estimator = DepthEstimator(config.depth_estimation)
    depth_map, depth_colored = depth_estimator.estimate(processed, return_colored=True)
    logger.info(f"✅ Profundidad estimada: rango [{depth_map.min():.3f}, {depth_map.max():.3f}]")
    
    # ====================
    # FASE 4: ANÁLISIS DE OCUPACIÓN
    # ====================
    logger.info(f"\n📊 FASE 4: Análisis de Ocupación por Cuadrículas (con refinamiento)")
    logger.info("-" * 40)
    
    grid_analyzer = GridAnalyzer(config.occupancy_analysis, enable_refinement=True)
    results = grid_analyzer.analyze_multiple_shelves(depth_map, shelves, processed)
    
    occupancy_grids = []
    occupancy_percentages = []
    stats_list = []
    
    for i, (occ_grid, occ_pct, stats) in enumerate(results):
        occupancy_grids.append(occ_grid)
        occupancy_percentages.append(occ_pct)
        stats_list.append(stats)
        
        level = grid_analyzer.classify_occupancy_level(occ_pct)
        level_emoji = {'high': '🟢', 'medium': '🟡', 'low': '🔴'}[level]
        
        logger.info(f"{level_emoji} Anaquel {i+1}: {occ_pct:.1f}% ({level})")
        logger.info(f"   - Celdas ocupadas: {stats['occupied_cells']}/{stats['total_cells']}")
        logger.info(f"   - Rango: [{stats['min_occupancy']:.2f}, {stats['max_occupancy']:.2f}]")
    
    # ====================
    # FASE 5: VISUALIZACIÓN
    # ====================
    logger.info(f"\n🎨 FASE 5: Generación de Visualizaciones")
    logger.info("-" * 40)
    
    visualizer = OccupancyVisualizer(config.visualization)
    
    # Crear overlay
    overlay = visualizer.create_overlay(processed, shelves, occupancy_percentages)
    logger.info("✅ Overlay de ocupación creado")
    
    # Crear visualización resumen
    summary_path = results_dir / f"{img_path.stem}_complete_analysis.png"
    fig = visualizer.create_summary_visualization(
        processed,
        depth_colored,
        overlay,
        stats_list,
        save_path=summary_path
    )
    plt.close(fig)
    logger.info(f"✅ Visualización resumen guardada: {summary_path}")
    
    # Crear heatmaps individuales por anaquel
    for i, (shelf, occ_grid) in enumerate(zip(shelves, occupancy_grids)):
        heatmap_path = results_dir / f"{img_path.stem}_shelf_{i+1}_heatmap.png"
        fig = visualizer.visualize_grid_heatmap(
            occ_grid,
            shelf,
            title=f"Anaquel {i+1} - Ocupación: {occupancy_percentages[i]:.1f}%"
        )
        plt.savefig(heatmap_path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        logger.info(f"✅ Heatmap guardado: {heatmap_path}")
    
    # ====================
    # RESUMEN FINAL
    # ====================
    logger.info("\n" + "=" * 80)
    logger.info("📋 RESUMEN FINAL")
    logger.info("=" * 80)
    
    avg_occupancy = sum(occupancy_percentages) / len(occupancy_percentages)
    logger.info(f"\n📊 Ocupación promedio global: {avg_occupancy:.1f}%")
    logger.info(f"📦 Total de anaqueles analizados: {len(shelves)}")
    logger.info(f"🔢 Total de celdas analizadas: {sum(s['total_cells'] for s in stats_list)}")
    
    logger.info(f"\n📁 Resultados guardados en: {results_dir}")
    logger.info(f"   - Visualización completa: {summary_path.name}")
    for i in range(len(shelves)):
        logger.info(f"   - Heatmap anaquel {i+1}: {img_path.stem}_shelf_{i+1}_heatmap.png")
    
    logger.success("\n🎉 ¡PIPELINE COMPLETADO EXITOSAMENTE!\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
