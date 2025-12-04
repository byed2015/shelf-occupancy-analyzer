"""Procesar todas las imágenes del dataset con el pipeline completo."""

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


def process_single_image(
    img_path: Path,
    preprocessor: ImagePreprocessor,
    edge_detector: EdgeDetector,
    line_detector: LineDetector,
    shelf_detector: ShelfDetector,
    depth_estimator: DepthEstimator,
    grid_analyzer: GridAnalyzer,
    visualizer: OccupancyVisualizer,
    results_dir: Path
) -> dict:
    """
    Procesa una imagen individual con el pipeline completo.
    
    Returns:
        Diccionario con estadísticas del procesamiento
    """
    logger.info(f"\n{'='*60}")
    logger.info(f"📸 Procesando: {img_path.name}")
    logger.info(f"{'='*60}")
    
    try:
        # 1. Cargar y preprocesar
        image = load_image(img_path, color_mode="BGR")
        processed = preprocessor.preprocess(image, apply_resize=False, apply_normalize=False)
        logger.info(f"✅ Imagen cargada y preprocesada: {image.shape}")
        
        # 2. Detección de estructura
        edges = edge_detector.detect(processed)
        all_lines = line_detector.detect(edges)
        h_lines = line_detector.filter_by_orientation(all_lines, "horizontal", tolerance=15)
        v_lines = line_detector.filter_by_orientation(all_lines, "vertical", tolerance=15)
        h_lines = line_detector.merge_similar_lines(h_lines)
        
        shelves = shelf_detector.detect_from_lines(h_lines, v_lines, processed.shape[:2])
        if not shelves:
            shelves = shelf_detector.detect_simple_grid(processed.shape[:2], n_rows=4)
        
        logger.info(f"✅ Detección: {len(h_lines)} líneas H, {len(v_lines)} líneas V, {len(shelves)} anaqueles")
        
        # 3. Estimación de profundidad
        depth_map, depth_colored = depth_estimator.estimate(processed, return_colored=True)
        logger.info(f"✅ Profundidad: [{depth_map.min():.3f}, {depth_map.max():.3f}]")
        
        # 4. Análisis de ocupación
        results = grid_analyzer.analyze_multiple_shelves(depth_map, shelves)
        
        occupancy_grids = []
        occupancy_percentages = []
        stats_list = []
        
        for occ_grid, occ_pct, stats in results:
            occupancy_grids.append(occ_grid)
            occupancy_percentages.append(occ_pct)
            stats_list.append(stats)
        
        avg_occupancy = sum(occupancy_percentages) / len(occupancy_percentages) if occupancy_percentages else 0
        logger.info(f"📊 Ocupación promedio: {avg_occupancy:.1f}%")
        
        # 5. Visualización
        overlay = visualizer.create_overlay(processed, shelves, occupancy_percentages)
        
        summary_path = results_dir / f"{img_path.stem}_analysis.png"
        fig = visualizer.create_summary_visualization(
            processed,
            depth_colored,
            overlay,
            stats_list,
            save_path=summary_path
        )
        plt.close(fig)
        logger.success(f"✅ Visualización guardada: {summary_path.name}")
        
        return {
            'filename': img_path.name,
            'shape': image.shape,
            'n_shelves': len(shelves),
            'n_h_lines': len(h_lines),
            'n_v_lines': len(v_lines),
            'avg_occupancy': avg_occupancy,
            'occupancy_per_shelf': occupancy_percentages,
            'total_cells': sum(s['total_cells'] for s in stats_list),
            'success': True,
            'error': None
        }
        
    except Exception as e:
        logger.error(f"❌ Error procesando {img_path.name}: {e}")
        return {
            'filename': img_path.name,
            'success': False,
            'error': str(e)
        }


def main():
    """Procesa todas las imágenes del dataset."""
    # Configurar logging
    logger.remove()
    logger.add(sys.stderr, level="INFO")
    
    logger.info("=" * 80)
    logger.info("🚀 PROCESAMIENTO MASIVO DE IMÁGENES - DATASET SKU-110K")
    logger.info("=" * 80)
    
    # Configuración
    config = load_config()
    
    # Directorios
    data_dir = Path("data")
    sample_dir = data_dir / "raw" / "sample"
    results_dir = data_dir / "results" / "batch_processing"
    results_dir.mkdir(parents=True, exist_ok=True)
    
    # Obtener todas las imágenes
    image_files = sorted(list(sample_dir.glob("sku110k_sample_*.jpg")))
    
    if not image_files:
        logger.error("No se encontraron imágenes del dataset SKU-110K")
        return False
    
    logger.info(f"\n📁 Encontradas {len(image_files)} imágenes para procesar\n")
    
    # Inicializar componentes del pipeline
    preprocessor = ImagePreprocessor(config.preprocessing)
    edge_detector = EdgeDetector(config.shelf_detection.canny)
    line_detector = LineDetector(config.shelf_detection.hough)
    shelf_detector = ShelfDetector(config.shelf_detection)
    depth_estimator = DepthEstimator(config.depth_estimation)
    grid_analyzer = GridAnalyzer(config.occupancy_analysis)
    visualizer = OccupancyVisualizer(config.visualization)
    
    # Procesar todas las imágenes
    all_results = []
    
    for i, img_path in enumerate(image_files, 1):
        logger.info(f"\n[{i}/{len(image_files)}]")
        
        result = process_single_image(
            img_path,
            preprocessor,
            edge_detector,
            line_detector,
            shelf_detector,
            depth_estimator,
            grid_analyzer,
            visualizer,
            results_dir
        )
        all_results.append(result)
    
    # Generar reporte final
    logger.info("\n" + "=" * 80)
    logger.info("📋 REPORTE FINAL DE PROCESAMIENTO")
    logger.info("=" * 80)
    
    successful = [r for r in all_results if r['success']]
    failed = [r for r in all_results if not r['success']]
    
    logger.info(f"\n✅ Procesadas exitosamente: {len(successful)}/{len(image_files)}")
    if failed:
        logger.warning(f"❌ Fallidas: {len(failed)}")
        for r in failed:
            logger.warning(f"   - {r['filename']}: {r['error']}")
    
    if successful:
        logger.info("\n📊 ESTADÍSTICAS GLOBALES:")
        avg_global_occupancy = sum(r['avg_occupancy'] for r in successful) / len(successful)
        total_shelves = sum(r['n_shelves'] for r in successful)
        total_cells = sum(r['total_cells'] for r in successful)
        
        logger.info(f"   • Ocupación promedio global: {avg_global_occupancy:.1f}%")
        logger.info(f"   • Total de anaqueles detectados: {total_shelves}")
        logger.info(f"   • Total de celdas analizadas: {total_cells}")
        
        logger.info("\n📈 DETALLES POR IMAGEN:")
        for r in successful:
            logger.info(f"   • {r['filename']}: {r['avg_occupancy']:.1f}% "
                       f"({r['n_shelves']} anaqueles, {r['total_cells']} celdas)")
    
    logger.info(f"\n📁 Todos los resultados guardados en: {results_dir}")
    logger.success("\n🎉 ¡PROCESAMIENTO MASIVO COMPLETADO!\n")
    
    return True


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
