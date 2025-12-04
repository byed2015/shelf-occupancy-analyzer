"""Módulo de visualización de resultados."""

from pathlib import Path
from typing import List, Optional, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from loguru import logger

from shelf_occupancy.config import VisualizationConfig
from shelf_occupancy.utils.geometry import BoundingBox


class OccupancyVisualizer:
    """Visualizador de ocupación de anaqueles."""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        """
        Inicializa el visualizador.
        
        Args:
            config: Configuración de visualización. Si None, usa valores por defecto.
        """
        if config is None:
            config = VisualizationConfig()
        self.config = config
        logger.info("OccupancyVisualizer inicializado")
    
    def create_overlay(
        self,
        image: np.ndarray,
        shelves: List[BoundingBox],
        occupancy_percentages: List[float]
    ) -> np.ndarray:
        """
        Crea overlay de ocupación sobre la imagen.
        
        Args:
            image: Imagen original
            shelves: Lista de bounding boxes de anaqueles
            occupancy_percentages: Porcentajes de ocupación por anaquel
        
        Returns:
            Imagen con overlay
        """
        overlay = image.copy()
        
        for shelf, occupancy in zip(shelves, occupancy_percentages):
            # Determinar color según nivel de ocupación
            color = self._get_color_for_occupancy(occupancy)
            
            # Dibujar rectángulo semi-transparente
            x, y, w, h = shelf.x, shelf.y, shelf.width, shelf.height
            
            # Crear máscara para el área
            mask = np.zeros_like(overlay)
            cv2.rectangle(mask, (x, y), (x + w, y + h), color, -1)
            
            # Aplicar con transparencia
            overlay = cv2.addWeighted(overlay, 1.0, mask, 0.3, 0)
            
            # Dibujar borde
            cv2.rectangle(overlay, (x, y), (x + w, y + h), color, 3)
            
            # Agregar texto con ocupación
            text = f"{occupancy:.1f}%"
            font_scale = 1.0
            thickness = 2
            
            # Calcular tamaño del texto
            (text_w, text_h), baseline = cv2.getTextSize(
                text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness
            )
            
            # Dibujar fondo para el texto
            text_x = x + 10
            text_y = y + 40
            cv2.rectangle(
                overlay,
                (text_x - 5, text_y - text_h - 5),
                (text_x + text_w + 5, text_y + baseline + 5),
                (0, 0, 0),
                -1
            )
            
            # Dibujar texto
            cv2.putText(
                overlay,
                text,
                (text_x, text_y),
                cv2.FONT_HERSHEY_SIMPLEX,
                font_scale,
                (255, 255, 255),
                thickness
            )
        
        return overlay
    
    def _get_color_for_occupancy(self, occupancy: float) -> Tuple[int, int, int]:
        """
        Obtiene color BGR según nivel de ocupación.
        
        Args:
            occupancy: Porcentaje de ocupación
        
        Returns:
            Color en formato BGR
        """
        if occupancy >= 70:
            # Alto: Verde
            color_rgb = self.config.colors['high_occupancy']
        elif occupancy >= 40:
            # Medio: Amarillo
            color_rgb = self.config.colors['medium_occupancy']
        else:
            # Bajo: Rojo
            color_rgb = self.config.colors['low_occupancy']
        
        # Convertir RGB a BGR
        return (color_rgb[2], color_rgb[1], color_rgb[0])
    
    def visualize_grid_heatmap(
        self,
        occupancy_grid: np.ndarray,
        shelf_bbox: BoundingBox,
        title: str = "Mapa de Ocupación"
    ) -> plt.Figure:
        """
        Crea mapa de calor de la cuadrícula de ocupación.
        
        Args:
            occupancy_grid: Matriz de ocupación [0, 1]
            shelf_bbox: Bounding box del anaquel
            title: Título del gráfico
        
        Returns:
            Figura de matplotlib
        """
        fig, ax = plt.subplots(figsize=(10, 6))
        
        # Crear heatmap
        im = ax.imshow(
            occupancy_grid,
            cmap='RdYlGn',  # Rojo-Amarillo-Verde
            vmin=0,
            vmax=1,
            aspect='auto'
        )
        
        # Agregar colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label('Nivel de Ocupación', rotation=270, labelpad=20)
        
        # Agregar valores en las celdas
        rows, cols = occupancy_grid.shape
        for i in range(rows):
            for j in range(cols):
                text = ax.text(
                    j, i, f'{occupancy_grid[i, j]:.2f}',
                    ha="center", va="center",
                    color="black", fontsize=8
                )
        
        # Configurar ejes
        ax.set_title(title, fontweight='bold', fontsize=14)
        ax.set_xlabel('Columnas')
        ax.set_ylabel('Filas')
        
        # Configurar ticks
        ax.set_xticks(np.arange(cols))
        ax.set_yticks(np.arange(rows))
        ax.set_xticklabels(np.arange(1, cols + 1))
        ax.set_yticklabels(np.arange(1, rows + 1))
        
        plt.tight_layout()
        
        return fig
    
    def create_summary_visualization(
        self,
        image: np.ndarray,
        depth_colored: np.ndarray,
        overlay: np.ndarray,
        stats: List[dict],
        save_path: Optional[Path] = None
    ) -> plt.Figure:
        """
        Crea visualización resumen completa.
        
        Args:
            image: Imagen original
            depth_colored: Mapa de profundidad colorizado
            overlay: Imagen con overlay de ocupación
            stats: Estadísticas de ocupación
            save_path: Path donde guardar (opcional)
        
        Returns:
            Figura de matplotlib
        """
        fig = plt.figure(figsize=(18, 12))
        gs = fig.add_gridspec(2, 3, hspace=0.3, wspace=0.3)
        
        # Imagen original
        ax1 = fig.add_subplot(gs[0, 0])
        ax1.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        ax1.set_title('Imagen Original', fontweight='bold', fontsize=12)
        ax1.axis('off')
        
        # Mapa de profundidad
        ax2 = fig.add_subplot(gs[0, 1])
        ax2.imshow(depth_colored)
        ax2.set_title('Mapa de Profundidad', fontweight='bold', fontsize=12)
        ax2.axis('off')
        
        # Overlay de ocupación
        ax3 = fig.add_subplot(gs[0, 2])
        ax3.imshow(cv2.cvtColor(overlay, cv2.COLOR_BGR2RGB))
        ax3.set_title('Análisis de Ocupación', fontweight='bold', fontsize=12)
        ax3.axis('off')
        
        # Gráfico de barras de ocupación por anaquel
        ax4 = fig.add_subplot(gs[1, :2])
        shelf_nums = [f"Anaquel {i+1}" for i in range(len(stats))]
        occupancies = [s['mean_occupancy'] * 100 for s in stats]
        colors_bar = [self._get_bar_color(occ) for occ in occupancies]
        
        bars = ax4.bar(shelf_nums, occupancies, color=colors_bar, alpha=0.7, edgecolor='black')
        ax4.set_ylabel('Ocupación (%)', fontweight='bold')
        ax4.set_title('Ocupación por Anaquel', fontweight='bold', fontsize=12)
        ax4.set_ylim(0, 100)
        ax4.grid(axis='y', alpha=0.3)
        
        # Agregar valores sobre las barras
        for bar, occ in zip(bars, occupancies):
            height = bar.get_height()
            ax4.text(
                bar.get_x() + bar.get_width() / 2., height,
                f'{occ:.1f}%',
                ha='center', va='bottom',
                fontweight='bold'
            )
        
        # Tabla de estadísticas
        ax5 = fig.add_subplot(gs[1, 2])
        ax5.axis('off')
        
        # Calcular promedios globales
        avg_occupancy = np.mean(occupancies)
        total_cells = sum(s['total_cells'] for s in stats)
        occupied_cells = sum(s['occupied_cells'] for s in stats)
        
        stats_text = f"""
        ESTADÍSTICAS GLOBALES
        {'=' * 30}
        
        Ocupación promedio: {avg_occupancy:.1f}%
        
        Total de anaqueles: {len(stats)}
        Total de celdas: {total_cells}
        Celdas ocupadas: {occupied_cells}
        
        Clasificación:
        🟢 Alto (>70%): {sum(1 for o in occupancies if o >= 70)}
        🟡 Medio (40-70%): {sum(1 for o in occupancies if 40 <= o < 70)}
        🔴 Bajo (<40%): {sum(1 for o in occupancies if o < 40)}
        """
        
        ax5.text(
            0.1, 0.9, stats_text,
            transform=ax5.transAxes,
            fontsize=10,
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5)
        )
        
        if save_path:
            plt.savefig(save_path, dpi=self.config.dpi, bbox_inches='tight')
            logger.info(f"Visualización guardada: {save_path}")
        
        return fig
    
    def _get_bar_color(self, occupancy: float) -> str:
        """Obtiene color para gráfico de barras según ocupación."""
        if occupancy >= 70:
            return 'green'
        elif occupancy >= 40:
            return 'yellow'
        else:
            return 'red'


def create_occupancy_overlay(
    image: np.ndarray,
    shelves: List[BoundingBox],
    occupancy_percentages: List[float]
) -> np.ndarray:
    """
    Función de conveniencia para crear overlay de ocupación.
    
    Args:
        image: Imagen original
        shelves: Lista de anaqueles
        occupancy_percentages: Porcentajes de ocupación
    
    Returns:
        Imagen con overlay
    """
    visualizer = OccupancyVisualizer()
    return visualizer.create_overlay(image, shelves, occupancy_percentages)
