"""Módulo de configuración del sistema."""

from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class CLAHEConfig(BaseModel):
    """Configuración para CLAHE."""
    clip_limit: float = 2.0
    tile_grid_size: Tuple[int, int] = (8, 8)


class BilateralFilterConfig(BaseModel):
    """Configuración para filtro bilateral."""
    d: int = 9
    sigma_color: float = 75.0
    sigma_space: float = 75.0


class PreprocessingConfig(BaseModel):
    """Configuración de preprocesamiento."""
    clahe: CLAHEConfig = Field(default_factory=CLAHEConfig)
    bilateral_filter: BilateralFilterConfig = Field(default_factory=BilateralFilterConfig)
    target_size: Optional[Tuple[int, int]] = (640, 640)
    normalize_values: bool = True


class CannyConfig(BaseModel):
    """Configuración para Canny."""
    low_threshold: int = 50
    high_threshold: int = 150
    aperture_size: int = 3


class HoughConfig(BaseModel):
    """Configuración para Hough Transform."""
    rho: int = 1
    theta: float = 0.017453293
    threshold: int = 100
    min_line_length: int = 100
    max_line_gap: int = 10


class ClusteringConfig(BaseModel):
    """Configuración para clustering de líneas."""
    angle_tolerance: float = 10.0
    distance_tolerance: float = 50.0
    min_lines_per_shelf: int = 2


class MorphologyConfig(BaseModel):
    """Configuración para operaciones morfológicas."""
    kernel_size: Tuple[int, int] = (5, 5)
    iterations: int = 2


class ShelfDetectionConfig(BaseModel):
    """Configuración de detección de anaqueles."""
    canny: CannyConfig = Field(default_factory=CannyConfig)
    hough: HoughConfig = Field(default_factory=HoughConfig)
    clustering: ClusteringConfig = Field(default_factory=ClusteringConfig)
    morphology: MorphologyConfig = Field(default_factory=MorphologyConfig)


class DepthPostprocessingConfig(BaseModel):
    """Configuración de post-procesamiento de profundidad."""
    bilateral_filter: BilateralFilterConfig = Field(default_factory=BilateralFilterConfig)
    resize_to_original: bool = True


class DepthEstimationConfig(BaseModel):
    """Configuración de estimación de profundidad."""
    model_name: str = "depth-anything/Depth-Anything-V2-Small-hf"
    device: str = "cuda"
    batch_size: int = 1
    normalize_depth: bool = True
    postprocessing: DepthPostprocessingConfig = Field(default_factory=DepthPostprocessingConfig)


class OccupancyThresholdsConfig(BaseModel):
    """Umbrales para ocupación."""
    depth_percentile: float = 0.3
    min_occupancy: float = 0.2
    std_threshold: float = 0.1


class NormalizationConfig(BaseModel):
    """Configuración de normalización."""
    method: str = "minmax"
    per_shelf: bool = True


class OccupancyAnalysisConfig(BaseModel):
    """Configuración de análisis de ocupación."""
    grid_size: Tuple[int, int] = (10, 5)
    thresholds: OccupancyThresholdsConfig = Field(default_factory=OccupancyThresholdsConfig)
    normalization: NormalizationConfig = Field(default_factory=NormalizationConfig)


class DatasetConfig(BaseModel):
    """Configuración del dataset."""
    name: str = "SKU-110K"
    raw_dir: str = "data/raw"
    processed_dir: str = "data/processed"
    results_dir: str = "data/results"
    sample_size: int = 10


class VisualizationConfig(BaseModel):
    """Configuración de visualización."""
    colors: Dict[str, List[int]] = Field(default_factory=lambda: {
        "high_occupancy": [0, 255, 0],
        "medium_occupancy": [255, 255, 0],
        "low_occupancy": [255, 0, 0]
    })
    occupancy_levels: Dict[str, float] = Field(default_factory=lambda: {
        "high": 0.7,
        "medium": 0.4
    })
    dpi: int = 150
    figsize: Tuple[int, int] = (15, 10)
    save_format: str = "png"


class Config(BaseSettings):
    """Configuración principal del sistema."""
    dataset: DatasetConfig = Field(default_factory=DatasetConfig)
    preprocessing: PreprocessingConfig = Field(default_factory=PreprocessingConfig)
    shelf_detection: ShelfDetectionConfig = Field(default_factory=ShelfDetectionConfig)
    depth_estimation: DepthEstimationConfig = Field(default_factory=DepthEstimationConfig)
    occupancy_analysis: OccupancyAnalysisConfig = Field(default_factory=OccupancyAnalysisConfig)
    visualization: VisualizationConfig = Field(default_factory=VisualizationConfig)

    @classmethod
    def from_yaml(cls, path: Path) -> "Config":
        """Carga configuración desde archivo YAML."""
        with open(path, "r") as f:
            config_dict = yaml.safe_load(f)
        return cls(**config_dict)

    def to_yaml(self, path: Path) -> None:
        """Guarda configuración a archivo YAML."""
        with open(path, "w") as f:
            yaml.dump(self.model_dump(), f, default_flow_style=False)


def load_config(config_path: Optional[Path] = None) -> Config:
    """
    Carga la configuración del sistema.
    
    Args:
        config_path: Path al archivo de configuración. Si es None, usa el default.
    
    Returns:
        Objeto Config con la configuración cargada.
    """
    if config_path is None:
        config_path = Path(__file__).parent.parent.parent / "config" / "config.yaml"
    else:
        config_path = Path(config_path)
    
    if config_path.exists():
        return Config.from_yaml(config_path)
    else:
        return Config()
