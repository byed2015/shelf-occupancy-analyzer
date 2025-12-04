"""Script para descargar y preparar el dataset SKU-110K."""

import argparse
import shutil
import zipfile
from pathlib import Path
from typing import List, Optional

import requests
from loguru import logger
from tqdm import tqdm


# URLs del dataset SKU-110K
# Dataset completo: http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz
# Google Drive mirror: https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd
DATASET_URLS = {
    "s3_full": "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz",
    "gdrive_full": "1iq93lCdhaPUN0fWbLieMtzfB1850pKwd",  # Google Drive file ID
}

# URLs directas del S3 para imágenes de test (más ligero)
S3_BASE_URL = "http://trax-geometry.s3.amazonaws.com/cvpr_challenge/"


def download_file(url: str, destination: Path, timeout: int = 30) -> bool:
    """
    Descarga un archivo con barra de progreso.
    
    Args:
        url: URL del archivo
        destination: Path donde guardar el archivo
        timeout: Timeout en segundos
    
    Returns:
        True si la descarga fue exitosa
    """
    logger.info(f"Descargando {url}")
    
    try:
        response = requests.get(url, stream=True, timeout=timeout)
        response.raise_for_status()
    except requests.exceptions.RequestException as e:
        logger.error(f"Error descargando {url}: {e}")
        return False
    
    total_size = int(response.headers.get('content-length', 0))
    
    destination.parent.mkdir(parents=True, exist_ok=True)
    
    try:
        with open(destination, 'wb') as f, tqdm(
            desc=destination.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
            unit_divisor=1024,
        ) as pbar:
            for chunk in response.iter_content(chunk_size=8192):
                size = f.write(chunk)
                pbar.update(size)
        
        logger.info(f"Descarga completada: {destination}")
        return True
    except Exception as e:
        logger.error(f"Error guardando archivo: {e}")
        return False


def extract_archive(archive_path: Path, extract_to: Path) -> None:
    """
    Extrae un archivo comprimido.
    
    Args:
        archive_path: Path al archivo comprimido
        extract_to: Directorio donde extraer
    """
    logger.info(f"Extrayendo {archive_path.name}")
    
    extract_to.mkdir(parents=True, exist_ok=True)
    
    if archive_path.suffix == '.zip':
        with zipfile.ZipFile(archive_path, 'r') as zip_ref:
            zip_ref.extractall(extract_to)
    elif archive_path.suffix == '.gz' or '.tar' in archive_path.name:
        import tarfile
        with tarfile.open(archive_path, 'r:*') as tar_ref:
            tar_ref.extractall(extract_to)
    
    logger.info(f"Extracción completada: {extract_to}")


def get_sample_images(images_dir: Path, n_samples: int = 10) -> List[Path]:
    """
    Obtiene una muestra de imágenes del dataset.
    
    Args:
        images_dir: Directorio con imágenes
        n_samples: Número de imágenes a seleccionar
    
    Returns:
        Lista de paths a imágenes
    """
    image_extensions = {'.jpg', '.jpeg', '.png'}
    all_images = [
        p for p in images_dir.rglob('*')
        if p.suffix.lower() in image_extensions
    ]
    
    if not all_images:
        logger.warning(f"No se encontraron imágenes en {images_dir}")
        return []
    
    # Seleccionar muestra
    import random
    random.seed(42)  # Para reproducibilidad
    sample = random.sample(all_images, min(n_samples, len(all_images)))
    
    logger.info(f"Seleccionadas {len(sample)} imágenes de muestra")
    return sample


def setup_dataset(data_dir: Path, n_samples: int = 10, download: bool = True) -> Path:
    """
    Configura el dataset SKU-110K descargando el archivo completo.
    
    Args:
        data_dir: Directorio raíz de datos
        n_samples: Número de imágenes de muestra
        download: Si descargar el dataset (si no, busca localmente)
    
    Returns:
        Path al directorio con imágenes de muestra
    """
    raw_dir = data_dir / "raw"
    sample_dir = data_dir / "raw" / "sample"
    
    raw_dir.mkdir(parents=True, exist_ok=True)
    sample_dir.mkdir(parents=True, exist_ok=True)
    
    # Verificar si ya hay imágenes de muestra reales
    existing_samples = list(sample_dir.glob("*.jpg")) + list(sample_dir.glob("*.png"))
    if len(existing_samples) >= n_samples:
        # Verificar que no sean todas sintéticas
        non_synthetic = [s for s in existing_samples if "synthetic" not in s.name]
        if len(non_synthetic) >= n_samples:
            logger.info(f"Ya existen {len(non_synthetic)} imágenes reales de muestra")
            return sample_dir
    
    if download:
        logger.info("Intentando descargar dataset SKU-110K real...")
        logger.info("Dataset: Imágenes de anaqueles de retail con productos densamente empaquetados")
        
        # Intentar descargar el dataset completo desde S3
        dataset_archive = raw_dir / "SKU110K_fixed.tar.gz"
        
        if not dataset_archive.exists():
            logger.info("Descargando SKU-110K dataset completo (~7.4 GB)...")
            logger.info("Este proceso puede tardar varios minutos...")
            success = download_file(DATASET_URLS["s3_full"], dataset_archive, timeout=600)
            
            if not success:
                logger.warning("Descarga de S3 falló. Intentando método alternativo...")
                logger.info("NOTA: Puedes descargar manualmente desde:")
                logger.info("  1. S3: http://trax-geometry.s3.amazonaws.com/cvpr_challenge/SKU110K_fixed.tar.gz")
                logger.info("  2. Google Drive: https://drive.google.com/file/d/1iq93lCdhaPUN0fWbLieMtzfB1850pKwd")
                logger.info("Y colocarlo en: " + str(dataset_archive))
                logger.warning("Usando imágenes sintéticas mientras tanto...")
                return create_synthetic_samples(sample_dir, n_samples)
        else:
            logger.info(f"Archivo ya descargado: {dataset_archive}")
        
        # Extraer el dataset
        sku110k_dir = raw_dir / "SKU110K_fixed"
        if not sku110k_dir.exists() or not any(sku110k_dir.iterdir()):
            logger.info("Extrayendo SKU-110K dataset (esto puede tardar varios minutos)...")
            try:
                extract_archive(dataset_archive, raw_dir)
                logger.success("Dataset extraído exitosamente")
            except Exception as e:
                logger.error(f"Error extrayendo archivo: {e}")
                return create_synthetic_samples(sample_dir, n_samples)
        else:
            logger.info(f"Dataset ya extraído: {sku110k_dir}")
        
        # Buscar imágenes en el dataset
        # El dataset tiene estructura: SKU110K_fixed/images/ con subdirectorios train, test, val
        images_dirs = [
            sku110k_dir / "images" / "test",
            sku110k_dir / "images" / "val", 
            sku110k_dir / "test_images",
            sku110k_dir / "val_images",
            sku110k_dir,  # Buscar en raíz también
        ]
        
        images_source = None
        for img_dir in images_dirs:
            if img_dir.exists():
                imgs = list(img_dir.glob("*.jpg")) + list(img_dir.glob("*.png"))
                if imgs:
                    images_source = img_dir
                    logger.info(f"Encontradas {len(imgs)} imágenes en {img_dir}")
                    break
        
        if images_source is None:
            # Buscar recursivamente
            logger.info("Buscando imágenes recursivamente...")
            images_source = sku110k_dir
    else:
        # Buscar imágenes existentes
        images_source = raw_dir
    
    # Copiar muestra
    sample_images = get_sample_images(images_source, n_samples)
    
    if not sample_images:
        logger.warning("No se encontraron imágenes reales del dataset.")
        logger.info("Creando imágenes sintéticas de anaqueles...")
        return create_synthetic_samples(sample_dir, n_samples)
    
    # Limpiar imágenes sintéticas previas
    for old_synthetic in sample_dir.glob("synthetic_shelf_*.jpg"):
        old_synthetic.unlink()
        logger.debug(f"Eliminada imagen sintética: {old_synthetic.name}")
    
    # Copiar imágenes reales
    for i, img_path in enumerate(sample_images):
        dest = sample_dir / f"sku110k_sample_{i:03d}{img_path.suffix}"
        if not dest.exists():
            shutil.copy(img_path, dest)
            logger.info(f"✅ Copiada imagen real: {dest.name} (origen: {img_path.name})")
    
    logger.success(f"Dataset SKU-110K configurado en: {sample_dir}")
    logger.info(f"Total de imágenes reales de muestra: {len(list(sample_dir.glob('sku110k_sample_*')))}")
    
    return sample_dir


def create_synthetic_samples(sample_dir: Path, n_samples: int) -> Path:
    """
    Crea imágenes sintéticas de anaqueles.
    
    Args:
        sample_dir: Directorio donde guardar las muestras
        n_samples: Número de imágenes a crear
    
    Returns:
        Path al directorio con las muestras
    """
    logger.info(f"Creando {n_samples} imágenes sintéticas de anaqueles...")
    
    import cv2
    import numpy as np
    
    for i in range(n_samples):
        # Crear imagen sintética variada
        width = np.random.randint(600, 1000)
        height = np.random.randint(400, 700)
        
        img = create_synthetic_shelf(width, height, seed=42 + i)
        
        dest = sample_dir / f"synthetic_shelf_{i:03d}.jpg"
        cv2.imwrite(str(dest), img)
        logger.info(f"Creada imagen sintética: {dest.name}")
    
    return sample_dir


def create_synthetic_shelf(width: int, height: int, seed: int = 42) -> 'np.ndarray':
    """Crea una imagen sintética de anaquel."""
    import cv2
    import numpy as np
    
    np.random.seed(seed)
    
    # Imagen base con ruido
    img = np.ones((height, width, 3), dtype=np.uint8) * 200
    noise = np.random.randint(-30, 30, (height, width, 3), dtype=np.int16)
    img = np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8)
    
    # Número aleatorio de anaqueles
    n_shelves = np.random.randint(3, 6)
    shelf_heights = np.linspace(height // 6, height - height // 6, n_shelves).astype(int)
    
    # Dibujar líneas de anaqueles
    for y in shelf_heights:
        thickness = np.random.randint(2, 5)
        cv2.line(img, (0, y), (width, y), (50, 50, 50), thickness)
    
    # Divisiones verticales aleatorias
    n_divisions = np.random.randint(3, 8)
    for x in np.linspace(0, width, n_divisions).astype(int):
        cv2.line(img, (x, 0), (x, height), (60, 60, 60), 2)
    
    # Simular productos
    colors = [
        (180, 100, 100), (100, 180, 100), (100, 100, 180),
        (180, 180, 100), (180, 100, 180), (100, 180, 180)
    ]
    
    for shelf_y in shelf_heights[:-1]:
        n_products = np.random.randint(3, 8)
        for _ in range(n_products):
            x = np.random.randint(10, width - 100)
            y = shelf_y + np.random.randint(10, min(150, shelf_heights[1] - shelf_heights[0] - 20))
            w = np.random.randint(40, 100)
            h = np.random.randint(60, 150)
            
            # Asegurar que no se sale de la imagen
            if y + h > height:
                h = height - y - 10
            
            color = colors[np.random.randint(0, len(colors))]
            cv2.rectangle(img, (x, y), (x + w, y + h), color, -1)
            cv2.rectangle(img, (x, y), (x + w, y + h), (0, 0, 0), 2)
    
    # Variación de iluminación
    overlay = np.zeros_like(img)
    center_x = np.random.randint(width // 4, 3 * width // 4)
    center_y = np.random.randint(height // 4, 3 * height // 4)
    radius = np.random.randint(300, 500)
    cv2.circle(overlay, (center_x, center_y), radius, (255, 255, 255), -1)
    img = cv2.addWeighted(img, 0.7, overlay, 0.3, 0)
    
    return img


def main():
    """Función principal."""
    parser = argparse.ArgumentParser(description="Descarga y prepara el dataset SKU-110K")
    parser.add_argument(
        "--data-dir",
        type=Path,
        default=Path("data"),
        help="Directorio de datos"
    )
    parser.add_argument(
        "--n-samples",
        type=int,
        default=10,
        help="Número de imágenes de muestra"
    )
    parser.add_argument(
        "--no-download",
        action="store_true",
        help="No descargar, usar imágenes existentes"
    )
    
    args = parser.parse_args()
    
    # Configurar logging
    logger.add("logs/dataset_setup.log", rotation="10 MB")
    
    try:
        sample_dir = setup_dataset(
            args.data_dir,
            args.n_samples,
            download=not args.no_download
        )
        logger.success(f"Dataset listo en: {sample_dir}")
    except Exception as e:
        logger.error(f"Error configurando dataset: {e}")
        raise


if __name__ == "__main__":
    main()
