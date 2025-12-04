# Dataset Images

Esta carpeta debe contener las imágenes del dataset SKU-110K.

**NOTA**: Las imágenes NO están incluidas en el repositorio Git debido a su tamaño (~2.5GB).

## Cómo obtener las imágenes

1. Descargar el dataset completo desde:
   - https://github.com/eg4000/SKU110K_CVPR19

2. Extraer las imágenes en esta carpeta:
   ```
   data/raw/SKU110K_fixed/images/
   ├── test_0.jpg
   ├── test_1.jpg
   ├── ...
   ├── train_0.jpg
   ├── train_1.jpg
   ├── ...
   ├── val_0.jpg
   ├── val_1.jpg
   └── ...
   ```

3. Verificar que tienes aproximadamente:
   - ~2900 imágenes de entrenamiento (train_*.jpg)
   - ~588 imágenes de validación (val_*.jpg)
   - ~2936 imágenes de prueba (test_*.jpg)

## Alternativa: Usar el script de descarga

```bash
cd shelf-occupancy-analyzer
uv run python src/shelf_occupancy/data/download_dataset.py
```

Este script descargará automáticamente el dataset y lo colocará en la ubicación correcta.
