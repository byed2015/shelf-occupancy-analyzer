# 📦 Shelf Occupancy Analyzer - Índice de Documentación

**Versión**: 2.0.0 (Normalización Local + Pipeline Simplificado) | **Estado**: ✅ Producción

---

## 🚀 Inicio Rápido

**¿Primera vez aquí?** Empieza por estos archivos en orden:

1. **[README.md](README.md)** - Overview general del proyecto
2. **[GETTING_STARTED.md](GETTING_STARTED.md)** - Guía técnica completa
3. **[PIPELINE_OPTIMIZATION.md](PIPELINE_OPTIMIZATION.md)** - Optimizaciones v2.0.0 ⭐ NUEVO
4. **[STREAMLIT_APP.md](STREAMLIT_APP.md)** - Guía de la aplicación Streamlit

---

## 📚 Documentación Principal

### Para Usuarios

| Archivo | Descripción | Para quién |
|---------|-------------|------------|
| **README.md** | Descripción general, instalación, uso básico | Todos los usuarios |
| **GETTING_STARTED.md** | Tutorial técnico completo con ejemplos | Desarrolladores |
| **PIPELINE_OPTIMIZATION.md** ⭐ | Auditoría y optimización del pipeline (v2.0.0) | Técnicos/Investigadores |
| **MEJORAS_IMPLEMENTADAS.md** | Detalles del sistema de refinamiento (v1.1.0) | Interesados en técnicas ML |
| **STREAMLIT_APP.md** | Guía completa de la aplicación web | Usuarios finales |
| **INDEX.md** | Índice de documentación (este archivo) | Todos |

### Para Desarrolladores

| Archivo | Descripción |
|---------|-------------|
| **shelf_occupancy_inference.py** | API simplificada para Streamlit |
| **visualize_pipeline.py** | Pipeline completo con visualización |
| **process_all_images.py** | Procesamiento batch |
| **app.py** | Aplicación Streamlit completa |
| **config/config.yaml** | Configuración centralizada |

---

## 🗂️ Estructura del Proyecto

```
shelf-occupancy-analyzer/
│
├── 📖 README.md                      ← Empieza aquí
├── 📚 GETTING_STARTED.md             ← Guía técnica completa
├── 📋 MEJORAS_IMPLEMENTADAS.md       ← Detalles del refinamiento
│
├── 🎨 visualize_pipeline.py          ← Script principal
├── 📦 process_all_images.py          ← Procesamiento batch
├── 🔌 shelf_occupancy_inference.py   ← API para Streamlit
│
├── ⚙️ config/
│   └── config.yaml                   ← Configuración central
│
├── 💻 src/shelf_occupancy/           ← Código fuente
│   ├── preprocessing/                ← Gaussian Blur (simplificado)
│   ├── detection/                    ← Bordes, líneas, cuadriláteros
│   ├── depth/                        ← Depth-Anything-V2
│   ├── analysis/                     ← Cuadrículas + normalización local
│   ├── visualization/                ← Overlays y heatmaps
│   └── utils/                        ← BoundingBox, Quadrilateral, I/O
│
├── 📊 data/
│   ├── raw/                          ← Imágenes originales
│   │   └── SKU110K_fixed/            ← Dataset
│   └── results/                      ← Salidas generadas
│       └── examples/                 ← Ejemplos de referencia
│
├── 📓 notebooks/                     ← Experimentación Jupyter
│
├── 🧪 tests/                         ← Tests unitarios
│
├── 📝 logs/                          ← Logs de ejecución
│
├── 🔧 requirements.txt               ← Dependencias pip
├── 🔧 pyproject.toml                 ← Configuración uv
└── 🔧 .gitignore                     ← Git ignore
```

---

## 🎯 Flujos de Trabajo Comunes

### 1. Procesar una Imagen

```powershell
uv run python visualize_pipeline.py \
  --image "data/raw/SKU110K_fixed/images/test_117.jpg" \
  --output-dir "data/results/mi_analisis"
```

Ver: [README.md § Uso Rápido](README.md#uso-rápido)

### 2. Procesamiento Batch

```powershell
uv run python process_all_images.py \
  --input-dir "data/raw/SKU110K_fixed/images" \
  --max-images 20
```

Ver: [GETTING_STARTED.md § Procesamiento Batch](GETTING_STARTED.md#procesamiento-batch)

### 3. Uso Programático

```python
from shelf_occupancy_inference import ShelfOccupancyAnalyzer

analyzer = ShelfOccupancyAnalyzer()
results = analyzer.process("imagen.jpg")
print(f"Ocupación: {results['avg_occupancy']:.1f}%")
```

Ver: [GETTING_STARTED.md § API de Inferencia](GETTING_STARTED.md#api-de-inferencia)

### 4. Integración con Streamlit

Ver ejemplo completo en: [GETTING_STARTED.md § Integración con Streamlit](GETTING_STARTED.md#integración-con-streamlit)

---

## 🔧 Configuración

Todas las configuraciones están centralizadas en `config/config.yaml`.

**Ajustes comunes**:

- **Para perspectivas extremas**: `shelf_detection.clustering.eps: 70`
- **Para mayor precisión**: `occupancy_analysis.grid_size: [15, 8]`
- **Para usar GPU**: `depth_estimation.device: "cuda"`

Ver guía completa: [GETTING_STARTED.md § Configuración Avanzada](GETTING_STARTED.md#configuración-avanzada)

---

## 🧪 Testing y Desarrollo

```powershell
# Ejecutar tests
uv run pytest

# Con cobertura
uv run pytest --cov=src --cov-report=html

# Test específico
uv run pytest tests/test_preprocessing.py -v
```

Ver: [GETTING_STARTED.md § Desarrollo y Testing](GETTING_STARTED.md#desarrollo-y-testing)

---

## 🆘 Ayuda y Troubleshooting

### Problemas Comunes

1. **"CUDA no disponible"** → Sistema usa CPU automáticamente (OK)
2. **"No se encontraron imágenes"** → Descargar dataset: `uv run python -m shelf_occupancy.data.download_dataset --n-samples 10`
3. **Ocupación incorrecta** → Ajustar `config.yaml` según [GETTING_STARTED.md § Troubleshooting](GETTING_STARTED.md#troubleshooting)

### Más Ayuda

- **Guía técnica completa**: [GETTING_STARTED.md](GETTING_STARTED.md)
- **Notebooks de ejemplo**: Carpeta `notebooks/`
- **Logs detallados**: Carpeta `logs/`

---

## 📦 Para Deployment

### Preparación para Streamlit

1. **Instalar Streamlit**: `pip install streamlit`
2. **Crear app** usando ejemplo en [GETTING_STARTED.md](GETTING_STARTED.md)
3. **Ejecutar**: `streamlit run streamlit_app.py`

### Archivos Necesarios

- ✅ `shelf_occupancy_inference.py` - API lista
- ✅ `requirements.txt` - Dependencias completas
- ✅ `config/config.yaml` - Configuración
- ✅ Código fuente en `src/shelf_occupancy/`

---

## 📚 Referencias Técnicas

- **Modelo de Profundidad**: [Depth-Anything-V2](https://huggingface.co/depth-anything/Depth-Anything-V2-Small-hf)
- **Dataset**: [SKU-110K](https://github.com/eg4000/SKU110K_CVPR19)
- **OpenCV Docs**: [Hough Transform](https://docs.opencv.org/4.x/d9/db0/tutorial_hough_lines.html)

---

## 📝 Historial de Versiones

- **v2.0.0** (Dic 2024): Normalización local por cuadrilátero, pipeline simplificado, 20% más rápido
- **v1.2.0** (Dic 2024): Arquitectura de cuadriláteros, filtrado absoluto, sin corrección perspectiva
- **v1.1.0** (Dic 2024): Sistema de refinamiento integrado (~20% mejora)
- **v1.0.0** (Nov 2024): Pipeline base con Depth-Anything-V2

---

**Mantenedor**: Proyecto Final - Visión Computarizada  
**Licencia**: MIT  
**Última Actualización**: Diciembre 2024
