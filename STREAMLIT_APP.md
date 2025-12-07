# 🎨 Aplicación Streamlit - Análisis de Ocupación de Anaqueles

Interfaz web interactiva para visualizar el pipeline completo de análisis de ocupación de anaqueles paso a paso.

## 🚀 Inicio Rápido

### Ejecutar la aplicación

```powershell
# Iniciar la aplicación Streamlit
uv run streamlit run app.py
```

La aplicación se abrirá automáticamente en tu navegador en `http://localhost:8501`

## 📖 Cómo usar la aplicación

### 1. **Seleccionar imagen**
   - En la barra lateral izquierda, usa el selector desplegable para elegir una imagen del dataset
   - Por defecto, se selecciona `test_49.jpg`
   - Hay más de 300 imágenes disponibles en el dataset SKU110K

### 2. **Analizar imagen**
   - Presiona el botón **🚀 Analizar Imagen** (verde) en la barra lateral
   - El sistema procesará la imagen automáticamente
   - Verás una barra de progreso durante el procesamiento

### 3. **Explorar resultados**

La aplicación muestra:

#### 📊 **Resumen de Resultados**
- Número de anaqueles detectados
- Ocupación promedio del anaquel completo
- Método de detección utilizado (Cuadriláteros o Bounding Boxes)

#### 🔄 **Pipeline de Procesamiento - 7 Pasos**

Cada paso se presenta con:
- **Visualización**: Imagen del resultado de ese paso
- **Explicación detallada**: Cómo ese paso contribuye al sistema completo
- **Métricas**: Información técnica específica del paso

Los 7 pasos del pipeline:

1. **📷 Paso 0: Imagen Original**
   - Punto de partida del análisis
   - Muestra la imagen sin procesar

2. **🔧 Paso 1: Preprocesamiento**
   - Conversión a escala de grises
   - Aplicación de filtro Gaussiano para reducir ruido

3. **🔍 Paso 2: Detección de Bordes**
   - Algoritmo Canny con umbrales automáticos
   - Identifica contornos de anaqueles y productos

4. **📐 Paso 3: Detección de Líneas**
   - Transformada de Hough
   - Líneas horizontales (verde) y verticales (azul)
   - Muestra ángulos dominantes

5. **📦 Paso 4: Detección de Anaqueles**
   - Segmentación en cuadriláteros inclinados
   - Cada anaquel etiquetado (S1, S2, S3...)

6. **🌊 Paso 5: Estimación de Profundidad**
   - Modelo Depth-Anything-V2
   - Colores cálidos = cerca (productos)
   - Colores fríos = lejos (vacío)

7. **📊 Paso 6: Análisis de Ocupación**
   - Resultado final con porcentajes
   - Código de colores:
     - 🟢 Verde: Alta ocupación (>70%)
     - 🟡 Amarillo: Ocupación media (30-70%)
     - 🔴 Rojo: Baja ocupación (<30%)

#### 📋 **Tabla de Resultados Detallados**
- Información por cada anaquel detectado
- Porcentajes de ocupación
- Estado (Alto/Medio/Bajo)
- Número de celdas ocupadas
- Desviación estándar

## 🎯 Características de la Aplicación

### ✨ Interfaz Interactiva
- Diseño responsivo de dos columnas
- Navegación intuitiva
- Visualizaciones de alta calidad

### 📚 Educativa
- Cada paso incluye explicación detallada
- Entiende cómo funciona el sistema completo
- Ideal para presentaciones y demostraciones

### 🎨 Diseño Profesional
- Esquema de colores coherente
- Cajas de explicación destacadas
- Métricas visuales atractivas

## 🛠️ Configuración Avanzada

### Puerto personalizado
```powershell
uv run streamlit run app.py --server.port 8080
```

### Modo headless (sin abrir navegador)
```powershell
uv run streamlit run app.py --server.headless true
```

### Habilitar CORS (para acceso remoto)
```powershell
uv run streamlit run app.py --server.enableCORS false
```

## 📁 Estructura de Archivos

```
shelf-occupancy-analyzer/
├── app.py                    # ← Aplicación Streamlit
├── visualize_pipeline.py     # Pipeline backend
├── data/
│   └── raw/
│       └── SKU110K_fixed/
│           └── images/       # Imágenes del dataset
└── config/
    └── config.yaml           # Configuración del sistema
```

## 🐛 Solución de Problemas

### Error: "No se encontraron imágenes"
- Verifica que exista la carpeta `data/raw/SKU110K_fixed/images/`
- Asegúrate de que contiene archivos `.jpg`

### Error: "ModuleNotFoundError"
- Ejecuta `uv sync` para instalar todas las dependencias
- Verifica que estés en el directorio correcto del proyecto

### La aplicación no carga
- Cierra otras instancias de Streamlit
- Prueba con un puerto diferente: `--server.port 8502`
- Verifica los logs en la terminal

### Procesamiento muy lento
- Primera ejecución: descarga el modelo Depth-Anything-V2 (~500MB)
- GPU recomendada pero funciona en CPU
- El procesamiento puede tardar 10-30 segundos por imagen

## 💡 Consejos de Uso

1. **Primera ejecución**: Espera a que se descargue el modelo de profundidad (~700MB)
2. **Explora diferentes imágenes**: Cada una tiene características únicas
3. **Lee las explicaciones**: Entender cada paso mejora el uso del sistema
4. **Compara resultados**: Prueba con `test_49.jpg`, `test_35.jpg`, `test_192.jpg`

## 📞 Soporte

Para reportar problemas o sugerir mejoras, consulta la documentación completa:
- **[README.md](README.md)**: Visión general del proyecto
- **[GETTING_STARTED.md](GETTING_STARTED.md)**: Guía técnica detallada
- **[INDEX.md](INDEX.md)**: Índice completo de documentación

---

**Versión de la Aplicación:** 2.0.0  
**Compatible con:** Pipeline v2.0.0 (Normalización Local por Cuadrilátero)
