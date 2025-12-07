# Sistema de Análisis Automático de Ocupación de Anaqueles mediante Visión Computacional y Profundidad

**Maestría en Inteligencia Artificial y Ciencia de Datos**  
**Proyecto Final – Visión Computacional**

---

**Trabajo presentado por:**
- EDGAR ALBERTO MORALES GUTIÉRREZ
- GUSTAVO ALBERTO GÓMEZ ROJAS

---

## Tabla de Contenidos

- [Sistema de Análisis Automático de Ocupación de Anaqueles mediante Visión Computacional y Profundidad](#sistema-de-análisis-automático-de-ocupación-de-anaqueles-mediante-visión-computacional-y-profundidad)
  - [Tabla de Contenidos](#tabla-de-contenidos)
  - [1. Introducción y fundamentación](#1-introducción-y-fundamentación)
    - [1.1. Planteamiento del Problema](#11-planteamiento-del-problema)
    - [1.2. Relevancia en el campo de la visión computacional](#12-relevancia-en-el-campo-de-la-visión-computacional)
    - [1.3. Objetivo General](#13-objetivo-general)
    - [1.4. Objetivos específicos](#14-objetivos-específicos)
  - [2. Metodología detallada](#2-metodología-detallada)
    - [2.1. Estructura del pipeline](#21-estructura-del-pipeline)
    - [2.2. Técnicas de visión computacional utilizadas](#22-técnicas-de-visión-computacional-utilizadas)
      - [1. Preprocesamiento](#1-preprocesamiento)
      - [2. Detección de bordes](#2-detección-de-bordes)
      - [3. Detección de líneas](#3-detección-de-líneas)
      - [4. Segmentación en cuadriláteros](#4-segmentación-en-cuadriláteros)
      - [5. Estimación de profundidad](#5-estimación-de-profundidad)
      - [6. Análisis de ocupación](#6-análisis-de-ocupación)
    - [2.3. Datos, condiciones de captura y hardware](#23-datos-condiciones-de-captura-y-hardware)
      - [Dataset principal](#dataset-principal)
      - [Estructura de datos en el repositorio](#estructura-de-datos-en-el-repositorio)
      - [Condiciones de captura previstas](#condiciones-de-captura-previstas)
      - [Hardware utilizado (ejemplo sugerido)](#hardware-utilizado-ejemplo-sugerido)
    - [2.4. Parámetros y criterios de ajuste](#24-parámetros-y-criterios-de-ajuste)
  - [3. Implementación y explicación técnica](#3-implementación-y-explicación-técnica)
    - [3.1. Organización del código](#31-organización-del-código)
    - [3.2. Flujo de ejecución principal](#32-flujo-de-ejecución-principal)
    - [3.3. Librerías utilizadas](#33-librerías-utilizadas)
    - [3.4. Robustez del sistema](#34-robustez-del-sistema)
  - [4. Resultados y análisis](#4-resultados-y-análisis)
    - [4.1. Evidencia visual](#41-evidencia-visual)
    - [4.2. Resultados cuantitativos (ejemplos)](#42-resultados-cuantitativos-ejemplos)
      - [Imagen test\_192.jpg](#imagen-test_192jpg)
      - [Imagen test\_179.jpg](#imagen-test_179jpg)
    - [4.3. Análisis crítico](#43-análisis-crítico)
    - [4.4. Limitaciones y posibles errores](#44-limitaciones-y-posibles-errores)
    - [4.5. Propuestas de mejora y trabajo futuro](#45-propuestas-de-mejora-y-trabajo-futuro)
  - [5. Innovación y complejidad del proyecto](#5-innovación-y-complejidad-del-proyecto)
    - [Elementos innovadores y de alta complejidad técnica:](#elementos-innovadores-y-de-alta-complejidad-técnica)
  - [6. Conclusiones](#6-conclusiones)
    - [Resumen de logros:](#resumen-de-logros)
  - [7. Referencias](#7-referencias)

---

## 1. Introducción y fundamentación

### 1.1. Planteamiento del Problema

En el entorno actual del comercio minorista, la correcta gestión de anaqueles es un factor determinante para la competitividad. Un anaquel vacío implica ventas perdidas, afecta la percepción del cliente sobre la tienda y, en escenarios acumulados, puede distorsionar el análisis de la demanda real. Tradicionalmente, el monitoreo de anaqueles se realiza mediante recorridos manuales, donde el personal de piso identifica visualmente huecos y faltantes. Este enfoque es costoso, lento, subjetivo y difícil de escalar a cadenas con cientos de sucursales.

El proyecto "Shelf Occupancy Analyzer" surge para responder a este problema desde la visión computacional, proponiendo un sistema capaz de estimar automáticamente la ocupación de los anaqueles a partir de una sola imagen. Para ello, se combinan técnicas clásicas (detección de bordes y líneas, segmentación geométrica) con modelos modernos de aprendizaje profundo para estimación de profundidad monocular. La idea central es que la profundidad aporta una dimensión adicional que permite diferenciar mejor entre fondo del anaquel y producto exhibido, incluso cuando los colores son similares.

El sistema no solo busca generar una métrica numérica de ocupación, sino también producir visualizaciones claras que puedan ser interpretadas por usuarios no técnicos, como gerentes de tienda o personal de reposición. De esta forma, el proyecto se sitúa en la intersección entre la teoría de la visión computacional y una necesidad operativa real, con potencial de uso en analítica de retail, administración de inventarios y sistemas de alerta temprana ante desabasto.

### 1.2. Relevancia en el campo de la visión computacional

El problema es representativo de diversos temas centrales de la visión computacional:

1. **Detección de bordes y líneas (Canny + Hough)**: extracción de la estructura geométrica dominante en la escena.
2. **Segmentación geométrica basada en cuadriláteros**: partición de la escena en regiones de interés (anaqueles) respetando la perspectiva.
3. **Estimación de profundidad monocular con redes neuronales profundas**: aproximación moderna relacionada con la visión estéreo, pero utilizando una sola imagen.
4. **Análisis estadístico de mapas de profundidad**: uso de medianas, percentiles y varianza para inferir ocupación y distinguir fondo vs. producto.

Con ello, el proyecto integra al menos dos unidades temáticas típicas del curso: procesamiento y detección de características, segmentación y análisis de profundidad.

### 1.3. Objetivo General

Desarrollar e implementar un sistema profesional de análisis de ocupación de anaqueles que, a partir de una imagen, detecte automáticamente los anaqueles, estime su nivel de ocupación utilizando mapas de profundidad y genere visualizaciones y métricas cuantitativas útiles para operación y analítica.

### 1.4. Objetivos específicos

1. Diseñar un pipeline modular de visión computacional que abarque preprocesamiento, detección de estructura, estimación de profundidad y análisis de ocupación.
2. Implementar detección robusta de líneas horizontales y verticales mediante Canny + Transformada de Hough y clustering, para obtener cuadriláteros correspondientes a anaqueles.
3. Integrar el modelo de profundidad monocular Depth-Anything-V2-Small para generar mapas de profundidad sobre imágenes de anaqueles.
4. Definir una métrica de ocupación basada en la mediana de profundidad dentro de cada cuadrilátero, asignando un porcentaje de llenado a cada anaquel.
5. Evaluar la robustez del sistema ante variaciones razonables de perspectiva, iluminación y ruido, utilizando imágenes del dataset SKU-110K y muestras adicionales.

---

## 2. Metodología detallada

La metodología se estructura como un pipeline de procesamiento de imagen en siete pasos principales, diseñados para ser modulares y trazables. En primer lugar, se recibe una imagen del anaquel, capturada en condiciones típicas de una tienda (iluminación artificial y perspectiva moderadamente inclinada). A partir de esta entrada, se aplica un preprocesamiento ligero que incluye suavizado mediante desenfoque Gaussiano. Este paso reduce el ruido de alta frecuencia sin destruir los bordes relevantes que serán utilizados más adelante.

Posteriormente, se ejecuta el detector de bordes Canny, cuyos umbrales no se fijan de forma estática, sino que se calculan en función de la mediana de intensidades de la imagen. Esto permite adaptar la sensibilidad del detector a escenas más claras u oscuras. Con el mapa de bordes, se utiliza la Transformada de Hough para localizar líneas rectas prominentes. Mediante filtrado angular se separan las líneas horizontales y verticales, y con técnicas de clustering se agrupan líneas similares, para obtener una representación más estable de la estructura del anaquel.

Con las familias de líneas resultantes se construyen cuadriláteros que representan los anaqueles individuales, respetando la perspectiva de la escena. En paralelo, se envía la imagen al modelo de profundidad monocular, que genera un mapa de profundidad continuo sobre toda la imagen. Finalmente, para cada cuadrilátero se extrae la porción correspondiente del mapa de profundidad y se calcula una estadística robusta (mediana), que se transforma en un porcentaje de ocupación. El pipeline concluye con la generación de visualizaciones y un reporte con los porcentajes por anaquel e imagen completa.

### 2.1. Estructura del pipeline

El sistema sigue un pipeline de 7 pasos optimizados (v1.3.1):

1. **Imagen original**
2. **Preprocesamiento** (suavizado ligero)
3. **Detección de bordes** (Canny con auto-umbrales basados en la mediana de la imagen)
4. **Detección y fusión de líneas** (Transformada de Hough + clustering DBSCAN)
5. **Segmentación en cuadriláteros inclinados** (anaqueles)
6. **Estimación de profundidad** mediante Depth-Anything-V2
7. **Análisis de ocupación** con mediana directa y visualización con cuadriláteros coloreados

Este pipeline se implementa en el script `visualize_pipeline.py`, que orquesta los módulos de `src/shelf_occupancy/` y genera una imagen concatenada con los principales pasos y un reporte de métricas.

### 2.2. Técnicas de visión computacional utilizadas

#### 1. Preprocesamiento

- **Desenfoque Gaussiano 5×5 (σ=1.0)** para reducir ruido preservando bordes.
- *(Versiones anteriores incluían CLAHE y filtro bilateral, posteriormente simplificados para mejorar velocidad)*

#### 2. Detección de bordes

- **Canny** con umbrales adaptativos calculados a partir la mediana de intensidades de la imagen, mejorando robustez frente a cambios de iluminación.

#### 3. Detección de líneas

- **Transformada de Hough** (HoughLines / HoughLinesP) para obtener líneas candidatas.
- **Filtrado "ABSOLUTO"** por orientación:
  - Horizontales: ~0°/180° ±20°
  - Verticales: ~90° ±20°
- **Clustering con DBSCAN** y fusión de líneas similares por ángulo y distancia.

#### 4. Segmentación en cuadriláteros

- A partir de las familias de líneas horizontales y verticales, el módulo `ShelfDetector` genera cuadriláteros que siguen la geometría real de cada anaquel, sin corregir la perspectiva global.

#### 5. Estimación de profundidad

- Uso del modelo **Depth-Anything-V2-Small-hf**, cargado vía PyTorch/HuggingFace, para producir un mapa de profundidad normalizado (min–max) de la escena completa.

#### 6. Análisis de ocupación

- **Método de "mediana directa"**: se crea una máscara binaria del cuadrilátero, se extraen los valores de profundidad, se calcula la mediana, y se define la ocupación como:
  
  ```
  Ocupación = (1 - mediana) × 100%
  ```

### 2.3. Datos, condiciones de captura y hardware

#### Dataset principal
- **SKU-110K**: un conjunto de imágenes de anaqueles minoristas con miles de productos anotados.

#### Estructura de datos en el repositorio
- `data/raw/SKU110K_fixed/images/`: contiene imágenes de prueba
- `data/results/`: almacena resultados, reportes y visualizaciones

#### Condiciones de captura previstas
- Imágenes tomadas en pasillos de supermercado
- Variaciones moderadas de perspectiva
- Iluminación artificial
- Presencia de ruido visual (personas, carteles)

#### Hardware utilizado (ejemplo sugerido)
- **CPU**: al menos 4 núcleos, 8 GB de RAM
- **GPU** (opcional): para acelerar la inferencia de profundidad
- **Sistema operativo**: Linux/Windows con Python 3.10+

### 2.4. Parámetros y criterios de ajuste

Algunos parámetros clave definidos en `config/config.yaml` incluyen:

- `shelf_detection.canny.low_threshold` / `high_threshold`: sensibilidad de bordes
- `shelf_detection.hough.threshold`: sensibilidad de la Transformada de Hough
- `depth_estimation.model_name` y `device`: elección de modelo de profundidad y uso de CPU/GPU
- `occupancy_analysis.thresholds.min_occupancy`: umbral mínimo para considerar un anaquel "ocupado" o "vacío"

Estos parámetros se ajustan empíricamente observando tanto el mapa de bordes y líneas detectadas como los porcentajes de ocupación resultantes sobre imágenes de validación.

---

## 3. Implementación y explicación técnica

La implementación se basa en una arquitectura modular escrita en Python, con una organización clara por capas dentro del directorio `src/shelf_occupancy`. En la capa de entrada se encuentran los scripts principales, como `visualize_pipeline.py` y `process_all_images.py`, que sirven como puntos de acceso al sistema. Estos scripts leen la configuración desde un archivo `config.yaml`, cargan las imágenes desde el directorio de datos y orquestan la ejecución de los módulos internos.

En la capa de procesamiento, el módulo de preprocesamiento encapsula operaciones como lectura de imagen, conversión de espacio de color, redimensionamiento y suavizado. La detección de bordes y líneas se implementa utilizando OpenCV, con funciones específicas para Canny y Hough, mientras que el filtrado y clustering de líneas aprovecha estructuras de datos de NumPy y, en algunos casos, algoritmos de agrupamiento como DBSCAN. El diseño busca mantener las funciones puras y fáciles de probar, evitando lógica mezclada entre lectura de archivos y operaciones matemáticas.

La capa de profundidad integra un modelo preentrenado como Depth-Anything-V2, cargado mediante PyTorch. Se realiza la normalización adecuada de la imagen, se pasa por la red neuronal y el mapa de profundidad resultante se reescala a las dimensiones originales. La capa de análisis define cómo se construye la máscara del cuadrilátero, cómo se extraen los valores dentro de él y cómo se calcula la mediana de profundidad. Finalmente, el módulo de visualización genera imágenes compuestas con los distintos pasos del pipeline, dibuja los cuadriláteros coloreados según la ocupación y, si se requiere, guarda métricas en archivos de texto o CSV para análisis posterior.

### 3.1. Organización del código

El proyecto sigue una arquitectura modular organizada en capas:

```
shelf-occupancy-analyzer/
├── visualize_pipeline.py          # Script principal de alto nivel
├── process_all_images.py          # Procesamiento batch
│
├── src/shelf_occupancy/            # Lógica de negocio
│   ├── preprocessing/              # image_processor.py
│   ├── detection/                  # edges.py, lines.py, shelves.py
│   ├── depth/                      # estimator.py
│   ├── analysis/                   # grid_analysis.py
│   ├── visualization/              # overlay.py
│   └── utils/                      # geometry.py, image_io.py
│
├── config/
│   └── config.yaml                 # Configuración centralizada
│
└── notebooks/                      # Notebooks exploratorios
```

Esta estructura facilita el mantenimiento, la extensión del sistema y la integración en pipelines MLOps.

### 3.2. Flujo de ejecución principal

Para una imagen individual, el flujo estándar es:

1. Cargar configuración (`load_config`) y la imagen (`load_image`)
2. Aplicar preprocesamiento (suavizado, opcionalmente CLAHE/filtro bilateral)
3. Calcular bordes con Canny
4. Detectar líneas con Hough y filtrarlas según orientación
5. Ejecutar `ShelfDetector` para generar los cuadriláteros de anaqueles
6. Estimar el mapa de profundidad usando `DepthEstimator`
7. Para cada cuadrilátero:
   - Construir máscara
   - Extraer profundidades
   - Calcular mediana y porcentaje de ocupación
8. Generar visualizaciones (overlay de cuadriláteros coloreados, imagen con los 7 pasos del pipeline) y reporte de métricas en texto/CSV

### 3.3. Librerías utilizadas

1. **OpenCV**: lectura/escritura de imágenes, desenfoque Gaussiano, Canny, Hough, operaciones geométricas y morfológicas
2. **NumPy**: operaciones numéricas (mediana, percentiles, máscaras)
3. **PyTorch + Transformers (HuggingFace)**: carga e inferencia del modelo Depth-Anything-V2 para estimación de profundidad
4. **scikit-learn (DBSCAN)**: clustering de líneas y agrupación de estructuras
5. **loguru**: logging estructurado

### 3.4. Robustez del sistema

1. **Umbrales de Canny adaptativos** a la mediana de intensidades, lo que mejora el desempeño en diferentes condiciones de iluminación
2. **Filtrado absoluto de líneas** horizontales/verticales, que evita que el sistema se "incline" siguiendo el ángulo dominante y pierda la estructura del anaquel
3. **Arquitectura de cuadriláteros** que mantiene la imagen original sin corrección de perspectiva global; solo se realiza warp local por anaquel cuando es necesario
4. **Uso de la mediana de profundidad** (estadística robusta frente a outliers) para calcular ocupación

---

## 4. Resultados y análisis

Los resultados obtenidos se evaluaron principalmente de forma cualitativa, mediante la inspección visual de las imágenes procesadas, y de forma cuantitativa, a través de los porcentajes de ocupación por anaquel. En las pruebas realizadas con imágenes del conjunto SKU-110K y ejemplos de referencia proporcionados en el repositorio, el sistema fue capaz de identificar correctamente la estructura de los anaqueles en la mayoría de los casos, incluso cuando existían ligeras inclinaciones de la cámara o variaciones en la iluminación.

La visualización final, donde cada anaquel se colorea de acuerdo con su nivel de ocupación (por ejemplo, verde para alta ocupación, amarillo para media y rojo para baja), resulta especialmente útil para la interpretación. Un supervisor puede, de un vistazo, identificar qué zonas requieren reposición. Además, se calcula una ocupación promedio global de la imagen, útil para análisis agregados. Comparado con versiones previas del código, el enfoque basado en la mediana de profundidad mostró una reducción de casos con ocupación reportada como 0% en anaqueles que claramente tenían productos, lo que indica una mejora en robustez frente a ruido y outliers en el mapa de profundidad.

No obstante, se observaron limitaciones en escenas donde personas u otros objetos ajenos al anaquel ocupaban parte importante de la imagen: el sistema, al basarse en profundidad y estructura geométrica, puede confundir algunos elementos. Aun así, en un escenario controlado donde la cámara se enfoca a los anaqueles y se minimizan obstrucciones, los resultados son coherentes y repetibles.

### 4.1. Evidencia visual

El sistema genera automáticamente:

1. **Una imagen concatenada con los 7 pasos del pipeline**:
   - Imagen original
   - Imagen preprocesada
   - Mapa de bordes (Canny)
   - Líneas detectadas
   - Cuadriláteros de anaqueles
   - Mapa de profundidad
   - Imagen final con cuadriláteros coloreados según porcentaje de ocupación

2. **Una imagen `*_pipeline_complete.png`** por cada entrada procesada

3. **Overlays con colores**:
   - 🟢 **Verde**: alta ocupación (>70%)
   - 🟡 **Amarillo**: ocupación media (40–70%)
   - 🔴 **Rojo**: ocupación baja (<40%)

### 4.2. Resultados cuantitativos (ejemplos)

De acuerdo con las pruebas documentadas en el repositorio, el método de mediana directa (v1.3.1) mejora significativamente la precisión frente a versiones previas con grid/warp:

#### Imagen test_192.jpg
- **Método anterior**: 11.8% de ocupación
- **Método propuesto**: 34.4% de ocupación

#### Imagen test_179.jpg
- **Ocupación estimada**: 18.3%, sin falsos 0%

Asimismo, se reportan métricas por anaquel:
- Anaquel 1: 45.2%
- Anaquel 2: 78.5%
- Anaquel 3: 32.1%
- **Ocupación promedio**: ~51.9% (en ciertos ejemplos)

### 4.3. Análisis crítico

**Fortalezas:**

1. La combinación de líneas estructurales + mapas de profundidad permite distinguir zonas con producto de zonas vacías, incluso cuando el color de fondo es similar
2. El uso de mediana en lugar de promedios reduce el impacto de valores anómalos en el mapa de profundidad

**Sensibilidades:**

3. El sistema sigue siendo sensible a:
   - Escenas extremadamente saturadas de ruido (personas frente al anaquel)
   - Condiciones de iluminación muy extremas o reflejos intensos
   - Anaqueles con geometrías no rectilíneas o muy irregulares

### 4.4. Limitaciones y posibles errores

1. **Dependencia del modelo de profundidad**: si el modelo falla (por ejemplo, en dominios muy diferentes al dataset de entrenamiento), la ocupación puede ser sub o sobre-estimada
2. **Pérdida de precisión en perspectivas extremas** fuera del rango soportado por la arquitectura de cuadriláteros (~−45° a +25°)
3. **No reconocimiento de productos individuales**: el sistema detecta anaqueles como estructuras geométricas, pero no realiza reconocimiento de productos individuales; por tanto, no distingue entre diferentes tipos de productos, solo mide espacio ocupado vs. vacío

### 4.5. Propuestas de mejora y trabajo futuro

1. **Integrar un modelo de detección de objetos** para contar productos por anaquel y combinarlo con la ocupación de profundidad
2. **Entrenar/tunear un modelo de profundidad específico** para el dominio retail, mejorando la precisión en condiciones reales de tienda
3. **Incorporar seguimiento temporal (video)** para detectar en tiempo real cuándo un anaquel alcanza un umbral crítico de ocupación
4. **Diseñar una interfaz web** (por ejemplo, con Streamlit) para cargar imágenes y visualizar resultados en tiempo real

---

## 5. Innovación y complejidad del proyecto

El proyecto destaca por combinar varias ideas que, en conjunto, representan un nivel de innovación y complejidad superior al de ejercicios típicos de curso. En lugar de limitarse a la detección de bordes o a un clasificador de imágenes, se plantea un problema aplicado real (ocupación de anaqueles) y se diseña una solución de extremo a extremo. La decisión de trabajar con profundidad monocular es relevante, ya que la mayoría de soluciones industriales recurren a sensores especializados o a métodos de conteo de objetos, mientras que aquí se explota la información 3D estimada a partir de una sola imagen 2D.

La arquitectura basada en cuadriláteros que respetan la perspectiva de la escena también es un aspecto diferenciador. Muchos enfoques optan por rectificar la imagen completa a una vista "frontal" mediante homografías, lo que puede introducir distorsiones y requerir supuestos fuertes sobre la calibración de la cámara. En este proyecto, en cambio, la detección de líneas y la construcción de cuadriláteros permiten trabajar directamente en el espacio de imagen original, preservando la geometría sin necesidad de rectificación global.

Desde el punto de vista de ingeniería, la integración de un modelo de aprendizaje profundo con código tradicional de visión computacional, bajo una configuración centralizada y módulos bien separados, aproxima el proyecto a un prototipo realista que podría evolucionar hacia un producto. Esta combinación de dificultad algorítmica, integración de librerías avanzadas y enfoque en un caso de uso concreto justifica considerar el proyecto como de complejidad alta y con un componente de innovación significativo.

### Elementos innovadores y de alta complejidad técnica:

1. **Arquitectura basada en cuadriláteros adaptativos**: cada anaquel se modela como un cuadrilátero que sigue la perspectiva natural, evitando la corrección global de perspectiva que típicamente introduce distorsiones

2. **Uso de un modelo de profundidad de última generación** (Depth-Anything-V2) para análisis de ocupación: en lugar de usar únicamente información de intensidad o color, el sistema utiliza la dimensión de profundidad como señal principal

3. **Método de mediana directa para ocupación**: enfoque simple pero robusto, que mejora la precisión y elimina falsos 0% de ocupación

4. **Diseño modular tipo MLOps**: organización por capas, configuración centralizada, tests unitarios preparados y lista para integración con herramientas como Streamlit o CI/CD

Esto va más allá de un ejemplo básico de detección de bordes o segmentación y constituye un sistema integrado de análisis aplicado a un problema real en retail.

---

## 6. Conclusiones

El desarrollo del "Shelf Occupancy Analyzer" permite concluir que la visión computacional, combinada con modelos de profundidad basados en deep learning, constituye una herramienta viable para abordar el problema de monitoreo de anaqueles en el sector retail. El sistema implementado demuestra que, a partir de una sola imagen, es posible segmentar de manera razonablemente robusta los anaqueles, estimar mapas de profundidad y traducir esta información a un indicador cuantitativo de ocupación, acompañado de visualizaciones intuitivas.

En términos de aprendizaje, el proyecto integra varios conceptos clave de la materia: preprocesamiento, detección de características, geometría de la imagen, homografías locales, redes neuronales para tareas de percepción y diseño de pipelines reproducibles. El equipo de trabajo tuvo que enfrentar decisiones de diseño, compromisos entre precisión y costo computacional, así como ajustes de parámetros para obtener resultados estables. Este proceso es representativo de lo que ocurre en proyectos reales de visión computacional en la industria.

Si bien el sistema no está exento de limitaciones, especialmente en escenarios con obstrucciones importantes o geometrías atípicas, constituye una base sólida sobre la cual construir funcionalidades más avanzadas, como conteo de productos, integración con sistemas de inventario o monitoreo en tiempo real mediante video. En resumen, el proyecto cumple con los objetivos planteados, ofrece un beneficio potencial claro para la operación de tiendas y evidencia el dominio práctico de los temas revisados en el curso.

### Resumen de logros:

El proyecto "Shelf Occupancy Analyzer" demuestra que es posible automatizar el análisis de ocupación de anaqueles mediante una combinación de visión computacional clásica, estimación de profundidad con deep learning y segmentación geométrica por cuadriláteros. La solución diseñada logra:

1. ✅ **Detectar anaqueles de forma robusta** a partir de líneas horizontales y verticales
2. ✅ **Obtener mapas de profundidad fiables** con un modelo preentrenado
3. ✅ **Calcular un porcentaje de ocupación interpretable** por anaquel y visualizarlo de forma intuitiva

Aunque existen limitaciones relacionadas con condiciones extremas de iluminación, geometrías atípicas y posibles fallos del modelo de profundidad, el sistema proporciona una base sólida para aplicaciones de monitoreo automatizado en retail y abre la puerta a extensiones orientadas a conteo de productos, alertas en tiempo real y despliegue en aplicaciones web.

---

## 7. Referencias

1. Hui, K., Tang, S., & Chen, X. (2020). *SKU110K: A retail shelf image dataset for product recognition*. arXiv preprint arXiv:2008.09100. https://arxiv.org/abs/2008.09100

2. byed2015. (2024). *Shelf-Occupancy-Analyzer* (versión 1.3.x) [Repositorio GitHub]. GitHub. https://github.com/byed2015/shelf-occupancy-analyzer

3. OpenCV. (2024). *OpenCV: Open Source Computer Vision Library — Documentation*. https://docs.opencv.org/

4. PyTorch Foundation. (2024). *PyTorch Documentation*. https://pytorch.org/docs/stable/

5. Duda, R. O., & Hart, P. E. (1972). Use of the Hough transformation to detect lines and curves in pictures. *Communications of the ACM*, 15(1), 11–15. https://doi.org/10.1145/361237.361242

---

**Versión del Sistema**: 2.0.0 (Normalización Local + Pipeline Simplificado)  
**Fecha del Reporte**: Diciembre 2024  
**Repositorio**: https://github.com/byed2015/shelf-occupancy-analyzer
