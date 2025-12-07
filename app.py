"""
Aplicación Streamlit para el Análisis de Ocupación de Anaqueles.

Esta aplicación proporciona una interfaz interactiva para visualizar
el pipeline completo de análisis de ocupación de anaqueles paso a paso.

Uso:
    streamlit run app.py
"""

import sys
from pathlib import Path
from typing import Dict, Optional
import streamlit as st
import cv2
import numpy as np
from PIL import Image
from loguru import logger

# Importar el visualizador del pipeline
from visualize_pipeline import PipelineVisualizer


# Configuración de la página
st.set_page_config(
    page_title="Análisis de Ocupación de Anaqueles",
    page_icon="📦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        font-weight: bold;
        color: #1f77b4;
        text-align: center;
        margin-bottom: 1rem;
    }
    .step-header {
        font-size: 1.5rem;
        font-weight: bold;
        color: #2ca02c;
        margin-top: 2rem;
        margin-bottom: 1rem;
        border-bottom: 2px solid #2ca02c;
        padding-bottom: 0.5rem;
    }
    .metric-box {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #1f77b4;
    }
    .explanation-box {
        background-color: #e8f4f8;
        padding: 1rem;
        border-radius: 0.5rem;
        border-left: 4px solid #17a2b8;
        margin-bottom: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# Diccionario con explicaciones de cada paso
STEP_EXPLANATIONS = {
    '0_original': {
        'title': '📷 Paso 0: Imagen Original',
        'explanation': """
        **Punto de partida del análisis.**
        
        Esta es la imagen sin procesar del anaquel que será analizada. 
        El sistema parte de fotografías tomadas directamente en el punto de venta.
        
        **Contribución al sistema:**
        - Proporciona los datos visuales base para todo el análisis
        - La calidad y resolución de esta imagen afecta directamente la precisión del sistema
        """
    },
    '1_preprocessed': {
        'title': '🔧 Paso 1: Preprocesamiento',
        'explanation': """
        **Conversión a escala de grises y suavizado.**
        
        La imagen se convierte a escala de grises y se aplica un filtro Gaussiano para reducir el ruido.
        
        **Contribución al sistema:**
        - Simplifica la imagen eliminando información de color innecesaria
        - Reduce el ruido que podría generar falsas detecciones
        - Mejora la eficiencia computacional al trabajar con un solo canal
        - Prepara la imagen para la detección de bordes más precisa
        """
    },
    '2_edges': {
        'title': '🔍 Paso 2: Detección de Bordes',
        'explanation': """
        **Algoritmo Canny con umbrales automáticos.**
        
        Se detectan los bordes de la imagen usando el algoritmo Canny con umbrales calculados 
        automáticamente basados en la mediana de intensidad de la imagen.
        
        **Contribución al sistema:**
        - Identifica los contornos de los anaqueles y productos
        - Los bordes son fundamentales para detectar las líneas de los anaqueles
        - El ajuste automático de umbrales hace el sistema robusto a diferentes condiciones de iluminación
        - Resalta las estructuras geométricas necesarias para el siguiente paso
        """
    },
    '3_lines': {
        'title': '📐 Paso 3: Detección de Líneas',
        'explanation': """
        **Transformada de Hough para detectar líneas horizontales y verticales.**
        
        Se detectan las líneas principales de la imagen y se clasifican en horizontales (verde) 
        y verticales (azul). Se calcula el ángulo dominante de cada orientación.
        
        **Contribución al sistema:**
        - Las líneas horizontales definen los límites superior e inferior de cada anaquel
        - Las líneas verticales definen los límites laterales
        - El ángulo dominante permite adaptar el sistema a anaqueles con perspectiva
        - La fusión de líneas similares elimina duplicados y mejora la precisión
        """
    },
    '4_shelves': {
        'title': '📦 Paso 4: Detección de Anaqueles',
        'explanation': """
        **Segmentación de anaqueles como cuadriláteros inclinados.**
        
        A partir de las líneas detectadas, se forman cuadriláteros que representan cada anaquel.
        El sistema respeta la perspectiva natural de la imagen.
        
        **Contribución al sistema:**
        - Define las regiones de interés (ROI) para el análisis de ocupación
        - Los cuadriláteros permiten adaptarse a la perspectiva de la fotografía
        - Cada región detectada se analizará independientemente
        - La precisión de esta segmentación es crucial para cálculos correctos de ocupación
        """
    },
    '5_depth': {
        'title': '🌊 Paso 5: Estimación de Profundidad',
        'explanation': """
        **Modelo de Deep Learning Depth-Anything-V2.**
        
        Se estima la profundidad de cada pixel usando un modelo pre-entrenado. 
        Los colores cálidos (amarillo/naranja) indican cercanía, colores fríos (azul/morado) indican lejanía.
        
        **Contribución al sistema:**
        - **Clave para calcular la ocupación:** los productos están más cerca (valores bajos de profundidad)
        - El fondo vacío del anaquel está más lejos (valores altos de profundidad)
        - Permite diferenciar productos de espacios vacíos sin entrenar un modelo específico
        - Robusto a diferentes tipos de productos, colores e iluminación
        """
    },
    '6_occupancy': {
        'title': '📊 Paso 6: Análisis de Ocupación (Normalización Local)',
        'explanation': """
        **Cálculo de ocupación con normalización independiente por anaquel.**
        
        Para cada anaquel se realiza un análisis **independiente**:
        1. Se mide la profundidad mínima y máxima **dentro del anaquel**
        2. Se normalizan las profundidades al rango [0, 1] local
        3. Se calcula la mediana normalizada
        4. Ocupación = mediana_normalizada × 100%
        
        **Visualización con código de colores:**
        - 🟢 Verde: Alta ocupación (>70%)
        - 🟡 Amarillo: Ocupación media (30-70%)
        - 🔴 Rojo: Baja ocupación (<30%)
        
        **Ventajas de la normalización local:**
        - **Más preciso:** Cada anaquel se analiza en su propio contexto
        - **Robusto:** No afectado por variaciones de iluminación entre anaqueles
        - **Equitativo:** Anaqueles oscuros/claros se miden igual de bien
        - **Accionable:** Resultados consistentes para toma de decisiones
        """
    }
}


def get_available_images() -> list[Path]:
    """Obtiene la lista de imágenes disponibles en la carpeta de dataset."""
    images_dir = Path("data/raw/SKU110K_fixed/images")
    if not images_dir.exists():
        return []
    
    image_files = sorted(images_dir.glob("test_*.jpg"))
    return image_files


def convert_cv2_to_pil(cv2_image: np.ndarray) -> Image.Image:
    """Convierte una imagen de OpenCV (BGR) a PIL (RGB)."""
    if len(cv2_image.shape) == 3:
        rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)
    else:
        rgb_image = cv2_image
    return Image.fromarray(rgb_image)


def display_step(step_name: str, step_image: np.ndarray, step_info: str):
    """
    Muestra un paso del pipeline con su explicación.
    
    Args:
        step_name: Nombre del paso
        step_image: Imagen del paso
        step_info: Información adicional del paso
    """
    explanation = STEP_EXPLANATIONS.get(step_name, {
        'title': step_name,
        'explanation': 'Sin descripción disponible.'
    })
    
    # Encabezado del paso
    st.markdown(f'<div class="step-header">{explanation["title"]}</div>', unsafe_allow_html=True)
    
    # Layout en dos columnas: imagen y explicación
    col1, col2 = st.columns([1.2, 1])
    
    with col1:
        # Mostrar imagen
        pil_image = convert_cv2_to_pil(step_image)
        st.image(pil_image, use_container_width=True)
        st.caption(f"ℹ️ {step_info}")
    
    with col2:
        # Mostrar explicación
        st.markdown(f'<div class="explanation-box">{explanation["explanation"]}</div>', 
                   unsafe_allow_html=True)


def main():
    """Función principal de la aplicación Streamlit."""
    
    # Título principal
    st.markdown('<div class="main-header">📦 Análisis de Ocupación de Anaqueles</div>', 
                unsafe_allow_html=True)
    
    st.markdown("""
    <div style="text-align: center; color: #666; margin-bottom: 2rem;">
        Sistema automático de análisis de ocupación usando visión computacional y Deep Learning
    </div>
    """, unsafe_allow_html=True)
    
    # Barra lateral para configuración
    with st.sidebar:
        st.header("⚙️ Configuración")
        
        # Selector de imagen
        available_images = get_available_images()
        
        if not available_images:
            st.error("❌ No se encontraron imágenes en `data/raw/SKU110K_fixed/images/`")
            return
        
        # Crear lista de nombres para el selectbox
        image_names = [img.name for img in available_images]
        
        selected_image_name = st.selectbox(
            "Selecciona una imagen:",
            image_names,
            index=image_names.index("test_49.jpg") if "test_49.jpg" in image_names else 0
        )
        
        selected_image_path = next(img for img in available_images if img.name == selected_image_name)
        
        st.markdown("---")
        
        # 🖼️ PREVISUALIZACIÓN AUTOMÁTICA DE LA IMAGEN SELECCIONADA
        st.markdown("### 🖼️ Imagen Seleccionada")
        try:
            # Cargar y mostrar la imagen
            preview_image = Image.open(selected_image_path)
            
            # Mostrar información de la imagen
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.image(preview_image, caption=f"📷 {selected_image_name}", use_container_width=True)
            
            with col2:
                st.markdown("**📐 Información:**")
                st.markdown(f"- **Nombre:** `{selected_image_name}`")
                st.markdown(f"- **Dimensiones:** {preview_image.width} × {preview_image.height} px")
                st.markdown(f"- **Modo:** {preview_image.mode}")
                st.markdown(f"- **Tamaño:** {selected_image_path.stat().st_size / 1024:.1f} KB")
        except Exception as e:
            st.warning(f"⚠️ No se pudo cargar la previsualización: {e}")
        
        st.markdown("---")
        
        # Información
        st.markdown("""
        ### 📖 Acerca del sistema
        
        Este sistema analiza fotografías de anaqueles para calcular automáticamente 
        el porcentaje de ocupación de cada nivel usando **normalización local por anaquel**.
        
        **Tecnologías utilizadas:**
        - OpenCV para visión computacional
        - Depth-Anything-V2 para estimación de profundidad
        - Transformada de Hough para detección de líneas
        - Normalización independiente por cuadrilátero
        """)
        
        st.markdown("---")
        
        # Botón de procesamiento
        process_button = st.button("🚀 Analizar Imagen", type="primary", use_container_width=True)
    
    # Área principal
    if process_button:
        # Configurar logger para capturar mensajes
        logger.remove()
        
        # Crear contenedores para el progreso
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        try:
            status_text.text("Inicializando pipeline...")
            progress_bar.progress(10)
            
            # Crear visualizador
            visualizer = PipelineVisualizer("config/config.yaml")
            
            status_text.text(f"Procesando {selected_image_name}...")
            progress_bar.progress(20)
            
            # Procesar imagen
            success = visualizer.process_image(selected_image_path)
            
            if not success:
                st.error("❌ Error al procesar la imagen. Revisa los logs para más detalles.")
                return
            
            progress_bar.progress(80)
            status_text.text("Generando visualizaciones...")
            
            # Limpiar barra de progreso
            progress_bar.progress(100)
            status_text.empty()
            progress_bar.empty()
            
            # Mostrar éxito
            st.success(f"✅ Imagen procesada exitosamente: **{selected_image_name}**")
            
            # Mostrar métricas principales
            if hasattr(visualizer, 'metadata'):
                st.markdown("---")
                st.markdown("### 📊 Resumen de Resultados")
                
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric(
                        label="Anaqueles Detectados",
                        value=visualizer.metadata['num_shelves']
                    )
                
                with col2:
                    st.metric(
                        label="Ocupación Promedio",
                        value=f"{visualizer.metadata['average_occupancy']:.1f}%"
                    )
                
                with col3:
                    uses_quad = visualizer.metadata.get('uses_quadrilaterals', False)
                    st.metric(
                        label="Método de Detección",
                        value="Cuadriláteros" if uses_quad else "Bounding Boxes"
                    )
            
            st.markdown("---")
            
            # Mostrar cada paso del pipeline
            st.markdown("### 🔄 Pipeline de Procesamiento")
            
            for step_name in sorted(visualizer.steps.keys()):
                step_image = visualizer.steps[step_name]
                step_info = visualizer.step_info.get(step_name, "")
                
                display_step(step_name, step_image, step_info)
            
            # Mostrar tabla de resultados por anaquel
            if hasattr(visualizer, 'metadata'):
                st.markdown("---")
                st.markdown("### 📋 Resultados Detallados por Anaquel")
                
                import pandas as pd
                
                # Crear DataFrame con resultados
                results_data = []
                for i, (occ_pct, stats) in enumerate(zip(
                    visualizer.metadata['occupancy_percentages'],
                    visualizer.metadata['stats']
                )):
                    results_data.append({
                        'Anaquel': f'S{i+1}',
                        'Ocupación (%)': f'{occ_pct:.1f}',
                        'Estado': '🟢 Alto' if occ_pct > 70 else ('🟡 Medio' if occ_pct > 30 else '🔴 Bajo'),
                        'Celdas Ocupadas': f"{stats.get('occupied_cells', 0)}/{stats.get('total_cells', 0)}",
                        'Desv. Estándar': f"{stats.get('std_occupancy', 0):.3f}"
                    })
                
                df = pd.DataFrame(results_data)
                st.dataframe(df, use_container_width=True, hide_index=True)
            
        except Exception as e:
            st.error(f"❌ Error durante el procesamiento: {str(e)}")
            import traceback
            with st.expander("Ver detalles del error"):
                st.code(traceback.format_exc())
    
    else:
        # Pantalla de inicio
        st.info("👈 Selecciona una imagen en la barra lateral y presiona el botón **Analizar Imagen** para comenzar.")
        
        # Mostrar imagen de ejemplo
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            st.image("https://via.placeholder.com/800x600/f0f2f6/1f77b4?text=Selecciona+una+imagen+para+comenzar", 
                    use_container_width=True)


if __name__ == "__main__":
    main()
