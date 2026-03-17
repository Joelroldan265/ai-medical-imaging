import os
from PIL import Image as PILImage
from agno.agent import Agent
from agno.models.google import Gemini
from agno.run.agent import RunOutput
import streamlit as st
from agno.tools.duckduckgo import DuckDuckGoTools
from agno.media import Image as AgnoImage

if "GOOGLE_API_KEY" not in st.session_state:
    st.session_state.GOOGLE_API_KEY = None

with st.sidebar:
    st.title("ℹ️ Configuración")
    
    if not st.session_state.GOOGLE_API_KEY:
        api_key = st.text_input(
            "Ingresa tu API Key de Google:",
            type="password"
        )
        st.caption(
            "Obtén tu API key en [Google AI Studio]"
            "(https://aistudio.google.com/apikey) 🔑"
        )
        if api_key:
            st.session_state.GOOGLE_API_KEY = api_key
            st.success("¡API Key guardada!")
            st.rerun()
    else:
        st.success("API Key configurada correctamente")
        if st.button("🔄 Cambiar API Key"):
            st.session_state.GOOGLE_API_KEY = None
            st.rerun()
    
    st.info(
        "Esta herramienta analiza imágenes médicas usando inteligencia artificial "
        "con visión computacional avanzada y experiencia radiológica."
    )
    st.warning(
        "⚠️ AVISO: Esta herramienta es solo para fines educativos e informativos. "
        "Todo análisis debe ser revisado por profesionales de salud calificados. "
        "No tomes decisiones médicas basándote únicamente en este análisis."
    )

medical_agent = Agent(
    model=Gemini(
        id="gemini-2.5-flash",
        api_key=st.session_state.GOOGLE_API_KEY
    ),
    tools=[DuckDuckGoTools()],
    markdown=True
) if st.session_state.GOOGLE_API_KEY else None

if not medical_agent:
    st.warning("Por favor configura tu API Key en el panel lateral para continuar")

# Consulta de análisis médico en español
query = """
Eres un experto altamente capacitado en imágenes médicas con amplio conocimiento en radiología y diagnóstico por imagen. Analiza la imagen médica del paciente y estructura tu respuesta de la siguiente manera:

### 1. Tipo de Imagen y Región
- Especifica la modalidad de imagen (Rayos X / MRI / CT / Ultrasonido / etc.)
- Identifica la región anatómica y posición del paciente
- Comenta sobre la calidad de la imagen y adecuación técnica

### 2. Hallazgos Principales
- Lista las observaciones primarias de forma sistemática
- Señala cualquier anomalía en la imagen con descripciones precisas
- Incluye medidas y densidades cuando sea relevante
- Describe ubicación, tamaño, forma y características
- Clasifica la severidad: Normal / Leve / Moderado / Severo

### 3. Evaluación Diagnóstica
- Proporciona el diagnóstico principal con nivel de confianza
- Lista diagnósticos diferenciales en orden de probabilidad
- Apoya cada diagnóstico con evidencia observada en la imagen
- Señala hallazgos críticos o urgentes

### 4. Explicación para el Paciente
- Explica los hallazgos en lenguaje simple y claro que el paciente pueda entender
- Evita términos médicos o proporciona definiciones claras
- Incluye analogías visuales si es útil
- Aborda las preguntas más comunes relacionadas con estos hallazgos

### 5. Contexto de Investigación
IMPORTANTE: Usa la herramienta de búsqueda DuckDuckGo para:
- Encontrar literatura médica reciente sobre casos similares
- Buscar protocolos de tratamiento estándar
- Proporcionar una lista de enlaces médicos relevantes
- Investigar avances tecnológicos relevantes
- Incluir 2-3 referencias clave para apoyar tu análisis

Formatea tu respuesta usando encabezados markdown claros y puntos. Sé conciso pero exhaustivo. Responde TODO en español.
"""

st.title("🏥 Agente de Diagnóstico por Imágenes Médicas")
st.write("Sube una imagen médica para obtener un análisis profesional con IA")

# Contenedores para mejor organización
upload_container = st.container()
image_container = st.container()
analysis_container = st.container()

with upload_container:
    uploaded_file = st.file_uploader(
        "Subir Imagen Médica",
        type=["jpg", "jpeg", "png", "dicom"],
        help="Formatos soportados: JPG, JPEG, PNG, DICOM"
    )

if uploaded_file is not None:
    with image_container:
        col1, col2, col3 = st.columns([1, 2, 1])
        with col2:
            image = PILImage.open(uploaded_file)
            width, height = image.size
            aspect_ratio = width / height
            new_width = 500
            new_height = int(new_width / aspect_ratio)
            resized_image = image.resize((new_width, new_height))
            
            st.image(
                resized_image,
                caption="Imagen Médica Subida",
                use_container_width=True
            )
            
            analyze_button = st.button(
                "🔍 Analizar Imagen",
                type="primary",
                use_container_width=True
            )
    
    with analysis_container:
        if analyze_button:
            with st.spinner("🔄 Analizando imagen... Por favor espera."):
                try:
                    temp_path = "temp_resized_image.png"
                    resized_image.save(temp_path)
                    
                    # Crear objeto AgnoImage
                    agno_image = AgnoImage(filepath=temp_path)
                    
                    # Ejecutar análisis
                    response: RunOutput = medical_agent.run(query, images=[agno_image])
                    st.markdown("### 📋 Resultados del Análisis")
                    st.markdown("---")
                    st.markdown(response.content)
                    st.markdown("---")
                    st.caption(
                        "Nota: Este análisis es generado por IA y debe ser revisado por "
                        "un profesional de salud calificado."
                    )
                except Exception as e:
                    st.error(f"Error en el análisis: {e}")
else:
    st.info("👆 Por favor sube una imagen médica para comenzar el análisis")