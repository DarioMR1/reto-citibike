# 🚴‍♂️ CitiBike Analytics & ML Platform

Una plataforma integral de análisis predictivo y detección de anomalías para datos de CitiBike, que incluye simulaciones avanzadas y análisis contrafactual.

## 📋 Características Principales

### 🤖 Modelos de Machine Learning

- **Modelos Supervisados**: Random Forest y Gradient Boosting para predicción de ingresos
- **Modelos No Supervisados**: Isolation Forest y Local Outlier Factor para detección de anomalías
- **Persistencia de Modelos**: Guardado y carga automática de modelos entrenados

### 🎮 Simulaciones Interactivas

- **Predicción Individual**: Predice ingresos para viajes específicos
- **Análisis por Lotes**: Evalúa múltiples escenarios simultáneamente
- **Detección de Anomalías**: Simula identificación de patrones anómalos
- **Análisis Contrafactual**: Estima pérdidas por eventos de desabasto
- **Optimizador**: Encuentra condiciones óptimas para maximizar ingresos

### 📊 Dashboard y Visualizaciones

- **KPIs en Tiempo Real**: Métricas clave del negocio
- **Análisis de Ingresos**: Por hora, usuario, estación y clima
- **Balance de Estaciones**: Identificación de déficit y excesos
- **Impacto Climático**: Relación entre clima e ingresos

## 🏗️ Arquitectura Modular

La aplicación está estructurada en módulos especializados:

```
reto-citibike/
├── app.py                    # Aplicación principal de Streamlit
├── database.py               # Conexiones y consultas a Snowflake
├── models.py                 # Clases de modelos ML supervisados y no supervisados
├── visualizations.py         # Funciones de gráficos y visualizaciones
├── simulations.py           # Motor de simulaciones y predicciones
├── requirements.txt         # Dependencias de Python
└── README.md               # Este archivo
```

## 🚀 Instalación y Uso

### 1. Preparar el Entorno

```bash
# Crear entorno virtual
python -m venv venv

# Activar entorno virtual
# En Windows:
venv\Scripts\activate
# En Mac/Linux:
source venv/bin/activate

# Instalar dependencias
pip install -r requirements.txt
```

### 2. Ejecutar la Aplicación

```bash
streamlit run app.py
```

### 3. Acceder a la Plataforma

Abre tu navegador y ve a `http://localhost:8501`

## 📖 Guía de Uso

### 🏠 Dashboard Principal

- **KPIs**: Visualiza métricas clave como ingresos totales, viajes y estaciones activas
- **Análisis de Ingresos**: Explora patrones temporales y por usuario
- **Balance de Estaciones**: Identifica estaciones con déficit o exceso de bicicletas
- **Impacto del Clima**: Analiza cómo la temperatura afecta los ingresos

### 🤖 Entrenamiento de Modelos

#### Modelos Supervisados

1. Ve a la pestaña "Entrenamiento de Modelos"
2. Selecciona "Modelos Supervisados"
3. Haz clic en "Entrenar Modelos Supervisados"
4. Espera a que se complete el entrenamiento (puede tomar varios minutos)
5. Revisa las métricas de comparación entre Random Forest y Gradient Boosting

#### Modelos No Supervisados

1. En la misma sección, selecciona "Modelos No Supervisados"
2. Haz clic en "Entrenar Detección de Anomalías"
3. Analiza las anomalías detectadas por consenso

### 🎮 Simulaciones y Predicciones

#### Predicción Individual

- Configura parámetros como estación, hora, temperatura, duración
- Obtén predicciones instantáneas de ingresos por minutos excedentes
- Compara con cálculos teóricos

#### Análisis por Lotes

- Selecciona qué parámetros variar (hora, temperatura, duración)
- Ejecuta análisis masivo para encontrar mejores escenarios
- Visualiza resultados en matrices interactivas

#### Detección de Anomalías

- Simula condiciones específicas de estación-hora
- Evalúa si el patrón es anómalo según los modelos entrenados
- Obtén scores de anomalía detallados

#### Análisis Contrafactual

- Simula eventos de desabasto específicos
- Compara impacto entre diferentes estaciones
- Estima pérdidas anuales proyectadas

#### Optimizador

- Encuentra automáticamente las mejores condiciones
- Maximiza ingresos para estaciones específicas
- Recibe recomendaciones operativas

## 🔧 Configuración Avanzada

### Gestión de Modelos

- **Guardar**: Los modelos se guardan automáticamente en `/models/`
- **Cargar**: Carga modelos previamente entrenados al iniciar
- **Estado**: Verifica el estado de entrenamiento en la barra lateral

### Parámetros de Conexión

La aplicación se conecta automáticamente a Snowflake con:

- **Account**: WSDIINJ-SG47948
- **Database**: RETO_CITIBIKE
- **Schema**: PUBLIC

### Cache de Datos

- Los datos del dashboard se cachean por 1 hora para mejor rendimiento
- Las consultas se optimizan para reducir tiempo de carga

## 📊 Integración con los Notebooks

Esta aplicación integra y mejora el trabajo realizado en los notebooks:

### Desde `modelos_supervisados.ipynb`:

- ✅ Feature Engineering automatizado (25 features)
- ✅ Entrenamiento de Random Forest y Gradient Boosting
- ✅ Evaluación con R² = 0.7939 (mejor performance)
- ✅ Análisis contrafactual de eventos de desabasto
- ✅ Estimación de $11,702.64 en pérdidas anuales

### Desde `modelos_no_supervisados.ipynb`:

- ✅ Isolation Forest y Local Outlier Factor
- ✅ Detección de 515 anomalías de consenso
- ✅ Análisis de $1,740.71 en pérdidas por anomalías
- ✅ Identificación de estaciones críticas (HB102)

### Mejoras Adicionales:

- 🎮 **Simulaciones Interactivas**: Capacidades "what-if" en tiempo real
- 📊 **Dashboard Intuitivo**: Visualizaciones profesionales con Plotly
- 🔄 **Persistencia**: Modelos se guardan y cargan automáticamente
- ⚡ **Optimización**: Cache y consultas optimizadas
- 🎯 **UX/UI**: Interfaz moderna y fácil de usar

## 🚨 Casos de Uso Principales

### 1. Análisis Operativo Diario

- Monitorear KPIs en tiempo real
- Identificar estaciones problemáticas
- Evaluar impacto del clima

### 2. Planificación Estratégica

- Simular diferentes escenarios operativos
- Optimizar redistribución de bicicletas
- Estimar ROI de mejoras

### 3. Detección Proactiva de Problemas

- Identificar anomalías antes de que se conviertan en problemas
- Predecir eventos de desabasto
- Alertas tempranas para operaciones

### 4. Análisis de Ingresos

- Predecir ingresos por minutos excedentes
- Identificar oportunidades de optimización
- Cuantificar pérdidas por desabasto

## 🤝 Soporte

Para soporte técnico o preguntas sobre la plataforma:

- 📧 Email: contacto@citibike-analytics.com
- 📖 Documentación: Ver este README
- 🐛 Problemas: Reportar en el repositorio

## 📈 Métricas de Rendimiento

### Modelos Supervisados:

- **Gradient Boosting**: R² = 0.7939, RMSE = $12.71
- **Random Forest**: R² = 0.7906, RMSE = $12.81
- **Features**: 25 variables predictivas
- **Datos**: 245,513 registros de entrenamiento

### Modelos No Supervisados:

- **Isolation Forest**: 10% contaminación esperada
- **Local Outlier Factor**: 20 vecinos, consenso robusto
- **Anomalías**: 515 detectadas por consenso (3.1%)
- **Datos**: 16,803 combinaciones estación-hora

---

**🚀 CitiBike Analytics Platform v1.0** - Desarrollado para análisis predictivo y detección de anomalías en datos de CitiBike
