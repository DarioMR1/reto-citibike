import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import time

# Importar módulos personalizados
from database import (
    get_training_data, get_anomaly_data, get_desabasto_events, 
    get_dashboard_data
)
from models import (
    supervised_model, unsupervised_model, 
    save_models, load_models
)
from visualizations import (
    plot_model_comparison, plot_feature_importance, plot_anomalies_scatter,
    plot_anomalies_by_hour, plot_revenue_analysis, plot_station_balance,
    plot_weather_impact, plot_counterfactual_analysis, create_kpi_metrics,
    display_kpi_row
)
from simulations import run_simulation_interface

# Configuración de la página
st.set_page_config(
    page_title="CitiBike Analytics & ML Platform",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS personalizado
st.markdown("""
<style>
    /* Black Modern UI Theme */
    .stApp {
        background-color: #0e1117;
        color: #fafafa;
    }
    
    .main-container {
        background: linear-gradient(135deg, #1f2937 0%, #111827 100%);
        color: white;
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        text-align: center;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
    }
    
    .main-header {
        font-size: 3rem;
        font-weight: 700;
        margin-bottom: 0.5rem;
        color: #f9fafb;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.5);
    }
    
    .main-subtitle {
        font-size: 1.2rem;
        opacity: 0.9;
        font-weight: 400;
        color: #d1d5db;
    }
    
    .section-header {
        font-size: 1.8rem;
        color: #f9fafb;
        font-weight: 600;
        margin: 2rem 0 1rem 0;
        padding-left: 0.5rem;
        border-left: 4px solid #3b82f6;
        background: rgba(59, 130, 246, 0.1);
        padding: 1rem;
        border-radius: 8px;
    }
    
    .metric-card {
        background: #1f2937;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.3);
        border: 1px solid #374151;
        margin: 0.5rem 0;
    }
    
    .status-badge {
        display: inline-block;
        padding: 0.4rem 1rem;
        border-radius: 25px;
        font-size: 0.875rem;
        font-weight: 600;
        margin: 0.25rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    
    .status-success {
        background: linear-gradient(135deg, #10b981, #059669);
        color: white;
        box-shadow: 0 2px 8px rgba(16, 185, 129, 0.3);
    }
    
    .status-warning {
        background: linear-gradient(135deg, #f59e0b, #d97706);
        color: white;
        box-shadow: 0 2px 8px rgba(245, 158, 11, 0.3);
    }
    
    .status-info {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    .sidebar-section {
        background: #1f2937;
        padding: 1.5rem;
        border-radius: 12px;
        margin: 1rem 0;
        border: 1px solid #374151;
        box-shadow: 0 2px 8px rgba(0, 0, 0, 0.2);
    }
    
    /* Streamlit component overrides for dark theme */
    .stSelectbox > div > div {
        background-color: #374151;
        border: 1px solid #4b5563;
        color: #f9fafb;
    }
    
    .stButton > button {
        background: linear-gradient(135deg, #3b82f6, #2563eb);
        color: white;
        border: none;
        border-radius: 8px;
        padding: 0.75rem 1.5rem;
        font-weight: 600;
        transition: all 0.3s ease;
        box-shadow: 0 4px 6px rgba(59, 130, 246, 0.25);
    }
    
    .stButton > button:hover {
        background: linear-gradient(135deg, #2563eb, #1d4ed8);
        transform: translateY(-2px);
        box-shadow: 0 8px 15px rgba(59, 130, 246, 0.4);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background-color: #111827;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        background-color: #1f2937;
        border-radius: 12px;
        padding: 0.5rem;
        border: 1px solid #374151;
    }
    
    .stTabs [data-baseweb="tab"] {
        background-color: transparent;
        border-radius: 8px;
        color: #9ca3af;
        font-weight: 500;
    }
    
    .stTabs [aria-selected="true"] {
        background-color: #3b82f6;
        color: white;
        box-shadow: 0 2px 8px rgba(59, 130, 246, 0.3);
    }
    
    /* Metrics containers */
    [data-testid="metric-container"] {
        background: #1f2937;
        border: 1px solid #374151;
        padding: 1.5rem;
        border-radius: 12px;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    /* Hide Streamlit branding */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
</style>
""", unsafe_allow_html=True)

# Variables de estado de la sesión
if 'training_data_loaded' not in st.session_state:
    st.session_state.training_data_loaded = False
if 'anomaly_data_loaded' not in st.session_state:
    st.session_state.anomaly_data_loaded = False
if 'models_trained' not in st.session_state:
    st.session_state.models_trained = False
if 'data_cache' not in st.session_state:
    st.session_state.data_cache = {}
if 'dashboard_data' not in st.session_state:
    st.session_state.dashboard_data = None

def main():
    """Función principal de la aplicación"""
    
    # Header principal
    st.markdown("""
    <div class="main-container">
        <h1 class="main-header">CitiBike Analytics & ML Platform</h1>
        <p class="main-subtitle">Advanced Analytics and Machine Learning Platform for CitiBike Operations</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Sidebar para navegación
    with st.sidebar:
        st.markdown("### 🚴 CitiBike Analytics")
        st.markdown("---")
        
        page = st.selectbox(
            "Navigation",
            [
                "📊 Dashboard",
                "🤖 Model Training", 
                "🎯 Simulations & Predictions",
                "⚠️ Anomaly Analysis",
                "🔮 Counterfactual Analysis"
            ]
        )
        
        st.markdown("---")
        
        # Estado de los modelos
        st.markdown('<div class="sidebar-section">', unsafe_allow_html=True)
        st.markdown("### 🎛️ System Status")
        
        # Intentar cargar modelos existentes
        if st.button("🔄 Load Saved Models", key="load_models_btn"):
            if load_models():
                st.success("✅ Models loaded successfully")
                st.session_state.models_trained = True
            else:
                st.info("ℹ️ No saved models found")
        
        if supervised_model.best_model is not None:
            st.markdown('<span class="status-badge status-success">✅ Supervised Model: Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-warning">⚠️ Supervised Model: Not Trained</span>', unsafe_allow_html=True)
            
        if unsupervised_model.iso_forest is not None:
            st.markdown('<span class="status-badge status-success">✅ Unsupervised Model: Ready</span>', unsafe_allow_html=True)
        else:
            st.markdown('<span class="status-badge status-warning">⚠️ Unsupervised Model: Not Trained</span>', unsafe_allow_html=True)
        
        st.markdown('</div>', unsafe_allow_html=True)
    
    # Contenido principal basado en la selección
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🤖 Model Training":
        show_model_training()
    elif page == "🎯 Simulations & Predictions":
        run_simulation_interface()
    elif page == "⚠️ Anomaly Analysis":
        show_anomaly_analysis()
    elif page == "🔮 Counterfactual Analysis":
        show_counterfactual_analysis()

@st.cache_data(ttl=3600)
def load_dashboard_data():
    """Carga datos del dashboard con cache"""
    return get_dashboard_data()

def show_dashboard():
    """Dashboard principal con KPIs y visualizaciones"""
    st.markdown('<h2 class="section-header">📊 Dashboard Overview</h2>', unsafe_allow_html=True)
    
    # Cargar datos con cache
    if st.session_state.dashboard_data is None:
        with st.spinner("🔄 Loading dashboard data..."):
            st.session_state.dashboard_data = load_dashboard_data()
    
    df_dashboard = st.session_state.dashboard_data
    
    if df_dashboard.empty:
        st.error("❌ Could not load dashboard data")
        return
    
    st.success(f"✅ Data loaded: {len(df_dashboard):,} records")
    
    # KPIs principales
    kpis = create_kpi_metrics(df_dashboard)
    display_kpi_row(kpis)
    
    # Tabs para diferentes análisis
    tab1, tab2, tab3, tab4 = st.tabs([
        "💰 Revenue Analysis",
        "⚖️ Station Balance", 
        "🌤️ Weather Impact",
        "🔍 Data Explorer"
    ])
    
    with tab1:
        st.plotly_chart(plot_revenue_analysis(df_dashboard), use_container_width=True)
    
    with tab2:
        balance_chart = plot_station_balance(df_dashboard)
        if balance_chart:
            st.plotly_chart(balance_chart, use_container_width=True)
        else:
            st.info("ℹ️ Station balance data not available")
    
    with tab3:
        weather_chart = plot_weather_impact(df_dashboard)
        if weather_chart:
            st.plotly_chart(weather_chart, use_container_width=True)
        else:
            st.info("ℹ️ Weather data not available")
    
    with tab4:
        st.subheader("🔍 Data Explorer")
        
        # Filtros con keys únicos
        col1, col2, col3 = st.columns(3)
        with col1:
            if 'MEMBER_CASUAL' in df_dashboard.columns:
                member_filter = st.selectbox("👤 User Type", 
                    ['All'] + df_dashboard['MEMBER_CASUAL'].unique().tolist(),
                    key="member_filter_unique")
        with col2:
            if 'BIKE_TYPE_ID' in df_dashboard.columns:
                bike_filter = st.selectbox("🚲 Bike Type",
                    ['All'] + df_dashboard['BIKE_TYPE_ID'].unique().tolist(),
                    key="bike_filter_unique")
        with col3:
            month_filter = st.selectbox("📅 Month",
                ['All'] + sorted(df_dashboard['MONTH'].unique().tolist()),
                key="month_filter_unique")
        
        # Aplicar filtros
        df_filtered = df_dashboard.copy()
        
        if 'member_filter' in locals() and member_filter != 'All':
            df_filtered = df_filtered[df_filtered['MEMBER_CASUAL'] == member_filter]
        if 'bike_filter' in locals() and bike_filter != 'All':
            df_filtered = df_filtered[df_filtered['BIKE_TYPE_ID'] == bike_filter]
        if month_filter != 'All':
            df_filtered = df_filtered[df_filtered['MONTH'] == month_filter]
        
        st.info(f"📋 Showing {len(df_filtered):,} filtered records out of {len(df_dashboard):,} total")
        
        # Mostrar datos filtrados
        if len(df_filtered) > 0:
            st.dataframe(df_filtered.head(1000), use_container_width=True)
        else:
            st.warning("⚠️ No records match the selected filters")

def show_model_training():
    """Interfaz de entrenamiento de modelos"""
    st.markdown('<h2 class="section-header">🤖 Model Training</h2>', unsafe_allow_html=True)
    
    model_tabs = st.tabs(["Supervised Models", "Unsupervised Models", "Results Comparison"])
    
    with model_tabs[0]:
        st.subheader("Supervised Model Training")
        st.write("Train models to predict excess minute revenue")
        
        if st.button("Train Supervised Models", type="primary"):
            with st.spinner("Loading training data..."):
                df_training = get_training_data()
            
            if df_training.empty:
                st.error("Could not load training data")
                return
            
            st.info(f"Data loaded: {len(df_training):,} records")
            
            # Entrenar modelos
            with st.spinner("Training models... (this may take several minutes)"):
                results = supervised_model.train(df_training)
            
            if results:
                st.success("Training completed successfully!")
                
                # Mostrar comparación
                fig_comparison = plot_model_comparison(results)
                st.plotly_chart(fig_comparison, use_container_width=True)
                
                # Feature importance
                if supervised_model.best_model and hasattr(supervised_model.best_model, 'feature_importances_'):
                    fig_importance = plot_feature_importance(
                        supervised_model.best_model,
                        supervised_model.features_cols
                    )
                    if fig_importance:
                        st.plotly_chart(fig_importance, use_container_width=True)
                
                # Guardar modelos
                if save_models():
                    st.success("✅ Models saved successfully")
                else:
                    st.warning("⚠️ Error saving models, but training completed")
                st.session_state.models_trained = True
    
    with model_tabs[1]:
        st.subheader("🔍 Entrenamiento de Modelos No Supervisados")
        st.write("Entrena modelos para detección de anomalías")
        
        if st.button("🚀 Entrenar Detección de Anomalías", type="primary"):
            with st.spinner("⏳ Cargando datos para detección de anomalías..."):
                df_anomalies = get_anomaly_data()
            
            if df_anomalies.empty:
                st.error("❌ No se pudieron cargar los datos de anomalías")
                return
            
            st.info(f"📊 Datos cargados: {len(df_anomalies):,} registros")
            
            # Entrenar modelo de anomalías
            with st.spinner("🔄 Entrenando detección de anomalías..."):
                anomaly_results = unsupervised_model.train_anomaly_detection(df_anomalies)
            
            if anomaly_results:
                st.success("✅ Detección de anomalías entrenada!")
                
                # Visualizaciones de anomalías
                col1, col2 = st.columns(2)
                
                with col1:
                    st.metric("Anomalías ISO Forest", anomaly_results['num_anomalias_iso'])
                    st.metric("Anomalías LOF", anomaly_results['num_anomalias_lof'])
                
                with col2:
                    st.metric("Consenso", anomaly_results['num_consenso'])
                    st.metric("% Anomalías", f"{(anomaly_results['num_consenso']/len(anomaly_results['df_clean']))*100:.1f}%")
                
                # Scatter plot de anomalías
                if len(anomaly_results['anomalias_consenso']) > 0:
                    fig_scatter = plot_anomalies_scatter(
                        anomaly_results['df_clean'],
                        anomaly_results['anomalias_consenso']
                    )
                    st.plotly_chart(fig_scatter, use_container_width=True)
                    
                    fig_hours = plot_anomalies_by_hour(anomaly_results['anomalias_consenso'])
                    st.plotly_chart(fig_hours, use_container_width=True)
                
                if save_models():
                    st.success("✅ Models saved successfully")
                else:
                    st.warning("⚠️ Error saving models, but training completed")
    
    with model_tabs[2]:
        st.subheader("📊 Comparación de Resultados")
        
        if supervised_model.best_model and unsupervised_model.iso_forest:
            col1, col2 = st.columns(2)
            
            with col1:
                st.success("✅ **Modelo Supervisado Entrenado**")
                st.write(f"**Tipo**: {supervised_model.model_type}")
                st.write(f"**R² Score**: {supervised_model.performance_metrics.get('r2', 0):.4f}")
                st.write(f"**RMSE**: ${supervised_model.performance_metrics.get('rmse', 0):.2f}")
                st.write(f"**MAE**: ${supervised_model.performance_metrics.get('mae', 0):.2f}")
            
            with col2:
                st.success("✅ **Modelo No Supervisado Entrenado**")
                if unsupervised_model.anomalies_data:
                    st.write(f"**Anomalías detectadas**: {unsupervised_model.anomalies_data['num_consenso']}")
                    st.write(f"**Total registros**: {len(unsupervised_model.anomalies_data['df_clean']):,}")
                    anomaly_rate = (unsupervised_model.anomalies_data['num_consenso'] / len(unsupervised_model.anomalies_data['df_clean'])) * 100
                    st.write(f"**Tasa de anomalías**: {anomaly_rate:.1f}%")
            
            st.info("🎉 **¡Ambos modelos están listos para simulaciones!**")
        else:
            st.warning("⚠️ Entrena ambos modelos para ver la comparación completa")

def show_anomaly_analysis():
    """Análisis detallado de anomalías"""
    st.markdown('<h2 class="section-header">⚠️ Anomaly Analysis</h2>', unsafe_allow_html=True)
    
    if unsupervised_model.anomalies_data is None:
        st.warning("⚠️ Necesitas entrenar el modelo de anomalías primero")
        return
    
    anomaly_data = unsupervised_model.anomalies_data
    
    # Métricas de anomalías
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Total Registros", f"{len(anomaly_data['df_clean']):,}")
    with col2:
        st.metric("Anomalías Consenso", anomaly_data['num_consenso'])
    with col3:
        st.metric("Tasa Anomalías", f"{(anomaly_data['num_consenso']/len(anomaly_data['df_clean']))*100:.1f}%")
    with col4:
        if len(anomaly_data['anomalias_consenso']) > 0:
            perdida_estimada = anomaly_data['anomalias_consenso']['INGRESO_PROMEDIO'].sum()
            st.metric("Pérdida Estimada", f"${perdida_estimada:,.2f}")
    
    # Análisis detallado
    if len(anomaly_data['anomalias_consenso']) > 0:
        anomalias = anomaly_data['anomalias_consenso']
        
        tab1, tab2, tab3 = st.tabs(["🔍 Top Anomalies", "📈 Temporal Analysis", "🏢 By Station"])
        
        with tab1:
            st.subheader("🚨 Top 20 Most Critical Anomalies")
            top_anomalias = anomalias.nlargest(20, 'INGRESO_PROMEDIO')
            st.dataframe(top_anomalias, use_container_width=True)
        
        with tab2:
            fig_hours = plot_anomalies_by_hour(anomalias)
            st.plotly_chart(fig_hours, use_container_width=True)
            
            # Análisis por día de la semana
            if 'IS_WEEKEND' in anomalias.columns:
                weekend_analysis = anomalias.groupby('IS_WEEKEND').size().reset_index(name='count')
                weekend_analysis['day_type'] = weekend_analysis['IS_WEEKEND'].map({0: 'Weekdays', 1: 'Weekends'})
                
                import plotly.express as px
                fig_weekend = px.pie(weekend_analysis, values='count', names='day_type',
                                   title='Anomaly Distribution: Weekdays vs Weekends')
                st.plotly_chart(fig_weekend, use_container_width=True)
        
        with tab3:
            if 'STATION_ID' in anomalias.columns:
                station_anomalies = anomalias.groupby('STATION_ID').size().reset_index(name='num_anomalias')
                station_anomalies = station_anomalies.sort_values('num_anomalias', ascending=False).head(15)
                
                fig_stations = px.bar(station_anomalies, x='STATION_ID', y='num_anomalias',
                                    title='Top 15 Stations with Most Anomalies')
                st.plotly_chart(fig_stations, use_container_width=True)

@st.cache_data(ttl=3600)
def load_counterfactual_data():
    """Carga datos de eventos de desabasto con cache"""
    return get_desabasto_events()

def show_counterfactual_analysis():
    """Análisis contrafactual detallado"""
    st.markdown('<h2 class="section-header">🔮 Counterfactual Analysis</h2>', unsafe_allow_html=True)
    
    if supervised_model.best_model is None:
        st.warning("Please train the supervised model first")
        return
    
    # Cargar eventos de desabasto con cache
    with st.spinner("Loading shortage events..."):
        df_desabasto = load_counterfactual_data()
    
    if df_desabasto.empty:
        st.info("No shortage events found in the data")
        return
    
    st.success(f"Shortage events loaded: {len(df_desabasto):,}")
    
    # Realizar análisis contrafactual
    if st.button("Execute Complete Counterfactual Analysis", type="primary"):
        with st.spinner("Simulating lost trips..."):
            # Simular ingresos perdidos
            total_ingresos_perdidos = 0
            viajes_simulados = 0
            
            progress_bar = st.progress(0)
            
            for idx, evento in df_desabasto.iterrows():
                viajes_perdidos = int(evento['VIAJES_PERDIDOS_ESTIMADOS'])
                
                for i in range(viajes_perdidos):
                    # Generar duración realista
                    duration = np.random.normal(evento['DURACION_PROMEDIO_ESTACION'], 15)
                    duration = max(10, min(120, duration))
                    
                    # Predecir ingreso
                    predicted_revenue = supervised_model.predict_single(
                        station_id=evento['STATION_ID'],
                        hour=evento['HOUR'],
                        is_weekend=evento['IS_WEEKEND'],
                        month=evento['MONTH'],
                        temperature=evento['TEMPERATURE_2M'],
                        humidity=evento['RELATIVE_HUMIDITY_2M'],
                        bike_type=1 if np.random.random() < evento['PROP_BIKE_TYPE_1'] else 2,
                        duration=duration
                    )
                    
                    total_ingresos_perdidos += predicted_revenue
                    viajes_simulados += 1
                
                # Actualizar progress bar
                progress_bar.progress((idx + 1) / len(df_desabasto))
        
        # Mostrar resultados
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric("Lost Revenue", f"${total_ingresos_perdidos:,.2f}")
        with col2:
            st.metric("Lost Trips", f"{viajes_simulados:,}")
        with col3:
            st.metric("Loss per Trip", f"${total_ingresos_perdidos/viajes_simulados:.2f}")
        with col4:
            st.metric("Affected Stations", df_desabasto['STATION_ID'].nunique())
        
        # Visualización de pérdidas por estación
        fig_counterfactual = plot_counterfactual_analysis(df_desabasto, total_ingresos_perdidos)
        if fig_counterfactual:
            st.plotly_chart(fig_counterfactual, use_container_width=True)
        
        # Análisis temporal de eventos
        eventos_por_hora = df_desabasto.groupby('HOUR').size().reset_index(name='num_eventos')
        fig_temporal = px.bar(eventos_por_hora, x='HOUR', y='num_eventos',
                             title='Temporal Distribution of Shortage Events')
        st.plotly_chart(fig_temporal, use_container_width=True)

if __name__ == "__main__":
    main()
