import streamlit as st
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from models import supervised_model, unsupervised_model
from database import get_desabasto_events
import plotly.express as px
import plotly.graph_objects as go

class Simulator:
    def __init__(self):
        self.scenarios = []
        
    def single_prediction_scenario(self):
        """Interfaz para predicción individual"""
        st.subheader("Individual Prediction Simulation")
        st.write("Predict excess minute revenue for a specific trip")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            station_id = st.selectbox(
                "Estación",
                options=['HB101', 'HB102', 'HB103', 'JC001', 'JC002', 'NY001', 'NY002'],
                index=0,
                key="single_pred_station"
            )
            
            hour = st.selectbox(
                "Hora del día",
                options=list(range(24)),
                index=17,
                key="single_pred_hour"
            )
            
            is_weekend = st.checkbox("¿Es fin de semana?", key="single_pred_weekend")
            
            month = st.selectbox(
                "Mes",
                options=list(range(1, 13)),
                index=5,
                format_func=lambda x: [
                    "Enero", "Febrero", "Marzo", "Abril", "Mayo", "Junio",
                    "Julio", "Agosto", "Septiembre", "Octubre", "Noviembre", "Diciembre"
                ][x-1],
                key="single_pred_month"
            )
        
        with col2:
            temperature = st.slider(
                "Temperatura (°C)",
                min_value=-10.0,
                max_value=40.0,
                value=20.0,
                step=0.5,
                key="single_pred_temp"
            )
            
            humidity = st.slider(
                "Humedad (%)",
                min_value=0,
                max_value=100,
                value=60,
                step=1,
                key="single_pred_humidity"
            )
            
            bike_type = st.selectbox(
                "Tipo de bicicleta",
                options=[1, 2],
                format_func=lambda x: "Clásica" if x == 1 else "Eléctrica",
                key="single_pred_bike_type"
            )
        
        with col3:
            duration = st.slider(
                "Duración del viaje (minutos)",
                min_value=5,
                max_value=120,
                value=35,
                step=1,
                key="single_pred_duration"
            )
            
            day_name = st.selectbox(
                "Día de la semana",
                options=["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"],
                format_func=lambda x: {
                    "Monday": "Lunes", "Tuesday": "Martes", "Wednesday": "Miércoles",
                    "Thursday": "Jueves", "Friday": "Viernes", "Saturday": "Sábado", "Sunday": "Domingo"
                }[x],
                key="single_pred_day_name"
            )
        
        if st.button("Make Prediction", type="primary", key="single_pred_button"):
            if supervised_model.best_model is None:
                st.error("Please train the supervised model first")
                return
            
            # Realizar predicción
            prediction = supervised_model.predict_single(
                station_id=station_id,
                hour=hour,
                is_weekend=is_weekend,
                month=month,
                temperature=temperature,
                humidity=humidity,
                bike_type=bike_type,
                duration=duration,
                day_name=day_name
            )
            
            # Mostrar resultados
            col1, col2, col3 = st.columns(3)
            
            with col1:
                st.metric(
                    "Predicted Revenue",
                    f"${prediction:.2f}"
                )
            
            with col2:
                # Calcular si habrá ingresos excedentes
                excedente_minutos = max(0, duration - 30)
                ingreso_teorico = excedente_minutos * 0.39
                st.metric(
                    "Theoretical Revenue",
                    f"${ingreso_teorico:.2f}"
                )
            
            with col3:
                diferencia = prediction - ingreso_teorico
                st.metric(
                    "Model Difference",
                    f"${diferencia:.2f}",
                    delta=f"{diferencia:.2f}"
                )
            
            # Explicación del resultado
            if prediction > 0:
                st.success(f"**Positive prediction**: Expected revenue of ${prediction:.2f} from excess minutes")
            else:
                st.info("**No excess revenue**: The model predicts this trip will not generate additional revenue")
            
            # Guardar escenario
            scenario = {
                'tipo': 'individual',
                'timestamp': datetime.now(),
                'parametros': {
                    'station_id': station_id, 'hour': hour, 'is_weekend': is_weekend,
                    'month': month, 'temperature': temperature, 'humidity': humidity,
                    'bike_type': bike_type, 'duration': duration
                },
                'resultado': prediction
            }
            self.scenarios.append(scenario)
    
    def batch_scenario_analysis(self):
        """Análisis de escenarios por lotes"""
        st.subheader("📊 Análisis de Escenarios por Lotes")
        st.write("Analiza múltiples combinaciones para encontrar los mejores escenarios")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.write("**Parámetros Variables:**")
            vary_hour = st.checkbox("Variar hora del día", value=True, key="batch_vary_hour")
            vary_temperature = st.checkbox("Variar temperatura", value=True, key="batch_vary_temp")
            vary_duration = st.checkbox("Variar duración del viaje", value=True, key="batch_vary_duration")
        
        with col2:
            st.write("**Parámetros Fijos:**")
            fixed_station = st.selectbox("Estación fija", ['HB101', 'HB102', 'HB103'], key="batch_fixed_station")
            fixed_month = st.selectbox("Mes fijo", list(range(1, 13)), index=5, key="batch_fixed_month")
            fixed_bike_type = st.selectbox("Tipo de bici fijo", [1, 2], key="batch_fixed_bike_type")
        
        if st.button("🔍 Ejecutar Análisis de Escenarios", key="batch_analysis_button"):
            if supervised_model.best_model is None:
                st.error("⚠️ Necesitas entrenar el modelo supervisado primero")
                return
                
            results = []
            
            # Definir rangos
            hours = list(range(24)) if vary_hour else [17]
            temperatures = list(range(5, 36, 5)) if vary_temperature else [20]
            durations = list(range(20, 81, 10)) if vary_duration else [35]
            
            progress_bar = st.progress(0)
            total_combinations = len(hours) * len(temperatures) * len(durations)
            current = 0
            
            for hour in hours:
                for temp in temperatures:
                    for duration in durations:
                        prediction = supervised_model.predict_single(
                            station_id=fixed_station,
                            hour=hour,
                            is_weekend=False,
                            month=fixed_month,
                            temperature=temp,
                            humidity=60,
                            bike_type=fixed_bike_type,
                            duration=duration
                        )
                        
                        results.append({
                            'hour': hour,
                            'temperature': temp,
                            'duration': duration,
                            'prediction': prediction
                        })
                        
                        current += 1
                        progress_bar.progress(current / total_combinations)
            
            # Convertir a DataFrame
            df_results = pd.DataFrame(results)
            
            # Visualizaciones
            if vary_hour and vary_temperature:
                fig = px.scatter(
                    df_results,
                    x='hour',
                    y='temperature',
                    size='prediction',
                    color='prediction',
                    title=f'Matriz de Predicciones - Estación {fixed_station}',
                    labels={'hour': 'Hora', 'temperature': 'Temperatura', 'prediction': 'Predicción ($)'}
                )
                st.plotly_chart(fig, use_container_width=True)
            
            # Top escenarios
            top_scenarios = df_results.nlargest(10, 'prediction')
            st.write("**🏆 Top 10 Mejores Escenarios:**")
            st.dataframe(top_scenarios)
            
            # Estadísticas
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Predicción Máxima", f"${df_results['prediction'].max():.2f}")
            with col2:
                st.metric("Predicción Promedio", f"${df_results['prediction'].mean():.2f}")
            with col3:
                st.metric("Escenarios Rentables", f"{(df_results['prediction'] > 0).sum()}")
    
    def anomaly_detection_scenario(self):
        """Simulación de detección de anomalías"""
        st.subheader("🚨 Simulación de Detección de Anomalías")
        st.write("Evalúa si una combinación estación-hora es anómala")
        
        if unsupervised_model.iso_forest is None:
            st.warning("⚠️ Necesitas entrenar el modelo de anomalías primero")
            return
        
        col1, col2 = st.columns(2)
        
        with col1:
            station_id = st.text_input("ID de Estación", value="HB102", key="anomaly_station_id")
            hour = st.selectbox("Hora", list(range(24)), index=17, key="anomaly_hour")
            is_weekend = st.checkbox("¿Es fin de semana?", key="anomaly_weekend")
            month = st.selectbox("Mes", list(range(1, 13)), index=5, key="anomaly_month")
        
        with col2:
            viajes_count = st.number_input("Número de viajes", value=50, min_value=1, key="anomaly_viajes_count")
            ingreso_promedio = st.number_input("Ingreso promedio ($)", value=2.5, min_value=0.0, key="anomaly_ingreso_promedio")
            duracion_promedio = st.number_input("Duración promedio (min)", value=35, min_value=5, key="anomaly_duracion_promedio")
            temperatura_promedio = st.number_input("Temperatura promedio (°C)", value=20.0, key="anomaly_temperatura_promedio")
        
        col3, col4 = st.columns(2)
        with col3:
            variabilidad_ingresos = st.number_input("Variabilidad ingresos", value=1.0, min_value=0.0, key="anomaly_variabilidad_ingresos")
        with col4:
            viajes_con_excedente = st.number_input("Viajes con excedente", value=5, min_value=0, key="anomaly_viajes_con_excedente")
        
        if st.button("🔍 Detectar Anomalía", key="anomaly_detect_button"):
            is_anomaly, score = unsupervised_model.predict_anomaly(
                station_id=station_id,
                hour=hour,
                is_weekend=is_weekend,
                month=month,
                viajes_count=viajes_count,
                ingreso_promedio=ingreso_promedio,
                duracion_promedio=duracion_promedio,
                temperatura_promedio=temperatura_promedio,
                variabilidad_ingresos=variabilidad_ingresos,
                viajes_con_excedente=viajes_con_excedente
            )
            
            col1, col2 = st.columns(2)
            
            with col1:
                if is_anomaly:
                    st.error("🚨 **ANOMALÍA DETECTADA**")
                    st.write("Esta combinación presenta patrones anómalos")
                else:
                    st.success("✅ **COMPORTAMIENTO NORMAL**")
                    st.write("Esta combinación presenta patrones normales")
            
            with col2:
                st.metric(
                    "Score de Anomalía",
                    f"{score:.4f}",
                    help="Valores más negativos indican mayor probabilidad de anomalía"
                )
    
    def counterfactual_scenario(self):
        """Análisis contrafactual de eventos de desabasto"""
        st.subheader("🔄 Análisis Contrafactual de Desabasto")
        st.write("Simula el impacto de eventos de desabasto en los ingresos")
        
        if supervised_model.best_model is None:
            st.error("⚠️ Necesitas entrenar el modelo supervisado primero")
            return
        
        # Opciones de simulación
        simulation_type = st.radio(
            "Tipo de simulación:",
            ["Evento específico", "Impacto por estación", "Análisis temporal"],
            key="counterfactual_simulation_type"
        )
        
        if simulation_type == "Evento específico":
            st.write("**Simula un evento de desabasto específico:**")
            
            col1, col2 = st.columns(2)
            
            with col1:
                station_id = st.selectbox("Estación afectada", ['HB101', 'HB102', 'HB103'], key="counter_station_id")
                hour = st.selectbox("Hora del evento", list(range(24)), index=17, key="counter_hour")
                duration_hours = st.slider("Duración del desabasto (horas)", 1, 8, 2, key="counter_duration_hours")
            
            with col2:
                viajes_perdidos = st.number_input("Viajes perdidos estimados", value=25, min_value=1, key="counter_viajes_perdidos")
                temperature = st.slider("Temperatura durante evento (°C)", -5, 35, 20, key="counter_temperature")
                day_type = st.selectbox("Tipo de día", ["Laboral", "Fin de semana"], key="counter_day_type")
            
            if st.button("💰 Calcular Pérdida Estimada", key="counter_calc_loss_button"):
                is_weekend = day_type == "Fin de semana"
                total_loss = 0
                
                for i in range(viajes_perdidos):
                    # Generar duración aleatoria realista
                    duration = np.random.normal(35, 15)
                    duration = max(10, min(120, duration))
                    
                    predicted_revenue = supervised_model.predict_single(
                        station_id=station_id,
                        hour=hour,
                        is_weekend=is_weekend,
                        month=6,  # Junio por defecto
                        temperature=temperature,
                        humidity=60,
                        bike_type=1,
                        duration=duration
                    )
                    
                    total_loss += predicted_revenue
                
                # Mostrar resultados
                col1, col2, col3 = st.columns(3)
                
                with col1:
                    st.metric("💸 Pérdida Total", f"${total_loss:.2f}")
                
                with col2:
                    st.metric("📊 Pérdida por Viaje", f"${total_loss/viajes_perdidos:.2f}")
                
                with col3:
                    st.metric("⏰ Pérdida por Hora", f"${total_loss/duration_hours:.2f}")
                
                # Impacto anual estimado
                eventos_anuales = st.slider("Eventos similares por año", 1, 50, 10, key="counter_eventos_anuales")
                impacto_anual = total_loss * eventos_anuales
                
                st.info(f"📈 **Impacto Anual Estimado**: ${impacto_anual:,.2f}")
        
        elif simulation_type == "Impacto por estación":
            st.write("**Compara el impacto de desabasto entre estaciones:**")
            
            stations = st.multiselect(
                "Selecciona estaciones para comparar",
                ['HB101', 'HB102', 'HB103', 'JC001', 'JC002'],
                default=['HB101', 'HB102', 'HB103'],
                key="counter_compare_stations"
            )
            
            standard_params = {
                'hour': 17,
                'viajes_perdidos': 30,
                'temperature': 20,
                'is_weekend': False
            }
            
            if st.button("📊 Comparar Estaciones", key="counter_compare_button") and stations:
                results = []
                
                for station in stations:
                    total_loss = 0
                    
                    for i in range(standard_params['viajes_perdidos']):
                        duration = np.random.normal(35, 15)
                        duration = max(10, min(120, duration))
                        
                        predicted_revenue = supervised_model.predict_single(
                            station_id=station,
                            hour=standard_params['hour'],
                            is_weekend=standard_params['is_weekend'],
                            month=6,
                            temperature=standard_params['temperature'],
                            humidity=60,
                            bike_type=1,
                            duration=duration
                        )
                        
                        total_loss += predicted_revenue
                    
                    results.append({
                        'Estación': station,
                        'Pérdida Estimada': total_loss,
                        'Pérdida por Viaje': total_loss / standard_params['viajes_perdidos']
                    })
                
                df_results = pd.DataFrame(results)
                
                # Visualizar
                fig = px.bar(
                    df_results,
                    x='Estación',
                    y='Pérdida Estimada',
                    title='Impacto Económico de Desabasto por Estación',
                    color='Pérdida Estimada'
                )
                st.plotly_chart(fig, use_container_width=True)
                
                st.dataframe(df_results)
    
    def what_if_optimizer(self):
        """Optimizador de escenarios 'What-If'"""
        st.subheader("🎯 Optimizador de Escenarios")
        st.write("Encuentra las condiciones óptimas para maximizar ingresos")
        
        if supervised_model.best_model is None:
            st.error("⚠️ Necesitas entrenar el modelo supervisado primero")
            return
        
        objective = st.selectbox(
            "Objetivo de optimización:",
            ["Maximizar ingresos", "Identificar horarios críticos", "Optimizar por clima"],
            key="optimizer_objective"
        )
        
        if objective == "Maximizar ingresos":
            station_id = st.selectbox("Estación objetivo", ['HB101', 'HB102', 'HB103'], key="optimizer_station_id")
            
            if st.button("🔍 Encontrar Condiciones Óptimas", key="optimizer_find_button"):
                best_scenario = None
                best_revenue = 0
                
                # Grid search sobre parámetros principales
                progress = st.progress(0)
                total_combinations = 24 * 12 * 2 * 7  # hour * month * weekend * duration_categories
                current = 0
                
                for hour in range(24):
                    for month in range(1, 13):
                        for is_weekend in [False, True]:
                            for duration_cat in [25, 35, 45, 60, 75, 90, 105]:
                                prediction = supervised_model.predict_single(
                                    station_id=station_id,
                                    hour=hour,
                                    is_weekend=is_weekend,
                                    month=month,
                                    temperature=20,  # Temperatura estándar
                                    humidity=60,
                                    bike_type=1,
                                    duration=duration_cat
                                )
                                
                                if prediction > best_revenue:
                                    best_revenue = prediction
                                    best_scenario = {
                                        'hour': hour,
                                        'month': month,
                                        'is_weekend': is_weekend,
                                        'duration': duration_cat,
                                        'revenue': prediction
                                    }
                                
                                current += 1
                                if current % 100 == 0:
                                    progress.progress(current / total_combinations)
                
                # Mostrar mejor escenario
                if best_scenario:
                    st.success("🎉 **Escenario Óptimo Encontrado:**")
                    
                    col1, col2 = st.columns(2)
                    
                    with col1:
                        st.write(f"**Estación**: {station_id}")
                        st.write(f"**Hora**: {best_scenario['hour']}:00")
                        st.write(f"**Mes**: {best_scenario['month']}")
                        st.write(f"**Tipo de día**: {'Fin de semana' if best_scenario['is_weekend'] else 'Laboral'}")
                    
                    with col2:
                        st.write(f"**Duración óptima**: {best_scenario['duration']} min")
                        st.metric("💰 Ingreso Máximo", f"${best_scenario['revenue']:.2f}")
                    
                    st.info(f"💡 **Recomendación**: Los viajes de {best_scenario['duration']} minutos a las {best_scenario['hour']}:00 horas en {['días laborales', 'fines de semana'][best_scenario['is_weekend']]} del mes {best_scenario['month']} maximizan los ingresos en la estación {station_id}")

# Instancia global del simulador
simulator = Simulator()

def run_simulation_interface():
    """Interfaz principal de simulaciones"""
    st.title("CitiBike Simulation Center")
    st.write("Explore different scenarios and make predictions with your trained models")
    
    simulation_tabs = st.tabs([
        "Individual Prediction",
        "Batch Analysis", 
        "Anomaly Detection",
        "Counterfactual Analysis",
        "Optimizer"
    ])
    
    with simulation_tabs[0]:
        simulator.single_prediction_scenario()
    
    with simulation_tabs[1]:
        simulator.batch_scenario_analysis()
    
    with simulation_tabs[2]:
        simulator.anomaly_detection_scenario()
    
    with simulation_tabs[3]:
        simulator.counterfactual_scenario()
    
    with simulation_tabs[4]:
        simulator.what_if_optimizer()
    
    # Historial de simulaciones
    if len(simulator.scenarios) > 0:
        with st.expander("Simulation History"):
            for i, scenario in enumerate(reversed(simulator.scenarios[-10:])):
                st.write(f"**{scenario['tipo'].title()}** - {scenario['timestamp'].strftime('%H:%M:%S')}: ${scenario['resultado']:.2f}") 