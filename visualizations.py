import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import numpy as np

def plot_model_comparison(results: dict):
    """Gráfico de comparación de modelos"""
    models = list(results.keys())
    r2_scores = [results[model]['r2'] for model in models]
    rmse_scores = [results[model]['rmse'] for model in models]
    
    fig = make_subplots(
        rows=1, cols=2,
        subplot_titles=('R² Score (Higher is better)', 'RMSE (Lower is better)'),
        specs=[[{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # R² Score
    fig.add_trace(
        go.Bar(x=models, y=r2_scores, name='R² Score', marker_color='lightblue'),
        row=1, col=1
    )
    
    # RMSE
    fig.add_trace(
        go.Bar(x=models, y=rmse_scores, name='RMSE', marker_color='lightcoral'),
        row=1, col=2
    )
    
    fig.update_layout(
        title_text="Model Performance Comparison",
        showlegend=False,
        height=400
    )
    
    return fig

def plot_feature_importance(model, feature_names):
    """Gráfico de importancia de features"""
    if hasattr(model, 'feature_importances_'):
        importance_df = pd.DataFrame({
            'feature': feature_names,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=True).tail(15)
        
        fig = px.bar(
            importance_df, 
            x='importance', 
            y='feature',
            orientation='h',
            title='Top 15 Most Important Features',
            labels={'importance': 'Importance', 'feature': 'Feature'}
        )
        fig.update_layout(height=500)
        return fig
    return None

def plot_prediction_distribution(predictions):
    """Distribución de predicciones"""
    fig = px.histogram(
        x=predictions,
        nbins=50,
        title='Revenue Prediction Distribution',
        labels={'x': 'Predicted Revenue ($)', 'y': 'Frequency'}
    )
    fig.add_vline(
        x=np.mean(predictions), 
        line_dash="dash", 
        line_color="red",
        annotation_text=f"Mean: ${np.mean(predictions):.2f}"
    )
    return fig

def plot_anomalies_scatter(df_clean, anomalias_consenso):
    """Scatter plot de anomalías"""
    fig = go.Figure()
    
    # Puntos normales
    normal_data = df_clean[
        (df_clean['anomalia_iso'] == 1) & 
        (df_clean['anomalia_lof'] == 1)
    ]
    
    fig.add_trace(go.Scatter(
        x=normal_data['INGRESO_PROMEDIO'],
        y=normal_data['VIAJES_COUNT'],
        mode='markers',
        name='Normal',
        marker=dict(color='blue', size=4, opacity=0.6)
    ))
    
    # Anomalías
    fig.add_trace(go.Scatter(
        x=anomalias_consenso['INGRESO_PROMEDIO'],
        y=anomalias_consenso['VIAJES_COUNT'],
        mode='markers',
        name='Anomalies',
        marker=dict(color='red', size=8, opacity=0.8)
    ))
    
    fig.update_layout(
        title='Anomaly Detection: Revenue vs Trips',
        xaxis_title='Average Revenue ($)',
        yaxis_title='Number of Trips',
        showlegend=True
    )
    
    return fig

def plot_anomalies_by_hour(anomalias_consenso):
    """Anomalías por hora"""
    anomalias_hora = anomalias_consenso.groupby('HOUR').size().reset_index(name='num_anomalias')
    
    fig = px.bar(
        anomalias_hora,
        x='HOUR',
        y='num_anomalias',
        title='Number of Anomalies by Hour of Day',
        labels={'HOUR': 'Hour', 'num_anomalias': 'Number of Anomalies'}
    )
    
    fig.update_layout(
        xaxis=dict(tickmode='linear', tick0=0, dtick=1)
    )
    
    return fig

def plot_revenue_analysis(df):
    """Análisis de ingresos por diferentes dimensiones"""
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=(
            'Revenue by Hour',
            'Revenue by User Type',
            'Revenue by Day of Week',
            'Revenue by Month'
        )
    )
    
    # Por hora
    ingresos_hora = df.groupby('HOUR')['INGRESO_MIN_EXCEDENTE'].sum().reset_index()
    fig.add_trace(
        go.Scatter(x=ingresos_hora['HOUR'], y=ingresos_hora['INGRESO_MIN_EXCEDENTE'], 
                  mode='lines+markers', name='Por Hora'),
        row=1, col=1
    )
    
    # Por tipo de usuario
    if 'MEMBER_CASUAL' in df.columns:
        ingresos_usuario = df.groupby('MEMBER_CASUAL')['INGRESO_MIN_EXCEDENTE'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=ingresos_usuario['MEMBER_CASUAL'], y=ingresos_usuario['INGRESO_MIN_EXCEDENTE'],
                  name='Por Usuario'),
            row=1, col=2
        )
    
    # Por día de la semana
    if 'DAY_NAME' in df.columns:
        ingresos_dia = df.groupby('DAY_NAME')['INGRESO_MIN_EXCEDENTE'].sum().reset_index()
        fig.add_trace(
            go.Bar(x=ingresos_dia['DAY_NAME'], y=ingresos_dia['INGRESO_MIN_EXCEDENTE'],
                  name='Por Día'),
            row=2, col=1
        )
    
    # Por mes
    ingresos_mes = df.groupby('MONTH')['INGRESO_MIN_EXCEDENTE'].sum().reset_index()
    fig.add_trace(
        go.Bar(x=ingresos_mes['MONTH'], y=ingresos_mes['INGRESO_MIN_EXCEDENTE'],
              name='Por Mes'),
        row=2, col=2
    )
    
    fig.update_layout(
        title_text="Revenue Analysis by Dimensions",
        showlegend=False,
        height=600
    )
    
    return fig

def plot_station_balance(df):
    """Balance de estaciones"""
    if 'START_STATION_ID' in df.columns and 'END_STATION_ID' in df.columns:
        # Salidas por estación
        salidas = df.groupby('START_STATION_ID').size().reset_index(name='salidas')
        salidas.rename(columns={'START_STATION_ID': 'station_id'}, inplace=True)
        
        # Llegadas por estación
        llegadas = df.groupby('END_STATION_ID').size().reset_index(name='llegadas')
        llegadas.rename(columns={'END_STATION_ID': 'station_id'}, inplace=True)
        
        # Merge y calcular balance
        balance = pd.merge(salidas, llegadas, on='station_id', how='outer').fillna(0)
        balance['balance_neto'] = balance['llegadas'] - balance['salidas']
        
        # Top 10 con déficit y exceso
        deficit = balance.nsmallest(10, 'balance_neto')
        exceso = balance.nlargest(10, 'balance_neto')
        
        fig = make_subplots(
            rows=1, cols=2,
            subplot_titles=('Top 10 Estaciones con Déficit', 'Top 10 Estaciones con Exceso')
        )
        
        fig.add_trace(
            go.Bar(x=deficit['station_id'], y=deficit['balance_neto'],
                  name='Déficit', marker_color='red'),
            row=1, col=1
        )
        
        fig.add_trace(
            go.Bar(x=exceso['station_id'], y=exceso['balance_neto'],
                  name='Exceso', marker_color='green'),
            row=1, col=2
        )
        
        fig.update_layout(
            title_text="Balance de Bicicletas por Estación",
            showlegend=False,
            height=400
        )
        
        return fig
    
    return None

def plot_weather_impact(df):
    """Impacto del clima en los ingresos"""
    if 'TEMPERATURE_2M' in df.columns:
        try:
            # Crear bins de temperatura - convertir a float para evitar problemas con Decimal
            df_temp = df.copy()
            df_temp['TEMPERATURE_2M'] = pd.to_numeric(df_temp['TEMPERATURE_2M'], errors='coerce')
            df_temp = df_temp.dropna(subset=['TEMPERATURE_2M'])
            
            if len(df_temp) > 0:
                df_temp['temp_bin'] = pd.cut(df_temp['TEMPERATURE_2M'], bins=10, labels=False)
                temp_ingresos = df_temp.groupby('temp_bin').agg({
                    'INGRESO_MIN_EXCEDENTE': 'mean',
                    'TEMPERATURE_2M': 'mean'
                }).reset_index()
                
                fig = px.scatter(
                    temp_ingresos,
                    x='TEMPERATURE_2M',
                    y='INGRESO_MIN_EXCEDENTE',
                    size='temp_bin',
                    title='Relación entre Temperatura e Ingresos',
                    labels={
                        'TEMPERATURE_2M': 'Temperatura (°C)',
                        'INGRESO_MIN_EXCEDENTE': 'Ingreso Promedio ($)'
                    }
                )
                
                return fig
        except Exception as e:
            print(f" Error al crear gráfico de clima: {str(e)}")
    
    return None

def plot_counterfactual_analysis(df_desabasto, ingresos_perdidos):
    """Análisis contrafactual de pérdidas"""
    if len(df_desabasto) > 0:
        # Pérdidas por estación
        perdidas_estacion = df_desabasto.groupby('STATION_ID').agg({
            'VIAJES_PERDIDOS_ESTIMADOS': 'sum'
        }).reset_index()
        perdidas_estacion['ingresos_perdidos'] = perdidas_estacion['VIAJES_PERDIDOS_ESTIMADOS'] * (ingresos_perdidos / df_desabasto['VIAJES_PERDIDOS_ESTIMADOS'].sum())
        perdidas_estacion = perdidas_estacion.nlargest(10, 'ingresos_perdidos')
        
        fig = px.bar(
            perdidas_estacion,
            x='STATION_ID',
            y='ingresos_perdidos',
            title='Top 10 Estaciones con Mayores Pérdidas Estimadas',
            labels={
                'STATION_ID': 'Estación',
                'ingresos_perdidos': 'Ingresos Perdidos ($)'
            }
        )
        
        return fig
    
    return None

def create_kpi_metrics(df, anomalies_data=None, counterfactual_loss=None):
    """Crea métricas KPI para mostrar en el dashboard"""
    kpis = {}
    
    if len(df) > 0:
        kpis['total_viajes'] = int(len(df))
        kpis['ingresos_totales'] = float(df['INGRESO_MIN_EXCEDENTE'].sum())
        kpis['ingreso_promedio'] = float(df['INGRESO_MIN_EXCEDENTE'].mean())
        kpis['estaciones_activas'] = int(df['START_STATION_ID'].nunique()) if 'START_STATION_ID' in df.columns else 0
        
        # Porcentaje de viajes con ingresos
        kpis['viajes_con_ingresos'] = float((df['INGRESO_MIN_EXCEDENTE'] > 0).mean() * 100)
    
    if anomalies_data:
        kpis['anomalias_detectadas'] = int(anomalies_data['num_consenso'])
        kpis['porcentaje_anomalias'] = float((anomalies_data['num_consenso'] / len(anomalies_data['df_clean'])) * 100)
    
    if counterfactual_loss:
        kpis['perdidas_estimadas'] = float(counterfactual_loss)
    
    return kpis 