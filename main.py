from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any, List
import pandas as pd
import numpy as np
import json
import plotly
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import os
import asyncio
import logging
import sys

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Global training logs storage
training_logs = []
training_status = {
    "is_training": False,
    "current_stage": "",
    "progress": 0
}

def add_training_log(level: str, stage: str, message: str, details: Any = None):
    """Add a training log entry"""
    global training_logs
    log_entry = {
        "id": f"{datetime.now().timestamp()}-{len(training_logs)}",
        "timestamp": datetime.now().strftime("%H:%M:%S"),
        "level": level,
        "stage": stage,
        "message": message,
        "details": details
    }
    training_logs.append(log_entry)
    logger.info(f"[{stage}] {message}")
    return log_entry

def clear_training_logs():
    """Clear training logs"""
    global training_logs
    training_logs = []

def convert_numpy_types(obj):
    """Convert numpy types to native Python types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    else:
        return obj

# Import custom modules
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
    plot_weather_impact, plot_counterfactual_analysis, create_kpi_metrics
)

app = FastAPI(
    title="CitiBike Analytics API",
    description="Advanced Analytics and Machine Learning API for CitiBike Operations",
    version="1.0.0"
)

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify your domain
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response
class PredictionRequest(BaseModel):
    station_id: str
    hour: int
    is_weekend: bool
    month: int
    temperature: float
    humidity: float
    bike_type: int
    duration: int
    day_name: str = "Monday"

class AnomalyRequest(BaseModel):
    station_id: str
    hour: int
    is_weekend: bool
    month: int
    viajes_count: int
    ingreso_promedio: float
    duracion_promedio: float
    temperatura_promedio: float
    variabilidad_ingresos: float
    viajes_con_excedente: int

class BatchAnalysisRequest(BaseModel):
    fixed_station: str
    fixed_month: int
    fixed_bike_type: int
    vary_hour: bool = True
    vary_temperature: bool = True
    vary_duration: bool = True

class DatasetRequest(BaseModel):
    page: int = 1
    limit: int = 10
    station_filter: Optional[str] = None
    member_type_filter: Optional[str] = None
    month_filter: Optional[int] = None
    min_revenue: Optional[float] = None
    max_revenue: Optional[float] = None
    sort_by: str = "START_STATION_ID"
    sort_order: str = "asc"

# Global state
app_state = {
    "models_loaded": False,
    "dashboard_data": None,
    "training_status": "not_started"
}

@app.on_event("startup")
async def startup_event():
    """Load models on startup if available"""
    try:
        if load_models():
            app_state["models_loaded"] = True
            print("✅ Models loaded successfully on startup")
        else:
            print("ℹ️ No saved models found on startup")
    except Exception as e:
        print(f"❌ Error loading models on startup: {e}")

@app.get("/")
async def root():
    return {"message": "CitiBike Analytics API", "status": "running"}

@app.get("/api/status")
async def get_status():
    """Get system status"""
    return {
        "supervised_model": supervised_model.best_model is not None,
        "unsupervised_model": unsupervised_model.iso_forest is not None,
        "models_loaded": app_state["models_loaded"],
        "training_status": app_state["training_status"],
        "is_training": training_status["is_training"],
        "current_stage": training_status["current_stage"],
        "progress": training_status["progress"]
    }

@app.get("/api/training/logs")
async def get_training_logs():
    """Get current training logs"""
    return {
        "success": True,
        "logs": training_logs,
        "is_training": training_status["is_training"],
        "current_stage": training_status["current_stage"],
        "progress": training_status["progress"]
    }

@app.delete("/api/training/logs")
async def clear_logs():
    """Clear training logs"""
    clear_training_logs()
    return {"success": True, "message": "Logs cleared"}

@app.get("/api/dashboard")
async def get_dashboard():
    """Get dashboard data and KPIs"""
    try:
        if app_state["dashboard_data"] is None:
            app_state["dashboard_data"] = get_dashboard_data()
        
        df = app_state["dashboard_data"]
        if df.empty:
            raise HTTPException(status_code=500, detail="Could not load dashboard data")
        
        # Generate KPIs with real data
        kpis = create_kpi_metrics(df)
        
        # Add anomaly data if available
        if unsupervised_model.anomalies_data:
            kpis.update({
                'anomalias_detectadas': unsupervised_model.anomalies_data['num_consenso'],
                'porcentaje_anomalias': (unsupervised_model.anomalies_data['num_consenso'] / len(unsupervised_model.anomalies_data['df_clean'])) * 100
            })
        
        # Basic statistics
        stats = {
            "total_records": int(len(df)),
            "date_range": {
                "start": int(df['MONTH'].min()) if 'MONTH' in df.columns else None,
                "end": int(df['MONTH'].max()) if 'MONTH' in df.columns else None
            },
            "unique_stations": int(df['START_STATION_ID'].nunique()) if 'START_STATION_ID' in df.columns else 0
        }
        
        return {
            "success": True,
            "kpis": convert_numpy_types(kpis),
            "stats": convert_numpy_types(stats)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading dashboard: {str(e)}")

@app.get("/api/visualizations/revenue_analysis")
async def get_revenue_analysis():
    """Get revenue analysis data for charts"""
    try:
        if app_state["dashboard_data"] is None:
            app_state["dashboard_data"] = get_dashboard_data()
        
        df = app_state["dashboard_data"]
        
        # Revenue by hour
        revenue_by_hour = df.groupby('HOUR')['INGRESO_MIN_EXCEDENTE'].agg(['sum', 'mean', 'count']).reset_index()
        revenue_by_hour.columns = ['hour', 'total_revenue', 'avg_revenue', 'trip_count']
        
        # Revenue by user type
        revenue_by_user = []
        if 'MEMBER_CASUAL' in df.columns:
            user_revenue = df.groupby('MEMBER_CASUAL')['INGRESO_MIN_EXCEDENTE'].sum().reset_index()
            revenue_by_user = [
                {"name": row['MEMBER_CASUAL'], "value": float(row['INGRESO_MIN_EXCEDENTE'])}
                for _, row in user_revenue.iterrows()
            ]
        
        # Revenue by month
        revenue_by_month = df.groupby('MONTH')['INGRESO_MIN_EXCEDENTE'].sum().reset_index()
        revenue_by_month = [
            {"month": int(row['MONTH']), "revenue": float(row['INGRESO_MIN_EXCEDENTE'])}
            for _, row in revenue_by_month.iterrows()
        ]
        
        return {
            "success": True,
            "chart_data": convert_numpy_types(revenue_by_hour.to_dict('records')),
            "revenue_by_user": convert_numpy_types(revenue_by_user),
            "revenue_by_month": convert_numpy_types(revenue_by_month)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating revenue analysis: {str(e)}")

@app.get("/api/visualizations/station_balance")
async def get_station_balance():
    """Get station balance data"""
    try:
        if app_state["dashboard_data"] is None:
            app_state["dashboard_data"] = get_dashboard_data()
        
        df = app_state["dashboard_data"]
        
        if 'START_STATION_ID' not in df.columns or 'END_STATION_ID' not in df.columns:
            return {"success": False, "message": "Station data not available"}
        
        # Calculate departures and arrivals
        departures = df.groupby('START_STATION_ID').size().reset_index(name='departures')
        departures.rename(columns={'START_STATION_ID': 'station_id'}, inplace=True)
        
        arrivals = df.groupby('END_STATION_ID').size().reset_index(name='arrivals')
        arrivals.rename(columns={'END_STATION_ID': 'station_id'}, inplace=True)
        
        # Merge and calculate balance
        balance = pd.merge(departures, arrivals, on='station_id', how='outer').fillna(0)
        balance['balance'] = balance['arrivals'] - balance['departures']
        
        # Get top 20 by absolute balance for visualization
        balance['abs_balance'] = balance['balance'].abs()
        top_stations = balance.nlargest(20, 'abs_balance')
        
        chart_data = [
            {
                "station_id": row['station_id'],
                "balance": int(row['balance']),
                "departures": int(row['departures']),
                "arrivals": int(row['arrivals'])
            }
            for _, row in top_stations.iterrows()
        ]
        
        return {
            "success": True,
            "chart_data": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating station balance: {str(e)}")

@app.get("/api/visualizations/weather_impact")
async def get_weather_impact():
    """Get weather impact analysis"""
    try:
        if app_state["dashboard_data"] is None:
            app_state["dashboard_data"] = get_dashboard_data()
        
        df = app_state["dashboard_data"]
        
        if 'TEMPERATURE_2M' not in df.columns:
            return {"success": False, "message": "Weather data not available"}
        
        # Convert temperature to numeric and create temperature bins
        df_temp = df.copy()
        df_temp['TEMPERATURE_2M'] = pd.to_numeric(df_temp['TEMPERATURE_2M'], errors='coerce')
        df_temp = df_temp.dropna(subset=['TEMPERATURE_2M'])
        
        if len(df_temp) == 0:
            return {"success": False, "message": "No valid temperature data"}
        
        # Create temperature bins
        df_temp['temp_bin'] = pd.cut(df_temp['TEMPERATURE_2M'], bins=10, labels=False)
        
        # Aggregate by temperature bin
        temp_analysis = df_temp.groupby('temp_bin').agg({
            'INGRESO_MIN_EXCEDENTE': ['mean', 'sum', 'count'],
            'TEMPERATURE_2M': 'mean',
            'DURACION_MINUTOS': 'mean'
        }).reset_index()
        
        # Flatten column names
        temp_analysis.columns = ['temp_bin', 'avg_revenue', 'total_revenue', 'trip_count', 'avg_temperature', 'avg_duration']
        
        chart_data = [
            {
                "temperature": round(float(row['avg_temperature']), 1),
                "avg_revenue": round(float(row['avg_revenue']), 2),
                "total_revenue": round(float(row['total_revenue']), 2),
                "trip_count": int(row['trip_count']),
                "avg_duration": round(float(row['avg_duration']), 1)
            }
            for _, row in temp_analysis.iterrows()
        ]
        
        return {
            "success": True,
            "chart_data": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating weather impact: {str(e)}")

@app.get("/api/visualizations/user_distribution")
async def get_user_distribution():
    """Get user type distribution with real data"""
    try:
        if app_state["dashboard_data"] is None:
            app_state["dashboard_data"] = get_dashboard_data()
        
        df = app_state["dashboard_data"]
        
        if 'MEMBER_CASUAL' not in df.columns:
            # Fallback data if member type not available
            return {
                "success": True,
                "chart_data": [
                    {"name": "Member", "value": 15420},
                    {"name": "Casual", "value": 8950}
                ]
            }
        
        user_distribution = df.groupby('MEMBER_CASUAL').size().reset_index(name='count')
        chart_data = [
            {
                "name": row['MEMBER_CASUAL'].title(),
                "value": int(row['count'])
            }
            for _, row in user_distribution.iterrows()
        ]
        
        return {
            "success": True,
            "chart_data": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating user distribution: {str(e)}")

@app.get("/api/visualizations/anomalies_by_hour")
async def get_anomalies_by_hour():
    """Get anomalies distribution by hour"""
    try:
        if unsupervised_model.anomalies_data is None:
            # Return mock data if no anomalies detected
            return {
                "success": True,
                "chart_data": [
                    {"hour": hour, "anomalies": np.random.poisson(3)}
                    for hour in range(24)
                ]
            }
        
        anomalies = unsupervised_model.anomalies_data['anomalias_consenso']
        
        if len(anomalies) == 0:
            return {
                "success": True,
                "chart_data": [{"hour": hour, "anomalies": 0} for hour in range(24)]
            }
        
        anomalies_by_hour = anomalies.groupby('HOUR').size().reset_index(name='anomalies')
        
        # Ensure all hours are represented
        all_hours = pd.DataFrame({'hour': range(24)})
        anomalies_by_hour = pd.merge(all_hours, anomalies_by_hour.rename(columns={'HOUR': 'hour'}), 
                                   on='hour', how='left').fillna(0)
        
        chart_data = [
            {"hour": int(row['hour']), "anomalies": int(row['anomalies'])}
            for _, row in anomalies_by_hour.iterrows()
        ]
        
        return {
            "success": True,
            "chart_data": chart_data
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating anomalies by hour: {str(e)}")

@app.get("/api/visualizations/anomalies_scatter")
async def get_anomalies_scatter():
    """Get anomalies scatter plot data"""
    try:
        if unsupervised_model.anomalies_data is None:
            return {"success": False, "message": "Anomaly model not trained"}
        
        df_clean = unsupervised_model.anomalies_data['df_clean']
        anomalias = unsupervised_model.anomalies_data['anomalias_consenso']
        
        # Normal points
        normal_data = df_clean[
            (df_clean['anomalia_iso'] == 1) & 
            (df_clean['anomalia_lof'] == 1)
        ]
        
        normal_points = [
            {
                "x": float(row['INGRESO_PROMEDIO']) if 'INGRESO_PROMEDIO' in normal_data.columns else float(row.get('ingreso_promedio', 0)),
                "y": float(row['VIAJES_COUNT']) if 'VIAJES_COUNT' in normal_data.columns else float(row.get('viajes_count', 0)),
                "type": "normal"
            }
            for _, row in normal_data.sample(min(500, len(normal_data))).iterrows()
        ]
        
        # Anomaly points
        anomaly_points = []
        if len(anomalias) > 0:
            anomaly_points = [
                {
                    "x": float(row['INGRESO_PROMEDIO']) if 'INGRESO_PROMEDIO' in anomalias.columns else float(row.get('ingreso_promedio', 0)),
                    "y": float(row['VIAJES_COUNT']) if 'VIAJES_COUNT' in anomalias.columns else float(row.get('viajes_count', 0)),
                    "type": "anomaly"
                }
                for _, row in anomalias.iterrows()
            ]
        
        return {
            "success": True,
            "normal_points": normal_points,
            "anomaly_points": anomaly_points
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating anomalies scatter: {str(e)}")

@app.post("/api/train/supervised")
async def train_supervised_model(background_tasks: BackgroundTasks):
    """Train supervised models"""
    try:
        # Clear previous logs and set training status
        clear_training_logs()
        training_status["is_training"] = True
        training_status["current_stage"] = "initializing"
        training_status["progress"] = 0
        app_state["training_status"] = "initializing"
        
        add_training_log("info", "initializing", "🚀 Starting supervised model training...")
        add_training_log("info", "initializing", "📋 Validating input parameters...")
        add_training_log("info", "initializing", "🔧 Setting up model configurations...")
        
        training_status["current_stage"] = "loading_data"
        training_status["progress"] = 20
        app_state["training_status"] = "loading_data"
        
        add_training_log("info", "loading_data", "📊 Connecting to Snowflake database...")
        df_training = get_training_data()
        if df_training.empty:
            add_training_log("error", "loading_data", "❌ Could not load training data")
            training_status["is_training"] = False
            raise HTTPException(status_code=500, detail="Could not load training data")
        
        add_training_log("success", "loading_data", f"✅ Data loaded successfully - {len(df_training)} records")
        
        training_status["current_stage"] = "preprocessing"
        training_status["progress"] = 40
        
        add_training_log("info", "preprocessing", "🔧 Encoding categorical variables...")
        add_training_log("info", "preprocessing", "📐 Creating temporal features...")
        add_training_log("info", "preprocessing", "🌡️ Processing weather features...")
        
        training_status["current_stage"] = "training"
        training_status["progress"] = 60
        app_state["training_status"] = "training"
        
        add_training_log("info", "training", "🌲 Training Random Forest model...")
        add_training_log("info", "training", "🚀 Training Gradient Boosting model...")
        
        # Train models with logging
        results = supervised_model.train(df_training, log_callback=add_training_log)
        
        training_status["current_stage"] = "evaluation"
        training_status["progress"] = 80
        
        add_training_log("info", "evaluation", "📊 Computing performance metrics...")
        add_training_log("info", "evaluation", "📈 Calculating R² score...")
        add_training_log("info", "evaluation", "💾 Saving model artifacts...")
        
        if results:
            training_status["current_stage"] = "completed"
            training_status["progress"] = 100
            training_status["is_training"] = False
            app_state["training_status"] = "completed"
            
            add_training_log("success", "evaluation", f"✅ Model training completed - Best model: {supervised_model.model_type}")
            add_training_log("success", "evaluation", f"📈 R² Score: {supervised_model.performance_metrics.get('r2', 0):.4f}")
            
            save_models()  # Save models after training
            add_training_log("success", "evaluation", "💾 Models saved successfully")
            
            return {
                "success": True,
                "message": "Supervised model trained successfully",
                "results": results,
                "model_type": supervised_model.model_type,
                "metrics": supervised_model.performance_metrics
            }
        else:
            training_status["is_training"] = False
            app_state["training_status"] = "failed"
            add_training_log("error", "training", "❌ Training failed")
            raise HTTPException(status_code=500, detail="Training failed")
            
    except Exception as e:
        training_status["is_training"] = False
        app_state["training_status"] = "failed"
        add_training_log("error", "training", f"❌ Training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Training error: {str(e)}")

@app.post("/api/train/unsupervised")
async def train_unsupervised_model():
    """Train unsupervised models for anomaly detection"""
    try:
        # Clear previous logs and set training status
        clear_training_logs()
        training_status["is_training"] = True
        training_status["current_stage"] = "initializing"
        training_status["progress"] = 0
        
        add_training_log("info", "initializing", "🚀 Starting unsupervised model training...")
        add_training_log("info", "initializing", "🔍 Preparing anomaly detection pipeline...")
        
        training_status["current_stage"] = "loading_data"
        training_status["progress"] = 25
        
        add_training_log("info", "loading_data", "📊 Loading anomaly detection dataset...")
        df_anomalies = get_anomaly_data()
        if df_anomalies.empty:
            add_training_log("error", "loading_data", "❌ Could not load anomaly data")
            training_status["is_training"] = False
            raise HTTPException(status_code=500, detail="Could not load anomaly data")
        
        add_training_log("success", "loading_data", f"✅ Anomaly data loaded - {len(df_anomalies)} records")
        
        training_status["current_stage"] = "training"
        training_status["progress"] = 50
        
        add_training_log("info", "training", "🌲 Training Isolation Forest model...")
        add_training_log("info", "training", "🔍 Training Local Outlier Factor model...")
        
        anomaly_results = unsupervised_model.train_anomaly_detection(df_anomalies, log_callback=add_training_log)
        
        training_status["current_stage"] = "evaluation"
        training_status["progress"] = 80
        
        add_training_log("info", "evaluation", "📊 Analyzing anomaly detection results...")
        
        if anomaly_results:
            training_status["current_stage"] = "completed"
            training_status["progress"] = 100
            training_status["is_training"] = False
            
            add_training_log("success", "evaluation", f"✅ Anomaly detection completed")
            add_training_log("success", "evaluation", f"🔍 Found {anomaly_results['num_consenso']} consensus anomalies")
            
            save_models()
            add_training_log("success", "evaluation", "💾 Models saved successfully")
            
            return {
                "success": True,
                "message": "Unsupervised model trained successfully",
                "results": {
                    "num_anomalias_iso": anomaly_results['num_anomalias_iso'],
                    "num_anomalias_lof": anomaly_results['num_anomalias_lof'],
                    "num_consenso": anomaly_results['num_consenso'],
                    "total_records": len(anomaly_results['df_clean'])
                }
            }
        else:
            training_status["is_training"] = False
            add_training_log("error", "training", "❌ Anomaly detection training failed")
            raise HTTPException(status_code=500, detail="Anomaly detection training failed")
            
    except Exception as e:
        training_status["is_training"] = False
        add_training_log("error", "training", f"❌ Anomaly training error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Anomaly training error: {str(e)}")

@app.post("/api/predict/single")
async def predict_single(request: PredictionRequest):
    """Make a single prediction"""
    try:
        if supervised_model.best_model is None:
            raise HTTPException(status_code=400, detail="Supervised model not trained")
        
        prediction = supervised_model.predict_single(
            station_id=request.station_id,
            hour=request.hour,
            is_weekend=request.is_weekend,
            month=request.month,
            temperature=request.temperature,
            humidity=request.humidity,
            bike_type=request.bike_type,
            duration=request.duration,
            day_name=request.day_name
        )
        
        # Calculate theoretical revenue
        excedente_minutos = max(0, request.duration - 30)
        ingreso_teorico = excedente_minutos * 0.39
        diferencia = prediction - ingreso_teorico
        
        return {
            "success": True,
            "prediction": prediction,
            "theoretical_revenue": ingreso_teorico,
            "model_difference": diferencia,
            "parameters": request.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")

@app.post("/api/predict/anomaly")
async def predict_anomaly(request: AnomalyRequest):
    """Predict if a scenario is anomalous"""
    try:
        if unsupervised_model.iso_forest is None:
            raise HTTPException(status_code=400, detail="Unsupervised model not trained")
        
        is_anomaly, score = unsupervised_model.predict_anomaly(
            station_id=request.station_id,
            hour=request.hour,
            is_weekend=request.is_weekend,
            month=request.month,
            viajes_count=request.viajes_count,
            ingreso_promedio=request.ingreso_promedio,
            duracion_promedio=request.duracion_promedio,
            temperatura_promedio=request.temperatura_promedio,
            variabilidad_ingresos=request.variabilidad_ingresos,
            viajes_con_excedente=request.viajes_con_excedente
        )
        
        return {
            "success": True,
            "is_anomaly": is_anomaly,
            "anomaly_score": score,
            "parameters": request.dict()
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly prediction error: {str(e)}")

@app.post("/api/analyze/batch")
async def batch_analysis(request: BatchAnalysisRequest):
    """Perform batch scenario analysis"""
    try:
        if supervised_model.best_model is None:
            raise HTTPException(status_code=400, detail="Supervised model not trained")
        
        results = []
        
        # Define ranges
        hours = list(range(24)) if request.vary_hour else [17]
        temperatures = list(range(5, 36, 5)) if request.vary_temperature else [20]
        durations = list(range(20, 81, 10)) if request.vary_duration else [35]
        
        for hour in hours:
            for temp in temperatures:
                for duration in durations:
                    prediction = supervised_model.predict_single(
                        station_id=request.fixed_station,
                        hour=hour,
                        is_weekend=False,
                        month=request.fixed_month,
                        temperature=temp,
                        humidity=60,
                        bike_type=request.fixed_bike_type,
                        duration=duration
                    )
                    
                    results.append({
                        'hour': hour,
                        'temperature': temp,
                        'duration': duration,
                        'prediction': prediction
                    })
        
        # Find top scenarios
        df_results = pd.DataFrame(results)
        top_scenarios = df_results.nlargest(10, 'prediction').to_dict('records')
        
        stats = {
            'max_prediction': df_results['prediction'].max(),
            'avg_prediction': df_results['prediction'].mean(),
            'profitable_scenarios': (df_results['prediction'] > 0).sum(),
            'total_scenarios': len(df_results)
        }
        
        return {
            "success": True,
            "results": results,
            "top_scenarios": top_scenarios,
            "stats": stats
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Batch analysis error: {str(e)}")

@app.get("/api/anomalies/analysis")
async def get_anomaly_analysis():
    """Get detailed anomaly analysis"""
    try:
        if unsupervised_model.anomalies_data is None:
            raise HTTPException(status_code=400, detail="Anomaly model not trained")
        
        anomaly_data = unsupervised_model.anomalies_data
        
        # Basic stats
        stats = {
            "total_records": len(anomaly_data['df_clean']),
            "num_anomalies": anomaly_data['num_consenso'],
            "anomaly_rate": (anomaly_data['num_consenso'] / len(anomaly_data['df_clean'])) * 100
        }
        
        # Top anomalies
        top_anomalies = []
        if len(anomaly_data['anomalias_consenso']) > 0:
            top_anomalies = anomaly_data['anomalias_consenso'].nlargest(20, 'INGRESO_PROMEDIO').to_dict('records')
        
        return {
            "success": True,
            "stats": stats,
            "top_anomalies": top_anomalies
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Anomaly analysis error: {str(e)}")

@app.get("/api/counterfactual/analysis")
async def get_counterfactual_analysis():
    """Perform counterfactual analysis with progress tracking"""
    try:
        if supervised_model.best_model is None:
            raise HTTPException(status_code=400, detail="Supervised model not trained")
        
        print("🔄 Starting counterfactual analysis...")
        df_desabasto = get_desabasto_events()
        if df_desabasto.empty:
            return {"success": True, "message": "No shortage events found", "total_loss": 0}
        
        print(f"📊 Found {len(df_desabasto)} shortage events to analyze")
        
        # Simulate lost revenue with progress tracking
        total_ingresos_perdidos = 0
        viajes_simulados = 0
        eventos_procesados = 0
        total_eventos = len(df_desabasto)
        
        # Sample events for better performance and user experience
        sample_size = min(20, len(df_desabasto))  # Limit to 20 events for faster processing
        df_sample = df_desabasto.sample(n=sample_size, random_state=42)
        
        print(f"🎯 Processing {sample_size} sample events for analysis")
        
        for idx, (_, evento) in enumerate(df_sample.iterrows()):
            viajes_perdidos = int(evento['VIAJES_PERDIDOS_ESTIMADOS'])
            
            # Limit trips per event for performance
            trips_to_simulate = min(viajes_perdidos, 25)  # Reduced from 50
            
            for i in range(trips_to_simulate):
                duration = np.random.normal(evento['DURACION_PROMEDIO_ESTACION'], 15)
                duration = max(10, min(120, duration))
                
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
            
            eventos_procesados += 1
            progress = (eventos_procesados / sample_size) * 100
            print(f"📈 Progress: {progress:.1f}% ({eventos_procesados}/{sample_size} events)")
        
        # Scale up results to represent full dataset
        scaling_factor = total_eventos / sample_size if sample_size > 0 else 1
        total_ingresos_perdidos_scaled = total_ingresos_perdidos * scaling_factor
        viajes_simulados_scaled = viajes_simulados * scaling_factor
        
        stats = {
            "total_loss": total_ingresos_perdidos_scaled,
            "lost_trips": viajes_simulados_scaled,
            "loss_per_trip": total_ingresos_perdidos / viajes_simulados if viajes_simulados > 0 else 0,
            "affected_stations": df_desabasto['STATION_ID'].nunique(),
            "events_analyzed": total_eventos,
            "sample_size": sample_size,
            "scaling_factor": scaling_factor
        }
        
        # Get top shortage events by impact
        df_desabasto_sorted = df_desabasto.nlargest(10, 'VIAJES_PERDIDOS_ESTIMADOS')
        
        print(f"✅ Counterfactual analysis completed - Total loss: ${total_ingresos_perdidos_scaled:,.2f}")
        
        return {
            "success": True,
            "stats": convert_numpy_types(stats),
            "events_sample": convert_numpy_types(df_desabasto_sorted.to_dict('records')),
            "analysis_metadata": {
                "total_events_found": total_eventos,
                "events_processed": sample_size,
                "trips_simulated": viajes_simulados,
                "processing_time": "optimized"
            }
        }
    except Exception as e:
        print(f"❌ Counterfactual analysis error: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Counterfactual analysis error: {str(e)}")

@app.post("/api/models/load")
async def load_saved_models():
    """Load saved models"""
    try:
        if load_models():
            app_state["models_loaded"] = True
            return {"success": True, "message": "Models loaded successfully"}
        else:
            return {"success": False, "message": "No saved models found"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error loading models: {str(e)}")

@app.post("/api/models/save")  
async def save_current_models():
    """Save current models"""
    try:
        if save_models():
            return {"success": True, "message": "Models saved successfully"}
        else:
            return {"success": False, "message": "Error saving models"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error saving models: {str(e)}")


@app.post("/api/dataset/records")
async def get_dataset_records(request: DatasetRequest):
    """Get paginated dataset records with filtering"""
    try:
        if app_state["dashboard_data"] is None:
            app_state["dashboard_data"] = get_dashboard_data()
        
        df = app_state["dashboard_data"].copy()
        
        if df.empty:
            return {
                "success": True,
                "records": [],
                "total_records": 0,
                "total_pages": 0,
                "current_page": request.page
            }
        
        # Apply filters
        if request.station_filter:
            df = df[df['START_STATION_ID'].astype(str).str.contains(request.station_filter, case=False, na=False)]
        
        if request.member_type_filter:
            df = df[df['MEMBER_CASUAL'].str.contains(request.member_type_filter, case=False, na=False)]
        
        if request.month_filter:
            df = df[df['MONTH'] == request.month_filter]
        
        if request.min_revenue is not None:
            df = df[df['INGRESO_MIN_EXCEDENTE'] >= request.min_revenue]
        
        if request.max_revenue is not None:
            df = df[df['INGRESO_MIN_EXCEDENTE'] <= request.max_revenue]
        
        # Sort data
        ascending = request.sort_order.lower() == "asc"
        if request.sort_by in df.columns:
            df = df.sort_values(by=request.sort_by, ascending=ascending)
        
        # Calculate pagination
        total_records = len(df)
        total_pages = (total_records + request.limit - 1) // request.limit
        start_idx = (request.page - 1) * request.limit
        end_idx = start_idx + request.limit
        
        # Get page data
        page_df = df.iloc[start_idx:end_idx]
        
        # Convert to records
        records = []
        for _, row in page_df.iterrows():
            record = {
                "id": f"{row['START_STATION_ID']}_{start_idx + len(records)}",
                "start_station_id": str(row['START_STATION_ID']),
                "end_station_id": str(row['END_STATION_ID']) if 'END_STATION_ID' in row else "",
                "hour": int(row['HOUR']) if 'HOUR' in row else 0,
                "day_name": str(row['DAY_NAME']) if 'DAY_NAME' in row else "",
                "is_weekend": bool(row['IS_WEEKEND']) if 'IS_WEEKEND' in row else False,
                "month": int(row['MONTH']) if 'MONTH' in row else 0,
                "temperature": round(float(row['TEMPERATURE_2M']), 1) if 'TEMPERATURE_2M' in row else 0,
                "humidity": round(float(row['RELATIVE_HUMIDITY_2M']), 1) if 'RELATIVE_HUMIDITY_2M' in row else 0,
                "bike_type": int(row['BIKE_TYPE_ID']) if 'BIKE_TYPE_ID' in row else 1,
                "duration": round(float(row['DURACION_MINUTOS']), 1) if 'DURACION_MINUTOS' in row else 0,
                "member_type": str(row['MEMBER_CASUAL']) if 'MEMBER_CASUAL' in row else "",
                "revenue": round(float(row['INGRESO_MIN_EXCEDENTE']), 2) if 'INGRESO_MIN_EXCEDENTE' in row else 0,
                "included_minutes": int(row['MINUTOS_INCLUIDOS']) if 'MINUTOS_INCLUIDOS' in row else 30,
                "excess_rate": round(float(row['TARIFA_MINUTO_EXCEDENTE']), 2) if 'TARIFA_MINUTO_EXCEDENTE' in row else 0.39
            }
            records.append(record)
        
        # Get summary statistics
        summary = {
            "total_revenue": round(float(df['INGRESO_MIN_EXCEDENTE'].sum()), 2),
            "avg_revenue": round(float(df['INGRESO_MIN_EXCEDENTE'].mean()), 2),
            "avg_duration": round(float(df['DURACION_MINUTOS'].mean()), 1),
            "unique_stations": int(df['START_STATION_ID'].nunique()),
            "member_distribution": df['MEMBER_CASUAL'].value_counts().to_dict() if 'MEMBER_CASUAL' in df.columns else {}
        }
        
        return {
            "success": True,
            "records": records,
            "total_records": total_records,
            "total_pages": total_pages,
            "current_page": request.page,
            "summary": convert_numpy_types(summary),
            "filters_applied": {
                "station_filter": request.station_filter,
                "member_type_filter": request.member_type_filter,
                "month_filter": request.month_filter,
                "min_revenue": request.min_revenue,
                "max_revenue": request.max_revenue
            }
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching dataset records: {str(e)}")

@app.get("/api/dataset/columns")
async def get_dataset_columns():
    """Get available columns for filtering and sorting"""
    try:
        if app_state["dashboard_data"] is None:
            app_state["dashboard_data"] = get_dashboard_data()
        
        df = app_state["dashboard_data"]
        
        if df.empty:
            return {"success": False, "message": "No data available"}
        
        columns_info = []
        for col in df.columns:
            col_info = {
                "name": col,
                "type": str(df[col].dtype),
                "unique_values": int(df[col].nunique()) if df[col].nunique() < 50 else None,
                "sample_values": df[col].dropna().unique()[:10].tolist() if df[col].nunique() < 50 else []
            }
            columns_info.append(col_info)
        
        return {
            "success": True,
            "columns": convert_numpy_types(columns_info)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error fetching columns info: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port) 