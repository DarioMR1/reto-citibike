import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor, IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
import joblib
import os
from typing import Tuple, Dict, Any, Callable, Optional
import warnings
warnings.filterwarnings('ignore')

class SupervisedModel:
    def __init__(self):
        self.best_model = None
        self.le_station = None
        self.features_cols = None
        self.model_type = None
        self.performance_metrics = {}
        
    def prepare_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """Prepara las features para el modelo supervisado"""
        df_modelo = df.copy()
        
        # 1. ENCODING DE VARIABLES CATEGÓRICAS
        if self.le_station is None:
            self.le_station = LabelEncoder()
            df_modelo['STATION_ID_ENCODED'] = self.le_station.fit_transform(df_modelo['STATION_ID'])
        else:
            # Para predicción, usar el encoder ya entrenado
            try:
                df_modelo['STATION_ID_ENCODED'] = self.le_station.transform(df_modelo['STATION_ID'])
            except ValueError:
                # Si hay estaciones nuevas, asignar el valor más común
                df_modelo['STATION_ID_ENCODED'] = 0
        
        # One-hot encoding para DAY_NAME
        df_day_dummies = pd.get_dummies(df_modelo['DAY_NAME'], prefix='DAY', drop_first=True)
        
        # 2. FEATURES TEMPORALES
        df_modelo['HOUR_SIN'] = np.sin(2 * np.pi * df_modelo['HOUR'] / 24)
        df_modelo['HOUR_COS'] = np.cos(2 * np.pi * df_modelo['HOUR'] / 24)
        df_modelo['IS_RUSH_HOUR'] = ((df_modelo['HOUR'] >= 7) & (df_modelo['HOUR'] <= 9)) | ((df_modelo['HOUR'] >= 17) & (df_modelo['HOUR'] <= 19))
        df_modelo['IS_NIGHT'] = (df_modelo['HOUR'] >= 22) | (df_modelo['HOUR'] <= 5)
        
        # 3. FEATURES CLIMÁTICAS
        df_modelo['TEMP_COMFORT'] = (df_modelo['TEMPERATURE_2M'] >= 15) & (df_modelo['TEMPERATURE_2M'] <= 25)
        df_modelo['HUMIDITY_HIGH'] = df_modelo['RELATIVE_HUMIDITY_2M'] > 70
        df_modelo['TEMP_APPARENT_DIFF'] = df_modelo['APPARENT_TEMPERATURE'] - df_modelo['TEMPERATURE_2M']
        
        # 4. FEATURES DE DURACIÓN
        df_modelo['DURACION_LARGA'] = df_modelo['DURACION_MINUTOS'] > 45
        df_modelo['DURACION_NORMALIZADA'] = df_modelo['DURACION_MINUTOS'] / df_modelo['MINUTOS_INCLUIDOS']
        
        # 5. COMBINAR FEATURES
        features_numericas = [
            'STATION_ID_ENCODED', 'HOUR', 'IS_WEEKEND', 'MONTH',
            'TEMPERATURE_2M', 'APPARENT_TEMPERATURE', 'RELATIVE_HUMIDITY_2M',
            'BIKE_TYPE_ID', 'DURACION_MINUTOS', 'MINUTOS_INCLUIDOS',
            'HOUR_SIN', 'HOUR_COS', 'IS_RUSH_HOUR', 'IS_NIGHT',
            'TEMP_COMFORT', 'HUMIDITY_HIGH', 'TEMP_APPARENT_DIFF',
            'DURACION_LARGA', 'DURACION_NORMALIZADA'
        ]
        
        # Combinar features numéricas + dummies de día
        df_modelo_final = pd.concat([
            df_modelo[features_numericas], 
            df_day_dummies
        ], axis=1)
        
        # Convertir booleanos a enteros
        bool_columns = df_modelo_final.select_dtypes(include=['bool']).columns
        df_modelo_final[bool_columns] = df_modelo_final[bool_columns].astype(int)
        
        # Asegurarse de que todas las columnas estén presentes
        if self.features_cols is None:
            self.features_cols = df_modelo_final.columns.tolist()
        else:
            # Reordenar columnas para que coincidan con el entrenamiento
            df_modelo_final = df_modelo_final.reindex(columns=self.features_cols, fill_value=0)
        
        return df_modelo_final
    
    def train(self, df: pd.DataFrame, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Entrena los modelos supervisados con optimizaciones de velocidad"""
        if log_callback:
            log_callback("info", "preprocessing", "🔧 Starting feature engineering...")
        
        # OPTIMIZACIÓN 1: Usar una muestra más pequeña para entrenamiento rápido
        if len(df) > 50000:
            df_sample = df.sample(n=50000, random_state=42)
            if log_callback:
                log_callback("info", "preprocessing", f"📊 Using sample of {len(df_sample)} records for faster training")
        else:
            df_sample = df
        
        # Preparar features
        df_modelo = self.prepare_features(df_sample)
        
        if log_callback:
            log_callback("success", "preprocessing", "✅ Feature engineering completed")
        
        # Preparar X e y
        if 'INGRESO_MIN_EXCEDENTE' in df_sample.columns:
            X = df_modelo
            y = df_sample['INGRESO_MIN_EXCEDENTE']
        else:
            if log_callback:
                log_callback("error", "preprocessing", "❌ Target column INGRESO_MIN_EXCEDENTE not found")
            return {}
        
        # Split
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=(y > 0)
        )
        
        if log_callback:
            log_callback("info", "training", f"📊 Training set: {len(X_train)} samples")
            log_callback("info", "training", f"📊 Test set: {len(X_test)} samples")
        
        results = {}
        
        # OPTIMIZACIÓN 2: Random Forest más pequeño y rápido
        if log_callback:
            log_callback("info", "training", "🌲 Training optimized Random Forest...")
        
        rf_model = RandomForestRegressor(
            n_estimators=50,  # Reducido de 100
            max_depth=10,     # Reducido de 15
            min_samples_split=20,  # Aumentado para menos splits
            min_samples_leaf=10,   # Aumentado para menos hojas
            random_state=42,
            n_jobs=-1,
            max_features='sqrt'  # Usar menos features por árbol
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluar RF
        y_pred_rf = rf_model.predict(X_test)
        rf_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_rf)),
            'mae': mean_absolute_error(y_test, y_pred_rf),
            'r2': r2_score(y_test, y_pred_rf)
        }
        results['RandomForest'] = rf_metrics
        
        if log_callback:
            log_callback("success", "training", f"✅ Random Forest completed - R²: {rf_metrics['r2']:.4f}")
        
        # OPTIMIZACIÓN 3: Gradient Boosting más pequeño y rápido
        if log_callback:
            log_callback("info", "training", "🚀 Training optimized Gradient Boosting...")
        
        gb_model = GradientBoostingRegressor(
            n_estimators=50,   # Reducido de 150
            max_depth=5,       # Reducido de 8
            learning_rate=0.15, # Aumentado para convergencia más rápida
            subsample=0.8,
            random_state=42,
            max_features='sqrt'  # Usar menos features
        )
        gb_model.fit(X_train, y_train)
        
        # Evaluar GB
        y_pred_gb = gb_model.predict(X_test)
        gb_metrics = {
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred_gb)),
            'mae': mean_absolute_error(y_test, y_pred_gb),
            'r2': r2_score(y_test, y_pred_gb)
        }
        results['GradientBoosting'] = gb_metrics
        
        if log_callback:
            log_callback("success", "training", f"✅ Gradient Boosting completed - R²: {gb_metrics['r2']:.4f}")
        
        # Seleccionar mejor modelo
        if rf_metrics['r2'] > gb_metrics['r2']:
            self.best_model = rf_model
            self.model_type = "Random Forest"
            self.performance_metrics = rf_metrics
            if log_callback:
                log_callback("success", "training", f"🏆 Best model: Random Forest (R² = {rf_metrics['r2']:.4f})")
        else:
            self.best_model = gb_model
            self.model_type = "Gradient Boosting"
            self.performance_metrics = gb_metrics
            if log_callback:
                log_callback("success", "training", f"🏆 Best model: Gradient Boosting (R² = {gb_metrics['r2']:.4f})")
        
        return results
    
    def predict_single(self, station_id: str, hour: int, is_weekend: bool, month: int,
                      temperature: float, humidity: float, bike_type: int, 
                      duration: int, day_name: str = "Monday") -> float:
        """Realiza una predicción individual"""
        if self.best_model is None:
            return 0.0
        
        # Crear datos de entrada
        data = {
            'STATION_ID': [station_id],
            'HOUR': [hour],
            'DAY_NAME': [day_name],
            'IS_WEEKEND': [is_weekend],
            'MONTH': [month],
            'TEMPERATURE_2M': [temperature],
            'APPARENT_TEMPERATURE': [temperature + 2],  # Aproximación
            'RELATIVE_HUMIDITY_2M': [humidity],
            'BIKE_TYPE_ID': [bike_type],
            'DURACION_MINUTOS': [duration],
            'MINUTOS_INCLUIDOS': [30],
            'TARIFA_MINUTO_EXCEDENTE': [0.39]
        }
        
        df_input = pd.DataFrame(data)
        X_pred = self.prepare_features(df_input)
        
        prediction = self.best_model.predict(X_pred)[0]
        return max(0, prediction)  # No permitir valores negativos

class UnsupervisedModel:
    def __init__(self):
        self.iso_forest = None
        self.lof = None
        self.scaler = None
        self.features_cols = None
        self.anomalies_data = None
        
    def prepare_anomaly_features(self, df: pd.DataFrame) -> np.ndarray:
        """Prepara features para detección de anomalías"""
        features_anomalias = [
            'HOUR', 'IS_WEEKEND', 'MONTH', 'VIAJES_COUNT',
            'INGRESO_PROMEDIO', 'DURACION_PROMEDIO', 'TEMPERATURA_PROMEDIO',
            'VARIABILIDAD_INGRESOS', 'VIAJES_CON_EXCEDENTE'
        ]
        
        # Filtrar columnas disponibles
        features_disponibles = [f for f in features_anomalias if f in df.columns]
        self.features_cols = features_disponibles
        
        df_clean = df[features_disponibles].copy()
        
        # Limpiar datos
        df_clean = df_clean.replace([np.inf, -np.inf], np.nan)
        for col in features_disponibles:
            if df_clean[col].dtype in ['float64', 'int64']:
                median_val = df_clean[col].median()
                df_clean[col] = df_clean[col].fillna(median_val)
        
        df_clean = df_clean.dropna()
        
        # Normalizar
        if self.scaler is None:
            self.scaler = StandardScaler()
            X_scaled = self.scaler.fit_transform(df_clean[features_disponibles])
        else:
            X_scaled = self.scaler.transform(df_clean[features_disponibles])
        
        return X_scaled, df_clean
    
    def train_anomaly_detection(self, df: pd.DataFrame, log_callback: Optional[Callable] = None) -> Dict[str, Any]:
        """Entrena modelos de detección de anomalías optimizados"""
        if log_callback:
            log_callback("info", "preprocessing", "🔧 Preparing anomaly features...")
        
        # OPTIMIZACIÓN: Usar muestra más pequeña para entrenamiento rápido
        if len(df) > 20000:
            df_sample = df.sample(n=20000, random_state=42)
            if log_callback:
                log_callback("info", "preprocessing", f"📊 Using sample of {len(df_sample)} records for faster training")
        else:
            df_sample = df
        
        X_scaled, df_clean = self.prepare_anomaly_features(df_sample)
        
        if log_callback:
            log_callback("success", "preprocessing", f"✅ Features prepared - {X_scaled.shape[1]} features")
        
        # OPTIMIZACIÓN: Isolation Forest más rápido
        if log_callback:
            log_callback("info", "training", "🌲 Training optimized Isolation Forest...")
        
        self.iso_forest = IsolationForest(
            contamination=0.1,
            random_state=42,
            n_jobs=-1,
            n_estimators=50,  # Reducido de 100
            max_samples=min(1000, len(df_clean))  # Limitar muestras por árbol
        )
        anomaly_scores_iso = self.iso_forest.fit_predict(X_scaled)
        anomaly_scores_iso_proba = self.iso_forest.decision_function(X_scaled)
        
        if log_callback:
            log_callback("success", "training", "✅ Isolation Forest training completed")
        
        # OPTIMIZACIÓN: LOF más rápido
        if log_callback:
            log_callback("info", "training", "🔍 Training optimized Local Outlier Factor...")
        
        self.lof = LocalOutlierFactor(
            n_neighbors=min(20, len(df_clean) // 10),  # Adaptar vecinos al tamaño
            contamination=0.1,
            n_jobs=-1
        )
        anomaly_scores_lof = self.lof.fit_predict(X_scaled)
        
        if log_callback:
            log_callback("success", "training", "✅ Local Outlier Factor training completed")
        
        # Agregar resultados al DataFrame
        df_clean = df_clean.copy()
        df_clean['anomalia_iso'] = anomaly_scores_iso
        df_clean['score_iso'] = anomaly_scores_iso_proba
        df_clean['anomalia_lof'] = anomaly_scores_lof
        
        # Anomalías consenso
        anomalias_consenso = df_clean[
            (df_clean['anomalia_iso'] == -1) & 
            (df_clean['anomalia_lof'] == -1)
        ].copy()
        
        self.anomalies_data = {
            'df_clean': df_clean,
            'anomalias_consenso': anomalias_consenso,
            'num_anomalias_iso': len(df_clean[df_clean['anomalia_iso'] == -1]),
            'num_anomalias_lof': len(df_clean[df_clean['anomalia_lof'] == -1]),
            'num_consenso': len(anomalias_consenso)
        }
        
        if log_callback:
            log_callback("success", "evaluation", f"🔍 Isolation Forest anomalies: {self.anomalies_data['num_anomalias_iso']}")
            log_callback("success", "evaluation", f"🔍 LOF anomalies: {self.anomalies_data['num_anomalias_lof']}")
            log_callback("success", "evaluation", f"🎯 Consensus anomalies: {self.anomalies_data['num_consenso']}")
        
        return self.anomalies_data
    
    def predict_anomaly(self, station_id: str, hour: int, is_weekend: bool, 
                       month: int, viajes_count: int, ingreso_promedio: float,
                       duracion_promedio: float, temperatura_promedio: float,
                       variabilidad_ingresos: float, viajes_con_excedente: int) -> Tuple[bool, float]:
        """Predice si una combinación estación-hora es anómala"""
        if self.iso_forest is None or self.scaler is None:
            return False, 0.0
        
        # Crear datos de entrada
        data = {
            'HOUR': [hour],
            'IS_WEEKEND': [is_weekend],
            'MONTH': [month],
            'VIAJES_COUNT': [viajes_count],
            'INGRESO_PROMEDIO': [ingreso_promedio],
            'DURACION_PROMEDIO': [duracion_promedio],
            'TEMPERATURA_PROMEDIO': [temperatura_promedio],
            'VARIABILIDAD_INGRESOS': [variabilidad_ingresos],
            'VIAJES_CON_EXCEDENTE': [viajes_con_excedente]
        }
        
        df_input = pd.DataFrame(data)
        
        # Normalizar
        X_scaled = self.scaler.transform(df_input[self.features_cols])
        
        # Predecir
        is_anomaly_iso = self.iso_forest.predict(X_scaled)[0] == -1
        anomaly_score = self.iso_forest.decision_function(X_scaled)[0]
        
        return is_anomaly_iso, anomaly_score

# Instancias globales
supervised_model = SupervisedModel()
unsupervised_model = UnsupervisedModel()

def save_models():
    """Guarda los modelos entrenados"""
    models_dir = "models"
    if not os.path.exists(models_dir):
        os.makedirs(models_dir)
    
    try:
        if supervised_model.best_model is not None:
            # Guardar solo los componentes necesarios del modelo supervisado
            supervised_data = {
                'best_model': supervised_model.best_model,
                'le_station': supervised_model.le_station,
                'features_cols': supervised_model.features_cols,
                'model_type': supervised_model.model_type,
                'performance_metrics': supervised_model.performance_metrics
            }
            joblib.dump(supervised_data, f"{models_dir}/supervised_model.pkl")
            print("✅ Supervised model saved successfully")
        
        if unsupervised_model.iso_forest is not None:
            # Guardar solo los componentes necesarios del modelo no supervisado
            unsupervised_data = {
                'iso_forest': unsupervised_model.iso_forest,
                'lof': unsupervised_model.lof,
                'scaler': unsupervised_model.scaler,
                'features_cols': unsupervised_model.features_cols,
                'anomalies_data': unsupervised_model.anomalies_data
            }
            joblib.dump(unsupervised_data, f"{models_dir}/unsupervised_model.pkl")
            print("✅ Unsupervised model saved successfully")
        
        return True
    except Exception as e:
        print(f"❌ Error saving models: {e}")
        return False

def load_models():
    """Carga los modelos guardados"""
    models_dir = "models"
    loaded_count = 0
    
    try:
        if os.path.exists(f"{models_dir}/supervised_model.pkl"):
            global supervised_model
            supervised_data = joblib.load(f"{models_dir}/supervised_model.pkl")
            supervised_model.best_model = supervised_data['best_model']
            supervised_model.le_station = supervised_data['le_station']
            supervised_model.features_cols = supervised_data['features_cols']
            supervised_model.model_type = supervised_data['model_type']
            supervised_model.performance_metrics = supervised_data['performance_metrics']
            print("✅ Supervised model loaded successfully")
            loaded_count += 1
        
        if os.path.exists(f"{models_dir}/unsupervised_model.pkl"):
            global unsupervised_model
            unsupervised_data = joblib.load(f"{models_dir}/unsupervised_model.pkl")
            unsupervised_model.iso_forest = unsupervised_data['iso_forest']
            unsupervised_model.lof = unsupervised_data['lof']
            unsupervised_model.scaler = unsupervised_data['scaler']
            unsupervised_model.features_cols = unsupervised_data['features_cols']
            unsupervised_model.anomalies_data = unsupervised_data['anomalies_data']
            print("✅ Unsupervised model loaded successfully")
            loaded_count += 1
        
        if loaded_count == 0:
            print("ℹ️ No saved models found")
            
        return loaded_count > 0
    except Exception as e:
        print(f"❌ Error loading models: {e}")
        return False 