import snowflake.connector
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Optional

# Cargar variables de entorno desde archivo .env
# Decisión: Se utiliza python-dotenv para separar la configuración sensible del código
load_dotenv()

class SnowflakeConnection:
    """
    Clase para manejar la conexión a Snowflake de manera centralizada.
    
    Propósito: Encapsular toda la lógica de conexión y consultas a la base de datos
    para facilitar el mantenimiento y reutilización del código.
    
    Decisión de diseño: Se utiliza el patrón Singleton implícito para asegurar
    una sola instancia de conexión por aplicación.
    """
    
    def __init__(self):
        # Cargar credenciales desde variables de entorno por seguridad
        # Esto evita hardcodear credenciales en el código fuente
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.database = os.getenv("SNOWFLAKE_DATABASE")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA")
        
        # Validación temprana de credenciales para fallar rápido si hay problemas
        if not all([self.account, self.user, self.password, self.database, self.schema]):
            raise ValueError("Faltan variables de entorno requeridas para Snowflake. Revisa tu archivo .env.")
    
    def ejecutar_consulta_df(self, sql_query: str) -> pd.DataFrame:
        """
        Ejecuta una consulta SQL y retorna un DataFrame de pandas.
        
        Propósito: Centralizar la ejecución de consultas con manejo robusto de errores
        y limpieza automática de recursos.
        
        Decisión: Se utiliza pandas DataFrame como formato de retorno estándar
        porque facilita el análisis de datos y es compatible con sklearn.
        """
        conn = None
        cursor = None
        try:
            print(f"Conectando a Snowflake...")
            # Configuración de conexión optimizada para estabilidad
            conn = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                database=self.database,
                schema=self.schema,
                insecure_mode=False,  # Mantener SSL habilitado por seguridad
                ocsp_fail_open=True   # Permitir conexión si OCSP falla (común en algunos entornos)
            )
            print(f"Conexión a Snowflake exitosa")
            
            cursor = conn.cursor()
            print(f"Ejecutando consulta...")
            cursor.execute(sql_query)
            
            print(f"Obteniendo resultados...")
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            
            print(f"Consulta ejecutada exitosamente. Se obtuvieron {len(df)} filas")
            return df
            
        except snowflake.connector.errors.ProgrammingError as pe:
            # Error específico de programación SQL
            print(f"Error de programación Snowflake: {str(pe)}")
            print(f"Consulta SQL: {sql_query[:200]}...")
            return pd.DataFrame()
        except snowflake.connector.errors.DatabaseError as de:
            # Error de base de datos
            print(f"Error de base de datos Snowflake: {str(de)}")
            return pd.DataFrame()
        except snowflake.connector.errors.Error as se:
            # Otros errores de Snowflake
            print(f"Error de Snowflake: {str(se)}")
            return pd.DataFrame()
        except Exception as e:
            # Manejo genérico de errores no esperados
            print(f"Error inesperado: {str(e)}")
            print(f"Tipo de error: {type(e).__name__}")
            import traceback
            print(f"Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
        finally:
            # Limpieza garantizada de recursos (patrón try-finally)
            # Esto previene memory leaks y conexiones colgadas
            if cursor:
                try:
                    cursor.close()
                    print("Cursor cerrado")
                except Exception as e:
                    print(f"Error cerrando cursor: {str(e)}")
            
            if conn:
                try:
                    conn.close()
                    print("Conexión cerrada")
                except Exception as e:
                    print(f"Error cerrando conexión: {str(e)}")

# Instancia global para reutilización en toda la aplicación
# Decisión: Singleton pattern para evitar múltiples conexiones innecesarias
try:
    db = SnowflakeConnection()
    print("Conexión Snowflake inicializada exitosamente")
except ValueError as e:
    print(f"Falló la inicialización de conexión Snowflake: {e}")
    db = None

def get_training_data():
    """
    Obtiene datos de entrenamiento para modelos supervisados.
    
    Propósito: Generar un dataset limpio para entrenar modelos de predicción de ingresos,
    excluyendo momentos de desabasto que podrían sesgar el modelo.
    
    Decisión de diseño: Se usa una CTE (Common Table Expression) para identificar
    eventos de desabasto y luego se excluyen con NOT EXISTS para mejor performance.
    """
    if db is None:
        print("Conexión de base de datos no disponible")
        return pd.DataFrame()
        
    consulta = """
    WITH momentos_desabasto AS (
        -- Identificar momentos críticos de desabasto para excluirlos del entrenamiento
        -- Esto previene que el modelo aprenda patrones de escasez como normales
        SELECT DISTINCT
            STATION_ID,
            TIME_ID
        FROM DIM_EVENTOBALANCE
        WHERE TIPO_EVENTO_BALANCE = 'DESABASTO'
        AND FLAG_EVENTO_CRITICO = TRUE
    )
    SELECT 
        t.START_STATION_ID AS STATION_ID,
        tiempo.TIME_ID,
        tiempo.HOUR,
        tiempo.DAY_NAME,
        tiempo.IS_WEEKEND,
        tiempo.MONTH,
        clima.TEMPERATURE_2M,
        clima.APPARENT_TEMPERATURE,
        clima.RELATIVE_HUMIDITY_2M,
        t.BIKE_TYPE_ID,
        t.DURACION_MINUTOS,
        tarifa.MINUTOS_INCLUIDOS,
        tarifa.TARIFA_MINUTO_EXCEDENTE,
        
        -- Cálculo de ingresos por minutos excedentes (variable objetivo)
        -- Solo se cobra extra si excede los minutos incluidos
        CASE
            WHEN t.DURACION_MINUTOS > tarifa.MINUTOS_INCLUIDOS THEN 
                (t.DURACION_MINUTOS - tarifa.MINUTOS_INCLUIDOS) * tarifa.TARIFA_MINUTO_EXCEDENTE
            ELSE 0
        END AS INGRESO_MIN_EXCEDENTE
        
    FROM FACT_CITIBIKE_WEATHER t
    JOIN DIM_TIME tiempo ON t.START_TIME_ID = tiempo.TIME_ID
    JOIN FACT_TEMPERATURE_ATMOSPHERE clima ON t.START_TIME_ID = clima.TIME_ID
    JOIN DIM_MEMBER_TYPE m ON t.MEMBER_TYPE_ID = m.MEMBER_TYPE_ID
    JOIN DIM_TARIFA tarifa ON t.MEMBER_TYPE_ID = tarifa.MEMBER_TYPE_ID 
                           AND t.BIKE_TYPE_ID = tarifa.BIKE_TYPE_ID
    WHERE m.MEMBER_CASUAL = 'casual'  -- Solo usuarios casuales (mayor variabilidad en duración)
    AND tiempo.YEAR = 2024
    AND NOT EXISTS (
        -- Excluir momentos de desabasto para datos de entrenamiento más limpios
        SELECT 1 FROM momentos_desabasto md
        WHERE md.STATION_ID = t.START_STATION_ID
        AND md.TIME_ID = t.START_TIME_ID
    )
    """
    return db.ejecutar_consulta_df(consulta)

def get_anomaly_data():
    """
    Obtiene datos agregados para detección de anomalías.
    
    Propósito: Crear features agregadas por estación-hora que permitan identificar
    patrones anómalos en el comportamiento de ingresos y uso.
    
    Decisión: Se agregan los datos para reducir dimensionalidad y crear features
    más estables para algoritmos de detección de anomalías.
    """
    if db is None:
        print("Conexión de base de datos no disponible")
        return pd.DataFrame()
        
    consulta = """
    WITH datos_base AS (
        -- Datos base con cálculos de ingresos para análisis de anomalías
        SELECT 
            t.START_STATION_ID AS STATION_ID,
            tiempo.HOUR,
            tiempo.DAY_NAME,
            tiempo.IS_WEEKEND,
            tiempo.MONTH,
            CASE
                WHEN t.DURACION_MINUTOS > tarifa.MINUTOS_INCLUIDOS THEN 
                    (t.DURACION_MINUTOS - tarifa.MINUTOS_INCLUIDOS) * tarifa.TARIFA_MINUTO_EXCEDENTE
                ELSE 0
            END AS INGRESO_MIN_EXCEDENTE,
            clima.TEMPERATURE_2M,
            t.DURACION_MINUTOS
        FROM FACT_CITIBIKE_WEATHER t
        JOIN DIM_TIME tiempo ON t.START_TIME_ID = tiempo.TIME_ID
        JOIN FACT_TEMPERATURE_ATMOSPHERE clima ON t.START_TIME_ID = clima.TIME_ID
        JOIN DIM_MEMBER_TYPE m ON t.MEMBER_TYPE_ID = m.MEMBER_TYPE_ID
        JOIN DIM_TARIFA tarifa ON t.MEMBER_TYPE_ID = tarifa.MEMBER_TYPE_ID 
                               AND t.BIKE_TYPE_ID = tarifa.BIKE_TYPE_ID
        WHERE m.MEMBER_CASUAL = 'casual'
        AND tiempo.YEAR = 2024
    )
    SELECT 
        STATION_ID,
        HOUR,
        IS_WEEKEND,
        MONTH,
        -- Métricas agregadas para detección de anomalías
        COUNT(*) as viajes_count,                           -- Volumen de actividad
        AVG(INGRESO_MIN_EXCEDENTE) as ingreso_promedio,     -- Ingreso promedio
        SUM(INGRESO_MIN_EXCEDENTE) as ingreso_total,        -- Ingreso total
        AVG(DURACION_MINUTOS) as duracion_promedio,         -- Duración promedio
        AVG(TEMPERATURE_2M) as temperatura_promedio,        -- Temperatura promedio
        STDDEV(INGRESO_MIN_EXCEDENTE) as variabilidad_ingresos,  -- Variabilidad de ingresos
        MAX(INGRESO_MIN_EXCEDENTE) as ingreso_maximo,       -- Ingreso máximo observado
        COUNT(CASE WHEN INGRESO_MIN_EXCEDENTE > 0 THEN 1 END) as viajes_con_excedente  -- Viajes rentables
    FROM datos_base
    GROUP BY STATION_ID, HOUR, IS_WEEKEND, MONTH
    HAVING COUNT(*) >= 5  -- Filtrar grupos con muy pocos registros para estabilidad
    ORDER BY STATION_ID, HOUR
    """
    return db.ejecutar_consulta_df(consulta)

def get_desabasto_events():
    """
    Obtiene eventos de desabasto para análisis contrafactual.
    
    Propósito: Identificar momentos donde hubo escasez de bicicletas y estimar
    los ingresos perdidos usando datos históricos de las estaciones.
    
    Decisión: Se combinan datos de eventos con estadísticas históricas de cada
    estación para hacer estimaciones más precisas de pérdidas.
    """
    if db is None:
        print("Conexión de base de datos no disponible")
        return pd.DataFrame()
        
    consulta = """
    WITH momentos_desabasto AS (
        -- Eventos críticos de desabasto con contexto temporal y climático
        SELECT DISTINCT
            eb.STATION_ID,
            eb.TIME_ID,
            eb.BALANCE_NET,  -- Balance negativo indica desabasto
            tiempo.HOUR,
            tiempo.DAY_NAME,
            tiempo.IS_WEEKEND,
            tiempo.MONTH,
            clima.TEMPERATURE_2M,
            clima.APPARENT_TEMPERATURE,
            clima.RELATIVE_HUMIDITY_2M
        FROM DIM_EVENTOBALANCE eb
        JOIN DIM_TIME tiempo ON eb.TIME_ID = tiempo.TIME_ID
        JOIN FACT_TEMPERATURE_ATMOSPHERE clima ON eb.TIME_ID = clima.TIME_ID
        WHERE eb.TIPO_EVENTO_BALANCE = 'DESABASTO'
        AND eb.FLAG_EVENTO_CRITICO = TRUE
        AND tiempo.YEAR = 2024
    ),
    stats_estacion AS (
        -- Estadísticas históricas de cada estación para estimaciones más precisas
        SELECT 
            t.START_STATION_ID AS STATION_ID,
            AVG(t.DURACION_MINUTOS) as duracion_promedio_estacion,
            COUNT(*) as viajes_totales,
            AVG(CASE WHEN t.BIKE_TYPE_ID = 1 THEN 1.0 ELSE 0.0 END) as prop_bike_type_1
        FROM FACT_CITIBIKE_WEATHER t
        JOIN DIM_TIME tiempo ON t.START_TIME_ID = tiempo.TIME_ID
        JOIN DIM_MEMBER_TYPE m ON t.MEMBER_TYPE_ID = m.MEMBER_TYPE_ID
        WHERE m.MEMBER_CASUAL = 'casual'
        AND tiempo.YEAR = 2024
        GROUP BY t.START_STATION_ID
    )
    SELECT 
        md.*,
        -- Usar estadísticas históricas o valores por defecto si no hay datos
        COALESCE(se.duracion_promedio_estacion, 35) as duracion_promedio_estacion,
        COALESCE(se.prop_bike_type_1, 0.7) as prop_bike_type_1,
        30 as MINUTOS_INCLUIDOS,
        -- Estimación de viajes perdidos basada en el balance negativo
        ROUND(ABS(md.BALANCE_NET) * 0.6) as viajes_perdidos_estimados
    FROM momentos_desabasto md
    LEFT JOIN stats_estacion se ON se.STATION_ID = md.STATION_ID
    """
    return db.ejecutar_consulta_df(consulta)

def get_dashboard_data():
    """
    Obtiene datos principales para el dashboard.
    
    Propósito: Consulta unificada que proporciona todos los datos necesarios
    para las visualizaciones y análisis del dashboard principal.
    
    Decisión: Una sola consulta comprehensiva para minimizar llamadas a la DB
    y asegurar consistencia en los datos mostrados.
    """
    if db is None:
        print("Conexión de base de datos no disponible")
        return pd.DataFrame()
        
    consulta = """
    SELECT 
        t.START_STATION_ID,
        t.END_STATION_ID,
        tiempo.HOUR,
        tiempo.DAY_NAME,
        tiempo.IS_WEEKEND,
        tiempo.MONTH,
        clima.TEMPERATURE_2M,
        clima.RELATIVE_HUMIDITY_2M,
        t.BIKE_TYPE_ID,
        t.DURACION_MINUTOS,
        m.MEMBER_CASUAL,
        -- Cálculo estándar de ingresos por excedente
        CASE
            WHEN t.DURACION_MINUTOS > tarifa.MINUTOS_INCLUIDOS THEN 
                (t.DURACION_MINUTOS - tarifa.MINUTOS_INCLUIDOS) * tarifa.TARIFA_MINUTO_EXCEDENTE
            ELSE 0
        END AS INGRESO_MIN_EXCEDENTE,
        tarifa.MINUTOS_INCLUIDOS,
        tarifa.TARIFA_MINUTO_EXCEDENTE
    FROM FACT_CITIBIKE_WEATHER t
    JOIN DIM_TIME tiempo ON t.START_TIME_ID = tiempo.TIME_ID
    JOIN FACT_TEMPERATURE_ATMOSPHERE clima ON t.START_TIME_ID = clima.TIME_ID
    JOIN DIM_MEMBER_TYPE m ON t.MEMBER_TYPE_ID = m.MEMBER_TYPE_ID
    JOIN DIM_TARIFA tarifa ON t.MEMBER_TYPE_ID = tarifa.MEMBER_TYPE_ID 
                           AND t.BIKE_TYPE_ID = tarifa.BIKE_TYPE_ID
    WHERE tiempo.YEAR = 2024  -- Enfoque en datos del año actual
    """
    return db.ejecutar_consulta_df(consulta) 