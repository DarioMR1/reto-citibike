import snowflake.connector
import pandas as pd
import os
from dotenv import load_dotenv
from typing import Optional

# Load environment variables from .env file
load_dotenv()

class SnowflakeConnection:
    def __init__(self):
        # Load credentials from environment variables
        self.account = os.getenv("SNOWFLAKE_ACCOUNT")
        self.user = os.getenv("SNOWFLAKE_USER")
        self.password = os.getenv("SNOWFLAKE_PASSWORD")
        self.database = os.getenv("SNOWFLAKE_DATABASE")
        self.schema = os.getenv("SNOWFLAKE_SCHEMA")
        
        # Validate that all required environment variables are present
        if not all([self.account, self.user, self.password, self.database, self.schema]):
            raise ValueError("❌ Missing required Snowflake environment variables. Please check your .env file.")
    
    def ejecutar_consulta_df(self, sql_query: str) -> pd.DataFrame:
        """Ejecuta una consulta SQL y retorna un DataFrame"""
        conn = None
        cursor = None
        try:
            print(f"🔄 Connecting to Snowflake...")
            conn = snowflake.connector.connect(
                account=self.account,
                user=self.user,
                password=self.password,
                database=self.database,
                schema=self.schema,
                insecure_mode=False,
                ocsp_fail_open=True
            )
            print(f"✅ Connected to Snowflake successfully")
            
            cursor = conn.cursor()
            print(f"🔄 Executing query...")
            cursor.execute(sql_query)
            
            print(f"🔄 Fetching results...")
            results = cursor.fetchall()
            columns = [col[0] for col in cursor.description]
            df = pd.DataFrame(results, columns=columns)
            
            print(f"✅ Query executed successfully. Retrieved {len(df)} rows")
            return df
            
        except snowflake.connector.errors.ProgrammingError as pe:
            print(f"❌ Snowflake Programming Error: {str(pe)}")
            print(f"❌ SQL Query: {sql_query[:200]}...")
            return pd.DataFrame()
        except snowflake.connector.errors.DatabaseError as de:
            print(f"❌ Snowflake Database Error: {str(de)}")
            return pd.DataFrame()
        except snowflake.connector.errors.Error as se:
            print(f"❌ Snowflake Error: {str(se)}")
            return pd.DataFrame()
        except Exception as e:
            print(f"❌ Unexpected error: {str(e)}")
            print(f"❌ Error type: {type(e).__name__}")
            import traceback
            print(f"❌ Traceback: {traceback.format_exc()}")
            return pd.DataFrame()
        finally:
            # Safely close cursor and connection
            if cursor:
                try:
                    cursor.close()
                    print("🔄 Cursor closed")
                except Exception as e:
                    print(f"⚠️ Error closing cursor: {str(e)}")
            
            if conn:
                try:
                    conn.close()
                    print("🔄 Connection closed")
                except Exception as e:
                    print(f"⚠️ Error closing connection: {str(e)}")

# Instancia global
try:
    db = SnowflakeConnection()
    print("✅ Snowflake connection initialized successfully")
except ValueError as e:
    print(f"❌ Failed to initialize Snowflake connection: {e}")
    db = None

def get_training_data():
    """Obtiene datos de entrenamiento para modelos supervisados"""
    if db is None:
        print("❌ Database connection not available")
        return pd.DataFrame()
        
    consulta = """
    WITH momentos_desabasto AS (
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
    WHERE m.MEMBER_CASUAL = 'casual'
    AND tiempo.YEAR = 2024
    AND NOT EXISTS (
        SELECT 1 FROM momentos_desabasto md
        WHERE md.STATION_ID = t.START_STATION_ID
        AND md.TIME_ID = t.START_TIME_ID
    )
    """
    return db.ejecutar_consulta_df(consulta)

def get_anomaly_data():
    """Obtiene datos para detección de anomalías"""
    if db is None:
        print("❌ Database connection not available")
        return pd.DataFrame()
        
    consulta = """
    WITH datos_base AS (
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
        COUNT(*) as viajes_count,
        AVG(INGRESO_MIN_EXCEDENTE) as ingreso_promedio,
        SUM(INGRESO_MIN_EXCEDENTE) as ingreso_total,
        AVG(DURACION_MINUTOS) as duracion_promedio,
        AVG(TEMPERATURE_2M) as temperatura_promedio,
        STDDEV(INGRESO_MIN_EXCEDENTE) as variabilidad_ingresos,
        MAX(INGRESO_MIN_EXCEDENTE) as ingreso_maximo,
        COUNT(CASE WHEN INGRESO_MIN_EXCEDENTE > 0 THEN 1 END) as viajes_con_excedente
    FROM datos_base
    GROUP BY STATION_ID, HOUR, IS_WEEKEND, MONTH
    HAVING COUNT(*) >= 5
    ORDER BY STATION_ID, HOUR
    """
    return db.ejecutar_consulta_df(consulta)

def get_desabasto_events():
    """Obtiene eventos de desabasto para análisis contrafactual"""
    if db is None:
        print("❌ Database connection not available")
        return pd.DataFrame()
        
    consulta = """
    WITH momentos_desabasto AS (
        SELECT DISTINCT
            eb.STATION_ID,
            eb.TIME_ID,
            eb.BALANCE_NET,
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
        COALESCE(se.duracion_promedio_estacion, 35) as duracion_promedio_estacion,
        COALESCE(se.prop_bike_type_1, 0.7) as prop_bike_type_1,
        30 as MINUTOS_INCLUIDOS,
        ROUND(ABS(md.BALANCE_NET) * 0.6) as viajes_perdidos_estimados
    FROM momentos_desabasto md
    LEFT JOIN stats_estacion se ON se.STATION_ID = md.STATION_ID
    """
    return db.ejecutar_consulta_df(consulta)

def get_dashboard_data():
    """Obtiene datos principales para el dashboard"""
    if db is None:
        print("❌ Database connection not available")
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
    WHERE tiempo.YEAR = 2024
    """
    return db.ejecutar_consulta_df(consulta) 