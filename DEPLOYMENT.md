# 🚀 Deployment Guide - Render

## Pre-requisitos

- Cuenta en [Render](https://render.com)
- Repositorio de GitHub con el código

## Configuración en Render

### 1. **Crear Web Service**

- Conecta tu repositorio de GitHub
- Selecciona el branch `main`

### 2. **Configuración del Service**

```
Name: citibike-analytics
Environment: Python 3
Build Command: pip install -r requirements.txt
Start Command: streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### 3. **Variables de Entorno**

En la sección "Environment Variables" de Render, agrega:

```
SNOWFLAKE_ACCOUNT=your_account_here
SNOWFLAKE_USER=your_user_here
SNOWFLAKE_PASSWORD=your_password_here
SNOWFLAKE_DATABASE=your_database_here
SNOWFLAKE_SCHEMA=your_schema_here
APP_ENV=production
DEBUG=False
```

### 4. **Configuración Avanzada**

```
Auto-Deploy: Yes
Instance Type: Starter (Free)
Region: Oregon (US West)
```

## 🔧 Comandos Importantes

### Start Command para Render:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0
```

### Build Command (opcional si hay problemas):

```bash
pip install --upgrade pip && pip install -r requirements.txt
```

## 📁 Archivos de Configuración

- `runtime.txt` - Especifica Python 3.12.3
- `.streamlit/config.toml` - Configuración optimizada para producción
- `requirements.txt` - Dependencias con versiones específicas
- `.env.example` - Template de variables de entorno

## 🛡️ Seguridad

- ✅ Variables de entorno en Render (no en código)
- ✅ `.env` en `.gitignore`
- ✅ Credenciales protegidas

## 🔍 Troubleshooting

### Error de Puerto

Si hay problemas con el puerto, usar:

```bash
streamlit run app.py --server.port=$PORT --server.address=0.0.0.0 --server.headless=true
```

### Error de Memoria

Si la app es muy pesada, considerar:

- Upgrade a plan pagado de Render
- Optimizar queries de Snowflake
- Usar más `@st.cache_data`

### Error de Build

Si falla el build:

1. Verificar `requirements.txt`
2. Verificar `runtime.txt`
3. Revisar logs en Render Dashboard

## 🌐 URL Final

Una vez desplegado, tu app estará disponible en:
`https://citibike-analytics.onrender.com`
