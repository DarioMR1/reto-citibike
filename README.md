# CitiBike Analytics & ML Platform

Advanced Analytics and Machine Learning Platform for CitiBike Operations built with **FastAPI** backend and **Next.js** frontend.

## 🏗️ Architecture

This project follows a modern, scalable architecture:

- **Backend**: FastAPI (deployed on Google Cloud Run)
- **Frontend**: Next.js (deployed on Vercel)
- **Database**: Snowflake
- **ML Stack**: scikit-learn, pandas, numpy
- **Visualization**: Plotly

## 📁 Project Structure

```
reto-citibike/
├── main.py                  # FastAPI application
├── database.py              # Snowflake database connections
├── models.py                # ML models (supervised & unsupervised)
├── visualizations.py        # Plotly charts and KPIs
├── requirements.txt         # Python dependencies
├── Dockerfile              # Container configuration
├── Makefile                # Deployment commands
├── env.example             # Environment variables template
└── README.md               # This file

reto-citibike-web/
├── app/
│   └── page.tsx            # Next.js main dashboard
├── components/ui/          # Reusable UI components
├── lib/utils.ts           # Utility functions
├── package.json           # Node.js dependencies
└── next.config.js         # Next.js configuration
```

## 🚀 Quick Start

### Backend (FastAPI)

1. **Setup environment**:

```bash
cd reto-citibike
cp env.example .env
# Edit .env with your Snowflake credentials
```

2. **Install dependencies**:

```bash
pip install -r requirements.txt
```

3. **Run locally**:

```bash
python main.py
# API will be available at http://localhost:8000
```

### Frontend (Next.js)

1. **Setup environment**:

```bash
cd reto-citibike-web
cp env.example .env.local
# Edit .env.local with your API URL
```

2. **Install dependencies**:

```bash
pnpm install
```

3. **Run locally**:

```bash
pnpm dev
# App will be available at http://localhost:3000
```

## 🚢 Deployment

### Backend to Google Cloud Run

```bash
cd reto-citibike
make deploy
```

### Frontend to Vercel

```bash
cd reto-citibike-web
vercel --prod
```

## 📊 Features

### Machine Learning

- **Supervised Models**: Revenue prediction using Random Forest and Gradient Boosting
- **Unsupervised Models**: Anomaly detection using Isolation Forest and LOF
- **Model Management**: Save/load trained models
- **Performance Metrics**: R², RMSE, MAE tracking

### Analytics & Predictions

- **Dashboard**: Real-time KPIs and visualizations
- **Single Predictions**: Individual trip revenue forecasting
- **Batch Analysis**: Scenario analysis across multiple parameters
- **Anomaly Detection**: Identify unusual patterns in operations
- **Counterfactual Analysis**: Estimate losses from shortage events

### API Endpoints

- `GET /api/status` - System status
- `GET /api/dashboard` - Dashboard data and KPIs
- `POST /api/train/supervised` - Train revenue prediction models
- `POST /api/train/unsupervised` - Train anomaly detection models
- `POST /api/predict/single` - Single trip prediction
- `POST /api/predict/anomaly` - Anomaly prediction
- `POST /api/analyze/batch` - Batch scenario analysis
- `GET /api/anomalies/analysis` - Anomaly analysis results
- `GET /api/counterfactual/analysis` - Counterfactual analysis

## 🛠️ Development

### Environment Variables

Backend (`.env`):

```bash
# Google Cloud Configuration
GOOGLE_CLOUD_PROJECT=your-project-id
GOOGLE_CLOUD_LOCATION=us-central1
SERVICE_NAME=citibike-analytics-api

# Snowflake Database Configuration
SNOWFLAKE_ACCOUNT=your_account_here
SNOWFLAKE_USER=your_user_here
SNOWFLAKE_PASSWORD=your_password_here
SNOWFLAKE_DATABASE=your_database_here
SNOWFLAKE_SCHEMA=your_schema_here

# Application Configuration
APP_ENV=production
DEBUG=False
PORT=8000
```

Frontend (`.env.local`):

```bash
NEXT_PUBLIC_API_URL=https://your-cloud-run-url
```

### Make Commands

- `make deploy` - Deploy to Cloud Run
- `make delete` - Remove service
- `make logs` - View service logs
- `make status` - Show service status

## 🔧 Technology Stack

- **Backend**: FastAPI, uvicorn, pydantic
- **ML/Data**: pandas, numpy, scikit-learn, plotly
- **Database**: snowflake-connector-python
- **Infrastructure**: Google Cloud Run, Docker
- **Frontend**: Next.js, React, TypeScript, Tailwind CSS
- **UI Components**: Radix UI, Lucide Icons

## 📈 Model Performance

The platform automatically evaluates multiple algorithms and selects the best performing model based on R² score:

- **Random Forest**: Robust ensemble method
- **Gradient Boosting**: Advanced boosting algorithm
- **Isolation Forest**: Anomaly detection
- **Local Outlier Factor**: Local anomaly detection

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Run tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License.
