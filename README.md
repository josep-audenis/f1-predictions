# F1 Predictions API

A production-ready FastAPI-based REST API for Formula 1 race predictions using machine learning models trained on historical F1 data.

[![Python 3.13+](https://img.shields.io/badge/python-3.13+-blue.svg)](https://www.python.org/downloads/)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.104+-green.svg)](https://fastapi.tiangolo.com/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)



## Features

- **Race Predictions**: ML-powered predictions for upcoming races and specific Grand Prix
- **Calendar Management**: Complete F1 race calendar with session times
- **Model Performance**: Query model metrics and compare different algorithms
- **Live Updates**: Update race data and retrain models on-demand
- **High Performance**: Built with FastAPI for exceptional speed
- **Interactive Docs**: Auto-generated Swagger UI and ReDoc documentation
- **CORS Support**: Configurable for frontend integration



## Table of Contents

- [Architecture](#architecture)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Installation](#installation)
- [Running the API](#running-the-api)
- [API Endpoints](#api-endpoints)
- [Training Models](#training-models)


## Architecture

This project follows a **separation of concerns** architecture with two distinct layers:

### Offline Layer (Training - `src/`)
- Data collection from FastF1 API
- Feature engineering
- Model training and evaluation
- Runs periodically (e.g., weekly)

### Online Layer (Serving - `app/`)
- FastAPI REST API
- Real-time predictions
- Model serving
- Runs continuously

### Why Separate?

- **Independent scaling**: Train on powerful machines, serve on lightweight instances
- **Different requirements**: Training needs hours, API needs <100ms responses
- **Clear boundaries**: Training code is experimental, API code is production-stable
- **Security**: Training logic isn't exposed in the production API

This separation pattern is used by major tech companies like Netflix (Metaflow), Uber (Michelangelo), Google (TFX), and Airbnb (Bighead).



## Quick Start

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Fix imports (important!)
python fix_imports.py

# 3. Set up directories
mkdir -p data/raw data/processed models/top1/quali models/top1/pre-quali

# 4. Train production models
python scripts/train_production_models.py

# 5. Start the API
python -m app.main
```

**That's it!** Your API is running at http://localhost:8000



## Project Structure

```
f1-predictions/
│
├── app/                           
│   ├── api/
│   │   └── routes/
│   │       ├── predictions.py     
│   │       ├── races.py           
│   │       ├── models.py          
│   │       └── training.py        
│   ├── core/
│   │   └── config.py              
│   ├── services/
│   │   ├── data_service.py        
│   │   └── prediction_service.py  
│   ├── utils/
│   │   └── calendar_utils.py      
│   └── main.py                    
│
├── src/                           
│   ├── data_collection/
│   │   ├── collect_season.py      
│   │   └── collection_utils.py
│   ├── features/
│   │   └── build_predictive_features.py  
│   ├── predictions/
│   │   ├── compare_models_top1_quali.py      
│   │   ├── compare_models_top1_pre_quali.py  
│   │   └── ...
│   └── utils/
│       ├── data_utils.py
│       └── model_utils.py
│
├── data/                          
│   ├── raw/                       
│   └── processed/                 
│
├── models/                        
│   └── top1/
│       ├── quali/                 
│       │   ├── GradientBoosting_model.pkl
│       │   ├── RandomForest_model.pkl
│       │   └── LogisticRegression_model.pkl
│       └── pre-quali/             
│           ├── GradientBoosting_model.pkl
│           ├── RandomForest_model.pkl
│           └── LogisticRegression_model.pkl
│
├── scripts/                       
│   ├── train_production_models.py 
│   ├── test_api.py                
│   └── fix_imports.py             
│
├── requirements.txt               
└── README.md                      
```



## Installation

### Prerequisites

- Python 3.13
- pip package manager

### Step 1: Navigate to Project

```bash
cd /path/to/f1-predictions
```

### Step 2: Install Dependencies

```bash
pip install -r requirements.txt
```

**Key Dependencies:**
- `fastapi` - Web framework
- `uvicorn` - ASGI server
- `pandas` - Data manipulation
- `scikit-learn` - ML models
- `fastf1` - F1 data API
- `pydantic-settings` - Configuration
- `joblib` - Model serialization



## Running the API

### Start the Server

```bash
# Standard way
python -m app.main

# Or using uvicorn directly
uvicorn app.main:app --reload --host 0.0.0.0 --port 8000
```

The `--reload` flag enables auto-reload on code changes (useful for development).

### Access Points

Once running, access:
- **API**: http://localhost:8000
- **Interactive Docs (Swagger UI)**: http://localhost:8000/docs
- **Alternative Docs (ReDoc)**: http://localhost:8000/redoc
- **OpenAPI Schema**: http://localhost:8000/openapi.json



## API Endpoints

### System Endpoints

| Method | Endpoint | Description |
|--|-|-|
| `GET` | `/` | Root endpoint with API info |
| `GET` | `/health` | Health check |

### Prediction Endpoints

| Method | Endpoint | Description |
|--|-|-|
| `GET` | `/api/v1/predictions/next-race` | Get predictions for next upcoming race |
| `GET` | `/api/v1/predictions/race/{year}/{grand_prix}` | Get predictions for specific race |

### Race Endpoints

| Method | Endpoint | Description |
|--|-|-|
| `GET` | `/api/v1/races/calendar/{year}` | Get full race calendar for year |
| `GET` | `/api/v1/races/next` | Get next race information |
| `GET` | `/api/v1/races/{year}/{grand_prix}` | Get specific race details |
| `GET` | `/api/v1/races/status` | Get next race status (quali completed?) |

### Model Endpoints

| Method | Endpoint | Description |
|--|-|-|
| `GET` | `/api/v1/models/performance/top1-quali` | Post-qualifying model metrics |
| `GET` | `/api/v1/models/performance/top1-pre-quali` | Pre-qualifying model metrics |
| `GET` | `/api/v1/models/best` | Best performing models |
| `GET` | `/api/v1/models/available` | List all available models |

### Training Endpoints

| Method | Endpoint | Description |
|--|-|-|
| `POST` | `/api/v1/training/retrain` | Trigger model retraining (background) |
| `POST` | `/api/v1/training/update-data` | Update with latest race data |



## Training Models

### Production Models (100% Data)

For deployment, train models using **100% of available data** (no train/test split):

```bash
python scripts/train_production_models.py
```

**This script:**
- Trains on 100% of data for maximum accuracy
- Creates both quali and pre-quali models
- Saves models to `models/top1/` directories
- Generates performance metadata
- Trains LogisticRegression, RandomForest, and GradientBoosting

**Output:**
```
models/top1/
├── quali/
│   ├── GradientBoosting_model.pkl   
│   ├── RandomForest_model.pkl
│   └── LogisticRegression_model.pkl
└── pre-quali/
    ├── GradientBoosting_model.pkl   
    ├── RandomForest_model.pkl
    └── LogisticRegression_model.pkl
```

### Complete Training Pipeline

If you need to collect new data and build features from scratch:

```bash
# Step 1: Collect race data
python -m src.data_collection.collect_season

# Step 2: Build features
python -m src.features.build_predictive_features

# Step 3: Train production models
python scripts/train_production_models.py
```

### Model Selection

The API uses **GradientBoosting** by default (hardcoded for simplicity and reliability).

To use a different model, edit `app/services/prediction_service.py`:

```python
def _load_best_model(self, quali_done: bool) -> any:
    model_name = "GradientBoosting"  # Change to "RandomForest" or "LogisticRegression"
    # ...
```



## Testing the API

Use the provided test script to verify all endpoints:

```bash
python scripts/test_api.py
```

This will test all endpoints and show which ones are working correctly.



### Health Check

Quick test to verify the API is running:

```bash
curl http://localhost:8000/health
```

Should return:
```json
{"status": "healthy"}
```



## Performance

- **Response Time**: <100ms average (with loaded models)
- **Model Size**: ~10-50 MB per model
- **Memory Usage**: ~500MB per instance
- **Throughput**: Handles hundreds of requests per second



## License

This project is licensed under the MIT License - see the LICENSE file for details.



## Roadmap

- [ ] Add more prediction types (podium, fastest lap)
- [ ] Real-time predictions during race weekends
- [ ] Historical prediction accuracy tracking
- [ ] Driver/team comparison endpoints
- [ ] WebSocket support for live updates
- [ ] Mobile app integration
