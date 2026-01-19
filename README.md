# Liver Disease Prediction System

A comprehensive machine learning-powered clinical decision support system for liver disease risk assessment. Built with React frontend and FastAPI backend, featuring XGBoost predictions, SHAP explainability, and clustering analysis.

![System Status](https://img.shields.io/badge/Status-Production%20Ready-green)
![Python](https://img.shields.io/badge/Python-3.8+-blue)
![React](https://img.shields.io/badge/React-18+-blue)
![FastAPI](https://img.shields.io/badge/FastAPI-Latest-green)

## Key Features Overview

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   Individual        │   Bulk CSV          │     SHAP            │
│   Prediction        │   Processing        │  Explainability     │
│                     │                     │                     │
│ • Real-time risk    │ • Upload CSV files  │ • Feature           │
│   assessment        │ • Batch processing  │   importance        │
│ • Lab value input   │ • Download results  │ • Clinical insights │
│ • Instant feedback  │ • Multiple patients │ • Model reasoning   │
├─────────────────────┼─────────────────────┴─────────────────────┤
│   ML Models         │        Technical Stack                    │
│                     │                                           │
│ • XGBoost (90%+)    │ Frontend: React + TypeScript + Tailwind  │
│ • Random Forest     │ Backend:  FastAPI + Python + Pydantic    │
│ • Clustering        │ ML Stack: XGBoost + SHAP + UMAP + KMeans │
│ • SHAP Values       │ Data:     Indian Liver Patient Dataset   │
└─────────────────────┴───────────────────────────────────────────┘
```

## Quick Start

### Setup Progress Checklist

```
Setup Steps:
□ 1. Install Python dependencies
□ 2. Install Node.js dependencies  
□ 3. Train ML models (first time only)
□ 4. Start backend server
□ 5. Start frontend server
□ 6. Open browser to localhost:5173
```

### Windows Users (Easiest)
```bash
# Step 1-3: Train models (first time only)
python _data_preparation.py      # ████████████████████ Processing data...
python _supervised_learning.py   # ████████████████████ Training models...
python _unsupervised_learning1.py # ████████████████████ Clustering analysis...

# Step 4-6: Start both servers
START_SERVER.bat                 # ████████████████████ Launching application...
```

### Manual Start (All Platforms)
```bash
# Terminal 1: Backend (Step 4)
cd backend
python -m uvicorn main:app --reload --port 8000
# ✓ Backend running at http://localhost:8000

# Terminal 2: Frontend (Step 5)
cd "Doctor-Friendly Liver Disease Dashboard"
npm install  # First time only
npm run dev
# ✓ Frontend running at http://localhost:5173
```

### Access Points
- **Frontend Application**: http://localhost:5173
- **Backend API Docs**: http://localhost:8000/docs
- **Health Check**: http://localhost:8000/api/health

## Features

### Clinical Analysis Capabilities

```
┌─────────────────────┬─────────────────────┬─────────────────────┐
│   Risk Assessment   │   Batch Processing  │   Explainability    │
│                     │                     │                     │
│ • Individual        │ • CSV file upload   │ • SHAP values       │
│   patient analysis  │ • Multiple patients │ • Feature           │
│ • Real-time         │ • Bulk predictions  │   importance        │
│   predictions       │ • Download results  │ • Clinical insights │
│ • Risk              │ • Summary           │ • Model reasoning   │
│   stratification    │   statistics        │ • Transparency      │
└─────────────────────┴─────────────────────┴─────────────────────┘
```

### Machine Learning Models

```
Model Comparison:
┌─────────────────┬─────────────┬─────────────┬─────────────┐
│    Feature      │   XGBoost   │Random Forest│ Logistic    │
├─────────────────┼─────────────┼─────────────┼─────────────┤
│ Accuracy        │ ⭐⭐⭐⭐⭐    │ ⭐⭐⭐⭐      │ ⭐⭐⭐        │
│ Speed           │ ⭐⭐⭐        │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐⭐     │
│ Interpretability│ ⭐⭐         │ ⭐⭐⭐        │ ⭐⭐⭐⭐⭐     │
│ Clinical Trust  │ ⭐⭐⭐        │ ⭐⭐⭐⭐      │ ⭐⭐⭐⭐⭐     │
│ Complexity      │ High        │ Medium      │ Low         │
└─────────────────┴─────────────┴─────────────┴─────────────┘
```

### Technical Stack
- **Frontend**: React 18 + TypeScript + Tailwind CSS
- **Backend**: FastAPI + Python 3.8+
- **ML**: XGBoost, scikit-learn, SHAP, UMAP
- **Data**: Indian Liver Patient Dataset (ILPD)

## Model Performance

```
Performance Comparison Matrix:
┌─────────────┬─────────────┬─────────────┬─────────────┐
│   XGBoost   │Random Forest│  Logistic   │   Target    │
│  (Winner)   │             │ Regression  │ (Minimum)   │
├─────────────┼─────────────┼─────────────┼─────────────┤
│ Recall: 91% │ Recall: 87% │ Recall: 87% │ Recall: 85% │
│ Prec:   84% │ Prec:   81% │ Prec:   82% │ Prec:   70% │
│ F1:    0.87 │ F1:    0.84 │ F1:    0.84 │ F1:    N/A  │
│ AUC:   0.92 │ AUC:   0.89 │ AUC:   0.90 │ AUC:   0.80 │
└─────────────┴─────────────┴─────────────┴─────────────┘
```

**Visual Performance Bars:**
```
XGBoost Recall:     ████████████████████████████████████████ 91%
Random Forest:      ████████████████████████████████████     87%
Logistic Reg:       ████████████████████████████████████     87%
Target (85%):       ████████████████████████████████████
                    ↑ All models exceed clinical targets
```

All models exceed clinical targets (Recall ≥85%, Precision ≥70%)

## System Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                        System Flow Diagram                         │
└─────────────────────────────────────────────────────────────────────┘

┌─────────────────┐    HTTP/REST     ┌─────────────────┐
│  React Frontend │ ◄──────────────► │ FastAPI Backend │
│   (Port 5173)   │                  │   (Port 8000)   │
│                 │    WebSocket     │                 │
│ • Patient Forms │ ◄──────────────► │ • API Routes    │
│ • Results UI    │                  │ • Validation    │
│ • Charts/Graphs │                  │ • Orchestration │
│ • Explanations  │                  │ • Error Handling│
└─────────────────┘                  └─────────┬───────┘
                                               │
                                               ▼
                                    ┌─────────────────┐
                                    │   ML Pipeline   │
                                    │ (saved_models/) │
                                    │                 │
                                    │ • XGBoost Model │
                                    │ • SHAP Explainer│
                                    │ • UMAP Reducer  │
                                    │ • K-Means       │
                                    │ • RobustScaler  │
                                    └─────────────────┘

Data Flow:
Patient Input → Validation → Preprocessing → ML Models → Results → Frontend
```

## Prerequisites

### Python Environment
```bash
# Create virtual environment
python -m venv .venv
.venv\Scripts\activate  # Windows
source .venv/bin/activate  # Linux/Mac

# Install dependencies
pip install -r backend/requirements.txt
```

### Node.js Environment
```bash
# Install Node.js 18+ from nodejs.org
cd "Doctor-Friendly Liver Disease Dashboard"
npm install
```

## Installation & Setup

### 1. Clone & Setup Environment
```bash
git clone <repository-url>
cd liver-disease-ml
python -m venv .venv
.venv\Scripts\activate
pip install -r backend/requirements.txt
```

### 2. Train ML Models
```bash
python _data_preparation.py      # Data preprocessing & feature engineering
python _supervised_learning.py   # Train XGBoost, Random Forest, Logistic Regression  
python _unsupervised_learning1.py # UMAP + K-Means clustering
```

### 3. Install Frontend Dependencies
```bash
cd "Doctor-Friendly Liver Disease Dashboard"
npm install
```

### 4. Start Application
```bash
# Option A: Automatic (Windows)
START_SERVER.bat

# Option B: Manual
# Terminal 1: Backend
cd backend && python -m uvicorn main:app --reload --port 8000

# Terminal 2: Frontend
cd "Doctor-Friendly Liver Disease Dashboard" && npm run dev
```

## Usage Examples

### Individual Patient Prediction
```python
# Example patient data
patient = {
    "age": 55,
    "gender": "Male", 
    "totalBilirubin": 4.2,    # High (normal: 0.3-1.2)
    "directBilirubin": 2.1,   # High (normal: 0.1-0.3)
    "alkalinePhosphatase": 285, # High (normal: 44-147)
    "sgptAlt": 95,            # High (normal: 10-40)
    "sgotAst": 128,           # High (normal: 10-40)
    "totalProteins": 5.8,     # Low (normal: 6.3-8.2)
    "albumin": 2.9,           # Low (normal: 3.5-5.0)
    "agRatio": 0.65           # Low (normal: 1.1-2.5)
}

# Expected result: High disease risk (90%+ probability)
```

### Bulk CSV Processing
Upload CSV with columns: `age,gender,totalBilirubin,directBilirubin,alkalinePhosphatase,sgptAlt,sgotAst,totalProteins,albumin,agRatio`

## API Endpoints

### REST API
- `POST /api/predict/individual` - Single patient prediction
- `POST /api/predict/bulk` - Bulk CSV processing
- `GET /api/health` - System health check
- `GET /docs` - Interactive API documentation

### WebSocket
- `WS /ws/predict` - Real-time bidirectional communication

## Testing

```bash
# Test backend setup
python test_backend.py

# Test API endpoints (requires running server)
python test_api.py
```

## Model Interpretability

### SHAP Values
- **Feature Importance**: Which biomarkers matter most
- **Individual Explanations**: Why this patient is high/low risk
- **Clinical Insights**: Understand model reasoning

### Risk Stratification
- **High Risk** (>70%): Immediate specialist referral
- **Medium Risk** (30-70%): Follow-up in 2-4 weeks  
- **Low Risk** (<30%): Routine monitoring

## Important Disclaimers

⚠️ **Medical Disclaimer**: This tool provides ML-based risk prediction support. It is NOT a medical diagnosis. Final diagnosis and treatment decisions must be made by qualified healthcare professionals.

⚠️ **Validation Required**: Before clinical use, this system requires proper validation, regulatory approval, and compliance with healthcare data regulations (HIPAA, GDPR).

## Troubleshooting

### Common Issues

**Backend won't start:**
- Check if port 8000 is in use: `netstat -an | findstr 8000`
- Ensure models are trained: Run training scripts first
- Install dependencies: `pip install -r backend/requirements.txt`

**Frontend won't start:**
- Install Node.js 18+
- Run `npm install` in frontend directory
- Check if port 5173 is available

**Predictions fail:**
- Verify `saved_models/` directory exists with model files
- Check backend logs for errors
- Ensure all training scripts completed successfully

**CORS errors:**
- Backend is configured to allow all origins for development
- Ensure backend is running before frontend makes requests

## Project Structure

```
liver-disease-ml/
├── backend/                          # FastAPI backend
│   ├── api/                         # REST API routes
│   ├── models/                      # Pydantic schemas
│   ├── utils/                       # ML prediction logic
│   └── main.py                      # FastAPI application
├── Doctor-Friendly Liver Disease Dashboard/  # React frontend
│   ├── src/app/                     # React components
│   ├── src/styles/                  # CSS styles
│   └── package.json                 # Dependencies
├── data/                            # Training data
│   ├── raw/                         # Original ILPD dataset
│   └── processed/                   # Preprocessed data
├── saved_models/                    # Trained ML models
├── outputs/                         # Visualizations & results
├── _data_preparation.py             # Data preprocessing
├── _supervised_learning.py          # Model training
├── _unsupervised_learning1.py       # Clustering analysis
├── predict.py                       # Core prediction logic
└── START_SERVER.bat                 # Windows startup script
```

## Contributing

1. Fork the repository
2. Create feature branch: `git checkout -b feature/new-feature`
3. Commit changes: `git commit -am 'Add new feature'`
4. Push to branch: `git push origin feature/new-feature`
5. Submit pull request

## License

This project is for demonstration and educational purposes. Not for clinical use without proper validation and regulatory approval.

## Support

For technical support, bug reports, or feature requests:
- Check the troubleshooting section above
- Review API documentation at `/docs`
- Examine log files in backend console

---

**Ready to predict liver disease risk with machine learning!**

Start with `START_SERVER.bat` and open http://localhost:5173 in your browser.