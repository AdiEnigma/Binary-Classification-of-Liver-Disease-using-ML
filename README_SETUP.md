# Liver Disease Prediction System - Complete Setup Guide

## 🚀 Quick Start Options

### Option 1: Windows Users (Easiest)
1. **Double-click `START_SERVER.bat`**
   - This will start both backend and frontend automatically
   - Two command windows will open (you can minimize them)

### Option 2: Manual Start (All Platforms)

#### Terminal 1: Backend
```bash
cd backend
python -m uvicorn main:app --reload --port 8000
```

#### Terminal 2: Frontend
```bash
cd "Doctor-Friendly Liver Disease Dashboard"
npm run dev
```

## ⚠️ Important: Train Models First!

Before making predictions, you need to train the models:

```bash
python _data_preparation.py
python _supervised_learning.py  
python _unsupervised_learning1.py
```

This will create the `saved_models/` directory with trained models.

## 📍 Access Points

Once both servers are running:

- **Frontend Application**: http://localhost:5173
- **Backend API Docs**: http://localhost:8000/docs
- **Backend Health Check**: http://localhost:8000/api/health

## 🎯 Using the Application

1. **Open browser** to http://localhost:5173
2. **Individual Patient Check**:
   - Enter patient clinical data
   - Click "Predict"
   - View risk score, confidence, and contributing factors
3. **Bulk CSV Analysis**:
   - Upload CSV file with patient data
   - View predictions for all patients
   - Download results as CSV

## 📊 What You Get

- ✅ Real-time disease risk prediction
- ✅ SHAP explainability (why the prediction was made)
- ✅ Clustering analysis (patient grouping)
- ✅ Bulk processing for multiple patients
- ✅ Downloadable results

## System Architecture

```
Frontend (React + TypeScript)
  ↓ HTTP/WebSocket
Backend (FastAPI)
  ↓ Model Loading
ML Models (XGBoost, UMAP+KMeans, SHAP)
```

## API Endpoints

- **Individual Prediction**: `POST /api/predict/individual`
- **Bulk CSV Prediction**: `POST /api/predict/bulk`
- **WebSocket**: `WS /ws/predict`
- **Health Check**: `GET /api/health`
- **API Documentation**: http://localhost:8000/docs

## 🔧 Troubleshooting

**Backend won't start:**
- Check if port 8000 is in use
- Ensure Python dependencies are installed: `pip install -r backend/requirements.txt`

**Frontend won't start:**
- Run `npm install` in the frontend directory
- Check if Node.js is installed

**Predictions fail:**
- Ensure models are trained (run training scripts)
- Check backend logs for errors
- Verify `saved_models/` directory exists with model files

**CORS errors:**
- Backend is configured to allow all origins
- Ensure backend is running before frontend makes requests

## 🎉 You're All Set!

The system is ready to use. Just start both servers and open your browser!

## Environment Variables (Optional)

Create `.env` file in frontend directory:
```
VITE_API_URL=http://localhost:8000
```

## Features

✅ Individual patient risk prediction
✅ Bulk CSV file processing
✅ Supervised learning (XGBoost) predictions
✅ Unsupervised learning (clustering) analysis
✅ SHAP explainability values
✅ Real-time WebSocket communication
