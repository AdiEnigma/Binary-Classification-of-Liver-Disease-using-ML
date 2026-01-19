"""
FastAPI Backend Application for Liver Disease Prediction
"""

from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from api import routes
from api.websocket import websocket_handler
import uvicorn


# Create FastAPI app
app = FastAPI(
    title="Liver Disease Prediction API",
    description="ML-based liver disease risk prediction API with supervised, unsupervised, and SHAP explainability",
    version="1.0.0"
)

# Configure CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Include REST API routes
app.include_router(routes.router)

# WebSocket endpoint
@app.websocket("/ws/predict")
async def websocket_endpoint(websocket: WebSocket):
    """WebSocket endpoint for bidirectional prediction requests"""
    await websocket_handler(websocket)


# Health check endpoint
@app.get("/api/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Try to load predictor to check if models are available
        from utils.predictor import get_predictor
        predictor = get_predictor()
        
        return {
            "status": "healthy",
            "models_loaded": True,
            "message": "All models loaded successfully"
        }
    except Exception as e:
        return JSONResponse(
            status_code=503,
            content={
                "status": "unhealthy",
                "models_loaded": False,
                "error": str(e)
            }
        )


# Root endpoint
@app.get("/")
async def root():
    """Root endpoint with API information"""
    return {
        "message": "Liver Disease Prediction API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/api/health",
        "endpoints": {
            "individual": "POST /api/predict/individual",
            "bulk": "POST /api/predict/bulk",
            "websocket": "WS /ws/predict"
        }
    }


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
