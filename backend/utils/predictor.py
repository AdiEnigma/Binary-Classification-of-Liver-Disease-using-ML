"""
Wrapper around predict.py for backend use
Handles model initialization and caching
"""

import sys
import os

# Add parent directory to path to import predict.py
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, BASE_DIR)

from predict import LiverDiseasePredictor
from typing import Dict, Any, Optional


# Global predictor instance (lazy loaded)
_predictor_instance: Optional[LiverDiseasePredictor] = None


def get_predictor(model_dir: str = "saved_models") -> LiverDiseasePredictor:
    """
    Get or create the global predictor instance.
    Uses singleton pattern to avoid reloading models on every request.
    
    Args:
        model_dir: Directory containing model files
        
    Returns:
        LiverDiseasePredictor instance
    """
    global _predictor_instance
    
    if _predictor_instance is None:
        print(f"Loading models from {model_dir}...")
        _predictor_instance = LiverDiseasePredictor(model_dir=model_dir)
        print("✓ All models loaded successfully")
    
    return _predictor_instance


def predict_complete(patient_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Wrapper function for complete prediction.
    
    Args:
        patient_data: Patient data dictionary
        
    Returns:
        Complete prediction results
    """
    predictor = get_predictor()
    return predictor.predict_complete(patient_data)
