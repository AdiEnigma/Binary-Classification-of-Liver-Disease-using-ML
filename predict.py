"""
Unified Prediction Pipeline for Liver Disease Diagnosis
Combines Supervised Learning (XGBoost), Unsupervised Learning (Clustering), and SHAP Explainability
"""

import os
import numpy as np
import pandas as pd
import joblib
from typing import Dict, List, Optional, Any
import warnings
warnings.filterwarnings('ignore')


class LiverDiseasePredictor:
    """
    Unified predictor that combines:
    - Supervised learning (XGBoost) for disease risk prediction
    - Unsupervised learning (UMAP + PSO-KMeans) for cluster analysis
    - SHAP explainer for feature contribution analysis
    """
    
    def __init__(self, model_dir: str = "saved_models"):
        """
        Initialize the predictor by loading all trained models and artifacts.
        
        Args:
            model_dir: Directory containing saved model files
        """
        self.model_dir = model_dir
        BASE_DIR = os.path.dirname(os.path.abspath(__file__))
        self.model_path = os.path.join(BASE_DIR, model_dir)
        
        # Load supervised learning models
        self.supervised_model = None
        self.supervised_scaler = None
        self.shap_explainer = None
        
        # Load unsupervised learning artifacts
        self.unsupervised_scaler = None
        self.umap_reducer = None
        self.cluster_centroids = None
        
        # Feature mapping: frontend names -> model names
        self.feature_mapping = {
            'age': 'age',
            'gender': 'gender',
            'totalBilirubin': 'total_bilirubin',
            'directBilirubin': 'direct_bilirubin',
            'alkalinePhosphatase': 'alkaline_phosphotase',
            'sgptAlt': 'alamine_aminotransferase',
            'sgotAst': 'aspartate_aminotransferase',
            'totalProteins': 'total_protiens',
            'albumin': 'albumin',
            'agRatio': 'albumin_and_globulin_ratio'
        }
        
        # Reverse mapping for output
        self.reverse_mapping = {v: k for k, v in self.feature_mapping.items()}
        
        # Clustering features (same as in unsupervised learning)
        self.clustering_features = [
            'total_bilirubin', 'direct_bilirubin', 'alkaline_phosphotase',
            'alamine_aminotransferase', 'aspartate_aminotransferase',
            'total_protiens', 'albumin', 'albumin_and_globulin_ratio'
        ]
        
        # Load all models
        self._load_models()
    
    def _load_models(self):
        """Load all trained models and artifacts from disk."""
        try:
            # Load supervised models
            xgb_path = os.path.join(self.model_path, "supervised_xgboost_model.pkl")
            scaler_path = os.path.join(self.model_path, "supervised_scaler.pkl")
            shap_path = os.path.join(self.model_path, "shap_explainer.pkl")
            
            if os.path.exists(xgb_path):
                self.supervised_model = joblib.load(xgb_path)
                print(f"✓ Loaded XGBoost model from {xgb_path}")
            else:
                raise FileNotFoundError(f"XGBoost model not found at {xgb_path}")
            
            if os.path.exists(scaler_path):
                self.supervised_scaler = joblib.load(scaler_path)
                print(f"✓ Loaded supervised scaler from {scaler_path}")
            else:
                raise FileNotFoundError(f"Supervised scaler not found at {scaler_path}")
            
            if os.path.exists(shap_path):
                self.shap_explainer = joblib.load(shap_path)
                print(f"✓ Loaded SHAP explainer from {shap_path}")
            else:
                print(f"⚠️  SHAP explainer not found at {shap_path}")
            
            # Load unsupervised artifacts
            unsup_scaler_path = os.path.join(self.model_path, "unsupervised_scaler.pkl")
            umap_path = os.path.join(self.model_path, "umap_reducer.pkl")
            centroids_path = os.path.join(self.model_path, "pso_kmeans_centroids.npy")
            
            if os.path.exists(unsup_scaler_path):
                self.unsupervised_scaler = joblib.load(unsup_scaler_path)
                print(f"✓ Loaded unsupervised scaler from {unsup_scaler_path}")
            else:
                print(f"⚠️  Unsupervised scaler not found at {unsup_scaler_path}")
            
            if os.path.exists(umap_path):
                self.umap_reducer = joblib.load(umap_path)
                print(f"✓ Loaded UMAP reducer from {umap_path}")
            else:
                print(f"⚠️  UMAP reducer not found at {umap_path}")
            
            if os.path.exists(centroids_path):
                self.cluster_centroids = np.load(centroids_path)
                print(f"✓ Loaded cluster centroids from {centroids_path}")
            else:
                print(f"⚠️  Cluster centroids not found at {centroids_path}")
                
        except Exception as e:
            raise RuntimeError(f"Error loading models: {str(e)}")
    
    def _convert_frontend_to_model_format(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Convert frontend patient data format to model input format.
        
        Frontend: {totalBilirubin, sgptAlt, ...}
        Model: {total_bilirubin, alamine_aminotransferase, ...}
        """
        model_data = {}
        
        # Map features
        for frontend_key, model_key in self.feature_mapping.items():
            if frontend_key in patient_data:
                model_data[model_key] = patient_data[frontend_key]
        
        return model_data
    
    def _prepare_supervised_input(self, patient_data: Dict[str, Any]) -> pd.DataFrame:
        """
        Prepare patient data for supervised model prediction.
        Converts to DataFrame with correct feature order and scaling.
        """
        # Convert to model format
        model_data = self._convert_frontend_to_model_format(patient_data)
        
        # Ensure gender is numeric (1 for Male, 0 for Female)
        if 'gender' in model_data:
            gender_val = model_data['gender']
            if isinstance(gender_val, str):
                model_data['gender'] = 1 if gender_val.lower() == 'male' else 0
        
        # Expected feature order (from training data)
        feature_order = [
            'age', 'gender', 'total_bilirubin', 'direct_bilirubin',
            'alkaline_phosphotase', 'alamine_aminotransferase',
            'aspartate_aminotransferase', 'total_protiens',
            'albumin', 'albumin_and_globulin_ratio'
        ]
        
        # Create DataFrame with correct column order
        patient_df = pd.DataFrame([model_data], columns=feature_order)
        
        # Scale using the same scaler from training
        patient_scaled = self.supervised_scaler.transform(patient_df)
        patient_scaled_df = pd.DataFrame(patient_scaled, columns=feature_order)
        
        return patient_scaled_df
    
    def _prepare_unsupervised_input(self, patient_data: Dict[str, Any]) -> np.ndarray:
        """
        Prepare patient data for unsupervised clustering.
        Extracts clustering features and applies scaling.
        """
        model_data = self._convert_frontend_to_model_format(patient_data)
        
        # Extract only clustering features
        clustering_data = {k: model_data.get(k, 0) for k in self.clustering_features}
        
        # Create DataFrame
        patient_df = pd.DataFrame([clustering_data], columns=self.clustering_features)
        
        # Scale using unsupervised scaler
        patient_scaled = self.unsupervised_scaler.transform(patient_df)
        
        return patient_scaled
    
    def predict_supervised(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Predict disease risk using supervised learning (XGBoost).
        
        Args:
            patient_data: Patient data dictionary (frontend format)
            
        Returns:
            Dictionary with prediction results:
            {
                "risk_probability": float (0-1),
                "prediction": str ("Disease Risk" or "No Disease Risk"),
                "confidence": str ("Low", "Medium", "High")
            }
        """
        if self.supervised_model is None:
            raise RuntimeError("Supervised model not loaded")
        
        # Prepare input
        patient_df = self._prepare_supervised_input(patient_data)
        
        # Predict probability
        probability = self.supervised_model.predict_proba(patient_df)[0, 1]
        
        # Binary prediction
        prediction = probability >= 0.5
        
        # Determine confidence
        if probability > 0.75 or probability < 0.25:
            confidence = "High"
        elif probability > 0.60 or probability < 0.40:
            confidence = "Medium"
        else:
            confidence = "Low"
        
        return {
            "risk_probability": float(probability),
            "prediction": "Disease Risk" if prediction else "No Disease Risk",
            "confidence": confidence
        }
    
    def predict_unsupervised(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Assign cluster using unsupervised learning (UMAP + KMeans).
        
        Args:
            patient_data: Patient data dictionary (frontend format)
            
        Returns:
            Dictionary with cluster analysis:
            {
                "cluster": int (0 or 1),
                "cluster_severity": str ("Low Risk" or "High Risk"),
                "distance_to_centroid": float,
                "similarity_score": float,
                "biomarker_profile": dict
            }
        """
        if self.umap_reducer is None or self.cluster_centroids is None:
            raise RuntimeError("Unsupervised models not loaded")
        
        # Prepare input
        patient_scaled = self._prepare_unsupervised_input(patient_data)
        
        # Apply UMAP projection
        patient_umap = self.umap_reducer.transform(patient_scaled)
        
        # Calculate distance to each centroid
        distances = [
            np.linalg.norm(patient_umap[0] - centroid)
            for centroid in self.cluster_centroids
        ]
        
        # Assign to closest cluster
        cluster = int(np.argmin(distances))
        distance_to_centroid = float(distances[cluster])
        
        # Map cluster to severity (cluster 1 typically has higher bilirubin = higher risk)
        # Based on unsupervised learning analysis: cluster 0 = low risk, cluster 1 = high risk
        cluster_severity = "High Risk" if cluster == 1 else "Low Risk"
        
        # Calculate similarity score (inverse of distance, normalized)
        max_distance = max(distances)
        similarity_score = float(1.0 - (distance_to_centroid / (max_distance + 1e-6)))
        
        # Extract biomarker profile
        model_data = self._convert_frontend_to_model_format(patient_data)
        biomarker_profile = {
            "total_bilirubin": model_data.get('total_bilirubin', 0),
            "direct_bilirubin": model_data.get('direct_bilirubin', 0),
            "alkaline_phosphatase": model_data.get('alkaline_phosphotase', 0),
            "alt": model_data.get('alamine_aminotransferase', 0),
            "ast": model_data.get('aspartate_aminotransferase', 0),
            "total_proteins": model_data.get('total_protiens', 0),
            "albumin": model_data.get('albumin', 0),
            "ag_ratio": model_data.get('albumin_and_globulin_ratio', 0)
        }
        
        return {
            "cluster": cluster,
            "cluster_severity": cluster_severity,
            "distance_to_centroid": distance_to_centroid,
            "similarity_score": similarity_score,
            "biomarker_profile": biomarker_profile
        }
    
    def explain_shap(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Calculate SHAP values for feature contribution analysis.
        
        Args:
            patient_data: Patient data dictionary (frontend format)
            
        Returns:
            Dictionary with SHAP explanations:
            {
                "shap_values": list[dict],  # Feature contributions
                "base_value": float,
                "top_contributing_factors": list[dict]
            }
        """
        if self.shap_explainer is None:
            raise RuntimeError("SHAP explainer not loaded")
        
        # Prepare input
        patient_df = self._prepare_supervised_input(patient_data)
        
        # Calculate SHAP values
        shap_values = self.shap_explainer.shap_values(patient_df)[0]  # Get for disease class
        
        # Get base value (expected value)
        base_value = float(self.shap_explainer.expected_value)
        
        # Get feature names
        feature_names = patient_df.columns.tolist()
        
        # Create SHAP contributions list
        shap_contributions = [
            {
                "feature": self.reverse_mapping.get(feat, feat),
                "shap_value": float(shap_val),
                "contribution": abs(float(shap_val)),
                "direction": "increases_risk" if shap_val > 0 else "decreases_risk"
            }
            for feat, shap_val in zip(feature_names, shap_values)
        ]
        
        # Sort by absolute contribution
        shap_contributions.sort(key=lambda x: x["contribution"], reverse=True)
        
        # Get top contributing factors (top 5)
        top_factors = shap_contributions[:5]
        
        # Map to frontend format
        top_factors_formatted = [
            {
                "feature": factor["feature"],
                "contribution": factor["contribution"],
                "shap_value": factor["shap_value"],
                "direction": factor["direction"]
            }
            for factor in top_factors
        ]
        
        return {
            "shap_values": shap_contributions,
            "base_value": base_value,
            "top_contributing_factors": top_factors_formatted
        }
    
    def predict_complete(self, patient_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Complete prediction combining all three methods:
        Supervised + Unsupervised + SHAP
        
        Args:
            patient_data: Patient data dictionary (frontend format)
            
        Returns:
            Dictionary with complete prediction results
        """
        results = {
            "supervised": None,
            "unsupervised": None,
            "shap": None
        }
        
        # Supervised prediction
        try:
            results["supervised"] = self.predict_supervised(patient_data)
        except Exception as e:
            print(f"Error in supervised prediction: {e}")
        
        # Unsupervised prediction
        try:
            results["unsupervised"] = self.predict_unsupervised(patient_data)
        except Exception as e:
            print(f"Error in unsupervised prediction: {e}")
        
        # SHAP explanation
        try:
            results["shap"] = self.explain_shap(patient_data)
        except Exception as e:
            print(f"Error in SHAP explanation: {e}")
        
        return results


# Example usage
if __name__ == "__main__":
    # Initialize predictor
    predictor = LiverDiseasePredictor()
    
    # Example patient data (frontend format)
    example_patient = {
        "age": 45,
        "gender": "Male",
        "totalBilirubin": 2.1,
        "directBilirubin": 0.8,
        "alkalinePhosphatase": 198,
        "sgptAlt": 89,
        "sgotAst": 112,
        "totalProteins": 6.8,
        "albumin": 3.2,
        "agRatio": 0.95
    }
    
    # Get complete prediction
    prediction = predictor.predict_complete(example_patient)
    
    print("\n" + "="*70)
    print("COMPLETE PREDICTION RESULTS")
    print("="*70)
    print("\nSupervised Prediction:")
    print(prediction["supervised"])
    print("\nUnsupervised Clustering:")
    print(prediction["unsupervised"])
    print("\nSHAP Top Factors:")
    for factor in prediction["shap"]["top_contributing_factors"]:
        print(f"  {factor['feature']}: {factor['shap_value']:.4f} ({factor['direction']})")
