"""
REST API routes for liver disease prediction
"""

from fastapi import APIRouter, HTTPException, UploadFile, File
from typing import List
import pandas as pd
import io
from models.schemas import (
    PatientDataRequest,
    CompletePredictionResponse,
    BulkPredictionResponse,
    BulkPredictionResult,
    ErrorResponse
)
from utils.predictor import predict_complete


router = APIRouter(prefix="/api/predict", tags=["predictions"])


@router.post("/individual", response_model=CompletePredictionResponse)
async def predict_individual(patient_data: PatientDataRequest):
    """
    Predict disease risk for a single patient.
    
    Returns:
        Complete prediction with supervised, unsupervised, and SHAP results
    """
    try:
        # Convert Pydantic model to dict
        patient_dict = patient_data.model_dump()
        
        # Get prediction
        result = predict_complete(patient_dict)
        
        # Build response
        response = CompletePredictionResponse(
            success=True,
            patient_id=patient_data.patientId,
            supervised=result["supervised"],
            unsupervised=result.get("unsupervised"),
            shap=result.get("shap")
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {str(e)}"
        )


@router.post("/bulk", response_model=BulkPredictionResponse)
async def predict_bulk(file: UploadFile = File(...)):
    """
    Predict disease risk for multiple patients from CSV file.
    
    Expected CSV columns:
    - age, gender, totalBilirubin, directBilirubin, alkalinePhosphatase,
    - sgptAlt, sgotAst, totalProteins, albumin, agRatio
    - Optional: patientId, name
    
    Returns:
        Bulk prediction results with summary statistics
    """
    try:
        # Read CSV file
        contents = await file.read()
        df = pd.read_csv(io.BytesIO(contents))
        
        # Convert DataFrame to list of dictionaries
        patients = df.to_dict('records')
        
        # Process each patient
        results = []
        for idx, patient_row in enumerate(patients):
            try:
                # Map CSV columns to PatientData format
                patient_dict = {
                    "age": float(patient_row.get("age", 0)),
                    "gender": str(patient_row.get("gender", "")).strip(),
                    "totalBilirubin": float(patient_row.get("totalBilirubin", 0)),
                    "directBilirubin": float(patient_row.get("directBilirubin", 0)),
                    "alkalinePhosphatase": float(patient_row.get("alkalinePhosphatase", 0)),
                    "sgptAlt": float(patient_row.get("sgptAlt", 0)),
                    "sgotAst": float(patient_row.get("sgotAst", 0)),
                    "totalProteins": float(patient_row.get("totalProteins", 0)),
                    "albumin": float(patient_row.get("albumin", 0)),
                    "agRatio": float(patient_row.get("agRatio", 0)),
                }
                
                # Optional fields
                if "patientId" in patient_row:
                    patient_dict["patientId"] = str(patient_row["patientId"])
                if "name" in patient_row:
                    patient_dict["name"] = str(patient_row["name"])
                
                # Get prediction
                result = predict_complete(patient_dict)
                
                # Build result object
                bulk_result = BulkPredictionResult(
                    patient_id=patient_dict.get("patientId"),
                    name=patient_dict.get("name"),
                    supervised=result["supervised"],
                    unsupervised=result.get("unsupervised"),
                    shap=result.get("shap")
                )
                
                results.append(bulk_result)
                
            except Exception as e:
                # Skip invalid rows
                print(f"Error processing row {idx}: {e}")
                continue
        
        # Calculate summary statistics
        if results:
            high_risk_count = sum(1 for r in results if r.supervised.risk_probability > 0.7)
            medium_risk_count = sum(1 for r in results if 0.3 <= r.supervised.risk_probability <= 0.7)
            low_risk_count = sum(1 for r in results if r.supervised.risk_probability < 0.3)
            avg_probability = sum(r.supervised.risk_probability for r in results) / len(results)
            
            summary = {
                "high_risk": high_risk_count,
                "medium_risk": medium_risk_count,
                "low_risk": low_risk_count,
                "average_probability": round(avg_probability, 3),
                "total_processed": len(results)
            }
        else:
            summary = {
                "high_risk": 0,
                "medium_risk": 0,
                "low_risk": 0,
                "average_probability": 0.0,
                "total_processed": 0
            }
        
        # Build response
        response = BulkPredictionResponse(
            success=True,
            total_patients=len(results),
            results=results,
            summary=summary
        )
        
        return response
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Bulk prediction failed: {str(e)}"
        )
