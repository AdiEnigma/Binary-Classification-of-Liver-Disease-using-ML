"""
Pydantic schemas for request/response validation
Matches frontend TypeScript interfaces
"""

from pydantic import BaseModel, Field
from typing import List, Optional, Literal
from datetime import datetime


# Request Models
class PatientDataRequest(BaseModel):
    """Patient data from frontend"""
    patientId: Optional[str] = None
    name: Optional[str] = None
    age: int = Field(ge=0, le=120, description="Patient age in years")
    gender: Literal["Male", "Female", ""] = Field(description="Patient gender")
    
    # Clinical parameters
    totalBilirubin: float = Field(ge=0.0, description="Total bilirubin in mg/dL")
    directBilirubin: float = Field(ge=0.0, description="Direct bilirubin in mg/dL")
    alkalinePhosphatase: float = Field(ge=0.0, description="Alkaline phosphatase in IU/L")
    sgptAlt: float = Field(ge=0.0, description="SGPT/ALT in IU/L")
    sgotAst: float = Field(ge=0.0, description="SGOT/AST in IU/L")
    totalProteins: float = Field(ge=0.0, description="Total proteins in g/dL")
    albumin: float = Field(ge=0.0, description="Albumin in g/dL")
    agRatio: float = Field(ge=0.0, description="Albumin/Globulin ratio")
    
    class Config:
        json_schema_extra = {
            "example": {
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
        }


# Response Models
class SupervisedPrediction(BaseModel):
    """Supervised learning prediction result"""
    risk_probability: float = Field(description="Disease risk probability (0-1)")
    prediction: str = Field(description="Prediction label")
    confidence: Literal["Low", "Medium", "High"] = Field(description="Confidence level")


class ClusterAnalysis(BaseModel):
    """Unsupervised clustering analysis result"""
    cluster: int = Field(description="Assigned cluster (0 or 1)")
    cluster_severity: str = Field(description="Risk severity based on cluster")
    distance_to_centroid: float = Field(description="Distance to cluster centroid")
    similarity_score: float = Field(description="Similarity score to cluster")
    biomarker_profile: dict = Field(description="Biomarker values")


class SHAPContribution(BaseModel):
    """SHAP feature contribution"""
    feature: str
    shap_value: float
    contribution: float
    direction: str  # "increases_risk" or "decreases_risk"


class SHAPExplanation(BaseModel):
    """SHAP explainability results"""
    shap_values: List[dict] = Field(description="All feature SHAP values")
    base_value: float = Field(description="Base expected value")
    top_contributing_factors: List[dict] = Field(description="Top 5 contributing factors")


class CompletePredictionResponse(BaseModel):
    """Complete prediction response combining all methods"""
    success: bool = True
    patient_id: Optional[str] = None
    supervised: SupervisedPrediction
    unsupervised: Optional[ClusterAnalysis] = None
    shap: Optional[SHAPExplanation] = None
    timestamp: datetime = Field(default_factory=datetime.now)
    
    class Config:
        json_encoders = {
            datetime: lambda v: v.isoformat()
        }


class BulkPredictionResult(BaseModel):
    """Single result in bulk prediction"""
    patient_id: Optional[str] = None
    name: Optional[str] = None
    # Include original patient data
    age: int
    gender: str
    totalBilirubin: float
    directBilirubin: float
    alkalinePhosphatase: float
    sgptAlt: float
    sgotAst: float
    totalProteins: float
    albumin: float
    agRatio: float
    # Prediction results
    supervised: SupervisedPrediction
    unsupervised: Optional[ClusterAnalysis] = None
    shap: Optional[SHAPExplanation] = None


class BulkPredictionResponse(BaseModel):
    """Bulk prediction response"""
    success: bool = True
    total_patients: int
    results: List[BulkPredictionResult]
    summary: dict = Field(description="Summary statistics")
    timestamp: datetime = Field(default_factory=datetime.now)


class ErrorResponse(BaseModel):
    """Error response"""
    success: bool = False
    error: str
    detail: Optional[str] = None
