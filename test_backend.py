"""Test script to verify backend setup"""
import sys
import os

print("=" * 70)
print("Testing Backend Setup")
print("=" * 70)

# Test 1: Import predict.py
print("\n1. Testing predict.py import...")
try:
    from predict import LiverDiseasePredictor
    print("   [OK] predict.py imports successfully")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test 2: Import backend modules
print("\n2. Testing backend module imports...")
try:
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from backend.models.schemas import PatientDataRequest, CompletePredictionResponse
    print("   [OK] Backend schemas import successfully")
    print(f"   [OK] PatientDataRequest has {len(PatientDataRequest.model_fields)} fields")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test 3: Test FastAPI app
print("\n3. Testing FastAPI app...")
try:
    from backend.main import app
    print("   [OK] FastAPI app imports successfully")
    print(f"   [OK] App title: {app.title}")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

# Test 4: Check saved_models directory
print("\n4. Checking saved_models directory...")
saved_models_path = os.path.join(os.path.dirname(__file__), "saved_models")
if os.path.exists(saved_models_path):
    files = os.listdir(saved_models_path)
    print(f"   [OK] Directory exists with {len(files)} files:")
    for f in files[:5]:
        print(f"     - {f}")
    if len(files) == 0:
        print("   [WARNING] No model files found. Run training scripts first.")
else:
    print("   [WARNING] Directory doesn't exist. It will be created when models are saved.")

# Test 5: Test Pydantic schema validation
print("\n5. Testing Pydantic schema validation...")
try:
    test_data = {
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
    patient = PatientDataRequest(**test_data)
    print("   [OK] PatientDataRequest validation works")
    print(f"   [OK] Validated patient age: {patient.age}, gender: {patient.gender}")
except Exception as e:
    print(f"   [ERROR] Error: {e}")

print("\n" + "=" * 70)
print("Backend Setup Test Complete!")
print("=" * 70)
print("\nNext steps:")
print("1. Train models: python _data_preparation.py && python _supervised_learning.py && python _unsupervised_learning1.py")
print("2. Start backend: cd backend && python -m uvicorn main:app --reload")
print("3. Test API: Visit http://localhost:8000/docs for interactive API documentation")
