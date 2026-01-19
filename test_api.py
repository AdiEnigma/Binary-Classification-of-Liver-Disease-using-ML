"""Test API endpoints"""
import requests
import time
import json

print("=" * 70)
print("Testing Backend API")
print("=" * 70)

base_url = "http://localhost:8000"

# Wait for server to be ready
print("\nWaiting for server to start...")
for i in range(10):
    try:
        response = requests.get(f"{base_url}/api/health", timeout=2)
        if response.status_code == 200:
            print(f"   [OK] Server is ready!")
            break
    except requests.exceptions.RequestException:
        if i < 9:
            time.sleep(1)
        else:
            print("   [ERROR] Server not responding. Is it running?")
            print("   Start server with: cd backend && python -m uvicorn main:app --reload")
            exit(1)

# Test 1: Root endpoint
print("\n1. Testing root endpoint...")
try:
    response = requests.get(f"{base_url}/")
    print(f"   [OK] Status: {response.status_code}")
    data = response.json()
    print(f"   [OK] API: {data.get('message')}")
    print(f"   [OK] Version: {data.get('version')}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 2: Health check
print("\n2. Testing health check endpoint...")
try:
    response = requests.get(f"{base_url}/api/health")
    print(f"   [OK] Status: {response.status_code}")
    data = response.json()
    print(f"   [OK] Status: {data.get('status')}")
    print(f"   [OK] Models loaded: {data.get('models_loaded')}")
except Exception as e:
    print(f"   [ERROR] {e}")

# Test 3: Individual prediction (will fail if models not loaded)
print("\n3. Testing individual prediction endpoint...")
test_patient = {
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

try:
    response = requests.post(
        f"{base_url}/api/predict/individual",
        json=test_patient,
        headers={"Content-Type": "application/json"}
    )
    print(f"   [OK] Status: {response.status_code}")
    if response.status_code == 200:
        data = response.json()
        print(f"   [OK] Prediction successful!")
        if 'supervised' in data:
            supervised = data['supervised']
            print(f"   [OK] Risk probability: {supervised.get('risk_probability', 0):.2%}")
            print(f"   [OK] Prediction: {supervised.get('prediction')}")
            print(f"   [OK] Confidence: {supervised.get('confidence')}")
    else:
        print(f"   [WARNING] Response: {response.text[:200]}")
except Exception as e:
    print(f"   [ERROR] {e}")
    print("   [INFO] This is expected if models are not trained yet.")

print("\n" + "=" * 70)
print("API Test Complete!")
print("=" * 70)
print("\nNote: If predictions fail, train models first:")
print("   python _data_preparation.py")
print("   python _supervised_learning.py")
print("   python _unsupervised_learning1.py")
