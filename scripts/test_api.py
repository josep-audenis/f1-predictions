import requests
import json
from datetime import datetime

BASE_URL = "http://localhost:8000"
API_URL = f"{BASE_URL}/api/v1"

def print_section(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}\n")

def test_endpoint(method, endpoint, description):
    print(f"Testing: {description}")
    print(f"Endpoint: {method} {endpoint}")
    
    try:
        if method == "GET":
            response = requests.get(f"{API_URL}{endpoint}")
        elif method == "POST":
            response = requests.post(f"{API_URL}{endpoint}")
        
        print(f"Status: {response.status_code}")
        
        if response.status_code == 200:
            data = response.json()
            print(f"Response preview: {json.dumps(data, indent=2)[:200]}...")
            print("SUCCESS\n")
            return True
        else:
            print(f"FAILED: {response.text}\n")
            return False
            
    except Exception as e:
        print(f"ERROR: {str(e)}\n")
        return False

def main():
    print_section("F1 Predictions API Test Suite")
    
    print_section("Basic Endpoints")
    
    try:
        response = requests.get(f"{BASE_URL}/health")
        print(f"Health Check: {response.json()}")
        print("SUCCESS\n")
    except Exception as e:
        print(f"Health check failed: {e}\n")
    
    print_section("Prediction Endpoints")
    test_endpoint("GET", "/predictions/next-race", "Get next race prediction")
    
    print_section("Race Endpoints")
    current_year = datetime.now().year
    test_endpoint("GET", f"/races/calendar/{current_year}", "Get race calendar")
    test_endpoint("GET", "/races/next", "Get next race info")
    test_endpoint("GET", "/races/status", "Get next race status")
    
    print_section("Model Endpoints")
    # test_endpoint("GET", "/models/best", "Get best models")
    test_endpoint("GET", "/models/available", "Get available models")
    test_endpoint("GET", "/models/performance/top1-quali", "Get top1-quali performance")
    test_endpoint("GET", "/models/performance/top1-pre-quali", "Get top1-pre-quali performance")
    
    
if __name__ == "__main__":
    main()
