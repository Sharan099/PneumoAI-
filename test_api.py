"""
Simple test script for the FastAPI endpoint
"""

import requests
from pathlib import Path
import json

API_URL = "http://localhost:8000"

def test_health():
    """Test health endpoint"""
    print("Testing /health endpoint...")
    response = requests.get(f"{API_URL}/health")
    print(f"Status: {response.status_code}")
    print(f"Response: {json.dumps(response.json(), indent=2)}")
    return response.status_code == 200

def test_predict(image_path: Path):
    """Test /predict endpoint"""
    if not image_path.exists():
        print(f"Image not found: {image_path}")
        return False
    
    print(f"\nTesting /predict endpoint with {image_path.name}...")
    
    with open(image_path, 'rb') as f:
        files = {'file': (image_path.name, f, 'image/jpeg')}
        response = requests.post(f"{API_URL}/predict", files=files)
    
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        result = response.json()
        print(f"Response: {json.dumps(result, indent=2)}")
        return True
    else:
        print(f"Error: {response.text}")
        return False

def main():
    """Main test function"""
    print("=" * 50)
    print("FastAPI Endpoint Tests")
    print("=" * 50)
    
    # Test health
    if not test_health():
        print("\n⚠ Health check failed. Is the server running?")
        print("Start the server with: python app.py")
        return
    
    # Test predict with a sample image
    base_dir = Path(__file__).parent
    test_dir = base_dir / "chest-xray-pneumonia" / "chest_xray" / "chest_xray" / "test"
    
    # Try to find a test image
    test_image = None
    for class_dir in ['NORMAL', 'PNEUMONIA']:
        class_path = test_dir / class_dir
        if class_path.exists():
            images = list(class_path.glob("*.jpeg"))
            if images:
                test_image = images[0]
                break
    
    if test_image:
        test_predict(test_image)
    else:
        print("\n⚠ No test images found. Skipping prediction test.")
    
    print("\n" + "=" * 50)
    print("Tests completed!")
    print("=" * 50)

if __name__ == "__main__":
    main()



