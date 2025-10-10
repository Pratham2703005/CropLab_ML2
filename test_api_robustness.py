#!/usr/bin/env python3
"""
Test script for API robustness verification
Tests all endpoints with valid and edge case scenarios
"""

import requests
import json
import time

def test_api_endpoints():
    base_url = "http://localhost:8000"
    
    # Test coordinates from your original example
    test_coordinates = [
        [77.2090, 28.6139],  # Delhi area
        [77.2100, 28.6149],
        [77.2110, 28.6159],
        [77.2120, 28.6169]
    ]
    
    print("🧪 Testing API Robustness...")
    
    # Test 1: Health Check
    try:
        response = requests.get(f"{base_url}/")
        print(f"✅ Health check: {response.status_code} - {response.json()}")
    except Exception as e:
        print(f"❌ Health check failed: {e}")
    
    # Test 2: Predict endpoint
    try:
        predict_payload = {
            "coordinates": test_coordinates,
            "t1": 2000,
            "t2": 3000
        }
        response = requests.post(f"{base_url}/predict", json=predict_payload, timeout=30)
        print(f"✅ Predict endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Predicted yield: {data.get('predicted_yield', 'N/A')}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Predict endpoint failed: {e}")
    
    # Test 3: Generate heatmap endpoint
    try:
        heatmap_payload = {
            "coordinates": test_coordinates,
            "t1": 2000,
            "t2": 3000
        }
        response = requests.post(f"{base_url}/generate_heatmap", json=heatmap_payload, timeout=60)
        print(f"✅ Heatmap endpoint: {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Predicted yield: {data.get('predicted_yield', 'N/A')}")
            print(f"   Has heatmap data: {'heatmap_data' in data}")
            print(f"   Suggestions count: {len(data.get('suggestions', []))}")
        else:
            print(f"   Response: {response.text}")
    except Exception as e:
        print(f"❌ Heatmap endpoint failed: {e}")
    
    # Test 4: Edge case - Empty coordinates
    try:
        edge_payload = {
            "coordinates": [],
            "t1": 2000,
            "t2": 3000
        }
        response = requests.post(f"{base_url}/predict", json=edge_payload, timeout=30)
        print(f"✅ Edge case (empty coords): {response.status_code}")
    except Exception as e:
        print(f"❌ Edge case test failed: {e}")
    
    # Test 5: Edge case - Invalid coordinates
    try:
        edge_payload = {
            "coordinates": [[999, 999]],
            "t1": 2000,
            "t2": 3000
        }
        response = requests.post(f"{base_url}/predict", json=edge_payload, timeout=30)
        print(f"✅ Edge case (invalid coords): {response.status_code}")
        if response.status_code == 200:
            data = response.json()
            print(f"   Fallback yield: {data.get('predicted_yield', 'N/A')}")
    except Exception as e:
        print(f"❌ Invalid coords test failed: {e}")

if __name__ == "__main__":
    print("🚀 Starting API robustness tests...")
    print("Make sure the server is running on localhost:8000")
    print("-" * 50)
    test_api_endpoints()
    print("-" * 50)
    print("✨ API robustness testing completed!")