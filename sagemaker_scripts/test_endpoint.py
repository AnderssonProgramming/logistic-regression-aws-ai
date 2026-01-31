#!/usr/bin/env python3
"""
Heart Disease Model - Test Endpoint Script
===========================================

⚠️  REQUIRES: Active deployed endpoint.
    Run deploy.py first, or use demo_deployment.py for local testing.

Run from SageMaker Code Editor terminal:
    python test_endpoint.py
"""

import boto3
import json

def predict_patient(patient_data, endpoint_name='heart-disease-prediction-endpoint'):
    """Send a prediction request to the deployed endpoint."""
    runtime_client = boto3.client('sagemaker-runtime')
    
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(patient_data)
    )
    
    return json.loads(response['Body'].read().decode())

def main():
    print("=" * 70)
    print("🧪 HEART DISEASE PREDICTION - ENDPOINT TEST")
    print("=" * 70)
    
    # Test cases
    test_cases = [
        {
            "name": "High-Risk Patient",
            "data": {
                "Age": 65,
                "Sex": 1,
                "Chest pain type": 4,
                "BP": 160,
                "Cholesterol": 320,
                "Max HR": 120,
                "ST depression": 2.5,
                "Number of vessels fluro": 2
            }
        },
        {
            "name": "Low-Risk Patient",
            "data": {
                "Age": 35,
                "Sex": 0,
                "Chest pain type": 1,
                "BP": 120,
                "Cholesterol": 180,
                "Max HR": 175,
                "ST depression": 0,
                "Number of vessels fluro": 0
            }
        },
        {
            "name": "Borderline Patient",
            "data": {
                "Age": 55,
                "Sex": 1,
                "Chest pain type": 2,
                "BP": 135,
                "Cholesterol": 240,
                "Max HR": 150,
                "ST depression": 1.0,
                "Number of vessels fluro": 1
            }
        }
    ]
    
    for test in test_cases:
        print(f"\n📋 {test['name']}")
        print("-" * 50)
        
        try:
            result = predict_patient(test['data'])
            print(f"   Age: {test['data']['Age']}, Cholesterol: {test['data']['Cholesterol']}")
            print(f"   Probability: {result['probability']:.2%}")
            print(f"   Risk Level: {result['risk_level']}")
            print(f"   Diagnosis: {'Heart Disease ⚠️' if result['has_heart_disease'] else 'No Heart Disease ✅'}")
        except Exception as e:
            print(f"   ❌ Error: {e}")
    
    print("\n" + "=" * 70)
    print("✅ TEST COMPLETE!")
    print("=" * 70)

if __name__ == "__main__":
    main()
