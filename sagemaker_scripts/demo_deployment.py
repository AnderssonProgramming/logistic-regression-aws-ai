#!/usr/bin/env python3
"""
Heart Disease Model - Deployment Demo
======================================

✅ USE THIS SCRIPT IN LEARNER LABS!
   This script demonstrates the full deployment process without creating
   an actual endpoint (which Learner Labs block).

Run from SageMaker Code Editor terminal:
    python demo_deployment.py
"""
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import json
import numpy as np
import logging

# Suppress INFO messages
logging.getLogger('sagemaker.config').setLevel(logging.WARNING)

def main():
    print("=" * 70)
    print("🚀 HEART DISEASE MODEL - DEPLOYMENT DEMO")
    print("=" * 70)
    print("\n⚠️  Note: This demo runs without creating a real endpoint")
    print("   (Learner Lab restricts sagemaker:CreateEndpointConfig)")

    # Step 1: Initialize
    print("\n📦 Step 1: Initializing SageMaker session...")
    sagemaker_session = sagemaker.Session()
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    role = sagemaker.get_execution_role()
    print(f"   ✅ Region: {region}")
    print(f"   ✅ Bucket: {bucket}")
    print(f"   ✅ Role: {role[:50]}...")

    # Step 2: Upload to S3
    print("\n📤 Step 2: Uploading model.tar.gz to S3...")
    s3_model_path = sagemaker_session.upload_data(
        path='model.tar.gz',
        bucket=bucket,
        key_prefix='heart-disease-model'
    )
    print(f"   ✅ S3 Path: {s3_model_path}")

    # Step 3: Create Model object
    print("\n🔧 Step 3: Creating SageMaker Model object...")
    model = SKLearnModel(
        model_data=s3_model_path,
        role=role,
        entry_point='inference.py',
        source_dir='model_artifacts',
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )
    print("   ✅ Model object created successfully")

    # Step 4: Show deployment config
    print("\n🌐 Step 4: Deployment Configuration")
    print("   " + "-" * 50)
    print("   📋 Endpoint Name: heart-disease-prediction-endpoint")
    print("   📋 Instance Type: ml.t2.medium")
    print("   📋 Instance Count: 1")
    print(f"   📋 Model Data: {s3_model_path}")
    print("   📋 Framework: scikit-learn 1.2-1")
    print("   " + "-" * 50)
    print("   ❌ Skipping actual deployment (Lab restriction)")

    # Step 5: Local inference test
    print("\n🧪 Step 5: Testing inference LOCALLY (simulating endpoint)...")
    
    # Load model artifacts
    weights = np.load('model_artifacts/weights.npy')
    bias = np.load('model_artifacts/bias.npy')[0]
    feat_mean = np.load('model_artifacts/feature_mean.npy')
    feat_std = np.load('model_artifacts/feature_std.npy')
    
    with open('model_artifacts/model_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    feature_names = metadata['features']
    
    # Test cases
    test_cases = [
        {
            "name": "High-Risk Patient",
            "data": {"Age": 65, "Sex": 1, "Chest pain type": 4, "BP": 160,
                     "Cholesterol": 320, "Max HR": 120, "ST depression": 2.5,
                     "Number of vessels fluro": 2}
        },
        {
            "name": "Low-Risk Patient", 
            "data": {"Age": 35, "Sex": 0, "Chest pain type": 1, "BP": 120,
                     "Cholesterol": 180, "Max HR": 175, "ST depression": 0,
                     "Number of vessels fluro": 0}
        },
        {
            "name": "Test Patient",
            "data": {"Age": 60, "Sex": 1, "Chest pain type": 3, "BP": 145,
                     "Cholesterol": 280, "Max HR": 140, "ST depression": 1.5,
                     "Number of vessels fluro": 1}
        }
    ]
    
    for test in test_cases:
        patient = test['data']
        features = np.array([patient[f] for f in feature_names])
        features_norm = (features - feat_mean) / feat_std
        z = np.dot(features_norm, weights) + bias
        probability = float(1 / (1 + np.exp(-z)))
        
        if probability < 0.3:
            risk = "Low"
        elif probability < 0.5:
            risk = "Moderate"
        elif probability < 0.7:
            risk = "High"
        else:
            risk = "Very High"
        
        diagnosis = "Heart Disease ⚠️" if probability >= 0.5 else "No Heart Disease ✅"
        
        print(f"\n   📋 {test['name']}")
        print(f"      Age: {patient['Age']}, Cholesterol: {patient['Cholesterol']}")
        print(f"      Probability: {probability:.2%}")
        print(f"      Risk Level: {risk}")
        print(f"      Diagnosis: {diagnosis}")

    # Summary
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT DEMO COMPLETE!")
    print("=" * 70)
    print("\n📊 Summary:")
    print("   ┌─────────────────────────────────┬────────┐")
    print("   │ Component                       │ Status │")
    print("   ├─────────────────────────────────┼────────┤")
    print("   │ SageMaker Session               │   ✅   │")
    print("   │ Model uploaded to S3            │   ✅   │")
    print("   │ SageMaker Model object          │   ✅   │")
    print("   │ Inference script (inference.py) │   ✅   │")
    print("   │ Local inference test            │   ✅   │")
    print("   │ Real endpoint deployment        │   ❌   │")
    print("   └─────────────────────────────────┴────────┘")
    print("\n💡 Why endpoint deployment failed:")
    print("   The Learner Lab policy explicitly denies:")
    print("   • sagemaker:CreateEndpointConfig")
    print("   • sagemaker:CreateEndpoint")
    print("\n🎯 In a full AWS account or different Lab, run:")
    print("   predictor = model.deploy(")
    print("       initial_instance_count=1,")
    print("       instance_type='ml.t2.medium',")
    print("       endpoint_name='heart-disease-prediction-endpoint'")
    print("   )")
    print("\n📁 All artifacts are ready in S3:")
    print(f"   {s3_model_path}")

if __name__ == "__main__":
    main()
