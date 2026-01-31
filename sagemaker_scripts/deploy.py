#!/usr/bin/env python3
"""
Heart Disease Model - SageMaker Deployment Script
==================================================

⚠️  REQUIRES: Full AWS account with sagemaker:CreateEndpoint permissions.
    Learner Labs typically BLOCK endpoint creation.

Run from SageMaker Code Editor terminal:
    python deploy.py
"""

import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import json
import logging

# Suppress INFO messages from sagemaker.config
logging.getLogger('sagemaker.config').setLevel(logging.WARNING)

def main():
    print("=" * 70)
    print("🚀 HEART DISEASE MODEL - SAGEMAKER DEPLOYMENT")
    print("=" * 70)
    
    # Step 1: Initialize SageMaker session
    print("\n📦 Step 1: Initializing SageMaker session...")
    sagemaker_session = sagemaker.Session()
    region = sagemaker_session.boto_region_name
    bucket = sagemaker_session.default_bucket()
    role = sagemaker.get_execution_role()
    
    print(f"   ✅ Region: {region}")
    print(f"   ✅ Bucket: {bucket}")
    print(f"   ✅ Role: {role[:50]}...")
    
    # Step 2: Upload model to S3
    print("\n📤 Step 2: Uploading model.tar.gz to S3...")
    s3_model_path = sagemaker_session.upload_data(
        path='model.tar.gz',
        bucket=bucket,
        key_prefix='heart-disease-model'
    )
    print(f"   ✅ S3 Path: {s3_model_path}")
    
    # Step 3: Create SageMaker Model
    print("\n🔧 Step 3: Creating SageMaker Model...")
    model = SKLearnModel(
        model_data=s3_model_path,
        role=role,
        entry_point='inference.py',
        source_dir='model_artifacts',
        framework_version='1.2-1',
        py_version='py3',
        sagemaker_session=sagemaker_session
    )
    print("   ✅ Model created")
    
    # Step 4: Deploy to endpoint
    print("\n🌐 Step 4: Deploying to real-time endpoint...")
    print("   ⏳ This may take 3-5 minutes...")
    
    endpoint_name = 'heart-disease-prediction-endpoint'
    
    predictor = model.deploy(
        initial_instance_count=1,
        instance_type='ml.t2.medium',  # t2.medium is valid for inference endpoints
        endpoint_name=endpoint_name
    )
    
    print(f"\n   ✅ Endpoint deployed: {endpoint_name}")
    print(f"   ✅ Endpoint ARN: arn:aws:sagemaker:{region}:endpoint/{endpoint_name}")
    
    # Step 5: Test the endpoint
    print("\n🧪 Step 5: Testing endpoint with sample patient...")
    
    test_patient = {
        "Age": 60,
        "Sex": 1,
        "Chest pain type": 3,
        "BP": 145,
        "Cholesterol": 280,
        "Max HR": 140,
        "ST depression": 1.5,
        "Number of vessels fluro": 1
    }
    
    runtime_client = boto3.client('sagemaker-runtime')
    
    response = runtime_client.invoke_endpoint(
        EndpointName=endpoint_name,
        ContentType='application/json',
        Body=json.dumps(test_patient)
    )
    
    result = json.loads(response['Body'].read().decode())
    
    print("\n   🏥 Heart Disease Risk Prediction")
    print("   " + "=" * 50)
    print(f"   Patient: Age={test_patient['Age']}, Cholesterol={test_patient['Cholesterol']}")
    print(f"   Probability: {result['probability']:.2%}")
    print(f"   Risk Level: {result['risk_level']}")
    print(f"   Diagnosis: {'Heart Disease ⚠️' if result['has_heart_disease'] else 'No Heart Disease ✅'}")
    
    print("\n" + "=" * 70)
    print("✅ DEPLOYMENT COMPLETE!")
    print("=" * 70)
    print(f"\n📌 Endpoint Name: {endpoint_name}")
    print("\n⚠️  IMPORTANT: Run 'python cleanup.py' when done to delete the endpoint!")
    print("   This will prevent ongoing AWS charges.")
    
    return endpoint_name

if __name__ == "__main__":
    main()
