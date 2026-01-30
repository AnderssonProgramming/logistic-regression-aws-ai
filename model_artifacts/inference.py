"""
SageMaker Inference Script for Heart Disease Prediction Model
This script handles model loading and inference for the deployed endpoint.
"""

import json
import numpy as np
import os

def model_fn(model_dir):
    """
    Load model artifacts from the model directory.
    This function is called once when the endpoint starts.

    Parameters:
    -----------
    model_dir : str
        Path to the directory containing model artifacts

    Returns:
    --------
    dict containing model components
    """
    print(f"Loading model from: {model_dir}")

    # Load weights and bias
    weights = np.load(os.path.join(model_dir, 'weights.npy'))
    bias = np.load(os.path.join(model_dir, 'bias.npy'))[0]

    # Load normalization parameters
    feature_mean = np.load(os.path.join(model_dir, 'feature_mean.npy'))
    feature_std = np.load(os.path.join(model_dir, 'feature_std.npy'))

    # Load metadata
    with open(os.path.join(model_dir, 'model_metadata.json'), 'r') as f:
        metadata = json.load(f)

    model = {
        'weights': weights,
        'bias': bias,
        'feature_mean': feature_mean,
        'feature_std': feature_std,
        'feature_names': metadata['features'],
        'metadata': metadata
    }

    print(f"Model loaded successfully. Features: {metadata['features']}")
    return model


def input_fn(request_body, request_content_type):
    """
    Deserialize and prepare the input data for inference.

    Parameters:
    -----------
    request_body : str
        The request payload
    request_content_type : str
        The content type of the request

    Returns:
    --------
    dict with patient data
    """
    if request_content_type == 'application/json':
        input_data = json.loads(request_body)
        return input_data
    else:
        raise ValueError(f"Unsupported content type: {request_content_type}")


def predict_fn(input_data, model):
    """
    Make prediction using the loaded model.

    Parameters:
    -----------
    input_data : dict
        Patient data with feature values
    model : dict
        Loaded model components

    Returns:
    --------
    dict with prediction results
    """
    # Extract features in correct order
    feature_names = model['feature_names']
    features = np.array([input_data[feat] for feat in feature_names])

    # Normalize
    features_normalized = (features - model['feature_mean']) / model['feature_std']

    # Sigmoid prediction
    z = np.dot(features_normalized, model['weights']) + model['bias']
    probability = 1 / (1 + np.exp(-z))

    # Classification
    threshold = 0.5
    prediction = 1 if probability >= threshold else 0

    # Risk level
    if probability < 0.3:
        risk_level = "Low"
    elif probability < 0.5:
        risk_level = "Moderate"
    elif probability < 0.7:
        risk_level = "High"
    else:
        risk_level = "Very High"

    return {
        'probability': float(probability),
        'prediction': int(prediction),
        'risk_level': risk_level,
        'has_heart_disease': bool(prediction),
        'input_features': input_data
    }


def output_fn(prediction, accept):
    """
    Serialize the prediction result.

    Parameters:
    -----------
    prediction : dict
        The prediction result
    accept : str
        The accept content type

    Returns:
    --------
    tuple of (response body, content type)
    """
    if accept == 'application/json':
        return json.dumps(prediction), 'application/json'
    else:
        raise ValueError(f"Unsupported accept type: {accept}")
