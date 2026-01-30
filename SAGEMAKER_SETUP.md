# AWS SageMaker Setup Guide

This document provides a detailed step-by-step guide for setting up AWS SageMaker to run the notebooks in this project.

## Prerequisites

- AWS Account with access to Amazon SageMaker
- LabRole IAM role configured

## Supported Configuration

| Resource | Limit |
|----------|-------|
| **Supported instance types** | ml.t3.medium, ml.t3.large, ml.t3.xlarge, ml.m5.large, ml.m5.xlarge, ml.c5.large, ml.c5.xlarge |
| **Maximum SageMaker Notebooks** | 2 |
| **Maximum SageMaker Apps** | 2 |

---

## Step 1: Create a SageMaker Domain

1. Navigate to **Amazon SageMaker** in the AWS Console
2. Choose **Domains** and then choose **Create domain**
3. Choose **Set up for organizations** and click **Set up**

### Step 1.1: Set up Domain Details & Users

- Give the domain a name (e.g., `myDomain`)
- Keep the **Login through IAM** default
- Leave **Who will use SageMaker** blank
- Choose **Next**

### Step 1.2: Configure Roles and ML Activities

- Choose **Use an existing role**
- Set the **Default execution role** to `LabRole`
- Choose **Next**

### Step 1.3: Configure Applications

> **Note:** Ignore and close any message about `servicequotas:RequestServiceQuotaIncrease` permissions issues.

#### SageMaker Studio Panel
- Choose **SageMaker Studio - New**

#### JupyterLab Panel
- Choose **Enable idle shutdown** and set it to **60 minutes**
- Choose **Allow users to set custom idle shutdown time** and set the Maximum to **600**

#### Canvas Panel
1. Choose **Configure Canvas**
2. Scroll down to **Canvas Ready-to-use models configuration**
3. Choose **Use an existing execution role**
4. For **Execution role name**, choose **Enter a custom IAM role ARN**
5. For **Custom IAM role ARN**, paste the ARN of LabRole:
   ```
   arn:aws:iam::ACCOUNT_ID:role/LabRole
   ```
   > Replace `ACCOUNT_ID` with your actual AWS Account ID

#### CodeEditor Panel
- Choose **Enable idle shutdown** and set it to **60 minutes**
- Choose **Allow users to set custom idle shutdown time** and set the Maximum to **600**

### Step 1.4: Customize Studio UI (Optional)

- Scroll to the bottom and choose **Next**

### Step 1.5: Set up Network Settings

Choose either **VPC only** or **Public internet access**:

**Example using VPC Only:**
1. Choose **VPC Only**
2. For **VPC for Studio to use**: Choose the **Default VPC**
3. Choose the **VPC Console link** (opens in a new browser tab)
4. In the **Resource map** tab, identify public subnets (connected to route tables that route to `igw-...` internet gateway)
5. Back in the SageMaker console, choose at least **two public subnets**
6. Choose the **default security group**
7. Choose **Next**

### Step 1.6: Configure Storage

- Scroll to the bottom and choose **Next**

### Step 1.7: Review and Create

- Scroll to the bottom and choose **Submit**
- Wait for the Domain to be created (typically **5 to 8 minutes**)
- Refresh the browser tab occasionally to check status

---

## Step 2: Create a SageMaker User Profile

1. In the list of SageMaker domains, choose the **name link** of the domain you created
2. In the **User profiles** tab, choose **Add user**
3. In **General settings**, for **Execution role** choose `LabRole` and choose **Next**
4. In Step 2: Choose **Next**
5. In Step 3: Choose **Next**
6. In Step 4: Keep **Inherit settings from domain** selected and choose **Next**
7. In Step 5: Scroll to the bottom and choose **Submit**

---

## Step 3: Create a Code Editor Space (Visual Studio Code Open Source)

1. Choose **Studio**
2. In the **Get started** panel, verify the user profile you created is selected
3. Choose **Open Studio**
4. Choose **Skip Tour for now**
5. From the **Applications** panel, choose **Code Editor**
6. Choose the **Create Code Editor space** button in the top right corner
7. Give it a name (e.g., `mySpace`) and choose **Create space**
8. Verify the space settings (defaults like `ml.t3.medium` are supported)
9. Choose **Run space**
10. Once the space has started, choose **Open Code Editor**

> A new browser tab opens, displaying an IDE. Refer to the [AWS Documentation](https://docs.aws.amazon.com/sagemaker/latest/dg/code-editor.html) to learn more about Code Editor in SageMaker Studio.

---

## Step 4: Use JupyterLab in SageMaker Studio

1. Choose **Studio**
2. From the **Applications** list in the top left corner, choose **JupyterLab**
3. Choose **Create JupyterLab space**
4. Give the space a name and choose **Create space**
5. Choose **Run space**
6. Wait for the JupyterLab space to start
7. Choose **Open JupyterLab**
8. In the JupyterLab UI, you can launch a notebook, console, or other resources

> If you have a running studio space, it will appear as a running App in the SageMaker Studio console.

---

## Step 5: Upload and Run Notebooks

Once you have JupyterLab running:

1. **Upload the notebooks:**
   - `01_part1_linreg_1feature.ipynb`
   - `02_part2_polyreg.ipynb`

2. **Run all cells** in each notebook to verify successful execution

3. **Verify outputs:** Ensure all plots are rendered correctly

---

## Using SageMaker Canvas (Optional)

1. First create a SageMaker domain and user profile (see steps above)
2. From the **User profiles** list, in the row with your profile, choose **Launch > Canvas**
3. The SageMaker Canvas console appears

> **Note:** There is limited support for SageMaker Canvas features. Many Ready-to-use models are not supported in Learner Labs.

---

## Tips to Preserve Your Budget

| Tip | Description |
|-----|-------------|
| **Monitor Dashboard** | Choose the SageMaker dashboard link to view recent activity including running jobs, models, or instances |
| **Stop Unused Resources** | Stop or delete anything that is running and no longer needed |
| **Stopped Instances** | When your session ends, running SageMaker notebook instances may be placed into a 'stopped' state. They will not automatically restart in new sessions |
| **Logout Sessions** | When using SageMaker Canvas or Studio, logout when done working |
| **Delete Unused Apps** | Consider deleting SageMaker Canvas and Studio apps that are no longer needed |

---

---

## Step 6: Deploy Heart Disease Model to SageMaker Endpoint

This section guides you through deploying the trained logistic regression model as a real-time inference endpoint.

### Prerequisites

Before starting deployment:
- ✅ Complete all cells in `heart_disease_lr_analysis.ipynb` (Steps 1-5)
- ✅ Verify `model.tar.gz` file was created
- ✅ Have JupyterLab space running in SageMaker Studio

### Step 6.1: Upload Model Artifacts to JupyterLab

1. Open your JupyterLab space in SageMaker Studio
2. In the file browser (left panel), click the **Upload** button (⬆️)
3. Upload the following files from your local project:
   - `model.tar.gz` (packaged model)
   - `model_artifacts/` folder (contains inference.py)
   - `heart_disease_prediction.csv` (optional, for testing)

### Step 6.2: Create a New Deployment Notebook

1. In JupyterLab, click **File > New > Notebook**
2. Select **Python 3** kernel
3. Name the notebook `deploy_model.ipynb`

### Step 6.3: Initialize SageMaker Session

Run the following code in the first cell:

```python
import boto3
import sagemaker
from sagemaker.sklearn.model import SKLearnModel
import json

# Initialize SageMaker session
sagemaker_session = sagemaker.Session()
region = sagemaker_session.boto_region_name
bucket = sagemaker_session.default_bucket()

# Get the execution role (LabRole)
role = sagemaker.get_execution_role()

print(f"✅ SageMaker Session Initialized")
print(f"   Region: {region}")
print(f"   Bucket: {bucket}")
print(f"   Role: {role}")
```

### Step 6.4: Upload Model to S3

```python
# Upload model.tar.gz to S3
s3_model_path = sagemaker_session.upload_data(
    path='model.tar.gz',
    bucket=bucket,
    key_prefix='heart-disease-model'
)

print(f"✅ Model uploaded to S3")
print(f"   S3 Path: {s3_model_path}")
```

### Step 6.5: Create SageMaker Model

```python
# Create the SageMaker Model
model = SKLearnModel(
    model_data=s3_model_path,
    role=role,
    entry_point='inference.py',
    source_dir='model_artifacts',
    framework_version='1.2-1',
    py_version='py3',
    sagemaker_session=sagemaker_session
)

print("✅ SageMaker Model created")
```

### Step 6.6: Deploy to Real-Time Endpoint

```python
# Deploy to a real-time endpoint
endpoint_name = 'heart-disease-prediction-endpoint'

predictor = model.deploy(
    initial_instance_count=1,
    instance_type='ml.t2.medium',  # Cost-effective for inference
    endpoint_name=endpoint_name
)

print(f"✅ Endpoint deployed successfully!")
print(f"   Endpoint Name: {endpoint_name}")
```

> **Note:** Deployment typically takes **3-5 minutes**. Wait for the cell to complete.

### Step 6.7: Test the Endpoint

Once deployed, test with a sample patient:

```python
import json

# Test patient data
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

# Invoke the endpoint
runtime_client = boto3.client('sagemaker-runtime')

response = runtime_client.invoke_endpoint(
    EndpointName=endpoint_name,
    ContentType='application/json',
    Body=json.dumps(test_patient)
)

# Parse response
result = json.loads(response['Body'].read().decode())

print("\n🏥 Heart Disease Risk Prediction")
print("=" * 50)
print(f"Patient Data: {test_patient}")
print(f"\nResults:")
print(f"   Probability: {result['probability']:.2%}")
print(f"   Risk Level: {result['risk_level']}")
print(f"   Diagnosis: {'Heart Disease ⚠️' if result['has_heart_disease'] else 'No Heart Disease ✅'}")
```

### Step 6.8: Cleanup - DELETE ENDPOINT (Important!)

**⚠️ CRITICAL: Always delete your endpoint when finished to avoid ongoing charges!**

```python
import boto3

# Initialize SageMaker client
sagemaker_client = boto3.client('sagemaker')
endpoint_name = 'heart-disease-prediction-endpoint'

# Delete the endpoint
sagemaker_client.delete_endpoint(EndpointName=endpoint_name)
print(f"✅ Endpoint '{endpoint_name}' deleted")

# Delete endpoint configuration
sagemaker_client.delete_endpoint_config(EndpointConfigName=endpoint_name)
print("✅ Endpoint configuration deleted")

# Optional: Delete the model
# model_name = 'heart-disease-sklearn-model'
# sagemaker_client.delete_model(ModelName=model_name)
# print("✅ Model deleted")

print("\n💰 Resources cleaned up - no ongoing charges!")
```

---

## Step 7: Verify Endpoint in AWS Console

1. Navigate to **Amazon SageMaker** in the AWS Console
2. Choose **Inference > Endpoints** from the left menu
3. Verify your `heart-disease-prediction-endpoint` appears with status **InService**
4. Click on the endpoint name to see details:
   - Instance type: `ml.t2.medium`
   - Instance count: 1
   - Created time
   - Endpoint ARN

---

## Endpoint Pricing Reference

| Instance Type | vCPU | Memory | Price/Hour (us-east-1) |
|--------------|------|--------|------------------------|
| ml.t2.medium | 2 | 4 GB | ~$0.05 |
| ml.t3.medium | 2 | 4 GB | ~$0.05 |
| ml.m5.large | 2 | 8 GB | ~$0.12 |

> **Tip:** Use `ml.t2.medium` for development/testing. Delete endpoints when not in use.

---

## Troubleshooting

### Limited Feature Support

- Some SageMaker JumpStart projects require more access permissions than available in Learner Labs
- Some SageMaker Canvas models (including many Ready-to-use models) may not be supported
- If a model is "powered by" an AWS service that is not supported, the model will not run

### Creating Notebook Instances

You can create SageMaker Notebook instances as an alternative to Studio spaces. The same instance type limitations apply.

### Common Deployment Issues

| Issue | Solution |
|-------|----------|
| **Endpoint stuck in "Creating"** | Wait up to 10 minutes. Check CloudWatch logs for errors. |
| **Permission Denied** | Verify LabRole is selected as execution role |
| **Model not found** | Ensure `model.tar.gz` was uploaded correctly to S3 |
| **Inference error** | Check `inference.py` syntax and feature names match training |
| **High latency** | First request may be slow (cold start). Subsequent requests faster. |

### Checking Endpoint Logs

```python
# View CloudWatch logs for debugging
logs_client = boto3.client('logs')
log_group = f"/aws/sagemaker/Endpoints/{endpoint_name}"
print(f"CloudWatch Log Group: {log_group}")
```
