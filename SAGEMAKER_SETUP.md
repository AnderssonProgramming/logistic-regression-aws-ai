# AWS SageMaker Setup Guide

Step-by-step guide for deploying the Heart Disease Prediction model using AWS SageMaker.

---

## ⚠️ Important: Learner Lab Limitations

**AWS Academy Learner Labs have restrictions** that prevent creating real-time endpoints:

| Action | Status |
|--------|--------|
| Create SageMaker Domain | ✅ Allowed |
| Create Code Editor Space | ✅ Allowed |
| Upload model to S3 | ✅ Allowed |
| Create SageMaker Model | ✅ Allowed |
| **Create Endpoint** | ❌ **Blocked** |
| **Create EndpointConfig** | ❌ **Blocked** |

**Solution:** Use `demo_deployment.py` which demonstrates everything except the blocked endpoint creation.

---

## Prerequisites

- AWS Account with access to Amazon SageMaker
- LabRole IAM role configured
- Completed notebook (`heart_disease_lr_analysis.ipynb`) with `model.tar.gz` generated

## Supported Instance Types

| Resource | Supported Types |
|----------|-----------------|
| **Studio/Code Editor** | ml.t3.medium, ml.t3.large, ml.m5.large |
| **Inference Endpoints** | ml.t2.medium, ml.t2.large, ml.m5.large |

> ⚠️ **Note:** `ml.t3.*` instances are NOT valid for inference endpoints!

---

## Step 1: Create a SageMaker Domain

1. Navigate to **Amazon SageMaker** in the AWS Console
2. Choose **Domains** → **Create domain**
3. Choose **Set up for organizations** → **Set up**

### Domain Configuration

1. **Domain Details:**
   - Name: `myDomain`
   - Keep **Login through IAM** default
   - Choose **Next**

2. **Roles:**
   - Choose **Use an existing role**
   - Set **Default execution role** to `LabRole`
   - Choose **Next**

3. **Applications:**
   - **SageMaker Studio:** Choose **SageMaker Studio - New**
   - **CodeEditor:** Enable idle shutdown (60 minutes)
   - Choose **Next**

4. **Network:**
   - Choose **VPC Only** or **Public internet access**
   - Select **Default VPC** and at least **two public subnets**
   - Choose **default security group**
   - Choose **Next** → **Submit**

5. Wait **5-8 minutes** for domain creation

---

## Step 2: Create a User Profile

1. In your domain, go to **User profiles** → **Add user**
2. Set **Execution role** to `LabRole`
3. Click **Next** through all steps → **Submit**

---

## Step 3: Create Code Editor Space

1. Go to **Studio** → Select your user profile → **Open Studio**
2. From **Applications**, choose **Code Editor**
3. Click **Create Code Editor space**
4. Name: `mySpace` → **Create space**
5. Verify instance type is `ml.t3.medium` → **Run space**
6. Click **Open Code Editor** (opens VS Code in browser)

---

## Step 4: Upload Project Files

In Code Editor, upload these files via drag & drop:

```
├── model.tar.gz              # Packaged model (REQUIRED)
├── model_artifacts/          # Model files folder
│   ├── inference.py
│   ├── weights.npy
│   ├── bias.npy
│   ├── feature_mean.npy
│   ├── feature_std.npy
│   └── model_metadata.json
└── sagemaker_scripts/        # Deployment scripts
    ├── demo_deployment.py    # ✅ Use this in Learner Lab
    ├── deploy.py             # Full deployment (needs permissions)
    ├── test_endpoint.py      # Test deployed endpoint
    └── cleanup.py            # Delete endpoint
```

---

## Step 5: Run Deployment

### Option A: Learner Lab (Recommended)

Open Terminal in Code Editor (`Terminal > New Terminal`) and run:

```bash
cd ~/your-project-folder
python sagemaker_scripts/demo_deployment.py
```

**Expected Output:**
```
🚀 HEART DISEASE MODEL - DEPLOYMENT DEMO
======================================================================
📦 Step 1: Initializing SageMaker session...
   ✅ Region: us-east-1
   ✅ Bucket: sagemaker-us-east-1-XXXX

📤 Step 2: Uploading model.tar.gz to S3...
   ✅ S3 Path: s3://sagemaker-us-east-1-XXXX/heart-disease-model/model.tar.gz

🔧 Step 3: Creating SageMaker Model object...
   ✅ Model object created successfully

🧪 Step 5: Testing inference LOCALLY (simulating endpoint)...
   📋 High-Risk Patient
      Probability: 99.38%
      Diagnosis: Heart Disease ⚠️

   📋 Low-Risk Patient
      Probability: 0.28%
      Diagnosis: No Heart Disease ✅

✅ DEPLOYMENT DEMO COMPLETE!
```

### Option B: Full AWS Account (When Permissions Allow)

```bash
# 1. Deploy endpoint (takes 3-5 minutes)
python sagemaker_scripts/deploy.py

# 2. Test the endpoint
python sagemaker_scripts/test_endpoint.py

# 3. CRITICAL: Delete endpoint to stop charges
python sagemaker_scripts/cleanup.py
```

---

## Troubleshooting

### Error: AccessDeniedException on CreateEndpointConfig

```
User is not authorized to perform: sagemaker:CreateEndpointConfig
with an explicit deny in an identity-based policy
```

**Cause:** Learner Lab policy blocks endpoint creation.  
**Solution:** Use `demo_deployment.py` instead.

### Error: ValidationException on ml.t3.medium

```
An error occurred (ValidationException)... ml.t3.medium... is not valid
```

**Cause:** `ml.t3.*` instances are only for Studio, not for endpoints.  
**Solution:** Use `ml.t2.medium` for endpoints.

### Error: ConnectTimeoutError to sts.amazonaws.com

```
ConnectTimeoutError... Connect timeout on endpoint URL
```

**Cause:** Network/VPC configuration issue.  
**Solution:** Wait and retry, or check VPC has internet access.

---

## Budget Tips

| Tip | Description |
|-----|-------------|
| **Stop Spaces** | Stop Code Editor spaces when not in use |
| **Delete Endpoints** | Always run `cleanup.py` after testing |
| **Monitor Dashboard** | Check SageMaker dashboard for running resources |
| **Idle Shutdown** | Enabled by default (60 min) but manually stop when done |

---

## Scripts Reference

| Script | Description |
|--------|-------------|
| `demo_deployment.py` | Safe demo for Learner Labs - tests everything locally |
| `deploy.py` | Full deployment - creates real endpoint |
| `test_endpoint.py` | Sends test patients to endpoint |
| `cleanup.py` | Deletes endpoint to stop charges |

See [sagemaker_scripts/README.md](sagemaker_scripts/README.md) for detailed documentation.
