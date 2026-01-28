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

## Troubleshooting

### Limited Feature Support

- Some SageMaker JumpStart projects require more access permissions than available in Learner Labs
- Some SageMaker Canvas models (including many Ready-to-use models) may not be supported
- If a model is "powered by" an AWS service that is not supported, the model will not run

### Creating Notebook Instances

You can create SageMaker Notebook instances as an alternative to Studio spaces. The same instance type limitations apply.
