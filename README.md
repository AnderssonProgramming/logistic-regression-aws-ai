# Heart Disease Risk Prediction: Logistic Regression Homework

## Exercise Summary

Implements logistic regression for heart disease prediction: EDA, training/visualization, regularization, and SageMaker deployment. This project builds logistic regression from scratch using NumPy—no scikit-learn for core training—to understand the mathematical foundations behind classification algorithms.

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on deploying the project on AWS SageMaker.

### Prerequisites

Requirements for running the notebooks:

- [Python 3.x](https://www.python.org/)
- [NumPy](https://numpy.org/) - Numerical computing
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [Matplotlib](https://matplotlib.org/) - Visualization
- [Jupyter Notebook](https://jupyter.org/) - For local execution
- [AWS Account](https://aws.amazon.com/) - For SageMaker execution

> **Not allowed:** scikit-learn, statsmodels, TensorFlow/PyTorch, or any high-level regression/optimization library for core training.

### Installing

A step by step series to get a development environment running:

1. Clone the repository

    ```bash
    git clone https://github.com/AnderssonProgramming/logistic-regression-aws-ai.git
    cd logistic-regression-aws-ai
    ```

2. Install the required libraries

    ```bash
    pip install numpy pandas matplotlib jupyter
    ```

3. Launch Jupyter Notebook

    ```bash
    jupyter notebook
    ```

4. Open and run the notebook:
   - `heart_disease_lr_analysis.ipynb`

## Introductory Context

Heart disease is the world's leading cause of death, claiming approximately 18 million lives each year, as reported by the World Health Organization. Predictive models like logistic regression can enable early identification of at-risk patients by analyzing clinical features such as age, cholesterol, and blood pressure. This not only improves treatment outcomes but also optimizes resource allocation in healthcare settings. In this homework, you'll implement logistic regression on the Heart Disease Dataset—a real-world UCI repository collection of 303 patient records with 14 features and a binary target (1 for disease presence, 0 for absence). You'll train models, visualize boundaries, apply regularization, and explore deployment via Amazon SageMaker to mimic a production pipeline.

### Motivation for Cloud Execution and Enterprise Context

This project is part of a four-week Machine Learning Bootcamp embedded in a course on Digital Transformation and Enterprise Architecture. In this context, machine learning is treated as a core architectural capability of modern enterprise systems.

Today, intelligence is increasingly considered a first-class quality attribute alongside scalability, availability, security, and performance. Intelligent behavior is no longer confined to offline analytics; it is embedded into platforms, decision-support services, and autonomous or semi-autonomous components.

As enterprise architects, it is not sufficient to understand what models do. We must also understand how they are built from first principles, executed and validated in controlled environments, and operated within cloud platforms.

## Dataset Description

### Source

**Kaggle Heart Disease Dataset** - Downloaded from [https://www.kaggle.com/datasets/neurocipher/heartdisease](https://www.kaggle.com/datasets/neurocipher/heartdisease)

> **About Kaggle:** Kaggle is a popular online platform for data science enthusiasts, hosting datasets, competitions, and notebooks—think of it as GitHub for data and ML projects (free to join at [kaggle.com](https://kaggle.com)).

### About the Dataset

This dataset contains real-world clinical attributes used to analyze and predict the presence or absence of heart disease. Each row represents one patient, and each column represents a medical measurement or diagnostic indicator.

**✨ Suitable for:**
- 📊 Exploratory Data Analysis (EDA)
- 🤖 Machine Learning / AI models
- 🧪 Binary classification
- 🔍 Feature importance analysis
- 🧠 Medical data science practice

### Dataset Overview

| Attribute | Description |
|-----------|-------------|
| **Samples** | 270 patient records |
| **Features** | 14 clinical attributes |
| **Target** | Binary (Presence = disease, Absence = no disease) |
| **Disease Rate** | ~44.4% presence rate (120 Presence / 150 Absence) |

### Column Descriptions (Data Dictionary)

| Column Name | Description | Values/Range |
|-------------|-------------|--------------|
| **Age** | Age of the patient | 29-77 years |
| **Sex** | Gender of the patient | 1 = Male, 0 = Female |
| **Chest pain type** | Type of chest pain | 1 = Typical angina, 2 = Atypical angina, 3 = Non-anginal pain, 4 = Asymptomatic |
| **BP** | Resting blood pressure | 94-200 mm Hg |
| **Cholesterol** | Serum cholesterol level | 126-564 mg/dL |
| **FBS over 120** | Fasting blood sugar > 120 mg/dL | 1 = True, 0 = False |
| **EKG results** | Resting electrocardiogram results | 0 = Normal, 1 = ST-T wave abnormality, 2 = Left ventricular hypertrophy |
| **Max HR** | Maximum heart rate achieved | 71-202 bpm |
| **Exercise angina** | Exercise-induced angina | 1 = Yes, 0 = No |
| **ST depression** | ST depression induced by exercise relative to rest | 0.0-6.2 |
| **Slope of ST** | Slope of the peak exercise ST segment | 1-3 |
| **Number of vessels fluro** | Number of major vessels (0-3) colored by fluoroscopy | 0-3 |
| **Thallium** | Thallium stress test result (categorical medical indicator) | 3-7 |
| **Heart Disease** | Target variable | Presence = Heart disease detected, Absence = No heart disease |

### Encoding Notes

- ✔ Categorical variables are numerically encoded for ML compatibility
- ✔ Target column uses text labels (Presence / Absence) for better interpretability
- ✔ Dataset is ready for Logistic Regression, Tree-based models, and Ensembles

### Source & Context

This dataset follows standard clinical encodings commonly used in heart disease research, similar to datasets used in:
- 🏥 Medical machine learning studies
- 🎓 Academic projects
- 🏆 Kaggle notebooks & benchmarks

> ⚠️ **Disclaimer:** This dataset is intended ONLY for educational and research purposes. It must NOT be used for real-world medical diagnosis or treatment decisions without professional clinical validation.

## Homework Steps

### Step 1: Load and Prepare the Dataset
- Download from Kaggle and load into Pandas
- Binarize target column (1=disease presence, 0=absence)
- EDA: Summarize stats, handle missing/outliers, plot class distribution
- 70/30 train/test split (stratified); normalize numerical features
- Select ≥6 features (e.g., Age, Cholesterol, BP, Max HR, ST Depression, Vessels)

### Step 2: Implement Basic Logistic Regression
- Implement sigmoid, cost (binary cross-entropy), gradient descent
- Train on full train set (α~0.01, 1000+ iterations)
- Plot cost vs. iterations
- Predict (threshold 0.5); evaluate accuracy/precision/recall/F1 on train/test

### Step 3: Visualize Decision Boundaries
- Select ≥3 feature pairs (e.g., Age-Cholesterol, BP-Max HR, ST Depression-Vessels)
- For each pair: subset to 2D, train model, plot boundary line + scatter
- Discuss separability/nonlinearity

### Step 4: Repeat with Regularization
- Add L2 to cost/gradients: λ/(2m)||w||²; dw += (λ/m)w
- Tune λ values: [0, 0.001, 0.01, 0.1, 1]
- Re-plot costs/boundaries (one pair: unreg vs. reg)
- Re-evaluate metrics and ||w||

### Step 5: Explore Deployment in Amazon SageMaker
- Export best model (w/b as NumPy array)
- Create SageMaker notebook instance; upload/run notebook
- Build/deploy simple endpoint for inference
- Test with sample input (e.g., Age=60, Chol=300)

## Repository Structure

```
/
├── README.md                              # Project documentation
├── SAGEMAKER_SETUP.md                     # SageMaker setup guide
├── heart_disease_lr_analysis.ipynb        # Main Logistic Regression notebook
├── heart_disease_prediction.csv           # Dataset file
├── dataset.py                             # Dataset utilities
├── LICENSE                                # MIT License
└── refering_notebooks/                    # Reference materials
    ├── week2_classification_hour1_final.ipynb
    ├── week2_classification_hour2_regularization_with_derivatives.ipynb
    └── APENDIX-RidgeVsGradientDescentInRegularizedLinearRegression.ipynb
```

### Notebook: Logistic Regression

`heart_disease_lr_analysis.ipynb` - Implements logistic regression from scratch using gradient descent to predict heart disease risk. Includes:
- **EDA**: Data exploration, visualization, and preprocessing
- **Implementation**: Sigmoid, cost function (binary cross-entropy), gradient descent
- **Visualization**: Decision boundary plots for multiple feature pairs
- **Regularization**: L2 regularization with hyperparameter tuning
- **Evaluation**: Accuracy, precision, recall, F1-score metrics

## Deployment

### AWS SageMaker Execution

To deploy and run this project on AWS SageMaker:

1. Upload the notebook to AWS SageMaker (Studio or Notebook Instances)
2. Run all cells successfully (no errors)
3. Export best model (w/b as NumPy array)
4. Create inference handler for patient inputs → probability output
5. Deploy endpoint and test with sample inputs

### Deployment Evidence

<!-- TODO: Add deployment screenshots here -->
<!-- 
Include ≥3 images:
1. Training job status screenshot
2. Endpoint configuration screenshot  
3. Inference response screenshot

Example:
![Training Job Status](images/sagemaker_training.png)
![Endpoint Config](images/sagemaker_endpoint.png)
![Inference Response](images/sagemaker_inference.png)
-->

**Sample Inference Test:**
- **Input:** Age=60, Chol=300
- **Output:** Prob=0.XX (risk level)
- **Endpoint ARN:** `[To be added after deployment]`

> **Deployment Benefits:** Enables real-time risk scoring for clinical decision support. Expected latency: ~XXms per inference.

### AWS SageMaker Execution Evidence

The successful execution of the notebook on AWS SageMaker is documented in the following video:

📹 **[aws-sagemaker-ai-notebooks-video.mp4](aws-sagemaker-ai-notebooks-video.mp4)**

The video demonstrates:
- ✅ Notebook open in AWS SageMaker JupyterLab
- ✅ Successful execution of all cells (no errors)
- ✅ Rendered plots and visualizations
- ✅ Complete training loop outputs
- ✅ Model deployment and endpoint testing

#### How notebooks were uploaded to SageMaker

For detailed step-by-step instructions on setting up AWS SageMaker (creating domains, user profiles, JupyterLab spaces, and uploading notebooks), see the **[SageMaker Setup Guide](SAGEMAKER_SETUP.md)**.

#### Comparison: Local Execution vs SageMaker Execution

| Aspect | Local Execution | AWS SageMaker |
|--------|-----------------|---------------|
| **Environment** | Personal machine with Jupyter | Cloud-based JupyterLab |
| **Setup** | Manual Python/library installation | Pre-configured ML environment |
| **Compute** | Limited to local hardware | Scalable instance types (ml.t3, ml.m5, etc.) |
| **Results** | ✅ Identical outputs | ✅ Identical outputs |
| **Plots** | ✅ Rendered correctly | ✅ Rendered correctly |

> **Conclusion:** Both environments produced identical results. The regression models, loss calculations, and visualizations behaved consistently across local and cloud execution, validating the portability of the implementation.

## Built With

- [Python](https://www.python.org/) - Programming language
- [NumPy](https://numpy.org/) - Numerical computing library
- [Pandas](https://pandas.pydata.org/) - Data manipulation and analysis
- [Matplotlib](https://matplotlib.org/) - Visualization library
- [AWS SageMaker](https://aws.amazon.com/sagemaker/) - Cloud ML platform

## Evaluation Criteria

| Criterion | Points | Description |
|-----------|--------|-------------|
| **EDA** | 10 | Data exploration, preprocessing, and insights |
| **Implementation** | 35 | Correctness of loss, gradients, and training loop |
| **Visualization/Analysis** | 20 | Quality of plots and interpretations |
| **Regularization** | 15 | L2 implementation and hyperparameter tuning |
| **Deployment/Repo** | 15 | SageMaker deployment with documented evidence |
| **Clarity** | 5 | Code comments and documentation quality |
| **Total** | **100** | |

## Authors

- **Andersson David Sánchez Méndez** - *Developer* - [AnderssonProgramming](https://github.com/AnderssonProgramming)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Machine Learning Bootcamp - Digital Transformation and Enterprise Architecture course
- UCI Machine Learning Repository for the original Heart Disease dataset
- Kaggle for dataset hosting and accessibility
- AWS SageMaker for cloud ML deployment capabilities
