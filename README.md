# Heart Disease Risk Prediction: Logistic Regression Homework

Implementation of logistic regression models from a dataset to predict heart disease based on age, sex, chest pain type, BP, cholesterol, FBS over 120, EKG results, max HR, exercise angina, ST depression, deployed on AWS SageMaker.

## Getting Started

These instructions will give you a copy of the project up and running on your local machine for development and testing purposes. See deployment for notes on deploying the project on AWS SageMaker.

### Prerequisites

Requirements for running the notebooks:

- [Python 3.x](https://www.python.org/)
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Inline plots only
- [Jupyter Notebook](https://jupyter.org/) - For local execution
- [AWS Account](https://aws.amazon.com/) - For SageMaker execution

> **Not allowed:** scikit-learn, statsmodels, TensorFlow/PyTorch, or any high-level regression/optimization library.

### Installing

A step by step series to get a development environment running:

1. Clone the repository

    ```bash
    git clone https://github.com/AnderssonProgramming/logistic-regression-aws-ai.git
    cd logistic-regression-aws-ai
    ```

2. Install the required libraries

    ```bash
    pip install numpy matplotlib jupyter
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

## Dataset and Notation

Use the following notation throughout:

| Symbol | Description | Units |
|--------|-------------|-------|
| **M** | Stellar mass | Solar mass (M⊙) |
| **T** | Effective stellar temperature | Kelvin (K) |
| **L** | Stellar luminosity | Solar luminosity (L⊙) |

### Part I Dataset (One Feature)

```python
M = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
L = [0.15, 0.35, 1.00, 2.30, 4.10, 7.00, 11.2, 17.5, 25.0, 35.0]
```

### Part II Dataset (Two Features)

```python
M = [0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.2, 2.4]
T = [3800, 4400, 5800, 6400, 6900, 7400, 7900, 8300, 8800, 9200]
L = [0.15, 0.35, 1.00, 2.30, 4.10, 7.00, 11.2, 17.5, 25.0, 35.0]
```

## Repository Structure

```
/
├── README.md                           # Project documentation
├── heart_disease_lr_analysis.ipynb      # Logistic Regression 
```

### Notebook: Logistic Regression

`heart_disease_lr_analysis.ipynb` - Implements logistic regression from scratch using gradient descent to model and predict the relationship in heart disease risk.

## Deployment

### AWS SageMaker Execution

To deploy and run this project on AWS SageMaker:

1. Upload the unique notebook to AWS SageMaker (Studio or Notebook Instances)
2. Run all cells successfully (no errors)
3. No model deployment, endpoints, or MLOps pipelines are required

### AWS SageMaker Execution Evidence

The successful execution of both notebooks on AWS SageMaker is documented in the following video:

📹 **[aws-sagemaker-ai-notebooks-video.mp4](aws-sagemaker-ai-notebooks-video.mp4)**

The video demonstrates:
- ✅ Both notebooks open in AWS SageMaker JupyterLab
- ✅ Successful execution of all cells (no errors)
- ✅ Rendered plots and visualizations
- ✅ Complete training loop outputs

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
- [Matplotlib](https://matplotlib.org/) - Visualization library
- [AWS SageMaker](https://aws.amazon.com/sagemaker/) - Cloud ML platform

## Evaluation Criteria

| Criterion | Description |
|-----------|-------------|
| Correctness | Implementation of loss, gradients, and training loop |
| Vectorization | Proper use of vectorization where required |
| Plots | Quality and completeness (dataset, cost surface, interaction cost, convergence) |
| Explanations | Quality of explanations and interpretations |
| SageMaker | Successful execution with documented evidence |

## Authors

- **Andersson David Sánchez Méndez** - *Developer* - [AnderssonProgramming](https://github.com/AnderssonProgramming)

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Acknowledgments

- Machine Learning Bootcamp - Digital Transformation and Enterprise Architecture course
- Inspiration from main-sequence stellar behavior models
- AWS SageMaker for cloud ML deployment capabilities
