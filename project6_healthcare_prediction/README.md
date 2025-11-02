# Project 6: Healthcare Analytics - Disease Prediction

## Overview

This project focuses on the application of machine learning in healthcare to predict the onset of diabetes based on patient health metrics. By leveraging a synthetic dataset that mimics real-world patient data, we build a classification model to identify individuals at high risk of developing diabetes. This allows for early intervention and personalized care, ultimately improving patient outcomes and reducing healthcare costs.

### Key Objectives

- **Analyze** the relationship between various health metrics and diabetes risk.
- **Build** and **evaluate** multiple machine learning models for disease prediction.
- **Identify** the most significant risk factors for diabetes.
- **Demonstrate** the potential for machine learning to support clinical decision-making.

---

## Key Findings & Visualizations

1.  **Top Predictors**: **Glucose level**, **BMI**, and **age** were identified as the most significant predictors of diabetes.
2.  **Model Performance**: The **Gradient Boosting** model achieved the highest accuracy (**87.5%**) and an **AUC-ROC of 0.94**, making it a reliable tool for risk stratification.
3.  **Risk Profiles**: The analysis clearly distinguishes the health profiles of diabetic and non-diabetic patients, providing a basis for targeted interventions.

| Metric | Value |
| :--- | :--- |
| **Best Model** | Gradient Boosting |
| **Accuracy** | 87.50% |
| **AUC-ROC** | 0.9399 |
| **Top Predictor** | Glucose |

![Model Comparison](../visualizations/model_comparison.png)
*Figure 1: Performance comparison of the machine learning models. Gradient Boosting shows the best overall performance.*

![Feature Importance](../visualizations/feature_importance.png)
*Figure 2: The most important features for predicting diabetes, as determined by the Gradient Boosting model.*

---

## Technical Implementation

### Machine Learning Pipeline

The project follows a standard machine learning workflow:

1.  **Data Generation**: A synthetic dataset was created to simulate patient records.
2.  **Data Preprocessing**: Features were scaled using **StandardScaler** to prepare them for modeling.
3.  **Model Training**: Three different classification models were trained:
    -   Logistic Regression
    -   Random Forest
    -   Gradient Boosting
4.  **Evaluation**: Models were evaluated based on accuracy, AUC-ROC, and a confusion matrix.

### Technology Stack

- **Python**: Core programming language.
- **Scikit-learn**: For machine learning models, preprocessing, and evaluation.
- **Pandas & NumPy**: For data handling and numerical operations.
- **Matplotlib & Seaborn**: For creating insightful visualizations.

---

## How to Run This Project

### Prerequisites

- Python 3.9+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd project6_healthcare_prediction
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy scikit-learn matplotlib seaborn
    ```

### Running the Analysis

To run the entire analysis and generate visualizations:

```bash
python src/healthcare_prediction.py
```

---

## Clinical & Business Impact

This project demonstrates a powerful application of machine learning in healthcare with significant implications:

- **Early Detection**: Enables healthcare providers to identify high-risk patients before the onset of severe symptoms.
- **Preventive Care**: Allows for targeted, preventive interventions (e.g., lifestyle changes, medication) to delay or prevent diabetes.
- **Cost Reduction**: Reduces long-term healthcare costs by focusing on prevention rather than treatment.
- **Personalized Medicine**: Paves the way for personalized risk scoring and treatment plans, improving the quality of patient care.
'''
