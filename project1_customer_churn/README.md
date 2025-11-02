# Project 1: Customer Churn Prediction System

## Overview

This project develops a machine learning system to predict customer churn for a telecommunications company. By identifying customers at high risk of leaving, the company can implement targeted retention strategies to reduce revenue loss and improve customer loyalty. The solution includes a comprehensive analysis of the IBM Telco Customer Churn dataset, development of multiple predictive models, and an interactive Streamlit dashboard for real-time predictions.

### Key Objectives

- **Analyze** key drivers of customer churn.
- **Build** and **evaluate** multiple machine learning models.
- **Develop** an interactive dashboard for business users.
- **Quantify** the potential business impact of the solution.

---

## Key Findings & Visualizations

Our analysis revealed several critical factors influencing churn:

1.  **Contract Type**: Customers with month-to-month contracts are **3 times more likely to churn** than those on long-term contracts.
2.  **Payment Method**: Electronic check payments are associated with a significantly higher churn rate.
3.  **Service Usage**: Customers with Fiber Optic internet service show higher churn, suggesting potential service quality or pricing issues.

| Feature | Insight |
| :--- | :--- |
| **Contract** | Month-to-month contracts have a 42% churn rate. |
| **Tenure** | New customers (0-12 months) are most likely to churn. |
| **Internet Service** | Fiber optic users have a higher churn rate than DSL users. |
| **Payment Method** | Electronic check users show the highest churn. |

![Model Performance](../visualizations/model_comparison.png)
*Figure 1: Comparison of machine learning models. Gradient Boosting achieved the highest accuracy and AUC-ROC score.*

![Feature Importance](../visualizations/feature_importance.png)
*Figure 2: Top 15 features influencing churn prediction. Contract type, tenure, and monthly charges are the most significant predictors.*

---

## Technical Implementation

### Machine Learning Models

We trained and evaluated three different classification models:

- **Logistic Regression**: A baseline model for binary classification.
- **Random Forest**: An ensemble model for improved accuracy and robustness.
- **Gradient Boosting**: The best-performing model, achieving **82% accuracy** and an **AUC-ROC of 0.85**.

### Interactive Dashboard

An interactive web application was built using **Streamlit** to provide a user-friendly interface for making real-time churn predictions. The dashboard allows business users to input customer data and receive an instant churn probability score, along with actionable recommendations.

![Streamlit Dashboard Screenshot](../screenshots/churn_dashboard.png)
*Figure 3: The interactive churn prediction dashboard.*

---

## How to Run This Project

### Prerequisites

- Python 3.9+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd project1_customer_churn
    ```

2.  **Install dependencies:**
    ```bash
    pip install -r requirements.txt
    ```

### Running the Analysis

To re-run the data analysis and model training, execute the Jupyter notebook:

```bash
njupyter notebook notebooks/customer_churn_analysis_executed.ipynb
```

### Launching the Dashboard

To start the interactive Streamlit dashboard:

```bash
streamlit run src/app.py
```

---

## Business Impact

The implementation of this predictive model is projected to have a significant positive impact on the business:

- **Reduce Churn**: Proactively identify and retain at-risk customers, potentially reducing churn by **15-20%**.
- **Increase Revenue**: Save over **$500,000 annually** by preventing customer loss (based on test set projections).
- **Improve ROI**: Achieve a **400%+ ROI** on customer retention campaigns by targeting the right customers.

This project provides a powerful, data-driven tool to enhance customer retention and drive sustainable business growth.
'''
