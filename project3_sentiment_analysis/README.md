# Project 3: Sentiment Analysis & Social Media Monitoring

## Overview

This project implements a Natural Language Processing (NLP) system to analyze and classify sentiment from product reviews. By automatically categorizing reviews as positive, negative, or neutral, businesses can gain valuable insights into customer feedback, identify product issues, and monitor brand perception in real-time.

### Key Objectives

- **Preprocess** and **clean** raw text data for NLP.
- **Build** and **evaluate** multiple sentiment classification models.
- **Visualize** sentiment distribution and model performance.
- **Identify** key terms associated with each sentiment.

---

## Key Findings & Visualizations

Our analysis of the product review dataset yielded the following key insights:

1.  **Sentiment Distribution**: The majority of reviews are **positive (50%)**, followed by negative (33%) and neutral (17%).
2.  **Model Performance**: The **Logistic Regression** model achieved the highest accuracy in classifying sentiment.
3.  **Key Terms**: Specific words were strongly associated with each sentiment, providing clear indicators of customer opinion.

| Metric | Insight |
| :--- | :--- |
| **Dominant Sentiment** | Positive |
| **Best Performing Model** | Logistic Regression |
| **Accuracy** | 100% on synthetic data |
| **Key Positive Words** | "amazing", "excellent", "love" |
| **Key Negative Words** | "terrible", "disappointed", "waste" |

![Sentiment Distribution](../visualizations/sentiment_distribution.png)
*Figure 1: Distribution of positive, negative, and neutral sentiments in the dataset.*

![Model Comparison](../visualizations/model_comparison.png)
*Figure 2: Performance comparison of the trained sentiment analysis models.*

---

## Technical Implementation

### NLP Pipeline

The project follows a standard NLP pipeline:

1.  **Text Preprocessing**: Lowercasing, removing punctuation and special characters.
2.  **Feature Extraction**: Using **TF-IDF (Term Frequency-Inverse Document Frequency)** to convert text into numerical features.
3.  **Model Training**: Training three different models:
    -   Naive Bayes
    -   Logistic Regression
    -   Random Forest

### Technology Stack

- **Python**: Core programming language.
- **Scikit-learn**: For machine learning models and metrics.
- **Pandas**: For data manipulation.
- **Matplotlib & Seaborn**: For data visualization.

---

## How to Run This Project

### Prerequisites

- Python 3.9+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd project3_sentiment_analysis
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas scikit-learn matplotlib seaborn
    ```

### Running the Analysis

To run the sentiment analysis script:

```bash
python src/sentiment_analysis.py
```

This will generate the dataset, train the models, and save all visualizations.

---

## Business Impact

This sentiment analysis system offers significant business value:

- **Enhanced Customer Understanding**: Gain deep insights into customer opinions and satisfaction levels.
- **Proactive Issue Resolution**: Quickly identify and address negative feedback and product issues.
- **Improved Product Development**: Use sentiment data to guide product improvements and feature development.
- **Brand Monitoring**: Track brand perception and the impact of marketing campaigns in real-time.
'''
