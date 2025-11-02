# Project 5: Stock Price Forecasting with Time Series Analysis

## Overview

This project explores the application of time series analysis for forecasting stock prices. Using historical stock data, we analyze trends, seasonality, and volatility to build a predictive model. The goal is to demonstrate proficiency in time series forecasting techniques and provide insights that could inform trading or investment strategies.

### Key Objectives

- **Analyze** historical stock price data to identify patterns.
- **Develop** a time series forecasting model.
- **Evaluate** the model's performance against actual price movements.
- **Visualize** trends, volatility, and forecast results.

---

## Key Findings & Visualizations

1.  **Clear Trend**: The stock data exhibits a clear upward trend over the three-year period.
2.  **Seasonality**: Noticeable seasonal patterns are present, with price peaks and troughs occurring at regular intervals.
3.  **Forecast Accuracy**: The simple moving average model provides a reasonable baseline forecast, with a Mean Absolute Percentage Error (MAPE) of **6.33%**.

| Metric | Value |
| :--- | :--- |
| **Forecast Model** | 30-Day Moving Average |
| **MAE** | $10.18 |
| **RMSE** | $12.07 |
| **MAPE** | 6.33% |

![Forecast vs. Actual](../visualizations/forecast_comparison.png)
*Figure 1: Comparison of the forecasted stock prices against the actual prices for the test period.*

![Price and Volume Trend](../visualizations/price_volume_trend.png)
*Figure 2: Historical stock price with 7, 30, and 90-day moving averages, along with trading volume.*

---

## Technical Implementation

### Time Series Analysis

The project involves several key steps:

1.  **Data Generation**: A synthetic dataset was created to simulate realistic stock price movements with trend and seasonality.
2.  **Feature Engineering**: Calculated moving averages, daily returns, and rolling volatility to enrich the dataset.
3.  **Forecasting Model**: A **30-day moving average** was used as a simple yet effective forecasting model.
4.  **Evaluation**: The model was evaluated using standard regression metrics (MAE, RMSE, MAPE).

### Technology Stack

- **Python**: Core programming language.
- **Pandas & NumPy**: For data manipulation and numerical analysis.
- **Matplotlib & Seaborn**: For data visualization.
- **Scikit-learn**: For model evaluation metrics.

---

## How to Run This Project

### Prerequisites

- Python 3.9+
- Pip (Python package installer)

### Installation

1.  **Clone the repository:**
    ```bash
    git clone <repository-url>
    cd project5_stock_forecasting
    ```

2.  **Install dependencies:**
    ```bash
    pip install pandas numpy matplotlib seaborn scikit-learn
    ```

### Running the Analysis

To run the forecasting script and generate all visualizations:

```bash
python src/stock_forecasting.py
```

---

## Business Impact

This project demonstrates the ability to analyze financial time series data and build predictive models. These skills are crucial for:

- **Algorithmic Trading**: Developing automated trading strategies based on price predictions.
- **Risk Management**: Assessing and mitigating investment risk by understanding volatility.
- **Portfolio Optimization**: Making informed decisions about asset allocation.
- **Financial Planning**: Forecasting future asset values for long-term financial planning.
'''
