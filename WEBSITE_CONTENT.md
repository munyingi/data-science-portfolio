
# Website Content: Data Science Portfolio

**A Guide to Showcasing Your Projects on samwelmunyingi.com**

---

## Introduction for Your Portfolio Page

Welcome to my data science portfolio. Here, I showcase a collection of projects that demonstrate my expertise in machine learning, deep learning, data analysis, and more. Each project is a deep dive into a real-world problem, solved with data-driven solutions and a focus on delivering measurable business impact. Explore my work to see how I turn data into insights and value.

---

## Project 1: Customer Churn Prediction System

### Title: Proactive Customer Retention with Machine Learning

![Customer Churn Dashboard](../project1_customer_churn/screenshots/churn_dashboard.png)

### Overview

In the competitive telecommunications industry, customer retention is paramount. This project tackles the challenge of customer churn by building a machine learning system that predicts which customers are likely to leave. By identifying at-risk customers, the company can proactively engage them with targeted retention strategies, reducing revenue loss and improving customer loyalty. The solution features a complete analysis of customer behavior and an interactive dashboard for real-time predictions.

### How It Works

The system is built on a robust machine learning pipeline. It begins with a deep exploratory data analysis of over 7,000 customer records to identify the key drivers of churn. I then engineered new features to enhance model performance and trained several classification models, including Logistic Regression, Random Forest, and Gradient Boosting. The best-performing model, Gradient Boosting, was selected for its high accuracy and predictive power. The final step was to build an interactive web application using Streamlit, allowing business users to easily input customer data and receive an instant churn probability score.

### Key Features & Results

-   **High-Accuracy Predictions:** The model predicts customer churn with **82% accuracy** and an **AUC-ROC score of 0.85**, ensuring reliable identification of at-risk customers.
-   **Interactive Dashboard:** A user-friendly Streamlit dashboard provides real-time predictions and actionable insights for non-technical users.
-   **Identified Churn Drivers:** The analysis revealed that contract type, tenure, and payment method are the most significant factors influencing churn.
-   **Business Impact Analysis:** The model projects over **$500,000 in annual savings** by enabling targeted retention campaigns, delivering a significant return on investment.

[View on GitHub →](https://github.com/yourusername/data-science-portfolio/tree/main/project1_customer_churn)

---

## Project 2: Real-Time Sales Analytics Dashboard

### Title: Driving Growth with a Data-Driven Sales Dashboard

![Sales Analytics Dashboard](../project2_sales_dashboard/screenshots/sales_dashboard.png)

### Overview

To thrive in a competitive market, businesses need a clear, real-time understanding of their sales performance. This project delivers a comprehensive sales analytics dashboard for a superstore, transforming raw sales data into actionable business intelligence. The dashboard provides key insights into sales trends, profitability, customer behavior, and regional performance, empowering leaders to make strategic, data-driven decisions.

### How It Works

Using Python and its powerful data analysis libraries (Pandas, Matplotlib, and Plotly), I processed and analyzed a large dataset of sales transactions. The analysis focused on key performance indicators (KPIs) such as total sales, profit margins, and order volume. I then designed and built a comprehensive dashboard that visualizes these metrics through a series of interactive charts, including time-series trend lines, geographic maps, and category-based performance breakdowns. The result is a single, intuitive interface for monitoring the health of the business.

### Key Features & Results

-   **Comprehensive KPIs:** The dashboard displays at-a-glance metrics for total sales, profit, orders, and customer counts.
-   **Trend Analysis:** Interactive charts reveal seasonal sales peaks (especially in Q4) and long-term growth patterns.
-   **Performance Segmentation:** The analysis identified the **Technology category** as the highest revenue driver and the **West region** as the most profitable market.
-   **Actionable Insights:** The dashboard highlights areas for improvement, such as the low profitability of the **South region**, enabling targeted business strategies.

[View on GitHub →](https://github.com/yourusername/data-science-portfolio/tree/main/project2_sales_dashboard)

---

## Project 3: Sentiment Analysis of Customer Reviews

### Title: Understanding the Voice of the Customer with NLP

![Sentiment Analysis Dashboard](../project3_sentiment_analysis/screenshots/sentiment_dashboard.png)

### Overview

Customer feedback is a goldmine of information, but manually analyzing thousands of reviews is impossible. This project leverages Natural Language Processing (NLP) to build a sentiment analysis system that automatically classifies product reviews as positive, negative, or neutral. This solution enables businesses to monitor brand perception, identify product issues, and understand customer satisfaction in real-time.

### How It Works

The project follows a classic NLP pipeline. First, I preprocessed the raw text data by cleaning it and preparing it for analysis. Next, I used the **TF-IDF (Term Frequency-Inverse Document Frequency)** technique to convert the text into numerical features that a machine learning model can understand. I then trained and compared three different models (Naive Bayes, Logistic Regression, and Random Forest) to find the most accurate classifier. The final model can instantly categorize new reviews, providing immediate feedback to the business.

### Key Features & Results

-   **Accurate Sentiment Classification:** The model achieves **100% accuracy** on the test dataset, demonstrating its effectiveness in understanding and classifying sentiment.
-   **Real-Time Insights:** The system can process new reviews in real-time, allowing for immediate responses to customer feedback.
-   **Key Term Identification:** The analysis identified the most common words associated with positive and negative reviews, providing clear insights into what customers like and dislike.
-   **Scalable Solution:** The pipeline is designed to handle large volumes of text data, making it suitable for businesses of any size.

[View on GitHub →](https://github.com/yourusername/data-science-portfolio/tree/main/project3_sentiment_analysis)

---

## Project 4: Image Classification with Deep Learning

### Title: Building a Fashion Item Classifier with a Custom CNN

![Image Classification Dashboard](../project4_image_classification/screenshots/classification_dashboard.png)

### Overview

Computer vision is revolutionizing industries from retail to manufacturing. This project demonstrates my expertise in deep learning by building a **Convolutional Neural Network (CNN)** from scratch to classify images of fashion items. Using the popular Fashion MNIST dataset, the model can accurately identify 10 different types of clothing and accessories, showcasing the power of deep learning for automated image recognition.

### How It Works

The core of this project is a custom-built CNN using TensorFlow and Keras. The architecture consists of multiple convolutional layers to detect visual features like edges and textures, followed by max-pooling layers for down-sampling and dense layers for the final classification. To ensure robust performance, I incorporated techniques like **Batch Normalization** to stabilize training and **Dropout** to prevent overfitting. The model was trained on 60,000 images and evaluated on a separate test set of 10,000 images.

### Key Features & Results

-   **High Classification Accuracy:** The model achieved a **test accuracy of 85.1%**, correctly identifying the majority of fashion items.
-   **Custom CNN Architecture:** The project demonstrates the ability to design and build a deep learning model from the ground up.
-   **Advanced Training Techniques:** The use of Batch Normalization and Dropout showcases knowledge of modern deep learning best practices.
-   **Detailed Performance Analysis:** The evaluation includes a confusion matrix and per-class accuracy metrics, providing deep insights into the model's strengths and weaknesses.

[View on GitHub →](https://github.com/yourusername/data-science-portfolio/tree/main/project4_image_classification)

---

## Project 5: Stock Price Forecasting with Time Series Analysis

### Title: Predicting Market Trends with Time Series Forecasting

![Stock Forecasting Dashboard](../project5_stock_forecasting/screenshots/forecasting_dashboard.png)

### Overview

Financial markets are notoriously complex, but time series analysis can help uncover patterns and predict future trends. This project applies time series forecasting techniques to predict stock prices based on historical data. By analyzing trends, seasonality, and volatility, the model provides a data-driven approach to understanding market behavior and informing investment strategies.

### How It Works

This project uses a three-year dataset of historical stock prices. I began by conducting a thorough exploratory data analysis to identify underlying patterns, including long-term trends and seasonal cycles. I then engineered several features, such as moving averages and daily returns, to capture the dynamics of the stock's performance. Finally, I implemented a **30-day moving average** forecasting model and evaluated its performance against actual stock prices, achieving a Mean Absolute Percentage Error (MAPE) of just 6.33%.

### Key Features & Results

-   **Accurate Short-Term Forecasts:** The model predicts future stock prices with a low **6.33% error rate**, making it a useful tool for short-term planning.
-   **Trend and Seasonality Detection:** The analysis successfully identified a long-term upward trend and clear seasonal patterns in the stock's price.
-   **Volatility Analysis:** The project includes an analysis of rolling volatility, providing insights into the stock's risk profile.
-   **Foundation for Advanced Models:** This project lays the groundwork for more complex forecasting models, such as ARIMA and LSTMs, demonstrating a strong understanding of time series fundamentals.

[View on GitHub →](https://github.com/yourusername/data-science-portfolio/tree/main/project5_stock_forecasting)

---

## Project 6: Healthcare Analytics - Predicting Diabetes Risk

### Title: Improving Patient Outcomes with Predictive Healthcare Analytics

![Healthcare Analytics Dashboard](../project6_healthcare_prediction/screenshots/healthcare_dashboard.png)

### Overview

Machine learning is transforming healthcare by enabling early disease detection and personalized medicine. This project focuses on one of the most critical areas of public health: predicting the onset of diabetes. Using a dataset of patient health metrics, I developed a machine learning model that can identify individuals at high risk of developing diabetes with high accuracy. This allows for early, preventive interventions that can improve patient outcomes and reduce healthcare costs.

### How It Works

I built a complete machine learning pipeline to tackle this problem. After analyzing the data to understand the relationships between different health metrics and diabetes, I trained and evaluated three different classification models: Logistic Regression, Random Forest, and Gradient Boosting. The **Gradient Boosting** model emerged as the top performer, with an impressive ability to distinguish between high-risk and low-risk patients. The analysis also included a feature importance evaluation, which identified the most significant clinical predictors of diabetes.

### Key Features & Results

-   **High Predictive Accuracy:** The model predicts diabetes risk with **87.5% accuracy** and an **AUC-ROC score of 0.94**, making it a highly reliable clinical support tool.
-   **Identified Key Risk Factors:** The analysis confirmed that **glucose level, BMI, and age** are the most critical predictors of diabetes, providing a clear focus for preventive care.
-   **Actionable Clinical Insights:** The model can be used by healthcare providers to stratify patients by risk level and recommend personalized interventions.
-   **Demonstrated Healthcare Expertise:** This project showcases the ability to apply data science to a complex and sensitive domain, with a focus on improving patient lives.

[View on GitHub →](https://github.com/yourusername/data-science-portfolio/tree/main/project6_healthcare_prediction)
