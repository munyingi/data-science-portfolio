
# Project 4: Image Classification with Deep Learning

## Overview

This project demonstrates the application of deep learning for image classification. We build a Convolutional Neural Network (CNN) to classify images from the Fashion MNIST dataset, which consists of 70,000 grayscale images in 10 different fashion categories. The goal is to achieve high classification accuracy and showcase expertise in building and training deep learning models from scratch.

### Key Objectives

- **Build** a custom CNN architecture for image classification.
- **Train** the model on the Fashion MNIST dataset.
- **Evaluate** model performance using key metrics.
- **Visualize** training progress, results, and model predictions.

---

## Key Findings & Visualizations

1.  **High Accuracy**: The CNN model achieved a test accuracy of **85.1%**, demonstrating its effectiveness in learning complex patterns from image data.
2.  **Class-Specific Performance**: The model performed exceptionally well on distinct classes like "Ankle boot" and "Trouser" but found it more challenging to distinguish between similar items like "T-shirt/top" and "Shirt".
3.  **Effective Architecture**: The combination of convolutional layers, max-pooling, and dropout proved to be a robust architecture for this task.

| Metric | Value |
| :--- | :--- |
| **Test Accuracy** | 85.10% |
| **Test Loss** | 0.4581 |
| **Best Performing Class** | Sneaker (99.5%) |
| **Worst Performing Class** | Shirt |

![Training History](../visualizations/training_history.png)
*Figure 1: Model accuracy and loss over training epochs, showing stable learning and convergence.*

![Confusion Matrix](../visualizations/confusion_matrix.png)
*Figure 2: Confusion matrix of the model's predictions on the test set. The diagonal represents correct classifications.*

---

## Technical Implementation

### Deep Learning Model

The core of this project is a CNN built with **TensorFlow** and **Keras**. The architecture includes:

-   Three convolutional blocks with increasing filter sizes (32, 64, 128).
-   **MaxPooling** layers for down-sampling.
-   **Batch Normalization** for stabilizing training.
-   **Dropout** layers to prevent overfitting.
-   A final set of dense layers for classification.

### Technology Stack

- **Python**: Core programming language.
- **TensorFlow/Keras**: For building and training the deep learning model.
- **NumPy**: For numerical operations.
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
    cd project4_image_classification
    ```

2.  **Install dependencies:**
    ```bash
    pip install tensorflow numpy matplotlib seaborn
    ```

### Running the Analysis

To train the model and generate all visualizations:

```bash
python src/image_classification.py
```

This script will download the dataset, build the model, train it, and save all output files.

---

## Business Impact

This project demonstrates the capability to build and deploy advanced deep learning solutions for real-world problems. The applications of image classification are vast and include:

- **E-commerce**: Automated product tagging and categorization.
- **Retail**: In-store item recognition and inventory management.
- **Manufacturing**: Visual quality control and defect detection.
- **Security**: Facial recognition and object detection systems.

By showcasing expertise in this area, it highlights the ability to tackle complex computer vision tasks that can drive significant business value.
