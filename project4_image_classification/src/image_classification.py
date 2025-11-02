"""
Image Classification with Deep Learning
CNN-based image classification using Fashion MNIST
Author: Samwel Munyingi
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("IMAGE CLASSIFICATION WITH DEEP LEARNING")
print("="*70)

# Install TensorFlow if needed
import subprocess
import sys

try:
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"\n✓ TensorFlow {tf.__version__} loaded")
except ImportError:
    print("\nInstalling TensorFlow...")
    subprocess.check_call([sys.executable, "-m", "pip", "install", "-q", "tensorflow"])
    import tensorflow as tf
    from tensorflow import keras
    from tensorflow.keras import layers
    print(f"✓ TensorFlow {tf.__version__} installed and loaded")

from sklearn.metrics import classification_report, confusion_matrix

# ============================================================================
# LOAD DATASET
# ============================================================================
print("\n" + "="*70)
print("LOADING FASHION MNIST DATASET")
print("="*70)

# Load Fashion MNIST
(X_train, y_train), (X_test, y_test) = keras.datasets.fashion_mnist.load_data()

print(f"✓ Dataset loaded successfully")
print(f"  Training samples: {X_train.shape[0]}")
print(f"  Test samples: {X_test.shape[0]}")
print(f"  Image shape: {X_train.shape[1]}x{X_train.shape[2]}")

# Class names
class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

# ============================================================================
# DATA PREPROCESSING
# ============================================================================
print("\n" + "="*70)
print("DATA PREPROCESSING")
print("="*70)

# Normalize pixel values
X_train_normalized = X_train / 255.0
X_test_normalized = X_test / 255.0

# Reshape for CNN (add channel dimension)
X_train_cnn = X_train_normalized.reshape(-1, 28, 28, 1)
X_test_cnn = X_test_normalized.reshape(-1, 28, 28, 1)

print("✓ Data normalized and reshaped")
print(f"  Training shape: {X_train_cnn.shape}")
print(f"  Test shape: {X_test_cnn.shape}")

# ============================================================================
# VISUALIZE SAMPLE IMAGES
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# Display sample images
fig, axes = plt.subplots(5, 10, figsize=(15, 8))
for i in range(50):
    ax = axes[i//10, i%10]
    ax.imshow(X_train[i], cmap='gray')
    ax.set_title(class_names[y_train[i]], fontsize=8)
    ax.axis('off')

plt.suptitle('Sample Fashion MNIST Images', fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../visualizations/sample_images.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Sample images saved")

# Class distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

train_counts = np.bincount(y_train)
test_counts = np.bincount(y_test)

axes[0].bar(range(10), train_counts, color='#3498db', edgecolor='black', linewidth=1.5)
axes[0].set_xticks(range(10))
axes[0].set_xticklabels(class_names, rotation=45, ha='right')
axes[0].set_title('Training Set Class Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)

axes[1].bar(range(10), test_counts, color='#2ecc71', edgecolor='black', linewidth=1.5)
axes[1].set_xticks(range(10))
axes[1].set_xticklabels(class_names, rotation=45, ha='right')
axes[1].set_title('Test Set Class Distribution', fontsize=14, fontweight='bold')
axes[1].set_ylabel('Count', fontsize=12)
axes[1].grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/class_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Class distribution saved")

# ============================================================================
# BUILD CNN MODEL
# ============================================================================
print("\n" + "="*70)
print("BUILDING CNN MODEL")
print("="*70)

model = keras.Sequential([
    # First Convolutional Block
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Second Convolutional Block
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.BatchNormalization(),
    
    # Third Convolutional Block
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.BatchNormalization(),
    
    # Flatten and Dense Layers
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(64, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(10, activation='softmax')
])

model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

print("✓ CNN model built successfully")
model.summary()

# ============================================================================
# TRAIN MODEL
# ============================================================================
print("\n" + "="*70)
print("TRAINING MODEL")
print("="*70)

# Use a subset for faster training
train_size = 10000
test_size = 2000

history = model.fit(
    X_train_cnn[:train_size], y_train[:train_size],
    epochs=10,
    batch_size=128,
    validation_split=0.2,
    verbose=1
)

print("\n✓ Model training completed")

# ============================================================================
# EVALUATE MODEL
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

test_loss, test_accuracy = model.evaluate(X_test_cnn[:test_size], y_test[:test_size], verbose=0)
print(f"\nTest Accuracy: {test_accuracy:.4f}")
print(f"Test Loss: {test_loss:.4f}")

# Predictions
y_pred = model.predict(X_test_cnn[:test_size], verbose=0)
y_pred_classes = np.argmax(y_pred, axis=1)

# ============================================================================
# VISUALIZE RESULTS
# ============================================================================

# Training history
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

axes[0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2, color='#3498db')
axes[0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2, color='#2ecc71')
axes[0].set_title('Model Accuracy Over Epochs', fontsize=14, fontweight='bold')
axes[0].set_xlabel('Epoch', fontsize=12)
axes[0].set_ylabel('Accuracy', fontsize=12)
axes[0].legend()
axes[0].grid(alpha=0.3)

axes[1].plot(history.history['loss'], label='Training Loss', linewidth=2, color='#e74c3c')
axes[1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2, color='#f39c12')
axes[1].set_title('Model Loss Over Epochs', fontsize=14, fontweight='bold')
axes[1].set_xlabel('Epoch', fontsize=12)
axes[1].set_ylabel('Loss', fontsize=12)
axes[1].legend()
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/training_history.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Training history saved")

# Confusion matrix
cm = confusion_matrix(y_test[:test_size], y_pred_classes)

fig, ax = plt.subplots(figsize=(12, 10))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
           xticklabels=class_names, yticklabels=class_names,
           cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title('Confusion Matrix - CNN Model', fontsize=14, fontweight='bold')
ax.set_ylabel('True Label', fontsize=12)
ax.set_xlabel('Predicted Label', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.yticks(rotation=0)
plt.tight_layout()
plt.savefig('../visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix saved")

# Prediction examples
fig, axes = plt.subplots(4, 5, figsize=(15, 12))
for i in range(20):
    ax = axes[i//5, i%5]
    ax.imshow(X_test[i], cmap='gray')
    
    true_label = class_names[y_test[i]]
    pred_label = class_names[y_pred_classes[i]]
    confidence = np.max(y_pred[i]) * 100
    
    color = 'green' if true_label == pred_label else 'red'
    ax.set_title(f'True: {true_label}\nPred: {pred_label}\n({confidence:.1f}%)', 
                fontsize=9, color=color, fontweight='bold')
    ax.axis('off')

plt.suptitle('Sample Predictions (Green=Correct, Red=Incorrect)', 
            fontsize=16, fontweight='bold', y=0.98)
plt.tight_layout()
plt.savefig('../visualizations/prediction_examples.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Prediction examples saved")

# Per-class accuracy
class_accuracy = []
for i in range(10):
    mask = y_test[:test_size] == i
    if mask.sum() > 0:
        acc = (y_pred_classes[mask] == i).sum() / mask.sum()
        class_accuracy.append(acc)
    else:
        class_accuracy.append(0)

fig, ax = plt.subplots(figsize=(12, 6))
bars = ax.bar(range(10), class_accuracy, color='#3498db', edgecolor='black', linewidth=1.5)
ax.set_xticks(range(10))
ax.set_xticklabels(class_names, rotation=45, ha='right')
ax.set_title('Per-Class Accuracy', fontsize=14, fontweight='bold')
ax.set_ylabel('Accuracy', fontsize=12)
ax.set_ylim([0, 1.0])
ax.grid(axis='y', alpha=0.3)

for i, bar in enumerate(bars):
    height = bar.get_height()
    ax.text(bar.get_x() + bar.get_width()/2., height,
           f'{height:.2%}', ha='center', va='bottom', fontweight='bold')

plt.tight_layout()
plt.savefig('../visualizations/per_class_accuracy.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Per-class accuracy saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\n📊 Model Architecture:")
print(f"   Type: Convolutional Neural Network (CNN)")
print(f"   Layers: 3 Conv blocks + 2 Dense layers")
print(f"   Parameters: {model.count_params():,}")

print(f"\n🎯 Model Performance:")
print(f"   Test Accuracy: {test_accuracy:.2%}")
print(f"   Test Loss: {test_loss:.4f}")

print(f"\n🏆 Best Performing Classes:")
best_classes = np.argsort(class_accuracy)[-3:][::-1]
for idx in best_classes:
    print(f"   {class_names[idx]}: {class_accuracy[idx]:.2%}")

print(f"\n💡 Key Insights:")
print("   • CNN effectively learns spatial features from images")
print("   • Batch normalization improves training stability")
print("   • Dropout prevents overfitting")
print("   • Some classes (e.g., shirts vs. t-shirts) are harder to distinguish")
print("   • Model achieves high accuracy with relatively simple architecture")

print("\n" + "="*70)
print("ANALYSIS COMPLETE - All visualizations saved!")
print("="*70)

# Save model
model.save('../src/fashion_mnist_model.h5')
print("\n✓ Model saved for deployment")
