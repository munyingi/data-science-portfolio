"""
Healthcare Analytics - Disease Prediction
Predicting diabetes risk using patient health metrics
Author: Samwel Munyingi
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_curve, roc_auc_score

print("="*70)
print("HEALTHCARE ANALYTICS - DIABETES RISK PREDICTION")
print("="*70)

# ============================================================================
# GENERATE SYNTHETIC HEALTH DATA
# ============================================================================
print("\n" + "="*70)
print("GENERATING HEALTHCARE DATA")
print("="*70)

np.random.seed(42)

n_samples = 1000

# Generate features
age = np.random.randint(20, 80, n_samples)
bmi = np.random.normal(28, 7, n_samples)
bmi = np.clip(bmi, 15, 50)
glucose = np.random.normal(120, 30, n_samples)
glucose = np.clip(glucose, 70, 200)
blood_pressure = np.random.normal(80, 15, n_samples)
blood_pressure = np.clip(blood_pressure, 50, 120)
insulin = np.random.normal(100, 50, n_samples)
insulin = np.clip(insulin, 0, 300)
skin_thickness = np.random.normal(25, 10, n_samples)
skin_thickness = np.clip(skin_thickness, 0, 60)
pregnancies = np.random.randint(0, 15, n_samples)
diabetes_pedigree = np.random.uniform(0.1, 2.5, n_samples)

# Generate outcome based on risk factors
risk_score = (
    (age > 45) * 0.2 +
    (bmi > 30) * 0.3 +
    (glucose > 140) * 0.4 +
    (blood_pressure > 90) * 0.15 +
    (insulin > 150) * 0.1 +
    (diabetes_pedigree > 1.0) * 0.2 +
    np.random.uniform(0, 0.3, n_samples)
)

outcome = (risk_score > 0.6).astype(int)

# Create DataFrame
df = pd.DataFrame({
    'Pregnancies': pregnancies,
    'Glucose': glucose,
    'BloodPressure': blood_pressure,
    'SkinThickness': skin_thickness,
    'Insulin': insulin,
    'BMI': bmi,
    'DiabetesPedigreeFunction': diabetes_pedigree,
    'Age': age,
    'Outcome': outcome
})

# Save data
df.to_csv('../data/diabetes_data.csv', index=False)
print(f"✓ Healthcare data generated: {len(df)} patients")
print(f"  Diabetic: {outcome.sum()} ({outcome.sum()/len(df)*100:.1f}%)")
print(f"  Non-diabetic: {len(df)-outcome.sum()} ({(len(df)-outcome.sum())/len(df)*100:.1f}%)")

# ============================================================================
# EXPLORATORY DATA ANALYSIS
# ============================================================================
print("\n" + "="*70)
print("EXPLORATORY DATA ANALYSIS")
print("="*70)

print("\n📊 Summary Statistics:")
print(df.describe())

# ============================================================================
# VISUALIZATIONS
# ============================================================================
print("\n" + "="*70)
print("GENERATING VISUALIZATIONS")
print("="*70)

# 1. Outcome distribution
fig, axes = plt.subplots(1, 2, figsize=(14, 5))

outcome_counts = df['Outcome'].value_counts()
colors = ['#2ecc71', '#e74c3c']

axes[0].bar(['Non-Diabetic', 'Diabetic'], outcome_counts.values, 
           color=colors, edgecolor='black', linewidth=1.5)
axes[0].set_title('Diabetes Outcome Distribution', fontsize=14, fontweight='bold')
axes[0].set_ylabel('Count', fontsize=12)
axes[0].grid(axis='y', alpha=0.3)
for i, v in enumerate(outcome_counts.values):
    axes[0].text(i, v, str(v), ha='center', va='bottom', fontweight='bold')

axes[1].pie(outcome_counts.values, labels=['Non-Diabetic', 'Diabetic'], 
           autopct='%1.1f%%', colors=colors, startangle=90,
           textprops={'fontsize': 12, 'fontweight': 'bold'})
axes[1].set_title('Diabetes Proportion', fontsize=14, fontweight='bold')

plt.tight_layout()
plt.savefig('../visualizations/outcome_distribution.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Outcome distribution saved")

# 2. Feature distributions by outcome
features = ['Glucose', 'BMI', 'Age', 'BloodPressure']
fig, axes = plt.subplots(2, 2, figsize=(16, 12))

for idx, feature in enumerate(features):
    ax = axes[idx//2, idx%2]
    
    df[df['Outcome']==0][feature].hist(bins=30, ax=ax, alpha=0.7, 
                                        color='#2ecc71', label='Non-Diabetic')
    df[df['Outcome']==1][feature].hist(bins=30, ax=ax, alpha=0.7, 
                                        color='#e74c3c', label='Diabetic')
    
    ax.set_title(f'{feature} Distribution by Outcome', fontsize=12, fontweight='bold')
    ax.set_xlabel(feature, fontsize=11)
    ax.set_ylabel('Frequency', fontsize=11)
    ax.legend()
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature distributions saved")

# 3. Correlation heatmap
fig, ax = plt.subplots(figsize=(12, 10))
corr = df.corr()
sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
           square=True, linewidths=1, cbar_kws={"shrink": 0.8}, ax=ax)
ax.set_title('Feature Correlation Matrix', fontsize=14, fontweight='bold')
plt.tight_layout()
plt.savefig('../visualizations/correlation_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Correlation matrix saved")

# 4. Box plots
fig, axes = plt.subplots(2, 4, figsize=(20, 10))
features_all = ['Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
               'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age']

for idx, feature in enumerate(features_all):
    ax = axes[idx//4, idx%4]
    df.boxplot(column=feature, by='Outcome', ax=ax, patch_artist=True,
              boxprops=dict(facecolor='#3498db', alpha=0.7),
              medianprops=dict(color='red', linewidth=2))
    ax.set_title(f'{feature} by Outcome', fontsize=11, fontweight='bold')
    ax.set_xlabel('Outcome (0=No, 1=Yes)', fontsize=10)
    ax.set_ylabel(feature, fontsize=10)
    plt.sca(ax)
    plt.xticks([1, 2], ['Non-Diabetic', 'Diabetic'])

plt.suptitle('Feature Comparison by Diabetes Outcome', fontsize=16, fontweight='bold', y=1.00)
plt.tight_layout()
plt.savefig('../visualizations/feature_boxplots.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Feature boxplots saved")

# ============================================================================
# MODEL DEVELOPMENT
# ============================================================================
print("\n" + "="*70)
print("MODEL DEVELOPMENT")
print("="*70)

# Prepare data
X = df.drop('Outcome', axis=1)
y = df['Outcome']

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

# Scale features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

print(f"✓ Data split: {len(X_train)} train, {len(X_test)} test")

# Train models
models = {
    'Logistic Regression': LogisticRegression(max_iter=1000, random_state=42),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=42)
}

results = {}

for name, model in models.items():
    print(f"\nTraining {name}...")
    model.fit(X_train_scaled, y_train)
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled)[:, 1]
    
    accuracy = accuracy_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_pred_proba)
    
    results[name] = {
        'model': model,
        'predictions': y_pred,
        'probabilities': y_pred_proba,
        'accuracy': accuracy,
        'auc': auc
    }
    
    print(f"  Accuracy: {accuracy:.4f}")
    print(f"  AUC-ROC: {auc:.4f}")

# ============================================================================
# MODEL EVALUATION
# ============================================================================
print("\n" + "="*70)
print("MODEL EVALUATION")
print("="*70)

# Model comparison
comparison_df = pd.DataFrame({
    'Model': list(results.keys()),
    'Accuracy': [results[m]['accuracy'] for m in results.keys()],
    'AUC-ROC': [results[m]['auc'] for m in results.keys()]
}).sort_values('AUC-ROC', ascending=False)

print("\nModel Performance:")
print(comparison_df.to_string(index=False))

# Best model
best_model_name = comparison_df.iloc[0]['Model']
best_model = results[best_model_name]['model']
best_predictions = results[best_model_name]['predictions']
best_probabilities = results[best_model_name]['probabilities']

print(f"\n✓ Best Model: {best_model_name}")

# Visualizations
fig, axes = plt.subplots(1, 2, figsize=(16, 6))

# Model comparison
x = np.arange(len(comparison_df))
width = 0.35

bars1 = axes[0].bar(x - width/2, comparison_df['Accuracy'], width, 
                    label='Accuracy', color='#3498db')
bars2 = axes[0].bar(x + width/2, comparison_df['AUC-ROC'], width, 
                    label='AUC-ROC', color='#2ecc71')

axes[0].set_xlabel('Model', fontsize=12)
axes[0].set_ylabel('Score', fontsize=12)
axes[0].set_title('Model Performance Comparison', fontsize=14, fontweight='bold')
axes[0].set_xticks(x)
axes[0].set_xticklabels(comparison_df['Model'], rotation=15, ha='right')
axes[0].legend()
axes[0].set_ylim([0, 1.0])
axes[0].grid(axis='y', alpha=0.3)

# ROC curves
for name in results.keys():
    fpr, tpr, _ = roc_curve(y_test, results[name]['probabilities'])
    axes[1].plot(fpr, tpr, label=f"{name} (AUC = {results[name]['auc']:.3f})", linewidth=2)

axes[1].plot([0, 1], [0, 1], 'k--', label='Random Classifier', linewidth=2)
axes[1].set_title('ROC Curves', fontsize=14, fontweight='bold')
axes[1].set_xlabel('False Positive Rate', fontsize=12)
axes[1].set_ylabel('True Positive Rate', fontsize=12)
axes[1].legend(loc='lower right')
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('../visualizations/model_comparison.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Model comparison saved")

# Confusion matrix
cm = confusion_matrix(y_test, best_predictions)

fig, ax = plt.subplots(figsize=(10, 8))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
           xticklabels=['Non-Diabetic', 'Diabetic'],
           yticklabels=['Non-Diabetic', 'Diabetic'],
           cbar_kws={'label': 'Count'}, ax=ax)
ax.set_title(f'Confusion Matrix - {best_model_name}', fontsize=14, fontweight='bold')
ax.set_ylabel('Actual', fontsize=12)
ax.set_xlabel('Predicted', fontsize=12)
plt.tight_layout()
plt.savefig('../visualizations/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()
print("✓ Confusion matrix saved")

# Feature importance (for tree-based models)
if hasattr(best_model, 'feature_importances_'):
    feature_importance = pd.DataFrame({
        'Feature': X.columns,
        'Importance': best_model.feature_importances_
    }).sort_values('Importance', ascending=False)
    
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.barh(range(len(feature_importance)), feature_importance['Importance'], color='#3498db')
    ax.set_yticks(range(len(feature_importance)))
    ax.set_yticklabels(feature_importance['Feature'])
    ax.set_xlabel('Importance', fontsize=12)
    ax.set_title('Feature Importance', fontsize=14, fontweight='bold')
    ax.invert_yaxis()
    ax.grid(axis='x', alpha=0.3)
    plt.tight_layout()
    plt.savefig('../visualizations/feature_importance.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✓ Feature importance saved")

# ============================================================================
# SUMMARY
# ============================================================================
print("\n" + "="*70)
print("ANALYSIS SUMMARY")
print("="*70)

print(f"\n📊 Dataset Overview:")
print(f"   Total Patients: {len(df)}")
print(f"   Diabetic: {outcome.sum()} ({outcome.sum()/len(df)*100:.1f}%)")
print(f"   Non-Diabetic: {len(df)-outcome.sum()} ({(len(df)-outcome.sum())/len(df)*100:.1f}%)")

print(f"\n🎯 Best Model Performance:")
print(f"   Model: {best_model_name}")
print(f"   Accuracy: {results[best_model_name]['accuracy']:.2%}")
print(f"   AUC-ROC: {results[best_model_name]['auc']:.4f}")

print(f"\n💡 Key Insights:")
print("   • Glucose level is the strongest predictor of diabetes")
print("   • BMI and age are significant risk factors")
print("   • Model achieves high accuracy in risk prediction")
print("   • Early detection enables preventive interventions")
print("   • Feature importance guides clinical decision-making")

print(f"\n🏥 Clinical Implications:")
print("   • High-risk patients can be identified early")
print("   • Targeted interventions for modifiable risk factors")
print("   • Cost-effective screening strategy")
print("   • Improved patient outcomes through prevention")

print("\n" + "="*70)
print("ANALYSIS COMPLETE - All visualizations saved!")
print("="*70)

# Save model
import joblib
joblib.dump(best_model, '../src/diabetes_model.pkl')
joblib.dump(scaler, '../src/scaler.pkl')
print("\n✓ Model and scaler saved for deployment")
