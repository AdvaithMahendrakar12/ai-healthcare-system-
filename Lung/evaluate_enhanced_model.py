"""
Enhanced Model Evaluation Script
=================================
Comprehensive evaluation of trained pneumonia detection model with advanced metrics,
visualizations, and interpretability analysis.

Author: Medical AI Team
Date: January 2026
"""

import tensorflow as tf
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import (confusion_matrix, classification_report, 
                            roc_curve, auc, precision_recall_curve,
                            average_precision_score)
import pandas as pd
import os
import json
from datetime import datetime

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# Configuration
IMG_SIZE = (320, 320)
BATCH_SIZE = 16
MODEL_PATH = "models/lung_model.h5"
TEST_DATA_DIR = "dataset/chest_xray/test"
RESULTS_DIR = "results"
PLOTS_DIR = "evaluation_plots"

# Create directories
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(PLOTS_DIR, exist_ok=True)

print("\n" + "="*80)
print("🔍 ENHANCED MODEL EVALUATION PIPELINE")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"🤖 Model: {MODEL_PATH}")
print(f"📂 Test Data: {TEST_DATA_DIR}")
print("="*80)

# ==========================================
# 1. LOAD MODEL
# ==========================================

print("\n📥 Loading model...")
try:
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"✅ Model loaded successfully")
    print(f"   Input shape: {model.input_shape}")
    print(f"   Output shape: {model.output_shape}")
    print(f"   Total parameters: {model.count_params():,}")
except Exception as e:
    print(f"❌ Error loading model: {e}")
    exit(1)

# ==========================================
# 2. PREPARE TEST DATA
# ==========================================

print("\n📊 Preparing test dataset...")

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1./255)

test_generator = test_datagen.flow_from_directory(
    TEST_DATA_DIR,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode='binary',
    shuffle=False
)

print(f"✅ Test samples: {test_generator.samples}")
print(f"✅ Class indices: {test_generator.class_indices}")
print(f"✅ Batch size: {BATCH_SIZE}")

# ==========================================
# 3. GENERATE PREDICTIONS
# ==========================================

print("\n🔮 Generating predictions...")
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=1)
y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
y_true = test_generator.classes
y_pred_probs_flat = y_pred_probs.flatten()

print(f"✅ Generated {len(y_pred_probs)} predictions")
print(f"   Prediction range: [{y_pred_probs_flat.min():.4f}, {y_pred_probs_flat.max():.4f}]")
print(f"   Mean prediction: {y_pred_probs_flat.mean():.4f}")

# ==========================================
# 4. CLASSIFICATION REPORT
# ==========================================

print("\n" + "="*80)
print("📋 CLASSIFICATION REPORT")
print("="*80)

class_names = list(test_generator.class_indices.keys())
report = classification_report(y_true, y_pred_classes, 
                               target_names=class_names,
                               digits=4)
print(report)

# Save report
report_path = f'{RESULTS_DIR}/classification_report_detailed.txt'
with open(report_path, 'w') as f:
    f.write("="*80 + "\n")
    f.write("COMPREHENSIVE CLASSIFICATION REPORT\n")
    f.write("Pneumonia Detection from Chest X-Rays\n")
    f.write("="*80 + "\n")
    f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write(f"Model: {MODEL_PATH}\n")
    f.write(f"Test Samples: {test_generator.samples}\n")
    f.write("="*80 + "\n\n")
    f.write(report)

print(f"✅ Saved: {report_path}")

# ==========================================
# 5. CONFUSION MATRIX
# ==========================================

print("\n📊 Computing confusion matrix...")

cm = confusion_matrix(y_true, y_pred_classes)
tn, fp, fn, tp = cm.ravel()

fig, axes = plt.subplots(1, 3, figsize=(20, 6))

# Plot 1: Confusion Matrix (Counts)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Count'}, ax=axes[0], 
            annot_kws={"size": 16, "weight": "bold"})
axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
axes[0].set_ylabel('True Label', fontsize=12)
axes[0].set_xlabel('Predicted Label', fontsize=12)

# Add percentages
cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        axes[0].text(j + 0.5, i + 0.75, f'({cm_percent[i, j]*100:.1f}%)',
                    ha='center', va='center', fontsize=11, color='red', fontweight='bold')

# Plot 2: Normalized Confusion Matrix
sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='RdYlGn', 
            xticklabels=class_names, yticklabels=class_names,
            cbar_kws={'label': 'Percentage'}, ax=axes[1],
            annot_kws={"size": 16, "weight": "bold"})
axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
axes[1].set_ylabel('True Label', fontsize=12)
axes[1].set_xlabel('Predicted Label', fontsize=12)

# Plot 3: Confusion Matrix Breakdown
breakdown_data = {
    'True\nNegatives': tn,
    'False\nPositives': fp,
    'False\nNegatives': fn,
    'True\nPositives': tp
}
colors = ['#2ecc71', '#e74c3c', '#e74c3c', '#2ecc71']
bars = axes[2].bar(list(breakdown_data.keys()), list(breakdown_data.values()),
                   color=colors, edgecolor='black', linewidth=2, alpha=0.7)
axes[2].set_ylabel('Count', fontsize=12, fontweight='bold')
axes[2].set_title('Prediction Breakdown', fontsize=14, fontweight='bold')
axes[2].grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars, breakdown_data.values()):
    height = bar.get_height()
    axes[2].text(bar.get_x() + bar.get_width()/2., height + 10,
                f'{int(value)}',
                ha='center', va='bottom', fontsize=13, fontweight='bold')

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/confusion_matrix_comprehensive.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {PLOTS_DIR}/confusion_matrix_comprehensive.png")

# ==========================================
# 6. ROC CURVE & AUC
# ==========================================

print("\n📈 Computing ROC curve...")

fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs_flat)
roc_auc = auc(fpr, tpr)

# Find optimal threshold (Youden's Index)
optimal_idx = np.argmax(tpr - fpr)
optimal_threshold = thresholds[optimal_idx]
optimal_tpr = tpr[optimal_idx]
optimal_fpr = fpr[optimal_idx]

plt.figure(figsize=(10, 8))
plt.plot(fpr, tpr, color='#e74c3c', lw=3.5, 
         label=f'ROC Curve (AUC = {roc_auc:.4f})')
plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2.5, linestyle='--', 
         label='Random Classifier (AUC = 0.5000)')
plt.scatter([optimal_fpr], [optimal_tpr], s=300, c='#2ecc71', 
            marker='*', edgecolors='black', linewidth=2.5,
            label=f'Optimal Threshold = {optimal_threshold:.4f}', zorder=5)

# Add annotations
plt.annotate(f'TPR: {optimal_tpr:.3f}\nFPR: {optimal_fpr:.3f}',
            xy=(optimal_fpr, optimal_tpr), xytext=(optimal_fpr + 0.15, optimal_tpr - 0.15),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=1.5))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate (1 - Specificity)', fontsize=13, fontweight='bold')
plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
plt.title('ROC Curve - Pneumonia Detection Model', fontsize=16, fontweight='bold')
plt.legend(loc="lower right", fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/roc_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {PLOTS_DIR}/roc_curve.png")

# ==========================================
# 7. PRECISION-RECALL CURVE
# ==========================================

print("\n📉 Computing Precision-Recall curve...")

precision, recall, pr_thresholds = precision_recall_curve(y_true, y_pred_probs_flat)
average_precision = average_precision_score(y_true, y_pred_probs_flat)

# Find optimal threshold for PR curve (F1 score)
f1_scores = 2 * (precision[:-1] * recall[:-1]) / (precision[:-1] + recall[:-1] + 1e-10)
optimal_pr_idx = np.argmax(f1_scores)
optimal_pr_threshold = pr_thresholds[optimal_pr_idx]
optimal_precision = precision[optimal_pr_idx]
optimal_recall = recall[optimal_pr_idx]
optimal_f1 = f1_scores[optimal_pr_idx]

plt.figure(figsize=(10, 8))
plt.plot(recall, precision, color='#3498db', lw=3.5, 
         label=f'PR Curve (AP = {average_precision:.4f})')
plt.scatter([optimal_recall], [optimal_precision], s=300, c='#e74c3c', 
            marker='*', edgecolors='black', linewidth=2.5,
            label=f'Optimal Threshold = {optimal_pr_threshold:.4f} (F1 = {optimal_f1:.4f})', zorder=5)

# Baseline (random classifier)
baseline = np.sum(y_true) / len(y_true)
plt.axhline(y=baseline, color='#95a5a6', linestyle='--', lw=2.5,
            label=f'Baseline (AP = {baseline:.4f})')

# Add annotations
plt.annotate(f'Precision: {optimal_precision:.3f}\nRecall: {optimal_recall:.3f}\nF1: {optimal_f1:.3f}',
            xy=(optimal_recall, optimal_precision), 
            xytext=(optimal_recall - 0.25, optimal_precision + 0.1),
            fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8),
            arrowprops=dict(arrowstyle='->', lw=1.5))

plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('Recall (Sensitivity)', fontsize=13, fontweight='bold')
plt.ylabel('Precision', fontsize=13, fontweight='bold')
plt.title('Precision-Recall Curve - Pneumonia Detection', fontsize=16, fontweight='bold')
plt.legend(loc="lower left", fontsize=11, framealpha=0.9)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/precision_recall_curve.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {PLOTS_DIR}/precision_recall_curve.png")

# ==========================================
# 8. DETAILED METRICS
# ==========================================

print("\n" + "="*80)
print("🎯 COMPREHENSIVE PERFORMANCE METRICS")
print("="*80)

# Calculate all metrics
accuracy = (tp + tn) / (tp + tn + fp + fn)
sensitivity = tp / (tp + fn)  # Recall, TPR
specificity = tn / (tn + fp)  # TNR
precision = tp / (tp + fp)     # PPV
f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
npv = tn / (tn + fn)          # Negative Predictive Value
fpr_val = fp / (fp + tn)      # False Positive Rate
fnr = fn / (fn + tp)          # False Negative Rate
fdr = fp / (fp + tp)          # False Discovery Rate
balanced_accuracy = (sensitivity + specificity) / 2
mcc = (tp * tn - fp * fn) / np.sqrt((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))

print(f"\n{'Primary Metrics:':<30}")
print(f"  {'Accuracy:':<25} {accuracy*100:>6.2f}%")
print(f"  {'Balanced Accuracy:':<25} {balanced_accuracy*100:>6.2f}%")
print(f"  {'AUC-ROC:':<25} {roc_auc:>6.4f}")
print(f"  {'Average Precision:':<25} {average_precision:>6.4f}")
print(f"  {'Matthews Corr. Coef.:':<25} {mcc:>6.4f}")

print(f"\n{'Positive Class (PNEUMONIA) Metrics:':<30}")
print(f"  {'Sensitivity (Recall/TPR):':<25} {sensitivity*100:>6.2f}%")
print(f"  {'Precision (PPV):':<25} {precision*100:>6.2f}%")
print(f"  {'F1-Score:':<25} {f1_score:>6.4f}")

print(f"\n{'Negative Class (NORMAL) Metrics:':<30}")
print(f"  {'Specificity (TNR):':<25} {specificity*100:>6.2f}%")
print(f"  {'NPV:':<25} {npv*100:>6.2f}%")

print(f"\n{'Error Rates:':<30}")
print(f"  {'False Positive Rate:':<25} {fpr_val*100:>6.2f}%")
print(f"  {'False Negative Rate:':<25} {fnr*100:>6.2f}%")
print(f"  {'False Discovery Rate:':<25} {fdr*100:>6.2f}%")

print(f"\n{'Optimal Thresholds:':<30}")
print(f"  {'ROC Threshold:':<25} {optimal_threshold:>6.4f}")
print(f"  {'PR Threshold:':<25} {optimal_pr_threshold:>6.4f}")

print(f"\n{'Confusion Matrix Values:':<30}")
print(f"  {'True Negatives:':<25} {tn:>6d}")
print(f"  {'False Positives:':<25} {fp:>6d}")
print(f"  {'False Negatives:':<25} {fn:>6d}")
print(f"  {'True Positives:':<25} {tp:>6d}")

print("="*80)

# ==========================================
# 9. METRICS VISUALIZATION
# ==========================================

print("\n📊 Creating metrics visualizations...")

fig, axes = plt.subplots(2, 2, figsize=(16, 12))

# Plot 1: Main Performance Metrics
metrics_dict = {
    'Accuracy': accuracy,
    'Sensitivity': sensitivity,
    'Specificity': specificity,
    'Precision': precision,
    'F1-Score': f1_score,
    'AUC-ROC': roc_auc,
    'Avg Precision': average_precision,
    'NPV': npv
}

colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', 
          '#9b59b6', '#1abc9c', '#34495e', '#16a085']
bars = axes[0, 0].barh(list(metrics_dict.keys()), list(metrics_dict.values()), 
                       color=colors, edgecolor='black', linewidth=1.5)
axes[0, 0].set_xlim([0, 1.05])
axes[0, 0].set_xlabel('Score', fontsize=12, fontweight='bold')
axes[0, 0].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
axes[0, 0].grid(True, alpha=0.3, axis='x')

for bar, value in zip(bars, metrics_dict.values()):
    axes[0, 0].text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                   f'{value:.3f}', va='center', fontsize=10, fontweight='bold')

# Plot 2: Error Analysis
error_metrics = {
    'False\nPositive\nRate': fpr_val,
    'False\nNegative\nRate': fnr,
    'False\nDiscovery\nRate': fdr
}

colors_error = ['#e74c3c', '#e67e22', '#c0392b']
bars2 = axes[0, 1].bar(list(error_metrics.keys()), list(error_metrics.values()),
                      color=colors_error, edgecolor='black', linewidth=1.5, alpha=0.7)
axes[0, 1].set_ylim([0, max(error_metrics.values()) * 1.2])
axes[0, 1].set_ylabel('Rate', fontsize=12, fontweight='bold')
axes[0, 1].set_title('Error Analysis', fontsize=14, fontweight='bold')
axes[0, 1].grid(True, alpha=0.3, axis='y')

for bar, value in zip(bars2, error_metrics.values()):
    height = bar.get_height()
    axes[0, 1].text(bar.get_x() + bar.get_width()/2., height + 0.01,
                   f'{value:.3f}\n({value*100:.1f}%)',
                   ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 3: Prediction Distribution
axes[1, 0].hist(y_pred_probs_flat[y_true == 0], bins=50, alpha=0.6, 
               label='NORMAL', color='#3498db', edgecolor='black')
axes[1, 0].hist(y_pred_probs_flat[y_true == 1], bins=50, alpha=0.6, 
               label='PNEUMONIA', color='#e74c3c', edgecolor='black')
axes[1, 0].axvline(x=0.5, color='black', linestyle='--', linewidth=2, label='Threshold (0.5)')
axes[1, 0].axvline(x=optimal_threshold, color='green', linestyle='--', linewidth=2, 
                  label=f'Optimal ({optimal_threshold:.3f})')
axes[1, 0].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
axes[1, 0].set_ylabel('Frequency', fontsize=12, fontweight='bold')
axes[1, 0].set_title('Prediction Distribution by True Class', fontsize=14, fontweight='bold')
axes[1, 0].legend()
axes[1, 0].grid(True, alpha=0.3)

# Plot 4: Calibration Analysis
axes[1, 1].scatter(y_pred_probs_flat, y_true, alpha=0.3, s=10, color='#3498db')
axes[1, 1].plot([0, 1], [0, 1], 'r--', lw=2, label='Perfect Calibration')
axes[1, 1].set_xlabel('Predicted Probability', fontsize=12, fontweight='bold')
axes[1, 1].set_ylabel('True Label', fontsize=12, fontweight='bold')
axes[1, 1].set_title('Model Calibration', fontsize=14, fontweight='bold')
axes[1, 1].legend()
axes[1, 1].grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{PLOTS_DIR}/comprehensive_metrics.png', dpi=300, bbox_inches='tight')
plt.close()
print(f"✅ Saved: {PLOTS_DIR}/comprehensive_metrics.png")

# ==========================================
# 10. SAVE RESULTS
# ==========================================

print("\n💾 Saving evaluation results...")

# Save detailed metrics
evaluation_results = {
    "evaluation_info": {
        "date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model_path": MODEL_PATH,
        "test_samples": int(test_generator.samples),
        "class_distribution": {
            "NORMAL": int(np.sum(y_true == 0)),
            "PNEUMONIA": int(np.sum(y_true == 1))
        }
    },
    "primary_metrics": {
        "accuracy": float(accuracy),
        "balanced_accuracy": float(balanced_accuracy),
        "auc_roc": float(roc_auc),
        "average_precision": float(average_precision),
        "matthews_correlation_coefficient": float(mcc)
    },
    "positive_class_metrics": {
        "sensitivity_recall_tpr": float(sensitivity),
        "precision_ppv": float(precision),
        "f1_score": float(f1_score)
    },
    "negative_class_metrics": {
        "specificity_tnr": float(specificity),
        "negative_predictive_value": float(npv)
    },
    "error_rates": {
        "false_positive_rate": float(fpr_val),
        "false_negative_rate": float(fnr),
        "false_discovery_rate": float(fdr)
    },
    "optimal_thresholds": {
        "roc_threshold": float(optimal_threshold),
        "pr_threshold": float(optimal_pr_threshold)
    },
    "confusion_matrix": {
        "true_negatives": int(tn),
        "false_positives": int(fp),
        "false_negatives": int(fn),
        "true_positives": int(tp),
        "matrix": cm.tolist()
    }
}

results_json_path = f'{RESULTS_DIR}/evaluation_results.json'
with open(results_json_path, 'w') as f:
    json.dump(evaluation_results, f, indent=2)
print(f"✅ Saved: {results_json_path}")

# Save metrics as CSV
metrics_df = pd.DataFrame({
    'Metric': list(metrics_dict.keys()),
    'Value': list(metrics_dict.values())
})
metrics_csv_path = f'{RESULTS_DIR}/performance_metrics.csv'
metrics_df.to_csv(metrics_csv_path, index=False)
print(f"✅ Saved: {metrics_csv_path}")

# ==========================================
# 11. FINAL SUMMARY
# ==========================================

print("\n" + "="*80)
print("✅ EVALUATION COMPLETED SUCCESSFULLY!")
print("="*80)
print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"\n📊 KEY RESULTS:")
print(f"   {'='*60}")
print(f"   Accuracy:          {accuracy*100:6.2f}%")
print(f"   AUC-ROC:           {roc_auc:6.4f}")
print(f"   Sensitivity:       {sensitivity*100:6.2f}%")
print(f"   Specificity:       {specificity*100:6.2f}%")
print(f"   F1-Score:          {f1_score:6.4f}")
print(f"   {'='*60}")

print(f"\n📁 GENERATED FILES:")
print(f"   Results:")
print(f"      ├── {results_json_path}")
print(f"      ├── {metrics_csv_path}")
print(f"      └── {report_path}")
print(f"   Visualizations:")
print(f"      ├── {PLOTS_DIR}/confusion_matrix_comprehensive.png")
print(f"      ├── {PLOTS_DIR}/roc_curve.png")
print(f"      ├── {PLOTS_DIR}/precision_recall_curve.png")
print(f"      └── {PLOTS_DIR}/comprehensive_metrics.png")
print("="*80)
print("\n🎉 Evaluation complete! Review the results in the output directories.")
print("="*80)
