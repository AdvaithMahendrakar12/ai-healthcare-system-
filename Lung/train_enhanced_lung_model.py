"""
Enhanced Lung/Pneumonia Detection Model Training Pipeline
=========================================================
This script implements a state-of-the-art deep learning model for pneumonia detection
from chest X-ray images using transfer learning and advanced optimization techniques.

Author: Medical AI Team
Date: January 2026
"""

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3, EfficientNetB4
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint, TensorBoard
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from datetime import datetime
import cv2
from tqdm import tqdm
import json

# Set style for plots
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# ==========================================
# CONFIGURATION
# ==========================================

class Config:
    """Training configuration parameters"""
    # Data paths
    DATA_DIR = 'dataset/chest_xray'
    MODEL_DIR = 'models'
    PLOTS_DIR = 'Pneumonia_plots'
    REPORTS_DIR = 'reports'
    LOGS_DIR = 'logs'
    
    # Model parameters
    IMG_SIZE = (320, 320)  # Optimal for EfficientNet
    BATCH_SIZE = 16
    EPOCHS_INITIAL = 20
    EPOCHS_FINETUNE = 25
    
    # Architecture
    BASE_MODEL = 'EfficientNetB3'  # Options: EfficientNetB3, EfficientNetB4
    
    # Learning rates
    INITIAL_LR = 1e-3
    FINETUNE_LR = 1e-5
    
    # Regularization
    DROPOUT_RATE = 0.5
    L2_REG = 0.01
    
    # Model name
    MODEL_NAME = "Pneumonia_Detection_Enhanced"
    TIMESTAMP = datetime.now().strftime('%Y%m%d_%H%M%S')

# Create directories
for directory in [Config.MODEL_DIR, Config.PLOTS_DIR, Config.REPORTS_DIR, Config.LOGS_DIR]:
    os.makedirs(directory, exist_ok=True)

print("\n" + "="*80)
print("🫁 ENHANCED PNEUMONIA DETECTION MODEL TRAINING PIPELINE")
print("="*80)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📂 Data Directory: {Config.DATA_DIR}")
print(f"🖼️  Image Size: {Config.IMG_SIZE}")
print(f"📦 Batch Size: {Config.BATCH_SIZE}")
print(f"🏗️  Base Model: {Config.BASE_MODEL}")
print("="*80)

# ==========================================
# 1. DATASET ANALYSIS & VISUALIZATION
# ==========================================

def analyze_dataset(data_dir):
    """
    Comprehensive dataset analysis
    - Count images per class and split
    - Analyze class distribution
    - Create visualizations
    """
    print("\n" + "="*70)
    print("📊 DATASET ANALYSIS")
    print("="*70)
    
    analysis_data = []
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            print(f"⚠️  Warning: {split_path} not found")
            continue
            
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                files = [f for f in os.listdir(class_path) 
                        if f.lower().endswith(('.png', '.jpg', '.jpeg'))]
                count = len(files)
                analysis_data.append({
                    'Split': split,
                    'Class': class_name,
                    'Count': count
                })
                print(f"   {split:5s} | {class_name:15s} | {count:5d} images")
    
    df_analysis = pd.DataFrame(analysis_data)
    
    # Save analysis
    df_analysis.to_csv(f'{Config.REPORTS_DIR}/dataset_analysis.csv', index=False)
    
    # Calculate statistics
    print("\n📈 Dataset Statistics:")
    pivot_table = df_analysis.pivot(index='Class', columns='Split', values='Count')
    print(pivot_table)
    
    total_by_class = df_analysis.groupby('Class')['Count'].sum()
    print("\n📊 Total Images per Class:")
    for cls, count in total_by_class.items():
        percentage = (count / total_by_class.sum()) * 100
        print(f"   {cls:15s}: {count:5,} ({percentage:.1f}%)")
    
    # Calculate class imbalance ratio
    imbalance_ratio = total_by_class.max() / total_by_class.min()
    print(f"\n⚖️  Class Imbalance Ratio: {imbalance_ratio:.2f}:1")
    
    # Visualization
    fig, axes = plt.subplots(1, 3, figsize=(18, 5))
    
    # Plot 1: Stacked bar chart by split
    df_pivot = df_analysis.pivot(index='Split', columns='Class', values='Count')
    df_pivot.plot(kind='bar', ax=axes[0], color=['#3498db', '#e74c3c'], width=0.7)
    axes[0].set_title('Dataset Distribution by Split', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Split', fontsize=12)
    axes[0].set_ylabel('Number of Images', fontsize=12)
    axes[0].legend(title='Class', fontsize=10)
    axes[0].grid(True, alpha=0.3)
    axes[0].set_xticklabels(axes[0].get_xticklabels(), rotation=0)
    
    # Plot 2: Pie chart
    colors = ['#3498db', '#e74c3c']
    total_by_class.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                        colors=colors, startangle=90, textprops={'fontsize': 11})
    axes[1].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    # Plot 3: Grouped bar chart
    df_analysis_pivot = df_analysis.pivot(index='Split', columns='Class', values='Count')
    x = np.arange(len(df_analysis_pivot.index))
    width = 0.35
    
    axes[2].bar(x - width/2, df_analysis_pivot.iloc[:, 0], width, 
                label=df_analysis_pivot.columns[0], color='#3498db', alpha=0.8)
    axes[2].bar(x + width/2, df_analysis_pivot.iloc[:, 1], width, 
                label=df_analysis_pivot.columns[1], color='#e74c3c', alpha=0.8)
    
    axes[2].set_title('Comparison by Split', fontsize=14, fontweight='bold')
    axes[2].set_xlabel('Split', fontsize=12)
    axes[2].set_ylabel('Count', fontsize=12)
    axes[2].set_xticks(x)
    axes[2].set_xticklabels(df_analysis_pivot.index)
    axes[2].legend()
    axes[2].grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig(f'{Config.PLOTS_DIR}/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"\n✅ Saved: {Config.PLOTS_DIR}/dataset_distribution.png")
    
    return df_analysis

def analyze_image_characteristics(data_dir, sample_size=200):
    """
    Analyze image characteristics (brightness, contrast, dimensions)
    """
    print("\n" + "="*70)
    print("🔍 IMAGE CHARACTERISTICS ANALYSIS")
    print("="*70)
    
    characteristics = []
    
    for split in ['train']:  # Analyze training set
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(data_dir, split, class_name)
            if not os.path.exists(class_path):
                continue
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:sample_size]
            
            print(f"\n   Analyzing {len(images)} {class_name} images...")
            
            for img_file in tqdm(images, desc=f"   {class_name}"):
                img_path = os.path.join(class_path, img_file)
                try:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    
                    if img is not None:
                        characteristics.append({
                            'Class': class_name,
                            'Mean_Brightness': np.mean(img),
                            'Std_Brightness': np.std(img),
                            'Min_Intensity': np.min(img),
                            'Max_Intensity': np.max(img),
                            'Median_Intensity': np.median(img),
                            'Width': img.shape[1],
                            'Height': img.shape[0]
                        })
                except Exception as e:
                    print(f"   ⚠️  Error reading {img_file}: {e}")
    
    df_chars = pd.DataFrame(characteristics)
    
    # Save characteristics
    df_chars.to_csv(f'{Config.REPORTS_DIR}/image_characteristics.csv', index=False)
    
    # Print statistics
    print("\n📊 Image Characteristics Statistics:")
    for class_name in ['NORMAL', 'PNEUMONIA']:
        class_data = df_chars[df_chars['Class'] == class_name]
        print(f"\n   {class_name}:")
        print(f"      Mean Brightness: {class_data['Mean_Brightness'].mean():.2f} ± {class_data['Mean_Brightness'].std():.2f}")
        print(f"      Mean Dimensions: {class_data['Width'].mean():.0f} x {class_data['Height'].mean():.0f}")
    
    # Visualization
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    
    metrics = ['Mean_Brightness', 'Std_Brightness', 'Min_Intensity', 
               'Max_Intensity', 'Median_Intensity']
    titles = ['Mean Brightness Distribution', 'Standard Deviation', 
              'Minimum Intensity', 'Maximum Intensity', 'Median Intensity']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        row, col = idx // 3, idx % 3
        ax = axes[row, col]
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            data = df_chars[df_chars['Class'] == class_name][metric]
            color = '#3498db' if class_name == 'NORMAL' else '#e74c3c'
            ax.hist(data, bins=30, alpha=0.6, label=class_name, color=color, edgecolor='black')
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(metric.replace('_', ' '), fontsize=10)
        ax.set_ylabel('Frequency', fontsize=10)
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    # Dimensions plot
    ax = axes[1, 2]
    for class_name in ['NORMAL', 'PNEUMONIA']:
        data = df_chars[df_chars['Class'] == class_name]
        color = '#3498db' if class_name == 'NORMAL' else '#e74c3c'
        ax.scatter(data['Width'], data['Height'], alpha=0.5, label=class_name, 
                  color=color, s=20)
    
    ax.set_title('Image Dimensions', fontsize=12, fontweight='bold')
    ax.set_xlabel('Width (pixels)', fontsize=10)
    ax.set_ylabel('Height (pixels)', fontsize=10)
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'{Config.PLOTS_DIR}/image_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {Config.PLOTS_DIR}/image_characteristics.png")
    
    return df_chars

def plot_sample_images(data_dir, samples_per_class=6):
    """Display sample images from each class"""
    print("\n" + "="*70)
    print("🖼️  SAMPLE IMAGES")
    print("="*70)
    
    fig, axes = plt.subplots(2, samples_per_class, figsize=(20, 7))
    
    for row, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
        class_path = os.path.join(data_dir, 'train', class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:samples_per_class]
        
        for col, img_file in enumerate(images):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_ylabel(class_name, fontsize=14, fontweight='bold')
    
    plt.suptitle('Sample Chest X-Ray Images', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    plt.savefig(f'{Config.PLOTS_DIR}/sample_images.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {Config.PLOTS_DIR}/sample_images.png")

# ==========================================
# 2. DATA PREPARATION
# ==========================================

def create_data_generators():
    """
    Create advanced data generators with augmentation
    
    Augmentation Strategy:
    - Training: Aggressive augmentation to improve generalization
    - Validation/Test: Only rescaling for consistent evaluation
    """
    print("\n" + "="*70)
    print("🔄 CREATING DATA GENERATORS")
    print("="*70)
    
    # Training augmentation - Medical imaging specific
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=15,           # X-rays can be slightly rotated
        width_shift_range=0.15,      # Slight horizontal shifts
        height_shift_range=0.15,     # Slight vertical shifts
        zoom_range=0.15,             # Zoom in/out
        horizontal_flip=True,        # X-rays can be flipped
        brightness_range=[0.85, 1.15],  # Brightness variation
        fill_mode='nearest',
        shear_range=5                # Minimal shear
    )
    
    # Validation and Test - only rescaling
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("\n📊 Loading datasets...")
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(Config.DATA_DIR, 'train'),
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        shuffle=True,
        seed=42
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(Config.DATA_DIR, 'val'),
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(Config.DATA_DIR, 'test'),
        target_size=Config.IMG_SIZE,
        batch_size=Config.BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"\n✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Test samples: {test_generator.samples}")
    print(f"✅ Class indices: {train_generator.class_indices}")
    
    # Calculate class weights for imbalanced dataset
    class_counts = np.bincount(train_generator.classes)
    total_samples = len(train_generator.classes)
    class_weights = {i: total_samples / (len(class_counts) * count) 
                    for i, count in enumerate(class_counts)}
    
    print(f"\n⚖️  Class Weights (for imbalanced data):")
    for cls, weight in class_weights.items():
        class_name = list(train_generator.class_indices.keys())[cls]
        print(f"   {class_name}: {weight:.3f}")
    
    return train_generator, val_generator, test_generator, class_weights

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================

def build_model():
    """
    Build enhanced model using EfficientNetB3 with custom head
    
    Architecture:
    - Base: EfficientNetB3 (pretrained on ImageNet)
    - Custom Head: GAP -> BatchNorm -> Dense(256) -> Dropout -> Dense(128) -> Output
    - Regularization: L2 regularization, Dropout, BatchNormalization
    """
    print("\n" + "="*70)
    print("🏗️  BUILDING MODEL ARCHITECTURE")
    print("="*70)
    
    # Load base model
    if Config.BASE_MODEL == 'EfficientNetB3':
        base_model = EfficientNetB3(
            weights='imagenet',
            include_top=False,
            input_shape=(*Config.IMG_SIZE, 3)
        )
    elif Config.BASE_MODEL == 'EfficientNetB4':
        base_model = EfficientNetB4(
            weights='imagenet',
            include_top=False,
            input_shape=(*Config.IMG_SIZE, 3)
        )
    else:
        raise ValueError(f"Unsupported base model: {Config.BASE_MODEL}")
    
    base_model.trainable = False  # Freeze for initial training
    
    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D(name='global_avg_pool')(x)
    x = BatchNormalization(name='bn_1')(x)
    x = Dense(256, activation='relu', 
              kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REG),
              name='dense_1')(x)
    x = Dropout(Config.DROPOUT_RATE, name='dropout_1')(x)
    x = BatchNormalization(name='bn_2')(x)
    x = Dense(128, activation='relu',
              kernel_regularizer=tf.keras.regularizers.l2(Config.L2_REG),
              name='dense_2')(x)
    x = Dropout(0.3, name='dropout_2')(x)
    predictions = Dense(1, activation='sigmoid', name='output')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions, name='Pneumonia_Detector')
    
    print(f"\n✅ Model Architecture:")
    print(f"   Base Model: {Config.BASE_MODEL}")
    print(f"   Input Shape: {Config.IMG_SIZE + (3,)}")
    print(f"   Total Layers: {len(model.layers)}")
    print(f"   Total Parameters: {model.count_params():,}")
    print(f"   Trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.trainable_weights]):,}")
    print(f"   Non-trainable Parameters: {sum([tf.keras.backend.count_params(w) for w in model.non_trainable_weights]):,}")
    
    return model, base_model

# ==========================================
# 4. TRAINING CALLBACKS
# ==========================================

def get_callbacks(phase='initial'):
    """Create training callbacks"""
    callbacks = [
        EarlyStopping(
            monitor='val_accuracy',
            patience=10 if phase == 'finetune' else 7,
            restore_best_weights=True,
            verbose=1,
            mode='max'
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=4 if phase == 'finetune' else 3,
            min_lr=1e-8,
            verbose=1,
            mode='min'
        ),
        ModelCheckpoint(
            f'{Config.MODEL_DIR}/lung_model_best_{phase}_{Config.TIMESTAMP}.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1,
            mode='max'
        ),
        TensorBoard(
            log_dir=f'{Config.LOGS_DIR}/{phase}_{Config.TIMESTAMP}',
            histogram_freq=1
        )
    ]
    
    return callbacks

# ==========================================
# 5. TRAINING FUNCTIONS
# ==========================================

def plot_training_history(history, phase_name, save_prefix=''):
    """Plot comprehensive training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(16, 12))
    
    epochs = range(1, len(history.history['accuracy']) + 1)
    
    # Accuracy plot
    axes[0, 0].plot(epochs, history.history['accuracy'], 
                    label='Training Accuracy', linewidth=2.5, color='#3498db', marker='o', markersize=4)
    axes[0, 0].plot(epochs, history.history['val_accuracy'], 
                    label='Validation Accuracy', linewidth=2.5, color='#e74c3c', marker='s', markersize=4)
    axes[0, 0].set_title(f'{phase_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch', fontsize=12)
    axes[0, 0].set_ylabel('Accuracy', fontsize=12)
    axes[0, 0].legend(fontsize=10)
    axes[0, 0].grid(True, alpha=0.3)
    axes[0, 0].set_ylim([0.5, 1.0])
    
    # Loss plot
    axes[0, 1].plot(epochs, history.history['loss'], 
                    label='Training Loss', linewidth=2.5, color='#3498db', marker='o', markersize=4)
    axes[0, 1].plot(epochs, history.history['val_loss'], 
                    label='Validation Loss', linewidth=2.5, color='#e74c3c', marker='s', markersize=4)
    axes[0, 1].set_title(f'{phase_name} - Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch', fontsize=12)
    axes[0, 1].set_ylabel('Loss', fontsize=12)
    axes[0, 1].legend(fontsize=10)
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate plot
    if 'lr' in history.history:
        axes[1, 0].plot(epochs, history.history['lr'], 
                       linewidth=2.5, color='#2ecc71', marker='d', markersize=4)
        axes[1, 0].set_title(f'{phase_name} - Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch', fontsize=12)
        axes[1, 0].set_ylabel('Learning Rate', fontsize=12)
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    else:
        axes[1, 0].axis('off')
    
    # Training summary
    final_train_acc = history.history['accuracy'][-1]
    final_val_acc = history.history['val_accuracy'][-1]
    best_val_acc = max(history.history['val_accuracy'])
    best_epoch = history.history['val_accuracy'].index(best_val_acc) + 1
    final_loss = history.history['val_loss'][-1]
    
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary - {phase_name}
    {'='*40}
    
    Final Training Accuracy:    {final_train_acc*100:.2f}%
    Final Validation Accuracy:  {final_val_acc*100:.2f}%
    Best Validation Accuracy:   {best_val_acc*100:.2f}%
    Best Epoch:                 {best_epoch}
    Final Validation Loss:      {final_loss:.4f}
    Total Epochs Trained:       {len(history.history['accuracy'])}
    
    Overfitting Check:
    Gap (Train-Val):            {(final_train_acc - final_val_acc)*100:.2f}%
    """
    
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=11, verticalalignment='center',
                    family='monospace',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    filename = f'{Config.PLOTS_DIR}/training_history_{save_prefix}_{phase_name.lower().replace(" ", "_")}.png'
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {filename}")

# ==========================================
# 6. EVALUATION FUNCTIONS
# ==========================================

def evaluate_model(model, test_generator):
    """
    Comprehensive model evaluation with multiple metrics
    """
    print("\n" + "="*70)
    print("📊 MODEL EVALUATION ON TEST SET")
    print("="*70)
    
    # Get predictions
    print("\n🔄 Generating predictions...")
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Classification Report
    print("\n" + "="*70)
    print("📋 CLASSIFICATION REPORT")
    print("="*70)
    class_names = list(test_generator.class_indices.keys())
    report = classification_report(y_true, y_pred_classes, 
                                   target_names=class_names,
                                   digits=4)
    print(report)
    
    # Save report
    with open(f'{Config.REPORTS_DIR}/classification_report.txt', 'w') as f:
        f.write("="*70 + "\n")
        f.write("CLASSIFICATION REPORT - PNEUMONIA DETECTION\n")
        f.write("="*70 + "\n")
        f.write(f"Model: {Config.BASE_MODEL}\n")
        f.write(f"Date: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("="*70 + "\n\n")
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Plot 1: Counts
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Count'}, ax=axes[0], annot_kws={"size": 14})
    axes[0].set_title('Confusion Matrix (Counts)', fontsize=14, fontweight='bold')
    axes[0].set_ylabel('True Label', fontsize=12)
    axes[0].set_xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            axes[0].text(j + 0.5, i + 0.7, f'({cm_percent[i, j]*100:.1f}%)',
                        ha='center', va='center', fontsize=10, color='red', fontweight='bold')
    
    # Plot 2: Normalized
    sns.heatmap(cm_percent, annot=True, fmt='.2%', cmap='RdYlGn', 
                xticklabels=class_names, yticklabels=class_names,
                cbar_kws={'label': 'Percentage'}, ax=axes[1], annot_kws={"size": 14})
    axes[1].set_title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('True Label', fontsize=12)
    axes[1].set_xlabel('Predicted Label', fontsize=12)
    
    plt.tight_layout()
    plt.savefig(f'{Config.PLOTS_DIR}/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {Config.PLOTS_DIR}/confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold (Youden's J statistic)
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='#e74c3c', lw=3, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='#95a5a6', lw=2, linestyle='--', 
             label='Random Classifier (AUC = 0.5)')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], s=250, c='#2ecc71', 
                marker='*', edgecolors='black', linewidth=2,
                label=f'Optimal Threshold = {optimal_threshold:.3f}', zorder=5)
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=13, fontweight='bold')
    plt.ylabel('True Positive Rate (Sensitivity)', fontsize=13, fontweight='bold')
    plt.title('ROC Curve - Pneumonia Detection', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{Config.PLOTS_DIR}/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {Config.PLOTS_DIR}/roc_curve.png")
    
    # Calculate detailed metrics
    tn, fp, fn, tp = cm.ravel()
    
    accuracy = (tp + tn) / (tp + tn + fp + fn)
    sensitivity = tp / (tp + fn)  # Recall, True Positive Rate
    specificity = tn / (tn + fp)  # True Negative Rate
    precision = tp / (tp + fp)
    f1_score = 2 * (precision * sensitivity) / (precision + sensitivity)
    npv = tn / (tn + fn)  # Negative Predictive Value
    fpr_val = fp / (fp + tn)  # False Positive Rate
    fnr = fn / (fn + tp)  # False Negative Rate
    
    print("\n" + "="*70)
    print("🎯 DETAILED PERFORMANCE METRICS")
    print("="*70)
    print(f"Overall Accuracy:           {accuracy*100:.2f}%")
    print(f"AUC-ROC Score:              {roc_auc:.4f}")
    print(f"\nClass-specific Metrics:")
    print(f"  Sensitivity (Recall/TPR): {sensitivity*100:.2f}%  [Ability to detect Pneumonia]")
    print(f"  Specificity (TNR):        {specificity*100:.2f}%  [Ability to detect Normal]")
    print(f"  Precision (PPV):          {precision*100:.2f}%  [Positive predictive value]")
    print(f"  F1-Score:                 {f1_score:.4f}")
    print(f"  NPV:                      {npv*100:.2f}%  [Negative predictive value]")
    print(f"\nError Rates:")
    print(f"  False Positive Rate:      {fpr_val*100:.2f}%")
    print(f"  False Negative Rate:      {fnr*100:.2f}%")
    print(f"\nOptimal Threshold:          {optimal_threshold:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  True Negatives:  {tn:4d}  |  False Positives: {fp:4d}")
    print(f"  False Negatives: {fn:4d}  |  True Positives:  {tp:4d}")
    print("="*70)
    
    # Create metrics visualization
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    
    # Bar plot of metrics
    metrics_dict = {
        'Accuracy': accuracy,
        'Sensitivity': sensitivity,
        'Specificity': specificity,
        'Precision': precision,
        'F1-Score': f1_score,
        'AUC-ROC': roc_auc
    }
    
    colors = ['#3498db', '#e74c3c', '#2ecc71', '#f39c12', '#9b59b6', '#1abc9c']
    bars = axes[0].barh(list(metrics_dict.keys()), list(metrics_dict.values()), 
                        color=colors, edgecolor='black', linewidth=1.5)
    axes[0].set_xlim([0, 1])
    axes[0].set_xlabel('Score', fontsize=12, fontweight='bold')
    axes[0].set_title('Model Performance Metrics', fontsize=14, fontweight='bold')
    axes[0].grid(True, alpha=0.3, axis='x')
    
    # Add value labels
    for i, (bar, value) in enumerate(zip(bars, metrics_dict.values())):
        axes[0].text(value + 0.02, i, f'{value:.3f}', 
                    va='center', fontsize=10, fontweight='bold')
    
    # Confusion matrix breakdown
    confusion_data = {
        'True\nNegatives': tn,
        'False\nPositives': fp,
        'False\nNegatives': fn,
        'True\nPositives': tp
    }
    
    colors_cm = ['#2ecc71', '#e74c3c', '#e74c3c', '#2ecc71']
    bars2 = axes[1].bar(list(confusion_data.keys()), list(confusion_data.values()),
                       color=colors_cm, edgecolor='black', linewidth=1.5, alpha=0.7)
    axes[1].set_ylabel('Count', fontsize=12, fontweight='bold')
    axes[1].set_title('Confusion Matrix Breakdown', fontsize=14, fontweight='bold')
    axes[1].grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bar, value in zip(bars2, confusion_data.values()):
        height = bar.get_height()
        axes[1].text(bar.get_x() + bar.get_width()/2., height + 5,
                    f'{int(value)}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(f'{Config.PLOTS_DIR}/performance_metrics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: {Config.PLOTS_DIR}/performance_metrics.png")
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'precision': precision,
        'f1_score': f1_score,
        'npv': npv,
        'fpr': fpr_val,
        'fnr': fnr,
        'optimal_threshold': optimal_threshold,
        'confusion_matrix': cm.tolist()
    }

# ==========================================
# 7. MAIN TRAINING PIPELINE
# ==========================================

def main():
    """Main training pipeline"""
    
    print("\n" + "="*80)
    print("STEP 1: DATASET ANALYSIS")
    print("="*80)
    
    # 1. Analyze dataset
    df_analysis = analyze_dataset(Config.DATA_DIR)
    df_chars = analyze_image_characteristics(Config.DATA_DIR, sample_size=200)
    plot_sample_images(Config.DATA_DIR)
    
    print("\n" + "="*80)
    print("STEP 2: DATA PREPARATION")
    print("="*80)
    
    # 2. Create data generators
    train_gen, val_gen, test_gen, class_weights = create_data_generators()
    
    print("\n" + "="*80)
    print("STEP 3: MODEL BUILDING")
    print("="*80)
    
    # 3. Build model
    model, base_model = build_model()
    
    # Print model summary
    print("\n📋 Model Summary:")
    model.summary()
    
    print("\n" + "="*80)
    print("STEP 4: INITIAL TRAINING (Transfer Learning)")
    print("="*80)
    
    # 4. Initial training with frozen base
    model.compile(
        optimizer=Adam(learning_rate=Config.INITIAL_LR),
        loss='binary_crossentropy',
        metrics=['accuracy', 
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    callbacks_initial = get_callbacks('initial')
    
    print(f"\n🏋️  Training with frozen base model...")
    print(f"   Learning Rate: {Config.INITIAL_LR}")
    print(f"   Epochs: {Config.EPOCHS_INITIAL}")
    
    history_initial = model.fit(
        train_gen,
        epochs=Config.EPOCHS_INITIAL,
        validation_data=val_gen,
        callbacks=callbacks_initial,
        class_weight=class_weights,  # Handle class imbalance
        verbose=1
    )
    
    plot_training_history(history_initial, "Initial Training", "phase1")
    
    print("\n" + "="*80)
    print("STEP 5: FINE-TUNING")
    print("="*80)
    
    # 5. Fine-tuning - Unfreeze and train top layers
    base_model.trainable = True
    
    # Freeze early layers, train only last 60 layers
    freeze_until = len(base_model.layers) - 60
    for layer in base_model.layers[:freeze_until]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in model.layers if layer.trainable])
    print(f"\n🔧 Unfreezing top layers of base model")
    print(f"   Total layers: {len(model.layers)}")
    print(f"   Trainable layers: {trainable_count}")
    print(f"   Frozen layers: {len(model.layers) - trainable_count}")
    
    # Recompile with lower learning rate
    model.compile(
        optimizer=Adam(learning_rate=Config.FINETUNE_LR),
        loss='binary_crossentropy',
        metrics=['accuracy',
                 tf.keras.metrics.Precision(name='precision'),
                 tf.keras.metrics.Recall(name='recall'),
                 tf.keras.metrics.AUC(name='auc')]
    )
    
    callbacks_finetune = get_callbacks('finetune')
    
    print(f"\n🏋️  Fine-tuning with unfrozen layers...")
    print(f"   Learning Rate: {Config.FINETUNE_LR}")
    print(f"   Epochs: {Config.EPOCHS_FINETUNE}")
    
    history_finetune = model.fit(
        train_gen,
        epochs=Config.EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=callbacks_finetune,
        class_weight=class_weights,
        verbose=1
    )
    
    plot_training_history(history_finetune, "Fine-Tuning", "phase2")
    
    print("\n" + "="*80)
    print("STEP 6: FINAL EVALUATION")
    print("="*80)
    
    # 6. Evaluate on test set
    metrics = evaluate_model(model, test_gen)
    
    print("\n" + "="*80)
    print("STEP 7: SAVING MODEL & METADATA")
    print("="*80)
    
    # 7. Save final model
    final_model_path = f'{Config.MODEL_DIR}/lung_model_final_{Config.TIMESTAMP}.h5'
    model.save(final_model_path)
    print(f"✅ Saved final model: {final_model_path}")
    
    # Also save as lung_model.h5 (for compatibility)
    model.save(f'{Config.MODEL_DIR}/lung_model.h5')
    print(f"✅ Saved model: {Config.MODEL_DIR}/lung_model.h5")
    
    # Save model metadata
    metadata = {
        "model_info": {
            "name": Config.MODEL_NAME,
            "version": "1.0",
            "architecture": Config.BASE_MODEL,
            "task": "Binary Classification - Pneumonia Detection",
            "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "input_shape": list(Config.IMG_SIZE) + [3],
            "output_classes": ["NORMAL", "PNEUMONIA"]
        },
        "training_config": {
            "image_size": list(Config.IMG_SIZE),
            "batch_size": Config.BATCH_SIZE,
            "initial_epochs": Config.EPOCHS_INITIAL,
            "finetune_epochs": Config.EPOCHS_FINETUNE,
            "initial_learning_rate": Config.INITIAL_LR,
            "finetune_learning_rate": Config.FINETUNE_LR,
            "optimizer": "Adam",
            "loss_function": "binary_crossentropy",
            "dropout_rate": Config.DROPOUT_RATE,
            "l2_regularization": Config.L2_REG
        },
        "dataset_info": {
            "training_samples": train_gen.samples,
            "validation_samples": val_gen.samples,
            "test_samples": test_gen.samples,
            "class_distribution": {
                k: int(v) for k, v in zip(train_gen.class_indices.keys(), 
                                         np.bincount(train_gen.classes))
            }
        },
        "performance_metrics": {
            "accuracy": float(metrics['accuracy']),
            "auc_roc": float(metrics['auc']),
            "sensitivity_recall": float(metrics['sensitivity']),
            "specificity": float(metrics['specificity']),
            "precision": float(metrics['precision']),
            "f1_score": float(metrics['f1_score']),
            "npv": float(metrics['npv']),
            "false_positive_rate": float(metrics['fpr']),
            "false_negative_rate": float(metrics['fnr']),
            "optimal_threshold": float(metrics['optimal_threshold']),
            "confusion_matrix": metrics['confusion_matrix']
        },
        "augmentation": {
            "rotation_range": 15,
            "width_shift_range": 0.15,
            "height_shift_range": 0.15,
            "zoom_range": 0.15,
            "horizontal_flip": True,
            "brightness_range": [0.85, 1.15]
        }
    }
    
    metadata_path = f'{Config.MODEL_DIR}/model_metadata.json'
    with open(metadata_path, 'w') as f:
        json.dump(metadata, f, indent=2)
    print(f"✅ Saved metadata: {metadata_path}")
    
    # Final summary
    print("\n" + "="*80)
    print("✅ TRAINING PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*80)
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 FINAL MODEL PERFORMANCE:")
    print(f"   {'='*60}")
    print(f"   Accuracy:          {metrics['accuracy']*100:6.2f}%")
    print(f"   AUC-ROC:           {metrics['auc']:6.4f}")
    print(f"   Sensitivity:       {metrics['sensitivity']*100:6.2f}%")
    print(f"   Specificity:       {metrics['specificity']*100:6.2f}%")
    print(f"   Precision:         {metrics['precision']*100:6.2f}%")
    print(f"   F1-Score:          {metrics['f1_score']:6.4f}")
    print(f"   {'='*60}")
    
    print(f"\n📁 GENERATED FILES:")
    print(f"   Models:")
    print(f"      ├── {Config.MODEL_DIR}/lung_model.h5")
    print(f"      ├── {final_model_path}")
    print(f"      └── {metadata_path}")
    print(f"   Reports:")
    print(f"      ├── {Config.REPORTS_DIR}/classification_report.txt")
    print(f"      ├── {Config.REPORTS_DIR}/dataset_analysis.csv")
    print(f"      └── {Config.REPORTS_DIR}/image_characteristics.csv")
    print(f"   Visualizations:")
    print(f"      ├── {Config.PLOTS_DIR}/dataset_distribution.png")
    print(f"      ├── {Config.PLOTS_DIR}/image_characteristics.png")
    print(f"      ├── {Config.PLOTS_DIR}/sample_images.png")
    print(f"      ├── {Config.PLOTS_DIR}/training_history_phase1_initial_training.png")
    print(f"      ├── {Config.PLOTS_DIR}/training_history_phase2_fine_tuning.png")
    print(f"      ├── {Config.PLOTS_DIR}/confusion_matrix.png")
    print(f"      ├── {Config.PLOTS_DIR}/roc_curve.png")
    print(f"      └── {Config.PLOTS_DIR}/performance_metrics.png")
    print(f"   Logs:")
    print(f"      └── {Config.LOGS_DIR}/")
    print("="*80)
    print("\n🎉 Ready for deployment!")
    print("="*80)

if __name__ == "__main__":
    main()
