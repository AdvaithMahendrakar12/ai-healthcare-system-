import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.applications import EfficientNetB3
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D, Dropout, BatchNormalization
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import os
from sklearn.metrics import confusion_matrix, roc_curve, auc, classification_report
from datetime import datetime
import cv2
from tqdm import tqdm

# Set style
sns.set_style("whitegrid")
plt.rcParams['figure.dpi'] = 300

# --- Configuration ---
IMG_SIZE = (300, 300)  # Increased for better feature extraction
BATCH_SIZE = 16  # Reduced for better gradient estimation
EPOCHS_INITIAL = 15
EPOCHS_FINETUNE = 20
DATA_DIR = '/Users/kalyan/Client project/Explainable AI/python/data/chest_xray'
MODEL_NAME = "Pneumonia_Detection"

# Create directories
os.makedirs('Pneumonia_plots', exist_ok=True)
os.makedirs('Pneumonia_models', exist_ok=True)
os.makedirs('Pneumonia_reports', exist_ok=True)

print("\n" + "="*70)
print("🫁 PNEUMONIA DETECTION MODEL TRAINING")
print("="*70)
print(f"📅 Started: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print(f"📂 Data Directory: {DATA_DIR}")
print(f"🖼️  Image Size: {IMG_SIZE}")
print(f"📦 Batch Size: {BATCH_SIZE}")

# ==========================================
# 1. DATA ANALYSIS & VISUALIZATION
# ==========================================

def analyze_dataset(data_dir):
    """Analyze the dataset structure and create EDA plots"""
    print("\n" + "="*60)
    print("📊 DATASET ANALYSIS")
    print("="*60)
    
    analysis_data = []
    
    for split in ['train', 'val', 'test']:
        split_path = os.path.join(data_dir, split)
        if not os.path.exists(split_path):
            continue
            
        for class_name in os.listdir(split_path):
            class_path = os.path.join(split_path, class_name)
            if os.path.isdir(class_path):
                count = len([f for f in os.listdir(class_path) 
                           if f.lower().endswith(('.png', '.jpg', '.jpeg'))])
                analysis_data.append({
                    'Split': split,
                    'Class': class_name,
                    'Count': count
                })
    
    df_analysis = pd.DataFrame(analysis_data)
    
    # Print statistics
    print("\n📈 Dataset Statistics:")
    print(df_analysis.pivot(index='Class', columns='Split', values='Count'))
    
    total_by_class = df_analysis.groupby('Class')['Count'].sum()
    print("\n📊 Total Images per Class:")
    for cls, count in total_by_class.items():
        print(f"   {cls}: {count:,}")
    
    # Plot 1: Class Distribution
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    # Bar plot by split
    df_pivot = df_analysis.pivot(index='Split', columns='Class', values='Count')
    df_pivot.plot(kind='bar', ax=axes[0], color=['#2ecc71', '#e74c3c'])
    axes[0].set_title('Images per Split and Class', fontsize=14, fontweight='bold')
    axes[0].set_xlabel('Split')
    axes[0].set_ylabel('Number of Images')
    axes[0].legend(title='Class')
    axes[0].grid(True, alpha=0.3)
    
    # Pie chart total
    total_by_class.plot(kind='pie', ax=axes[1], autopct='%1.1f%%', 
                        colors=['#2ecc71', '#e74c3c'], startangle=90)
    axes[1].set_title('Overall Class Distribution', fontsize=14, fontweight='bold')
    axes[1].set_ylabel('')
    
    plt.tight_layout()
    plt.savefig('plots/dataset_distribution.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: plots/dataset_distribution.png")
    
    return df_analysis

def analyze_image_characteristics(data_dir, sample_size=100):
    """Analyze image characteristics like brightness, contrast"""
    print("\n" + "="*60)
    print("🔍 IMAGE CHARACTERISTICS ANALYSIS")
    print("="*60)
    
    characteristics = []
    
    for split in ['train']:
        for class_name in ['NORMAL', 'PNEUMONIA']:
            class_path = os.path.join(data_dir, split, class_name)
            if not os.path.exists(class_path):
                continue
            
            images = [f for f in os.listdir(class_path) 
                     if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:sample_size]
            
            print(f"   Analyzing {len(images)} {class_name} images...")
            
            for img_file in tqdm(images, desc=f"{class_name}"):
                img_path = os.path.join(class_path, img_file)
                img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                
                if img is not None:
                    characteristics.append({
                        'Class': class_name,
                        'Mean_Brightness': np.mean(img),
                        'Std_Brightness': np.std(img),
                        'Min_Intensity': np.min(img),
                        'Max_Intensity': np.max(img)
                    })
    
    df_chars = pd.DataFrame(characteristics)
    
    # Plot characteristics
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    metrics = ['Mean_Brightness', 'Std_Brightness', 'Min_Intensity', 'Max_Intensity']
    titles = ['Mean Brightness Distribution', 'Standard Deviation Distribution',
              'Minimum Intensity Distribution', 'Maximum Intensity Distribution']
    
    for idx, (metric, title) in enumerate(zip(metrics, titles)):
        ax = axes[idx // 2, idx % 2]
        
        for class_name in ['NORMAL', 'PNEUMONIA']:
            data = df_chars[df_chars['Class'] == class_name][metric]
            color = '#2ecc71' if class_name == 'NORMAL' else '#e74c3c'
            ax.hist(data, bins=30, alpha=0.6, label=class_name, color=color)
        
        ax.set_title(title, fontsize=12, fontweight='bold')
        ax.set_xlabel(metric.replace('_', ' '))
        ax.set_ylabel('Frequency')
        ax.legend()
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('Pneumonia_plots/image_characteristics.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Pneumonia_plots/image_characteristics.png")
    
    return df_chars

def plot_sample_images(data_dir):
    """Display sample images from each class"""
    print("\n" + "="*60)
    print("🖼️  SAMPLE IMAGES")
    print("="*60)
    
    fig, axes = plt.subplots(2, 6, figsize=(18, 6))
    
    for row, class_name in enumerate(['NORMAL', 'PNEUMONIA']):
        class_path = os.path.join(data_dir, 'train', class_name)
        images = [f for f in os.listdir(class_path) 
                 if f.lower().endswith(('.png', '.jpg', '.jpeg'))][:6]
        
        for col, img_file in enumerate(images):
            img_path = os.path.join(class_path, img_file)
            img = cv2.imread(img_path)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            
            axes[row, col].imshow(img)
            axes[row, col].axis('off')
            if col == 0:
                axes[row, col].set_title(class_name, fontsize=12, fontweight='bold')
    
    plt.suptitle('Sample X-Ray Images', fontsize=16, fontweight='bold')
    plt.tight_layout()
    plt.savefig('Pneumonia_plots/sample_images.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Pneumonia_plots/sample_images.png")

# ==========================================
# 2. DATA PREPARATION WITH AUGMENTATION
# ==========================================

def create_data_generators():
    """Create enhanced data generators with aggressive augmentation"""
    print("\n" + "="*60)
    print("🔄 CREATING DATA GENERATORS")
    print("="*60)
    
    # Training augmentation - aggressive for X-rays
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='nearest'
    )
    
    # Validation and Test - only rescaling
    val_test_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'train'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=True
    )
    
    val_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'val'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    test_generator = val_test_datagen.flow_from_directory(
        os.path.join(DATA_DIR, 'test'),
        target_size=IMG_SIZE,
        batch_size=BATCH_SIZE,
        class_mode='binary',
        shuffle=False
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Test samples: {test_generator.samples}")
    print(f"✅ Class indices: {train_generator.class_indices}")
    
    return train_generator, val_generator, test_generator

# ==========================================
# 3. MODEL ARCHITECTURE
# ==========================================

def build_model():
    """Build an improved model using EfficientNetB3"""
    print("\n" + "="*60)
    print("🏗️  BUILDING MODEL")
    print("="*60)
    
    # Use EfficientNetB3 - better than MobileNetV2 for medical imaging
    base_model = EfficientNetB3(
        weights='imagenet',
        include_top=False,
        input_shape=(*IMG_SIZE, 3)
    )
    
    base_model.trainable = False  # Freeze initially
    
    # Custom classification head
    x = base_model.output
    x = GlobalAveragePooling2D()(x)
    x = BatchNormalization()(x)
    x = Dense(256, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.5)(x)
    x = BatchNormalization()(x)
    x = Dense(128, activation='relu', kernel_regularizer=tf.keras.regularizers.l2(0.01))(x)
    x = Dropout(0.3)(x)
    predictions = Dense(1, activation='sigmoid')(x)
    
    model = Model(inputs=base_model.input, outputs=predictions)
    
    print(f"✅ Model created with {len(model.layers)} layers")
    print(f"   Base model: EfficientNetB3")
    print(f"   Total parameters: {model.count_params():,}")
    
    return model, base_model

# ==========================================
# 4. TRAINING FUNCTIONS
# ==========================================

def plot_training_history(history, phase_name):
    """Plot training metrics"""
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy
    axes[0, 0].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[0, 0].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[0, 0].set_title(f'{phase_name} - Accuracy', fontsize=14, fontweight='bold')
    axes[0, 0].set_xlabel('Epoch')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].legend()
    axes[0, 0].grid(True, alpha=0.3)
    
    # Loss
    axes[0, 1].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0, 1].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0, 1].set_title(f'{phase_name} - Loss', fontsize=14, fontweight='bold')
    axes[0, 1].set_xlabel('Epoch')
    axes[0, 1].set_ylabel('Loss')
    axes[0, 1].legend()
    axes[0, 1].grid(True, alpha=0.3)
    
    # Learning Rate (if available)
    if 'lr' in history.history:
        axes[1, 0].plot(history.history['lr'], linewidth=2, color='green')
        axes[1, 0].set_title(f'{phase_name} - Learning Rate', fontsize=14, fontweight='bold')
        axes[1, 0].set_xlabel('Epoch')
        axes[1, 0].set_ylabel('Learning Rate')
        axes[1, 0].set_yscale('log')
        axes[1, 0].grid(True, alpha=0.3)
    
    # Metrics summary
    final_acc = history.history['val_accuracy'][-1]
    final_loss = history.history['val_loss'][-1]
    best_acc = max(history.history['val_accuracy'])
    
    axes[1, 1].axis('off')
    summary_text = f"""
    Training Summary
    
    Final Validation Accuracy: {final_acc*100:.2f}%
    Final Validation Loss: {final_loss:.4f}
    Best Validation Accuracy: {best_acc*100:.2f}%
    Total Epochs: {len(history.history['accuracy'])}
    """
    axes[1, 1].text(0.1, 0.5, summary_text, fontsize=12, verticalalignment='center',
                    bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(f'Pneumonia_plots/training_history_{phase_name.lower().replace(" ", "_")}.png', 
                dpi=300, bbox_inches='tight')
    plt.close()
    print(f"✅ Saved: Pneumonia_plots/training_history_{phase_name.lower().replace(' ', '_')}.png")

# ==========================================
# 5. EVALUATION FUNCTIONS
# ==========================================

def evaluate_model(model, test_generator):
    """Comprehensive model evaluation"""
    print("\n" + "="*60)
    print("📊 MODEL EVALUATION")
    print("="*60)
    
    # Get predictions
    print("🔄 Generating predictions...")
    test_generator.reset()
    y_pred_probs = model.predict(test_generator, verbose=1)
    y_pred_classes = (y_pred_probs > 0.5).astype(int).flatten()
    y_true = test_generator.classes
    
    # Classification Report
    print("\n📋 Classification Report:")
    report = classification_report(y_true, y_pred_classes, 
                                   target_names=['NORMAL', 'PNEUMONIA'],
                                   digits=4)
    print(report)
    
    # Save report
    with open('reports/classification_report.txt', 'w') as f:
        f.write(report)
    
    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred_classes)
    
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                xticklabels=['NORMAL', 'PNEUMONIA'],
                yticklabels=['NORMAL', 'PNEUMONIA'],
                cbar_kws={'label': 'Count'})
    plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    
    # Add percentages
    cm_percent = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            plt.text(j + 0.5, i + 0.7, f'({cm_percent[i, j]*100:.1f}%)',
                    ha='center', va='center', fontsize=10, color='red')
    
    plt.tight_layout()
    plt.savefig('Pneumonia_plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Pneumonia_plots/confusion_matrix.png")
    
    # ROC Curve
    fpr, tpr, thresholds = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    
    # Find optimal threshold
    optimal_idx = np.argmax(tpr - fpr)
    optimal_threshold = thresholds[optimal_idx]
    
    plt.figure(figsize=(10, 8))
    plt.plot(fpr, tpr, color='darkorange', lw=3, 
             label=f'ROC Curve (AUC = {roc_auc:.4f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', label='Random Classifier')
    plt.scatter([fpr[optimal_idx]], [tpr[optimal_idx]], s=200, c='red', 
                marker='o', label=f'Optimal Threshold = {optimal_threshold:.3f}')
    
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate', fontsize=12)
    plt.ylabel('True Positive Rate', fontsize=12)
    plt.title('ROC Curve - Pneumonia Detection', fontsize=16, fontweight='bold')
    plt.legend(loc="lower right", fontsize=11)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig('Pneumonia_plots/roc_curve.png', dpi=300, bbox_inches='tight')
    plt.close()
    print("✅ Saved: Pneumonia_plots/roc_curve.png")
    
    # Metrics summary
    accuracy = np.mean(y_pred_classes == y_true)
    sensitivity = cm[1, 1] / (cm[1, 0] + cm[1, 1])  # True Positive Rate
    specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # True Negative Rate
    
    print(f"\n🎯 Final Metrics:")
    print(f"   Accuracy: {accuracy*100:.2f}%")
    print(f"   AUC-ROC: {roc_auc:.4f}")
    print(f"   Sensitivity (Recall): {sensitivity*100:.2f}%")
    print(f"   Specificity: {specificity*100:.2f}%")
    print(f"   Optimal Threshold: {optimal_threshold:.4f}")
    
    return {
        'accuracy': accuracy,
        'auc': roc_auc,
        'sensitivity': sensitivity,
        'specificity': specificity,
        'optimal_threshold': optimal_threshold
    }

# ==========================================
# 6. MAIN TRAINING PIPELINE
# ==========================================

def main():
    # 1. Dataset Analysis
    df_analysis = analyze_dataset(DATA_DIR)
    df_chars = analyze_image_characteristics(DATA_DIR, sample_size=100)
    plot_sample_images(DATA_DIR)
    
    # 2. Create Data Generators
    train_gen, val_gen, test_gen = create_data_generators()
    
    # 3. Build Model
    model, base_model = build_model()
    
    # 4. Initial Training (Transfer Learning)
    print("\n" + "="*60)
    print("🏋️  PHASE 1: INITIAL TRAINING")
    print("="*60)
    
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_initial = [
        EarlyStopping(monitor='val_accuracy', patience=7, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
        ModelCheckpoint('models/pneumonia_best_initial.h5', monitor='val_accuracy', 
                       save_best_only=True, verbose=1)
    ]
    
    history_initial = model.fit(
        train_gen,
        epochs=EPOCHS_INITIAL,
        validation_data=val_gen,
        callbacks=callbacks_initial,
        verbose=1
    )
    
    plot_training_history(history_initial, "Initial Training")
    
    # 5. Fine-Tuning
    print("\n" + "="*60)
    print("🔧 PHASE 2: FINE-TUNING")
    print("="*60)
    
    base_model.trainable = True
    
    # Freeze early layers, train only last 50 layers
    for layer in base_model.layers[:-50]:
        layer.trainable = False
    
    trainable_count = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"   Trainable layers in base model: {trainable_count}")
    
    model.compile(
        optimizer=Adam(learning_rate=1e-5),  # Very low learning rate
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    
    callbacks_finetune = [
        EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=4, min_lr=1e-8, verbose=1),
        ModelCheckpoint('models/pneumonia_best_finetuned.h5', monitor='val_accuracy',
                       save_best_only=True, verbose=1)
    ]
    
    history_finetune = model.fit(
        train_gen,
        epochs=EPOCHS_FINETUNE,
        validation_data=val_gen,
        callbacks=callbacks_finetune,
        verbose=1
    )
    
    plot_training_history(history_finetune, "Fine-Tuning")
    
    # 6. Final Evaluation
    metrics = evaluate_model(model, test_gen)
    
    # 7. Save Final Model
    print("\n" + "="*60)
    print("💾 SAVING FINAL MODEL")
    print("="*60)
    
    model.save('models/pneumonia_model_final.h5')
    print("✅ Saved: models/pneumonia_model_final.h5")
    
    # Save metadata
    metadata = {
        "model_name": MODEL_NAME,
        "training_date": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "architecture": "EfficientNetB3",
        "input_shape": list(IMG_SIZE) + [3],
        "metrics": {
            "accuracy": float(metrics['accuracy']),
            "auc": float(metrics['auc']),
            "sensitivity": float(metrics['sensitivity']),
            "specificity": float(metrics['specificity']),
            "optimal_threshold": float(metrics['optimal_threshold'])
        },
        "training_config": {
            "batch_size": BATCH_SIZE,
            "initial_epochs": EPOCHS_INITIAL,
            "finetune_epochs": EPOCHS_FINETUNE,
            "optimizer": "Adam",
            "loss": "binary_crossentropy"
        }
    }
    
    import json
    with open('models/model_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    print("✅ Saved: models/model_metadata.json")
    
    # Final Summary
    print("\n" + "="*70)
    print("✅ TRAINING COMPLETE!")
    print("="*70)
    print(f"📅 Completed: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"\n📊 Final Performance:")
    print(f"   Accuracy: {metrics['accuracy']*100:.2f}%")
    print(f"   AUC-ROC: {metrics['auc']:.4f}")
    print(f"   Sensitivity: {metrics['sensitivity']*100:.2f}%")
    print(f"   Specificity: {metrics['specificity']*100:.2f}%")
    print(f"\n📁 Generated Files:")
    print(f"   ├── models/pneumonia_model_final.h5")
    print(f"   ├── models/pneumonia_best_initial.h5")
    print(f"   ├── models/pneumonia_best_finetuned.h5")
    print(f"   ├── models/model_metadata.json")
    print(f"   └── plots/")
    print(f"       ├── dataset_distribution.png")
    print(f"       ├── image_characteristics.png")
    print(f"       ├── sample_images.png")
    print(f"       ├── training_history_initial_training.png")
    print(f"       ├── training_history_fine_tuning.png")
    print(f"       ├── confusion_matrix.png")
    print(f"       └── roc_curve.png")
    print("="*70)

if __name__ == "__main__":
    main()