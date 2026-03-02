"""
🚀 FIXED ADVANCED BRAIN TUMOR CLASSIFICATION TRAINING
======================================================
Optimized for Maximum Accuracy with:
- EfficientNetV2 backbone
- Class weight balancing
- Label smoothing
- Cosine annealing learning rate
- Test-Time Augmentation (TTA)
- Three-stage training
- Fixed mixed precision issues
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, Model
from tensorflow.keras.applications import EfficientNetV2S
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
)
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import json
import math
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

print("="*70)
print("🧠 ADVANCED BRAIN TUMOR CLASSIFICATION - HIGH ACCURACY TRAINING")
print("="*70)
print(f"TensorFlow Version: {tf.__version__}")
gpus = tf.config.list_physical_devices('GPU')
print(f"GPU Available: {len(gpus) > 0}")
if gpus:
    for gpu in gpus:
        try:
            tf.config.experimental.set_memory_growth(gpu, True)
            print(f"  GPU: {gpu.name}")
        except:
            pass
print("="*70)

# ============================================================================
# CONFIGURATION
# ============================================================================
class Config:
    # Data paths
    TRAIN_DIR = 'brain_tumor/cleaned_data/Training'
    TEST_DIR = 'brain_tumor/cleaned_data/Testing'
    
    # Model paths
    MODEL_PATH = 'brain_tumor/models/brain_tumor_model.h5'
    MODEL_BEST = 'brain_tumor/models/brain_tumor_best.h5'
    MODEL_FINAL = 'brain_tumor/models/brain_tumor_final.h5'
    
    # Training parameters
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 16
    EPOCHS_STAGE1 = 25
    EPOCHS_STAGE2 = 35
    EPOCHS_STAGE3 = 15
    
    # Learning rates
    INITIAL_LR = 1e-3
    FINE_TUNE_LR = 1e-4
    FINAL_LR = 5e-5
    MIN_LR = 1e-7
    
    # Regularization
    LABEL_SMOOTHING = 0.1
    VALIDATION_SPLIT = 0.15
    
    # Classes
    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']
    NUM_CLASSES = 4
    
    # TTA
    TTA_STEPS = 5

config = Config()

# ============================================================================
# DATA LOADING
# ============================================================================
def create_data_generators():
    """Create data generators with augmentation"""
    print("\n📊 Setting up data augmentation...")
    
    # Training augmentation - conservative for medical images
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=20,
        width_shift_range=0.2,
        height_shift_range=0.2,
        shear_range=0.2,
        zoom_range=0.2,
        horizontal_flip=True,
        vertical_flip=True,
        brightness_range=[0.8, 1.2],
        fill_mode='constant',
        cval=0,
        validation_split=config.VALIDATION_SPLIT
    )
    
    test_datagen = ImageDataGenerator(rescale=1./255)
    
    print("📥 Loading datasets...")
    
    train_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='training',
        shuffle=True,
        seed=42
    )
    
    val_generator = train_datagen.flow_from_directory(
        config.TRAIN_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        subset='validation',
        shuffle=False,
        seed=42
    )
    
    test_generator = test_datagen.flow_from_directory(
        config.TEST_DIR,
        target_size=config.IMG_SIZE,
        batch_size=config.BATCH_SIZE,
        class_mode='categorical',
        shuffle=False
    )
    
    print(f"✅ Training samples: {train_generator.samples}")
    print(f"✅ Validation samples: {val_generator.samples}")
    print(f"✅ Test samples: {test_generator.samples}")
    
    # Calculate class weights
    print("\n📊 Computing class weights...")
    class_counts = []
    for class_name in config.CLASSES:
        idx = train_generator.class_indices[class_name]
        count = sum(train_generator.classes == idx)
        class_counts.append(count)
        print(f"  {class_name}: {count} samples")
    
    total = sum(class_counts)
    class_weights = {i: total / (len(class_counts) * count) 
                     for i, count in enumerate(class_counts)}
    
    print("\n📊 Class weights:")
    for i, class_name in enumerate(config.CLASSES):
        print(f"  {class_name}: {class_weights[i]:.3f}")
    
    return train_generator, val_generator, test_generator, class_weights

# ============================================================================
# MODEL BUILDING
# ============================================================================
def build_model():
    """Build EfficientNetV2-based model"""
    print(f"\n🏗️ Building model...")
    
    base_model = EfficientNetV2S(
        weights='imagenet',
        include_top=False,
        input_shape=(*config.IMG_SIZE, 3),
        include_preprocessing=True
    )
    
    base_model.trainable = False
    
    inputs = keras.Input(shape=(*config.IMG_SIZE, 3))
    
    # Base model
    x = base_model(inputs, training=False)
    
    # Dual pooling for richer features
    gap = layers.GlobalAveragePooling2D()(x)
    gmp = layers.GlobalMaxPooling2D()(x)
    x = layers.Concatenate()([gap, gmp])
    
    # Classification head
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(512, activation='swish', 
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.4)(x)
    
    x = layers.Dense(256, activation='swish',
                     kernel_regularizer=keras.regularizers.l2(1e-4))(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.3)(x)
    
    outputs = layers.Dense(config.NUM_CLASSES, activation='softmax')(x)
    
    model = keras.Model(inputs, outputs, name='BrainTumorClassifier')
    
    print("✅ Model built successfully!")
    print(f"📊 Total parameters: {model.count_params():,}")
    
    return model, base_model

# ============================================================================
# LEARNING RATE SCHEDULE
# ============================================================================
def cosine_decay_with_warmup(epoch, total_epochs, initial_lr, warmup_epochs=3, min_lr=1e-7):
    """Cosine decay with warmup"""
    if epoch < warmup_epochs:
        return initial_lr * (epoch + 1) / warmup_epochs
    else:
        progress = (epoch - warmup_epochs) / (total_epochs - warmup_epochs)
        return min_lr + (initial_lr - min_lr) * (1 + math.cos(math.pi * progress)) / 2

# ============================================================================
# TRAINING STAGES
# ============================================================================
def train_stage1(model, train_gen, val_gen, class_weights):
    """Stage 1: Train head only"""
    print("\n" + "="*70)
    print("🚀 STAGE 1: TRAINING HEAD (Base Frozen)")
    print("="*70)
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.INITIAL_LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=['accuracy', 
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    # Custom learning rate schedule
    def lr_schedule_stage1(epoch):
        return cosine_decay_with_warmup(
            epoch, config.EPOCHS_STAGE1, config.INITIAL_LR, 
            warmup_epochs=3, min_lr=config.MIN_LR
        )
    
    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_schedule_stage1, verbose=1),
        EarlyStopping(
            monitor='val_accuracy',
            patience=8,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            'brain_tumor/models/stage1_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_STAGE1,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    best_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Stage 1 Complete! Best Val Accuracy: {best_acc*100:.2f}%")
    return history

def train_stage2(model, base_model, train_gen, val_gen, class_weights):
    """Stage 2: Fine-tune top layers"""
    print("\n" + "="*70)
    print("🔥 STAGE 2: FINE-TUNING TOP LAYERS")
    print("="*70)
    
    # Unfreeze top 30% of base
    base_model.trainable = True
    num_layers = len(base_model.layers)
    fine_tune_from = int(num_layers * 0.7)
    
    for layer in base_model.layers[:fine_tune_from]:
        layer.trainable = False
    
    trainable = sum([1 for layer in base_model.layers if layer.trainable])
    print(f"📊 Unfrozen: {trainable}/{num_layers} layers ({trainable/num_layers*100:.1f}%)")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.FINE_TUNE_LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    def lr_schedule_stage2(epoch):
        return cosine_decay_with_warmup(
            epoch, config.EPOCHS_STAGE2, config.FINE_TUNE_LR,
            warmup_epochs=5, min_lr=config.MIN_LR
        )
    
    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_schedule_stage2, verbose=1),
        EarlyStopping(
            monitor='val_accuracy',
            patience=12,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            config.MODEL_BEST,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=config.MIN_LR,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_STAGE2,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    best_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Stage 2 Complete! Best Val Accuracy: {best_acc*100:.2f}%")
    return history

def train_stage3(model, base_model, train_gen, val_gen, class_weights):
    """Stage 3: Full fine-tuning"""
    print("\n" + "="*70)
    print("⚡ STAGE 3: FULL FINE-TUNING")
    print("="*70)
    
    base_model.trainable = True
    print(f"📊 All layers trainable")
    
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=config.FINAL_LR),
        loss=keras.losses.CategoricalCrossentropy(label_smoothing=config.LABEL_SMOOTHING),
        metrics=['accuracy',
                 keras.metrics.Precision(name='precision'),
                 keras.metrics.Recall(name='recall'),
                 keras.metrics.AUC(name='auc')]
    )
    
    def lr_schedule_stage3(epoch):
        return cosine_decay_with_warmup(
            epoch, config.EPOCHS_STAGE3, config.FINAL_LR,
            warmup_epochs=2, min_lr=config.MIN_LR
        )
    
    callbacks = [
        keras.callbacks.LearningRateScheduler(lr_schedule_stage3, verbose=1),
        EarlyStopping(
            monitor='val_accuracy',
            patience=10,
            restore_best_weights=True,
            verbose=1
        ),
        ModelCheckpoint(
            config.MODEL_FINAL,
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        )
    ]
    
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=config.EPOCHS_STAGE3,
        callbacks=callbacks,
        class_weight=class_weights,
        verbose=1
    )
    
    best_acc = max(history.history['val_accuracy'])
    print(f"\n✅ Stage 3 Complete! Best Val Accuracy: {best_acc*100:.2f}%")
    return history

# ============================================================================
# TEST-TIME AUGMENTATION
# ============================================================================
def predict_with_tta(model, test_gen, tta_steps=5):
    """Predict with Test-Time Augmentation"""
    print(f"\n🔮 Predicting with TTA ({tta_steps} augmentations)...")
    
    tta_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=10,
        width_shift_range=0.05,
        height_shift_range=0.05,
        horizontal_flip=True,
        zoom_range=0.05
    )
    
    all_predictions = []
    
    # Original
    test_gen.reset()
    pred = model.predict(test_gen, verbose=0)
    all_predictions.append(pred)
    
    # Augmented
    for i in range(tta_steps - 1):
        tta_gen = tta_datagen.flow_from_directory(
            config.TEST_DIR,
            target_size=config.IMG_SIZE,
            batch_size=config.BATCH_SIZE,
            class_mode='categorical',
            shuffle=False,
            seed=42 + i
        )
        pred = model.predict(tta_gen, verbose=0)
        all_predictions.append(pred)
    
    return np.mean(all_predictions, axis=0)

# ============================================================================
# EVALUATION
# ============================================================================
def evaluate_model(model, test_gen, use_tta=True):
    """Evaluate model"""
    print("\n" + "="*70)
    print("📊 EVALUATING MODEL")
    print("="*70)
    
    # Standard evaluation
    test_gen.reset()
    test_results = model.evaluate(test_gen, verbose=1)
    
    print(f"\n📊 Standard Evaluation:")
    print(f"  Loss:      {test_results[0]:.4f}")
    print(f"  Accuracy:  {test_results[1]*100:.2f}%")
    print(f"  Precision: {test_results[2]*100:.2f}%")
    print(f"  Recall:    {test_results[3]*100:.2f}%")
    print(f"  AUC:       {test_results[4]:.4f}")
    
    # Predictions
    if use_tta:
        y_pred_probs = predict_with_tta(model, test_gen, config.TTA_STEPS)
    else:
        test_gen.reset()
        y_pred_probs = model.predict(test_gen, verbose=1)
    
    y_pred = np.argmax(y_pred_probs, axis=1)
    y_true = test_gen.classes
    
    tta_accuracy = np.mean(y_pred == y_true)
    print(f"\n📊 {'TTA' if use_tta else 'Standard'} Accuracy: {tta_accuracy*100:.2f}%")
    
    # Classification report
    print("\n📋 Classification Report:")
    report = classification_report(y_true, y_pred, target_names=config.CLASSES, digits=4)
    print(report)
    
    # Confusion matrix
    cm = confusion_matrix(y_true, y_pred)
    
    # Per-class accuracy
    class_accuracy = cm.diagonal() / cm.sum(axis=1)
    print("\n📊 Per-Class Accuracy:")
    for i, class_name in enumerate(config.CLASSES):
        print(f"  {class_name}: {class_accuracy[i]*100:.2f}%")
    
    # Save results
    results = {
        'test_loss': float(test_results[0]),
        'test_accuracy': float(test_results[1]),
        'test_accuracy_tta': float(tta_accuracy) if use_tta else None,
        'test_precision': float(test_results[2]),
        'test_recall': float(test_results[3]),
        'test_auc': float(test_results[4]),
        'class_accuracy': {config.CLASSES[i]: float(class_accuracy[i]) 
                          for i in range(len(config.CLASSES))},
        'confusion_matrix': cm.tolist()
    }
    
    with open('brain_tumor/results/test_results_final.json', 'w') as f:
        json.dump(results, f, indent=4)
    
    print("\n✅ Results saved")
    return cm, results

# ============================================================================
# VISUALIZATION
# ============================================================================
def plot_results(histories, cm):
    """Plot training results"""
    print("\n📊 Creating visualizations...")
    
    # Combine histories
    combined = {'accuracy': [], 'val_accuracy': [], 'loss': [], 'val_loss': []}
    for h in histories:
        for key in combined:
            if key in h.history:
                combined[key].extend(h.history[key])
    
    fig = plt.figure(figsize=(20, 10))
    
    # Accuracy
    plt.subplot(2, 3, 1)
    plt.plot(combined['accuracy'], label='Train', linewidth=2, color='#2196F3')
    plt.plot(combined['val_accuracy'], label='Val', linewidth=2, color='#FF5722')
    plt.title('Model Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Loss
    plt.subplot(2, 3, 2)
    plt.plot(combined['loss'], label='Train', linewidth=2, color='#2196F3')
    plt.plot(combined['val_loss'], label='Val', linewidth=2, color='#FF5722')
    plt.title('Model Loss', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    # Best accuracy progress
    plt.subplot(2, 3, 3)
    val_acc = combined['val_accuracy']
    best = [max(val_acc[:i+1]) for i in range(len(val_acc))]
    plt.plot(best, linewidth=2, color='#4CAF50')
    plt.fill_between(range(len(best)), best, alpha=0.3, color='#4CAF50')
    plt.title('Best Val Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.grid(True, alpha=0.3)
    
    # Confusion Matrix
    plt.subplot(2, 3, 4)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES,
                annot_kws={'size': 12})
    plt.title('Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    # Normalized CM
    plt.subplot(2, 3, 5)
    cm_norm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    sns.heatmap(cm_norm, annot=True, fmt='.2%', cmap='Greens',
                xticklabels=config.CLASSES,
                yticklabels=config.CLASSES,
                annot_kws={'size': 11})
    plt.title('Normalized Confusion Matrix', fontsize=14, fontweight='bold')
    plt.ylabel('True')
    plt.xlabel('Predicted')
    
    # Per-class accuracy
    plt.subplot(2, 3, 6)
    class_acc = cm.diagonal() / cm.sum(axis=1)
    colors = ['#E91E63', '#9C27B0', '#4CAF50', '#FF9800']
    bars = plt.bar(config.CLASSES, class_acc * 100, color=colors, edgecolor='black')
    plt.title('Per-Class Accuracy', fontsize=14, fontweight='bold')
    plt.xlabel('Class')
    plt.ylabel('Accuracy (%)')
    plt.ylim(0, 105)
    for bar, acc in zip(bars, class_acc):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 1,
                f'{acc*100:.1f}%', ha='center', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='y')
    
    plt.tight_layout()
    plt.savefig('brain_tumor/results/training_results_final.png', dpi=300, bbox_inches='tight')
    print("✅ Visualization saved")
    plt.close()

# ============================================================================
# SAVE MODEL
# ============================================================================
def save_model(model):
    """Save model for deployment"""
    print("\n💾 Saving model...")
    
    model.save(config.MODEL_PATH)
    print(f"✅ Model saved: {config.MODEL_PATH}")
    
    class_info = {
        'classes': config.CLASSES,
        'num_classes': config.NUM_CLASSES,
        'class_indices': {name: i for i, name in enumerate(config.CLASSES)},
        'input_size': list(config.IMG_SIZE)
    }
    
    with open('brain_tumor/models/class_labels.json', 'w') as f:
        json.dump(class_info, f, indent=4)
    print("✅ Class labels saved")


def save_training_contract(results):
    """Save standardized training contract for app config synchronization."""
    if os.path.exists(config.MODEL_FINAL):
        resolved_model_path = config.MODEL_FINAL
    elif os.path.exists(config.MODEL_BEST):
        resolved_model_path = config.MODEL_BEST
    else:
        resolved_model_path = config.MODEL_PATH

    contract = {
        "disease_type": "brain_tumor",
        "model_path": resolved_model_path,
        "model_type": "classification",
        "img_size": config.IMG_SIZE[0],
        "labels": ["Glioma Tumor", "Meningioma Tumor", "No Tumor (Healthy)", "Pituitary Tumor"],
        "labels_path": None,
        "architecture": "EfficientNetV2S",
        "description": "Detects and classifies brain tumors from MRI scans.",
        "metrics": {
            "test_accuracy": float(results.get("test_accuracy", 0.0)),
            "test_accuracy_tta": float(results.get("test_accuracy_tta", 0.0)) if results.get("test_accuracy_tta") is not None else None,
            "test_precision": float(results.get("test_precision", 0.0)),
            "test_recall": float(results.get("test_recall", 0.0)),
            "test_auc": float(results.get("test_auc", 0.0)),
        },
        "created_at": datetime.now().isoformat()
    }

    with open('brain_tumor/models/contract_latest.json', 'w') as f:
        json.dump(contract, f, indent=4)
    print("✅ Training contract saved: brain_tumor/models/contract_latest.json")

# ============================================================================
# MAIN
# ============================================================================
def main():
    """Main training pipeline"""
    
    if not os.path.exists(config.TRAIN_DIR):
        print("❌ ERROR: Cleaned dataset not found!")
        print("➡️  Run: python brain_tumor/1_clean_dataset.py first")
        return
    
    os.makedirs('brain_tumor/results', exist_ok=True)
    os.makedirs('brain_tumor/models', exist_ok=True)
    
    # Load data
    train_gen, val_gen, test_gen, class_weights = create_data_generators()
    
    # Build model
    model, base_model = build_model()
    
    # Three-stage training
    histories = []
    histories.append(train_stage1(model, train_gen, val_gen, class_weights))
    histories.append(train_stage2(model, base_model, train_gen, val_gen, class_weights))
    histories.append(train_stage3(model, base_model, train_gen, val_gen, class_weights))
    
    # Load best model
    print("\n📥 Loading best model...")
    if os.path.exists(config.MODEL_FINAL):
        model = keras.models.load_model(config.MODEL_FINAL)
    elif os.path.exists(config.MODEL_BEST):
        model = keras.models.load_model(config.MODEL_BEST)
    
    # Evaluate
    cm, results = evaluate_model(model, test_gen, use_tta=True)
    
    # Plot
    plot_results(histories, cm)
    
    # Save
    save_model(model)
    save_training_contract(results)
    
    # Summary
    print("\n" + "="*70)
    print("🎉 TRAINING COMPLETE!")
    print("="*70)
    print(f"✅ Final Accuracy: {results['test_accuracy']*100:.2f}%")
    print(f"✅ TTA Accuracy:   {results['test_accuracy_tta']*100:.2f}%")
    print(f"✅ Model: {config.MODEL_PATH}")
    print("\n📊 Per-Class Performance:")
    for cls, acc in results['class_accuracy'].items():
        print(f"   {cls}: {acc*100:.2f}%")
    print("="*70)

if __name__ == "__main__":
    main()
