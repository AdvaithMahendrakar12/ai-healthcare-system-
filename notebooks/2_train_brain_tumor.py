"""
🧠 BRAIN TUMOR CLASSIFICATION MODEL TRAINING
Run in VS Code: Right-click → Run Python File in Terminal
"""

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.applications import EfficientNetB0
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix
import warnings
warnings.filterwarnings('ignore')

print("="*60)
print("🧠 BRAIN TUMOR CLASSIFICATION - TRAINING SCRIPT")
print("="*60)
print(f"TensorFlow Version: {tf.__version__}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("="*60)

# Configuration
class Config:
    TRAIN_DIR = 'datasets/brain_tumor/Training'
    TEST_DIR = 'datasets/brain_tumor/Testing'
    MODEL_SAVE_PATH = 'models/brain_tumor_model.h5'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT = 0.2
    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

config = Config()

# Verify dataset
if not os.path.exists(config.TRAIN_DIR):
    print(f"❌ ERROR: Dataset not found at {config.TRAIN_DIR}")
    exit(1)

print("✅ Dataset found!")

# Data Augmentation
print("\n📊 Setting up data augmentation...")
train_datagen = ImageDataGenerator(
    rescale=1./255,
    rotation_range=25,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    vertical_flip=True,
    fill_mode='nearest',
    validation_split=config.VALIDATION_SPLIT
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Load Data
print("📥 Loading datasets...")
train_generator = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    subset='training',
    shuffle=True
)

val_generator = train_datagen.flow_from_directory(
    config.TRAIN_DIR,
    target_size=config.IMG_SIZE,
    batch_size=config.BATCH_SIZE,
    class_mode='categorical',
    subset='validation',
    shuffle=False
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

# Build Model
print("\n🏗️ Building model...")
base_model = EfficientNetB0(
    weights='imagenet',
    include_top=False,
    input_shape=(*config.IMG_SIZE, 3)
)
base_model.trainable = False

model = keras.Sequential([
    base_model,
    layers.GlobalAveragePooling2D(),
    layers.BatchNormalization(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(256, activation='relu'),
    layers.Dropout(0.3),
    layers.Dense(4, activation='softmax')
])

model.compile(
    optimizer=keras.optimizers.Adam(learning_rate=config.LEARNING_RATE),
    loss='categorical_crossentropy',
    metrics=['accuracy', keras.metrics.Precision(), keras.metrics.Recall()]
)

print("✅ Model built successfully!")
print(f"📊 Total parameters: {model.count_params():,}")

# Callbacks
callbacks = [
    EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
    ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, min_lr=1e-7, verbose=1),
    ModelCheckpoint(config.MODEL_SAVE_PATH, monitor='val_accuracy', save_best_only=True, verbose=1)
]

# Train
print("\n🚀 Starting training...")
print("="*60)

history = model.fit(
    train_generator,
    validation_data=val_generator,
    epochs=config.EPOCHS,
    callbacks=callbacks,
    verbose=1
)

print("="*60)
print("✅ Training completed!")

# Evaluate
print("\n📊 Evaluating on test set...")
test_results = model.evaluate(test_generator)

print("\n" + "="*60)
print("📈 FINAL RESULTS")
print("="*60)
print(f"✅ Test Accuracy:  {test_results[1]*100:.2f}%")
print(f"✅ Test Precision: {test_results[2]*100:.2f}%")
print(f"✅ Test Recall:    {test_results[3]*100:.2f}%")
print("="*60)

# Predictions
print("\n🔮 Generating predictions...")
test_generator.reset()
y_pred_probs = model.predict(test_generator, verbose=0)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = test_generator.classes

# Confusion Matrix
cm = confusion_matrix(y_true, y_pred)
print("\n📋 Classification Report:")
print(classification_report(y_true, y_pred, target_names=config.CLASSES))

# Plot results
plt.figure(figsize=(15, 4))

plt.subplot(1, 3, 1)
plt.plot(history.history['accuracy'], label='Train')
plt.plot(history.history['val_accuracy'], label='Validation')
plt.title('Model Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 2)
plt.plot(history.history['loss'], label='Train')
plt.plot(history.history['val_loss'], label='Validation')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)

plt.subplot(1, 3, 3)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=config.CLASSES, yticklabels=config.CLASSES)
plt.title('Confusion Matrix')
plt.ylabel('True')
plt.xlabel('Predicted')

plt.tight_layout()
plt.savefig('models/brain_tumor_training_results.png', dpi=150, bbox_inches='tight')
print("📊 Results plot saved to: models/brain_tumor_training_results.png")

print("\n" + "="*60)
print("🎉 BRAIN TUMOR MODEL TRAINING COMPLETE!")
print("="*60)
print("✅ Next: Run 'python notebooks/3_train_chatbot.py'")