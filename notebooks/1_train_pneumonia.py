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
print(" LUNG CANCER CLASSIFICATION - TRAINING SCRIPT")
print("="*60)
print(f"TensorFlow Version: {tf._version_}")
print(f"GPU Available: {len(tf.config.list_physical_devices('GPU')) > 0}")
print("="*60)

# Configuration
class Config:
    TRAIN_DIR = 'datasets/lung_cancer/Training'
    TEST_DIR = 'datasets/lung_cancer/Testing'
    MODEL_SAVE_PATH = 'models/lung_cancer_model.h5'
    IMG_SIZE = (224, 224)
    BATCH_SIZE = 32
    EPOCHS = 20
    LEARNING_RATE = 0.0001
    VALIDATION_SPLIT = 0.2
    CLASSES = ['glioma', 'meningioma', 'notumor', 'pituitary']

config = Config()