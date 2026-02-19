# 🫁 Lung/Pneumonia Detection Model

A state-of-the-art deep learning system for automated pneumonia detection from chest X-ray images using EfficientNetB3 and transfer learning.

![License](https://img.shields.io/badge/license-MIT-blue.svg)
![Python](https://img.shields.io/badge/python-3.8%2B-blue)
![TensorFlow](https://img.shields.io/badge/TensorFlow-2.10%2B-orange)
![Status](https://img.shields.io/badge/status-production--ready-green)

## 📋 Quick Overview

- **Model Architecture:** EfficientNetB3 with custom classification head
- **Task:** Binary classification (NORMAL vs PNEUMONIA)
- **Performance:** 94.23% accuracy, 97.51% AUC-ROC, 96.92% sensitivity
- **Dataset:** Chest X-Ray Images (5,856 images)
- **Training Time:** ~2-3 hours on NVIDIA V100 GPU

## 🚀 Quick Start

### 1. Install Dependencies
```bash
pip install tensorflow==2.10.0 numpy pandas scikit-learn matplotlib seaborn opencv-python tqdm
```

### 2. Train the Model
```bash
python train_enhanced_lung_model.py
```

### 3. Evaluate the Model
```bash
python evaluate_enhanced_model.py
```

### 4. Make Predictions
```python
import tensorflow as tf
import cv2
import numpy as np

# Load model
model = tf.keras.models.load_model('models/lung_model.h5')

# Load and preprocess image
img = cv2.imread('chest_xray.jpg')
img = cv2.resize(img, (320, 320))
img = img / 255.0
img = np.expand_dims(img, axis=0)

# Predict
prediction = model.predict(img)[0][0]

if prediction > 0.5:
    print(f"PNEUMONIA detected (confidence: {prediction*100:.1f}%)")
else:
    print(f"NORMAL (confidence: {(1-prediction)*100:.1f}%)")
```

## 📁 Project Structure

```
Lung/
├── train_enhanced_lung_model.py    # Main training script
├── evaluate_enhanced_model.py      # Comprehensive evaluation
├── gradcam.py                      # Explainability visualization
├── DOCUMENTATION.md                # Complete documentation (70+ pages)
├── README.md                       # This file
├── dataset/
│   └── chest_xray/
│       ├── train/                  # Training data (5,216 images)
│       ├── val/                    # Validation data (16 images)
│       └── test/                   # Test data (624 images)
├── models/                         # Saved models
├── Pneumonia_plots/                # Training visualizations
├── reports/                        # Performance reports
├── results/                        # Evaluation results
└── logs/                           # TensorBoard logs
```

## 🎯 Key Features

### Advanced Model Architecture
- **Base:** EfficientNetB3 (pretrained on ImageNet)
- **Custom Head:** Dense layers with BatchNorm and Dropout
- **Regularization:** L2 regularization, Dropout (50%, 30%)
- **Parameters:** 12.8M trainable parameters

### Sophisticated Training Pipeline
- **Two-Phase Training:** Transfer learning + Fine-tuning
- **Data Augmentation:** 7 transformations for robust learning
- **Class Balancing:** Weighted loss for imbalanced dataset
- **Callbacks:** EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

### Comprehensive Evaluation
- **15+ Metrics:** Accuracy, AUC-ROC, Sensitivity, Specificity, F1, etc.
- **Visualizations:** Confusion matrix, ROC curve, PR curve
- **Interpretability:** Grad-CAM heatmaps
- **Clinical Metrics:** PPV, NPV, diagnostic accuracy

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| **Accuracy** | 94.23% |
| **AUC-ROC** | 0.9751 |
| **Sensitivity (Recall)** | 96.92% |
| **Specificity** | 89.74% |
| **Precision** | 94.03% |
| **F1-Score** | 0.9545 |
| **NPV** | 94.59% |

### Confusion Matrix
```
                Predicted
             NORMAL  PNEUMONIA
Actual NORMAL   210      24
       PNEUMONIA  12     378
```

### Clinical Interpretation
- ✅ **High Sensitivity (96.92%):** Excellent at detecting pneumonia cases
- ✅ **High AUC-ROC (0.9751):** Outstanding discriminative ability
- ✅ **High Precision (94.03%):** Trustworthy positive diagnoses
- ✅ **High NPV (94.59%):** Reliable negative results

## 🔬 Technical Details

### Dataset
- **Total Images:** 5,856 chest X-rays
- **Classes:** NORMAL (1,583) vs PNEUMONIA (4,273)
- **Format:** JPEG, grayscale
- **Preprocessed Size:** 320x320 pixels
- **Source:** Kaggle Chest X-Ray Images (Pneumonia)

### Data Augmentation
```python
- Rotation: ±15 degrees
- Width/Height shift: ±15%
- Zoom: ±15%
- Horizontal flip: Yes
- Brightness: [0.85, 1.15]
- Shear: ±5 degrees
```

### Training Configuration
```python
Image Size:         320x320x3
Batch Size:         16
Initial Epochs:     20
Fine-tune Epochs:   25
Initial LR:         0.001
Fine-tune LR:       0.00001
Optimizer:          Adam
Loss Function:      Binary Crossentropy (weighted)
```

## 📚 Documentation

For complete documentation including:
- Algorithm explanations
- Dataset cleaning procedures
- Model architecture details
- Training strategies
- Evaluation metrics
- Clinical interpretation
- Deployment guides
- Troubleshooting

**See:** [DOCUMENTATION.md](DOCUMENTATION.md) (70+ pages)

## 🛠️ Advanced Usage

### Custom Training Configuration
```python
# Edit Config class in train_enhanced_lung_model.py
class Config:
    DATA_DIR = 'dataset/chest_xray'
    IMG_SIZE = (320, 320)
    BATCH_SIZE = 16
    EPOCHS_INITIAL = 20
    EPOCHS_FINETUNE = 25
    BASE_MODEL = 'EfficientNetB3'  # or 'EfficientNetB4'
    INITIAL_LR = 1e-3
    FINETUNE_LR = 1e-5
```

### Grad-CAM Visualization
```bash
python gradcam.py --image path/to/xray.jpg --model models/lung_model.h5
```

### TensorBoard Monitoring
```bash
tensorboard --logdir=logs/ --port=6006
```

### Batch Prediction
```python
from tensorflow.keras.preprocessing.image import ImageDataGenerator

test_datagen = ImageDataGenerator(rescale=1./255)
test_gen = test_datagen.flow_from_directory(
    'test_images/',
    target_size=(320, 320),
    batch_size=16,
    class_mode='binary',
    shuffle=False
)

predictions = model.predict(test_gen)
```

## 🚢 Deployment

### Flask API
```python
from flask import Flask, request, jsonify
import tensorflow as tf
from PIL import Image
import numpy as np

app = Flask(__name__)
model = tf.keras.models.load_model('models/lung_model.h5')

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['image']
    img = Image.open(file.stream).convert('RGB')
    img = img.resize((320, 320))
    img_array = np.array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)
    
    prediction = model.predict(img_array)[0][0]
    
    return jsonify({
        'prediction': 'PNEUMONIA' if prediction > 0.5 else 'NORMAL',
        'confidence': float(prediction),
        'probability': {
            'NORMAL': float(1 - prediction),
            'PNEUMONIA': float(prediction)
        }
    })

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
```

### TensorFlow Lite (Mobile)
```python
# Convert to TFLite
converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]
tflite_model = converter.convert()

with open('models/lung_model.tflite', 'wb') as f:
    f.write(tflite_model)
```

### Docker
```dockerfile
FROM tensorflow/tensorflow:2.10.0-gpu

WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt

COPY . .

CMD ["python", "app.py"]
```

## 📈 Results Visualization

The training pipeline generates comprehensive visualizations:

1. **Dataset Analysis**
   - Class distribution
   - Image characteristics
   - Sample images

2. **Training Curves**
   - Accuracy over epochs
   - Loss over epochs
   - Learning rate schedule

3. **Evaluation Plots**
   - Confusion matrix (counts & normalized)
   - ROC curve with optimal threshold
   - Precision-Recall curve
   - Performance metrics bar chart
   - Prediction distribution

All visualizations are saved in `Pneumonia_plots/` and `evaluation_plots/`.

## ⚙️ System Requirements

### Training
- **GPU:** NVIDIA GTX 1060+ (6GB+ VRAM)
- **RAM:** 16GB+
- **Storage:** 50GB+ SSD
- **Time:** 2-3 hours

### Inference
- **GPU:** Optional (50-80ms) vs CPU (200-300ms)
- **RAM:** 4GB+
- **Storage:** 200MB (model file)

## 🔍 Model Interpretability

### Grad-CAM Heatmaps
The model includes Grad-CAM visualization to show which regions of the X-ray influenced the prediction:

- **Red/Warm colors:** High attention areas
- **Blue/Cool colors:** Low attention areas
- **Clinical value:** Verify model is focusing on relevant lung regions

### Uncertainty Quantification
```python
# Monte Carlo Dropout for uncertainty
predictions = []
for _ in range(100):
    pred = model(image, training=True)  # Keep dropout active
    predictions.append(pred)

mean_pred = np.mean(predictions)
uncertainty = np.std(predictions)

print(f"Prediction: {mean_pred:.3f} ± {uncertainty:.3f}")
```

## 🏥 Clinical Use Cases

### 1. Screening & Triage
- **Scenario:** Emergency department with high patient volume
- **Use:** Prioritize suspected pneumonia cases for radiologist review
- **Benefit:** Faster diagnosis, reduced wait times

### 2. Second Opinion
- **Scenario:** Support junior radiologists or general practitioners
- **Use:** AI provides second opinion alongside human diagnosis
- **Benefit:** Reduced diagnostic errors, increased confidence

### 3. Telemedicine
- **Scenario:** Remote healthcare settings without on-site radiologist
- **Use:** Preliminary diagnosis while awaiting expert review
- **Benefit:** Faster treatment initiation in remote areas

### 4. Quality Assurance
- **Scenario:** Regular audit of radiology reports
- **Use:** Flag discrepancies between AI and human diagnosis
- **Benefit:** Continuous quality improvement

## ⚠️ Important Disclaimers

### Medical Disclaimer
- This model is a **decision support tool**, not a diagnostic device
- **Always require radiologist confirmation** before clinical decisions
- Consider clinical presentation, patient history, and symptoms
- Not a substitute for professional medical judgment
- Regulatory approval required for clinical use (FDA, CE marking)

### Limitations
- Trained on specific X-ray equipment and protocols
- Does not distinguish pneumonia subtypes (bacterial, viral, fungal)
- Cannot assess severity or complications
- Not suitable for CT scans or other imaging modalities
- Limited pediatric training data

## 🤝 Contributing

Contributions are welcome! Areas for improvement:

1. **Model Enhancements**
   - Ensemble models
   - Multi-class classification (pneumonia subtypes)
   - Severity grading
   - Attention mechanisms

2. **Data Improvements**
   - Additional datasets (NIH ChestX-ray14, CheXpert)
   - Multi-view X-rays (frontal + lateral)
   - Temporal data (X-ray series)

3. **Clinical Features**
   - DICOM integration
   - HL7 FHIR compatibility
   - Mobile app development

4. **Research**
   - Federated learning
   - Self-supervised pretraining
   - Adversarial robustness

## 📝 Citation

If you use this model in your research or application, please cite:

```bibtex
@software{pneumonia_detection_2026,
  title = {Pneumonia Detection System using Deep Learning},
  author = {Medical AI Team},
  year = {2026},
  version = {1.0},
  url = {https://github.com/your-repo/lung-model}
}
```

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

```
MIT License

Copyright (c) 2026 Medical AI Team

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction...
```

## 🔗 Resources

### Documentation
- [Complete Documentation](DOCUMENTATION.md) - 70+ pages covering all aspects
- [Training Guide](#quick-start) - Step-by-step training instructions
- [API Reference](DOCUMENTATION.md#api-reference) - Function documentation

### Related Papers
- EfficientNet: Rethinking Model Scaling (Tan & Le, 2019)
- CheXNet: Radiologist-Level Pneumonia Detection (Rajpurkar et al., 2017)
- Identifying Medical Diagnoses by Deep Learning (Kermany et al., 2018)

### Datasets
- [Kaggle Chest X-Ray Images](https://www.kaggle.com/paultimothymooney/chest-xray-pneumonia)
- [NIH ChestX-ray14](https://nihcc.app.box.com/v/ChestXray-NIHCC)
- [CheXpert](https://stanfordmlgroup.github.io/competitions/chexpert/)
- [MIMIC-CXR](https://physionet.org/content/mimic-cxr/)

### Tools & Libraries
- [TensorFlow](https://www.tensorflow.org/)
- [Keras](https://keras.io/)
- [scikit-learn](https://scikit-learn.org/)
- [OpenCV](https://opencv.org/)

## 📞 Support

For questions, issues, or contributions:

- **GitHub Issues:** Create an issue for bug reports or feature requests
- **Documentation:** See [DOCUMENTATION.md](DOCUMENTATION.md) for detailed guides
- **Email:** Contact the development team

## 🙏 Acknowledgments

This project builds upon:
- Public chest X-ray datasets from medical institutions
- Open-source deep learning frameworks (TensorFlow, Keras)
- Medical imaging research community
- Clinical radiology expertise

Special thanks to:
- Kaggle for hosting the Chest X-Ray dataset
- Google for the EfficientNet architecture
- The TensorFlow team for excellent tools

---

## 📊 Project Statistics

- **Lines of Code:** ~2,000+
- **Documentation:** 70+ pages
- **Training Visualizations:** 8+ plots
- **Evaluation Metrics:** 15+ metrics
- **Model Parameters:** 12.8M trainable
- **Model Size:** ~155 MB
- **Training Time:** 2-3 hours
- **Inference Time:** 50-80ms (GPU), 200-300ms (CPU)

---

**Built with ❤️ for better healthcare through AI**

**Version:** 1.0  
**Last Updated:** January 2026  
**Status:** Production-Ready ✅

---

## Quick Links

- [📚 Complete Documentation](DOCUMENTATION.md)
- [🚀 Quick Start](#quick-start)
- [📊 Performance](#performance-metrics)
- [🛠️ Advanced Usage](#advanced-usage)
- [🚢 Deployment](#deployment)
- [🏥 Clinical Use Cases](#clinical-use-cases)
- [⚠️ Disclaimers](#important-disclaimers)

---

**Happy Coding! 🎉**
