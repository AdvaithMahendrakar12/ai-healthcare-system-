# app.py - Multi-Disease Medical AI Diagnostic System

import os
import numpy as np
from flask import Flask, render_template, request, jsonify, url_for
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
from PIL import Image

# --- Configuration ---
# Model Paths
MODELS_CONFIG = {
    'lung_cancer': {
        'model_path': 'models/pneumonia_model.h5',
        'labels': ['Normal', 'Pneumonia/Lung Cancer Detected'],
        'img_size': 224,
        'description': 'Analyzes chest X-ray images to detect signs of lung cancer and pneumonia'
    },
    'fetal_ultrasound': {
        'model_path': 'models/keras/fetal_us_model.h5',
        'labels_path': 'models/tflite/fetal_us_labels.txt',
        'img_size': 224,
        'description': 'Classifies fetal ultrasound images for prenatal health assessment'
    },
    'brain_hemorrhage': {
        'model_path': 'models/brain_tumor_model.h5',
        'labels': ['No Tumor', 'Glioma', 'Meningioma', 'Pituitary'],
        'img_size': 224,
        'description': 'Detects brain tumors and hemorrhages from MRI scans'
    }
}

UPLOAD_FOLDER = 'static/uploads'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}

# Initialize Flask App
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Global dictionary to store loaded models
models = {}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def load_all_models():
    """Loads all ML models into global memory on server start."""
    print("\n" + "="*60)
    print(" 🏥 Medical AI Diagnostic System - Loading Models")
    print("="*60)
    
    for disease_key, config in MODELS_CONFIG.items():
        print(f"\n📦 Loading {disease_key.replace('_', ' ').title()} model...")
        try:
            model_path = config['model_path']
            if os.path.exists(model_path):
                models[disease_key] = {
                    'model': load_model(model_path, compile=False),
                    'config': config
                }
                
                # Load labels from file if specified
                if 'labels_path' in config and os.path.exists(config['labels_path']):
                    with open(config['labels_path'], 'r') as f:
                        models[disease_key]['labels'] = [line.strip() for line in f.readlines()]
                else:
                    models[disease_key]['labels'] = config.get('labels', ['Unknown'])
                
                print(f"   ✅ Loaded successfully!")
                print(f"   📋 Classes: {models[disease_key]['labels']}")
            else:
                print(f"   ⚠️  Model file not found: {model_path}")
                models[disease_key] = None
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            models[disease_key] = None
    
    print("\n" + "="*60)
    print(" 🚀 Server Ready!")
    print("="*60 + "\n")

def preprocess_image(image_path, img_size=224):
    """Preprocesses the uploaded image for model prediction."""
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array

def make_prediction(disease_type, image_path):
    """Makes a prediction using the appropriate model."""
    if disease_type not in models or models[disease_type] is None:
        return None, "Model not available"
    
    model_data = models[disease_type]
    model = model_data['model']
    labels = model_data['labels']
    img_size = model_data['config']['img_size']
    
    try:
        input_data = preprocess_image(image_path, img_size)
        predictions = model.predict(input_data, verbose=0)
        
        predicted_index = np.argmax(predictions[0])
        confidence = float(predictions[0][predicted_index] * 100)
        label = labels[predicted_index]
        
        # Get all class probabilities
        all_predictions = [
            {'label': labels[i], 'confidence': float(predictions[0][i] * 100)}
            for i in range(len(labels))
        ]
        all_predictions.sort(key=lambda x: x['confidence'], reverse=True)
        
        return {
            'label': label,
            'confidence': confidence,
            'all_predictions': all_predictions,
            'is_critical': predicted_index != 0  # Assuming index 0 is always "Normal"
        }, None
    except Exception as e:
        return None, str(e)

# ==================== ROUTES ====================

@app.route('/')
def index():
    """Landing page with all diagnostic options."""
    return render_template('index.html')

@app.route('/diagnose/<disease_type>', methods=['GET', 'POST'])
def diagnose(disease_type):
    """Generic diagnosis route for all disease types."""
    if disease_type not in MODELS_CONFIG:
        return render_template('error.html', message="Invalid diagnosis type"), 404
    
    config = MODELS_CONFIG[disease_type]
    result = None
    image_filename = None
    error = None
    
    if request.method == 'POST':
        if 'file' not in request.files:
            error = 'No file uploaded'
        else:
            file = request.files['file']
            if file.filename == '':
                error = 'No file selected'
            elif not allowed_file(file.filename):
                error = 'Invalid file type. Please upload JPG, PNG, or GIF images.'
            else:
                filename = secure_filename(file.filename)
                # Add disease type prefix to avoid conflicts
                filename = f"{disease_type}_{filename}"
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_filename = filename
                
                result, error = make_prediction(disease_type, file_path)
    
    return render_template(
        'diagnose.html',
        disease_type=disease_type,
        disease_name=disease_type.replace('_', ' ').title(),
        config=config,
        result=result,
        image_filename=image_filename,
        error=error
    )

@app.route('/about')
def about():
    """About page with information about the system."""
    return render_template('about.html')

@app.route('/api/predict/<disease_type>', methods=['POST'])
def api_predict(disease_type):
    """REST API endpoint for predictions."""
    if disease_type not in MODELS_CONFIG:
        return jsonify({'error': 'Invalid disease type'}), 400
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    if not allowed_file(file.filename):
        return jsonify({'error': 'Invalid file type'}), 400
    
    filename = secure_filename(file.filename)
    filename = f"api_{disease_type}_{filename}"
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)
    
    result, error = make_prediction(disease_type, file_path)
    
    if error:
        return jsonify({'error': error}), 500
    
    return jsonify({
        'success': True,
        'disease_type': disease_type,
        'prediction': result
    })

# Run the app
if __name__ == '__main__':
    load_all_models()
    app.run(debug=True, host='0.0.0.0', port=5000)
