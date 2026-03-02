# app.py - Multi-Disease Medical AI Diagnostic System

import os
import json
import numpy as np
import cv2
import h5py
import tempfile
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Dict, List, Optional

from flask import Flask, render_template, request, jsonify, session
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from werkzeug.utils import secure_filename
import tensorflow.keras.backend as K

from chatbot.data.utils.chatbot import load_model as load_nlp_model, chat as nlp_chat
from chatbot.retriever import load_index, retrieve

# --- Configuration ---
# --- MODELS_CONFIG START ---
MODELS_CONFIG = {'pneumonia': {'model_path': 'Lung/models/lung_model_final_20260113_125327.h5',
               'labels': ['NORMAL', 'PNEUMONIA'],
               'img_size': 320,
               'model_type': 'classification',
               'description': 'Analyzes chest X-ray images to detect pneumonia with 74% accuracy using EfficientNetB3',
               'accuracy': 0.74,
               'architecture': 'EfficientNetB3'},
 'brain_tumor': {'model_path': 'brain_tumor/models/brain_tumor_final.h5',
                 'labels': ['Glioma Tumor', 'Meningioma Tumor', 'No Tumor (Healthy)', 'Pituitary Tumor'],
                 'img_size': 224,
                 'model_type': 'classification',
                 'description': 'Detects brain tumors from MRI scans with 92% accuracy. Classifies into Glioma, '
                                'Meningioma, Pituitary tumors, or healthy brain.',
                 'accuracy': 0.92,
                 'architecture': 'EfficientNetV2S'},
 'fetal_ultrasound': {'model_path': 'Fetal_Ultrasound/training/fetal_ultrasound_attention_unet_20260213_235033_best.h5',
                      'labels': ['Fetal Head Contour Segmentation'],
                      'img_size': 256,
                      'model_type': 'segmentation',
                      'description': 'Performs semantic segmentation to detect and outline fetal head in ultrasound '
                                     'images using Attention U-Net architecture',
                      'dice_coefficient': 0.285,
                      'architecture': 'Attention U-Net'}}
# --- MODELS_CONFIG END ---

UPLOAD_FOLDER = 'static/uploads'
RESULTS_FOLDER = 'static/results'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'gif'}
CHAT_HISTORY_KEY = 'chat_history_by_topic'
LATEST_DIAGNOSIS_KEY = 'latest_diagnosis_by_topic'
CHAT_PROMPT_PATH = Path('chatbot/prompts/system_prompt.txt')


# Initialize Flask App
app = Flask(__name__)
app.secret_key = os.getenv('FLASK_SECRET_KEY', 'local-dev-secret-change-me')
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['RESULTS_FOLDER'] = RESULTS_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size
app.config['CHAT_MAX_TURNS'] = int(os.getenv('CHAT_MAX_TURNS', '6'))
app.config['CHAT_MAX_TOKENS'] = int(os.getenv('CHAT_MAX_TOKENS', '512'))
app.config['CHAT_TEMPERATURE'] = float(os.getenv('CHAT_TEMPERATURE', '0.2'))

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(RESULTS_FOLDER, exist_ok=True)

# Global dictionary to store loaded models
models: Dict[str, Optional[Dict]] = {}
nlp_system = {
    'model': None,
    'words': None,
    'classes': None,
    'intents': None
}


# ==================== CUSTOM METRICS FOR SEGMENTATION ====================

def dice_coef(y_true, y_pred, smooth=1e-6):
    """Dice coefficient for segmentation evaluation."""
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    intersection = K.sum(y_true_f * y_pred_f)
    return (2. * intersection + smooth) / (K.sum(y_true_f) + K.sum(y_pred_f) + smooth)


def dice_coef_loss(y_true, y_pred):
    """Dice loss function."""
    return 1 - dice_coef(y_true, y_pred)


def iou_score(y_true, y_pred, smooth=1e-6):
    """IoU (Intersection over Union) score."""
    intersection = K.sum(K.abs(y_true * y_pred))
    union = K.sum(y_true) + K.sum(y_pred) - intersection
    return (intersection + smooth) / (union + smooth)


# Custom objects for loading segmentation models
CUSTOM_OBJECTS = {
    'dice_coef': dice_coef,
    'dice_coef_loss': dice_coef_loss,
    'iou_score': iou_score
}


# ==================== CHAT HELPERS ====================

def _safe_topic() -> str:
    return 'general'


def _get_chat_store() -> Dict[str, List[Dict[str, str]]]:
    store = session.get(CHAT_HISTORY_KEY)
    if isinstance(store, dict):
        return store
    return {}


def get_chat_history() -> List[Dict[str, str]]:
    store = _get_chat_store()
    return list(store.get('general', []))


def append_chat_history(role: str, content: str) -> None:
    store = _get_chat_store()
    topic_history = list(store.get('general', []))
    topic_history.append({
        'role': role,
        'content': content,
        'timestamp': datetime.now(timezone.utc).isoformat(),
    })

    max_messages = max(2, app.config['CHAT_MAX_TURNS'] * 2)
    store['general'] = topic_history[-max_messages:]
    session[CHAT_HISTORY_KEY] = store
    session.modified = True


def reset_chat() -> int:
    store = {}
    session[CHAT_HISTORY_KEY] = store
    session.modified = True
    return 0


def _get_latest_diagnosis_store() -> Dict[str, Dict]:
    store = session.get(LATEST_DIAGNOSIS_KEY)
    if isinstance(store, dict):
        return store
    return {}


def get_latest_diagnosis() -> Optional[Dict]:
    return _get_latest_diagnosis_store().get('general')


def store_latest_diagnosis(prediction: Dict) -> None:
    latest = _get_latest_diagnosis_store()

    summary = {
        'label': prediction.get('label'),
        'confidence': round(float(prediction.get('confidence', 0.0)), 2),
        'model_type': prediction.get('model_type', 'classification'),
        'is_critical': bool(prediction.get('is_critical', False)),
        'timestamp': datetime.now(timezone.utc).isoformat(),
    }

    if prediction.get('model_type') == 'classification':
        summary['top_classes'] = prediction.get('all_predictions', [])[:3]
    else:
        summary['coverage_percent'] = round(float(prediction.get('coverage_percent', 0.0)), 2)

    latest['general'] = summary
    session[LATEST_DIAGNOSIS_KEY] = latest
    session.modified = True


def _load_prompt_template() -> str:
    if CHAT_PROMPT_PATH.exists():
        return CHAT_PROMPT_PATH.read_text(encoding='utf-8')
    return (
        'You are a general medical assistant. '
        'Educational support only, not a medical diagnosis.'
    )


def _diagnosis_context_text() -> Optional[str]:
    latest = get_latest_diagnosis()
    if not latest:
        return None

    context = [
        f"Latest prediction: {latest.get('label')}",
        f"Confidence: {latest.get('confidence')}%",
        f"Critical flag: {latest.get('is_critical')}",
    ]

    if latest.get('model_type') == 'segmentation':
        context.append(f"Coverage: {latest.get('coverage_percent', 0.0)}%")
    else:
        top = latest.get('top_classes') or []
        if top:
            context.append(f"Top classes: {json.dumps(top)}")

    return "\n".join(context)


def _ensure_disclaimer(text: str) -> str:
    disclaimer = 'Educational support only, not a medical diagnosis.'
    if disclaimer.lower() in text.lower():
        return text
    return f"{text.rstrip()}\n\n{disclaimer}"


def build_chat_messages(user_message: str, use_latest_diagnosis: bool) -> List[Dict[str, str]]:
    system_prompt = _load_prompt_template()

    messages: List[Dict[str, str]] = [
        {'role': 'system', 'content': system_prompt}
    ]

    if use_latest_diagnosis:
        ctx = _diagnosis_context_text()
        if ctx:
            messages.append({'role': 'system', 'content': f'Latest diagnostic context:\n{ctx}'})

    for msg in get_chat_history():
        role = msg.get('role', '')
        content = msg.get('content', '')
        if role in ('user', 'assistant') and content:
            messages.append({'role': role, 'content': content})

    messages.append({'role': 'user', 'content': user_message})
    return messages


# ==================== KERAS VERSION COMPATIBILITY ====================

def _patch_keras3_model_config(config_str):
    """
    Patch a Keras 3.x model config JSON so it can be loaded by Keras 2.x.

    Keras 3 introduces several config changes that break Keras 2 loading:
      1. InputLayer uses 'batch_shape' instead of 'batch_input_shape'
      2. All layers store 'dtype' as a DTypePolicy dict instead of a plain string
      3. Layers may include unknown keys: 'dtype_policy', 'dtype_policy_name',
         'quantization_mode', 'build_config', 'compiled_trainable'
      4. `inbound_nodes` format changed from a list of `[layer_name, node_id, tensor_id, kwargs]`
         to a list of `{'args': [...], 'kwargs': {...}}`
    """
    try:
        config = json.loads(config_str)
    except Exception:
        return config_str  # Return as-is if not valid JSON

    # Keys in layer configs not understood by Keras 2
    unknown_layer_keys = [
        'dtype_policy', 'dtype_policy_name', 'quantization_mode',
        'build_config', 'compiled_trainable', 'activity_regularizer',
    ]

    def simplify_dtype(dtype_val):
        if isinstance(dtype_val, dict):
            if dtype_val.get('class_name') == 'DTypePolicy':
                return dtype_val.get('config', {}).get('name', 'float32')
            return dtype_val.get('config', {}).get('name', 'float32')
        return dtype_val

    def fix_inbound_nodes(inbound_nodes):
        if not isinstance(inbound_nodes, list) or len(inbound_nodes) == 0:
            return inbound_nodes

        if isinstance(inbound_nodes[0], list):
            return inbound_nodes

        def extract_keras_tensors(obj):
            results = []
            if isinstance(obj, dict) and obj.get('class_name') == '__keras_tensor__':
                kh = obj.get('config', {}).get('keras_history', [])
                if len(kh) >= 3:
                    results.append(kh)
            elif isinstance(obj, list):
                for item in obj:
                    results.extend(extract_keras_tensors(item))
            return results

        fixed_nodes = []
        for node in inbound_nodes:
            if not (isinstance(node, dict) and 'args' in node):
                fixed_nodes.append(node)
                continue
            args = node.get('args', [])
            kwargs = node.get('kwargs', {})
            histories = extract_keras_tensors(args)
            node_connections = []
            for kh in histories:
                node_connections.append([kh[0], kh[1], kh[2], kwargs])
            if node_connections:
                fixed_nodes.append(node_connections)
            else:
                fixed_nodes.append(node)
        return fixed_nodes

    def fix_layer_config(layer_cfg):
        if not isinstance(layer_cfg, dict):
            return layer_cfg
        if 'dtype' in layer_cfg:
            layer_cfg['dtype'] = simplify_dtype(layer_cfg['dtype'])
        for key in unknown_layer_keys:
            layer_cfg.pop(key, None)
        return layer_cfg

    def fix_config(obj):
        if isinstance(obj, dict):
            class_name = obj.get('class_name', '')
            if 'config' in obj and isinstance(obj['config'], dict):
                layer_cfg = obj['config']
                if class_name == 'InputLayer':
                    if 'batch_shape' in layer_cfg and 'batch_input_shape' not in layer_cfg:
                        layer_cfg['batch_input_shape'] = layer_cfg.pop('batch_shape')
                fix_layer_config(layer_cfg)

            if 'inbound_nodes' in obj:
                obj['inbound_nodes'] = fix_inbound_nodes(obj['inbound_nodes'])

            return {k: fix_config(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [fix_config(item) for item in obj]
        return obj

    patched = fix_config(config)
    return json.dumps(patched)


def load_model_compat(model_path, custom_objects=None, compile=False):
    """
    Load a Keras .h5 model with robust compatibility handling for Keras 3 -> Keras 2.
    """
    try:
        return load_model(model_path, custom_objects=custom_objects, compile=compile)
    except (TypeError, ValueError) as e:
        err_str = str(e)
        known_issues = [
            'batch_shape',
            'Unrecognized keyword arguments',
            'DTypePolicy',
            'Unknown dtype policy',
            'dtype_policy',
        ]
        if not any(kw in err_str for kw in known_issues):
            raise
        print('   ⚠️  Keras version mismatch detected. Applying compatibility patch...')
        print(f'   📋 Error was: {err_str[:120]}')

    tmp_dir = tempfile.mkdtemp(prefix='keras_compat_')
    tmp_path = os.path.join(tmp_dir, 'model_compat.h5')
    try:
        shutil.copy2(model_path, tmp_path)
        with h5py.File(tmp_path, 'r+') as f:
            if 'model_config' in f.attrs:
                original_cfg = f.attrs['model_config']
                if isinstance(original_cfg, bytes):
                    original_cfg = original_cfg.decode('utf-8')
                patched_cfg = _patch_keras3_model_config(original_cfg)
                f.attrs['model_config'] = patched_cfg
                print('   🔧 Patched model_config successfully')
            else:
                print('   ⚠️  No model_config attribute found in H5 file')

        return load_model(tmp_path, custom_objects=custom_objects, compile=compile)
    finally:
        shutil.rmtree(tmp_dir, ignore_errors=True)


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


def load_all_models():
    """Loads all ML models into global memory on server start."""
    print("\n" + "=" * 60)
    print(" 🏥 Medical AI Diagnostic System - Loading Models")
    print("=" * 60)

    for disease_key, config in MODELS_CONFIG.items():
        print(f"\n📦 Loading {disease_key.replace('_', ' ').title()} model...")
        print(f"   📂 Path: {config['model_path']}")
        print(f"   🏗️  Architecture: {config.get('architecture', 'Unknown')}")
        print(f"   🎯 Type: {config.get('model_type', 'classification')}")

        try:
            model_path = config['model_path']
            if os.path.exists(model_path):
                if config.get('model_type') == 'segmentation':
                    model = load_model_compat(model_path, custom_objects=CUSTOM_OBJECTS, compile=False)
                else:
                    model = load_model_compat(model_path, compile=False)

                models[disease_key] = {
                    'model': model,
                    'config': config
                }

                if 'labels_path' in config and os.path.exists(config['labels_path']):
                    with open(config['labels_path'], 'r', encoding='utf-8') as f:
                        models[disease_key]['labels'] = [line.strip() for line in f.readlines()]
                else:
                    models[disease_key]['labels'] = config.get('labels', ['Unknown'])

                print('   ✅ Loaded successfully!')
                print(f"   📋 Output: {models[disease_key]['labels']}")

                if 'accuracy' in config:
                    print(f"   📊 Accuracy: {config['accuracy'] * 100:.1f}%")
                if 'dice_coefficient' in config:
                    print(f"   📊 Dice Coefficient: {config['dice_coefficient']:.3f}")
            else:
                print(f"   ⚠️  Model file not found: {model_path}")
                models[disease_key] = None
        except Exception as e:
            print(f"   ❌ Error loading model: {e}")
            import traceback
            traceback.print_exc()
            models[disease_key] = None

    print("\n📦 Loading Generative NLP Chatbot model...")
    try:
        nlp_model, nlp_words, nlp_classes, nlp_intents = load_nlp_model()
        nlp_system['model'] = nlp_model
        nlp_system['words'] = nlp_words
        nlp_system['classes'] = nlp_classes
        nlp_system['intents'] = nlp_intents
        print("   ✅ NLP Chatbot loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading NLP Chatbot: {e}")

    print("\n📦 Loading FAISS Knowledge Base index...")
    try:
        index, qa_store = load_index()
        print("   ✅ FAISS Index loaded successfully!")
    except Exception as e:
        print(f"   ❌ Error loading FAISS Index: {e}")

    loaded_count = sum(1 for m in models.values() if m is not None)
    total_count = len(MODELS_CONFIG)

    print("\n" + "=" * 60)
    print(f" 🚀 Server Ready! ({loaded_count}/{total_count} models loaded)")
    print("=" * 60 + "\n")


def preprocess_image_classification(image_path, img_size=224):
    img = load_img(image_path, target_size=(img_size, img_size))
    img_array = img_to_array(img)
    img_array /= 255.0
    img_array = np.expand_dims(img_array, axis=0)
    return img_array


def preprocess_image_segmentation(image_path, img_size=256):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError('Could not read image')

    img = cv2.resize(img, (img_size, img_size))
    img = img.astype(np.float32) / 255.0
    img = np.expand_dims(img, axis=-1)
    img = np.expand_dims(img, axis=0)
    return img


def create_segmentation_overlay(original_image_path, mask, output_path, img_size=256):
    original = cv2.imread(original_image_path)
    original = cv2.resize(original, (img_size, img_size))

    mask_uint8 = (mask * 255).astype(np.uint8)

    overlay = original.copy()
    overlay[:, :, 1] = np.where(mask_uint8 > 127, 255, overlay[:, :, 1])

    result = cv2.addWeighted(original, 0.6, overlay, 0.4, 0)
    contours, _ = cv2.findContours(mask_uint8, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(result, contours, -1, (0, 255, 0), 2)

    cv2.imwrite(output_path, result)
    mask_output_path = output_path.replace('.png', '_mask.png')
    cv2.imwrite(mask_output_path, mask_uint8)

    return output_path, mask_output_path


def make_prediction_classification(disease_type, image_path, model_data):
    model = model_data['model']
    labels = model_data['labels']
    img_size = model_data['config']['img_size']

    input_data = preprocess_image_classification(image_path, img_size)
    predictions = model.predict(input_data, verbose=0)
    
    # Handle binary sigmoid output (single neuron) vs multi-class softmax
    if len(labels) == 2 and predictions[0].shape[0] == 1:
        # Sigmoid output: predictions[0] = [P(positive_class)]
        # Convert to two-class probabilities: [P(negative), P(positive)]
        prob_positive = predictions[0][0]
        predictions = np.array([[1 - prob_positive, prob_positive]])

    predicted_index = np.argmax(predictions[0])
    confidence = float(predictions[0][predicted_index] * 100)
    label = labels[predicted_index]

    all_predictions = [
        {'label': labels[i], 'confidence': float(predictions[0][i] * 100)}
        for i in range(len(labels))
    ]
    all_predictions.sort(key=lambda x: x['confidence'], reverse=True)

    if disease_type == 'brain_tumor':
        is_critical = predicted_index != 2
    else:
        is_critical = predicted_index != 0

    return {
        'label': label,
        'confidence': confidence,
        'all_predictions': all_predictions,
        'is_critical': is_critical,
        'model_type': 'classification'
    }


def make_prediction_segmentation(disease_type, image_path, model_data):
    model = model_data['model']
    img_size = model_data['config']['img_size']

    input_data = preprocess_image_segmentation(image_path, img_size)
    predicted_mask = model.predict(input_data, verbose=0)[0]
    predicted_mask_binary = (predicted_mask > 0.5).astype(np.float32)

    positive_pixels = np.sum(predicted_mask_binary)
    total_pixels = predicted_mask_binary.size
    coverage_percent = (positive_pixels / total_pixels) * 100

    filename = os.path.basename(image_path)
    result_filename = f'result_{filename}'
    result_path = os.path.join(app.config['RESULTS_FOLDER'], result_filename)

    overlay_path, mask_path = create_segmentation_overlay(
        image_path,
        predicted_mask_binary[:, :, 0],
        result_path,
        img_size
    )

    is_detected = coverage_percent > 0.5

    return {
        'label': 'Fetal Head Detected' if is_detected else 'No Clear Fetal Head',
        'confidence': float(coverage_percent * 10),
        'coverage_percent': float(coverage_percent),
        'is_critical': not is_detected,
        'segmentation_overlay': os.path.basename(overlay_path),
        'segmentation_mask': os.path.basename(mask_path),
        'model_type': 'segmentation',
        'metrics': {
            'positive_pixels': int(positive_pixels),
            'total_pixels': int(total_pixels),
            'mean_confidence': float(np.mean(predicted_mask))
        }
    }


def make_prediction(disease_type, image_path):
    if disease_type not in models or models[disease_type] is None:
        return None, 'Model not available'

    model_data = models[disease_type]
    model_type = model_data['config'].get('model_type', 'classification')

    try:
        if model_type == 'segmentation':
            result = make_prediction_segmentation(disease_type, image_path, model_data)
        else:
            result = make_prediction_classification(disease_type, image_path, model_data)
        return result, None
    except Exception as e:
        import traceback
        traceback.print_exc()
        return None, str(e)


# ==================== ROUTES ====================

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/chat')
def chat_page():
    return render_template(
        'chat.html',
        history=get_chat_history(),
        latest_diagnosis=get_latest_diagnosis(),
    )


@app.route('/diagnose/<disease_type>', methods=['GET', 'POST'])
def diagnose(disease_type):
    if disease_type not in MODELS_CONFIG:
        return render_template('error.html', message='Invalid diagnosis type'), 404

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
                filename = f'{disease_type}_{filename}'
                file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
                file.save(file_path)
                image_filename = filename

                result, error = make_prediction(disease_type, file_path)
                if result and not error:
                    store_latest_diagnosis(result)

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
    return render_template('about.html')


@app.route('/api/predict/<disease_type>', methods=['POST'])
def api_predict(disease_type):
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
    filename = f'api_{disease_type}_{filename}'
    file_path = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(file_path)

    result, error = make_prediction(disease_type, file_path)

    if error:
        return jsonify({'error': error}), 500

    store_latest_diagnosis(result)

    return jsonify({
        'success': True,
        'disease_type': disease_type,
        'prediction': result
    })


@app.route('/api/chat/health', methods=['GET'])
def api_chat_health():
    is_ready = nlp_system['model'] is not None
    status = 200 if is_ready else 503
    return jsonify({"ok": is_ready, "status": "online" if is_ready else "offline"}), status


@app.route('/api/chat/reset', methods=['POST'])
def api_chat_reset():
    history_length = reset_chat()
    return jsonify({
        'success': True,
        'message': 'Chat history reset',
        'history_length': history_length,
    })


@app.route('/api/chat', methods=['POST'])
def api_chat():
    payload = request.get_json(silent=True)
    if not payload:
        return jsonify({'success': False, 'error': 'Expected JSON body'}), 400

    message = str(payload.get('message', '')).strip()
    if not message:
        return jsonify({'success': False, 'error': 'message is required'}), 400

    if nlp_system['model'] is None:
        return jsonify({'success': False, 'error': 'NLP Model is not loaded.'}), 503

    try:
        # 1. Try FAISS lookup first
        faiss_results = retrieve(message, top_k=1, threshold=0.35)
        
        if faiss_results:
            top_match = faiss_results[0]
            reply = top_match['answer']
            confidence = top_match['score'] * 100  # Convert to percentage
            intent = f"kb_match ({top_match['category']})"
        else:
            # 2. Fallback to Keras NLP intent generative responses
            result = nlp_chat(
                message,
                nlp_system['model'],
                nlp_system['words'],
                nlp_system['classes'],
                nlp_system['intents']
            )
            reply = result['response']
            confidence = result['confidence']
            intent = result['intent']
            
    except Exception as exc:
        return jsonify({
            'success': False,
            'error': 'Chat processing failed',
            'details': str(exc),
        }), 500

    reply = _ensure_disclaimer(reply)
    append_chat_history('user', message)
    append_chat_history('assistant', reply)

    return jsonify({
        'success': True,
        'reply': reply,
        'intent': intent,
        'confidence': round(confidence, 1),
        'history_length': len(get_chat_history()),
    })


# Run the app
if __name__ == '__main__':
    load_all_models()
    app.run(debug=True, use_reloader=False, host='0.0.0.0', port=5000)
