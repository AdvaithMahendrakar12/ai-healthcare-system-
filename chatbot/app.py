"""
Advanced HealthBot Flask Application
=====================================
Run: python app.py
Visit: http://localhost:5000
"""

import os
import sys
from datetime import datetime
from flask import Flask, render_template, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# ─── STARTUP ──────────────────────────────────────────────────

print("\n" + "="*55)
print("  🏥 HealthBot — RAG + BioBERT Medical Chatbot")
print("="*55)

# Pre-load index at startup (not per request)
INDEX_READY = False
try:
    from utils.retriever import load_index
    from utils.chatbot import chat as bot_chat
    load_index()
    INDEX_READY = True
    print("✅ FAISS index loaded and ready!")
except FileNotFoundError as e:
    print(f"⚠️  {e}")
    print("   Run: python data/download_datasets.py")
    print("   Then: python model/build_faiss_index.py")
    from utils.chatbot import chat as bot_chat
except Exception as e:
    print(f"⚠️  Startup error: {e}")
    from utils.chatbot import chat as bot_chat

print("="*55 + "\n")

# ─── ROUTES ───────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html', index_ready=INDEX_READY)


@app.route('/api/chat', methods=['POST'])
def chat_endpoint():
    data = request.get_json()
    if not data or 'message' not in data:
        return jsonify({"error": "No message provided"}), 400

    message = data['message'].strip()
    if not message:
        return jsonify({"error": "Empty message"}), 400
    if len(message) > 600:
        return jsonify({"error": "Message too long"}), 400

    if not INDEX_READY:
        return jsonify({
            "response": (
                "⚠️ The AI model index is not built yet.\n\n"
                "Please run these commands:\n"
                "1. `python data/download_datasets.py`\n"
                "2. `python model/build_faiss_index.py`\n"
                "3. Restart `python app.py`"
            ),
            "confidence": 0,
            "category": "error",
            "timestamp": datetime.now().strftime("%H:%M")
        })

    result = bot_chat(message)
    result["timestamp"] = datetime.now().strftime("%H:%M")
    return jsonify(result)


@app.route('/api/health')
def health_check():
    return jsonify({
        "status": "ok",
        "index_ready": INDEX_READY,
        "model": "BioBERT + FAISS RAG",
        "version": "2.0.0"
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(debug=False, host='0.0.0.0', port=port)
