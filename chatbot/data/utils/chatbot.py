"""
Chatbot NLP Prediction Utility
"""

import json
import pickle
import random
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
import os

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

lemmatizer = WordNetLemmatizer()

# Define base paths relative to this file's directory (chatbot/data/utils/chatbot.py -> chatbot/)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
MODEL_DIR = os.path.join(BASE_DIR, 'model')
DATA_DIR = os.path.join(BASE_DIR, 'data')

def load_model():
    """Load the trained Keras model and supporting files."""
    from tensorflow.keras.models import load_model as keras_load
    model = keras_load(os.path.join(MODEL_DIR, 'chatbot_model.keras'))
    words = pickle.load(open(os.path.join(MODEL_DIR, 'words.pkl'), 'rb'))
    classes = pickle.load(open(os.path.join(MODEL_DIR, 'classes.pkl'), 'rb'))
    with open(os.path.join(DATA_DIR, 'intents.json'), encoding='utf-8') as f:
        intents = json.load(f)
    return model, words, classes, intents


def clean_up_sentence(sentence):
    sentence_words = nltk.word_tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(w.lower()) for w in sentence_words]
    return sentence_words


def bag_of_words(sentence, words):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for s in sentence_words:
        for i, w in enumerate(words):
            if w == s:
                bag[i] = 1
    return np.array(bag)


def predict_class(sentence, model, words, classes, threshold=0.25):
    bow = bag_of_words(sentence, words)
    res = model.predict(np.array([bow]), verbose=0)[0]
    
    # Filter by threshold
    results = [[i, r] for i, r in enumerate(res) if r > threshold]
    results.sort(key=lambda x: x[1], reverse=True)
    
    return [{"intent": classes[r[0]], "probability": str(r[1])} for r in results]


def get_response(intents_list, intents_json):
    if not intents_list:
        return "I'm not sure I understand. Could you rephrase? For emergencies, please call your local emergency number. ⚠️"
    
    tag = intents_list[0]['intent']
    probability = float(intents_list[0]['probability'])
    
    # Low confidence fallback
    if probability < 0.4:
        return "I'm not confident about that. Could you provide more details about your health concern? For medical emergencies, always call emergency services. ⚠️"
    
    for intent in intents_json['intents']:
        if intent['tag'] == tag:
            return random.choice(intent['responses'])
    
    return "I couldn't find relevant information. Please consult a healthcare professional. ⚠️"


def chat(message, model, words, classes, intents):
    """Main chat function — call this from Flask."""
    intents_list = predict_class(message, model, words, classes)
    response = get_response(intents_list, intents)
    
    # Get confidence for UI display
    confidence = float(intents_list[0]['probability']) if intents_list else 0.0
    intent_tag = intents_list[0]['intent'] if intents_list else 'unknown'
    
    return {
        "response": response,
        "intent": intent_tag,
        "confidence": round(confidence * 100, 1)
    }