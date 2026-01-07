"""
💬 MEDICAL CHATBOT TRAINING
Run in VS Code: Right-click → Run Python File in Terminal
"""

import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np
import pickle
import os

print("="*60)
print("💬 MEDICAL CHATBOT - TRAINING SCRIPT")
print("="*60)

# Load FAQ dataset
faq_path = 'datasets/chatbot_data/medical_faq.csv'

if not os.path.exists(faq_path):
    print(f"❌ ERROR: FAQ dataset not found at {faq_path}")
    print("Run: python datasets/chatbot_data/create_faq.py")
    exit(1)

print(f"📥 Loading FAQ data from {faq_path}...")
faq_data = pd.read_csv(faq_path)
print(f"✅ Loaded {len(faq_data)} Q&A pairs")

# Load Sentence-BERT model
print("\n📦 Loading Sentence-BERT model...")
model = SentenceTransformer('all-MiniLM-L6-v2')
print("✅ Model loaded!")

# Encode all questions
print("\n🔄 Encoding questions...")
question_embeddings = model.encode(faq_data['question'].tolist(), show_progress_bar=True)
print(f"✅ Encoded {len(question_embeddings)} questions")

# Save chatbot data
print("\n💾 Saving chatbot data...")
os.makedirs('models', exist_ok=True)

chatbot_data = {
    'embeddings': question_embeddings,
    'faq_data': faq_data,
    'model_name': 'all-MiniLM-L6-v2'
}

with open('models/chatbot_data.pkl', 'wb') as f:
    pickle.dump(chatbot_data, f)

print("✅ Chatbot data saved to: models/chatbot_data.pkl")

# Test chatbot function
def get_answer(user_query):
    query_embedding = model.encode([user_query])
    similarities = cosine_similarity(query_embedding, question_embeddings)[0]
    best_idx = np.argmax(similarities)
    confidence = similarities[best_idx]
    
    if confidence > 0.5:
        return faq_data.iloc[best_idx]['answer'], confidence
    else:
        return "I'm not sure about that. Please consult a medical professional.", 0.0

# Test queries
print("\n🧪 Testing chatbot:")
print("="*60)

test_queries = [
    "What causes pneumonia?",
    "Tell me about brain tumors",
    "How do I prevent pneumonia?",
    "What is MRI?"
]

for query in test_queries:
    answer, confidence = get_answer(query)
    print(f"\n❓ Query: {query}")
    print(f"💬 Answer: {answer}")
    print(f"📊 Confidence: {confidence*100:.1f}%")
    print("-"*60)

print("\n" + "="*60)
print("🎉 MEDICAL CHATBOT TRAINING COMPLETE!")
print("="*60)
print("✅ All models trained! Ready to run Streamlit app!")