"""
Build FAISS Vector Index
=========================
Uses BioBERT (S-BioBert) to encode all Q&A pairs into embeddings,
then indexes them in FAISS for fast similarity search.

Run: python model/build_faiss_index.py
Time: ~5-20 min depending on dataset size and hardware
"""

import os
import json
import pickle
import numpy as np
from tqdm import tqdm
from sentence_transformers import SentenceTransformer
import faiss

os.makedirs('model', exist_ok=True)

# ─── CONFIG ───────────────────────────────────────────────────
MODEL_NAME = "pritamdeka/S-BioBert-snli-multinli-stsb"
# Alternative if slow: "all-MiniLM-L6-v2" (faster, slightly less medical)
# Best alternative: "NLP4Science/biosses-sentence-transformers"

BATCH_SIZE = 64
KB_PATH = "knowledge_base/medical_qa.json"
INDEX_PATH = "model/faiss_index.bin"
STORE_PATH = "model/qa_store.pkl"

# ─── LOAD DATA ────────────────────────────────────────────────

if not os.path.exists(KB_PATH):
    print(f"❌ {KB_PATH} not found. Run: python data/download_datasets.py first")
    exit(1)

with open(KB_PATH, encoding='utf-8') as f:
    qa_data = json.load(f)

print(f"✅ Loaded {len(qa_data)} Q&A pairs from knowledge base")

# Remove duplicates by question
seen = set()
unique_qa = []
for item in qa_data:
    q = item['question'].strip().lower()
    if q not in seen and len(q) > 5:
        seen.add(q)
        unique_qa.append(item)

print(f"✅ {len(unique_qa)} unique Q&A pairs after deduplication")

# ─── LOAD BioBERT MODEL ───────────────────────────────────────

print(f"\n⏳ Loading BioBERT model: {MODEL_NAME}")
print("   (First run downloads ~420MB — subsequent runs use cache)")
embedder = SentenceTransformer(MODEL_NAME)
print("✅ BioBERT loaded!")

# ─── ENCODE QUESTIONS ─────────────────────────────────────────

print(f"\n⏳ Encoding {len(unique_qa)} questions with BioBERT...")
print("   This takes 5-20 minutes. Grab a coffee ☕")

questions = [item['question'] for item in unique_qa]

embeddings = embedder.encode(
    questions,
    batch_size=BATCH_SIZE,
    show_progress_bar=True,
    convert_to_numpy=True,
    normalize_embeddings=True   # L2-normalize for cosine similarity with dot product
)

print(f"✅ Encoded! Shape: {embeddings.shape}")

# ─── BUILD FAISS INDEX ────────────────────────────────────────

dim = embeddings.shape[1]   # 768 for BioBERT

# IndexFlatIP = Inner Product (cosine similarity with normalized vectors)
# For large datasets (>100k), use IndexIVFFlat for speed
if len(unique_qa) > 100_000:
    print("\n⏳ Building IVF FAISS index (large dataset mode)...")
    nlist = 200
    quantizer = faiss.IndexFlatIP(dim)
    index = faiss.IndexIVFFlat(quantizer, dim, nlist, faiss.METRIC_INNER_PRODUCT)
    index.train(embeddings)
    index.add(embeddings)
    index.nprobe = 20   # search 20 clusters per query
else:
    print("\n⏳ Building Flat FAISS index...")
    index = faiss.IndexFlatIP(dim)   # exact search
    index.add(embeddings)

print(f"✅ FAISS index built! Total vectors: {index.ntotal}")

# ─── SAVE ─────────────────────────────────────────────────────

faiss.write_index(index, INDEX_PATH)
print(f"✅ FAISS index saved → {INDEX_PATH}")

with open(STORE_PATH, 'wb') as f:
    pickle.dump(unique_qa, f)
print(f"✅ Q&A store saved → {STORE_PATH}")

print(f"\n{'='*50}")
print(f"🎉 Index complete!")
print(f"   Vectors indexed: {index.ntotal}")
print(f"   Embedding dim:   {dim}")
print(f"   Index type:      {'IVF' if len(unique_qa) > 100_000 else 'Flat'}")
print(f"{'='*50}")
print(f"\nNext step: python app.py")
