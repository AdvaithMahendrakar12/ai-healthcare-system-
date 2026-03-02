"""
Add Kaggle / MedQuAD datasets to knowledge base
Run AFTER downloading CSVs manually from Kaggle
"""

import os
import json
import glob
import xml.etree.ElementTree as ET
import pandas as pd

os.makedirs('knowledge_base', exist_ok=True)

# Load existing knowledge base
kb_path = 'knowledge_base/medical_qa.json'
if os.path.exists(kb_path):
    with open(kb_path) as f:
        all_qa = json.load(f)
    print(f"Loaded {len(all_qa)} existing Q&A pairs")
else:
    all_qa = []

added = 0

# ── Kaggle Medical Q&A CSV ─────────────────────────────────────
csv_path = 'data/raw/medical_qa.csv'
if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    # Try common column names
    q_col = next((c for c in df.columns if 'question' in c.lower()), None)
    a_col = next((c for c in df.columns if 'answer' in c.lower()), None)
    if q_col and a_col:
        for _, row in df.iterrows():
            q = str(row[q_col]).strip()
            a = str(row[a_col]).strip()
            if len(q) > 5 and len(a) > 10:
                all_qa.append({"question": q, "answer": a,
                               "source": "kaggle_medical_qa", "category": "general"})
                added += 1
        print(f"✅ Added {added} pairs from Kaggle Medical Q&A CSV")
    else:
        print(f"⚠️  Could not find question/answer columns in {csv_path}. Columns: {list(df.columns)}")

# ── Disease-Symptom CSV ────────────────────────────────────────
sym_path = 'data/raw/symptom_Description.csv'
if os.path.exists(sym_path):
    df = pd.read_csv(sym_path)
    before = len(all_qa)
    for _, row in df.iterrows():
        disease = str(row.get('Disease', '')).strip()
        desc = str(row.get('Description', '')).strip()
        if disease and desc:
            all_qa.append({
                "question": f"What is {disease}? What are symptoms of {disease}?",
                "answer": desc,
                "source": "kaggle_symptom_disease",
                "category": "general"
            })
    print(f"✅ Added {len(all_qa)-before} pairs from Disease-Symptom dataset")

# ── MedQuAD XML files ──────────────────────────────────────────
medquad_dir = 'data/raw/medquad'
if os.path.exists(medquad_dir):
    xml_files = glob.glob(f'{medquad_dir}/**/*.xml', recursive=True)
    before = len(all_qa)
    for xml_file in xml_files:
        try:
            tree = ET.parse(xml_file)
            root = tree.getroot()
            for qa_pair in root.findall('.//QAPair'):
                q = qa_pair.findtext('Question', '').strip()
                a = qa_pair.findtext('Answer', '').strip()
                if q and a:
                    all_qa.append({
                        "question": q, "answer": a[:800],
                        "source": "medquad",
                        "category": root.get('qtype', 'general')
                    })
        except Exception:
            pass
    print(f"✅ Added {len(all_qa)-before} pairs from MedQuAD ({len(xml_files)} XML files)")

# Save updated knowledge base
with open(kb_path, 'w', encoding='utf-8') as f:
    json.dump(all_qa, f, ensure_ascii=False, indent=2)

print(f"\n✅ Total Q&A pairs: {len(all_qa)}")
print(f"✅ Saved to {kb_path}")
print(f"\nNext step: python model/build_faiss_index.py")
