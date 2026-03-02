"""
Dataset Downloader
==================
Automatically downloads free medical datasets.
Run: python data/download_datasets.py
"""

import os
import json
import pandas as pd
from datasets import load_dataset
from tqdm import tqdm

os.makedirs('data/raw', exist_ok=True)
os.makedirs('knowledge_base', exist_ok=True)

print("=" * 60)
print("  HealthBot Dataset Downloader")
print("=" * 60)

all_qa = []   # Will collect all Q&A pairs here

# ─────────────────────────────────────────────────────────────
# DATASET 1: ruslanmv/ai-medical-chatbot (HuggingFace, FREE)
# 250,000+ medical conversation pairs
# https://huggingface.co/datasets/ruslanmv/ai-medical-chatbot
# ─────────────────────────────────────────────────────────────
print("\n[1/4] Downloading AI Medical Chatbot dataset (HuggingFace)...")
try:
    ds = load_dataset("ruslanmv/ai-medical-chatbot", split="train")
    count = 0
    for row in tqdm(ds, desc="Processing"):
        q = str(row.get("Patient", "") or row.get("question", "")).strip()
        a = str(row.get("Doctor", "") or row.get("answer", "")).strip()
        if len(q) > 10 and len(a) > 20:
            all_qa.append({
                "question": q,
                "answer": a,
                "source": "ai-medical-chatbot",
                "category": "general"
            })
            count += 1
        if count >= 30000:   # cap at 30k for speed; remove cap for full dataset
            break
    print(f"   ✅ Loaded {count} pairs from AI Medical Chatbot")
except Exception as e:
    print(f"   ⚠️  Failed: {e}")

# ─────────────────────────────────────────────────────────────
# DATASET 2: pubmed_qa (HuggingFace, FREE)
# Research-level biomedical QA — covers lungs, brain, fetal
# https://huggingface.co/datasets/pubmed_qa
# ─────────────────────────────────────────────────────────────
print("\n[2/4] Downloading PubMedQA (covers lungs, brain, fetal research)...")
try:
    ds2 = load_dataset("pubmed_qa", "pqa_labeled", split="train")
    count = 0
    for row in tqdm(ds2, desc="Processing"):
        q = str(row.get("question", "")).strip()
        # Combine context + long_answer for a rich answer
        contexts = row.get("context", {})
        ctx_text = ""
        if isinstance(contexts, dict):
            for texts in contexts.get("contexts", []):
                if isinstance(texts, list):
                    ctx_text += " ".join(texts) + " "
                else:
                    ctx_text += str(texts) + " "
        a = str(row.get("long_answer", ctx_text)).strip()
        if len(q) > 10 and len(a) > 30:
            all_qa.append({
                "question": q,
                "answer": a[:800],   # truncate very long answers
                "source": "pubmed_qa",
                "category": "research"
            })
            count += 1
    print(f"   ✅ Loaded {count} pairs from PubMedQA")
except Exception as e:
    print(f"   ⚠️  Failed: {e}")

# ─────────────────────────────────────────────────────────────
# DATASET 3: MedMCQA (HuggingFace, FREE)
# Medical college entrance Q&A — all specialties
# https://huggingface.co/datasets/openlifescienceai/medmcqa
# ─────────────────────────────────────────────────────────────
print("\n[3/4] Downloading MedMCQA (all medical specialties)...")
try:
    ds3 = load_dataset("openlifescienceai/medmcqa", split="train")
    count = 0
    for row in tqdm(ds3, desc="Processing"):
        q = str(row.get("question", "")).strip()
        # Use the correct option as answer
        options = [
            row.get("opa", ""), row.get("opb", ""),
            row.get("opc", ""), row.get("opd", "")
        ]
        correct_idx = row.get("cop", 0)
        exp = str(row.get("exp", "") or "").strip()
        try:
            correct_ans = str(options[correct_idx])
        except (IndexError, TypeError):
            correct_ans = options[0] if options else ""
        
        full_answer = correct_ans
        if exp and len(exp) > 10:
            full_answer += f" — {exp}"

        if len(q) > 10 and len(full_answer) > 10:
            all_qa.append({
                "question": q,
                "answer": full_answer,
                "source": "medmcqa",
                "category": row.get("subject_name", "general")
            })
            count += 1
        if count >= 20000:
            break
    print(f"   ✅ Loaded {count} pairs from MedMCQA")
except Exception as e:
    print(f"   ⚠️  Failed: {e}")

# ─────────────────────────────────────────────────────────────
# DATASET 4: Mental Health FAQ (HuggingFace)
# Brain / mental health specific
# ─────────────────────────────────────────────────────────────
print("\n[4/4] Downloading Mental Health FAQ dataset...")
try:
    ds4 = load_dataset("heliosbrahma/mental_health_chatbot_dataset", split="train")
    count = 0
    for row in tqdm(ds4, desc="Processing"):
        text = str(row.get("text", "")).strip()
        # Format: "Human: ...Assistant: ..."
        if "Human:" in text and "Assistant:" in text:
            parts = text.split("Assistant:")
            q_part = parts[0].replace("Human:", "").strip()
            a_part = parts[1].strip() if len(parts) > 1 else ""
            if len(q_part) > 5 and len(a_part) > 10:
                all_qa.append({
                    "question": q_part,
                    "answer": a_part,
                    "source": "mental_health_faq",
                    "category": "mental_health"
                })
                count += 1
    print(f"   ✅ Loaded {count} pairs from Mental Health FAQ")
except Exception as e:
    print(f"   ⚠️  Failed: {e}")

# ─────────────────────────────────────────────────────────────
# ADD SPECIALIZED Q&A: Lungs, Brain, Fetal (curated)
# ─────────────────────────────────────────────────────────────
print("\n[+] Adding curated specialized Q&A (lungs, brain, fetal)...")
specialized = [
    # LUNGS
    {"question": "What are symptoms of pneumonia?", "answer": "Pneumonia symptoms include cough with phlegm or pus, fever, chills, and difficulty breathing. Symptoms can vary from mild to severe. Older adults may have lower-than-normal body temperature instead of fever.", "category": "lungs"},
    {"question": "What causes COPD?", "answer": "COPD (Chronic Obstructive Pulmonary Disease) is primarily caused by long-term exposure to irritating gases or particulate matter, most often cigarette smoke. Other causes include air pollution, workplace dust, and genetic factors (Alpha-1 antitrypsin deficiency).", "category": "lungs"},
    {"question": "What is asthma and how is it treated?", "answer": "Asthma is a condition where airways narrow, swell and produce extra mucus, making breathing difficult. Treatment includes quick-relief inhalers (bronchodilators), long-term control medications (inhaled corticosteroids), and avoiding triggers.", "category": "lungs"},
    {"question": "What are signs of lung cancer?", "answer": "Lung cancer signs include a cough that doesn't go away, coughing up blood, shortness of breath, chest pain, hoarseness, losing weight without trying, bone pain, and headache. Early-stage lung cancer often produces no symptoms.", "category": "lungs"},
    {"question": "What is pulmonary fibrosis?", "answer": "Pulmonary fibrosis is a lung disease that occurs when lung tissue becomes damaged and scarred. The thickened, stiff tissue makes it harder to breathe. It worsens over time and can be caused by many conditions including prolonged exposure to certain toxins.", "category": "lungs"},
    {"question": "How does tuberculosis affect the lungs?", "answer": "TB bacteria destroy lung tissue forming cavities. Symptoms include persistent cough (sometimes with blood), chest pain, weakness, weight loss, fever, and night sweats. TB is treated with a 6-month course of antibiotics.", "category": "lungs"},
    {"question": "What is pleural effusion?", "answer": "Pleural effusion is the buildup of excess fluid between the layers of the pleura outside the lungs. Causes include heart failure, pneumonia, cancer, and kidney disease. Symptoms include chest pain, shortness of breath, and dry cough.", "category": "lungs"},
    {"question": "What is bronchitis and how long does it last?", "answer": "Bronchitis is inflammation of the bronchial tubes. Acute bronchitis usually lasts 2-3 weeks and is often caused by viruses. Chronic bronchitis lasts at least 3 months per year for 2 consecutive years and is usually associated with smoking.", "category": "lungs"},
    # BRAIN
    {"question": "What are symptoms of a brain stroke?", "answer": "Stroke symptoms: FAST — Face drooping, Arm weakness, Speech difficulty, Time to call emergency. Also sudden severe headache, vision problems, and dizziness. Call emergency services immediately — time is critical with stroke.", "category": "brain"},
    {"question": "What is epilepsy and how is it managed?", "answer": "Epilepsy is a neurological disorder causing recurrent seizures. Management includes anti-seizure medications, dietary therapies (ketogenic diet), nerve stimulation, and surgery in some cases. About 70% of epilepsy patients can control seizures with medication.", "category": "brain"},
    {"question": "What causes Alzheimer's disease?", "answer": "Alzheimer's is caused by abnormal protein deposits (amyloid plaques and tau tangles) disrupting brain cell communication. Risk factors include age, family history, genetics (APOE-e4 gene), head injuries, and cardiovascular risk factors.", "category": "brain"},
    {"question": "What is a brain tumor?", "answer": "A brain tumor is a mass of abnormal cells in the brain. Primary tumors originate in the brain; secondary tumors spread from elsewhere. Symptoms include headaches, seizures, vision problems, difficulty speaking, and personality changes.", "category": "brain"},
    {"question": "What is Parkinson's disease?", "answer": "Parkinson's is a progressive nervous system disorder affecting movement, causing tremors, stiffness, and slowing of movement. It occurs when neurons producing dopamine in the brain die. Treatment includes medications, physical therapy, and surgery.", "category": "brain"},
    {"question": "What is meningitis?", "answer": "Meningitis is inflammation of the membranes (meninges) surrounding the brain and spinal cord, usually caused by infection. Symptoms include sudden high fever, stiff neck, severe headache, nausea, vomiting, and sensitivity to light. It's a medical emergency.", "category": "brain"},
    {"question": "What are migraine headaches?", "answer": "Migraines are intense, recurring headaches often with nausea, vomiting, and sensitivity to light and sound. They can last hours to days. Triggers include hormonal changes, certain foods, stress, and sleep disruption. Treatment includes triptans, preventive medications, and avoiding triggers.", "category": "brain"},
    {"question": "What is multiple sclerosis?", "answer": "MS is a disease where the immune system attacks the protective myelin sheath covering nerves. This disrupts communication between brain and body. Symptoms vary widely including fatigue, vision problems, numbness, and walking difficulties. There is no cure but treatments slow progression.", "category": "brain"},
    # FETAL/PREGNANCY
    {"question": "What are common fetal development issues?", "answer": "Common fetal issues include neural tube defects (spina bifida), Down syndrome, heart defects, cleft lip/palate, and growth restriction. Regular prenatal care, folic acid supplementation, and genetic screening help detect and manage these conditions.", "category": "fetal"},
    {"question": "What is preeclampsia during pregnancy?", "answer": "Preeclampsia is a pregnancy complication with high blood pressure and signs of organ damage (usually kidney). Symptoms include high BP, protein in urine, severe headaches, vision changes, and swelling. It requires prompt medical treatment and can lead to premature delivery.", "category": "fetal"},
    {"question": "What is gestational diabetes?", "answer": "Gestational diabetes is high blood sugar that develops during pregnancy. It affects how cells use sugar and can cause complications for mother and baby. Management includes diet, exercise, blood sugar monitoring, and sometimes insulin. It usually resolves after delivery.", "category": "fetal"},
    {"question": "What causes miscarriage?", "answer": "Most miscarriages are caused by chromosomal abnormalities in the embryo. Other causes include uterine abnormalities, hormonal problems, thyroid disorders, diabetes, autoimmune conditions, and certain infections. Most occur in the first trimester.", "category": "fetal"},
    {"question": "What is fetal growth restriction?", "answer": "Fetal growth restriction (FGR) is when a baby grows slower than expected in the womb. Causes include placenta problems, maternal hypertension, infections, chromosomal abnormalities, and smoking. It requires close monitoring and may require early delivery.", "category": "fetal"},
    {"question": "What are signs of healthy fetal movement?", "answer": "Healthy fetal movement typically starts 16-25 weeks. You should feel 10 movements within 2 hours. A decrease in movement can indicate fetal distress. Count kicks daily after 28 weeks. Contact your doctor immediately if you notice a significant decrease.", "category": "fetal"},
    {"question": "What is placenta previa?", "answer": "Placenta previa is when the placenta covers the cervical opening. It causes painless vaginal bleeding in the third trimester. Depending on severity, delivery may require a C-section. Women with this condition need careful monitoring throughout pregnancy.", "category": "fetal"},
    {"question": "What prenatal tests are important?", "answer": "Key prenatal tests include: blood tests (blood type, anemia, infections), genetic screening (NIPT, nuchal translucency), anatomy ultrasound at 18-20 weeks, glucose screening (24-28 weeks), and Group B strep test (35-37 weeks). Discuss with your OB what's right for you.", "category": "fetal"},
    # HEART
    {"question": "What are symptoms of a heart attack?", "answer": "Heart attack symptoms include chest pain/pressure/tightening, pain radiating to arm/jaw/neck, nausea, sweating, shortness of breath, and lightheadedness. Women may have less typical symptoms. Call emergency services IMMEDIATELY — every minute matters.", "category": "heart"},
    {"question": "What is heart failure?", "answer": "Heart failure means the heart cannot pump blood efficiently. Symptoms include shortness of breath, fatigue, swollen legs/ankles, rapid heartbeat, and reduced ability to exercise. Treatment includes medications, lifestyle changes, and sometimes surgery or devices.", "category": "heart"},
]

for qa in specialized:
    qa["source"] = "curated_specialized"
    all_qa.append(qa)

print(f"   ✅ Added {len(specialized)} specialized Q&A pairs")

# ─────────────────────────────────────────────────────────────
# SAVE COMBINED KNOWLEDGE BASE
# ─────────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  Total Q&A pairs collected: {len(all_qa)}")
print(f"{'='*60}")

with open('knowledge_base/medical_qa.json', 'w', encoding='utf-8') as f:
    json.dump(all_qa, f, ensure_ascii=False, indent=2)

print(f"✅ Saved to knowledge_base/medical_qa.json")
print(f"\nNext step: python model/build_faiss_index.py")

# ─────────────────────────────────────────────────────────────
# MANUAL DATASET INSTRUCTIONS
# ─────────────────────────────────────────────────────────────
print("""
─────────────────────────────────────────────────────────────
ADDITIONAL DATASETS (Manual Download Required):
─────────────────────────────────────────────────────────────

1. MedQuAD (47,457 medical Q&A from NIH):
   → https://github.com/abachaa/MedQuAD
   → git clone https://github.com/abachaa/MedQuAD.git data/raw/medquad
   → Run: python data/add_medquad.py (see below)

2. Kaggle Medical Q&A (16k pairs):
   → https://www.kaggle.com/datasets/pythonafroz/medical-question-and-answer-for-ai-training
   → Download and place CSV in data/raw/medical_qa.csv

3. Kaggle Disease-Symptom Dataset:
   → https://www.kaggle.com/datasets/itachi9604/disease-symptom-description-dataset
   → Download and place in data/raw/

To add Kaggle datasets to knowledge base, run:
   python data/add_kaggle_datasets.py
─────────────────────────────────────────────────────────────
""")
