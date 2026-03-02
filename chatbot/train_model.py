"""
Health Chatbot - IMPROVED NLP Model Training Script
=====================================================
Key improvements for higher accuracy:
  1. Data augmentation (synonym expansion)
  2. Deeper model (512 -> 256 -> 128)
  3. Smaller batch size (better for small datasets)
  4. ReduceLROnPlateau callback
  5. EarlyStopping with restore_best_weights
  6. 500 max epochs (stops early when converged)
  7. Smaller val_split (0.10) to give model more training data
"""

import json
import pickle
import random
import os
import numpy as np
import nltk
from nltk.stem import WordNetLemmatizer
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split

nltk.download('punkt', quiet=True)
nltk.download('wordnet', quiet=True)
nltk.download('punkt_tab', quiet=True)

os.makedirs('model', exist_ok=True)
os.makedirs('data', exist_ok=True)

lemmatizer = WordNetLemmatizer()

# ─── LOAD DATA ────────────────────────────────────────────────────────────────

with open('data/intents.json', 'r', encoding='utf-8') as f:
    intents = json.load(f)

words = []
classes = []
documents = []
ignore_chars = ['?', '!', '.', ',', "'", '"', '-']

# ─── DATA AUGMENTATION ────────────────────────────────────────────────────────
# Expand small dataset by swapping common synonyms
synonyms = {
    "have":  ["got", "experiencing", "suffering from"],
    "feel":  ["am feeling", "feeling", "experience"],
    "pain":  ["ache", "discomfort", "soreness"],
    "bad":   ["severe", "terrible", "intense"],
    "I":     ["I've been", "I am", "I'm"],
    "my":    ["the"],
}

def augment_pattern(pattern):
    augmented = []
    word_list = pattern.split()
    for i, w in enumerate(word_list):
        if w.lower() in synonyms:
            for syn in synonyms[w.lower()][:2]:
                new = word_list.copy()
                new[i] = syn
                augmented.append(' '.join(new))
    return augmented

# Build documents + augmented copies
for intent in intents['intents']:
    for pattern in intent['patterns']:
        wl = nltk.word_tokenize(pattern)
        words.extend(wl)
        documents.append((wl, intent['tag']))
        # augmented versions
        for aug in augment_pattern(pattern):
            aug_wl = nltk.word_tokenize(aug)
            words.extend(aug_wl)
            documents.append((aug_wl, intent['tag']))
    if intent['tag'] not in classes:
        classes.append(intent['tag'])

words = [lemmatizer.lemmatize(w.lower()) for w in words if w not in ignore_chars]
words = sorted(set(words))
classes = sorted(set(classes))

print(f"✅ {len(documents)} patterns after augmentation (was 125 before)")
print(f"✅ {len(words)} unique lemmatized words")
print(f"✅ {len(classes)} intent classes")

pickle.dump(words,   open('model/words.pkl',   'wb'))
pickle.dump(classes, open('model/classes.pkl', 'wb'))

# ─── BAG OF WORDS ─────────────────────────────────────────────────────────────

training = []
output_empty = [0] * len(classes)

for document in documents:
    word_patterns = [lemmatizer.lemmatize(w.lower()) for w in document[0]]
    bag = [1 if w in word_patterns else 0 for w in words]
    output_row = list(output_empty)
    output_row[classes.index(document[1])] = 1
    training.append([bag, output_row])

random.shuffle(training)
training = np.array(training, dtype=object)
train_x = np.array(list(training[:, 0]), dtype=np.float32)
train_y = np.array(list(training[:, 1]), dtype=np.float32)

# Smaller val split → more data for training
X_train, X_val, y_train, y_val = train_test_split(
    train_x, train_y, test_size=0.10, random_state=42
)
print(f"✅ Training: {len(X_train)} | Validation: {len(X_val)}")

# ─── MODEL ────────────────────────────────────────────────────────────────────

model = Sequential([
    Dense(512, input_shape=(len(train_x[0]),), activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(256, activation='relu'),
    BatchNormalization(),
    Dropout(0.3),
    Dense(128, activation='relu'),
    Dropout(0.2),
    Dense(len(train_y[0]), activation='softmax')
])

model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

model.summary()

# ─── CALLBACKS ────────────────────────────────────────────────────────────────

callbacks = [
    EarlyStopping(
        monitor='val_accuracy',
        patience=40,
        restore_best_weights=True,
        min_delta=0.002
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=15,
        min_lr=1e-6,
        verbose=1
    )
]

# ─── TRAIN ────────────────────────────────────────────────────────────────────

print("\n🚀 Training started...\n")
history = model.fit(
    X_train, y_train,
    epochs=500,          # high ceiling; EarlyStopping cuts it short
    batch_size=4,        # small batch = better gradients on small data
    validation_data=(X_val, y_val),
    callbacks=callbacks,
    verbose=1
)

# ─── RESULTS ──────────────────────────────────────────────────────────────────

val_loss,   val_acc   = model.evaluate(X_val,   y_val,   verbose=0)
train_loss, train_acc = model.evaluate(X_train, y_train, verbose=0)

print(f"\n{'='*50}")
print(f"  Train Accuracy:      {train_acc*100:.2f}%")
print(f"  Validation Accuracy: {val_acc*100:.2f}%")
print(f"{'='*50}")

if val_acc < 0.75:
    print("\n⚠️  Still low. BEST FIX: add more patterns to intents.json")
    print("   Each intent needs 15-25 varied patterns.")
elif val_acc < 0.90:
    print("\n🟡 Good! Add more patterns per intent to push past 90%.")
else:
    print("\n🎉 Excellent accuracy!")

model.save('model/chatbot_model.keras')
print("\n✅ Model saved → model/chatbot_model.keras")
print("✅ Run: python app.py")