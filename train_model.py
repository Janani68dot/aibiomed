import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import SGDClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
import joblib
import os
import random
from features import extract_features

print("🚀 Starting V3+ Self-Learning Training System...")

# 🧪 HIGH-FIDELITY SYNTHETIC PATTERNS
healthy_templates = [
    "The boy is stealing cookies from the jar while his mother washes dishes at the sink.",
    "Today I went to the market and bought fresh vegetables, fruits, and some bread for dinner.",
    "My family enjoyed a wonderful dinner together last evening with good conversation.",
    "I remember when we first moved to this neighborhood many years ago.",
    "The weather today is beautiful with clear skies and a gentle breeze.",
    "I am planning to visit the museum this weekend to see the new art exhibit.",
    "Listening to classical music helps me relax and focus on my work.",
    "We should take a walk in the park while the sun is still out.",
    "The local library has a great collection of historical biographies.",
    "It is important to maintain a healthy lifestyle with regular exercise and balanced meals."
]

dementia_templates = [
    "Uh... the boy he um... cookies... jar? Mother washing... dishes... sink I think.",
    "Today market... um... vegetables... fruits maybe... bread? I don't know.",
    "Family dinner... uh... last night... yesterday... good... or was it?",
    "We moved... neighborhood... many years... when? I forget.",
    "Weather today... beautiful? Clear skies... breeze... um yes.",
    "Uh... museum... art... this weekend? I... um... think so.",
    "Music... uh... helps... focus? I... forgot... what was I saying?",
    "Walk... park... sun... um... where is the park again?",
    "Library... uh... books... history... um... what's that called?",
    "Health... uh... exercise... um... food? I... don't... remember."
]

def generate_variant(text, is_dementia):
    words = text.split()
    if is_dementia:
        fillers = ["uh", "um", "uhh", "umm", "...", "...uh..."]
        for _ in range(random.randint(1, 4)):
            pos = random.randint(0, len(words))
            words.insert(pos, random.choice(fillers))
        if len(words) > 5:
            for _ in range(random.randint(1, 3)):
                words.pop(random.randint(0, len(words)-1))
    return " ".join(words)

print("📊 Generating 10,000 baseline samples...")
texts, labels = [], []
for _ in range(5000):
    base_h = random.choice(healthy_templates)
    texts.append(generate_variant(base_h, False))
    labels.append(0)
    base_d = random.choice(dementia_templates)
    texts.append(generate_variant(base_d, True))
    labels.append(1)

df = pd.DataFrame({'text': texts, 'label': labels})
print("🧮 Extracting features...")
df['features'] = df['text'].apply(extract_features)
X = np.array(df['features'].tolist())
y = df['label']

# SGDClassifier requires scaling for best performance in online learning
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 🚆 Initial Training with SGD (Online capable)
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

model = SGDClassifier(
    loss='log_loss',  # Log loss for probabilistic output
    max_iter=1000, 
    tol=1e-3, 
    random_state=42, 
    class_weight='balanced'
)

# Initial fit
model.fit(X_train, y_train)

# 📈 Results
y_pred = model.predict(X_test)
acc = accuracy_score(y_test, y_pred)
print(f"\n✅ Initial Baseline Accuracy: {acc:.4%}")

# 💾 Save Model and Scaler
os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/alzheimer_model.pkl')
joblib.dump(scaler, 'model/scaler.pkl')
print("💎 Self-Learning Model and Scaler saved.")
