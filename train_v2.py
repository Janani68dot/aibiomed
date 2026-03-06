import pandas as pd
import spacy
import numpy as np
import os
import joblib
import textstat
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import f1_score, classification_report

# Load SpaCy
nlp = spacy.load("en_core_web_sm")
FILLERS = set(["uh", "um", "er", "uhm", "umm", "hmm", "well", "like", "actually", "basically", "you know", "right"])

def extract_advanced_features(text):
    try:
        if not text or not isinstance(text, str) or len(text.strip()) == 0:
            return [0.0] * 18
        doc = nlp(text.lower())
        tokens = [t.text for t in doc if t.is_alpha]
        total_tokens = len(tokens)
        if total_tokens == 0: return [0.0] * 18
        words = [t.text for t in doc if t.is_alpha and not t.is_stop]
        vw = len(words)
        ttr = float(len(set(tokens)) / total_tokens)
        avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
        sents = list(doc.sents)
        avg_sent_len = float(total_tokens / len(sents)) if sents else 0.0
        pos_counts = Counter([t.pos_ for t in doc])
        pron_ratio = float(pos_counts['PRON'] / total_tokens)
        noun_ratio = float(pos_counts['NOUN'] / total_tokens)
        verb_ratio = float(pos_counts['VERB'] / total_tokens)
        adj_ratio = float(pos_counts['ADJ'] / total_tokens)
        filler_count = sum(1 for t in tokens if t in FILLERS)
        filler_ratio = float(filler_count / total_tokens)
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        rep_bigrams = len(bigrams) - len(set(bigrams))
        repetition_ratio = float(rep_bigrams / len(bigrams)) if bigrams else 0.0
        read_ease = float(textstat.flesch_reading_ease(text))
        fog_index = float(textstat.gunning_fog(text))
        coord_conjs = sum(1 for t in doc if t.dep_ == 'cc')
        coord_ratio = float(coord_conjs / total_tokens)
        ent_ratio = float(len(doc.ents) / total_tokens)
        temporal_words = set(['yesterday', 'today', 'tomorrow', 'now', 'then', 'before', 'after', 'when'])
        temp_count = sum(1 for t in tokens if t in temporal_words)
        temp_ratio = float(temp_count / total_tokens)
        unique_tokens = len(set(tokens))
        brunet = float(np.log(total_tokens) / np.log(unique_tokens)) if unique_tokens > 1 else 0.0
        is_short = 1.0 if total_tokens < 8 else 0.0 # Reduced threshold
        return [ttr, avg_word_len, avg_sent_len, pron_ratio, noun_ratio, verb_ratio, adj_ratio, filler_ratio, repetition_ratio, read_ease, fog_index, coord_ratio, ent_ratio, temp_ratio, brunet, is_short, float(total_tokens), float(vw)]
    except:
        return [0.0] * 18

def generate_balanced_data(n=2500):
    healthy_templates = [
        "The young boy is reaching for the cookie jar while his mother is busy washing the dishes.",
        "I remember clearly visiting the local museum last summer with my entire family.",
        "Regular exercise and a balanced diet are essential for maintaining good cognitive health.",
        "Today I am going to college and I have many classes to attend.", # New medium
        "Hello, how are you today?", # New short
        "I am going to the store.", # New short
        "The weather is very nice outside.", # New short
        "I had a great breakfast this morning.", # New short
        "It is a beautiful morning to walk.",
        "Please tell me the time."
    ]
    dementia_templates = [
        "Boy... uh... cookies... the jar? Mother is... um... washing something. Dishes, I think.",
        "Museum... last... when was it? I saw some... uh... old things. I don't remember.",
        "Eating... good food. Exercise. It is... uh... important for... you know.",
        "Today... uh... college? I think... um... classes.", # dementia variant of short
        "Hello... um... who? How are... uh... you?",
        "Store... uh... going. I need... um... milk?",
        "Weather... is... uh... it has sun.",
        "Eat... um... bread. Good."
    ]
    texts, labels = [], []
    for _ in range(n // 2):
        texts.append(np.random.choice(healthy_templates))
        labels.append(0)
        texts.append(np.random.choice(dementia_templates))
        labels.append(1)
    return pd.DataFrame({'text': texts, 'label': labels})

print("📊 Generating 2500 balanced samples...")
df = generate_balanced_data(2500)
X = np.array(df['text'].apply(extract_advanced_features).tolist(), dtype=float)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🚀 Training Calibrated Stacking Model (Single-thread for stability)...")
stack_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=100, learning_rate=0.05, max_depth=4, n_jobs=1)),
        ('lgbm', LGBMClassifier(n_estimators=100, learning_rate=0.05, num_leaves=15, verbose=-1, n_jobs=1))
    ], 
    final_estimator=LogisticRegression(),
    cv=5,
    n_jobs=1 # Critical for Windows stability
)
stack_model.fit(X_train, y_train)

y_pred = stack_model.predict(X_test)
print(f"✅ Holdout F1-Score: {f1_score(y_test, y_pred):.4f}")

os.makedirs('model', exist_ok=True)
joblib.dump(stack_model, 'model/advanced_alzheimer_model.pkl')
features_list = ['ttr', 'avg_word_len', 'avg_sent_len', 'pron_ratio', 'noun_ratio', 'verb_ratio', 'adj_ratio', 'filler_ratio', 'repetition_ratio', 'read_ease', 'fog_index', 'coord_ratio', 'ent_ratio', 'temp_ratio', 'brunet', 'is_short', 'total_tokens', 'vw']
joblib.dump(features_list, 'model/features_list.pkl')
print("✨ Calibrated Model and Features List Saved.")
