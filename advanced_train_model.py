import pandas as pd
import spacy
import numpy as np
import re
import os
import joblib
import textstat
from collections import Counter
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.metrics import accuracy_score, classification_report, f1_score

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
        if total_tokens == 0:
            return [0.0] * 18

        words = [t.text for t in doc if t.is_alpha and not t.is_stop]
        vw = len(words)

        # 1. Lexical Diversity (TTR)
        ttr = float(len(set(tokens)) / total_tokens)
        
        # 2. Avg Word Length
        avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
        
        # 3. Sentence Complexity
        sentences = list(doc.sents)
        avg_sent_len = float(total_tokens / len(sentences)) if sentences else 0.0
        
        # 4. POS Ratios
        pos_counts = Counter([t.pos_ for t in doc])
        pron_ratio = float(pos_counts['PRON'] / total_tokens)
        noun_ratio = float(pos_counts['NOUN'] / total_tokens)
        verb_ratio = float(pos_counts['VERB'] / total_tokens)
        adj_ratio = float(pos_counts['ADJ'] / total_tokens)
        
        # 5. Filler Words
        filler_count = sum(1 for t in tokens if t in FILLERS)
        filler_ratio = float(filler_count / total_tokens)

        # 6. Repetitions (Bi-gram)
        bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
        rep_bigrams = len(bigrams) - len(set(bigrams))
        repetition_ratio = float(rep_bigrams / len(bigrams)) if bigrams else 0.0

        # 7. Readability
        read_ease = float(textstat.flesch_reading_ease(text))
        fog_index = float(textstat.gunning_fog(text))
        
        # 8. Semantic/Structural
        coord_conjs = sum(1 for t in doc if t.dep_ == 'cc')
        coord_ratio = float(coord_conjs / total_tokens)
        
        # 9. Named Entities
        ent_ratio = float(len(doc.ents) / total_tokens)
        
        # 10. Temporal Markers
        temporal_words = set(['yesterday', 'today', 'tomorrow', 'now', 'then', 'before', 'after', 'when'])
        temp_count = sum(1 for t in tokens if t in temporal_words)
        temp_ratio = float(temp_count / total_tokens)

        # 11. Brunet's Index
        unique_tokens = len(set(tokens))
        if total_tokens > 0 and unique_tokens > 1:
            brunet = float(np.log(total_tokens) / np.log(unique_tokens))
        else:
            brunet = 0.0

        # 12. Short text penalty
        is_short = 1.0 if total_tokens < 15 else 0.0

        return [
            ttr, avg_word_len, avg_sent_len, pron_ratio, noun_ratio, 
            verb_ratio, adj_ratio, filler_ratio, repetition_ratio, 
            read_ease, fog_index, coord_ratio, ent_ratio, temp_ratio, 
            brunet, is_short, float(total_tokens), float(vw)
        ]
    except Exception as e:
        print(f"Error extracting features for text: '{text[:50]}...' -> {e}")
        return [0.0] * 18

def generate_augmented_data(n=2000):
    healthy_templates = [
        "The young boy is reaching for the cookie jar while his mother is busy washing the dishes.",
        "I remember clearly visiting the local museum last summer with my entire family.",
        "Regular exercise and a balanced diet are essential for maintaining good cognitive health.",
        "The intricate details of the painting captured the essence of the morning sunrise perfectly.",
        "Yesterday was a productive day as I managed to complete all my pending tasks efficiently.",
        "The scientific research published last week provides new insights into neural plasticity.",
        "Navigating through the bustling city streets requires focus and a good sense of direction.",
        "Modern technology has significantly changed the way we communicate and share information.",
        "The complex plot of the novel kept me engaged until the very last chapter.",
        "Speaking clearly and concisely helps in delivering an effective presentation to the board."
    ]

    dementia_templates = [
        "Boy... uh... cookies... the jar? Mother is... um... washing something. Dishes, I think.",
        "Museum... last... when was it? I saw some... uh... old things. I don't remember.",
        "Eating... good food. Exercise. It is... uh... important for... you know.",
        "The, uh, picture? It has... colors. Morning... sun... um... very nice.",
        "Yesterday? I did... things. Tasks? Yes, tasks. I think I... um... finished.",
        "Science... papers? New things about... uh... brains. I read it... somewhere.",
        "Walking... in the city. Many people. Uh... I got... um... a bit confused.",
        "Phones... computers. They, uh, change everything. We... um... talk differently.",
        "The book... was... uh... long. Stories about... um... people. I forgot the end.",
        "Talking... um... is hard. I... uh... want to say... the words... um... are missing."
    ]

    texts, labels = [], []
    for _ in range(n // 2):
        texts.append(np.random.choice(healthy_templates))
        labels.append(0)
        
        base = np.random.choice(dementia_templates)
        words = base.split()
        if len(words) > 5:
            idx = np.random.randint(0, len(words))
            words.insert(idx, np.random.choice(list(FILLERS)))
        texts.append(" ".join(words))
        labels.append(1)
    return pd.DataFrame({'text': texts, 'label': labels})

print("📊 Generating 2200 samples...")
df = generate_augmented_data(2200)

print("🧪 Extracting 18 advanced features...")
X_list = df['text'].apply(extract_advanced_features).tolist()
X = np.array(X_list, dtype=float)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🚀 Building Stacking Classifier (XGBoost + LightGBM)...")
base_models = [
    ('xgb', XGBClassifier(n_estimators=150, learning_rate=0.03, max_depth=6, random_state=42)),
    ('lgbm', LGBMClassifier(n_estimators=150, learning_rate=0.03, num_leaves=31, random_state=42, verbose=-1))
]
stack_model = StackingClassifier(
    estimators=base_models, 
    final_estimator=LogisticRegression(),
    cv=5
)

stack_model.fit(X_train, y_train)
y_pred = stack_model.predict(X_test)
print(f"✅ Holdout F1-Score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

os.makedirs('model', exist_ok=True)
joblib.dump(stack_model, 'model/advanced_alzheimer_model.pkl')
features_list = [
    "ttr", "avg_word_len", "avg_sent_len", "pron_ratio", "noun_ratio", 
    "verb_ratio", "adj_ratio", "filler_ratio", "repetition_ratio", 
    "read_ease", "fog_index", "coord_ratio", "ent_ratio", "temp_ratio", 
    "brunet", "is_short", "total_tokens", "vw"
]
joblib.dump(features_list, 'model/features_list.pkl')
print("✨ Ultimate Model Saved.")
