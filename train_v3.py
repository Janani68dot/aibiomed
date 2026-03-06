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
        
        # Aggressive filler detection
        filler_count = sum(1 for t in tokens if t in FILLERS)
        # Check for injected markers too
        filler_count += text.count("uhm") + text.count("...uhm...")
        
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
        is_short = 1.0 if total_tokens < 6 else 0.0 
        return [ttr, avg_word_len, avg_sent_len, pron_ratio, noun_ratio, verb_ratio, adj_ratio, filler_ratio, repetition_ratio, read_ease, fog_index, coord_ratio, ent_ratio, temp_ratio, brunet, is_short, float(total_tokens), float(vw)]
    except:
        return [0.0] * 18

def generate_v3_data(n=10000):
    healthy = [
        "The quick brown fox jumps over the lazy dog.",
        "I am going to the university to attend my morning lectures.",
        "The weather report says it will be sunny all through the weekend.",
        "I clearly remember our trip to the mountains last December.",
        "A healthy lifestyle involves regular physical activity and proper nutrition.",
        "She is reading a fascinating book about history in the library.",
        "The garden is full of vibrant flowers and tall green trees.",
        "We are planning a small gathering for my friend's birthday tonight.",
        "Today I am going to college and I have many classes to attend.",
        "It is a beautiful morning to walk in the park.",
        "Please tell me the time of the next flight to London.",
        "I enjoy cooking dinner for my family on Sunday evenings.",
        "The scientist published a groundbreaking paper on climate change.",
        "They are building a new community center in the neighborhood."
    ]
    dementia = [
        "uhm... the... uhm... fox... jumps... over... uhm... dog?",
        "Going... uhm... school? No... uhm... college. Many... uhm... things to do.",
        "Sun... uh... hot? Today... uhm... weather... I think.",
        "Mountains... uhm... cold. Last... uhm... when was it?",
        "Walking... uhm... good. Eating... uhm... also good. For... uhm... health.",
        "Book... uhm... read. About... uhm... long ago.",
        "Flowers... uhm... red. Trees... uhm... big. In... uhm... outside.",
        "Party... uhm... tonight. Friends... uhm... come.",
        "College... uhm... today. Classes... uhm... yeah.",
        "Morning... uhm... walk. Park... uhm... nice.",
        "Time... uhm... clock? Flight... uhm... gone.",
        "Dinner... uhm... cook. Family... uhm... eat.",
        "Paper... uhm... news. Climate... uhm... change... uhm... bad.",
        "Building... uhm... house. Center... uhm... where?"
    ]
    texts, labels = [], []
    for _ in range(n // 2):
        texts.append(np.random.choice(healthy))
        labels.append(0)
        texts.append(np.random.choice(dementia))
        labels.append(1)
    return pd.DataFrame({'text': texts, 'label': labels})

print("📊 Generating 10,000 V3 Ultra-Precision samples...")
df = generate_v3_data(10000)
X = np.array(df['text'].apply(extract_advanced_features).tolist(), dtype=float)
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print("🚀 Training V3 Stacking Model (100% Precision Target)...")
stack_model = StackingClassifier(
    estimators=[
        ('xgb', XGBClassifier(n_estimators=200, learning_rate=0.01, max_depth=6, n_jobs=1)),
        ('lgbm', LGBMClassifier(n_estimators=200, learning_rate=0.01, num_leaves=31, verbose=-1, n_jobs=1))
    ], 
    final_estimator=LogisticRegression(),
    cv=10,
    n_jobs=1
)
stack_model.fit(X_train, y_train)

y_pred = stack_model.predict(X_test)
print(f"✅ Holdout F1-Score: {f1_score(y_test, y_pred):.4f}")
print(classification_report(y_test, y_pred))

os.makedirs('model', exist_ok=True)
joblib.dump(stack_model, 'model/advanced_alzheimer_model.pkl')
features_list = ['ttr', 'avg_word_len', 'avg_sent_len', 'pron_ratio', 'noun_ratio', 'verb_ratio', 'adj_ratio', 'filler_ratio', 'repetition_ratio', 'read_ease', 'fog_index', 'coord_ratio', 'ent_ratio', 'temp_ratio', 'brunet', 'is_short', 'total_tokens', 'vw']
joblib.dump(features_list, 'model/features_list.pkl')
print("✨ V3 Ultra-Precision Model Saved.")
