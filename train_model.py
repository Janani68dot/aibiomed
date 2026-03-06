import pandas as pd
import spacy
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    doc = nlp(text.lower())
    words = [t.text for t in doc if t.is_alpha and not t.is_stop]
    vw = len(words)
    
    if vw < 5:
        return [0.5, 10.0, 0.1, 0.2]
    
    uw = len(set(words))
    vocab_rich = uw / vw
    sent_len = np.mean([len(sent) for sent in doc.sents]) if doc.sents else 10
    pron = len([t for t in doc if t.pos_ == 'PRON'])
    pron_ratio = pron / vw
    rep = 1 - vocab_rich
    
    return [vocab_rich, sent_len, pron_ratio, rep]

# MORE DIVERSE synthetic data
healthy_texts = [
    "The boy steals cookies while mother washes dishes oblivious to the chaos.",
    "I visited the vibrant market buying fresh produce and artisan bread today.",
    "Family dinner was delightful with engaging conversations and laughter.",
    "Planning my schedule meticulously for the important meeting tomorrow.",
    "Reading complex novels expands my vocabulary and sharpens my mind."
]

dementia_texts = [
    "Boy... cookies... uh... mother wash dishes... um... what else?",
    "Market today... vegetables... fruits? Don't remember exactly.",
    "Dinner family... good... when was it? Yesterday maybe.",
    "Schedule... meeting... tomorrow? I think so... not sure.",
    "Book reading... words... difficult... uh... forgot the story."
]

# Generate DIVERSE dataset
np.random.seed(42)
texts, labels = [], []
for _ in range(300):  # More samples
    texts.extend([np.random.choice(healthy_texts) for _ in range(2)])
    labels.extend([0, 0])
    texts.extend([np.random.choice(dementia_texts) for _ in range(2)])
    labels.extend([1, 1])

df = pd.DataFrame({'text': texts, 'label': labels})
df['features'] = df['text'].apply(extract_features)
X = np.array(df['features'].tolist())
y = df['label']

# Train
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
model = RandomForestClassifier(n_estimators=200, random_state=42, class_weight='balanced')
model.fit(X_train, y_train)

print("✅ Model retrained with diverse data!")
print("Test Accuracy:", accuracy_score(y_test, model.predict(X_test)))
print("\nClassification Report:")
print(classification_report(y_test, model.predict(X_test)))

os.makedirs('model', exist_ok=True)
joblib.dump(model, 'model/alzheimer_model.pkl')
