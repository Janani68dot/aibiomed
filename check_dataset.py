import pandas as pd
import numpy as np
import os  # ← THIS WAS MISSING

print("🔥 Creating REALISTIC dementia dataset (400 samples)...")

# Realistic healthy vs dementia speech patterns
healthy_patterns = [
    "The boy is stealing cookies from the jar while his mother washes dishes at the sink.",
    "Today I went to the market and bought fresh vegetables, fruits, and some bread for dinner.",
    "My family enjoyed a wonderful dinner together last evening with good conversation.",
    "I remember when we first moved to this neighborhood many years ago.",
    "The weather today is beautiful with clear skies and gentle breeze."
]

dementia_patterns = [
    "Uh... the boy he um... cookies... jar? Mother washing... dishes... sink I think.",
    "Today market... um... vegetables... fruits maybe... bread? I don't know.",
    "Family dinner... uh... last night... yesterday... good... or was it?",
    "We moved... neighborhood... many years... when? I forget.",
    "Weather today... beautiful? Clear skies... breeze... um yes."
]

# Generate 400 balanced samples
texts, labels = [], []
for i in range(200):
    texts.append(np.random.choice(healthy_patterns))
    labels.append(0)  # Healthy
    texts.append(np.random.choice(dementia_patterns)) 
    labels.append(1)  # Dementia

df = pd.DataFrame({'text': texts, 'label': labels})

# Create data folder and save
os.makedirs('data', exist_ok=True)
df.to_csv('data/dementia.csv', index=False)

print("✅ Dataset created: data/dementia.csv")
print("Shape:", df.shape)
print("Columns:", df.columns.tolist())
print("\nLabel distribution:")
print(df['label'].value_counts())
print("\nSample healthy (label=0):")
print(df[df['label']==0]['text'].iloc[0])
print("\nSample dementia (label=1):")
print(df[df['label']==1]['text'].iloc[0])
