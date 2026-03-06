import spacy
import textstat
import numpy as np
from collections import Counter

nlp = spacy.load("en_core_web_sm")
FILLERS = set(["uh", "um", "er", "uhm", "umm", "hmm", "well", "like", "actually", "basically", "you know", "right"])

def extract_advanced_features(text):
    if not text or len(text.strip()) == 0:
        return [0.0] * 18

    doc = nlp(text.lower())
    tokens = [t.text for t in doc if t.is_alpha]
    total_tokens = len(tokens)
    words = [t.text for t in doc if t.is_alpha and not t.is_stop]
    vw = len(words)

    # 1. Lexical Diversity (TTR)
    ttr = float(len(set(tokens)) / total_tokens) if total_tokens > 0 else 0.0
    
    # 2. Avg Word Length
    avg_word_len = float(np.mean([len(t) for t in tokens])) if tokens else 0.0
    
    # 3. Sentence Complexity
    sentences = list(doc.sents)
    avg_sent_len = float(total_tokens / len(sentences)) if sentences else 0.0
    
    # 4. POS Ratios
    pos_counts = Counter([t.pos_ for t in doc])
    pron_ratio = float(pos_counts['PRON'] / total_tokens) if total_tokens > 0 else 0.0
    noun_ratio = float(pos_counts['NOUN'] / total_tokens) if total_tokens > 0 else 0.0
    verb_ratio = float(pos_counts['VERB'] / total_tokens) if total_tokens > 0 else 0.0
    adj_ratio = float(pos_counts['ADJ'] / total_tokens) if total_tokens > 0 else 0.0
    
    # 5. Filler Words
    filler_count = sum(1 for t in tokens if t in FILLERS)
    filler_ratio = float(filler_count / total_tokens) if total_tokens > 0 else 0.0

    # 6. Repetitions (Bi-gram)
    bigrams = [tuple(tokens[i:i+2]) for i in range(len(tokens)-1)]
    rep_bigrams = len(bigrams) - len(set(bigrams))
    repetition_ratio = float(rep_bigrams / len(bigrams)) if bigrams else 0.0

    # 7. Readability
    read_ease = float(textstat.flesch_reading_ease(text))
    fog_index = float(textstat.gunning_fog(text))
    
    # 8. Semantic/Structural
    coord_conjs = sum(1 for t in doc if t.dep_ == 'cc')
    coord_ratio = float(coord_conjs / total_tokens) if total_tokens > 0 else 0.0
    
    # 9. Named Entities
    ent_ratio = float(len(doc.ents) / total_tokens) if total_tokens > 0 else 0.0
    
    # 10. Temporal Markers
    temporal_words = set(['yesterday', 'today', 'tomorrow', 'now', 'then', 'before', 'after', 'when'])
    temp_count = sum(1 for t in tokens if t in temporal_words)
    temp_ratio = float(temp_count / total_tokens) if total_tokens > 0 else 0.0

    # 11. Brunet's Index
    brunet = float(np.log(total_tokens) / np.log(len(set(tokens)))) if len(set(tokens)) > 1 else 0.0

    # 12. Short text penalty
    is_short = float(1 if total_tokens < 15 else 0)

    res = [
        ttr, avg_word_len, avg_sent_len, pron_ratio, noun_ratio, 
        verb_ratio, adj_ratio, filler_ratio, repetition_ratio, 
        read_ease, fog_index, coord_ratio, ent_ratio, temp_ratio, 
        brunet, is_short, float(total_tokens), float(vw)
    ]
    return res

test_text = "Boy... uh... cookies... the jar? Mother is... um... washing something. Dishes, I think."
features = extract_advanced_features(test_text)
print(f"Features type: {type(features)}")
print(f"Features length: {len(features)}")
for i, f in enumerate(features):
    print(f"Feature {i}: {f} (type: {type(f)})")

# Check numpy conversion
arr = np.array([features], dtype=float)
print("Numpy conversion successful")
