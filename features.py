import spacy
import re
import numpy as np

# Load NLP model once at module level or use a global instance
nlp = spacy.load("en_core_web_sm")

def extract_features(text):
    """
    Extracts clinical biomarkers for dementia detection.
    Optimized for high accuracy on disfluency-heavy speech.
    """
    if not text or len(text.strip()) < 5:
        # Default baseline for insufficient data
        return [0.5, 0.0, 10.0, 0.1, 0.0]

    text_clean = text.lower()
    doc = nlp(text_clean)
    
    # 1. Linguistic Metrics (Vocabulary Flow)
    words = [t.text for t in doc if t.is_alpha]
    total_words = len(words)
    if total_words < 1: 
        return [0.0, 1.0, 0.0, 0.5, 0.0]
    
    unique_words = len(set(words))
    vocab_rich = unique_words / total_words
    
    # 2. Disfluency Metrics (The "uh/um" Biomarker)
    # Heavily penalize specific markers mentioned by the user
    fillers = len(re.findall(r'\b(uh|um|er|uhm|umm|hesitation)\b', text_clean))
    hesitations = text_clean.count("...")
    disfluency_score = (fillers + hesitations) / max(total_words, 1)
    
    # 3. Syntactic Complexity (Logopenia Detection)
    sentences = [sent for sent in doc.sents if len(sent) > 1]
    avg_sent_len = np.mean([len(sent) for sent in sentences]) if sentences else 6
    
    # 4. Semantic Poverty (Pronoun-to-Noun ratio / Semantic Anomia)
    pronouns = len([t for t in doc if t.pos_ == 'PRON'])
    nouns = len([t for t in doc if t.pos_ == 'NOUN'])
    pron_ratio = pronouns / max(nouns + pronouns, 1)
    
    # 5. Repetition Index (Palilalia)
    reps = 0
    if len(words) > 1:
        for i in range(len(words)-1):
            if words[i] == words[i+1]: reps += 1
    rep_index = reps / max(total_words, 1)

    # Return a fixed-length feature vector for the model
    # [VocabRich, DisfluencyVal, SentLen, PronRatio, RepetitionVal]
    return [
        vocab_rich, 
        min(disfluency_score, 1.0), 
        min(avg_sent_len / 20.0, 1.0), 
        pron_ratio, 
        min(rep_index, 1.0)
    ]
