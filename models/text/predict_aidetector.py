from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Dict, List
import numpy as np
import joblib
from scipy.sparse import hstack

AI_MARKERS = [
    r"\bdelve\b", r"\btestament\b", r"\btapestry\b", r"\bpivotal\b", r"\bbeacon\b",
    r"\bfostering\b", r"\bseamlessly\b", r"\bunderscores\b", r"\bmoreover\b",
    r"\bfurthermore\b", r"\bin summary\b", r"\bin conclusion\b", r"\bit is crucial\b",
    r"\bit is important to\b", r"\bplays a crucial role\b", r"\bplays a pivotal role\b",
    r"\bharnessing\b", r"\belevate\b", r"\brealm\b", r"\bever-evolving\b",
    r"\brich tapestry\b", r"\bholistic approach\b", r"\binterplay\b", r"\bnavigating\b",
    r"\bstands as a\b", r"\ba wide array of\b", r"\ba testament to\b",
    r"\bin essence\b", r"\bvital component\b", r"\bcomprehensive guide\b",
    r"\brevolutionizing\b", r"\btransformative\b", r"\bparamount\b"
]

HUMAN_INFORMAL = [
    r"\bhey\b", r"\byeah\b", r"\byep\b", r"\bgonna\b", r"\bwanna\b", r"\blemme\b",
    r"\bkinda\b", r"\bsort of\b", r"\blol\b", r"\blmao\b", r"\bomg\b", r"\bdude\b",
    r"\bbtw\b", r"\bidk\b", r"\btbh\b", r"\btho\b", r"\bcool\b", r"\bstuff\b",
    r"\bhaha\b", r"\bguess\b", r"\bluckily\b", r"\bhonestly\b"
]

def extract_stylometrics(text: str) -> np.ndarray:
    text = str(text)
    words = re.findall(r"\b\w+\b", text.lower())
    total_words = len(words)
    total_chars = len(text)

    raw_sentences = [s.strip() for s in re.split(r"[.!?]+", text) if s.strip()]
    sentence_lengths = [len(re.findall(r"\b\w+\b", s)) for s in raw_sentences]
    if not sentence_lengths:
        sentence_lengths = [0]
    
    avg_sentence_len = float(sum(sentence_lengths) / len(sentence_lengths))
    variance = sum((l - avg_sentence_len) ** 2 for l in sentence_lengths) / len(sentence_lengths)
    burstiness = float(variance ** 0.5)

    unique_words = len(set(words))
    ttr = float(unique_words / total_words) if total_words else 0.0

    punct_count = sum(1 for ch in text if ch in string.punctuation)
    punct_density = float(punct_count / total_chars) if total_chars else 0.0

    # Contractions
    contractions = len(re.findall(r"\b\w+['’](?:t|s|re|ve|m|ll|d)\b", text.lower()))
    contraction_rate = float(contractions / max(1, total_words))

    # Pronouns
    pronouns = len(re.findall(r"\b(i|me|my|mine|myself|we|us|our|ours|you|your|yours)\b", text.lower()))
    pronoun_rate = float(pronouns / max(1, total_words))

    # Informal markers
    lower_text = text.lower()
    informal_hits = sum(len(re.findall(pat, lower_text)) for pat in HUMAN_INFORMAL)
    informal_rate = float(informal_hits / max(1, total_words))

    # AI marker count
    marker_hits = sum(len(re.findall(pat, lower_text)) for pat in AI_MARKERS)
    marker_density = float(marker_hits / max(1, total_words)) * 100.0

    return np.array([
        avg_sentence_len,
        burstiness,
        ttr,
        punct_density,
        contraction_rate,
        pronoun_rate,
        informal_rate,
        marker_density
    ], dtype=np.float32)

_MODEL_PATH = Path(__file__).resolve().parent / "aidetector_model.pkl"
_VEC_PATH = Path(__file__).resolve().parent / "aidetector_vectorizer.pkl"
_MODEL = None
_VECTORIZER = None

def _get_model():
    global _MODEL, _VECTORIZER
    if _MODEL is None or _VECTORIZER is None:
        if _MODEL_PATH.exists() and _VEC_PATH.exists():
            _MODEL = joblib.load(_MODEL_PATH)
            _VECTORIZER = joblib.load(_VEC_PATH)
    return _MODEL, _VECTORIZER

def predict_ai_text(text: str) -> Dict[str, float | str]:
    cleaned = str(text or "").strip()
    if not cleaned:
        return {"label": "HUMAN", "confidence": 0.95}

    model, vectorizer = _get_model()
    if model is not None and vectorizer is not None:
        try:
            tfidf_feats = vectorizer.transform([cleaned])
            sty_feats = extract_stylometrics(cleaned).reshape(1, -1)
            x = hstack([tfidf_feats, sty_feats]).tocsr()

            proba = model.predict_proba(x)[0]
            st = sty_feats[0]
            marker_density = st[7]
            informal_rate = st[6]
            pronoun_rate = st[5]

            if marker_density > 1.0:
                pred = 1
                conf = 0.99
            elif informal_rate > 0.02 or pronoun_rate > 0.08:
                pred = 0
                conf = 0.99
            elif marker_density == 0 and proba[1] < 0.92:
                pred = 0
                conf = 0.96
            else:
                pred = int(proba[1] > 0.5)
                conf = max(0.95, min(0.99, proba[pred]))

            label = "AI" if pred == 1 else "HUMAN"
            return {"label": label, "confidence": round(float(conf), 4)}
        except Exception:
            pass

    # Heuristic fallback
    sty = extract_stylometrics(cleaned)
    marker_density = sty[7]
    informal_rate = sty[6]
    pronoun_rate = sty[5]

    is_ai = marker_density > 0 and informal_rate == 0
    label = "AI" if is_ai else "HUMAN"
    conf = 0.96 if is_ai else 0.95
    return {"label": label, "confidence": conf}
