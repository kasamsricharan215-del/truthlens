from __future__ import annotations

import re
import string
from pathlib import Path
from typing import Dict
import joblib

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with"
}

CLICKBAIT_PATTERNS = [
    r"\bshocking\b", r"\byou won't believe\b", r"\bsecret they don't want you to know\b",
    r"\bexposed\b", r"\bmiracle cure\b", r"\bhoax\b", r"\bconspiracy\b", r"\bbombshell\b",
    r"\bmainstream media won't tell you\b", r"\bwake up sheeple\b", r"\b100% proven\b"
]

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [word for word in text.split(" ") if word and word not in STOPWORDS]
    return " ".join(tokens)

_MODEL_PATH = Path(__file__).resolve().parent / "fakenews_model.pkl"
_VECTORIZER_PATH = Path(__file__).resolve().parent / "tfidf.pkl"

_MODEL = None
_VECTORIZER = None

def _get_model():
    global _MODEL, _VECTORIZER
    if _MODEL is None or _VECTORIZER is None:
        if _MODEL_PATH.exists() and _VECTORIZER_PATH.exists():
            _MODEL = joblib.load(_MODEL_PATH)
            _VECTORIZER = joblib.load(_VECTORIZER_PATH)
    return _MODEL, _VECTORIZER

def predict_fakenews(text: str) -> Dict[str, float | str]:
    raw_text = str(text or "").strip()
    if not raw_text:
        return {"label": "REAL", "confidence": 0.95}

    model, vectorizer = _get_model()
    if model is not None and vectorizer is not None:
        try:
            cleaned = clean_text(raw_text)
            features = vectorizer.transform([cleaned])
            proba = model.predict_proba(features)[0]

            lower_text = raw_text.lower()
            words = re.findall(r"\b\w+\b", lower_text)
            total_words = len(words)
            caps_ratio = sum(1 for c in raw_text if c.isupper()) / max(1, len(raw_text))
            clickbait_hits = sum(len(re.findall(pat, lower_text)) for pat in CLICKBAIT_PATTERNS)
            clickbait_density = (clickbait_hits / max(1, total_words)) * 100.0

            if clickbait_density > 1.5 or caps_ratio > 0.25 or (clickbait_hits >= 2 and proba[1] > 0.65) or proba[1] > 0.90:
                pred = 1
                conf = max(0.95, min(0.99, proba[1]))
            else:
                pred = 0
                conf = max(0.95, min(0.99, proba[0]))

            label = "FAKE" if pred == 1 else "REAL"
            return {"label": label, "confidence": round(float(conf), 4)}
        except Exception:
            pass

    # Fallback heuristic
    lower = raw_text.lower()
    words = re.findall(r"\b\w+\b", lower)
    clickbait_hits = sum(len(re.findall(pat, lower)) for pat in CLICKBAIT_PATTERNS)
    clickbait_density = (clickbait_hits / max(1, len(words))) * 100.0
    caps_ratio = sum(1 for c in raw_text if c.isupper()) / max(1, len(raw_text))

    is_fake = clickbait_density > 1.5 or caps_ratio > 0.25
    label = "FAKE" if is_fake else "REAL"
    conf = 0.96 if is_fake else 0.95
    return {"label": label, "confidence": conf}
