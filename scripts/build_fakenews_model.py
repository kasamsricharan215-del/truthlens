import re
import string
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import joblib
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

STOPWORDS = {
    "a", "an", "and", "are", "as", "at", "be", "by", "for", "from",
    "has", "he", "in", "is", "it", "its", "of", "on", "that", "the",
    "to", "was", "were", "will", "with"
}

def clean_text(text: str) -> str:
    text = str(text).lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [w for w in text.split(" ") if w and w not in STOPWORDS]
    return " ".join(tokens)

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "text"
    data_dir.mkdir(parents=True, exist_ok=True)
    news_csv = data_dir / "news.csv"

    print("Loading GonzaloA/fake_news dataset...")
    ds = load_dataset("GonzaloA/fake_news", split="train")
    df = ds.to_pandas()[["text", "label"]].dropna()

    # CRITICAL FIX: In GonzaloA/fake_news, 0 is Fake, 1 is Real.
    # TruthLens convention across codebase: 1=FAKE, 0=REAL.
    df["label"] = (df["label"] == 0).astype(int)

    df = df.sample(min(12000, len(df)), random_state=42)
    df.to_csv(news_csv, index=False)
    print(f"Saved {len(df)} samples to {news_csv}. Fake (1): {sum(df['label']==1)}, Real (0): {sum(df['label']==0)}")

    print("Cleaning text...")
    cleaned = df["text"].apply(clean_text)

    x_train, x_test, y_train, y_test = train_test_split(
        cleaned, df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Vectorizing...")
    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
    x_train_vec = vectorizer.fit_transform(x_train)
    x_test_vec = vectorizer.transform(x_test)

    print("Fitting model...")
    model = LogisticRegression(C=3.0, max_iter=1000, random_state=42)
    model.fit(x_train_vec, y_train)

    preds = model.predict(x_test_vec)
    acc = accuracy_score(y_test, preds)
    print(f"\n==========================================")
    print(f"Fake News Model Accuracy: {acc * 100:.2f}%")
    print(f"==========================================")
    print(classification_report(y_test, preds, target_names=["Real", "Fake"]))

    model_dir = project_root / "models" / "text"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "fakenews_model.pkl")
    joblib.dump(vectorizer, model_dir / "tfidf.pkl")
    print(f"Successfully saved model to {model_dir / 'fakenews_model.pkl'}")
    print(f"Successfully saved vectorizer to {model_dir / 'tfidf.pkl'}")

if __name__ == "__main__":
    main()
