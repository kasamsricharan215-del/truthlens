import re
import string
from collections import Counter
from pathlib import Path
from datasets import load_dataset
import pandas as pd
import numpy as np
import joblib
from scipy.sparse import hstack
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import train_test_split

AI_MARKERS = [
    r"\bdelve\b", r"\btestament\b", r"\btapestry\b", r"\bpivotal\b", r"\bbeacon\b",
    r"\bfostering\b", r"\bseamlessly\b", r"\bunderscores\b", r"\bmoreover\b",
    r"\bfurthermore\b", r"\bin summary\b", r"\bin conclusion\b", r"\bit is crucial\b",
    r"\bit is important to\b", r"\bplays a crucial role\b", r"\bplays a pivotal role\b",
    r"\bharnessing\b", r"\belevate\b", r"\brealm\b", r"\bever-evolving\b",
    r"\brich tapestry\b", r"\bholistic approach\b", r"\binterplay\b", r"\bnavigating\b",
    r"\bstands as a\b", r"\ba wide array of\b", r"\ba testament to\b",
    r"\bin essence\b", r"\bvital component\b", r"\bcomprehensive guide\b"
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

    # First & second person pronouns (I, me, my, we, us, our, you, your)
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

def main():
    project_root = Path(__file__).resolve().parents[1]
    data_dir = project_root / "data" / "text"
    data_dir.mkdir(parents=True, exist_ok=True)
    csv_path = data_dir / "ai_vs_human.csv"

    print("Loading aadityaubhat/GPT-wiki-intro from cache...")
    ds = load_dataset("aadityaubhat/GPT-wiki-intro", split="train[:12000]")

    records = []
    for row in ds:
        human_text = row["wiki_intro"]
        ai_text = row["generated_intro"]
        if human_text and len(human_text.strip()) > 30:
            records.append({"text": human_text.strip(), "label": 0})
        if ai_text and len(ai_text.strip()) > 30:
            records.append({"text": ai_text.strip(), "label": 1})

    # Add diverse conversational and blog/news human texts
    extra_human = [
        "hey man, i was thinking we should probably grab some lunch later if you are free? lemme know",
        "Just finished watching the new episode and honestly it was so good, definitely recommend it to anyone who enjoys sci-fi.",
        "Can anyone help me figure out why my computer keeps restarting randomly when I open Chrome? It is driving me crazy.",
        "Had such an amazing time at the beach yesterday with the family. The weather was perfect and the water was warm.",
        "I really don't think that is how that works, you might want to double check the documentation before deploying.",
        "Thanks for sending over the invoice! I will forward it to accounting first thing tomorrow morning.",
        "Anyone know a good place to get coffee around downtown? Looking for somewhere quiet to read.",
        "Yeah absolutely, let me check my schedule for next Tuesday and I will get back to you by this afternoon.",
        "The recipe called for two tablespoons of sugar but I only added one and it still turned out way too sweet.",
        "I was walking my dog in the park this morning when we ran into our old neighbor from three years ago!"
    ]
    for text in extra_human * 100:  # Upsample human diverse conversational
        records.append({"text": text, "label": 0})

    extra_ai = [
        "In today's fast-paced digital world, artificial intelligence plays a pivotal role in revolutionizing various sectors.",
        "It is crucial to understand that fostering collaboration serves as a testament to the power of human ingenuity.",
        "Furthermore, by delving into the multifaceted nuances of this technology, organizations can unlock unprecedented opportunities.",
        "Moreover, this framework seamlessly integrates diverse workflows, thereby elevating operational efficiency to new heights.",
        "In summary, the interplay between innovation and sustainability stands as a beacon for future advancements.",
        "As we navigate the ever-evolving landscape of modern technology, it becomes paramount to embrace a holistic approach.",
        "Delving into the rich tapestry of historical traditions underscores the profound significance of cultural heritage.",
        "Harnessing the potential of clean energy is a vital component in our collective quest to combat climate change.",
        "In conclusion, these groundbreaking developments not only enrich our understanding but also pave the way for sustainable growth.",
        "It is important to remember that achieving long-term excellence requires unwavering commitment and strategic foresight."
    ]
    for text in extra_ai * 100:
        records.append({"text": text, "label": 1})

    df = pd.DataFrame(records)
    df.to_csv(csv_path, index=False)
    print(f"Dataset samples: {len(df)} | Human: {sum(df['label']==0)}, AI: {sum(df['label']==1)}")

    x_train_text, x_test_text, y_train, y_test = train_test_split(
        df["text"], df["label"], test_size=0.2, random_state=42, stratify=df["label"]
    )

    print("Extracting TF-IDF...")
    vectorizer = TfidfVectorizer(max_features=25000, ngram_range=(1, 2), sublinear_tf=True)
    x_train_tfidf = vectorizer.fit_transform(x_train_text)
    x_test_tfidf = vectorizer.transform(x_test_text)

    print("Extracting stylometrics...")
    x_train_sty = np.vstack([extract_stylometrics(t) for t in x_train_text])
    x_test_sty = np.vstack([extract_stylometrics(t) for t in x_test_text])

    x_train = hstack([x_train_tfidf, x_train_sty]).tocsr()
    x_test = hstack([x_test_tfidf, x_test_sty]).tocsr()

    print("Training model...")
    model = LogisticRegression(C=2.5, max_iter=1000, random_state=42)
    model.fit(x_train, y_train)

    preds = model.predict(x_test)
    acc = accuracy_score(y_test, preds)
    print(f"\n==========================================")
    print(f"AI Text Detector Accuracy: {acc * 100:.2f}%")
    print(f"==========================================")
    print(classification_report(y_test, preds, target_names=["Human", "AI"]))

    model_dir = project_root / "models" / "text"
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, model_dir / "aidetector_model.pkl")
    joblib.dump(vectorizer, model_dir / "aidetector_vectorizer.pkl")
    print("Saved aidetector_model.pkl and aidetector_vectorizer.pkl")

if __name__ == "__main__":
    main()
