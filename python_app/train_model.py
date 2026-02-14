"""
Professional training pipeline for the phishing email classifier.

Usage:
    python train_model.py                    # train on data/train.csv
    python train_model.py --data path.csv   # custom dataset

CSV format: one of
    - text,label   with label in (0,1) or (legitimate, phishing) or (safe, phishing)
    - content,is_phishing
"""
from __future__ import annotations

import argparse
import csv
import os
import re
import string
from pathlib import Path

import joblib
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
)
from sklearn.model_selection import train_test_split


# Must match ml_model.clean_text exactly so inference matches training
def clean_text(text: str) -> str:
    text = (text or "").lower()
    text = re.sub(r"http\S+", " URL ", text)
    text = re.sub(r"\d+", " NUM ", text)
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text.strip()


def load_csv(path: str) -> tuple[list[str], np.ndarray]:
    """Load (texts, labels) from CSV. label 1 = phishing, 0 = legitimate."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset not found: {path}")
    texts = []
    labels = []
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        reader = csv.DictReader(f)
        if not reader.fieldnames:
            raise ValueError("CSV has no header")
        # use first row headers; normalize to find text/label columns
        raw_fields = [n.strip() for n in reader.fieldnames]
        fieldnames_lower = [n.lower() for n in raw_fields]
        text_col = None
        label_col = None
        for i, name in enumerate(fieldnames_lower):
            if name in ("text", "content", "body", "email", "message"):
                text_col = raw_fields[i]
                break
        if not text_col:
            text_col = raw_fields[0]
        for i, name in enumerate(fieldnames_lower):
            if name in ("label", "is_phishing", "phishing", "class", "category"):
                label_col = raw_fields[i]
                break
        if not label_col:
            label_col = raw_fields[1] if len(raw_fields) > 1 else "label"
        for row in reader:
            t = (row.get(text_col) or "").strip()
            if not t:
                continue
            raw_label = (row.get(label_col) or "0").strip().lower()
            if raw_label in ("1", "phishing", "phish", "malicious", "1.0", "yes", "true"):
                labels.append(1)
            else:
                labels.append(0)
            texts.append(t)
    if not texts:
        raise ValueError("No rows loaded from CSV")
    return texts, np.array(labels, dtype=np.int64)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train phishing email classifier")
    parser.add_argument(
        "--data",
        type=str,
        default=None,
        help="Path to CSV (default: data/train.csv)",
    )
    parser.add_argument(
        "--test-size",
        type=float,
        default=0.2,
        help="Fraction for test set (default: 0.2)",
    )
    parser.add_argument(
        "--max-features",
        type=int,
        default=50_000,
        help="TfidfVectorizer max_features (default: 50000)",
    )
    parser.add_argument(
        "--ngram-range",
        type=str,
        default="1,2",
        help="N-gram range as min,max (default: 1,2)",
    )
    parser.add_argument(
        "--C",
        type=float,
        default=1.0,
        help="LogisticRegression inverse regularization (default: 1.0)",
    )
    parser.add_argument(
        "--max-iter",
        type=int,
        default=2000,
        help="LogisticRegression max_iter (default: 2000)",
    )
    parser.add_argument(
        "--out-dir",
        type=str,
        default=None,
        help="Directory to save model and preprocessor (default: script dir)",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed (default: 42)",
    )
    args = parser.parse_args()

    base_dir = Path(__file__).resolve().parent
    data_path = args.data or (base_dir / "data" / "train.csv")
    out_dir = Path(args.out_dir) if args.out_dir else base_dir

    print("Loading dataset...")
    texts, y = load_csv(str(data_path))
    n_phish = int(y.sum())
    n_legit = len(y) - n_phish
    print(f"  Total: {len(texts)}, Phishing: {n_phish}, Legitimate: {n_legit}")

    print("Cleaning text...")
    X_cleaned = [clean_text(t) for t in texts]

    X_train, X_test, y_train, y_test = train_test_split(
        X_cleaned, y, test_size=args.test_size, random_state=args.seed, stratify=y
    )

    ngram_parts = [int(x.strip()) for x in args.ngram_range.split(",")]
    ngram_range = (ngram_parts[0], ngram_parts[1]) if len(ngram_parts) >= 2 else (1, 2)

    print("Building vectorizer and model...")
    vectorizer = TfidfVectorizer(
        max_features=args.max_features,
        ngram_range=ngram_range,
        sublinear_tf=True,
        min_df=2,
        max_df=0.95,
        strip_accents="unicode",
        lowercase=True,
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)

    model = LogisticRegression(
        max_iter=args.max_iter,
        C=args.C,
        class_weight="balanced",
        random_state=args.seed,
        solver="lbfgs",
    )
    model.fit(X_train_vec, y_train)

    y_pred = model.predict(X_test_vec)
    acc = accuracy_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    print("\nTest set performance:")
    print(f"  Accuracy: {acc:.4f}")
    print(f"  F1 (phishing): {f1:.4f}")
    print("\nClassification report:")
    print(classification_report(y_test, y_pred, target_names=["Legitimate", "Phishing"]))
    print("Confusion matrix:")
    print(confusion_matrix(y_test, y_pred))

    out_dir.mkdir(parents=True, exist_ok=True)
    model_path = out_dir / "model.pkl"
    preprocessor_path = out_dir / "preprocessor.pkl"
    joblib.dump(model, model_path)
    joblib.dump(vectorizer, preprocessor_path)
    print(f"\nSaved model to {model_path}")
    print(f"Saved preprocessor to {preprocessor_path}")


if __name__ == "__main__":
    main()
