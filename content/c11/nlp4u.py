"""
Chapter 11: Natural Language Processing for Utilities
Analyze maintenance logs and regulatory documents using NLP (spaCy).
"""

import pandas as pd
import re
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

def generate_maintenance_logs():
    """
    Generate synthetic maintenance logs with failure vs. routine labels.
    """
    logs = [
        "Transformer oil leak detected near bushing. Immediate repair required.",
        "Routine inspection of substation breakers completed.",
        "Severe vibration detected on cooling fan motor.",
        "Preventive maintenance: tested relay settings.",
        "Burn marks observed on conductor, risk of fault high.",
        "Monthly cleaning of control room performed."
    ]
    labels = [1, 0, 1, 0, 1, 0]  # 1=Failure, 0=Routine
    return pd.DataFrame({"log": logs, "failure": labels})

def classify_logs(df):
    """
    Train a TF-IDF + Logistic Regression model to classify logs.
    """
    tfidf = TfidfVectorizer()
    X = tfidf.fit_transform(df["log"])
    y = df["failure"]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    model = LogisticRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)

    print("Maintenance Log Classification Report:")
    print(classification_report(y_test, preds, target_names=["Routine", "Failure"]))

def extract_entities(text):
    """
    Extract equipment and issues from regulatory or maintenance text.
    """
    nlp = spacy.load("en_core_web_sm")
    doc = nlp(text)
    print("Named Entities:")
    for ent in doc.ents:
        print(f"{ent.text} ({ent.label_})")

    equipment_terms = re.findall(r"(transformer|breaker|relay|conductor|fan)", text, re.IGNORECASE)
    print("\nDetected Equipment Terms:", equipment_terms)

if __name__ == "__main__":
    # Classify logs
    df_logs = generate_maintenance_logs()
    classify_logs(df_logs)

    # Extract entities from regulatory document
    sample_text = """
    NERC CIP compliance audit found gaps in relay testing documentation.
    Transformer T-103 requires oil quality testing per IEEE C57 standards.
    """
    extract_entities(sample_text)
