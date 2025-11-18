# scripts/02_train_model.py
import joblib
from pathlib import Path
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, f1_score

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

MODELS_DIR.mkdir(exist_ok=True)
REPORTS_DIR.mkdir(exist_ok=True)

def main():
    train_df = pd.read_csv(DATA_DIR / "train.csv")
    val_df = pd.read_csv(DATA_DIR / "val.csv")

    X_train, y_train = train_df["clean_text"], train_df["category"]
    X_val, y_val = val_df["clean_text"], val_df["category"]

    pipeline = Pipeline(
        [
            ("tfidf", TfidfVectorizer(
                ngram_range=(1, 3),
                min_df=2,
                max_features=50000,
            )),
            ("clf", LogisticRegression(
                max_iter=200,
                class_weight="balanced",
                n_jobs=-1,
            )),
        ]
    )

    pipeline.fit(X_train, y_train)

    y_pred = pipeline.predict(X_val)
    macro_f1 = f1_score(y_val, y_pred, average="macro")
    print("Validation macro F1:", macro_f1)

    with open(REPORTS_DIR / "metrics_report.txt", "w") as f:
        f.write(f"Validation macro F1: {macro_f1:.4f}\n\n")
        f.write(classification_report(y_val, y_pred))

    # Save separate components if you want
    tfidf = pipeline.named_steps["tfidf"]
    clf = pipeline.named_steps["clf"]

    joblib.dump(tfidf, MODELS_DIR / "tfidf_vectorizer.joblib")
    joblib.dump(clf, MODELS_DIR / "logistic_regression.joblib")

if __name__ == "__main__":
    main()
