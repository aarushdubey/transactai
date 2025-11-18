# scripts/03_evaluate_model.py
import joblib
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix, f1_score
import seaborn as sns

DATA_DIR = Path("data")
MODELS_DIR = Path("models")
REPORTS_DIR = Path("reports")

def main():
    test_df = pd.read_csv(DATA_DIR / "test.csv")
    X_test, y_test = test_df["clean_text"], test_df["category"]

    tfidf = joblib.load(MODELS_DIR / "tfidf_vectorizer.joblib")
    clf = joblib.load(MODELS_DIR / "logistic_regression.joblib")

    X_vec = tfidf.transform(X_test)
    y_pred = clf.predict(X_vec)
    macro_f1 = f1_score(y_test, y_pred, average="macro")
    print("Test macro F1:", macro_f1)

    with open(REPORTS_DIR / "metrics_report.txt", "a") as f:
        f.write("\n\n=== Test Metrics ===\n")
        f.write(f"Test macro F1: {macro_f1:.4f}\n\n")
        f.write(classification_report(y_test, y_pred))

    cm = confusion_matrix(y_test, y_pred, labels=clf.classes_)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt="d",
                xticklabels=clf.classes_,
                yticklabels=clf.classes_)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.tight_layout()
    plt.savefig(REPORTS_DIR / "confusion_matrix.png")

if __name__ == "__main__":
    main()
