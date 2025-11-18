# scripts/01_prepare_data.py
import pandas as pd
from pathlib import Path
from sklearn.model_selection import train_test_split

DATA_DIR = Path("data")

def clean_text(text: str) -> str:
    if not isinstance(text, str):
        return ""
    text = text.lower()
    return " ".join(text.split())

def main():
    df = pd.read_csv(DATA_DIR / "transactions_dataset.csv")
    df["clean_text"] = df["raw_description"].apply(clean_text)

    train_df, temp_df = train_test_split(
        df, test_size=0.3, random_state=42, stratify=df["category"]
    )
    val_df, test_df = train_test_split(
        temp_df, test_size=0.5, random_state=42, stratify=temp_df["category"]
    )

    train_df.to_csv(DATA_DIR / "train.csv", index=False)
    val_df.to_csv(DATA_DIR / "val.csv", index=False)
    test_df.to_csv(DATA_DIR / "test.csv", index=False)
    print("Prepared train/val/test splits.")

if __name__ == "__main__":
    main()
