import csv
import sys
from pathlib import Path
try:
    import joblib
    import pandas as pd
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.metrics import classification_report
    from sklearn.model_selection import train_test_split
    from sklearn.naive_bayes import MultinomialNB
    from sklearn.pipeline import make_pipeline
except ModuleNotFoundError as err:
    sys.exit(
        f"Missing package: {err.name}\n"
        "Install with:  python -m pip install scikit-learn pandas joblib\n"
        "(use the same Python that VS Code has selected)"
    )

BASE_DIR = Path(__file__).resolve().parent
DATA_FILE = BASE_DIR / "SMSSpamCollection"
MODEL_FILE = BASE_DIR / "spam_model.joblib"
def load_data():
    if not DATA_FILE.exists():
        sys.exit(f"Dataset not found: {DATA_FILE}")

    df = pd.read_csv(
        DATA_FILE,
        sep="\t",
        header=None,
        names=["label", "text"],
        quoting=csv.QUOTE_NONE,
        encoding="utf-8",
    )
    df["is_spam"] = (df["label"] == "spam").astype(int)
    return df

def train():
    df = load_data()
    X_train, X_test, y_train, y_test = train_test_split(
        df["text"],
        df["is_spam"],
        test_size=0.2,
        random_state=42,
        stratify=df["is_spam"],
    ) 

    model = make_pipeline(
        TfidfVectorizer(lowercase=True, stop_words="english", ngram_range=(1, 2)),
        MultinomialNB(alpha=0.1),
    )
    model.fit(X_train, y_train)
    print("Evaluation on unseen messages:")
    print(
        classification_report(
            y_test, model.predict(X_test), target_names=["Not spam", "Spam"]
        )
    )
    joblib.dump(model, MODEL_FILE)
    return


# ========================================= Continue ===========================================