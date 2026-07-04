from pathlib import Path
import joblib
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "models"
MODELS_DIR.mkdir(parents=True, exist_ok=True)

MODEL_PATH = MODELS_DIR / "model.joblib"
PREPROCESSOR_PATH = MODELS_DIR / "preprocessor.joblib"

from src.data_loader import load_data
from src.preprocessing import build_preprocessor, NUMERIC_FEATURES, CATEGORICAL_FEATURES

TARGET = "Churn"
THRESHOLD = 0.35
FINAL_PARAMS = {"C": 0.01, "solver": "liblinear"}


def main():
    df = load_data()
    feature_columns = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = df[feature_columns]
    y = df[TARGET]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    preprocessor = build_preprocessor()
    model = LogisticRegression(
        max_iter=1000,
        class_weight="balanced",
        C=FINAL_PARAMS["C"],
        solver=FINAL_PARAMS["solver"],
    )

    pipeline = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("classifier", model),
    ])

    pipeline.fit(X_train, y_train)

    y_prob = pipeline.predict_proba(X_test)[:, 1]
    y_pred = (y_prob >= THRESHOLD).astype(int)

    print("Algorithm: LogisticRegression")
    print("Final hyperparameters:", FINAL_PARAMS)
    print("Threshold:", THRESHOLD)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Precision:", precision_score(y_test, y_pred, zero_division=0))
    print("Recall:", recall_score(y_test, y_pred, zero_division=0))
    print("F1 Score:", f1_score(y_test, y_pred, zero_division=0))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))
    print("\nClassification Report:\n", classification_report(y_test, y_pred, zero_division=0))

    models_dir = Path("models")
    models_dir.mkdir(exist_ok=True)
    joblib.dump(preprocessor, models_dir / "preprocessor.joblib")
    joblib.dump(model, models_dir / "model.joblib")
    joblib.dump(
        {
            "features": feature_columns,
            "target": TARGET,
            "threshold": THRESHOLD,
            "params": FINAL_PARAMS,
        },
        models_dir / "metadata.joblib",
    )


if __name__ == "__main__":
    main()

