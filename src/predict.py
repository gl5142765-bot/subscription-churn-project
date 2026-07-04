src/predict.py <<'PY'
from pathlib import Path
import joblib
import pandas as pd


from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"
PREPROCESSOR_PATH = BASE_DIR / "models" / "preprocessor.joblib"

model = joblib.load(MODEL_PATH)
preprocessor = joblib.load(PREPROCESSOR_PATH)

def predict_single(customer_dict):
    preprocessor = joblib.load(MODELS_DIR / "preprocessor.joblib")
    model = joblib.load(MODELS_DIR / "model.joblib")
    metadata = joblib.load(MODELS_DIR / "metadata.joblib")

    features = metadata["features"]
    threshold = metadata["threshold"]

    customer_df = pd.DataFrame([customer_dict])
    customer_df = customer_df.reindex(columns=features)

    X_processed = preprocessor.transform(customer_df)
    probability = model.predict_proba(X_processed)[:, 1][0]
    label = int(probability >= threshold)

    return {"probability": float(probability), "label": label}
PY

