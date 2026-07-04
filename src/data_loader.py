from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_PATH = BASE_DIR / "data" / "churn.csv"


def load_data(csv_path=DATA_PATH):
    df = pd.read_csv(csv_path)
    return df


