from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field, field_validator
import pandas as pd

from pathlib import Path
import joblib

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "model.joblib"

model = joblib.load(MODEL_PATH)
app = FastAPI()



class ChurnRequest(BaseModel):
    AccountAge: int = Field(..., ge=0)
    MonthlyCharges: float = Field(..., ge=0)
    ViewingHoursPerWeek: float = Field(..., ge=0)
    ContentDownloadsPerMonth: int = Field(..., ge=0)

    SubscriptionType: str
    ContentType: str
    DeviceRegistered: str
    GenrePreference: str

    @field_validator("SubscriptionType", "ContentType", "DeviceRegistered", "GenrePreference")
    @classmethod
    def validate_non_empty_strings(cls, value: str) -> str:
        value = value.strip()
        if not value:
            raise ValueError("must not be empty or only whitespace")
        return value


class ChurnResponse(BaseModel):
    probability: float
    prob: float
    threshold: float
    label: int
    risk_band: str
    message: str


@app.exception_handler(RequestValidationError)
async def validation_exception_handler(request: Request, exc: RequestValidationError):
    formatted_errors = []

    for err in exc.errors():
        field_name = " -> ".join(str(x) for x in err["loc"])
        formatted_errors.append({
            "field": field_name,
            "message": err["msg"],
            "type": err["type"]
        })

    return JSONResponse(
        status_code=422,
        content={
            "error": "Invalid input",
            "message": "Request validation failed",
            "details": formatted_errors
        }
    )


model = joblib.load("models/model.joblib")
preprocessor = joblib.load("models/preprocessor.joblib")
metadata = joblib.load("models/metadata.joblib")
features = metadata["features"]
threshold = metadata["threshold"]


@app.post("/predict_churn", response_model=ChurnResponse)
def predict_churn(data: ChurnRequest):
    input_data = [[
        data.AccountAge,
        data.MonthlyCharges,
        data.ViewingHoursPerWeek,
        data.ContentDownloadsPerMonth,
        data.SubscriptionType,
        data.ContentType,
        data.DeviceRegistered,
        data.GenrePreference
    ]]

    input_df = pd.DataFrame(input_data, columns=features)

    processed_input = preprocessor.transform(input_df)
    prob = model.predict_proba(processed_input)[0][1]
    label = 1 if prob >= threshold else 0

    if prob < 0.3:
        risk_band = "low"
    elif prob <= 0.7:
        risk_band = "medium"
    else:
        risk_band = "high"

    return {
        "probability": prob,
        "prob": prob,
        "threshold": threshold,
        "label": label,
        "risk_band": risk_band,
        "message": "Customer is likely to churn" if label == 1 else "Customer is unlikely to churn"
    }