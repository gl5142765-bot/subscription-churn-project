# Streaming Membership Churn Dashboard

A machine learning project that predicts whether a streaming subscriber is likely to churn. The application is designed as a small product demo: a user enters subscriber profile, subscription, and engagement details, and the system returns churn probability, a risk label, and a simple retention recommendation.

Subscriber churn is a practical business problem for streaming platforms because cancellation risk directly affects recurring revenue, retention campaigns, and content strategy. When engagement drops, billing feels less valuable, or content preferences are not met, a subscriber becomes more likely to cancel. Predicting this risk early allows teams to intervene before churn happens with targeted recommendations, pricing offers, or engagement nudges.

This project frames the churn problem in a streaming-style context using a subscription churn dataset adapted for dashboard-based prediction. The final system combines a trained machine learning model, a FastAPI inference service, and a Streamlit UI deployed as a live demo.

---

## Live demo

- **Streamlit dashboard:**  
  https://subscription-churn-project-8ksqnzkog28vsrx9qhksmz.streamlit.app/
- **FastAPI docs (API):**  
  https://subscription-churn-project.onrender.com/docs

---

## Problem statement

Streaming platforms depend on recurring subscriptions, so retaining existing users is just as important as acquiring new ones. As subscribers change their viewing habits, switch devices, or upgrade/downgrade plans, their likelihood of churn can change significantly. A single view of profile, billing, and engagement data is often not enough for teams to quickly estimate churn risk user-by-user.

In a real product setting, churn prediction can support customer success, CRM campaigns, recommendation systems, and product analytics. Instead of manually reviewing raw behavioral data, teams can use a model-driven workflow that estimates churn probability and classifies subscribers into low, medium, or high-risk groups for action.

This dashboard demonstrates that workflow in a portfolio-ready format. It packages a trained churn model inside a FastAPI backend, exposes a prediction endpoint, and provides a Streamlit UI where non-technical users can test subscriber scenarios and immediately view risk outputs as probabilities, labels, and business-facing messages.

---

## Dataset

This project uses a subscription churn dataset from Kaggle and adapts it to a streaming membership use case:

- **Dataset source:** [Streaming Subscription Churn Model – Kaggle](https://www.kaggle.com/competitions/streaming-subscription-churn-model/data)  
  The dataset contains subscription behavior and churn labels for supervised learning.[web:1649]

The dashboard uses selected profile, billing, and engagement-style features to simulate a streaming churn workflow:

- Account age (months)
- Monthly charges (numeric billing amount)
- Viewing hours per week
- Content downloads per month
- Subscription type (Basic / Standard / Premium)
- Preferred content type (Movies / TV Shows / Both)
- Registered device (Mobile, Tablet, TV, Laptop)
- Genre preference (Action, Comedy, Drama, Horror, Sci‑Fi, Romance)

---

## Model summary

**Model name:** `SUBSCIRPTION CHURN PREDICTION`  
*A supervised classification pipeline for streaming membership churn.*

- **Task:** Binary churn prediction (likely to churn vs. unlikely to churn).
- **Preprocessing:** Numeric and categorical features handled via a scikit‑learn preprocessing pipeline (e.g. imputation, encoding, scaling).
- **Estimator:** A scikit‑learn classifier (e.g. tree‑based model or ensemble) trained on the subscription churn dataset.
- **Thresholding:** Predictions are turned into labels via a configurable decision threshold.

The API response includes:

- `probability` / `prob`: churn probability between 0 and 1.
- `threshold`: the cutoff used to classify churn vs non‑churn.
- `label`: 1 (churn) or 0 (no churn).
- `risk_band`: `"low"`, `"medium"`, or `"high"`.
- `message`: short explanation for business users.

**Example API output:**

```json
{
  "probability": 0.6784650729677506,
  "prob": 0.6784650729677506,
  "threshold": 0.7,
  "label": 0,
  "risk_band": "medium",
  "message": "Customer is unlikely to churn"
}
```

In this scenario, the model estimates a churn probability of ~0.68, which is below the decision threshold of 0.7. The label is `0` (unlikely to churn) and the risk band is `"medium"`, signaling a user worth monitoring but not yet flagged as high risk.

---

## Architecture

High‑level architecture:

```text
Data -> preprocessing + trained model -> FastAPI API -> Streamlit dashboard -> cloud deployment
```

### Flow

1. **UI input (Streamlit):**  
   User enters subscriber profile, subscription, and engagement details in the dashboard.

2. **Request (Streamlit → FastAPI):**  
   The UI sends the input as JSON to the FastAPI backend via the `/predict_churn` endpoint.

3. **Validation (FastAPI + Pydantic):**  
   FastAPI validates the request using a `ChurnRequest` Pydantic model (numeric ranges and non‑empty strings).

4. **Inference (model pipeline):**  
   The backend loads the preprocessor, model, and metadata artifacts from the `models/` folder, transforms the input, and computes churn probability.

5. **Response (FastAPI → Streamlit):**  
   The API returns probability, threshold, label, risk band, and message in a JSON payload.

6. **Visualization (Streamlit):**  
   The dashboard shows metrics, risk label, badges, and a recommendation message. An expander allows viewing the raw API response for debugging.

---

## Project structure

```text
subscription-churn-project/
├── app/
│   ├── main.py        # FastAPI backend (prediction API)
│   └── ui.py          # Streamlit frontend (dashboard UI)
├── data/              # Raw / processed data (if needed)
├── models/            # Preprocessor, model, metadata artifacts (.joblib)
├── notebooks/         # Training and exploration notebooks
├── src/               # Supporting code (feature engineering, utilities)
├── tests/             # Tests for backend logic and data checks
├── README.md          # Project documentation (this file)
├── FUTURE_WORK.md     # Planned improvements and next steps
└── requirements.txt   # Pinned dependencies for reproducible deployment
```

---

## Setup instructions

### 1. Create a virtual environment

#### Windows

```bash
python -m venv venv
venv\Scripts\activate
```

#### macOS / Linux

```bash
python -m venv venv
source venv/bin/activate
```

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

---

## Run the project locally

### Start FastAPI backend

From the project root, run:

```bash
uvicorn app.main:app --reload
```

FastAPI will run at:

```text
http://127.0.0.1:8000
```

Swagger docs:

```text
http://127.0.0.1:8000/docs
```

### Start Streamlit frontend

Open a new terminal, activate the same virtual environment, and run:

```bash
streamlit run app/ui.py
```

Streamlit will run at:

```text
http://localhost:8501
```

### Notes

- Make sure the FastAPI server is running before using the Streamlit dashboard.
- Monthly charges are used as provided in the dataset (numeric billing amount) and treated as a continuous feature.
- Keep all paths relative to the project root.
- Store trained model, preprocessor, and metadata artefacts inside the `models/` folder.
- For cloud deployment, ensure `requirements.txt` pins the scikit‑learn version used to train and serialise the model artefacts to avoid version‑mismatch errors.

---

## Deployed services

- **FastAPI docs (API):**  
  https://subscription-churn-project.onrender.com/docs

- **Streamlit dashboard:**  
  https://subscription-churn-project-8ksqnzkog28vsrx9qhksmz.streamlit.app/

The Streamlit app is configured to call the deployed FastAPI backend instead of localhost, enabling a fully hosted demo that can be shared in portfolios and interviews.

---

## How to use this dashboard (for PM / Customer Success)

This dashboard is designed for a product manager, growth analyst, or customer success stakeholder who wants a quick estimate of subscriber churn risk.

1. **Fill in subscriber profile:**
   - Account age (months).
   - Registered device (Mobile, Tablet, TV, Laptop).
   - Genre preference (e.g. Action, Drama).

2. **Enter subscription information:**
   - Subscription type (Basic, Standard, Premium).
   - Monthly charges.
   - Preferred content type (Movies, TV Shows, Both).

3. **Add engagement metrics:**
   - Viewing hours per week.
   - Content downloads per month.

4. **Click “Predict churn”:**
   - The dashboard calls the FastAPI backend and generates a prediction.

5. **Interpret the results:**
   - Churn probability (0–1).
   - Risk label and band (LOW / MEDIUM / HIGH).
   - Recommendation message such as:
     - *“High risk – user is likely to cancel; consider personalised recommendations or special offers.”*
     - *“Customer is unlikely to churn”* for medium or low‑risk scenarios.

Using this workflow, non‑technical users can quickly test scenarios and reason about how profile, plan, and engagement changes affect churn risk.

---

## Example prediction

A typical medium‑risk scenario might look like this in the raw API response:

```json
{
  "probability": 0.6784650729677506,
  "prob": 0.6784650729677506,
  "threshold": 0.7,
  "label": 0,
  "risk_band": "medium",
  "message": "Customer is unlikely to churn"
}
```

Business interpretation:

- Probability is ~0.68, slightly below the decision threshold of 0.7.
- Label `0` means “unlikely to churn” according to the threshold.
- Risk band `"medium"` identifies the user as worth monitoring but not yet flagged as high risk.
- The message provides a quick summary for PM / CS without needing to inspect raw numbers.

---

## Screenshots
<img width="1882" height="897" alt="image" src="https://github.com/user-attachments/assets/e11467ff-9323-4f61-bda6-a525e3c617a3" />

<img width="1906" height="855" alt="image" src="https://github.com/user-attachments/assets/54397bbf-9477-444d-870c-5b036d4e480c" />


## Demo features

- Subscriber churn prediction for streaming‑style subscriptions.
- Churn probability output with configurable threshold.
- Risk label and risk band (low / medium / high).
- Business‑friendly recommendation message.
- FastAPI backend serving a `/predict_churn` endpoint.
- Streamlit dashboard UI with metrics, badges, and debug expander.
- Live cloud deployment (FastAPI on Render, UI on Streamlit Community Cloud).

---

## Future work

See `FUTURE_WORK.md` for ideas such as:

- Logging predictions and monitoring data/usage drift.
- Periodic retraining schedule for the churn model.
- SHAP explanations to show which behaviours drive churn.
- Authentication and multi‑user access (teams, roles).
- Integrating with a recommendation system to auto‑suggest shows to at‑risk users.

This repository plus the live demo are intended to be portfolio‑ready and suitable to discuss in interviews as a streaming churn prediction product.
