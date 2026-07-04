# Streaming Membership Churn Dashboard

A machine learning project that predicts whether a streaming subscriber is likely to churn.  
The project uses a FastAPI backend for model inference and a Streamlit frontend for an interactive dashboard.

## Project Structure

```text
subscription-churn-project/
├── app/
│   ├── main.py
│   └── ui.py
├── data/
├── models/
├── notebooks/
├── src/
├── tests/
└── requirements.txt
```

## Setup Instructions

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

## Run the project

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

## Notes

- Make sure the FastAPI server is running before using the Streamlit dashboard.
- Monthly Charges is used as provided in the dataset [USD] and treated as a numeric billing amount.”
- Keep all paths relative to the project root.
- Store trained models inside the `models/` folder.

## Demo Features

- Subscriber churn prediction
- Churn probability output
- Risk label display
- Recommendation message
- Clean Streamlit dashboard UI
