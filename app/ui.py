import streamlit as st
import requests

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_PATH = BASE_DIR / "models" / "churn_model.pkl"

API_BASE_URL = "https://subscription-churn-project.onrender.com"
PREDICT_URL = f"{API_BASE_URL}/predict_churn"

st.set_page_config(
    page_title="Streaming membership churn dashboard",
    page_icon="📺",
    layout="wide"
)

st.title("📺 Streaming membership churn dashboard")
st.caption(
    "This is a streaming membership churn prediction demo. "
    "It uses a subscription churn dataset from Kaggle, adapted to simulate streaming behavior."
)

top_col1, top_col2 = st.columns([2, 1])

with top_col1:
    st.markdown(
        """
        Use this demo to estimate the probability that a subscriber may churn.
        Enter profile, subscription, and engagement details, then click **Predict churn**
        to view churn probability, risk level, and a simple retention recommendation.
        """
    )

with top_col2:
    st.markdown(
        """
        <div style="
            background-color:#111827;
            padding:18px;
            border-radius:12px;
            border:1px solid #2d3748;
        ">
            <h4 style="margin-top:0; margin-bottom:10px;">Insights</h4>
            <p style="margin-bottom:0; color:#d1d5db;">
                In this dataset, subscribers with very low recent engagement and shorter
                account age tend to show higher churn risk.
            </p>
        </div>
        """,
        unsafe_allow_html=True
    )

st.subheader("Prediction inputs")

col1, col2, col3 = st.columns(3)

with col1:
    st.markdown("### Subscriber profile")
    account_age = st.number_input("Account age (months)", min_value=0, max_value=120, value=12, step=1)
    device_registered = st.selectbox("Registered device", ["Mobile", "Tablet", "TV", "Laptop"])
    genre_preference = st.selectbox("Genre preference", ["Action", "Comedy", "Drama", "Horror", "Sci-Fi", "Romance"])

with col2:
    st.markdown("### Subscription details")
    subscription_type = st.selectbox("Subscription type", ["Basic", "Standard", "Premium"])
    monthly_charges = st.number_input("Monthly charges", min_value=0.0, max_value=5000.0, value=499.0, step=1.0)
    content_type = st.selectbox("Preferred Content Type", ["Movies", "TV Shows", "Both"])

with col3:
    st.markdown("### Engagement")
    viewing_hours_per_week = st.number_input("Viewing hours per week", min_value=0.0, max_value=168.0, value=12.0, step=1.0)
    content_downloads_per_month = st.number_input("Content downloads per month", min_value=0, max_value=100, value=2, step=1)

st.markdown("")
predict_clicked = st.button("Predict churn", use_container_width=True)

if predict_clicked:
    payload = {
        "AccountAge": account_age,
        "MonthlyCharges": monthly_charges,
        "ViewingHoursPerWeek": viewing_hours_per_week,
        "ContentDownloadsPerMonth": content_downloads_per_month,
        "SubscriptionType": subscription_type,
        "ContentType": content_type,
        "DeviceRegistered": device_registered,
        "GenrePreference": genre_preference
    }

    try:
        response = requests.post(PREDICT_URL, json=payload, timeout=30)

        if response.status_code == 200:
            result = response.json()
            probability = result["probability"]
            percentage = probability * 100
            risk_band = result["risk_band"].lower()

            st.success("Prediction generated successfully.")

            metric_col1, metric_col2, metric_col3 = st.columns([1.2, 1, 1.2])

            with metric_col1:
                st.metric("Churn probability", f"{probability:.2f}", f"{percentage:.1f}%")

            with metric_col2:
                st.metric("Risk label", risk_band.upper())

            with metric_col3:
                if risk_band == "high":
                    st.badge("High risk", color="red")
                elif risk_band == "medium":
                    st.badge("Medium risk", color="orange")
                else:
                    st.badge("Low risk", color="green")

            if risk_band == "high":
                st.error(
                    "High risk – user is likely to cancel; consider personalized recommendations or special offers.",
                    icon=":material/warning:"
                )
            elif risk_band == "medium":
                st.warning(
                    "Medium risk – user may churn; consider engagement nudges, reminders, or targeted content recommendations.",
                    icon=":material/trending_up:"
                )
            else:
                st.success(
                    "Low risk – user is likely to stay engaged with the platform.",
                    icon=":material/check_circle:"
                )

            with st.expander("Show raw API response (debug)", icon=":material/code:"):
                st.json(result)

        else:
            st.error("Prediction failed.")
            with st.expander("Show error details", icon=":material/error:"):
                try:
                    st.json(response.json())
                except Exception:
                    st.write(response.text)

    except requests.exceptions.ConnectionError:
        st.error(
            f"Could not connect to FastAPI server. Make sure the backend is running on {API_BASE_URL}",
            icon=":material/cloud_off:"
        )

    except requests.exceptions.Timeout:
        st.error(
            "The request timed out. Render free instances can take time to wake up. Please try again.",
            icon=":material/schedule:"
        )
