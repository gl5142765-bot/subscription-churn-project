import requests

url = "http://127.0.0.1:8000/predict_churn"

payload = {
    "AccountAge": 12,
    "MonthlyCharges": 450,
    "ViewingHoursPerWeek": 18,
    "ContentDownloadsPerMonth": 4,
    "SubscriptionType": "Premium",
    "ContentType": "Movies",
    "DeviceRegistered": "Mobile",
    "GenrePreference": "Action"
}

response = requests.post(url, json=payload)

print("Status code:", response.status_code)
print("Response JSON:", response.json())