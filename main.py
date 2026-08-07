import joblib
import pandas as pd
from pathlib import Path

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from pydantic import BaseModel, Field
from typing import Literal

# Load model
model = joblib.load("Mental_Health_Model.pkl")

# Countries that get their own bucket
top_countries = [
    "Other",
    "India",
    "USA",
    "Canada",
    "Australia",
    "UK",
    "Germany",
    "Mexico",
    "Turkey",
    "France",
]

app = FastAPI()

# Enable CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Path to HTML file
BASE_DIR = Path(__file__).resolve().parent


# Input model
class StudentData(BaseModel):
    age: int = Field(..., ge=10, le=100)
    gender: Literal["Male", "Female"]
    country: str
    academic_level: Literal["Undergraduate", "Graduate", "High School"]
    most_used_platform: Literal[
        "Facebook",
        "LinkedIn",
        "Instagram",
        "Snapchat",
        "Twitter",
        "YouTube",
        "TikTok",
        "LINE",
        "KakaoTalk",
        "VKontakte",
        "WhatsApp",
        "WeChat",
    ]
    purpose_of_use: Literal[
        "Networking",
        "Education",
        "Entertainment",
        "News",
    ]
    avg_daily_usage_hours: float = Field(..., ge=0, le=24)
    daily_unlocks: int = Field(..., ge=0)
    study_hours: float = Field(..., ge=0, le=24)
    physical_activity_hours: float = Field(..., ge=0, le=24)
    sleep_hours_per_night: float = Field(..., ge=0, le=24)
    stress_level: Literal["Low", "Medium", "High", "Very High"]


# Output model
class PredictionResponse(BaseModel):
    predicted_mental_health_score: float


# Serve HTML page
@app.get("/")
def home():
    html_file = BASE_DIR / "Mental Health.html"
    return FileResponse(html_file)


# Prediction endpoint
@app.post("/predict", response_model=PredictionResponse)
def predict(data: StudentData):

    grouped_country = (
        data.country if data.country in top_countries else "Other"
    )

    input_row = pd.DataFrame([{
        "Age": data.age,
        "Gender": data.gender,
        "Academic_Level": data.academic_level,
        "Most_Used_Platform": data.most_used_platform,
        "Purpose_Of_Use": data.purpose_of_use,
        "Avg_Daily_Usage_Hours": data.avg_daily_usage_hours,
        "Daily_Unlocks": data.daily_unlocks,
        "Study_Hours": data.study_hours,
        "Physical_Activity_Hours": data.physical_activity_hours,
        "Sleep_Hours_Per_Night": data.sleep_hours_per_night,
        "Stress_Level": data.stress_level,
        "Grouped_Countries": grouped_country,
    }])

    prediction = model.predict(input_row)[0]

    return PredictionResponse(
        predicted_mental_health_score=round(float(prediction), 2)
    )
