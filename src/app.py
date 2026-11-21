
from fastapi import FastAPI
from pydantic import BaseModel
import joblib
import numpy as np

# Initialize FastAPI
app = FastAPI(
    title="Iris Prediction API",
    description="Predict Iris flower species using Random Forest",
    version="1.0"
)

# Load trained model
model = joblib.load("models/iris_model.pkl")


# Define input data
class IrisInput(BaseModel):
    sepal_length: float
    sepal_width: float
    petal_length: float
    petal_width: float


# Home endpoint
@app.get("/")
def home():
    return {
        "message": "Iris Prediction API is running successfully!"
    }


# Prediction endpoint
@app.post("/predict")
def predict(data: IrisInput):

    features = np.array([[
        data.sepal_length,
        data.sepal_width,
        data.petal_length,
        data.petal_width
    ]])

    prediction = model.predict(features)[0]

    species = [
        "Setosa",
        "Versicolor",
        "Virginica"
    ]

    return {
        "predicted_class": int(prediction),
        "predicted_species": species[prediction]
    }
