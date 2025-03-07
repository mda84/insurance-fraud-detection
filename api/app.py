from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import numpy as np
from src.utils import load_model, preprocess_input
from src.llm_integration import generate_explanation

app = FastAPI(title="Vehicle Claim Fraud Detection API")

# Load the trained model (ensure that the model was saved in ../model/model_xgb.pkl)
model_path = "../model/model_xgb.pkl"
try:
    model = load_model(model_path)
except Exception as e:
    raise RuntimeError(f"Failed to load model from {model_path}") from e

class ClaimData(BaseModel):
    AgeOfVehicle: str
    AgeOfPolicyHolder: str
    NumberOfSuppliments: str
    NumberOfCars: str
    Year: str
    PoliceReportFiled: str = None
    WitnessPresent: str = None

@app.get("/")
def read_root():
    return {"message": "Welcome to the Vehicle Claim Fraud Detection API."}

@app.post("/predict")
def predict_fraud(claim: ClaimData):
    try:
        # Preprocess the input to get feature values.
        features = preprocess_input(claim.dict())
        features = np.array(features).reshape(1, -1)
        fraud_prob = model.predict_proba(features)[0][1]
        prediction = int(fraud_prob > 0.5)
        return {
            "fraud_probability": fraud_prob,
            "fraud_prediction": prediction
        }
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/explain")
def explain_claim(claim: ClaimData):
    try:
        explanation = generate_explanation(claim.dict())
        return {"explanation": explanation}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
