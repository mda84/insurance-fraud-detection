import pandas as pd
import numpy as np
import re

def extract_number(x):
    """
    Extract the first numeric value from a string using regex.
    Returns the float value if found; otherwise returns np.nan.
    """
    try:
        s = str(x)
        match = re.search(r"[-+]?\d*\.\d+|\d+", s)
        if match:
            return float(match.group())
        else:
            return np.nan
    except Exception:
        return np.nan

def create_features_and_target(data: pd.DataFrame):
    """
    Processes the raw dataset to extract features and create a target variable for fraud detection.
    
    If the dataset does not have a "Fraud" column, a synthetic target is created.
    For demonstration, a claim is marked as fraudulent if both "PoliceReportFiled" and "WitnessPresent" are "No".
    
    The function processes the following columns: 
      - "AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars", and "Year"
    by extracting numeric values and filling missing values with the column mean.
    """
    df = data.copy()
    
    # Create synthetic target if "Fraud" column does not exist.
    if "Fraud" not in df.columns:
        df["Fraud"] = ((df["PoliceReportFiled"].str.lower() == "no") & 
                       (df["WitnessPresent"].str.lower() == "no")).astype(int)
    
    # Select features for training.
    features = df[["AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars", "Year"]].copy()
    
    # Convert each feature column to numeric using extract_number.
    for col in ["AgeOfVehicle", "AgeOfPolicyHolder", "NumberOfSuppliments", "NumberOfCars", "Year"]:
        features[col] = features[col].apply(extract_number)
    
    # Fill missing values with the mean of each column.
    features = features.fillna(features.mean())
    
    target = df["Fraud"]
    
    return features, target

if __name__ == "__main__":
    data = pd.read_csv("../data/fraud_oracle.csv")
    features, target = create_features_and_target(data)
    print("Features preview:")
    print(features.head())
    print("\nTarget preview:")
    print(target.head())
