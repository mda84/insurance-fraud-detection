import joblib

def load_model(model_path: str):
    """
    Loads a trained model from disk.
    """
    model = joblib.load(model_path)
    return model

def preprocess_input(data: dict):
    """
    Processes raw input data into a format suitable for model prediction.
    Expected keys: 'AgeOfVehicle', 'AgeOfPolicyHolder', 'NumberOfSuppliments', 'NumberOfCars', 'Year'.
    This function extracts the numeric part from strings (e.g., "3 years").
    """
    try:
        features = [
            float(data.get("AgeOfVehicle", "0").split()[0]),
            float(data.get("AgeOfPolicyHolder", "0").split()[0]),
            float(data.get("NumberOfSuppliments", "0").split()[0]) if isinstance(data.get("NumberOfSuppliments", "0"), str) and data.get("NumberOfSuppliments", "0").split()[0].replace('.','',1).isdigit() else 0,
            float(data.get("NumberOfCars", "0").split()[0]),
            float(data.get("Year", "0"))
        ]
    except Exception as e:
        raise ValueError("Invalid input data format") from e
    return features
