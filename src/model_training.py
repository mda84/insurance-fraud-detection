import os
import pandas as pd
import joblib
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
from feature_engineering import create_features_and_target

def train_model(data: pd.DataFrame):
    """
    Splits the data, trains an XGBoost model, and returns the trained model.
    """
    features, target = create_features_and_target(data)
    X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)
    model = xgb.XGBClassifier(use_label_encoder=False, eval_metric="logloss")
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    print("Classification Report:")
    print(classification_report(y_test, preds))
    return model

def save_model(model, filename: str = "../model/model_xgb.pkl"):
    """
    Saves the trained model to disk.
    """
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    joblib.dump(model, filename)
    print(f"Model saved as {filename}")

if __name__ == "__main__":
    data_path = os.path.join("..", "data", "fraud_oracle.csv")
    data = pd.read_csv(data_path)
    model = train_model(data)
    save_model(model)
