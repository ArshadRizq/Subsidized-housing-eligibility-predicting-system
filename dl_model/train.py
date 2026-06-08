import os
import sys
import joblib
import numpy as np
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.metrics import ground_truth_score

DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
MODEL_PATH = os.path.join(os.path.dirname(__file__), 'model.joblib')
SCALER_PATH = os.path.join(os.path.dirname(__file__), 'scaler.joblib')


def train():
    df = pd.read_csv(DATA_PATH)
    features = ['Income', 'Household_Size', 'Work_Experience', 'Age', 'Number_of_Dependents']
    X = df[features].values.astype(float)
    y = np.array([ground_truth_score(*row) for row in X])

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    model = MLPRegressor(
        hidden_layer_sizes=(64, 32, 16),
        activation='relu',
        solver='adam',
        max_iter=500,
        random_state=42,
        early_stopping=True,
        validation_fraction=0.1,
        n_iter_no_change=20,
        verbose=False,
    )

    model.fit(X_train_scaled, y_train)

    y_pred = model.predict(X_test_scaled)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"DL Model Training Complete")
    print(f"  Test MAE : {mae:.4f}")
    print(f"  Test RMSE: {rmse:.4f}")
    print(f"  Test R2  : {r2:.4f}")
    print(f"  Epochs   : {model.n_iter_}")

    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)
    print(f"  Model saved to: {MODEL_PATH}")
    print(f"  Scaler saved to: {SCALER_PATH}")

    return model, scaler


def predict_dl(inc, hh, we, age, dep, model=None, scaler=None):
    if model is None:
        model = joblib.load(MODEL_PATH)
    if scaler is None:
        scaler = joblib.load(SCALER_PATH)
    X = np.array([[inc, hh, we, age, dep]], dtype=float)
    X_scaled = scaler.transform(X)
    score = float(model.predict(X_scaled)[0])
    score = max(0, min(100, score))
    label = 'Layak' if score >= 55 else 'Tidak Layak'
    return score, label


if __name__ == '__main__':
    train()
