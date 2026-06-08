"""Train multiple ML regression models for eligibility score prediction.

Trains and saves: Random Forest, XGBoost, SVR, Decision Tree, Ridge, KNN,
and the existing MLPRegressor. All use the same train/test split and scaler.
"""

import os
import sys
import joblib
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from sklearn.ensemble import RandomForestRegressor
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import Ridge
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.neural_network import MLPRegressor

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))
from utils.metrics import ground_truth_score

BASE_DIR = os.path.dirname(__file__)
DATA_PATH = os.path.join(BASE_DIR, '..', 'data', 'data.csv')
MODEL_DIR = BASE_DIR

FEATURES = ['Income', 'Household_Size', 'Work_Experience', 'Age', 'Number_of_Dependents']

MODEL_DEFS = {
    'mlp': {
        'name': 'MLP Regressor',
        'file': 'model.joblib',
        'scaler': 'scaler.joblib',
        'model': MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu', solver='adam', max_iter=500,
            random_state=42, early_stopping=True,
            validation_fraction=0.1, n_iter_no_change=20, verbose=False,
        ),
        'shared_scaler': True,
    },
    'rf': {
        'name': 'Random Forest',
        'file': 'rf_model.joblib',
        'model': RandomForestRegressor(
            n_estimators=200, max_depth=20, min_samples_leaf=4,
            random_state=42, n_jobs=-1,
        ),
    },
    'xgb': {
        'name': 'XGBoost',
        'file': 'xgb_model.joblib',
        'model': None,
    },
    'svr': {
        'name': 'SVR',
        'file': 'svr_model.joblib',
        'model': SVR(kernel='rbf', C=100, gamma='scale'),
    },
    'dt': {
        'name': 'Decision Tree',
        'file': 'dt_model.joblib',
        'model': DecisionTreeRegressor(max_depth=15, min_samples_leaf=5, random_state=42),
    },
    'ridge': {
        'name': 'Ridge',
        'file': 'ridge_model.joblib',
        'model': Ridge(alpha=1.0, random_state=42),
    },
    'knn': {
        'name': 'KNN',
        'file': 'knn_model.joblib',
        'model': KNeighborsRegressor(n_neighbors=15, weights='distance'),
    },
}


def _init_xgb():
    try:
        import xgboost as xgb
        return xgb.XGBRegressor(
            n_estimators=200, max_depth=8, learning_rate=0.1,
            subsample=0.8, colsample_bytree=0.8,
            random_state=42, verbosity=0,
        )
    except ImportError:
        return None


def train_all():
    print("=" * 60)
    print("Training all ML models for eligibility score prediction")
    print("=" * 60)

    df = pd.read_csv(DATA_PATH)
    X = df[FEATURES].values.astype(float)
    y = np.array([ground_truth_score(*row) for row in X])

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    joblib.dump(scaler, os.path.join(MODEL_DIR, 'scaler.joblib'))
    print(f"  Scaler saved to: {os.path.join(MODEL_DIR, 'scaler.joblib')}")

    results = []

    for key, cfg in MODEL_DEFS.items():
        print(f"\n--- Training {cfg['name']} ---")

        if key == 'xgb':
            model = _init_xgb()
            if model is None:
                print("  SKIPPED: xgboost not installed")
                results.append((cfg['name'], None, None, None, None))
                continue
        elif key == 'mlp':
            model = cfg['model']
        else:
            model = cfg['model']

        X_tr = X_train_scaled if cfg.get('shared_scaler', False) else X_train
        X_te = X_test_scaled if cfg.get('shared_scaler', False) else X_test

        model.fit(X_tr, y_train)

        y_pred = model.predict(X_te)
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)

        path = os.path.join(MODEL_DIR, cfg['file'])
        joblib.dump(model, path)

        print(f"  MAE : {mae:.4f}")
        print(f"  RMSE: {rmse:.4f}")
        print(f"  R2  : {r2:.4f}")
        print(f"  Saved to: {path}")

        results.append((cfg['name'], mae, rmse, r2, path))

    print("\n" + "=" * 60)
    print("Training Complete — Summary")
    print("=" * 60)
    print(f"{'Model':<20} {'MAE':<10} {'RMSE':<10} {'R2':<10}")
    print("-" * 50)
    for name, mae, rmse, r2, path in results:
        if mae is not None:
            print(f"{name:<20} {mae:<10.4f} {rmse:<10.4f} {r2:<10.4f}")
        else:
            print(f"{name:<20} {'SKIPPED':<10}")
    print("=" * 60)

    return results


if __name__ == '__main__':
    train_all()
