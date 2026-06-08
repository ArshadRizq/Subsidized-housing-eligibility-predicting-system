"""Central registry for all trained ML models.

Lazy-loads models on first access and provides a consistent interface:
    models = load_all_models()
    score = models['Random Forest'](inc, hh, we, age, dep)
"""

import os
import joblib
import numpy as np

BASE_DIR = os.path.dirname(__file__)
FEATURE_NAMES = ['Income', 'Household_Size', 'Work_Experience', 'Age', 'Number_of_Dependents']

MODEL_CATALOG = {
    'MLP Regressor': {
        'file': 'model.joblib',
        'uses_scaler': True,
    },
    'Random Forest': {
        'file': 'rf_model.joblib',
        'uses_scaler': False,
    },
    'XGBoost': {
        'file': 'xgb_model.joblib',
        'uses_scaler': False,
    },
    'SVR': {
        'file': 'svr_model.joblib',
        'uses_scaler': True,
    },
    'Decision Tree': {
        'file': 'dt_model.joblib',
        'uses_scaler': False,
    },
    'Ridge': {
        'file': 'ridge_model.joblib',
        'uses_scaler': True,
    },
    'KNN': {
        'file': 'knn_model.joblib',
        'uses_scaler': True,
    },
}

_cache = {}


def _load_scaler():
    path = os.path.join(BASE_DIR, 'scaler.joblib')
    if not os.path.exists(path):
        return None
    return joblib.load(path)


def load_all_models():
    if _cache:
        return _cache

    scaler = _load_scaler()

    for name, cfg in MODEL_CATALOG.items():
        path = os.path.join(BASE_DIR, cfg['file'])
        if not os.path.exists(path):
            continue

        model = joblib.load(path)
        model_scaler = scaler if cfg['uses_scaler'] else None

        def _make_predict(m, s):
            def predict(inc, hh, we, age, dep):
                X = np.array([[inc, hh, we, age, dep]], dtype=float)
                if s is not None:
                    X = s.transform(X)
                score = float(m.predict(X)[0])
                score = max(0, min(100, score))
                label = 'Layak' if score >= 55 else 'Tidak Layak'
                return score, label
            return predict

        _cache[name] = _make_predict(model, model_scaler)

    return _cache


def predict_all(inc, hh, we, age, dep):
    """Run all models on a single input. Returns {name: (score, label)}."""
    models = load_all_models()
    return {name: fn(inc, hh, we, age, dep) for name, fn in models.items()}


def batch_predict_all(X_array, available_models=None):
    """Run all models on a 2D array (N x 5). Returns {name: scores_array, labels_array}.

    Much faster than predict_all per row — uses model.predict(vectorized).
    """
    if available_models is None:
        available_models = load_all_models()

    from sklearn.preprocessing import StandardScaler

    BASE_DIR_LOCAL = os.path.dirname(__file__)
    scaler_path = os.path.join(BASE_DIR_LOCAL, 'scaler.joblib')
    scaler = joblib.load(scaler_path) if os.path.exists(scaler_path) else None

    results = {}
    for name, cfg in MODEL_CATALOG.items():
        path = os.path.join(BASE_DIR_LOCAL, cfg['file'])
        if not os.path.exists(path):
            continue
        model = joblib.load(path)
        X_input = scaler.transform(X_array) if cfg['uses_scaler'] and scaler else X_array
        scores = np.clip(model.predict(X_input).flatten(), 0, 100)
        labels = np.array(['Layak' if s >= 55 else 'Tidak Layak' for s in scores])
        results[name] = (scores, labels)

    return results
