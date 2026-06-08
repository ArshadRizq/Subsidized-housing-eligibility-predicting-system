"""Integration test: batch evaluation matches documented notebook metrics."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import pandas as pd
from utils.metrics import ground_truth_score, confusion_matrix_manual
from fuzzy_logic.inference import fuzzy_mamdani, fuzzy_sugeno, THRESHOLD


DATA_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'data.csv')
FEATURE_NAMES = ['Income', 'Household_Size', 'Work_Experience', 'Age', 'Number_of_Dependents']
GT_THRESHOLD = 60  # ground truth uses >= 60


def load_and_evaluate():
    """Run the same batch eval as the app, return metrics dict."""
    df = pd.read_csv(DATA_PATH)
    X = df[FEATURE_NAMES].values.astype(float)

    gt_scores = np.array([ground_truth_score(*row) for row in X])
    gt_labels = np.array(['Layak' if s >= GT_THRESHOLD else 'Tidak Layak' for s in gt_scores])

    m_scores, m_labels = [], []
    s_scores, s_labels = [], []

    for row in X:
        inc, hh, we, age, dep = row
        ms, ml, _ = fuzzy_mamdani(inc, hh, we, age, dep)
        ss, sl, _ = fuzzy_sugeno(inc, hh, we, age, dep)
        m_scores.append(ms)
        m_labels.append(ml)
        s_scores.append(ss)
        s_labels.append(sl)

    m_scores = np.array(m_scores)
    s_scores = np.array(s_scores)
    m_labels = np.array(m_labels)
    s_labels = np.array(s_labels)

    acc_m = float(np.mean(m_labels == gt_labels))
    acc_s = float(np.mean(s_labels == gt_labels))
    mae_m = float(np.mean(np.abs(m_scores - gt_scores)))
    mae_s = float(np.mean(np.abs(s_scores - gt_scores)))
    rmse_m = float(np.sqrt(np.mean((m_scores - gt_scores) ** 2)))
    rmse_s = float(np.sqrt(np.mean((s_scores - gt_scores) ** 2)))
    agreement = float(np.mean(m_labels == s_labels))

    return {
        'acc_m': acc_m, 'acc_s': acc_s,
        'mae_m': mae_m, 'mae_s': mae_s,
        'rmse_m': rmse_m, 'rmse_s': rmse_s,
        'agreement': agreement,
    }


# Expected metrics from notebook (documented in AGENTS.md)
EXPECTED = {
    'acc_m': 0.7490,
    'acc_s': 0.7490,
    'mae_m': 13.1484,
    'mae_s': 13.0097,
    'rmse_m': 16.8397,
    'rmse_s': 16.6296,
    'agreement': 0.9994,
}


class TestBatchEvaluation:
    """Integration test: full 10k-row evaluation must match notebook metrics."""

    def test_mamdani_accuracy(self):
        """Mamdani accuracy should match notebook value (74.90%)."""
        result = load_and_evaluate()
        assert result['acc_m'] == pytest.approx(EXPECTED['acc_m'], abs=0.001)

    def test_sugeno_accuracy(self):
        """Sugeno accuracy should match notebook value (74.90%)."""
        result = load_and_evaluate()
        assert result['acc_s'] == pytest.approx(EXPECTED['acc_s'], abs=0.001)

    def test_mamdani_mae(self):
        """Mamdani MAE should match notebook value (13.1484)."""
        result = load_and_evaluate()
        assert result['mae_m'] == pytest.approx(EXPECTED['mae_m'], abs=0.01)

    def test_sugeno_mae(self):
        """Sugeno MAE should match notebook value (13.0097)."""
        result = load_and_evaluate()
        assert result['mae_s'] == pytest.approx(EXPECTED['mae_s'], abs=0.01)

    def test_mamdani_rmse(self):
        """Mamdani RMSE should match notebook value (16.8397)."""
        result = load_and_evaluate()
        assert result['rmse_m'] == pytest.approx(EXPECTED['rmse_m'], abs=0.01)

    def test_sugeno_rmse(self):
        """Sugeno RMSE should match notebook value (16.6296)."""
        result = load_and_evaluate()
        assert result['rmse_s'] == pytest.approx(EXPECTED['rmse_s'], abs=0.01)

    def test_mamdani_sugeno_agreement(self):
        """Mamdani-Sugeno label agreement should be 99.94%."""
        result = load_and_evaluate()
        assert result['agreement'] == pytest.approx(EXPECTED['agreement'], abs=0.001)


import pytest
