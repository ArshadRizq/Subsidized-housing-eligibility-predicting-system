"""Shared evaluation utilities — ground truth scoring and confusion matrices."""

import numpy as np


def ground_truth_score(inc, hh, we, age, dep):
    """Compute the ground truth eligibility score (0-100) based on subsidy criteria.

    Same logic as notebook section 3.
    """
    score = 0
    if inc <= 70000:
        score += 40
    elif inc <= 150000:
        score += 25
    elif inc <= 400000:
        score += 10
    if hh >= 5:
        score += 20
    elif hh >= 3:
        score += 15
    else:
        score += 5
    if 2 <= we <= 20:
        score += 20
    elif we > 20:
        score += 10
    else:
        score += 5
    if 21 <= age <= 55:
        score += 15
    elif 56 <= age <= 65:
        score += 10
    else:
        score += 5
    if dep >= 3:
        score += 5
    elif dep >= 1:
        score += 3
    return score


def confusion_matrix_manual(gt, pred):
    """Build 2x2 confusion matrix from ground truth and prediction labels."""
    classes = ['Layak', 'Tidak Layak']
    cm = np.zeros((2, 2), dtype=int)
    idx = {c: i for i, c in enumerate(classes)}
    for g, p in zip(gt, pred):
        cm[idx[g]][idx[p]] += 1
    return cm, classes
