"""Tests for utils/metrics.py — ground_truth_score and confusion_matrix_manual."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from utils.metrics import ground_truth_score, confusion_matrix_manual


# ---------------------------------------------------------------------------
# ground_truth_score
# ---------------------------------------------------------------------------

class TestGroundTruthScore:
    def test_returns_integer_between_0_and_100(self):
        """Score should always be an int in [0, 100]."""
        for params in [
            (0, 1, 0, 18, 0),
            (1000000, 7, 50, 70, 5),
            (70000, 3, 10, 35, 2),
        ]:
            score = ground_truth_score(*params)
            assert isinstance(score, int)
            assert 0 <= score <= 100

    def test_max_score_poor_large_family(self):
        """Poorest, largest family, mid-career, productive age, many dependents -> max score."""
        # inc <= 70000 => +40, hh >= 5 => +20, 2 <= we <= 20 => +20,
        # 21 <= age <= 55 => +15, dep >= 3 => +5 => total 100
        score = ground_truth_score(30000, 6, 10, 35, 4)
        assert score == 100

    def test_min_score_rich_single(self):
        """Richest, smallest family, no experience, young, no dependents -> min score."""
        # inc > 400000 => +10, hh < 3 => +5, we < 2 => +5,
        # age <= 20 => +5, dep < 1 => +0 => total 25
        # Wait, let me re-read the function...
        # inc <= 70000: +40
        # inc <= 150000: +25 (but not <= 70000)
        # inc <= 400000: +10 (but not <= 150000)
        # else: +0? Let me check...
        # Actually: if inc <= 70000: +40; elif inc <= 150000: +25; elif inc <= 400000: +10
        # No else for income — so income > 400000 gets +0?
        # Wait, looking at the code again:
        # if inc <= 70000: score += 40
        # elif inc <= 150000: score += 25
        # elif inc <= 400000: score += 10
        # So if inc > 400000, none of these fire => score += 0
        #
        # hh >= 5: +20; hh >= 3: +15; else: +5
        # we: 2 <= we <= 20: +20; we > 20: +10; else: +5
        # age: 21 <= age <= 55: +15; 56 <= age <= 65: +10; else: +5
        # dep >= 3: +5; dep >= 1: +3; else: +0
        #
        # Min: inc=1_000_000 (0), hh=1 (+5), we=0 (+5), age=18 (+5), dep=0 (+0) = 15
        score = ground_truth_score(1_000_000, 1, 0, 18, 0)
        assert score == 15

    def test_poverty_line_income(self):
        """Income <= 70000 contributes +40."""
        score_70k = ground_truth_score(70000, 1, 0, 18, 0)
        score_71k = ground_truth_score(71000, 1, 0, 18, 0)
        assert score_70k == 40 + 5 + 5 + 5 + 0  # 55
        assert score_71k == 25 + 5 + 5 + 5 + 0  # 40

    def test_low_income_threshold(self):
        """Income between 70001 and 150000 contributes +25."""
        score = ground_truth_score(100000, 1, 0, 18, 0)
        assert score == 25 + 5 + 5 + 5  # = 40

    def test_mid_income_threshold(self):
        """Income between 150001 and 400000 contributes +10."""
        score = ground_truth_score(300000, 1, 0, 18, 0)
        assert score == 10 + 5 + 5 + 5  # = 25

    def test_high_income_zero_contribution(self):
        """Income > 400000 contributes 0 from income component."""
        score = ground_truth_score(500000, 1, 0, 18, 0)
        assert score == 0 + 5 + 5 + 5  # = 15

    def test_household_large(self):
        """Household >= 5 contributes +20."""
        assert ground_truth_score(500000, 5, 0, 18, 0) == 0 + 20 + 5 + 5

    def test_household_medium(self):
        """Household >= 3 and < 5 contributes +15."""
        for hh in [3, 4]:
            score = ground_truth_score(500000, hh, 0, 18, 0)
            assert score == 0 + 15 + 5 + 5

    def test_household_small(self):
        """Household < 3 contributes +5."""
        for hh in [1, 2]:
            score = ground_truth_score(500000, hh, 0, 18, 0)
            assert score == 0 + 5 + 5 + 5

    def test_workexp_mid_career(self):
        """Work experience between 2 and 20 inclusive contributes +20."""
        for we in [2, 10, 20]:
            score = ground_truth_score(500000, 1, we, 18, 0)
            assert score == 0 + 5 + 20 + 5

    def test_workexp_senior(self):
        """Work experience > 20 contributes +10."""
        score = ground_truth_score(500000, 1, 25, 18, 0)
        assert score == 0 + 5 + 10 + 5

    def test_workexp_junior(self):
        """Work experience < 2 contributes +5."""
        score = ground_truth_score(500000, 1, 0, 18, 0)
        assert score == 0 + 5 + 5 + 5

    def test_age_productive(self):
        """Age between 21 and 55 inclusive contributes +15."""
        for age in [21, 35, 55]:
            score = ground_truth_score(500000, 1, 0, age, 0)
            assert score == 0 + 5 + 5 + 15

    def test_age_senior(self):
        """Age between 56 and 65 inclusive contributes +10."""
        for age in [56, 60, 65]:
            score = ground_truth_score(500000, 1, 0, age, 0)
            assert score == 0 + 5 + 5 + 10

    def test_age_young(self):
        """Age <= 20 contributes +5."""
        for age in [18, 19, 20]:
            score = ground_truth_score(500000, 1, 0, age, 0)
            assert score == 0 + 5 + 5 + 5

    def test_age_elderly(self):
        """Age > 65 contributes +5."""
        for age in [66, 70]:
            score = ground_truth_score(500000, 1, 0, age, 0)
            assert score == 0 + 5 + 5 + 5

    def test_dependents_many(self):
        """Dependents >= 3 contributes +5."""
        for dep in [3, 4, 5]:
            score = ground_truth_score(500000, 1, 0, 18, dep)
            assert score == 0 + 5 + 5 + 5 + 5

    def test_dependents_some(self):
        """Dependents >= 1 and < 3 contributes +3."""
        for dep in [1, 2]:
            score = ground_truth_score(500000, 1, 0, 18, dep)
            assert score == 0 + 5 + 5 + 5 + 3

    def test_dependents_none(self):
        """No dependents contributes +0."""
        score = ground_truth_score(500000, 1, 0, 18, 0)
        assert score == 0 + 5 + 5 + 5 + 0

    def test_boundary_income_70000(self):
        """Exactly 70000 gets the +40 bracket."""
        assert ground_truth_score(70000, 1, 0, 18, 0) == 55

    def test_boundary_income_150000(self):
        """Exactly 150000 gets the +25 bracket."""
        assert ground_truth_score(150000, 1, 0, 18, 0) == 40

    def test_boundary_income_400000(self):
        """Exactly 400000 gets the +10 bracket."""
        assert ground_truth_score(400000, 1, 0, 18, 0) == 25

    def test_boundary_age_21(self):
        """Exactly 21 is in productive age bracket."""
        assert ground_truth_score(500000, 1, 0, 21, 0) == 15 + 5 + 5  # age + hh + we + inc = 15 + 5 + 5 + 0

    def test_boundary_age_55(self):
        """Exactly 55 is in productive age bracket."""
        assert ground_truth_score(500000, 1, 0, 55, 0) == 15 + 5 + 5

    def test_boundary_age_56(self):
        """Exactly 56 is in senior age bracket."""
        assert ground_truth_score(500000, 1, 0, 56, 0) == 10 + 5 + 5

    def test_boundary_hh_3(self):
        """Exactly 3 household members => medium bracket."""
        assert ground_truth_score(500000, 3, 0, 18, 0) == 15 + 5 + 5

    def test_boundary_hh_5(self):
        """Exactly 5 household members => large bracket."""
        assert ground_truth_score(500000, 5, 0, 18, 0) == 20 + 5 + 5


# ---------------------------------------------------------------------------
# confusion_matrix_manual
# ---------------------------------------------------------------------------

class TestConfusionMatrix:
    def test_perfect_predictions(self):
        """Perfect predictions produce a diagonal matrix."""
        gt = ['Layak', 'Tidak Layak', 'Layak', 'Tidak Layak']
        pred = ['Layak', 'Tidak Layak', 'Layak', 'Tidak Layak']
        cm, classes = confusion_matrix_manual(gt, pred)
        assert classes == ['Layak', 'Tidak Layak']
        assert cm[0][0] == 2  # TP Layak
        assert cm[1][1] == 2  # TN Tidak Layak
        assert cm[0][1] == 0  # FP
        assert cm[1][0] == 0  # FN

    def test_all_wrong(self):
        """All predictions wrong produce anti-diagonal."""
        gt = ['Layak', 'Layak']
        pred = ['Tidak Layak', 'Tidak Layak']
        cm, _ = confusion_matrix_manual(gt, pred)
        assert cm[0][1] == 2  # All predicted Tidak Layak but are Layak

    def test_mixed_results(self):
        """Mixed predictions produce correct counts."""
        gt = ['Layak', 'Layak', 'Tidak Layak', 'Tidak Layak']
        pred = ['Layak', 'Tidak Layak', 'Layak', 'Tidak Layak']
        cm, _ = confusion_matrix_manual(gt, pred)
        assert cm[0][0] == 1  # TP
        assert cm[0][1] == 1  # FN
        assert cm[1][0] == 1  # FP
        assert cm[1][1] == 1  # TN

    def test_all_same_class(self):
        """All ground truth same class."""
        gt = ['Layak', 'Layak', 'Layak']
        pred = ['Layak', 'Layak', 'Layak']
        cm, _ = confusion_matrix_manual(gt, pred)
        assert cm[0][0] == 3
        assert cm[0][1] == 0
        assert cm[1][0] == 0
        assert cm[1][1] == 0

    def test_empty_inputs(self):
        """Empty lists produce zeros matrix."""
        cm, classes = confusion_matrix_manual([], [])
        assert np.array_equal(cm, np.zeros((2, 2), dtype=int))

    def test_class_order(self):
        """Classes should be ['Layak', 'Tidak Layak'] in that order."""
        _, classes = confusion_matrix_manual(['Layak'], ['Tidak Layak'])
        assert classes == ['Layak', 'Tidak Layak']
