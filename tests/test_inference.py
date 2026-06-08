"""Tests for fuzzy_logic/inference.py — fuzzify, centroid, weighted_average,
Mamdani, and Sugeno inference."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fuzzy_logic.inference import (
    fuzzify, mamdani_inference, centroid, fuzzy_mamdani,
    sugeno_inference, weighted_average, fuzzy_sugeno,
    UNIVERSE, THRESHOLD, SUGENO_CONST,
)
from fuzzy_logic.rules import RULES


# ---------------------------------------------------------------------------
# fuzzify
# ---------------------------------------------------------------------------

class TestFuzzify:
    def test_returns_all_five_variables(self):
        """fuzzify should return dict with 5 variable keys."""
        fv = fuzzify(50000, 4, 10, 35, 2)
        assert set(fv.keys()) == {'Income', 'Household', 'WorkExp', 'Age', 'Dependents'}

    def test_each_variable_has_correct_linguistic_terms(self):
        """Each fuzzified variable should contain the right term names."""
        fv = fuzzify(50000, 4, 10, 35, 2)
        assert set(fv['Income'].keys()) == {'Rendah', 'Menengah_Bawah', 'Menengah', 'Tinggi'}
        assert set(fv['Household'].keys()) == {'Kecil', 'Sedang', 'Besar'}
        assert set(fv['WorkExp'].keys()) == {'Baru', 'Cukup', 'Berpengalaman'}
        assert set(fv['Age'].keys()) == {'Muda', 'Produktif', 'Senior'}
        assert set(fv['Dependents'].keys()) == {'Sedikit', 'Sedang', 'Banyak'}

    def test_all_memberships_are_floats_in_0_1(self):
        """Every membership value should be a float between 0 and 1."""
        fv = fuzzify(50000, 4, 10, 35, 2)
        for var, terms in fv.items():
            for term, val in terms.items():
                assert isinstance(val, (float, np.floating))
                assert 0.0 <= val <= 1.0, f"{var}.{term} = {val} out of range"

    def test_low_income_senario(self):
        """Very low income + large family should produce expected pattern."""
        fv = fuzzify(30000, 6, 2, 22, 4)
        assert fv['Income']['Rendah'] > 0.9
        assert fv['Household']['Besar'] > 0.5
        assert fv['Dependents']['Banyak'] > 0.5


# ---------------------------------------------------------------------------
# centroid
# ---------------------------------------------------------------------------

class TestCentroid:
    def test_normal_case(self):
        """Centroid of a uniform distribution over [0,100]."""
        agg = np.ones(len(UNIVERSE))
        c = centroid(agg)
        assert c == pytest.approx(50.0, abs=0.5)

    def test_single_peak_at_zero(self):
        """If only the first element is non-zero, centroid is near 0."""
        agg = np.zeros(len(UNIVERSE))
        agg[0] = 1.0
        c = centroid(agg)
        assert c == pytest.approx(0.0, abs=0.3)

    def test_single_peak_at_end(self):
        """If only the last element is non-zero, centroid is near 100."""
        agg = np.zeros(len(UNIVERSE))
        agg[-1] = 1.0
        c = centroid(agg)
        assert c == pytest.approx(100.0, abs=0.3)

    def test_all_zero_default(self):
        """When all firings are zero, centroid defaults to 50.0."""
        agg = np.zeros(len(UNIVERSE))
        assert centroid(agg) == 50.0

    def test_output_range(self):
        """Centroid should always be in [0, 100]."""
        for _ in range(20):
            agg = np.random.rand(len(UNIVERSE))
            c = centroid(agg)
            assert 0.0 <= c <= 100.0


# ---------------------------------------------------------------------------
# weighted_average
# ---------------------------------------------------------------------------

class TestWeightedAverage:
    def test_normal_case(self):
        """Weighted average of [(0.5, 30), (0.8, 60)]."""
        firings = [(0.5, 30.0), (0.8, 60.0)]
        expected = (0.5 * 30 + 0.8 * 60) / (0.5 + 0.8)
        assert weighted_average(firings) == pytest.approx(expected)

    def test_single_firing(self):
        """Single firing should return its constant."""
        assert weighted_average([(0.7, 85.0)]) == pytest.approx(85.0)

    def test_unequal_weights(self):
        """Higher weight should pull average toward its constant."""
        firings = [(0.1, 15.0), (0.9, 85.0)]
        result = weighted_average(firings)
        assert result > 50.0  # pulled toward 85

    def test_empty_firings_default(self):
        """Empty firing list should return 50.0."""
        assert weighted_average([]) == 50.0


# ---------------------------------------------------------------------------
# mamdani_inference
# ---------------------------------------------------------------------------

class TestMamdaniInference:
    def test_returns_array_of_universe_length(self):
        """Aggregated set should be same length as UNIVERSE."""
        fv = fuzzify(50000, 4, 10, 35, 2)
        agg = mamdani_inference(fv)
        assert len(agg) == len(UNIVERSE)

    def test_output_between_0_and_1(self):
        """All aggregated values should be in [0, 1]."""
        fv = fuzzify(50000, 4, 10, 35, 2)
        agg = mamdani_inference(fv)
        assert np.all(agg >= 0.0) and np.all(agg <= 1.0)

    def test_extreme_rich_single_person_should_have_low_eligibility(self):
        """High income, small household, experienced → should be 'Tidak_Layak'."""
        fv = fuzzify(900000, 1, 30, 45, 0)
        agg = mamdani_inference(fv)
        score = centroid(agg)
        assert score < THRESHOLD


# ---------------------------------------------------------------------------
# fuzzy_mamdani (end-to-end)
# ---------------------------------------------------------------------------

class TestFuzzyMamdani:
    def test_returns_tuple_of_three(self):
        """fuzzy_mamdani returns (score, label, agg_array)."""
        result = fuzzy_mamdani(50000, 4, 10, 35, 2)
        assert len(result) == 3

    def test_score_between_0_and_100(self):
        """Mamdani score should always be in [0, 100]."""
        for params in [
            (50000, 4, 10, 35, 2),
            (900000, 1, 30, 60, 0),
            (0, 7, 0, 18, 5),
            (INCOME_CAP, 3, 5, 30, 2),
        ]:
            score, label, _ = fuzzy_mamdani(*params)
            assert 0.0 <= score <= 100.0, f"score={score} out of [0,100] for {params}"

    def test_label_based_on_threshold(self):
        """Label should be 'Layak' when score >= THRESHOLD and 'Tidak Layak' otherwise."""
        score, label, _ = fuzzy_mamdani(50000, 4, 10, 35, 2)
        expected = 'Layak' if score >= THRESHOLD else 'Tidak Layak'
        assert label == expected

    def test_low_income_large_family_tends_layak(self):
        """Low income + large family tends to be eligible (Layak)."""
        score, label, _ = fuzzy_mamdani(10000, 7, 5, 30, 5)
        assert label == 'Layak', f"Expected Layak, got {label} (score={score:.2f})"
        assert score >= 80.0, f"Score {score:.2f} too low for extreme poverty case"

    def test_high_income_tends_not_layak(self):
        """High income + small family tends not to be eligible."""
        score, label, _ = fuzzy_mamdani(900000, 2, 15, 40, 1)
        assert label == 'Tidak Layak', f"Expected Tidak Layak, got {label} (score={score:.2f})"


# ---------------------------------------------------------------------------
# sugeno_inference
# ---------------------------------------------------------------------------

class TestSugenoInference:
    def test_returns_list_of_tuples(self):
        """sugeno_inference returns list of (strength, constant) tuples."""
        fv = fuzzify(50000, 4, 10, 35, 2)
        firings = sugeno_inference(fv)
        assert isinstance(firings, list)
        if firings:
            s, v = firings[0]
            assert 0.0 < s <= 1.0
            assert v in SUGENO_CONST.values()

    def test_extreme_rich_few_firings(self):
        """High income should lead to very few (or zero) active rules."""
        fv = fuzzify(900000, 2, 15, 40, 1)
        firings = sugeno_inference(fv)
        # Most rules won't fire for high income
        assert len(firings) < len(RULES)


# ---------------------------------------------------------------------------
# fuzzy_sugeno (end-to-end)
# ---------------------------------------------------------------------------

class TestFuzzySugeno:
    def test_returns_tuple_of_three(self):
        """fuzzy_sugeno returns (score, label, firings_list)."""
        result = fuzzy_sugeno(50000, 4, 10, 35, 2)
        assert len(result) == 3

    def test_score_between_0_and_100(self):
        """Sugeno score should always be in [0, 100]."""
        for params in [
            (50000, 4, 10, 35, 2),
            (900000, 1, 30, 60, 0),
            (0, 7, 0, 18, 5),
            (INCOME_CAP, 3, 5, 30, 2),
        ]:
            score, label, _ = fuzzy_sugeno(*params)
            assert 0.0 <= score <= 100.0, f"score={score} out of [0,100] for {params}"

    def test_label_based_on_threshold(self):
        """Label should be 'Layak' when score >= THRESHOLD and 'Tidak Layak' otherwise."""
        score, label, _ = fuzzy_sugeno(50000, 4, 10, 35, 2)
        expected = 'Layak' if score >= THRESHOLD else 'Tidak Layak'
        assert label == expected

    def test_sugeno_scores_reasonable_range(self):
        """Sugeno scores should fall within [15, 85] (range of constants)."""
        for params in [
            (50000, 4, 10, 35, 2),
            (900000, 1, 30, 60, 0),
            (0, 7, 0, 18, 5),
        ]:
            score, _, _ = fuzzy_sugeno(*params)
            assert 15.0 <= score <= 85.0, f"score={score} out of [15,85] for {params}"


# ---------------------------------------------------------------------------
# Mamdani vs Sugeno agreement
# ---------------------------------------------------------------------------

class TestMamdaniVsSugeno:
    def test_agree_on_obvious_cases(self):
        """Mamdani and Sugeno should agree on obvious extreme cases."""
        # Very poor, large family → both should say Layak
        m_score, m_label, _ = fuzzy_mamdani(30000, 6, 1, 25, 4)
        s_score, s_label, _ = fuzzy_sugeno(30000, 6, 1, 25, 4)
        assert m_label == s_label, f"Mamdani={m_label}, Sugeno={s_label}"

        # Very rich → both should say Tidak Layak
        m_score, m_label, _ = fuzzy_mamdani(900000, 2, 15, 50, 1)
        s_score, s_label, _ = fuzzy_sugeno(900000, 2, 15, 50, 1)
        assert m_label == s_label, f"Mamdani={m_label}, Sugeno={s_label}"


# Helper import
from fuzzy_logic.core import INCOME_CAP
import pytest
