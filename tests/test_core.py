"""Tests for fuzzy_logic/core.py — trimf, trapmf, and membership functions."""

import sys
import os
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

from fuzzy_logic.core import trimf, trapmf, INCOME_CAP
from fuzzy_logic.core import mf_income, mf_household, mf_workexp, mf_age, mf_dependents, mf_output


# ---------------------------------------------------------------------------
# trimf
# ---------------------------------------------------------------------------

class TestTrimf:
    def test_vertex_returns_one(self):
        """At x == b, trimf should return exactly 1.0."""
        assert trimf(5, 0, 5, 10) == 1.0

    def test_below_range_returns_zero(self):
        """At x < a, trimf should return 0.0."""
        assert trimf(-1, 0, 5, 10) == 0.0

    def test_above_range_returns_zero(self):
        """At x > c, trimf should return 0.0."""
        assert trimf(11, 0, 5, 10) == 0.0

    def test_left_edge_returns_zero(self):
        """At x == a, trimf should return 0.0."""
        assert trimf(0, 0, 5, 10) == 0.0

    def test_right_edge_returns_zero(self):
        """At x == c, trimf should return 0.0."""
        assert trimf(10, 0, 5, 10) == 0.0

    def test_ascending_slope(self):
        """On the left of b, trimf should ramp up linearly."""
        result = trimf(2, 0, 5, 10)
        expected = (2 - 0) / (5 - 0)
        assert result == pytest.approx(expected)

    def test_descending_slope(self):
        """On the right of b, trimf should ramp down linearly."""
        result = trimf(8, 0, 5, 10)
        expected = (10 - 8) / (10 - 5)
        assert result == pytest.approx(expected)

    def test_midpoint_ascending(self):
        """Halfway between a and b gives 0.5."""
        assert trimf(2.5, 0, 5, 10) == pytest.approx(0.5)

    def test_midpoint_descending(self):
        """Halfway between b and c gives 0.5."""
        assert trimf(7.5, 0, 5, 10) == pytest.approx(0.5)

    def test_symmetric(self):
        """trimf(a+d) should equal trimf(c-d) for any offset d."""
        d = 2
        assert trimf(0 + d, 0, 5, 10) == pytest.approx(trimf(10 - d, 0, 5, 10))


# ---------------------------------------------------------------------------
# trapmf
# ---------------------------------------------------------------------------

class TestTrapmf:
    def test_plateau_returns_one(self):
        """Between b and c, trapmf should return exactly 1.0."""
        assert trapmf(3, 0, 2, 5, 7) == 1.0
        assert trapmf(4, 0, 2, 5, 7) == 1.0

    def test_below_range_returns_zero(self):
        """At x < a, trapmf should return 0.0."""
        assert trapmf(-1, 0, 2, 5, 7) == 0.0

    def test_above_range_returns_zero(self):
        """At x > d, trapmf should return 0.0."""
        assert trapmf(8, 0, 2, 5, 7) == 0.0

    def test_left_edge_returns_zero(self):
        """At x == a, trapmf should return 0.0."""
        assert trapmf(0, 0, 2, 5, 7) == 0.0

    def test_right_edge_returns_zero(self):
        """At x == d, trapmf should return 0.0."""
        assert trapmf(7, 0, 2, 5, 7) == 0.0

    def test_ascending_slope(self):
        """On the left of b, trapmf should ramp up linearly."""
        result = trapmf(1, 0, 2, 5, 7)
        expected = (1 - 0) / (2 - 0)
        assert result == pytest.approx(expected)

    def test_descending_slope(self):
        """On the right of c, trapmf should ramp down linearly."""
        result = trapmf(6, 0, 2, 5, 7)
        expected = (7 - 6) / (7 - 5)
        assert result == pytest.approx(expected)

    def test_vertical_ramp_left_b_eq_a(self):
        """When b == a, left slope is vertical: x == a/b gives 1.0."""
        assert trapmf(2, 2, 2, 5, 7) == 1.0  # x == a == b → instant rise

    def test_vertical_ramp_right_d_eq_c(self):
        """When d == c, right slope is vertical: x == c/d gives 1.0."""
        assert trapmf(5, 0, 2, 5, 5) == 1.0  # x == c == d → instant drop

    def test_midpoint_ascending(self):
        """Halfway between a and b returns 0.5."""
        assert trapmf(1, 0, 2, 5, 7) == pytest.approx(0.5)

    def test_midpoint_descending(self):
        """Halfway between c and d returns 0.5."""
        assert trapmf(6, 0, 2, 5, 7) == pytest.approx(0.5)

    def test_full_plateau(self):
        """Multiple x values across plateau all give 1.0."""
        for x in np.linspace(2.01, 4.99, 10):
            assert trapmf(x, 0, 2, 5, 7) == pytest.approx(1.0)


# ---------------------------------------------------------------------------
# mf_income
# ---------------------------------------------------------------------------

class TestMfIncome:
    def test_returns_correct_keys(self):
        """mf_income should return exactly 4 linguistic terms."""
        result = mf_income(50000)
        assert set(result.keys()) == {'Rendah', 'Menengah_Bawah', 'Menengah', 'Tinggi'}

    def test_low_income_rendah_high(self):
        """Very low income should have high 'Rendah' membership."""
        result = mf_income(10000)
        assert result['Rendah'] == pytest.approx(1.0)
        assert result['Tinggi'] == pytest.approx(0.0)

    def test_high_income_rendah_zero(self):
        """High income should have zero 'Rendah' membership."""
        result = mf_income(500000)
        assert result['Rendah'] == pytest.approx(0.0)

    def test_income_cap_applied(self):
        """Income over INCOME_CAP should be capped."""
        result = mf_income(INCOME_CAP * 2)
        result_capped = mf_income(INCOME_CAP)
        for term in result:
            assert result[term] == result_capped[term]

    def test_mid_income_menengah(self):
        """Mid-range income should activate 'Menengah'."""
        result = mf_income(400000)
        assert result['Menengah'] > 0

    def test_all_memberships_between_0_and_1(self):
        """All membership values should be in [0, 1]."""
        for inc in [0, 50000, 100000, 250000, 500000, 800000, INCOME_CAP]:
            result = mf_income(inc)
            for term, val in result.items():
                assert 0.0 <= val <= 1.0, f"{term} = {val} out of [0,1] for inc={inc}"


# ---------------------------------------------------------------------------
# mf_household
# ---------------------------------------------------------------------------

class TestMfHousehold:
    def test_returns_correct_keys(self):
        """mf_household should return exactly 3 linguistic terms."""
        result = mf_household(3)
        assert set(result.keys()) == {'Kecil', 'Sedang', 'Besar'}

    def test_all_memberships_between_0_and_1(self):
        """All membership values should be in [0, 1]."""
        for hh in range(1, 8):
            result = mf_household(hh)
            for term, val in result.items():
                assert 0.0 <= val <= 1.0, f"{term} = {val} out of [0,1] for hh={hh}"


# ---------------------------------------------------------------------------
# mf_workexp
# ---------------------------------------------------------------------------

class TestMfWorkexp:
    def test_returns_correct_keys(self):
        """mf_workexp should return exactly 3 linguistic terms."""
        result = mf_workexp(5)
        assert set(result.keys()) == {'Baru', 'Cukup', 'Berpengalaman'}

    def test_all_memberships_between_0_and_1(self):
        """All membership values should be in [0, 1]."""
        for we in [0, 2, 5, 10, 20, 30, 50]:
            result = mf_workexp(we)
            for term, val in result.items():
                assert 0.0 <= val <= 1.0, f"{term} = {val} out of [0,1] for we={we}"


# ---------------------------------------------------------------------------
# mf_age
# ---------------------------------------------------------------------------

class TestMfAge:
    def test_returns_correct_keys(self):
        """mf_age should return exactly 3 linguistic terms."""
        result = mf_age(30)
        assert set(result.keys()) == {'Muda', 'Produktif', 'Senior'}

    def test_all_memberships_between_0_and_1(self):
        """All membership values should be in [0, 1]."""
        for age in [18, 25, 35, 45, 55, 65, 70]:
            result = mf_age(age)
            for term, val in result.items():
                assert 0.0 <= val <= 1.0, f"{term} = {val} out of [0,1] for age={age}"


# ---------------------------------------------------------------------------
# mf_dependents
# ---------------------------------------------------------------------------

class TestMfDependents:
    def test_returns_correct_keys(self):
        """mf_dependents should return exactly 3 linguistic terms."""
        result = mf_dependents(2)
        assert set(result.keys()) == {'Sedikit', 'Sedang', 'Banyak'}

    def test_all_memberships_between_0_and_1(self):
        """All membership values should be in [0, 1]."""
        for dep in range(0, 6):
            result = mf_dependents(dep)
            for term, val in result.items():
                assert 0.0 <= val <= 1.0, f"{term} = {val} out of [0,1] for dep={dep}"


# ---------------------------------------------------------------------------
# mf_output
# ---------------------------------------------------------------------------

class TestMfOutput:
    def test_returns_correct_keys(self):
        """mf_output should return exactly 4 linguistic terms."""
        result = mf_output(50)
        assert set(result.keys()) == {'Tidak_Layak', 'Kurang_Layak', 'Cukup_Layak', 'Layak'}

    def test_all_memberships_between_0_and_1(self):
        """All membership values should be in [0, 1]."""
        for score in range(0, 101, 10):
            result = mf_output(score)
            for term, val in result.items():
                assert 0.0 <= val <= 1.0, f"{term} = {val} out of [0,1] for score={score}"

    def test_extreme_scores(self):
        """Score 0 should be 'Tidak_Layak'=1; score 100 should be 'Layak'=1."""
        result_0 = mf_output(0)
        assert result_0['Tidak_Layak'] == pytest.approx(1.0)
        assert result_0['Layak'] == pytest.approx(0.0)

        result_100 = mf_output(100)
        assert result_100['Layak'] == pytest.approx(1.0)
        assert result_100['Tidak_Layak'] == pytest.approx(0.0)


# Need this import for floating-point comparisons
import pytest
