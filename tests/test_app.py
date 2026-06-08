"""Smoke tests — verify the app and its modules import without errors."""

import sys
import os

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class TestAppImports:
    """Verify that all modules can be imported without errors."""

    def test_fuzzy_logic_core_imports(self):
        """fuzzy_logic.core imports without error."""
        from fuzzy_logic.core import trimf, trapmf
        from fuzzy_logic.core import mf_income, mf_household, mf_workexp, mf_age, mf_dependents, mf_output
        from fuzzy_logic.core import INCOME_CAP
        assert callable(trimf)
        assert callable(trapmf)
        assert INCOME_CAP == 1_000_000

    def test_fuzzy_logic_rules_imports(self):
        """fuzzy_logic.rules imports without error."""
        from fuzzy_logic.rules import RULES
        assert len(RULES) == 33

    def test_fuzzy_logic_inference_imports(self):
        """fuzzy_logic.inference imports without error."""
        from fuzzy_logic.inference import (
            fuzzify, mamdani_inference, centroid, fuzzy_mamdani,
            sugeno_inference, weighted_average, fuzzy_sugeno,
            UNIVERSE, THRESHOLD, SUGENO_CONST,
        )
        assert len(UNIVERSE) == 500
        assert THRESHOLD == 55
        assert callable(fuzzy_mamdani)
        assert callable(fuzzy_sugeno)

    def test_utils_metrics_imports(self):
        """utils.metrics imports without error."""
        from utils.metrics import ground_truth_score, confusion_matrix_manual
        assert callable(ground_truth_score)
        assert callable(confusion_matrix_manual)

    def test_utils_plots_imports(self):
        """utils.plots imports without error."""
        from utils.plots import (
            plot_membership_functions,
            plot_mamdani_defuzz,
            plot_sugeno_defuzz,
            plot_score_distribution,
            plot_scatter_comparison,
            plot_segment_comparison,
        )
        assert callable(plot_membership_functions)
        assert callable(plot_mamdani_defuzz)

    def test_all_rules_have_valid_outputs(self):
        """Every rule's output term should exist in SUGENO_CONST."""
        from fuzzy_logic.rules import RULES
        from fuzzy_logic.inference import SUGENO_CONST
        for rule in RULES:
            output_term = rule[5]
            assert output_term in SUGENO_CONST, f"Rule {rule} has unknown output '{output_term}'"
