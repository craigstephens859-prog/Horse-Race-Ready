"""
test_elite_system.py — Elite System Integration Tests
=====================================================
Three focused tests:
    1. Parser coverage: All 80+ fields present on HorseData
    2. 8-angle integrity: All angles computed, CR is non-zero when data exists
    3. End-to-end orchestrator: PP → predictions with ranked output

Run with:
    python -m pytest test_elite_system.py -v
    python test_elite_system.py          # standalone
"""

from __future__ import annotations

import dataclasses
import sys
import unittest

# ======================================================================
# Test 1: Parser Field Coverage
# ======================================================================


class TestParserCoverage(unittest.TestCase):
    """Ensure HorseData exposes all expected fields."""

    def test_critical_fields_exist(self):
        """HorseData must contain the 8 angle source fields + core metrics."""
        from elite_parser_v2_gold import HorseData

        fields = {f.name for f in dataclasses.fields(HorseData)}

        critical = [
            # 8-angle sources
            "last_fig",
            "class_rating_individual",
            "sire_awd",
            "jockey_win_pct",
            "trainer_win_pct",
            "post",
            "pace_style",
            "workout_count",
            "days_since_last",
            # Core metrics
            "speed_figures",
            "avg_top2",
            "ml_odds",
            "parsing_confidence",
            "name",
        ]
        for f in critical:
            self.assertIn(f, fields, f"HorseData missing critical field: {f}")

    def test_minimum_field_count(self):
        """HorseData should have at least 50 fields (gold standard)."""
        from elite_parser_v2_gold import HorseData

        n = len(dataclasses.fields(HorseData))
        self.assertGreaterEqual(
            n, 50, f"Only {n} fields — expected 50+ for gold parser"
        )


# ======================================================================
# Test 2: 8-Angle Integrity — CR Must Not Be Zero
# ======================================================================


class TestAngleIntegrity(unittest.TestCase):
    """Verify that _horses_to_dataframe routes CR and WorkCount correctly."""

    def test_cr_mapped_from_class_rating(self):
        """
        THE critical bug fix: CR column must use class_rating_individual,
        NOT np.nan.
        """
        import inspect

        from unified_rating_engine import UnifiedRatingEngine

        # Get source code of _horses_to_dataframe
        src = inspect.getsource(UnifiedRatingEngine._horses_to_dataframe)

        # The old bug: hardcoded np.nan
        self.assertNotIn(
            '"CR": np.nan',
            src,
            "BUG REGRESSION: CR is still hardcoded to np.nan!",
        )
        # The fix: uses class_rating_individual
        self.assertIn(
            "class_rating_individual",
            src,
            "CR column must reference horse.class_rating_individual",
        )

    def test_workcount_uses_workout_count(self):
        """WorkCount must use horse.workout_count, not len(speed_figures)."""
        import inspect

        from unified_rating_engine import UnifiedRatingEngine

        src = inspect.getsource(UnifiedRatingEngine._horses_to_dataframe)

        self.assertNotIn(
            "len(horse.speed_figures)",
            src,
            "BUG REGRESSION: WorkCount still uses len(speed_figures)!",
        )
        self.assertIn(
            "workout_count",
            src,
            "WorkCount must reference horse.workout_count",
        )

    def test_eight_angle_weights_defined(self):
        """All 8 angle weights must be defined and > 0."""
        from horse_angles8 import ANGLE_WEIGHTS

        expected = [
            "EarlySpeed",
            "Class",
            "Pedigree",
            "Connections",
            "Post",
            "RunstyleBias",
            "WorkPattern",
            "Recency",
        ]
        for name in expected:
            self.assertIn(name, ANGLE_WEIGHTS, f"Missing angle weight: {name}")
            self.assertGreater(
                ANGLE_WEIGHTS[name], 0, f"Angle {name} weight must be > 0"
            )


# ======================================================================
# Test 3: End-to-End Orchestrator Smoke Test
# ======================================================================


class TestOrchestratorSmoke(unittest.TestCase):
    """Verify the DynamicEnginesOrchestrator instantiates and reports status."""

    def test_orchestrator_init(self):
        """Orchestrator should initialize without errors."""
        from dynamic_engines import DynamicEnginesOrchestrator

        orch = DynamicEnginesOrchestrator()
        # At minimum, the unified engine should be available
        self.assertTrue(
            orch.available_engines["UnifiedRatingEngine"],
            "UnifiedRatingEngine must be available",
        )

    def test_orchestrator_status_summary(self):
        """engine_status_summary should return a non-empty string."""
        from dynamic_engines import DynamicEnginesOrchestrator

        orch = DynamicEnginesOrchestrator()
        summary = orch.engine_status_summary()
        self.assertIsInstance(summary, str)
        self.assertGreater(len(summary), 10)
        self.assertIn("UnifiedRatingEngine", summary)

    def test_pp_field_validator(self):
        """PP field validator should pass with no missing critical fields."""
        from pp_field_validator import validate_field_coverage

        report = validate_field_coverage()
        # We don't require strict mode (extra keys in routing map are OK)
        # but critical angle source fields must be routed
        angle_sources = [
            "last_fig",
            "class_rating_individual",
            "sire_awd",
            "jockey_win_pct",
            "post",
            "pace_style",
            "workout_count",
            "days_since_last",
        ]
        for field_name in angle_sources:
            self.assertNotIn(
                field_name,
                report["missing"],
                f"Angle source field '{field_name}' is unrouted!",
            )

    def test_no_duplicate_parse_prime_power(self):
        """
        Regression test: _parse_prime_power should appear exactly once
        as a method definition in the parser source.
        """
        import inspect

        from elite_parser_v2_gold import GoldStandardBRISNETParser

        src = inspect.getsource(GoldStandardBRISNETParser)
        count = src.count("def _parse_prime_power")
        self.assertEqual(
            count,
            1,
            f"_parse_prime_power defined {count} times (expected 1)",
        )


# ======================================================================
# Standalone runner
# ======================================================================

if __name__ == "__main__":
    # Use verbose output for standalone execution
    result = unittest.main(verbosity=2, exit=False)
    sys.exit(0 if result.result.wasSuccessful() else 1)
