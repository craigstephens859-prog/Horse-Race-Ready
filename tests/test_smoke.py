"""Smoke tests for core rating-engine functions in app.py.

These tests verify that critical functions:
  1. Don't crash on typical inputs
  2. Return the correct type
  3. Handle edge cases (empty, None, boundary values)

Strategy: We extract individual pure functions from app.py source using AST
parsing, then exec them in an isolated namespace.  This avoids importing the
full 15K-line Streamlit app (which has top-level UI code that can't run in
a test harness).

Run:  python -m pytest tests/test_smoke.py -v
"""

from __future__ import annotations

import ast
import logging
import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import pytest

# ---------------------------------------------------------------------------
# Source extraction — pull pure functions out of app.py without executing it
# ---------------------------------------------------------------------------

ROOT = Path(__file__).resolve().parent.parent
APP_PY = ROOT / "app.py"

_FUNCTIONS_TO_EXTRACT = [
    "safe_float",
    "_distance_bucket_from_text",
    "distance_bucket",
    "analyze_pace_figures",
    "calculate_layoff_factor",
    "calculate_form_trend",
    "calculate_hot_trainer_bonus",
    "analyze_class_movement",
    "softmax_from_rating",
    "fair_probs_from_ratings",
]


def _extract_function_source(source: str, func_name: str) -> str | None:
    """Extract a single top-level function's source from the app.py text."""
    tree = ast.parse(source)
    lines = source.splitlines()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.FunctionDef) and node.name == func_name:
            return "\n".join(lines[node.lineno - 1 : node.end_lineno])
    return None


def _build_namespace() -> dict[str, Any]:
    """Build a namespace containing extracted functions and their deps."""
    source = APP_PY.read_text(encoding="utf-8")

    ns: dict[str, Any] = {
        "pd": pd,
        "np": np,
        "Any": Any,
        "__builtins__": __builtins__,
        "logger": logging.getLogger("test_smoke"),
    }

    # Pull important constants (MODEL_CONFIG, DIST_BUCKET_*, etc.)
    for m in re.finditer(
        r"^((?:DIST_BUCKET_|MODEL_CONFIG)\w*\s*=\s*)", source, re.MULTILINE
    ):
        # For multi-line dicts, extract from AST instead
        pass

    # Use AST to grab top-level constants
    tree = ast.parse(source)
    src_lines = source.splitlines()
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and (
                    target.id.startswith("DIST_BUCKET_") or target.id == "MODEL_CONFIG"
                ):
                    const_src = "\n".join(src_lines[node.lineno - 1 : node.end_lineno])
                    try:
                        exec(const_src, ns)  # noqa: S102
                    except Exception:
                        pass

    # Extract and exec each function
    for name in _FUNCTIONS_TO_EXTRACT:
        func_src = _extract_function_source(source, name)
        if func_src:
            try:
                exec(func_src, ns)  # noqa: S102
            except Exception as exc:
                ns[name] = None
                print(f"[test_smoke] Could not load {name}: {exc}")

    return ns


_ns = _build_namespace()


def _fn(name: str):
    """Helper — get a function from the namespace or skip the test."""
    fn = _ns.get(name)
    if fn is None or not callable(fn):
        pytest.skip(f"{name} could not be extracted")
    return fn


# ═══════════════════════════════════════════════════════════════════════════
# 1. safe_float — universal value converter
# ═══════════════════════════════════════════════════════════════════════════


class TestSafeFloat:
    def setup_method(self):
        self.f = _fn("safe_float")

    def test_int(self):
        assert self.f(42) == pytest.approx(42.0)

    def test_float(self):
        assert self.f(3.14) == pytest.approx(3.14)

    def test_string(self):
        assert self.f("123.45") == pytest.approx(123.45)

    def test_percentage(self):
        assert self.f("75.6%") == pytest.approx(75.6)

    def test_none(self):
        assert self.f(None) == pytest.approx(0.0)

    def test_custom_default(self):
        assert self.f(None, default=-1.0) == pytest.approx(-1.0)

    def test_empty(self):
        assert self.f("") == pytest.approx(0.0)

    def test_garbage(self):
        assert self.f("not-a-number") == pytest.approx(0.0)


# ═══════════════════════════════════════════════════════════════════════════
# 2. distance_bucket — maps distance text to sprint/mid/route
# ═══════════════════════════════════════════════════════════════════════════


class TestDistanceBucket:
    def setup_method(self):
        self.f = _fn("distance_bucket")

    def test_sprint(self):
        assert isinstance(self.f("5f"), str)

    def test_route(self):
        assert isinstance(self.f("1 1/8 mi"), str)

    def test_empty(self):
        assert isinstance(self.f(""), str)

    def test_garbage(self):
        assert isinstance(self.f("xyz"), str)


# ═══════════════════════════════════════════════════════════════════════════
# 3. analyze_pace_figures — pace bonus calculator
# ═══════════════════════════════════════════════════════════════════════════


class TestAnalyzePaceFigures:
    def setup_method(self):
        self.f = _fn("analyze_pace_figures")

    def test_normal(self):
        bonus = self.f(e1_vals=[90, 88, 85], e2_vals=[88, 86, 84], lp_vals=[92, 90, 88])
        assert isinstance(bonus, float)
        assert -0.20 <= bonus <= 0.25

    def test_with_pars(self):
        bonus = self.f(
            e1_vals=[90, 88],
            e2_vals=[88, 86],
            lp_vals=[92, 90],
            e1_par=87,
            e2_par=85,
            lp_par=89,
        )
        assert isinstance(bonus, float)

    def test_too_few(self):
        assert self.f(e1_vals=[90], e2_vals=[88], lp_vals=[92]) == pytest.approx(0.0)

    def test_empty(self):
        assert self.f(e1_vals=[], e2_vals=[], lp_vals=[]) == pytest.approx(0.0)

    def test_none_pars(self):
        bonus = self.f(
            e1_vals=[90, 88],
            e2_vals=[88, 86],
            lp_vals=[92, 90],
            e1_par=None,
            e2_par=None,
            lp_par=None,
        )
        assert isinstance(bonus, float)


# ═══════════════════════════════════════════════════════════════════════════
# 4. calculate_layoff_factor — days-off impact
# ═══════════════════════════════════════════════════════════════════════════


class TestCalculateLayoffFactor:
    def setup_method(self):
        self.f = _fn("calculate_layoff_factor")

    def test_fresh(self):
        assert self.f(days_since_last=7) > 0

    def test_normal(self):
        assert isinstance(self.f(days_since_last=30), float)

    def test_long_layoff(self):
        assert self.f(days_since_last=200) < 0

    def test_workouts_mitigate(self):
        no_works = self.f(days_since_last=120, num_workouts=0)
        with_works = self.f(days_since_last=120, num_workouts=5)
        assert with_works > no_works

    def test_none_workouts(self):
        assert isinstance(self.f(days_since_last=60, num_workouts=None), float)

    def test_max_penalty(self):
        assert self.f(days_since_last=365) >= -3.0


# ═══════════════════════════════════════════════════════════════════════════
# 5. calculate_hot_trainer_bonus — trainer quality
# ═══════════════════════════════════════════════════════════════════════════


class TestCalculateHotTrainerBonus:
    def setup_method(self):
        self.f = _fn("calculate_hot_trainer_bonus")

    def test_hot_trainer(self):
        bonus = self.f(trainer_win_pct=22.0, is_hot_l14=True)
        assert isinstance(bonus, float)
        assert bonus > 0

    def test_cold_trainer(self):
        bonus = self.f(trainer_win_pct=0.0)
        assert isinstance(bonus, float)
        assert bonus <= 0

    def test_none_starts(self):
        assert isinstance(self.f(trainer_win_pct=15.0, trainer_starts=None), float)


# ═══════════════════════════════════════════════════════════════════════════
# 6. analyze_class_movement — class up/down
# ═══════════════════════════════════════════════════════════════════════════


class TestAnalyzeClassMovement:
    def setup_method(self):
        self.f = _fn("analyze_class_movement")

    def test_normal(self):
        past = [
            {"class": "Clm", "purse": 10000},
            {"class": "Clm", "purse": 12000},
            {"class": "Alw", "purse": 20000},
        ]
        result = self.f(past, "Alw")
        assert isinstance(result, dict)
        assert "class_change" in result
        assert "bonus" in result

    def test_empty(self):
        assert self.f([], "Alw")["class_change"] == "unknown"

    def test_single_race(self):
        assert self.f([{"class": "Clm"}], "Clm")["class_change"] == "unknown"


# ═══════════════════════════════════════════════════════════════════════════
# 7. fair_probs_from_ratings — probability engine
# ═══════════════════════════════════════════════════════════════════════════


class TestFairProbsFromRatings:
    def setup_method(self):
        self.f = _fn("fair_probs_from_ratings")

    def test_basic(self):
        df = pd.DataFrame({"Horse": ["A", "B", "C"], "R": [90.0, 85.0, 80.0]})
        probs = self.f(df)
        assert isinstance(probs, dict)
        assert len(probs) == 3
        assert abs(sum(probs.values()) - 1.0) < 0.01

    def test_highest_rated_wins(self):
        df = pd.DataFrame({"Horse": ["Best", "Worst"], "R": [100.0, 50.0]})
        probs = self.f(df)
        assert probs["Best"] > probs["Worst"]

    def test_empty_df(self):
        assert self.f(pd.DataFrame()) == {}

    def test_none(self):
        assert self.f(None) == {}

    def test_missing_columns(self):
        assert self.f(pd.DataFrame({"Name": ["A"], "Score": [10]})) == {}

    def test_nan_ratings(self):
        df = pd.DataFrame({"Horse": ["A", "B", "C"], "R": [90.0, np.nan, 80.0]})
        probs = self.f(df)
        assert len(probs) == 3
        assert all(np.isfinite(v) for v in probs.values())
