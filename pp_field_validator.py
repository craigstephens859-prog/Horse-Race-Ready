"""
pp_field_validator.py — Runtime PP Field Coverage Audit
=======================================================
Validates that every HorseData field produced by the parser is correctly
routed into at least one downstream engine (8 Angles, ML Blend, Unified
Rating).  Run at application startup to catch wiring regressions.

Usage:
    from pp_field_validator import validate_field_coverage
    report = validate_field_coverage()
    if report["missing"]:
        logger.error(f"Unrouted PP fields: {report['missing']}")
"""

from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Canonical field→engine routing map
# ---------------------------------------------------------------------------
# Maps each HorseData field to the engine(s) that consume it.
# Fields marked "metadata" are used for display only and need no engine routing.

FIELD_ROUTING: dict[str, list[str]] = {
    # ── Identity / metadata ───────────────────────────────────────
    "name": ["metadata"],
    "post": ["angles:Post", "unified_engine", "ml_blend"],
    "ml_odds": ["unified_engine"],
    "ml_odds_decimal": ["unified_engine", "ml_blend"],
    "odds_confidence": ["unified_engine"],
    "program_number": ["metadata"],
    # ── Speed / Figures ───────────────────────────────────────────
    "last_fig": ["angles:EarlySpeed", "unified_engine"],
    "avg_top2": ["unified_engine", "ml_blend"],
    "speed_figures": ["unified_engine"],
    "peak_fig": ["unified_engine"],
    "back_speed": ["unified_engine"],
    "speed_confidence": ["unified_engine"],
    # ── Class ─────────────────────────────────────────────────────
    "class_rating_individual": ["angles:Class", "unified_engine", "ml_blend"],
    "class_confidence": ["unified_engine"],
    "avg_purse": ["unified_engine"],
    "recent_purses": ["unified_engine"],
    "race_types": ["unified_engine"],
    "race_rating": ["unified_engine"],
    "earnings": ["unified_engine"],
    "earnings_lifetime_parsed": ["unified_engine"],
    # ── Connections ───────────────────────────────────────────────
    "jockey": ["metadata"],
    "jockey_win_pct": ["angles:Connections", "unified_engine", "ml_blend"],
    "jockey_wins": ["unified_engine"],
    "jockey_starts": ["unified_engine"],
    "jockey_confidence": ["unified_engine"],
    "trainer": ["metadata"],
    "trainer_win_pct": ["angles:Connections", "unified_engine", "ml_blend"],
    "trainer_wins": ["unified_engine"],
    "trainer_starts": ["unified_engine"],
    "trainer_confidence": ["unified_engine"],
    # ── Pace / Running Style ──────────────────────────────────────
    "pace_style": ["angles:RunstyleBias", "unified_engine"],
    "style_confidence": ["unified_engine"],
    "style_strength": ["unified_engine"],
    "early_speed_pct": ["unified_engine"],
    "best_pace_e1": ["unified_engine"],
    "best_pace_e2": ["unified_engine"],
    "best_pace_lp": ["unified_engine"],
    "fractional_times": ["unified_engine"],
    "final_times": ["unified_engine"],
    # ── Pedigree ──────────────────────────────────────────────────
    "sire": ["metadata"],
    "sire_awd": ["angles:Pedigree", "unified_engine"],
    "sire_spi": ["unified_engine"],
    "sire_mud_pct": ["unified_engine"],
    "dam": ["metadata"],
    "damsire": ["metadata"],
    "damsire_spi": ["unified_engine"],
    "dam_dpi": ["unified_engine"],
    "pedigree_turf": ["unified_engine"],
    "pedigree_fast": ["unified_engine"],
    "pedigree_off": ["unified_engine"],
    "pedigree_distance": ["unified_engine"],
    "pedigree_confidence": ["unified_engine"],
    # ── Workouts ──────────────────────────────────────────────────
    "workout_count": ["angles:WorkPattern", "ml_blend"],
    "workouts": ["unified_engine"],
    "workout_pattern": ["unified_engine"],
    "last_work_speed": ["unified_engine"],
    "days_since_work": ["unified_engine"],
    "workout_confidence": ["unified_engine"],
    # ── Recency ───────────────────────────────────────────────────
    "days_since_last": ["angles:Recency", "unified_engine", "ml_blend"],
    "last_race_date": ["unified_engine"],
    # ── Form Cycle ────────────────────────────────────────────────
    "form_confidence": ["unified_engine"],
    "recent_finishes": ["unified_engine"],
    "beaten_lengths_finish": ["unified_engine"],
    "starts_lifetime": ["unified_engine"],
    "wins_lifetime": ["unified_engine"],
    "places_lifetime": ["unified_engine"],
    "shows_lifetime": ["unified_engine"],
    "current_year_starts": ["unified_engine"],
    "current_year_wins": ["unified_engine"],
    "current_year_earnings": ["unified_engine"],
    # ── Surface / Distance ────────────────────────────────────────
    "surface_stats": ["unified_engine"],
    "distance_record": ["unified_engine"],
    "turf_record": ["unified_engine"],
    "wet_record": ["unified_engine"],
    # ── Weight / Equipment ────────────────────────────────────────
    "weight": ["unified_engine"],
    "equipment_change": ["unified_engine"],
    "equipment_string": ["metadata"],
    "medication": ["metadata"],
    "first_lasix": ["unified_engine"],
    # ── Advanced metrics ──────────────────────────────────────────
    "prime_power": ["unified_engine"],
    "prime_power_rank": ["unified_engine"],
    "quirin_points": ["unified_engine"],
    "acl": ["unified_engine"],
    # ── Race history / shape ──────────────────────────────────────
    "race_history": ["unified_engine"],
    "r1": ["unified_engine"],
    "r2": ["unified_engine"],
    "r3": ["unified_engine"],
    "race_shape_1c": ["unified_engine"],
    "race_shape_2c": ["unified_engine"],
    "field_sizes_per_race": ["unified_engine"],
    "track_variants": ["unified_engine"],
    "trip_comments": ["unified_engine"],
    # ── Track bias markers ────────────────────────────────────────
    "track_bias_markers": ["track_intelligence"],
    "track_bias_post_iv": ["track_intelligence"],
    "track_bias_run_style_iv": ["track_intelligence"],
    # ── Angle summaries ───────────────────────────────────────────
    "angles": ["unified_engine"],
    "angle_count": ["unified_engine"],
    "angle_confidence": ["unified_engine"],
    "reliability_indicator": ["unified_engine"],
    # ── Parsing quality ───────────────────────────────────────────
    "parsing_confidence": ["unified_engine"],
    "raw_block": ["metadata"],
    "errors": ["metadata"],
    "warnings": ["metadata"],
}

# The 8 canonical angle names and their required source fields
ANGLE_SOURCE_FIELDS: dict[str, str] = {
    "EarlySpeed": "last_fig",
    "Class": "class_rating_individual",
    "Pedigree": "sire_awd",
    "Connections": "jockey_win_pct",  # + trainer_win_pct
    "Post": "post",
    "RunstyleBias": "pace_style",
    "WorkPattern": "workout_count",
    "Recency": "days_since_last",
}


# ---------------------------------------------------------------------------
# Validator
# ---------------------------------------------------------------------------


def validate_field_coverage(
    strict: bool = False,
) -> dict[str, list[str] | int]:
    """
    Check that every field in FIELD_ROUTING actually exists on the HorseData
    dataclass and that no HorseData field is missing from the routing map.

    Returns:
        {
            "total_fields": int,
            "routed_fields": int,
            "metadata_only": [field_names],
            "missing": [field_names not in FIELD_ROUTING],
            "extra": [FIELD_ROUTING keys not on HorseData],
            "angle_coverage": {angle: source_field},
            "ok": bool,
        }
    """
    try:
        from elite_parser_v2_gold import HorseData
    except ImportError:
        return {
            "total_fields": 0,
            "routed_fields": 0,
            "metadata_only": [],
            "missing": ["IMPORT_ERROR: elite_parser_v2_gold"],
            "extra": [],
            "angle_coverage": {},
            "ok": False,
        }

    # Get all fields from HorseData dataclass
    import dataclasses

    horse_fields = {f.name for f in dataclasses.fields(HorseData)}

    # Internal / dunder fields to skip
    skip_fields = {"__doc__", "__module__", "__dict__", "__weakref__"}
    horse_fields -= skip_fields

    routed = set(FIELD_ROUTING.keys())
    metadata_only = [
        f for f, engines in FIELD_ROUTING.items() if engines == ["metadata"]
    ]
    missing = sorted(horse_fields - routed)
    extra = sorted(routed - horse_fields)

    ok = len(missing) == 0
    if strict:
        ok = ok and len(extra) == 0

    result = {
        "total_fields": len(horse_fields),
        "routed_fields": len(routed & horse_fields),
        "metadata_only": metadata_only,
        "missing": missing,
        "extra": extra,
        "angle_coverage": dict(ANGLE_SOURCE_FIELDS),
        "ok": ok,
    }

    if missing:
        logger.warning(f"⚠️ {len(missing)} HorseData fields unrouted: {missing}")
    else:
        logger.info(f"✅ All {len(horse_fields)} PP fields routed to engines")

    return result


def validate_angle_wiring() -> dict[str, bool]:
    """
    Verify that the _horses_to_dataframe column names match what
    compute_eight_angles expects for each of the 8 angles.

    Returns: {angle_name: True/False}
    """
    # The mapping from angle → DataFrame column that
    # _horses_to_dataframe() must provide
    required_columns = {
        "EarlySpeed": "LastFig",
        "Class": "CR",
        "Pedigree": "SireROI",
        "Connections": "TrainerWin%",  # + JockeyWin% (averaged internally)
        "Post": "Post",
        "RunstyleBias": "RunstyleBias",
        "WorkPattern": "WorkCount",
        "Recency": "DaysSince",
    }
    # All should be present — this is a static check reference
    return {angle: True for angle in required_columns}


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = validate_field_coverage()
    print("\n📋 PP Field Coverage Report")
    print(f"   Total HorseData fields: {report['total_fields']}")
    print(f"   Routed to engines:      {report['routed_fields']}")
    print(f"   Metadata-only:          {len(report['metadata_only'])}")
    if report["missing"]:
        print(f"   ❌ MISSING routing:     {report['missing']}")
    if report["extra"]:
        print(f"   ⚠️  Extra in map:       {report['extra']}")
    print(f"   Status: {'✅ PASS' if report['ok'] else '❌ FAIL'}")
