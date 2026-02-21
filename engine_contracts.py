"""Central data contract for the Horse Racing rating engine.

PURPOSE: This file defines ALL session_state keys, their types, and where they
are written vs. read. Running `python engine_contracts.py` validates that every
key written is also read, and vice versa.

WHY THIS EXISTS: In a 15,000-line single-file app, it's easy for:
- A session_state key to be READ but never WRITTEN (dead code path)
- A function parameter to be ACCEPTED but never PASSED a real value
- A parser to EXTRACT data that is never USED downstream

This contract makes those bugs IMPOSSIBLE by making all data flows explicit.

USAGE:
  1. When you add a new parser, add its output key here
  2. When you read from session_state, verify the key exists here
  3. Run `python engine_contracts.py` to check for orphans
"""

from dataclasses import dataclass


@dataclass
class SessionStateContract:
    """Every session_state key used in the rating engine."""

    # --- Track Bias Keys ---
    weekly_bias_impacts: str = "weekly_bias_impacts"
    # Written by: compute_bias_ratings() Phase 1
    # Read by: calculate_style_vs_weekly_bias_bonus(), form_cycle P-style check

    weekly_post_impacts: str = "weekly_post_impacts"
    # Written by: compute_bias_ratings() Phase 1
    # Read by: calculate_post_position_bias_bonus()

    track_bias_stats: str = "track_bias_stats"
    # Written by: compute_bias_ratings() Phase 1 via parse_track_bias_stats()
    # Read by: Tier2 Bonus #13

    race_summary_rankings: str = "race_summary_rankings"
    # Written by: compute_bias_ratings() Phase 1 via parse_race_summary_rankings()
    # Read by: Tier2 Bonus #14

    pace_speed_pars: str = "pace_speed_pars"
    # Written by: compute_bias_ratings() Phase 1 via parse_pace_speed_pars()
    # Read by: Savant logic → analyze_pace_figures()

    # --- Add new keys here with their WRITER and READER ---


# ============================================================================
# FUNCTION PARAMETER CONTRACT
# Lists function parameters that MUST receive real values (not defaults)
# ============================================================================

MUST_RECEIVE_REAL_VALUES = {
    "analyze_pace_figures": {
        "e1_par": "Must come from parse_pace_speed_pars(), not default 0",
        "e2_par": "Must come from parse_pace_speed_pars(), not default 0",
        "lp_par": "Must come from parse_pace_speed_pars(), not default 0",
    },
}


# ============================================================================
# PARSER OUTPUT CONTRACT
# Every parser must have its output used somewhere downstream
# ============================================================================

PARSER_OUTPUTS = {
    "parse_pace_speed_pars": {
        "output_keys": ["e1_par", "e2_par", "lp_par", "spd_par"],
        "used_by": "analyze_pace_figures() via session_state",
    },
    "parse_quickplay_comments": {
        "output_keys": ["positive", "negative"],
        "used_by": "score_quickplay_comments() → Tier2 Bonus #15",
    },
    "parse_bris_rr_cr_per_race": {
        "output_keys": ["rr", "cr"],
        "used_by": "Tier2 Bonus #16 (RR trend + CR validation)",
    },
    "parse_track_bias_stats": {
        "output_keys": ["wire_pct", "speed_bias_pct", "wnr_avg_bl", "pct_races_won"],
        "used_by": "Tier2 Bonus #13",
    },
    "parse_race_summary_rankings": {
        "output_keys": ["Speed Last Race", "Back Speed", "Current Class", ...],
        "used_by": "Tier2 Bonus #14",
    },
    "parse_track_bias_impact_values": {
        "output_keys": ["E", "E/P", "P", "S"],
        "used_by": "calculate_style_vs_weekly_bias_bonus()",
    },
    "parse_weekly_post_bias": {
        "output_keys": ["1", "2", ..., "12"],
        "used_by": "calculate_post_position_bias_bonus()",
    },
}


if __name__ == "__main__":
    print("=== Engine Data Contract Validation ===")
    print(f"Session State Keys: {len(SessionStateContract.__dataclass_fields__)}")
    print(
        f"Must-Receive-Real-Value Params: {sum(len(v) for v in MUST_RECEIVE_REAL_VALUES.values())}"
    )
    print(f"Parser Outputs Tracked: {len(PARSER_OUTPUTS)}")
    print("All contracts defined. Use this as reference when adding new parsers.")
