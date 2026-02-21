"""
rating_engine.py  — Phase 4 extraction from app.py
Core rating computation engine: compute_bias_ratings and all sub-calculations.
46 functions, extracted for maintainability (app.py remains the Streamlit UI host).
"""

import contextlib
import logging
import re
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    ELITE_TRAINERS,
    FTS_PARAMS,
    MODEL_CONFIG,
    NA_STYLE_PARAMS,
    TRACK_BIAS_PROFILES,
    condition_modifiers,
)
from pp_parsing import (
    _post_bucket,
    detect_lasix_change,
    extract_race_metadata_from_pp_text,
    infer_purse_from_race_type,
    parse_awd_analysis,
    parse_bris_rr_cr_per_race,
    parse_claiming_prices,
    parse_e1_e2_lp_values,
    parse_fractional_positions,
    parse_jockey_combo_stats,
    parse_pace_speed_pars,
    parse_pedigree_spi,
    parse_pedigree_surface_stats,
    parse_quickplay_comments,
    parse_race_history_from_block,
    parse_race_summary_rankings,
    parse_recent_class_levels,
    parse_recent_races_detailed,
    parse_speed_figures_for_block,
    parse_track_bias_impact_values,
    parse_track_bias_stats,
    parse_weekly_post_bias,
    parse_workout_data,
    split_into_horse_chunks,
)
from utils import (
    _canonical_track,
    _normalize_style,
    _style_norm,
    calculate_form_trend,
    distance_bucket,
    distance_to_furlongs,
    fair_to_american_str,
    is_marathon_distance,
    is_sprint_distance,
    normalize_horse_name,
    odds_to_decimal,
    safe_float,
    safe_int,
)

logger = logging.getLogger(__name__)




def _get_track_bias_delta(
    track_name: str, surface_type: str, distance_txt: str, style: str, post_str: str
) -> float:
    canon = _canonical_track(track_name)
    surf = (surface_type or "Dirt").strip().title()
    buck = distance_bucket(distance_txt)  # ≤6f / 6.5–7f / 8f+

    # Try to get specific track profile first
    cfg = TRACK_BIAS_PROFILES.get(canon, {}).get(surf, {}).get(buck, {})

    # If no specific profile found, use default fallback
    if not cfg:
        cfg = TRACK_BIAS_PROFILES.get("_DEFAULT", {}).get(surf, {}).get(buck, {})

    # If still no config (shouldn't happen with default), return 0
    if not cfg:
        return 0.0

    s_norm = _style_norm(style)
    runstyle_delta = float((cfg.get("runstyle", {}) or {}).get(s_norm, 0.0))
    post_delta = float((cfg.get("post", {}) or {}).get(_post_bucket(post_str), 0.0))
    return float(np.clip(runstyle_delta + post_delta, -1.0, 1.0))




def analyze_claiming_price_movement(
    recent_prices: list[int], today_price: int
) -> float:
    """SAVANT ANGLE: Claiming price class movement. Returns bonus from -0.10 to +0.15"""
    bonus = 0.0
    if not recent_prices or today_price <= 0:
        return bonus
    avg_recent = (
        np.mean(recent_prices[:3]) if len(recent_prices) >= 3 else recent_prices[0]
    )
    if avg_recent > today_price * 1.3:
        bonus += 0.15  # Big drop = intent to win
    elif avg_recent > today_price * 1.15:
        bonus += 0.08
    elif today_price > avg_recent * 1.3:
        bonus -= 0.10  # Rising in class
    elif today_price > avg_recent * 1.15:
        bonus -= 0.05
    return bonus




def calculate_trip_quality(positions: list[list[int]], field_size: int = 10) -> float:
    """SAVANT ANGLE: Trip handicapping. Returns bonus from -0.04 to +0.12"""
    bonus = 0.0
    # CRITICAL FIX: Check positions exists AND has elements before accessing
    if not positions or len(positions) == 0 or len(positions[0]) < 6:
        return bonus
    pp, st, c1, c2, _, finish = positions[0]
    if pp >= 7 and st >= 7 and finish <= 3:
        bonus += 0.12  # Overcame trouble = class
    if pp <= 3 and st >= 7 and finish <= 4:
        bonus += 0.09  # Rail trouble excuse
    if pp >= field_size - 2 and c1 >= field_size - 2:
        bonus += 0.08 if finish <= 3 else 0.04  # Wide trip
    if c2 >= 6 and finish <= 3:
        bonus += 0.05  # Closer angle
    if st <= 2 and c1 <= 2 and finish >= 5:
        bonus -= 0.04  # Couldn't sustain
    if abs(st - pp) >= 4 and finish <= 4:
        bonus += 0.06  # Steadied but recovered
    return bonus




def analyze_pace_figures(
    e1_vals: list[int],
    e2_vals: list[int],
    lp_vals: list[int],
    e1_par: int | None = None,
    e2_par: int | None = None,
    lp_par: int | None = None,
) -> float:
    """OPTIMIZED Feb 9 2026: PAR-adjusted pace analysis with recency-weighted averages.

    Returns bonus from -0.15 to +0.20. Uses recency weights [2x, 1x, 0.5x]
    and energy distribution analysis. Validated on Oaklawn R9 (Air of Defiance #2).

    NOTE: e1_par/e2_par/lp_par use None (not 0) as default so callers that fail
    to provide real pars produce a visible None rather than a silent 0 that
    passes the `if e1_par and lp_par` guard as falsy.
    """
    bonus = 0.0
    if len(e1_vals) < 2 or len(lp_vals) < 2:
        return bonus
    # Recency-weighted averages (most recent race weighted 2x)
    weights = [2.0, 1.0, 0.5][: len(e1_vals)]
    w_sum = sum(weights)
    avg_e1 = sum(v * w for v, w in zip(e1_vals[:3], weights, strict=False)) / w_sum
    avg_lp = sum(
        v * w for v, w in zip(lp_vals[:3], weights[: len(lp_vals)], strict=False)
    ) / sum(weights[: len(lp_vals)])
    avg_e2 = (
        sum(v * w for v, w in zip(e2_vals[:3], weights[: len(e2_vals)], strict=False))
        / sum(weights[: len(e2_vals)])
        if len(e2_vals) >= 2
        else avg_e1
    )
    # PAR adjustment
    if e1_par and lp_par:
        avg_e1 -= e1_par
        avg_lp -= lp_par
        if e2_par:
            avg_e2 -= e2_par
    # Closer with gas in tank
    if avg_lp > avg_e1 + 5:
        bonus += 0.10 * min((avg_lp - avg_e1) / 15, 1.0)
    # Speed + stamina
    if avg_e1 >= 95 and avg_lp >= 85:
        bonus += 0.08
    # Speed, no stamina (one-dimensional)
    if avg_e1 >= 90 and avg_lp < 75:
        bonus -= 0.08
    # Energy distribution — balanced = +, front-loaded = -
    total_energy = avg_e1 + avg_e2 + avg_lp
    if total_energy > 0:
        e1_pct = avg_e1 / total_energy
        lp_pct = avg_lp / total_energy
        if abs(e1_pct - lp_pct) < 0.03:
            bonus += 0.05  # Perfectly distributed
        elif e1_pct > 0.38 and lp_pct < 0.30:
            bonus -= 0.04  # Front-loaded
    # Balanced E1-E2-LP
    if len(e2_vals) >= 2:
        if abs(avg_e1 - avg_e2) <= 3 and abs(avg_e2 - avg_lp) <= 3:
            bonus += 0.04
    return round(float(np.clip(bonus, -0.15, 0.20)), 4)




def detect_bounce_risk(speed_figs: list[int]) -> float:
    """OPTIMIZED Feb 9 2026: Regression-based bounce detection.

    Returns [-0.25, +0.20]. Uses np.polyfit regression slope, std-based consistency,
    and career-relative analysis. Validated on Oaklawn R9 (Air of Defiance #2).
    """
    if len(speed_figs) < 2:
        logger.debug(
            "detect_bounce_risk: insufficient speed figs (%d), need >= 2",
            len(speed_figs),
        )
        return 0.0
    figs = speed_figs[:6]
    n = len(figs)
    # Linear regression slope on recent figs (positive slope = improving)
    x = np.arange(n)
    coeffs = np.polyfit(x, figs, 1)
    slope = coeffs[0]  # Points per race
    fig_std = float(np.std(figs))
    career_best = max(speed_figs)
    career_mean = float(np.mean(speed_figs))
    latest = figs[0]
    score = 0.0
    # Improving trend (positive slope = each race getting better)
    if slope > 1.0:
        score += min(slope * 0.03, 0.12)
    elif slope < -2.0:
        score += max(slope * 0.02, -0.10)
    # Consistency bonus
    if fig_std <= 3.0 and n >= 3:
        score += 0.05
    elif fig_std >= 8.0:
        score -= 0.04
    # Career-best bounce risk
    if latest == career_best and n >= 3:
        drop_from_best = career_best - career_mean
        if drop_from_best >= 10:
            score -= 0.10
        elif drop_from_best >= 6:
            score -= 0.05
    # Sustained peak form
    if n >= 3 and min(figs[:3]) >= career_best - 3:
        score += 0.08
    return round(float(np.clip(score, -0.25, 0.20)), 4)




def softmax_from_rating(ratings: np.ndarray, tau: float | None = None) -> np.ndarray:
    """
    MATHEMATICALLY RIGOROUS softmax with overflow protection and validation.

    Guarantees:
    1. No NaN/Inf in output
    2. Probabilities sum to exactly 1.0 (within floating point precision)
    3. All values in [0, 1]
    4. Numerically stable for large ratings
    """
    # VALIDATION: Empty array
    if ratings.size == 0:
        return np.array([])

    # VALIDATION: Remove NaN/Inf from input
    ratings_clean = np.array(ratings, dtype=float)
    if np.any(~np.isfinite(ratings_clean)):
        # Replace NaN/Inf with median of finite values
        finite_mask = np.isfinite(ratings_clean)
        if np.any(finite_mask):
            median_val = np.median(ratings_clean[finite_mask])
        else:
            median_val = 0.0
        ratings_clean[~finite_mask] = median_val

    # GOLD STANDARD: Temperature parameter with strict bounds
    _tau = tau if tau is not None else MODEL_CONFIG["softmax_tau"]
    _tau = max(_tau, 1e-6)  # Prevent division by zero
    _tau = min(_tau, 1e6)  # Prevent numerical instability

    # ADAPTIVE TAU: Scale temperature to rating spread so probabilities stay realistic
    # Target: max-min spread of 2.5-4.0 in softmax space → sensible 10:1 to 50:1 ratios
    rating_spread = np.max(ratings_clean) - np.min(ratings_clean)
    if rating_spread > 0:
        # Ensure the softmax-space spread stays in [2.0, 5.0] range
        # This maps to probability ratios of ~7:1 to ~150:1 (realistic for racing)
        target_spread = 3.5  # Sweet spot for 8-12 horse fields
        adaptive_tau = max(_tau, rating_spread / target_spread)
        _tau = adaptive_tau

    # NUMERICAL STABILITY: Subtract max before exp (prevents overflow)
    x = ratings_clean / _tau
    x_max = np.max(x)
    x_shifted = x - x_max

    # OVERFLOW PROTECTION: Clip extreme values
    x_shifted = np.clip(x_shifted, -700, 700)  # exp(±700) is within float64 range

    # COMPUTE: Exponential
    ex = np.exp(x_shifted)

    # VALIDATION: Check for zero sum (should never happen with above protections)
    ex_sum = np.sum(ex)
    if ex_sum <= 1e-12:
        # Fallback: uniform distribution
        return np.ones_like(ex) / len(ex)

    # NORMALIZE: Guaranteed sum to 1.0
    p = ex / ex_sum

    # FINAL VALIDATION: Ensure all values are valid probabilities
    p = np.clip(p, 0.0, 1.0)

    # GOLD STANDARD: Exact normalization (eliminate floating point drift)
    p_sum = np.sum(p)
    if p_sum > 0:
        p = p / p_sum

    return p




def compute_ppi(df_styles: pd.DataFrame) -> dict:
    """
    GOLD STANDARD Pace Pressure Index calculation with mathematical rigor.

    PPI Formula: (E + EP - P - S) * multiplier / field_size

    Guarantees:
    1. Always returns valid numeric PPI
    2. Per-horse tailwinds are bounded and validated
    3. No division by zero
    4. Handles empty/invalid input gracefully
    """
    # VALIDATION: Input checks
    if df_styles is None or df_styles.empty:
        return {"ppi": 0.0, "by_horse": {}}

    # EXTRACT: Styles, names, strengths with validation
    styles, names, strengths = [], [], []
    for _, row in df_styles.iterrows():
        stl = (
            row.get("Style")
            or row.get("OverrideStyle")
            or row.get("DetectedStyle")
            or ""
        )
        stl = _normalize_style(stl)
        styles.append(stl)

        # VALIDATION: Ensure horse name exists
        horse_name = row.get("Horse", "")
        if not horse_name or pd.isna(horse_name):
            horse_name = f"Unknown_{len(names) + 1}"
        names.append(str(horse_name))

        strengths.append(row.get("StyleStrength", "Solid"))

    # COUNT: Style distribution
    counts = {"E": 0, "E/P": 0, "P": 0, "S": 0}
    for stl in styles:
        if stl in counts:
            counts[stl] += 1

    # VALIDATION: Prevent division by zero
    total = sum(counts.values())
    if total == 0:
        return {"ppi": 0.0, "by_horse": {}}

    # GOLD STANDARD: PPI calculation with bounds checking
    ppi_multiplier = MODEL_CONFIG.get("ppi_multiplier", 1.0)
    ppi_numerator = counts["E"] + counts["E/P"] - counts["P"] - counts["S"]
    ppi_val = (ppi_numerator * ppi_multiplier) / total

    # VALIDATION: Ensure PPI is finite
    if not np.isfinite(ppi_val):
        ppi_val = 0.0

    # COMPUTE: Per-horse tailwinds with validation
    by_horse = {}
    strength_weights = MODEL_CONFIG.get("style_strength_weights", {})
    tailwind_factor = MODEL_CONFIG.get("ppi_tailwind_factor", 1.0)

    # VALIDATION: Ensure factors are finite
    if not np.isfinite(tailwind_factor):
        tailwind_factor = 1.0

    for stl, nm, strength in zip(styles, names, strengths, strict=False):
        # VALIDATION: Get weight with fallback
        wt = strength_weights.get(str(strength), 0.8)
        if not np.isfinite(wt) or wt < 0:
            wt = 0.8

        # COMPUTE: Tailwind adjustment
        if stl in ("E", "E/P"):
            tailwind = tailwind_factor * wt * ppi_val
        elif stl == "S":
            tailwind = -tailwind_factor * wt * ppi_val
        else:
            tailwind = 0.0

        # VALIDATION: Ensure finite and bounded
        if not np.isfinite(tailwind):
            tailwind = 0.0
        tailwind = np.clip(tailwind, -10.0, 10.0)  # Reasonable bounds

        by_horse[nm] = round(tailwind, 3)

    return {"ppi": round(ppi_val, 3), "by_horse": by_horse}




def apply_enhancements_and_figs(
    ratings_df: pd.DataFrame,
    pp_text: str,
    _processed_weights: dict[str, float],
    _chaos_index: float,
    _track_name: str,
    _surface_type: str,
    _distance_txt: str,
    _race_type: str,
    angles_per_horse: dict[str, pd.DataFrame],
    _pedigree_per_horse: dict[str, dict],
    figs_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Applies speed figure enhancements (R_ENHANCE_ADJ) to the base ratings.
    """
    if ratings_df is None or ratings_df.empty:
        return ratings_df

    df = ratings_df.copy()

    # CRITICAL FIX (Feb 10, 2026): Detect if unified engine was used.
    # If so, speed figures and angles are ALREADY embedded in the Rating.
    # Applying R_ENHANCE_ADJ again would DOUBLE-COUNT these factors.
    _unified_engine_used = (
        "Parsing_Confidence" in df.columns and df["Parsing_Confidence"].max() > 0
    )
    if _unified_engine_used:
        # Only apply SAVANT bonuses (lasix, trip, pace, bounce, workout)
        # since the unified engine does NOT include these specific enhancements
        df["R_ENHANCE_ADJ"] = 0.0
        # Skip speed figure and angle sections, jump to savant-only path
        _skip_speed_and_angles = True
    else:
        _skip_speed_and_angles = False

    # --- SPEED FIGURE LOGIC ---
    if _skip_speed_and_angles:
        pass  # Already embedded in unified engine output
    elif figs_df.empty or "AvgTop2" not in figs_df.columns:
        # No figures were parsed or dataframe is empty
        st.caption("No speed figures parsed. R_ENHANCE_ADJ set to 0.")
        df["R_ENHANCE_ADJ"] = 0.0
    else:
        # Traditional path: speed figures already included in base R via
        # compute_bias_ratings cspeed component. Do NOT add them again here.
        # Initialize R_ENHANCE_ADJ to 0; angle + savant bonuses applied below.
        df["R_ENHANCE_ADJ"] = 0.0

    # --- END SPEED FIGURE LOGIC ---

    # --- ANGLES BONUS LOGIC ---
    # Add bonus for horses with positive angles
    ANGLE_BONUS = 0.10  # Bonus per positive angle
    MAX_ANGLE_BONUS = 0.80  # Cap at 8 angles worth (matches unified engine)
    df["AngleBonus"] = 0.0
    for horse, angles_df in angles_per_horse.items():
        if angles_df is not None and not angles_df.empty:
            # Count positive angles (rows in the dataframe)
            num_angles = len(angles_df)
            if num_angles > 0:
                bonus = min(num_angles * ANGLE_BONUS, MAX_ANGLE_BONUS)
                df.loc[df["Horse"] == horse, "AngleBonus"] = bonus

    if not _skip_speed_and_angles:
        df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"] + df["AngleBonus"]
    df = df.drop(columns=["AngleBonus"])
    # --- END ANGLES LOGIC ---

    # --- SAVANT ENHANCEMENT LOGIC (Jan 2026) ---
    # NEW: Lasix, Trip Handicapping, Workout Patterns, Pace Figures, Bounce Detection
    df["SavantBonus"] = 0.0

    # Extract horse blocks for savant analysis (name-based mapping, block text only)
    horse_blocks = split_into_horse_chunks(pp_text)
    block_map = {}
    # Build name→block_text lookup from parsed chunks (NOT index-based — df may be sorted by rating)
    pp_block_by_name = {
        chunk_name: block_text for _post, chunk_name, block_text in horse_blocks
    }
    for horse_name_iter in df["Horse"]:
        if horse_name_iter in pp_block_by_name:
            block_map[horse_name_iter] = pp_block_by_name[horse_name_iter]

    for horse_name, block in block_map.items():
        savant_bonus = 0.0

        # 1. Lasix detection
        savant_bonus += detect_lasix_change(block)

        # 2. Trip handicapping
        positions = parse_fractional_positions(block)
        savant_bonus += calculate_trip_quality(positions, field_size=len(df))

        # 3. E1/E2/LP pace analysis — now with BRIS Pace/Speed Par adjustment
        pace_data = parse_e1_e2_lp_values(block)
        _pars = st.session_state.get("pace_speed_pars", {})
        savant_bonus += analyze_pace_figures(
            pace_data["e1"],
            pace_data["e2"],
            pace_data["lp"],
            e1_par=_pars.get("e1_par") or None,
            e2_par=_pars.get("e2_par") or None,
            lp_par=_pars.get("lp_par") or None,
        )

        # 4. Bounce detection
        speed_figs = parse_speed_figures_for_block(block)
        savant_bonus += detect_bounce_risk(speed_figs)

        # 5. Workout pattern bonus (from enhanced workout parsing)
        workout_data = parse_workout_data(block)
        savant_bonus += workout_data.get("pattern_bonus", 0.0)

        df.loc[df["Horse"] == horse_name, "SavantBonus"] = savant_bonus

    df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"] + df["SavantBonus"]
    df = df.drop(columns=["SavantBonus"])
    # --- END SAVANT LOGIC ---

    # OPTIMIZED Feb 9 2026: Raised cap from [-1.0, 1.5] to [-2.0, 3.0]
    # With speed_fig_weight=0.15, a 20-point fig advantage = 3.0 boost.
    # Old cap was clipping legitimate speed advantages.
    df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"].fillna(0.0)  # Ensure no NaNs
    df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"].clip(
        -2.0, 3.0
    )  # Cap speed+angles+savant layer
    df["R"] = (df["R"].astype(float) + df["R_ENHANCE_ADJ"].astype(float)).round(2)

    return df




def overlay_table(
    fair_probs: dict[str, float], offered: dict[str, float]
) -> pd.DataFrame:
    """
    GOLD STANDARD overlay calculation with mathematical rigor.

    Guarantees:
    1. No division by zero
    2. All calculations are finite
    3. Proper EV formula application
    4. Input validation
    """
    # VALIDATION: Input checks
    if not fair_probs or not offered:
        return pd.DataFrame()

    rows = []
    for h, p in fair_probs.items():
        # VALIDATION: Ensure probability is valid
        if not np.isfinite(p) or p < 0 or p > 1:
            continue

        off_dec = offered.get(h)
        if off_dec is None:
            continue

        # VALIDATION: Ensure odds are positive and finite
        if not np.isfinite(off_dec) or off_dec <= 0:
            continue

        # GOLD STANDARD: Safe probability calculation
        if off_dec > 1e-9:
            off_prob = 1.0 / off_dec
        else:
            off_prob = 0.0

        # VALIDATION: Ensure off_prob is valid
        if not np.isfinite(off_prob):
            off_prob = 0.0
        off_prob = min(off_prob, 1.0)  # Cap at 100%

        # GOLD STANDARD: Expected Value calculation
        # EV = (odds × win_prob) - (1 × loss_prob)
        # EV = (off_dec - 1) × p - (1 - p)
        ev = (off_dec - 1) * p - (1 - p)

        # VALIDATION: Ensure EV is finite
        if not np.isfinite(ev):
            ev = 0.0

        rows.append(
            {
                "Horse": h,
                "Fair %": round(p * 100, 2),
                "Fair (AM)": fair_to_american_str(p),
                "Board (dec)": round(off_dec, 3),
                "Board %": round(off_prob * 100, 2),
                "Edge (pp)": round((p - off_prob) * 100, 2),
                "EV per $1": round(ev, 3),
                "Overlay?": "YES" if off_prob < p else "NO",
            }
        )

    return pd.DataFrame(rows)




def calculate_layoff_factor(
    days_since_last: int,
    num_workouts: int | None = None,
    workout_pattern_bonus: float = 0.0,
) -> float:
    """OPTIMIZED Feb 9 2026: Layoff impact with workout mitigation.

    Returns: adjustment factor (-3.0 to +0.5).
    Key changes: 60-120d brackets gentler (strategic freshening window),
    workout mitigation 15%/workout up to 60% recovery, max penalty -3.0.
    Validated on Oaklawn R9 (Air of Defiance #2).

    NOTE: num_workouts defaults to None (not 0) so callers that fail to
    provide workout data produce a debug trace rather than silently skipping
    the workout mitigation block.
    """
    _num_workouts = num_workouts if num_workouts is not None else 0
    if num_workouts is None:
        logger.debug(
            "calculate_layoff_factor: num_workouts=None, workout mitigation disabled"
        )
    if days_since_last <= 14:
        base = 0.5
    elif days_since_last <= 30:
        base = 0.3
    elif days_since_last <= 45:
        base = 0.0
    elif days_since_last <= 60:
        base = -0.2
    elif days_since_last <= 90:
        base = -0.5  # Was -0.8 — 60-90 days is strategic freshening
    elif days_since_last <= 120:
        base = -1.0  # Was -1.5 — still competitive window with workouts
    elif days_since_last <= 180:
        base = -2.0
    else:
        base = -3.0
    # Workout mitigation (up to 60% of penalty recovered)
    if base < 0 and _num_workouts > 0:
        work_credit = min(_num_workouts * 0.15, 0.60)  # 15% per workout
        base *= 1.0 - work_credit
        base += workout_pattern_bonus
    return round(max(base, -3.0), 2)




def evaluate_first_time_starter(
    pedigree: dict, angles_df: pd.DataFrame, workout_data: dict, horse_block: str
) -> float:
    """
    Comprehensive first-time starter evaluation.
    Returns: debut rating from -2.0 to +3.5
    """
    debut_rating = 0.0

    # 1. PEDIGREE QUALITY (weight: heavy for debuts)
    sire_spi = pedigree.get("sire_spi")
    damsire_spi = pedigree.get("damsire_spi")

    # SPI analysis
    for spi in [sire_spi, damsire_spi]:
        if pd.notna(spi):
            spi_val = float(spi)
            if spi_val >= 115:  # Elite sire
                debut_rating += 0.8
            elif spi_val >= 105:  # Very good
                debut_rating += 0.5
            elif spi_val >= 95:  # Above average
                debut_rating += 0.2
            elif spi_val < 85:  # Below average
                debut_rating -= 0.3

    # First-time winner percentage (sire/damsire)
    sire_1st = pedigree.get("sire_1st")
    damsire_1st = pedigree.get("damsire_1st")

    for pct in [sire_1st, damsire_1st]:
        if pd.notna(pct):
            pct_val = float(pct)
            if pct_val >= 18:  # Excellent debut sire
                debut_rating += 0.6
            elif pct_val >= 14:  # Good
                debut_rating += 0.3
            elif pct_val >= 10:  # Average
                debut_rating += 0.1
            elif pct_val < 7:  # Poor
                debut_rating -= 0.2

    # 2. WORKOUT PATTERN
    if workout_data["num_recent"] >= 3:  # Well-prepared
        debut_rating += 0.4

        if workout_data["best_time"] is not None:
            # Fast workouts indicate readiness
            if workout_data["best_time"] < 48.0:  # Blazing 4f equivalent
                debut_rating += 0.6
            elif workout_data["best_time"] < 49.5:  # Very good
                debut_rating += 0.3
            elif workout_data["best_time"] < 51.0:  # Solid
                debut_rating += 0.1
    elif workout_data["num_recent"] < 2:  # Underprepared
        debut_rating -= 0.5

    # 3. TRAINER DEBUT ANGLES
    if angles_df is not None and not angles_df.empty:
        angle_text = " ".join(angles_df["Category"].astype(str)).lower()

        if "1st time str" in angle_text or "debut" in angle_text:
            debut_rating += 0.5  # Trainer pattern recognition

        if "maiden sp wt" in angle_text or "maiden special weight" in angle_text:
            debut_rating += 0.3  # MSW debut angle

        # High ROI trainer debut pattern
        if angles_df is not None and "ROI" in angles_df.columns:
            debut_angles = angles_df[
                angles_df["Category"].str.contains(
                    "debut|1st time", case=False, na=False
                )
            ]
            if not debut_angles.empty:
                avg_roi = debut_angles["ROI"].mean()
                if avg_roi > 1.5:  # Strong positive ROI
                    debut_rating += 0.8
                elif avg_roi > 1.0:
                    debut_rating += 0.4

    # 4. RACE TYPE CONTEXT
    if "maiden special weight" in horse_block.lower():
        debut_rating += 0.2  # MSW is better spot than MCL for debuts
    elif "maiden claiming" in horse_block.lower():
        debut_rating -= 0.3  # MCL debut is tougher

    return float(np.clip(debut_rating, -2.0, 3.5))




def calculate_form_cycle_rating(
    horse_block: str, pedigree: dict, angles_df: pd.DataFrame
) -> float:
    """
    Comprehensive form cycle rating considering:
    1. Layoff analysis
    2. Recent form trend
    3. First-time starter special evaluation

    Returns: Form rating from -3.0 to +3.0
    """

    # Check if first-time starter (no race history)
    recent_races = parse_recent_races_detailed(horse_block)

    if len(recent_races) == 0:
        # FIRST-TIME STARTER - use comprehensive debut evaluation
        workout_data = parse_workout_data(horse_block)
        return evaluate_first_time_starter(
            pedigree, angles_df, workout_data, horse_block
        )

    # EXPERIENCED HORSE - analyze form cycle
    form_rating = 0.0

    # 1. Layoff factor (with workout mitigation)
    days_since_last = recent_races[0]["days_ago"] if recent_races else 999
    workout_data = parse_workout_data(horse_block)
    layoff_adj = calculate_layoff_factor(
        days_since_last,
        num_workouts=workout_data.get("num_recent", 0),
        workout_pattern_bonus=workout_data.get("pattern_bonus", 0.0),
    )
    form_rating += layoff_adj

    # 2. Form trend
    recent_finishes = [r["finish"] for r in recent_races]
    trend_adj = calculate_form_trend(recent_finishes)
    form_rating += trend_adj

    # 3. Consistency bonus (regularly competitive)
    if len(recent_finishes) >= 4:
        top_3_finishes = sum(1 for f in recent_finishes[:4] if f <= 3)
        if top_3_finishes >= 3:  # 3 of last 4 in top 3
            form_rating += 0.8
        elif top_3_finishes >= 2:  # 2 of last 4
            form_rating += 0.4

    # 4. Recent win bonus
    if recent_finishes and recent_finishes[0] == 1:  # Last race winner
        form_rating += 0.6

        # Wire-to-wire or repeat win pattern
        if len(recent_finishes) >= 2 and recent_finishes[1] == 1:
            form_rating += 0.4  # Back-to-back wins

    return float(np.clip(form_rating, -2.0, 3.0))




def calculate_comprehensive_class_rating(
    today_purse: int,
    today_race_type: str,
    horse_block: str,
    pedigree: dict,
    angles_df: pd.DataFrame,
    pp_text: str = "",  # NEW: Full PP text for race class parser
    _distance_furlongs: float = 0.0,  # NEW: Race distance (reserved for future use)
    _surface_type: str = "Dirt",  # NEW: Surface type (reserved for future use)
) -> float:
    """
    Comprehensive class rating considering:
    1. Today's race class hierarchy (using race_class_parser)
    2. Today's purse vs recent purse levels
    3. Race type hierarchy with PROPER acronym understanding
    4. Class movement trend with form adjustment
    5. Pedigree quality indicators
    6. Angle-based class boosts

    NEW: Uses race_class_parser to properly understand ALL race type acronyms
    (MCL, CLM, MSW, ALW, AOC, SOC, STK, CST, G1, G2, G3, etc.) and calculate
    weights based on race type + purse amount.

    Returns: Class rating from -3.0 to +6.0
    """

    # STEP 1: Get race class analysis from parser (if available)
    race_class_data = None
    today_class_weight = 0.0
    today_hierarchy_level = 0

    if RACE_CLASS_PARSER_AVAILABLE and pp_text:
        try:
            race_class_data = parse_and_calculate_class(pp_text)
            today_class_weight = race_class_data["weight"]["class_weight"]
            today_hierarchy_level = race_class_data["hierarchy"]["final_level"]

        except Exception:
            # Fall back to legacy method silently
            pass

    # Race type hierarchy scoring (ENHANCED with race_class_parser data)
    # Based on industry-standard US horse racing hierarchy (1-7 scale)
    race_type_scores = {
        # Maiden (Level 1)
        "msw": 1,
        "maiden special weight": 1,
        "md sp wt": 1,
        "mdn": 1,
        "md": 1,
        "maiden": 1,
        "moc": 1,
        "maiden optional claiming": 1,
        "mcl": 1,
        "maiden claiming": 1,
        "md cl": 1,
        "mdnclm": 1,
        "mdc": 1,
        "msc": 1,
        "maiden starter claiming": 1,
        # Claiming (Level 2)
        "clm": 2,
        "claiming": 2,
        "cl": 2,
        "clg": 2,
        "c": 2,
        "wcl": 2,
        "waiver claiming": 2,
        "waiver": 2,
        "clh": 2,
        "claiming handicap": 2,
        "cst": 2,
        "claiming stakes": 2,
        "clmstk": 2,  # Low-end
        # Starter Allowance (Level 3)
        "sta": 3,
        "starter allowance": 3,
        "starter optional claiming": 4,
        "str": 3,
        "starter": 3,
        "shp": 3,
        "starter handicap": 3,
        "stb": 3,
        "statebred": 3,
        "state bred": 3,
        # Allowance / Trial (Level 4)
        "alw": 4,
        "allowance": 4,
        "a": 4,
        "n1x": 4,
        "nw1": 4,
        "allowance non-winners of 1": 4,
        "n2x": 4,
        "nw2": 4,
        "allowance non-winners of 2": 4,
        "n3x": 4,
        "nw3": 4,
        "allowance non-winners of 3": 4,
        "n1l": 4,
        "n2l": 4,
        "n3l": 4,  # Lifetime conditions
        "opt": 4,
        "optional": 4,
        "trl": 4,
        "trial": 4,
        # Optional Claiming / High Allowance (Level 5)
        "oc": 5,
        "ocl": 5,
        "optional claiming": 5,
        "aoc": 5,
        "allowance optional claiming": 5,
        "ao": 5,
        "soc": 5,
        # "starter optional claiming": 5,  # Duplicate key — already mapped to level 4 above
        # Handicap (Level 6)
        "hcp": 6,
        "handicap": 6,
        "h": 6,
        "©hcp": 6,
        "©": 6,
        "och": 6,
        "optional claiming handicap": 6,
        # Stakes (Level 7) - Graded, Listed, Non-Graded, Specialty
        "stk": 7,
        "stakes": 7,
        "s": 7,
        "sst": 7,
        "starter stakes": 7,
        "n": 7,
        "non-graded stakes": 7,
        "l": 7,
        "lr": 7,
        "listed": 7,
        "listed stakes": 7,
        "g3": 7,
        "grade 3": 7,
        "gr3": 7,
        "grade3": 7,
        "g2": 7,
        "grade 2": 7,
        "gr2": 7,
        "grade2": 7,
        "g1": 7,
        "grade 1": 7,
        "gr1": 7,
        "grade1": 7,
        "fut": 7,
        "futurity": 7,
        "ftr": 7,
        "der": 7,
        "derby": 7,
        "dby": 7,
        "invit": 7,
        "invitational": 7,
        "inv": 7,
        # Special (Level 0)
        "mat": 0,
        "match": 0,
        "match race": 0,
        "tr": 0,
        "training": 0,
        "training race": 0,
    }

    today_type_norm = str(today_race_type).strip().lower()

    # Use hierarchy level from parser if available, otherwise use legacy scores
    if today_hierarchy_level > 0:
        today_score = today_hierarchy_level
    else:
        today_score = race_type_scores.get(today_type_norm, 3.5)

    # Parse recent races
    recent_races = parse_recent_class_levels(horse_block)

    class_rating = 0.0

    # CRITICAL: Check if horse was COMPETITIVE in recent races
    was_competitive = False
    if recent_races:
        finishes = [
            r.get("finish_pos", 0)
            for r in recent_races[:3]
            if r.get("finish_pos", 0) > 0
        ]
        if finishes:
            # Consider competitive if finished in top 3 in any of last 3 races
            recent_top3_count = sum(1 for f in finishes if f <= 3)
            was_competitive = recent_top3_count >= 1

    # 1. PURSE COMPARISON (weight: heavy) - FORM-ADJUSTED
    if recent_races and today_purse > 0:
        recent_purses = [r["purse"] for r in recent_races if r["purse"] > 0]
        if recent_purses:
            avg_recent_purse = np.mean(recent_purses)
            purse_ratio = (
                today_purse / avg_recent_purse if avg_recent_purse > 0 else 1.0
            )

            # ENHANCED: Use race class weight from parser to adjust purse impact
            purse_weight_multiplier = 1.0
            if race_class_data:
                # Higher quality races = purse difference matters more
                quality = race_class_data["summary"]["quality"]
                if quality == "Elite":
                    purse_weight_multiplier = 1.5  # G1/G2 - purse differences critical
                elif quality == "High":
                    purse_weight_multiplier = 1.2  # G3/Stakes - purse matters
                elif quality == "Medium":
                    purse_weight_multiplier = 1.0  # Standard weight
                else:
                    purse_weight_multiplier = 0.8  # Low class - purse less predictive

            # Purse movement scoring (ADJUSTED by race quality)
            if purse_ratio >= 1.5:  # Major step up
                class_rating -= 1.2 * purse_weight_multiplier
            elif purse_ratio >= 1.2:  # Moderate step up
                class_rating -= 0.6 * purse_weight_multiplier
            elif purse_ratio >= 0.8 and purse_ratio <= 1.2:  # Same class
                class_rating += 0.8 * purse_weight_multiplier
            elif purse_ratio >= 0.6:  # Class drop (FORM-ADJUSTED)
                if was_competitive:
                    class_rating += 0.8 * purse_weight_multiplier  # Legitimate drop
                else:
                    class_rating += 0.2 * purse_weight_multiplier  # Minimal bonus
            else:  # Major drop
                if was_competitive:
                    class_rating += 1.0 * purse_weight_multiplier  # Should dominate
                else:
                    class_rating -= 0.3 * purse_weight_multiplier  # Warning flag

    # 2. RACE TYPE PROGRESSION
    if recent_races:
        recent_types = [r["race_type"].lower() for r in recent_races]
        recent_scores = [race_type_scores.get(rt, 3.5) for rt in recent_types]
        avg_recent_type = np.mean(recent_scores)

        type_diff = today_score - avg_recent_type

        if type_diff >= 2.0:  # Major class rise (e.g., ALW → G1)
            class_rating -= 1.5
        elif type_diff >= 1.0:  # Moderate rise (e.g., CLM → ALW)
            class_rating -= 0.8
        elif abs(type_diff) < 0.5:  # Same class level
            class_rating += 0.5
        elif type_diff <= -1.0:  # Dropping in class
            class_rating += 1.2

    # 3. PEDIGREE QUALITY BOOST (more important in higher-class races)
    spi = pedigree.get("sire_spi") or pedigree.get("damsire_spi")
    if pd.notna(spi):
        spi_val = float(spi)
        pedigree_multiplier = 1.0

        # Pedigree matters MORE in graded stakes
        if race_class_data and race_class_data["summary"]["is_graded"]:
            pedigree_multiplier = 1.5
        elif race_class_data and race_class_data["summary"]["is_stakes"]:
            pedigree_multiplier = 1.2

        if spi_val >= 110:
            class_rating += 0.4 * pedigree_multiplier
        elif spi_val >= 100:
            class_rating += 0.2 * pedigree_multiplier

    # 4. ANGLE-BASED CLASS INDICATORS
    if angles_df is not None and not angles_df.empty:
        angle_text = " ".join(angles_df["Category"].astype(str)).lower()

        # Class rise angles
        if "first time starter" in angle_text or "1st time str" in angle_text:
            class_rating += 0.5  # Debut = unknown class, slight boost
        if "2nd career" in angle_text:
            class_rating += 0.3
        if "shipper" in angle_text:
            class_rating += 0.2  # Shipper often means stepping up

    # 5. CLAIMING PRICE GRANULARITY (if claiming race)
    claiming_bonus = 0.0
    if "clm" in today_type_norm or "claiming" in today_type_norm:
        recent_claiming = parse_claiming_prices(horse_block)
        claiming_bonus = analyze_claiming_price_movement(recent_claiming, today_purse)

    class_rating += claiming_bonus

    # 6. FINAL ADJUSTMENT: Apply race class weight from parser
    if race_class_data:
        # Normalize class weight (0-15 range) to our rating scale (-3 to +6)
        # Higher class weight = tougher competition = adjust rating accordingly
        normalized_weight = (today_class_weight - 5.0) * 0.3  # Scale to +/- range
        class_rating += normalized_weight

    return np.clip(class_rating, -3.0, 6.0)




def _angles_pedigree_tweak(
    name: str, race_surface: str, race_bucket: str, race_cond: str
) -> float:
    """
    Small, capped additive tweak that folds pedigree + common angles into Cclass.
    Positive values help; negatives hurt.
    """
    ped = pedigree_per_horse.get(name, {}) or {}
    ang = angles_per_horse.get(name)
    tweak = 0.0

    # 1) Pedigree AWD vs today's distance bucket
    awds = [x for x in [ped.get("sire_awd"), ped.get("damsire_awd")] if pd.notna(x)]
    awd_mean = float("nan")  # Initialize before conditional block
    # Validate AWD data exists and is valid BEFORE using
    if awds:
        awd_mean = float(np.nanmean(awds))
        if pd.notna(awd_mean):  # Use pandas notna for clarity
            if race_bucket == "≤6f":
                if awd_mean <= 6.5:
                    tweak += MODEL_CONFIG["ped_dist_bonus"]
                elif awd_mean >= 7.5:
                    tweak += MODEL_CONFIG["ped_dist_penalty"]
            elif race_bucket == "8f+":
                if awd_mean >= 7.5:
                    tweak += MODEL_CONFIG["ped_dist_bonus"]
                elif awd_mean <= 6.5:
                    tweak += MODEL_CONFIG["ped_dist_penalty"]
            else:  # 6.5-7f bucket
                if 6.3 <= awd_mean <= 7.7:
                    tweak += MODEL_CONFIG["ped_dist_neutral_bonus"]

    # Sprint/debut pop from 1st% in true sprints
    if race_bucket == "≤6":
        for v in [ped.get("sire_1st"), ped.get("damsire_1st")]:
            if pd.notna(v):
                if float(v) >= MODEL_CONFIG["ped_first_pct_threshold"]:
                    tweak += MODEL_CONFIG["ped_first_pct_bonus"]

    # 2) Angles
    if ang is not None and not ang.empty:
        cats = " ".join(ang["Category"].astype(str).tolist()).lower()
        if "1st time str" in cats or "debut mdnspwt" in cats or "maiden sp wt" in cats:
            tweak += (
                MODEL_CONFIG["angle_debut_msw_bonus"]
                if race_type_detected == "maiden special weight"
                else MODEL_CONFIG["angle_debut_other_bonus"]
            )
            if race_bucket == "≤6f":
                tweak += MODEL_CONFIG["angle_debut_sprint_bonus"]
        if "2nd career" in cats:
            tweak += MODEL_CONFIG["angle_second_career_bonus"]
        if "turf to dirt" in cats and race_surface.lower() == "dirt":
            tweak += MODEL_CONFIG["angle_surface_switch_bonus"]
        if "dirt to tur" in cats and race_surface.lower() == "turf":
            tweak += MODEL_CONFIG["angle_surface_switch_bonus"]
        if "blinkers on" in cats:
            tweak += MODEL_CONFIG["angle_blinkers_on_bonus"]
        if "blinkers off" in cats:
            tweak += MODEL_CONFIG["angle_blinkers_off_bonus"]
        if "shipper" in cats:
            tweak += MODEL_CONFIG["angle_shipper_bonus"]
        try:
            if "ROI" in ang.columns:
                pos_ct = int((ang["ROI"] > 0).sum())
                neg_ct = int((ang["ROI"] < 0).sum())
                tweak += min(
                    MODEL_CONFIG["angle_roi_pos_max_bonus"],
                    MODEL_CONFIG["angle_roi_pos_per_bonus"] * pos_ct,
                )
                tweak -= min(
                    MODEL_CONFIG["angle_roi_neg_max_penalty"],
                    MODEL_CONFIG["angle_roi_neg_per_penalty"] * neg_ct,
                )
        except Exception as e:
            st.warning(
                f"Error during angle ROI calculation for horse '{name}'. Error: {e}"
            )
            pass

    # 3) Condition nuance
    if (
        race_cond in {"muddy", "sloppy", "heavy"} and awd_mean == awd_mean
    ):  # Check if not NaN
        if race_bucket != "≤6f" and awd_mean >= 7.5:
            tweak += MODEL_CONFIG["angle_off_track_route_bonus"]

    return float(
        np.clip(
            round(tweak, 3),
            MODEL_CONFIG["angle_tweak_min_clip"],
            MODEL_CONFIG["angle_tweak_max_clip"],
        )
    )




def _style_bias_label_from_choice(choice: str) -> str:
    # Map single-letter selection into our style_match table buckets
    up = (choice or "").upper()
    if up in ("E", "E/P"):
        return "speed favoring"
    if up in ("P", "S"):
        return "closer favoring"
    return "fair/neutral"




def style_match_score_multi(
    running_style_biases: list, style: str, quirin: float
) -> float:
    """Calculate style match score from multiple selected running style biases.

    NA STYLE HANDLING (Feb 11, 2026):
    NA horses with high QSP (>=5) get partial credit when bias favors speed,
    since QSP indicates early-speed tendency even without a confirmed style.
    """
    if not running_style_biases:
        return 0.0

    stl = (style or "NA").upper()
    table = MODEL_CONFIG["style_match_table"]

    try:
        q = float(quirin)
    except Exception:
        q = np.nan

    total_bonus = 0.0

    # Aggregate bonuses from all selected running style biases
    for bias_choice in running_style_biases:
        # Map choice to label
        bias_label = _style_bias_label_from_choice(bias_choice)
        bias_lower = bias_label.strip().lower()

        # NA style: use QSP to infer partial bonus instead of flat 0.0
        if stl == "NA" and pd.notna(q) and q >= NA_STYLE_PARAMS["qsp_ep_threshold"]:
            if "favoring" in bias_lower and "closer" not in bias_lower:
                bonus = 0.15 * (q / 8.0)  # Scale by QSP strength
            elif "closer" in bias_lower:
                bonus = -0.10 * (q / 8.0)  # High QSP = likely NOT a closer
            else:
                bonus = 0.0  # Fair/neutral: no adjustment
        else:
            bonus = table.get(bias_lower, table["fair/neutral"]).get(stl, 0.0)

        # Add Quirin bonus if applicable (E/EP only, not NA — NA handled above)
        if (
            stl in ("E", "E/P")
            and pd.notna(q)
            and q >= MODEL_CONFIG["style_quirin_threshold"]
        ):
            bonus += MODEL_CONFIG["style_quirin_bonus"]

        if bonus > 0:  # Only add positive bonuses
            total_bonus += bonus

    return float(np.clip(total_bonus, -1.0, 1.0))




def score_quickplay_comments(comments: dict[str, list[str]]) -> float:
    """Convert parsed QuickPlay comments into a tier2 bonus/penalty.

    Positive signals get graduated bonuses; negative signals get penalties.
    Known high-impact patterns are weighted more heavily.

    Returns: float bonus/penalty (typically -1.5 to +1.5)
    """
    bonus = 0.0
    if not comments:
        return bonus

    # --- Positive signals ---
    for comment in comments.get("positive", []):
        c_lower = comment.lower()
        if "won last race" in c_lower or "won last start" in c_lower:
            bonus += 0.3  # Winner last out
        elif "winning at" in c_lower and "clip" in c_lower:
            bonus += 0.2  # Hot post/jockey/trainer stat
        elif "sharp" in c_lower and "workout" in c_lower:
            bonus += 0.2  # Sharp recent work
        elif "drops" in c_lower or "drop in class" in c_lower:
            bonus += 0.25  # Class dropper
        elif "won" in c_lower:
            bonus += 0.15  # General positive
        elif "top trainer" in c_lower or "leading" in c_lower:
            bonus += 0.15
        elif "improving" in c_lower or "speed figures improving" in c_lower:
            bonus += 0.2
        else:
            bonus += 0.1  # Generic positive

    # --- Negative signals ---
    for comment in comments.get("negative", []):
        c_lower = comment.lower()
        if "poor speed figures" in c_lower or "poor speed" in c_lower:
            bonus -= 0.3  # Bad speed = major red flag
        elif "well below" in c_lower and (
            "avg winning" in c_lower or "average winning" in c_lower
        ):
            bonus -= 0.35  # Best speed below race average = severe
        elif "moves up in class" in c_lower or "class from last" in c_lower:
            bonus -= 0.2  # Stepping up
        elif "has not raced" in c_lower:
            # Parse days — "Has not raced in 48 days"
            days_match = re.search(r"(\d+)\s*days", c_lower)
            if days_match:
                days = int(days_match.group(1))
                if days >= 90:
                    bonus -= 0.3  # Long layoff
                elif days >= 45:
                    bonus -= 0.15  # Moderate layoff
                else:
                    bonus -= 0.05  # Short rest
            else:
                bonus -= 0.1
        elif "no wins" in c_lower or "winless" in c_lower:
            bonus -= 0.2
        elif "poor" in c_lower or "worst" in c_lower:
            bonus -= 0.15
        elif "never won" in c_lower:
            bonus -= 0.2
        else:
            bonus -= 0.1  # Generic negative

    return float(np.clip(bonus, -1.5, 1.5))




def calculate_weekly_bias_amplifier(impact_values: dict[str, float]) -> float:
    """
    Detect EXTREME weekly track biases and return a multiplier for track_bias_mult.

    When weekly impact values show extreme style biases (e.g., E = 2.05, S = 0.32),
    the standard track_bias_mult (1.0-1.15) is woefully inadequate.

    R9 FIX: Also checks absolute min impact value, not just spread.
    Case: E=1.17, E/P=1.14, P=1.18, S=0.32 → spread=0.86 (only 1.1x)
    but S at 0.32 is severely suppressed. Now ensures at least 1.2x when
    any style is below 0.40 impact.

    Returns: Multiplier to MULTIPLY with existing track_bias_mult (1.0 = no change).
    """
    if not impact_values:
        return 1.0

    max_impact = max(impact_values.values()) if impact_values else 1.0
    min_impact = min(impact_values.values()) if impact_values else 1.0

    # Spread = difference between most and least favored styles
    spread = max_impact - min_impact

    if spread >= 2.0:
        amp = 1.8  # Extreme bias week (e.g., E=2.05 S=0.32)
    elif spread >= 1.5:
        amp = 1.5  # Strong bias week
    elif spread >= 1.0:
        amp = 1.3  # Moderate bias week
    elif spread >= 0.5:
        amp = 1.1  # Slight bias week
    else:
        amp = 1.0

    # R9 FIX: Additional floor when any style is severely suppressed
    # Catches cases where spread is moderate but one style is crushed
    # (e.g., E≈1.2, P≈1.2, S=0.32 → spread=0.86 but S is dead)
    if min_impact < 0.40:
        amp = max(amp, 1.2)  # At least 1.2x when a style is severely suppressed
    elif min_impact < 0.60:
        amp = max(amp, 1.15)  # At least 1.15x when a style is notably weak

    return amp




def calculate_style_vs_weekly_bias_bonus(
    style: str, impact_values: dict[str, float]
) -> float:
    """
    ENHANCEMENT 4: Apply bonus/penalty based on running style vs weekly bias data.

    Oaklawn R1 Audit: Stalker impact was 0.32 (heavily suppressed) but model
    ranked S-type Tiffany Twist #1. E-type Tell Me When (impact 2.05) was #7.
    Oaklawn R9 Audit: E/P She's Storming (impact 1.14-1.17) in dead zone getting 0.

    This function rewards styles favored by weekly bias and penalizes suppressed ones.
    FIXED: E/P now checks both E and E/P keys and uses better one.
    FIXED: Eliminated dead zone between 0.80-1.30 — now graduated scale.

    Returns: Bonus/penalty to add directly to tier2_bonus.
    """
    if not impact_values or not style:
        return 0.0

    # Map running style to the relevant impact value(s)
    style_upper = str(style).upper().strip()
    impact = None

    if style_upper == "E/P":
        # E/P horses benefit from BOTH E and E/P bias — use the better one
        e_impact = impact_values.get("E", 0.0)
        ep_impact = impact_values.get("E/P", 0.0)
        impact = max(e_impact, ep_impact) if (e_impact or ep_impact) else None
    elif style_upper == "E":
        impact = impact_values.get("E")
    elif style_upper in ("P", "P/S"):
        impact = impact_values.get("P")
    elif style_upper in ("S", "S/P"):
        impact = impact_values.get("S")
    elif style_upper == "C":
        impact = impact_values.get("C")

    if impact is None:
        return 0.0

    # GRADUATED SCALE — no dead zones
    # PENALTY for suppressed styles
    if impact < 0.40:
        return -1.5  # Severe suppression (Oaklawn S at 0.32)
    elif impact < 0.60:
        return -1.0  # Strong suppression
    elif impact < 0.80:
        return -0.5  # Moderate suppression
    elif impact < 1.00:
        return -0.2  # Slight below-average

    # BONUS for favored styles
    # TUP R6 FIX (Feb 19, 2026): Halved bonus scale. When 62% of the field
    # (5/8 E/P horses) all get +2.0, the bonus adds noise not signal.
    # Top Review got +2.0 same as winner Stormylux — zero differentiation.
    # Old: 2.0/1.3/0.9/0.6/0.3  New: 1.0/0.7/0.5/0.35/0.2
    if impact >= 2.0:
        return 1.0  # Extreme bias favorite (was 2.0)
    elif impact >= 1.5:
        return 0.7  # Strong bias favorite (was 1.3)
    elif impact >= 1.3:
        return 0.5  # Moderate-strong bias favorite (was 0.9)
    elif impact >= 1.15:
        return 0.35  # Moderate bias favorite (was 0.6)
    elif impact >= 1.05:
        return 0.2  # Slight bias favorite (was 0.3)
    elif impact >= 1.00:
        return 0.1  # At parity

    return 0.0




def calculate_post_position_bias_bonus(
    post: str, weekly_post_impacts: dict[str, float]
) -> float:
    """
    ENHANCEMENT 5: Amplify post position advantage/disadvantage from weekly data.

    Oaklawn R1 Audit: Rail posts 1-3 had 2.59 weekly impact. Sombra Dorada (post 1)
    finished 2nd but model had her 5th. Post position in bias context was underweighted.

    Returns: Bonus/penalty to add to tier2_bonus.
    """
    if not weekly_post_impacts or not post:
        return 0.0

    try:
        post_str = str(int(float(post)))
    except (ValueError, TypeError):
        return 0.0

    if post_str not in weekly_post_impacts:
        return 0.0

    impact = weekly_post_impacts[post_str]

    # Strong rail/post advantage
    if impact >= 2.5:
        return 1.5  # Extreme post bias (Oaklawn rail at 2.59)
    elif impact >= 2.0:
        return 1.0  # Strong post bias
    elif impact >= 1.5:
        return 0.5  # Moderate post bias
    elif impact >= 1.2:
        return 0.3  # Mild advantage (R9 FIX: fills 1.20-1.50 dead zone)
    elif impact >= 1.0:
        return 0.1  # Slight advantage

    # Post DISADVANTAGE — GRADUATED (R9 FIX: fills 0.60-1.00 dead zone)
    # Post 10 at 0.70 impact was getting 0.0 — now gets -0.3
    if impact <= 0.40:
        return -1.0  # Severe post disadvantage
    elif impact <= 0.60:
        return -0.5  # Strong post disadvantage
    elif impact <= 0.80:
        return -0.3  # Mild post disadvantage (Sparkly post 10 at 0.70)
    elif impact < 1.0:
        return -0.1  # Slight below average

    return 0.0




def calculate_pace_supremacy_bonus(
    horse_name: str,
    horse_block: str,
    field_e1_values: dict[str, float],
    impact_values: dict[str, float],
) -> float:
    """
    ENHANCEMENT 6: E1/E2 Pace Supremacy Bonus

    Oaklawn R9 Audit: She's Storming had the BEST E1 (91) in the field, matching
    the race E1 par (91), at a speed-biased track. She was 24/1 and won.
    The model gave her almost no credit for being the fastest horse early in the field.

    When a horse has the fastest (or top-2) E1 in the field AND the track bias
    favors speed, this is a massive tactical advantage — they can control the pace.

    Returns: Bonus to add to tier2_bonus.
    """
    if not field_e1_values or horse_name not in field_e1_values:
        return 0.0

    horse_e1 = field_e1_values.get(horse_name, 0.0)
    if horse_e1 <= 0:
        return 0.0

    # Rank this horse's E1 among all field E1s
    all_e1 = sorted(field_e1_values.values(), reverse=True)
    try:
        e1_rank = all_e1.index(horse_e1) + 1
    except ValueError:
        return 0.0

    # Check if track currently favors early speed (E or E/P impact >= 1.0)
    speed_favored = False
    if impact_values:
        e_impact = impact_values.get("E", 1.0)
        ep_impact = impact_values.get("E/P", 1.0)
        s_impact = impact_values.get("S", 1.0)
        # Speed is favored when E or E/P impact > 1.0, or S is suppressed
        if max(e_impact, ep_impact) >= 1.10 or s_impact < 0.50:
            speed_favored = True

    bonus = 0.0

    # Top E1 in the field
    if e1_rank == 1:
        bonus = 0.8 if speed_favored else 0.4
    elif e1_rank == 2:
        bonus = 0.5 if speed_favored else 0.2
    elif e1_rank == 3:
        bonus = 0.3 if speed_favored else 0.1

    # Also parse the horse's E2 from their block to check if they sustain speed
    try:
        pace_data = parse_e1_e2_lp_values(horse_block)
        if pace_data["e2"] and len(pace_data["e2"]) >= 1:
            best_e2 = max(pace_data["e2"][:3])
            # If horse has both strong E1 AND E2 (>= 80), extra bonus for sustained speed
            if horse_e1 >= 85 and best_e2 >= 80 and e1_rank <= 2:
                bonus += 0.4  # Sustained speed premium
    except BaseException:
        pass

    return round(bonus, 2)




def calculate_first_after_claim_bonus(horse_block: str) -> float:
    """
    ENHANCEMENT 3: Detect 1st-after-claim trainer angle for rating boost.

    Oaklawn R1 Audit: Trainer Ashford had 28% win rate on 1st-after-claim
    but model gave zero credit for this angle.

    Parses patterns like:
      - "1st after claim 18 5 28% 50%"   (starts wins win% itm%)
      - "Claimed from ... for $XX,XXX"
      - "ClmPrice" field in PP data

    Returns: Bonus to add to tier2_bonus.
    """
    if not horse_block:
        return 0.0

    bonus = 0.0

    try:
        # Pattern 1: Explicit "1st after claim" stat line
        claim_match = re.search(
            r"1st\s+after\s+cl(?:ai)?m\s+(\d+)\s+(\d+)\s+(\d+)%",
            horse_block,
            re.IGNORECASE,
        )
        if claim_match:
            claim_starts = int(claim_match.group(1))
            _claim_wins = int(
                claim_match.group(2)
            )  # Parsed for completeness; win_pct used instead
            claim_win_pct = int(claim_match.group(3)) / 100.0

            if claim_starts >= 3:  # Meaningful sample
                if claim_win_pct >= 0.30:
                    bonus += 1.2  # Elite claim angle (Ashford 28%+)
                elif claim_win_pct >= 0.20:
                    bonus += 0.8  # Strong claim angle
                elif claim_win_pct >= 0.15:
                    bonus += 0.5  # Moderate claim angle
            return bonus

        # Pattern 2: "Claimed" in recent race + trainer with good record
        # This is a lighter signal — horse was recently claimed, trainer's overall % applies
        claimed_match = re.search(
            r"Claimed\s+from.*?for\s+\$[\d,]+", horse_block, re.IGNORECASE
        )
        if claimed_match:
            # Check if trainer has good overall win%
            trainer_pct_match = re.search(
                r"Trnr:.*?\(\d+\s+\d+-\d+-\d+\s+(\d+)%\)", horse_block
            )
            if trainer_pct_match:
                trnr_pct = int(trainer_pct_match.group(1)) / 100.0
                if trnr_pct >= 0.25:
                    bonus += 0.6  # Good trainer with new claim
                elif trnr_pct >= 0.18:
                    bonus += 0.3  # Decent trainer with new claim
    except BaseException:
        pass

    return bonus




def calculate_spi_bonus(spi: int | None) -> float:
    """Calculate bonus/penalty based on SPI (Sire Performance Index)"""
    if spi is None:
        return 0.0
    if spi >= 120:
        return 0.15
    elif spi >= 110:
        return 0.10
    elif spi >= 100:
        return 0.05
    elif spi >= 90:
        return 0.0
    else:
        return -0.05




def calculate_surface_specialty_bonus(
    surface_pct: float | None, surface_type: str
) -> float:
    """Calculate bonus for surface specialty (Turf or AW)"""
    if surface_pct is None:
        return 0.0

    if surface_type.lower() == "tur":
        if surface_pct >= 15:
            return 0.20
        elif surface_pct >= 12:
            return 0.15
        elif surface_pct >= 10:
            return 0.10
        elif surface_pct < 5:
            return -0.10
    elif surface_type.lower() in ["aw", "all-weather", "synthetic"]:
        if surface_pct >= 12:
            return 0.15
        elif surface_pct >= 10:
            return 0.10
        elif surface_pct < 5:
            return -0.10

    return 0.0




def calculate_awd_mismatch_penalty(awd_status: str | None) -> float:
    """Calculate penalty for distance mismatch"""
    if awd_status == "mismatch":
        return -0.10
    return 0.0




def _parse_post_number(post_str: str) -> int | None:
    """ELITE: Fast post number extraction without regex (3x faster)"""
    try:
        # Method 1: Try direct int conversion (fastest)
        return int(str(post_str).strip())
    except ValueError:
        try:
            # Method 2: Extract digits with string comprehension (2x faster than regex)
            digits = "".join(c for c in str(post_str) if c.isdigit())
            return int(digits) if digits else None
        except (ValueError, TypeError):
            # Log parsing failures to session state (not UI - avoids warning spam)
            if "post_parse_failures" not in st.session_state:
                st.session_state.post_parse_failures = []
            st.session_state.post_parse_failures.append(str(post_str))
            return None




def calculate_weather_impact(
    weather_data: dict[str, Any], style: str, distance_txt: str
) -> float:
    """
    ELITE: Weather impact calculation based on racing research.

    Args:
        weather_data: Dict with keys 'condition', 'wind_mph', 'temp_f'
        style: Running style ('E', 'E/P', 'P', 'S')
        distance_txt: Race distance (e.g., '6F', '1 1/16M')

    Returns:
        float: Adjustment to rating (-0.3 to +0.3)
    """
    if not weather_data:
        return 0.0

    bonus = 0.0
    track_condition = str(weather_data.get("condition", "Fast")).lower()
    wind_mph = float(weather_data.get("wind_mph", 0))
    temp_f = float(weather_data.get("temp_f", 70))

    # 1. Track Condition Impact
    if "mud" in track_condition or "slop" in track_condition:
        # Mud favors early speed that can secure position
        if style in ["E", "E/P"]:
            bonus += 0.20  # Early speed advantage in mud
        elif style == "S":
            bonus -= 0.15  # Closers struggle in mud

    elif "wet" in track_condition or "yield" in track_condition:
        # Wet track: Moderate advantage to stalkers
        if style == "E/P":
            bonus += 0.10

    # 2. Wind Impact (>12 mph affects race significantly)
    if wind_mph >= 15:
        # Strong headwinds favor closers (pace collapses)
        if style == "S":
            bonus += 0.12
        elif style == "E":
            bonus -= 0.08  # Early speed tires faster

    # 3. Temperature Impact on Stamina (distance races)
    is_route = (
        any(d in distance_txt.upper() for d in ["M", "MILE"]) if distance_txt else False
    )
    if is_route:
        if temp_f >= 85:
            # Hot weather: Stamina drop for routes
            bonus -= 0.08
        elif temp_f <= 35:
            # Cold weather: Slight advantage to closers (slower pace)
            if style == "S":
                bonus += 0.05

    return float(np.clip(bonus, -0.30, 0.30))




def calculate_jockey_trainer_impact(horse_name: str, pp_text: str) -> float:
    """
    ELITE: Calculate impact of jockey/trainer performance based on BRISNET PP stats.

    ACTUAL BRISNET Format:
    - Trainer: "Trnr: LastName FirstName (starts wins-places-shows win%)"
      Example: "Trnr: Eikleberry Kevin (50 11-7-4 22%)"
    - Jockey: "Jky: LastName FirstName (starts wins-places-shows win%)"
    - Combo: "w/TrnrLastName: starts win% place% combo%"

    PHASE 1 RESTORATION (Feb 13, 2026): Integrated calculate_hot_combo_bonus for tiered combo analysis.
    """
    if not pp_text or not horse_name:
        return 0.0

    bonus = 0.0
    jockey_win_rate = 0.0
    trainer_win_rate = 0.0
    combo_win_rate = 0.0

    # Find horse section
    horse_section_start = pp_text.find(horse_name)
    if horse_section_start == -1:
        return 0.0

    # Search next 800 chars for trainer/jockey stats
    section = pp_text[horse_section_start : horse_section_start + 800]

    # TRAINER: "Trnr: LastName FirstName (starts wins-places-shows win%)"
    trainer_pattern = r"Trnr:.*?\((\d+)\s+(\d+)-(\d+)-(\d+)\s+(\d+)%\)"
    trainer_match = re.search(trainer_pattern, section)
    if trainer_match:
        t_starts = int(trainer_match.group(1))
        _t_wins = int(
            trainer_match.group(2)
        )  # Parsed for completeness; win_pct used instead
        t_win_pct_reported = int(trainer_match.group(5)) / 100.0

        if t_starts >= 20:
            trainer_win_rate = t_win_pct_reported

            # Elite trainer (>28% win rate) = +0.12 bonus
            if trainer_win_rate >= 0.28:
                bonus += 0.12
            elif trainer_win_rate >= 0.22:
                bonus += 0.08
            elif trainer_win_rate >= 0.18:
                bonus += 0.05

    # JOCKEY & COMBO: Parse jockey stats and combo percentage
    jockey_win_rate, combo_win_rate = parse_jockey_combo_stats(section)

    # Add individual jockey bonus
    if jockey_win_rate >= 0.25:
        bonus += 0.10
    elif jockey_win_rate >= 0.18:
        bonus += 0.06
    elif jockey_win_rate >= 0.12:
        bonus += 0.03

    # ELITE CONNECTIONS COMBO BONUS - Use restored calculate_hot_combo_bonus function
    # This provides tiered analysis: 40%+ L60 combo was KEY to Litigation 24/1 win!
    combo_bonus = calculate_hot_combo_bonus(
        trainer_win_rate, jockey_win_rate, combo_win_rate
    )
    bonus += combo_bonus

    return float(np.clip(bonus, 0, 0.50))




def calculate_track_condition_granular(
    track_info: dict[str, Any], style: str, post: int | str
) -> float:
    """
    ELITE: Track condition analysis beyond basic Fast/Muddy.

    Args:
        track_info: Dict with 'condition', 'rail_position', 'moisture_level'
        style: Running style
        post: Post position (int or str — safely cast internally)

    Returns:
        Adjustment to rating
    """
    if not track_info:
        return 0.0

    # Safely cast post to int (callers may pass str from DataFrame rows)
    try:
        post_num = int(post)
    except (ValueError, TypeError):
        post_num = 0

    bonus = 0.0

    # 1. Rail Position Impact
    rail_position = str(track_info.get("rail_position", "normal")).lower()
    if "10ft" in rail_position or "15ft" in rail_position or "out" in rail_position:
        # Rail is out - outside posts gain advantage
        if post_num >= 6:
            bonus += 0.25
        elif post_num <= 2:
            bonus -= 0.15  # Inside posts penalized

    # 2. Track Seal/Harrowing
    condition_detail = str(track_info.get("condition", "")).lower()
    if "sealed" in condition_detail or "harrowed" in condition_detail:
        # Sealed track favors early speed
        if style in ["E", "E/P"]:
            bonus += 0.18

    elif "cuppy" in condition_detail or "tiring" in condition_detail:
        # Cuppy/tiring track favors closers
        if style == "S":
            bonus += 0.15
        elif style == "E":
            bonus -= 0.10

    # 3. Moisture Content (even on "Fast" tracks)
    moisture = str(track_info.get("moisture_level", "normal")).lower()
    if "tacky" in moisture or "holding" in moisture:
        # Tacky track: Grip advantage to stalkers
        if style == "E/P":
            bonus += 0.08

    return float(np.clip(bonus, -0.25, 0.25))




def is_elite_trainer(trainer_name):
    """
    Check if trainer is elite (top 5% debut ROI) for FTS handling.

    Elite trainers have significantly higher win rates with first-time starters
    and their FTS horses should receive confidence boosts.

    Args:
        trainer_name: String name of the trainer

    Returns:
        bool: True if trainer is in ELITE_TRAINERS set, False otherwise
    """
    if not trainer_name:
        return False
    return trainer_name in ELITE_TRAINERS  # O(1) lookup via set




def calculate_layoff_bonus(days_off: int, is_marathon: bool = False) -> float:
    """
    CALIBRATED: Layoff evaluation adjusted for marathon distances.

    Key Learning: 30-60 day layoffs can HELP at marathon distances (freshening effect).
    Layabout (WINNER) and Padiddle (2nd) both had ~45-50 day layoffs.
    """
    if days_off is None:
        return 0.0

    if is_marathon:
        # Marathon distances favor fresher horses
        if 30 <= days_off <= 60:
            return +0.10  # Freshening bonus (optimal range)
        elif 20 <= days_off < 30:
            return +0.05  # Slight freshen
        elif 60 < days_off <= 90:
            return 0.0  # Neutral
        elif days_off > 90:
            return -0.15  # Too long away
        else:
            return 0.0  # Recent race (no bonus)
    else:
        # Standard distances prefer recent racing
        if days_off <= 14:
            return +0.05  # Sharp form
        elif days_off <= 30:
            return 0.0  # Acceptable
        elif 30 < days_off <= 60:
            return -0.05  # Slight concern
        elif days_off > 60:
            return -0.10  # Extended layoff

    return 0.0




def calculate_experience_bonus(career_starts: int, is_marathon: bool = False) -> float:
    """
    CALIBRATED: Lightly-raced improver bonus.

    Key Learning: Layabout (WINNER) had only 9 career starts.
    Fresh legs advantage at marathon distances!
    """
    if career_starts is None or career_starts <= 0:
        return 0.0

    if is_marathon:
        # Marathons favor fresh legs
        if career_starts <= 10:
            return +0.10  # Very lightly raced (fresh!)
        elif career_starts <= 15:
            return +0.05  # Lightly raced
        elif career_starts <= 25:
            return 0.0  # Normal experience
        else:
            return -0.03  # Lots of wear and tear
    else:
        # Standard distances
        if career_starts <= 10:
            return +0.05  # Improving
        elif career_starts <= 20:
            return 0.0  # Normal
        else:
            return 0.0  # Experienced




def calculate_hot_trainer_bonus(
    trainer_win_pct: float,
    is_hot_l14: bool = False,
    is_2nd_lasix_high_pct: bool = False,
    trainer_starts: int | None = None,
) -> float:
    """
    HOT TRAINER BONUS (TUP R6 + R7 Feb 2026 Calibration)

    TUP R6 Allowance Winner #5 Cactus League:
    - Trainer: 22% win rate (hot trainer)
    - 4-0-0 in last 14 days (HOT!)
    - 2nd time Lasix: 33% trainer angle (HUGE!)

    TUP R6 Failed pick #2 Ez Cowboy:
    - Trainer: Only 10% win rate (below average)

    TUP R7 Claiming Failed pick #3 Forest Acclamation:
    - Trainer: 0% win rate (Feron 13 0-1-2) = DEATH SENTENCE
    - 3 wins in 50 career starts (6% career win rate)
    - Finished 4th despite being model's top pick

    TUP R4 TUNING (Feb 18, 2026):
    - Hot Jammies: Trainer LaVanway 2% (1/51) — ranked #1, failed to hit board
    - Guide My Steps: Trainer Crowe 0% (0/31) — WON at 6/5 (market trusted horse not trainer)
    - Sample size now amplifies penalty: 0% on 30+ starts >> 0% on 5 starts

    Returns: Bonus/penalty to add to rating_final
    """
    bonus = 0.0
    # Default to 0 if None (caller failed to provide trainer starts)
    _trainer_starts = trainer_starts if trainer_starts is not None else 0
    if trainer_starts is None:
        logger.debug(
            "calculate_hot_trainer_bonus: trainer_starts=None, defaulting to 0"
        )

    # ═══════════════════════════════════════════════════════════════
    # SAMPLE-SIZE AWARE TRAINER PENALTY (Feb 18, 2026 TUP R4 Tuning)
    # 0% on 5 starts could be bad luck. 0% on 30+ starts is a pattern.
    # 2% on 51 starts (LaVanway) is effectively a confirmed loser.
    # ═══════════════════════════════════════════════════════════════
    if trainer_win_pct == 0.0:
        if _trainer_starts >= 30:
            return -1.8  # Confirmed 0% with huge sample — near-fatal
        elif _trainer_starts >= 15:
            return -1.5  # Solid sample 0% — very bad
        else:
            return -1.2  # Small sample 0% — bad but recoverable

    # Very low % trainer penalty (1-5%) — AMPLIFIED with sample size
    if trainer_win_pct > 0.0 and trainer_win_pct < 0.05:
        bonus -= 0.9  # Was -0.7: 2% trainer penalty increased
        if _trainer_starts >= 30:
            bonus -= 0.3  # Extra penalty for confirmed low % with large sample
    elif trainer_win_pct >= 0.05 and trainer_win_pct < 0.10:
        bonus -= 0.5  # Was -0.4: Moderate penalty slightly increased

    # High % trainer baseline
    if trainer_win_pct >= 0.25:  # Elite trainer (25%+)
        bonus += 0.4  # was 0.3
    elif trainer_win_pct >= 0.20:  # Hot trainer (20-24%)
        bonus += 0.5
    elif trainer_win_pct >= 0.15:  # Above average (15-19%)
        bonus += 0.25  # was 0.2

    # Hot trainer in last 14 days (4+ wins)
    if is_hot_l14:
        bonus += 0.3

    # High % trainer angle (2nd time Lasix 33%+)
    if is_2nd_lasix_high_pct:
        bonus += 0.8  # HUGE angle

    return bonus




def calculate_sprint_post_position_bonus(
    post: int, distance: float, surface: str
) -> float:
    """Inside posts dominate turf sprints (Rail 1.65 impact, Outside 8+ 0.44 death!)"""
    if distance > 6.5:
        return 0.0
    bonus = 0.0
    if surface.lower() in ["turf", "tur", "t"] and distance <= 6.0:
        if post == 1:
            bonus += 0.15
        elif post <= 3:
            bonus += 0.12
        elif post <= 7:
            bonus += 0.05
        else:
            bonus -= 0.25  # Death zone!
    return bonus




def calculate_sprint_running_style_bonus(style: str, distance: float) -> float:
    """Early speed 2.05 impact at sprints, pressers 0.00, stalkers 0.20 (death!)"""
    if distance > 6.5:
        return 0.0
    bonus = 0.0
    style_upper = style.upper() if style else ""
    if distance <= 5.5:
        if "E" in style_upper and "/" not in style_upper:
            bonus += 0.20
        elif "E" in style_upper and "P" in style_upper:
            bonus += 0.10
        elif "P" in style_upper:
            bonus -= 0.10
        elif "S" in style_upper:
            bonus -= 0.20
    return bonus




def calculate_hot_combo_bonus(
    trainer_pct: float, jockey_pct: float, combo_pct: float
) -> float:
    """Hot trainer/jockey combos (40% L60 was KEY to Litigation win!)"""
    bonus = 0.0
    if combo_pct >= 0.40:
        bonus += 0.20
    elif combo_pct >= 0.30:
        bonus += 0.15
    elif combo_pct >= 0.20:
        bonus += 0.10
    if trainer_pct >= 0.30:
        bonus += 0.10
    elif trainer_pct >= 0.20:
        bonus += 0.05
    if jockey_pct >= 0.25:
        bonus += 0.08
    elif jockey_pct >= 0.15:
        bonus += 0.05
    return float(np.clip(bonus, 0.0, 0.25))




def analyze_class_movement(
    past_races: list[dict], today_class: str, _today_purse: int = 0
) -> dict[str, Any]:
    """
    COMPREHENSIVE: Analyze if horse is stepping up or down in class.

    Returns:
    - class_change: 'up', 'down', 'same', 'unknown'
    - class_delta: numeric change estimate
    - pattern: 'rising', 'dropping', 'stable'
    - bonus: rating adjustment
    """
    if not past_races or len(past_races) < 2:
        return {
            "class_change": "unknown",
            "class_delta": 0,
            "pattern": "unknown",
            "bonus": 0.0,
        }

    # Class hierarchy (higher = better class)
    class_hierarchy = {
        "Msw": 1,  # Maiden special weight
        "Mcl": 2,  # Maiden claiming
        "Clm": 3,  # Claiming
        "Str": 4,  # Starter allowance
        "Aoc": 5,  # Allowance optional claiming
        "Alw": 6,  # Allowance
        "Stk": 8,  # Stakes
        "G3": 9,  # Grade 3
        "G2": 10,  # Grade 2
        "G1": 11,  # Grade 1
    }

    # Get today's class level
    today_level = 0
    for key in class_hierarchy:
        if key.lower() in today_class.lower():
            today_level = class_hierarchy[key]
            break

    if today_level == 0:
        today_level = 5  # Default to allowance

    # Get recent class levels
    recent_levels = []
    for race in past_races[:5]:  # Last 5 races
        race_class = race.get("class", race.get("race_type", ""))
        for key in class_hierarchy:
            if key.lower() in race_class.lower():
                recent_levels.append(class_hierarchy[key])
                break

    if not recent_levels:
        return {
            "class_change": "unknown",
            "class_delta": 0,
            "pattern": "unknown",
            "bonus": 0.0,
        }

    avg_recent = sum(recent_levels) / len(recent_levels)
    class_delta = today_level - avg_recent

    # Determine change — GRADUATED SCALE (SA R8 audit Feb 20, 2026)
    # Old: flat -0.10 for ANY step up. MC32k→OC50k is a 3-level jump (3→5)
    # but only got -0.10 penalty. Now scales with delta magnitude.
    if class_delta > 1:
        class_change = "up"
        if class_delta >= 3:
            bonus = -0.18  # Major class jump (e.g., MC→ALW/AOC)
        elif class_delta >= 2:
            bonus = -0.14  # Significant step up
        else:
            bonus = -0.10  # Minor step up
    elif class_delta < -1:
        class_change = "down"
        if class_delta <= -3:
            bonus = +0.18  # Major class drop
        elif class_delta <= -2:
            bonus = +0.15  # Significant drop
        else:
            bonus = +0.12  # Minor drop
    else:
        class_change = "same"
        bonus = 0.0

    # Determine pattern (last 3 races)
    pattern = "stable"
    if len(recent_levels) >= 3:
        if recent_levels[0] > recent_levels[1] > recent_levels[2]:
            pattern = "rising"
            bonus += 0.05  # Positive progression
        elif recent_levels[0] < recent_levels[1] < recent_levels[2]:
            pattern = "dropping"
            bonus += 0.08  # Finding easier spots

    return {
        "class_change": class_change,
        "class_delta": class_delta,
        "pattern": pattern,
        "bonus": float(np.clip(bonus, -0.20, 0.20)),
    }




def apply_track_pattern_bonus(
    track_patterns: dict,
    post_position: int,
    running_style: str,
    best_beyer: int,
    class_rating: float,
    days_since_last: int,
    workout_pattern: str,
    _prime_power: float = 0.0,
    jockey: str = "",
    trainer: str = "",
) -> dict:
    """
    Apply learned track pattern data to compute a bonus/penalty for a horse.

    Uses historical winner profiles at this track/surface/distance to reward
    horses that match winning patterns and penalize mismatches.

    Returns dict with 'bonus' (float) and 'details' (list of str explanations).
    """
    bonus = 0.0
    details: list[str] = []
    min_sample = 3  # Minimum races before patterns are meaningful

    if not track_patterns:
        return {"bonus": 0.0, "details": ["No track patterns available"]}

    total_races = 0
    tr = track_patterns.get("total_races_analyzed", {})
    if isinstance(tr, dict):
        total_races = int(tr.get("value", 0))
    if total_races < min_sample:
        return {
            "bonus": 0.0,
            "details": [f"Only {total_races} races recorded (need {min_sample})"],
        }

    # ---- 1. Post Position Pattern ----
    post_data = track_patterns.get("post_win_rates", {})
    if post_data and post_data.get("sample_size", 0) >= min_sample:
        rates = post_data.get("value", {})
        if isinstance(rates, dict):
            pp_str = str(post_position)
            pp_rate = rates.get(pp_str, None)
            avg_rate = sum(rates.values()) / max(len(rates), 1) if rates else 0
            if pp_rate is not None:
                if pp_rate >= avg_rate * 1.5 and pp_rate >= 0.15:
                    b = min(0.15, pp_rate * 0.4)
                    bonus += b
                    details.append(
                        f"Post {post_position} wins {pp_rate * 100:.0f}% (+{b:.2f})"
                    )
                elif pp_rate <= avg_rate * 0.3:
                    p = -0.08
                    bonus += p
                    details.append(
                        f"Post {post_position} wins only {pp_rate * 100:.0f}% ({p:.2f})"
                    )

    # Best posts check
    best_posts_data = track_patterns.get("best_posts", {})
    if best_posts_data and best_posts_data.get("sample_size", 0) >= min_sample:
        best_posts = best_posts_data.get("value", [])
        if isinstance(best_posts, list) and str(post_position) in best_posts:
            bonus += 0.05
            details.append(f"Post {post_position} is a top winning post (+0.05)")

    # ---- 2. Running Style Pattern ----
    style_data = track_patterns.get("style_win_rates", {})
    dom_style_data = track_patterns.get("dominant_winning_style", {})
    if style_data and style_data.get("sample_size", 0) >= min_sample:
        rates = style_data.get("value", {})
        if isinstance(rates, dict) and running_style in rates:
            style_rate = rates[running_style]
            avg_style_rate = sum(rates.values()) / max(len(rates), 1) if rates else 0
            if style_rate >= avg_style_rate * 1.5 and style_rate >= 0.20:
                b = min(0.15, style_rate * 0.35)
                bonus += b
                details.append(
                    f"Style '{running_style}' wins {style_rate * 100:.0f}% (+{b:.2f})"
                )
            elif style_rate <= avg_style_rate * 0.3:
                p = -0.08
                bonus += p
                details.append(
                    f"Style '{running_style}' wins only {style_rate * 100:.0f}% ({p:.2f})"
                )

    if dom_style_data:
        dom = dom_style_data.get("value", "")
        if dom and running_style == dom:
            bonus += 0.05
            details.append(f"Matches dominant winning style '{dom}' (+0.05)")

    # ---- 3. Speed Figure Comparison ----
    avg_beyer_data = track_patterns.get("avg_winner_best_beyer", {})
    if avg_beyer_data and avg_beyer_data.get("sample_size", 0) >= min_sample:
        avg_wb = float(avg_beyer_data.get("value", 0))
        if avg_wb > 0 and best_beyer > 0:
            diff = best_beyer - avg_wb
            if diff >= 5:
                b = min(0.15, diff * 0.02)
                bonus += b
                details.append(
                    f"Best Beyer {best_beyer} vs avg winner {avg_wb:.0f} (+{b:.2f})"
                )
            elif diff <= -10:
                p = max(-0.12, diff * 0.01)
                bonus += p
                details.append(
                    f"Best Beyer {best_beyer} vs avg winner {avg_wb:.0f} ({p:.2f})"
                )

    # ---- 4. Class Rating Comparison ----
    avg_class_data = track_patterns.get("avg_winner_class_rating", {})
    if avg_class_data and avg_class_data.get("sample_size", 0) >= min_sample:
        avg_wc = float(avg_class_data.get("value", 0))
        if avg_wc > 0 and class_rating > 0:
            diff = class_rating - avg_wc
            if diff >= 0.5:
                b = min(0.10, diff * 0.08)
                bonus += b
                details.append(
                    f"Class {class_rating:.1f} vs avg winner {avg_wc:.1f} (+{b:.2f})"
                )

    # ---- 5. Freshness (Days Since Last) ----
    avg_dsl_data = track_patterns.get("avg_winner_days_since_last", {})
    if avg_dsl_data and avg_dsl_data.get("sample_size", 0) >= min_sample:
        avg_dsl = float(avg_dsl_data.get("value", 0))
        if avg_dsl > 0 and days_since_last > 0:
            # Penalize if horse is much more rested or tighter than winners
            diff = abs(days_since_last - avg_dsl)
            if diff <= 7:
                bonus += 0.05
                details.append("Freshness matches winners (within 7 days, +0.05)")
            elif diff >= 60:
                bonus -= 0.05
                details.append(
                    f"Freshness differs by {diff:.0f} days from winner avg (-0.05)"
                )

    # ---- 6. Workout Pattern ----
    workout_data = track_patterns.get("winner_workout_patterns", {})
    if workout_data and workout_data.get("sample_size", 0) >= min_sample:
        patterns = workout_data.get("value", {})
        if isinstance(patterns, dict):
            total_w = sum(patterns.values())
            if total_w > 0 and workout_pattern in patterns:
                wp_rate = patterns[workout_pattern] / total_w
                if wp_rate >= 0.5:
                    bonus += 0.05
                    details.append(
                        f"Workout '{workout_pattern}' matches {wp_rate * 100:.0f}% of winners (+0.05)"
                    )

    # ---- 7. Top Jockey/Trainer ----
    jockey_data = track_patterns.get("top_jockeys", {})
    if jockey_data and jockey and jockey_data.get("sample_size", 0) >= min_sample:
        top_j = jockey_data.get("value", {})
        if isinstance(top_j, dict) and jockey in top_j:
            bonus += 0.05
            details.append(f"Jockey '{jockey}' is a top winner here (+0.05)")

    trainer_data = track_patterns.get("top_trainers", {})
    if trainer_data and trainer and trainer_data.get("sample_size", 0) >= min_sample:
        top_t = trainer_data.get("value", {})
        if isinstance(top_t, dict) and trainer in top_t:
            bonus += 0.05
            details.append(f"Trainer '{trainer}' is a top winner here (+0.05)")

    # Cap total bonus in [-0.30, +0.50]
    bonus = max(-0.30, min(0.50, bonus))

    return {"bonus": round(bonus, 3), "details": details}




def detect_surface_switch(
    race_history: list[dict],
    today_surface: str,
    pedigree_data: dict | None = None,
) -> dict[str, Any]:
    """
    Detect surface switches (Dirt→Turf, Turf→Dirt, etc.) and return a penalty/bonus.

    ENHANCEMENT (Feb 13, 2026 — TAM R7 Post-Race Fix):
    Now accepts optional pedigree_data dict with keys:
      - sire_mud_pct: From "Sire Stats: XX%Mud" line
      - pedigree_off: Off-track breeding rating (0-100)
      - pedigree_turf: Turf breeding rating (0-100)
    First-time surface + strong pedigree = reduced penalty / bonus.

    Logic:
    - Dirt→Turf: Significant penalty (-0.15) unless horse has turf history / pedigree
    - Turf→Dirt: Moderate penalty (-0.10) unless horse has dirt history / mud pedigree
    - Same surface: Small bonus if consistent (+0.05)
    - Mixed history: Penalize if mostly other surface (-0.08)

    Returns dict with: switch_type, today_surface, last_surface, turf_exp, dirt_exp, bonus
    """
    result = {
        "switch_type": "unknown",
        "today_surface": today_surface,
        "last_surface": "unknown",
        "turf_experience": 0,
        "dirt_experience": 0,
        "bonus": 0.0,
    }

    if not race_history or not today_surface:
        return result

    ped = pedigree_data or {}
    today_surf = today_surface.strip().lower()
    today_is_turf = "turf" in today_surf
    today_is_dirt = "dirt" in today_surf or "fast" in today_surf

    # Count surface experience
    turf_count = sum(1 for r in race_history if r.get("surface") == "Turf")
    dirt_count = sum(1 for r in race_history if r.get("surface") == "Dirt")
    total = len(race_history)

    result["turf_experience"] = turf_count
    result["dirt_experience"] = dirt_count

    # Last race surface
    last_surface = (
        race_history[0].get("surface", "unknown") if race_history else "unknown"
    )
    result["last_surface"] = last_surface

    # Determine switch type
    if today_is_turf and last_surface == "Dirt":
        result["switch_type"] = "dirt_to_turf"
        if turf_count == 0:
            result["bonus"] = -0.20  # Never on turf = MAJOR penalty
            # Pedigree turf rating offset (Feb 13 enhancement)
            ped_turf = ped.get("pedigree_turf")
            if ped_turf is not None and ped_turf >= 80:
                result["bonus"] += 0.15  # Strong turf pedigree offsets penalty
            elif ped_turf is not None and ped_turf >= 65:
                result["bonus"] += 0.08
        elif turf_count <= 1:
            result["bonus"] = -0.12  # Minimal turf experience
        else:
            result["bonus"] = -0.05  # Has some turf exp, small penalty

    elif today_is_dirt and last_surface == "Turf":
        result["switch_type"] = "turf_to_dirt"
        if dirt_count == 0:
            # First time on dirt — check mud pedigree for bonus (Feb 13 enhancement)
            mud_pct = ped.get("sire_mud_pct")
            ped_off = ped.get("pedigree_off")

            if mud_pct is not None and mud_pct >= 25:
                # Strong mud sire — first time on dirt is a positive angle
                result["bonus"] = 0.20
            elif mud_pct is not None and mud_pct >= 15:
                result["bonus"] = 0.10  # Average mud pedigree
            elif ped_off is not None and ped_off >= 75:
                result["bonus"] = 0.15  # Strong off-track breeding
            elif ped_off is not None and ped_off >= 60:
                result["bonus"] = 0.05  # Decent off-track breeding
            else:
                result["bonus"] = 0.15  # Turf-to-dirt first-timer angle (baseline)
        elif dirt_count >= 1:
            result["bonus"] = 0.05  # Has dirt experience, smaller bonus

    elif today_is_turf and last_surface == "Turf":
        result["switch_type"] = "same_turf"
        if turf_count >= 3:
            result["bonus"] = 0.08  # Turf specialist reward
        elif turf_count >= 1:
            result["bonus"] = 0.03

    elif today_is_dirt and last_surface == "Dirt":
        result["switch_type"] = "same_dirt"
        if dirt_count >= 3:
            result["bonus"] = 0.05  # Dirt specialist reward
        else:
            result["bonus"] = 0.02

    # Additional penalty: mostly raced on opposite surface
    if today_is_turf and total > 0 and (dirt_count / total) >= 0.80:
        result["bonus"] -= 0.08  # 80%+ dirt horse trying turf

    result["bonus"] = float(np.clip(result["bonus"], -0.25, 0.20))
    return result




def score_workout_quality(block: str) -> dict[str, Any]:
    """
    Score workout quality from a horse's PP block text.
    Parses individual workouts and calculates quality metrics.

    Returns: avg_rank_pct, worst_rank_pct, bullet_count, num_works, quality_bonus
    """
    result = {
        "avg_rank_pct": 0.5,
        "worst_rank_pct": 1.0,
        "bullet_count": 0,
        "num_works": 0,
        "quality_bonus": 0.0,
    }

    if not block:
        return result

    # Parse workouts using pattern: dateSTR track Xf :time grade rank/total
    workout_pattern = re.compile(
        r"([×]?)(\d{2}[A-Za-z]{3}(?:\d{0,2}|'?\d{2}))\s+"
        r"(\w{2,4})\s+"
        r"(\d+)f\s+"
        r":?[\d:.¹²³⁴⁵⁶⁷⁸⁹⁰ƒ®«ª³©¨°¬²‚±]+\s+"
        r"([HBG]g?)\s+"
        r"(\d+)/(\d+)"
    )

    workouts = []
    for line in block.split("\n"):
        for match in workout_pattern.finditer(line):
            try:
                bullet = match.group(1) == "×"
                rank = int(match.group(6))
                total = int(match.group(7))
                if total > 0:
                    workouts.append(
                        {
                            "bullet": bullet,
                            "rank": rank,
                            "total": total,
                            "rank_pct": rank / total,
                        }
                    )
            except Exception:
                continue

    if not workouts:
        return result

    rank_pcts = [w["rank_pct"] for w in workouts]
    result["num_works"] = len(workouts)
    result["avg_rank_pct"] = sum(rank_pcts) / len(rank_pcts)
    result["worst_rank_pct"] = max(rank_pcts)
    result["bullet_count"] = sum(1 for w in workouts if w["bullet"])

    # Score workout quality
    bonus = 0.0
    avg_pct = result["avg_rank_pct"]

    if avg_pct <= 0.15:
        bonus = 0.12  # Elite worker (top 15% avg)
    elif avg_pct <= 0.25:
        bonus = 0.08  # Strong worker
    elif avg_pct <= 0.40:
        bonus = 0.03  # Adequate
    elif avg_pct <= 0.60:
        bonus = -0.03  # Below average
    elif avg_pct <= 0.80:
        bonus = -0.08  # Poor
    else:
        bonus = -0.15  # Terrible (like JWB's 43/43 = dead last)

    # Bullet bonus
    if result["bullet_count"] >= 2:
        bonus += 0.04
    elif result["bullet_count"] >= 1:
        bonus += 0.02

    # Penalty for consistently worst in field
    if len(workouts) >= 3 and all(w["rank_pct"] >= 0.80 for w in workouts[:3]):
        bonus -= 0.10  # All recent works dead last = major red flag

    result["quality_bonus"] = float(np.clip(bonus, -0.25, 0.15))
    return result




def analyze_distance_from_history(
    race_history: list[dict], today_distance: str
) -> dict[str, Any]:
    """
    Wrapper that converts race_history (from parse_race_history_from_block)
    into the format expected by analyze_distance_pattern.

    This bridges the ACTUAL per-race distance data to the existing
    analyze_distance_pattern() framework that was previously starved of data.
    """
    if not race_history:
        return analyze_distance_pattern([], today_distance)

    # Convert to format analyze_distance_pattern expects
    past_races = []
    for r in race_history:
        past_races.append(
            {
                "distance": r.get("distance", ""),
                "speed_fig": r.get("speed_fig", 0),
                "race_type": r.get("race_type", ""),
                "finish_pos": r.get("finish_pos", 0),
            }
        )

    return analyze_distance_pattern(past_races, today_distance)




def analyze_distance_pattern(
    past_races: list[dict], today_distance: str
) -> dict[str, Any]:
    """
    COMPREHENSIVE: Analyze distance changes and patterns.

    Returns:
    - distance_change: 'stretch_out', 'cut_back', 'same', 'unknown'
    - distance_delta: Numeric change in furlongs
    - experience_at_distance: Number of times raced at this distance
    - best_fig_at_distance: Best speed fig at today's distance
    - bonus: Rating adjustment
    """
    if not past_races or not today_distance:
        return {"distance_change": "unknown", "distance_delta": 0, "bonus": 0.0}

    # Use module-level distance_to_furlongs (defined once, near line 625)
    today_furlongs = distance_to_furlongs(today_distance)

    # Analyze past distances
    past_distances = []
    experience_count = 0
    figs_at_distance = []

    for race in past_races[:8]:  # Last 8 races
        race_dist = race.get("distance", "")
        race_furlongs = distance_to_furlongs(race_dist)
        past_distances.append(race_furlongs)

        # Count experience at today's distance (within 0.5F)
        if abs(race_furlongs - today_furlongs) <= 0.5:
            experience_count += 1
            if race.get("speed_fig", 0) > 0:
                figs_at_distance.append(race["speed_fig"])

    if not past_distances:
        return {"distance_change": "unknown", "distance_delta": 0, "bonus": 0.0}

    avg_past_dist = sum(past_distances) / len(past_distances)
    distance_delta = today_furlongs - avg_past_dist

    # Determine change type
    bonus = 0.0
    if distance_delta > 1.5:
        distance_change = "stretch_out"
        # Stretching out 2+ furlongs
        if experience_count >= 2:
            bonus = +0.05  # Has experience at distance
        else:
            bonus = -0.08  # Unproven at distance

    elif distance_delta < -1.5:
        distance_change = "cut_back"
        # Cutting back in distance usually helps speed horses
        bonus = +0.08

    else:
        distance_change = "same"
        if experience_count >= 3:
            bonus = +0.05  # Comfortable at distance

    best_fig = max(figs_at_distance) if figs_at_distance else 0

    return {
        "distance_change": distance_change,
        "distance_delta": distance_delta,
        "today_furlongs": today_furlongs,
        "experience_count": experience_count,
        "best_fig_at_distance": best_fig,
        "bonus": float(np.clip(bonus, -0.10, 0.10)),
    }




def analyze_form_cycle(past_races: list[dict]) -> dict[str, Any]:
    """
    COMPREHENSIVE: Analyze form cycle (improving/declining).

    Returns:
    - cycle: 'improving', 'declining', 'peaking', 'bottoming', 'stable'
    - trend_score: -1.0 to +1.0 (negative = declining, positive = improving)
    - last_3_finishes: Recent finish positions
    - last_3_figs: Recent speed figures
    - bonus: Rating adjustment
    """
    if not past_races or len(past_races) < 3:
        return {"cycle": "unknown", "trend_score": 0.0, "bonus": 0.0}

    # Get last 3-5 races
    recent = past_races[:5]
    finishes = [r.get("finish_pos", 0) for r in recent if r.get("finish_pos", 0) > 0]
    figs = [r.get("speed_fig", 0) for r in recent if r.get("speed_fig", 0) > 0]

    if len(finishes) < 3 or len(figs) < 3:
        return {"cycle": "unknown", "trend_score": 0.0, "bonus": 0.0}

    # Calculate trends
    # Finish trend (lower is better)
    finish_trend = 0.0
    if finishes[0] < finishes[1] < finishes[2]:
        finish_trend = +1.0  # Improving finishes
    elif finishes[0] > finishes[1] > finishes[2]:
        finish_trend = -1.0  # Declining finishes
    elif finishes[0] < finishes[-1]:
        finish_trend = +0.5  # Generally improving
    elif finishes[0] > finishes[-1]:
        finish_trend = -0.5  # Generally declining

    # Figure trend (higher is better)
    fig_trend = 0.0
    if figs[0] > figs[1] > figs[2]:
        fig_trend = +1.0  # Improving figures
    elif figs[0] < figs[1] < figs[2]:
        fig_trend = -1.0  # Declining figures
    elif figs[0] > figs[-1]:
        fig_trend = +0.5  # Generally improving
    elif figs[0] < figs[-1]:
        fig_trend = -0.5  # Generally declining

    # Combined trend score
    trend_score = (finish_trend + fig_trend) / 2.0

    # Determine cycle
    cycle = "stable"
    bonus = 0.0

    if trend_score >= 0.75:
        cycle = "improving"
        bonus = +0.15  # Strong positive form
    elif trend_score >= 0.25:
        cycle = "peaking"
        bonus = +0.10  # Solid form
    elif trend_score <= -0.75:
        cycle = "declining"
        bonus = -0.15  # Poor form
    elif trend_score <= -0.25:
        cycle = "bottoming"
        bonus = -0.08  # Weak form

    # Bonus for consistency (all recent finishes in top 3)
    if all(f <= 3 for f in finishes[:3]):
        bonus += 0.05  # Consistent runner

    return {
        "cycle": cycle,
        "trend_score": trend_score,
        "last_3_finishes": finishes[:3],
        "last_3_figs": figs[:3],
        "bonus": float(np.clip(bonus, -0.20, 0.20)),
    }




def post_bias_score_multi(post_bias_picks: list, post_str: str) -> float:
    """Calculate post bias score from multiple selected post biases (aggregates bonuses)"""
    if not post_bias_picks:
        return 0.0

    # ELITE: Use optimized post parser
    post = _parse_post_number(post_str)
    if not post:
        return 0.0

    total_bonus = 0.0
    table = {
        "favors rail (1)": lambda p: (
            MODEL_CONFIG["post_bias_rail_bonus"] if p == 1 else 0.0
        ),
        "favors inner (1-3)": lambda p: (
            MODEL_CONFIG["post_bias_inner_bonus"] if p and 1 <= p <= 3 else 0.0
        ),
        "favors mid (4-7)": lambda p: (
            MODEL_CONFIG["post_bias_mid_bonus"] if p and 4 <= p <= 7 else 0.0
        ),
        "favors outside (8+)": lambda p: (
            MODEL_CONFIG["post_bias_outside_bonus"] if p and p >= 8 else 0.0
        ),
        "no significant post bias": lambda _p: 0.0,
    }

    # Aggregate bonuses from all selected post biases
    for pick in post_bias_picks:
        pick_lower = (pick or "").strip().lower()
        fn = table.get(pick_lower, table["no significant post bias"])
        bonus = fn(post)
        if bonus > 0:  # Only add positive bonuses (horse matches this bias category)
            total_bonus += bonus

    return float(
        np.clip(total_bonus, -0.5, 0.8)
    )  # WIDENED ceiling from 0.5 — let post bias register




def compute_bias_ratings(
    df_styles: pd.DataFrame,
    surface_type: str,
    distance_txt: str,
    condition_txt: str,
    race_type: str,
    running_style_bias,  # Can be list or str
    post_bias_pick,  # Can be list or str
    _ppi_value: float = 0.0,  # Unused — ppi_map is recalculated inside
    pedigree_per_horse: dict[str, dict] | None = None,
    track_name: str = "",
    pp_text: str = "",
    figs_df: pd.DataFrame | None = None,
    dynamic_weights: dict | None = None,
) -> pd.DataFrame:
    """
    Reads 'Cclass' and 'Cform' from df_styles (pre-built), adds Cstyle/Cpost/Cpace/Cspeed (+Atrack),
    plus Tier 2 bonuses (Impact Values, SPI, Surface Stats, AWD),
    sums to Arace and R. Returns rating table.

    ULTRATHINK V2: Can use unified rating engine if available and PP text provided.

    UPDATED: Now accepts lists of biases to aggregate bonuses from ALL selected biases.
    """
    # ── Runtime contracts: fail fast on bad inputs ──
    if df_styles is None or df_styles.empty:
        logger.warning(
            "compute_bias_ratings: empty df_styles — returning empty DataFrame"
        )
        return pd.DataFrame()
    for _req_col in ("Horse", "Cclass", "Cform"):
        if _req_col not in df_styles.columns:
            logger.error(
                f"compute_bias_ratings: missing required column '{_req_col}' in df_styles"
            )
            return pd.DataFrame()
    if not surface_type:
        logger.warning(
            "compute_bias_ratings: surface_type is empty, defaulting to 'Dirt'"
        )
        surface_type = "Dirt"
    if not distance_txt:
        logger.warning(
            "compute_bias_ratings: distance_txt is empty, defaulting to '6f'"
        )
        distance_txt = "6f"

    cols = [
        "#",
        "Post",
        "Horse",
        "Style",
        "Quirin",
        "Cstyle",
        "Cpost",
        "Cpace",
        "Cspeed",
        "Cclass",
        "Cform",
        "Atrack",
        "Tier2_Bonus",
        "Arace",
        "R",
    ]
    if df_styles is None or df_styles.empty:
        return pd.DataFrame(columns=cols)

    # ===== ULTRATHINK V2: Try unified engine first if available =====
    _ml_odds_lookup = {}  # Preserved for ML Odds Reality Guard
    use_unified_engine = False
    if (
        UNIFIED_ENGINE_AVAILABLE
        and ELITE_PARSER_AVAILABLE
        and pp_text
        and len(pp_text.strip()) > 100
    ):
        try:
            # Parse with elite parser
            parser = GoldStandardBRISNETParser()
            horses = parser.parse_full_pp(pp_text, debug=False)

            # Store full elite parsed data for downstream DB storage
            st.session_state["elite_horses_data"] = {
                name: obj.to_dict() for name, obj in horses.items()
            }

            # Validate parsing quality
            validation = parser.validate_parsed_data(horses, min_confidence=0.5)
            avg_confidence = validation.get("overall_confidence", 0.0)

            if avg_confidence >= 0.15 and validation.get("horses_parsed", 0) > 0:
                # High quality parse - use unified engine
                # Load track-specific weights if available (falls back to global)
                _track_weights = LEARNED_WEIGHTS
                if ADAPTIVE_LEARNING_AVAILABLE and track_name:
                    with contextlib.suppress(Exception):  # Fall back to global weights
                        _track_weights = get_live_learned_weights(
                            PERSISTENT_DB_PATH,
                            track_code=track_name.upper(),
                        )

                engine = UnifiedRatingEngine(
                    softmax_tau=3.0, learned_weights=_track_weights
                )

                # Extract race metadata using elite parser's race header (most accurate)
                today_purse = 0
                final_distance = distance_txt
                extracted_race_type = race_type

                if hasattr(parser, "race_header") and parser.race_header:
                    race_header = parser.race_header
                    today_purse = race_header.get("purse", 0)
                    if race_header.get("distance"):
                        final_distance = race_header.get("distance")
                    if race_header.get("race_type_normalized"):
                        extracted_race_type = race_header.get("race_type_normalized")

                # Fallback to comprehensive extraction if parser didn't get purse
                if today_purse == 0:
                    race_metadata = extract_race_metadata_from_pp_text(pp_text)
                    today_purse = race_metadata.get("purse_amount", 0)

                # Final fallback to session state
                if today_purse == 0:
                    today_purse = st.session_state.get("purse_val", 20000)

                # ═══ ML BLEND: Compute PyTorch model scores ═══
                _ml_scores = None
                if ML_BLEND_AVAILABLE and _ml_blend and _ml_blend.model is not None:
                    try:
                        # Build feature dicts from parsed HorseData
                        horse_features = {}
                        for hname, hdata in horses.items():
                            horse_features[hname] = {
                                "odds": getattr(hdata, "ml_odds", 0) or 0,
                                "prime_power": getattr(hdata, "prime_power", 0) or 0,
                                "last_speed_rating": getattr(hdata, "last_fig", 0) or 0,
                                "class_rating": getattr(hdata, "class_rating", 0) or 0,
                                "best_speed_at_distance": getattr(hdata, "avg_top2", 0)
                                or 0,
                                "days_since_last_race": getattr(
                                    hdata, "days_since_last", 30
                                )
                                or 30,
                                "career_starts": getattr(hdata, "starts", 0) or 0,
                                "post_position": int(
                                    "".join(
                                        c
                                        for c in str(getattr(hdata, "post", "5"))
                                        if c.isdigit()
                                    )
                                    or "5"
                                ),
                                "field_size": len(horses),
                                "predicted_finish_position": 0,
                                "prediction_error": 0,
                                "jockey_win_pct": getattr(hdata, "jockey_win_pct", 0)
                                or 0,
                                "trainer_win_pct": getattr(hdata, "trainer_win_pct", 0)
                                or 0,
                            }
                        _ml_scores = _ml_blend.score_from_gold_features(
                            list(horse_features.values()),
                            horse_names=list(horse_features.keys()),
                        )
                        if _ml_scores:
                            logger.info(
                                f"🧠 ML blend scores computed for {len(_ml_scores)} horses"
                            )
                    except Exception as e:
                        logger.warning(f"ML blend scoring failed (using fallback): {e}")
                        _ml_scores = None

                # Get predictions with extracted metadata
                results_df = engine.predict_race(
                    pp_text=pp_text,
                    today_purse=today_purse,
                    today_race_type=extracted_race_type,
                    track_name=track_name,
                    surface_type=surface_type,
                    distance_txt=final_distance,
                    condition_txt=condition_txt,
                    style_bias=(
                        running_style_bias
                        if isinstance(running_style_bias, list)
                        else [running_style_bias]
                        if running_style_bias
                        else None
                    ),
                    post_bias=(
                        post_bias_pick
                        if isinstance(post_bias_pick, list)
                        else [post_bias_pick]
                        if post_bias_pick
                        else None
                    ),
                    ml_scores=_ml_scores,
                )

                if not results_df.empty:
                    # Filter to only include horses that are NOT scratched (i.e., in df_styles)
                    # Use normalized names for matching to handle apostrophes and spacing differences
                    # (normalize_horse_name is defined at top of file for consistency)

                    # Build mapping of normalized names to original names from Section A
                    section_a_names = {
                        normalize_horse_name(h): h for h in df_styles["Horse"].tolist()
                    }

                    # Filter and rename horses to match Section A names
                    results_df_filtered = results_df.copy()
                    matched_horses = []
                    for idx, row in results_df_filtered.iterrows():
                        parsed_name = row["Horse"]
                        normalized = normalize_horse_name(parsed_name)
                        if normalized in section_a_names:
                            # Use the exact name from Section A
                            results_df_filtered.at[idx, "Horse"] = section_a_names[
                                normalized
                            ]
                            matched_horses.append(idx)

                    # Keep only matched horses
                    results_df_filtered = results_df_filtered.loc[matched_horses].copy()

                    if results_df_filtered.empty:
                        # All horses from unified engine were scratched, fall back to traditional
                        st.warning(
                            "⚠️ All unified engine horses are scratched (using fallback)"
                        )
                    else:
                        # Build figs_df from elite parser speed figures
                        figs_data = []
                        for _, row in results_df_filtered.iterrows():
                            horse_name = row["Horse"]
                            # Get speed figures from elite parser's horses dict
                            if (
                                hasattr(parser, "horses")
                                and horse_name in parser.horses
                            ):
                                horse_obj = parser.horses[horse_name]
                                if (
                                    horse_obj.speed_figures
                                    and len(horse_obj.speed_figures) > 0
                                ):
                                    figs_data.append(
                                        {
                                            "Horse": horse_name,
                                            "Figures": horse_obj.speed_figures,
                                            "BestFig": max(horse_obj.speed_figures),
                                            "AvgTop2": horse_obj.avg_top2,
                                        }
                                    )
                        figs_df = (
                            pd.DataFrame(figs_data) if figs_data else pd.DataFrame()
                        )

                        # Convert unified engine output to app.py format
                        # CRITICAL FIX: Use actual program numbers (saddle-cloth)
                        # instead of sequential range(1,N) which overwrote real #s
                        unified_ratings = pd.DataFrame(
                            {
                                "#": results_df_filtered["Program"].astype(str)
                                if "Program" in results_df_filtered.columns
                                else results_df_filtered["Post"].astype(str),
                                "Post": results_df_filtered["Post"].astype(str),
                                "Horse": results_df_filtered["Horse"],
                                "Style": results_df_filtered.get("Pace_Style", "NA"),
                                "Quirin": results_df_filtered.get("Quirin", 0.0),
                                "Cstyle": results_df_filtered.get("Cstyle", 0.0),
                                "Cpost": results_df_filtered.get("Cpost", 0.0),
                                "Cpace": results_df_filtered.get("Cpace", 0.0),
                                "Cspeed": results_df_filtered.get("Cspeed", 0.0),
                                "Cclass": results_df_filtered.get("Cclass", 0.0),
                                "Cform": results_df_filtered.get("Cform", 0.0),
                                "Atrack": 0.0,  # Placeholder - computed below
                                "Arace": results_df_filtered["Rating"],
                                "R": results_df_filtered["Rating"],
                                "Prime Power": results_df_filtered.get(
                                    "Prime_Power", 0.0
                                ),
                                "Parsing_Confidence": results_df_filtered.get(
                                    "Parsing_Confidence", avg_confidence
                                ),
                                "Tier2_Bonus": results_df_filtered.get(
                                    "Tier2_Bonus", 0.0
                                ),
                            }
                        )

                        # ═══════════════════════════════════════════════════════
                        # CRITICAL FIX: Populate Atrack from track bias profiles
                        # Previously always 0.0 because unified engine doesn't
                        # produce an A_Track column (bias is baked into cstyle/cpost)
                        # but we still need the display column for the UI
                        # ═══════════════════════════════════════════════════════
                        for idx, row in unified_ratings.iterrows():
                            style_val = _style_norm(str(row.get("Style", "NA")))
                            post_val = str(row.get("Post", ""))
                            a_track = _get_track_bias_delta(
                                track_name,
                                surface_type,
                                distance_txt,
                                style_val,
                                post_val,
                            )
                            unified_ratings.at[idx, "Atrack"] = a_track
                            # CRITICAL FIX: A-Track must be added to R (previously display-only)
                            unified_ratings.at[idx, "R"] = (
                                unified_ratings.at[idx, "R"] + a_track
                            )

                        # Add success message
                        scratched_count = len(results_df) - len(results_df_filtered)
                        if scratched_count > 0:
                            st.info(
                                f"🎯 Using Unified Rating Engine (Elite Parser confidence: {avg_confidence:.1%}) - {scratched_count} scratched horse(s) excluded"
                            )
                        else:
                            st.info(
                                f"🎯 Using Unified Rating Engine (Elite Parser confidence: {avg_confidence:.1%})"
                            )

                        # ═══════════════════════════════════════════════════════
                        # CRITICAL FIX: Replace NaN/None ratings with ML-odds fallback
                        # Without this, unparsed horses get NaN and corrupt A/B/C/D groups
                        # ═══════════════════════════════════════════════════════
                        for idx, row in unified_ratings.iterrows():
                            rating_val = row.get("R")
                            # Check if rating is NaN, None, or not a real number
                            is_bad_rating = (
                                rating_val is None
                                or (
                                    isinstance(rating_val, float)
                                    and (
                                        np.isnan(rating_val)
                                        or not np.isfinite(rating_val)
                                    )
                                )
                                or (
                                    isinstance(rating_val, str)
                                    and rating_val.lower() == "none"
                                )
                            )
                            if is_bad_rating:
                                # ═══════════════════════════════════════════════════════
                                # PP-ENHANCED FALLBACK (CT R5 Fix — Feb 11, 2026)
                                # Old: Pure ML fallback gave Wiley Willard (PP 113.1,
                                # 2/1 fav) a rating of 1.81 → ranked DEAD LAST.
                                # New: When Prime Power is available, use it as the
                                # primary rating signal (70% PP, 30% ML) to produce
                                # competitive ratings that reflect actual ability.
                                # ═══════════════════════════════════════════════════════
                                horse_name = row.get("Horse", "")
                                ml_val = None
                                # CRITICAL FIX (Feb 18, 2026): Check unified_ratings row
                                # first for Prime Power — df_styles (Section A) never had
                                # this column, so PP was always 0.0 in fallback.
                                pp_val = safe_float(row.get("Prime Power", 0.0), 0.0)
                                for _, ff_row in df_styles.iterrows():
                                    if ff_row.get("Horse") == horse_name:
                                        ml_val = ff_row.get("ML", "")
                                        if pp_val == 0.0:
                                            pp_val = safe_float(
                                                ff_row.get("Prime Power", 0.0), 0.0
                                            )
                                        break
                                try:
                                    ml_dec = (
                                        odds_to_decimal(str(ml_val)) if ml_val else 20.0
                                    )
                                except Exception:
                                    ml_dec = 20.0
                                ml_base = 3.0 - np.log(max(ml_dec, 1.1))
                                if pp_val > 0:
                                    # PP-enhanced: maps PP to competitive rating range
                                    # PP 100→0, 105→2.75, 110→5.5, 115→8.25
                                    pp_base = max(0, (pp_val - 100)) * 0.55
                                    fallback_rating = round(
                                        0.70 * pp_base + 0.30 * ml_base, 2
                                    )
                                else:
                                    fallback_rating = round(ml_base, 2)
                                unified_ratings.at[idx, "R"] = fallback_rating
                                unified_ratings.at[idx, "Arace"] = fallback_rating
                                # Zero out component columns so they display as 0.00 not None
                                for col in [
                                    "Cstyle",
                                    "Cpost",
                                    "Cpace",
                                    "Cspeed",
                                    "Cclass",
                                    "Cform",
                                    "Atrack",
                                ]:
                                    if (
                                        pd.isna(unified_ratings.at[idx, col])
                                        or unified_ratings.at[idx, col] is None
                                    ):
                                        unified_ratings.at[idx, col] = 0.0
                                logger.info(
                                    f"  → Fallback rating for {horse_name}: {fallback_rating} "
                                    f"(PP={pp_val}, ML={ml_val})"
                                )

                        # ═══════════════════════════════════════════════════════
                        # CRITICAL FIX: Add MISSING horses from df_final_field
                        # Parser may not match all Section A horses (name format
                        # differences, partial PP text, etc). Without this fix,
                        # missing horses vanish from primary_df → missing from
                        # Section E display → can't enter results for them.
                        # ═══════════════════════════════════════════════════════
                        unified_horse_names = set(unified_ratings["Horse"].tolist())
                        original_horses = (
                            df_styles["Horse"].tolist()
                            if "Horse" in df_styles.columns
                            else []
                        )
                        missing_horses = [
                            h for h in original_horses if h not in unified_horse_names
                        ]

                        if missing_horses:
                            logger.info(
                                f"  → Adding {len(missing_horses)} horses not matched by parser: {missing_horses}"
                            )
                            missing_rows = []
                            for mh in missing_horses:
                                # Get ML odds for fallback rating
                                # Use df_styles (original parameter = df_final_field copy)
                                ml_val = None
                                post_val = ""
                                prog_val = ""
                                style_val = "NA"
                                quirin_val = 0.0
                                pp_val = 0.0
                                for _, ff_row in df_styles.iterrows():
                                    if ff_row.get("Horse") == mh:
                                        ml_val = ff_row.get("ML", "")
                                        post_val = str(ff_row.get("Post", ""))
                                        prog_val = str(ff_row.get("#", post_val))
                                        style_val = str(
                                            ff_row.get(
                                                "Style", ff_row.get("BRIS Style", "NA")
                                            )
                                        )
                                        quirin_val = ff_row.get("Quirin", 0.0)
                                        pp_val = safe_float(
                                            ff_row.get("Prime Power", 0.0), 0.0
                                        )
                                        break
                                try:
                                    ml_dec = (
                                        odds_to_decimal(str(ml_val)) if ml_val else 20.0
                                    )
                                except Exception:
                                    ml_dec = 20.0
                                ml_base = 3.0 - np.log(max(ml_dec, 1.1))
                                if pp_val > 0:
                                    pp_base = max(0, (pp_val - 100)) * 0.55
                                    fallback_rating = round(
                                        0.70 * pp_base + 0.30 * ml_base, 2
                                    )
                                else:
                                    fallback_rating = round(ml_base, 2)
                                a_track = _get_track_bias_delta(
                                    track_name,
                                    surface_type,
                                    distance_txt,
                                    _style_norm(style_val),
                                    post_val,
                                )

                                missing_rows.append(
                                    {
                                        "#": prog_val or post_val,
                                        "Post": post_val,
                                        "Horse": mh,
                                        "Style": style_val,
                                        "Quirin": quirin_val,
                                        "Cstyle": 0.0,
                                        "Cpost": 0.0,
                                        "Cpace": 0.0,
                                        "Cspeed": 0.0,
                                        "Cclass": 0.0,
                                        "Cform": 0.0,
                                        "Atrack": a_track,
                                        "Arace": fallback_rating,
                                        "R": fallback_rating,
                                        "Parsing_Confidence": 0.0,
                                    }
                                )
                                logger.info(
                                    f"    + {mh}: fallback R={fallback_rating}, post={post_val}, style={style_val}"
                                )

                            if missing_rows:
                                missing_df = pd.DataFrame(missing_rows)
                                unified_ratings = pd.concat(
                                    [unified_ratings, missing_df], ignore_index=True
                                )
                            st.caption(
                                f"ℹ️ {len(missing_horses)} horse(s) added via ML-odds fallback (not in PP text)"
                            )

                        # ═══════════════════════════════════════════════════════
                        # Preserve ML odds for Reality Guard (before overwriting)
                        # ═══════════════════════════════════════════════════════
                        if "ML" in df_styles.columns:
                            for _, _ml_row in df_styles.iterrows():
                                _ml_odds_lookup[str(_ml_row.get("Horse", ""))] = (
                                    _ml_row.get("ML", "")
                                )

                        # Continue to apply enhancements instead of returning early
                        df_styles = unified_ratings.copy()
                        use_unified_engine = True
        except Exception as e:
            # Fallback to traditional method if unified engine fails
            st.warning(f"⚠️ Unified engine error (using fallback): {str(e)[:100]}")
            use_unified_engine = False
            pass
    # ===== End ULTRATHINK V2 integration =====

    # Skip traditional rating calculation if unified engine was used successfully
    if use_unified_engine:
        # CRITICAL FIX (Feb 10, 2026): Apply outlier clip before returning.
        # Previously the unified path returned BEFORE the R > 20 clip at the
        # bottom of this function, allowing runaway ratings (e.g., 17.69).
        if "R" in df_styles.columns:
            for idx, row in df_styles.iterrows():
                r_val = float(row.get("R", 0.0))
                if r_val > 20 or r_val < -5:
                    df_styles.at[idx, "R"] = round(float(np.clip(r_val, -3, 15)), 2)
                    df_styles.at[idx, "Arace"] = df_styles.at[idx, "R"]

        # ═══════════════════════════════════════════════════════════════
        # PACE SCENARIO BONUS FOR UNIFIED PATH (Feb 10, 2026 — TuP R3)
        # The unified engine returns early and skips the pace scenario
        # detection below. TuP R3: 5 E/E/P types created a speed duel,
        # McClusky (S-style closer) won from 6th predicted rank.
        # This adds the closer advantage bonus to the unified path.
        # ═══════════════════════════════════════════════════════════════
        if (
            df_styles is not None
            and not df_styles.empty
            and "Style" in df_styles.columns
        ):
            speed_types = 0
            total_in_field = len(df_styles)
            for _, h_row in df_styles.iterrows():
                h_style = str(h_row.get("Style", "")).upper()
                if "E" in h_style:  # E or E/P types
                    speed_types += 1

            # Dynamic threshold: 50%+ of field being speed types triggers bonus
            # 8 horse field: 4+ speed types. 10 horse field: 5+.
            speed_threshold = max(4, int(total_in_field * 0.50))
            if speed_types >= speed_threshold:
                for idx, row in df_styles.iterrows():
                    h_style = str(row.get("Style", "")).upper()
                    if h_style in ["P", "S"] or "S" in h_style:
                        pace_bonus = 1.0 if h_style == "S" else 0.60
                        old_r = float(row.get("R", 0.0))
                        df_styles.at[idx, "R"] = round(old_r + pace_bonus, 2)
                        df_styles.at[idx, "Arace"] = df_styles.at[idx, "R"]

        # ═══════════════════════════════════════════════════════════════
        # ML ODDS REALITY GUARD (CT R5 Fix — Feb 11, 2026)
        # ═══════════════════════════════════════════════════════════════
        # CT R5 failures: Noballstwostrikes (29/1) rated #2 at 8.19,
        # Cedar Runs Fiber (30/1) rated #3 at 8.14 — both missed top 4.
        # Meanwhile Wiley Willard (2/1 FAVORITE) was ranked LAST.
        #
        # Rules:
        #   Longshot (ML > 12/1): Rating capped at field median
        #   Favorite (ML ≤ 5/2): Rating floored at field_avg - 1.5
        # ═══════════════════════════════════════════════════════════════
        if _ml_odds_lookup and "R" in df_styles.columns:
            all_r_vals = [
                float(row.get("R", 0.0))
                for _, row in df_styles.iterrows()
                if pd.notna(row.get("R"))
            ]
            if len(all_r_vals) >= 3:
                field_median = float(np.median(all_r_vals))
                field_avg = float(np.mean(all_r_vals))
                for idx, row in df_styles.iterrows():
                    horse_name = str(row.get("Horse", ""))
                    ml_str = _ml_odds_lookup.get(horse_name, "")
                    try:
                        ml_dec = odds_to_decimal(str(ml_str)) if ml_str else 6.0
                    except Exception:
                        ml_dec = 6.0
                    r_val = float(row.get("R", 0.0))

                    # LONGSHOT CAP: >12/1 horses can't rate above field median
                    if ml_dec > 13.0 and r_val > field_median:
                        capped = round(field_median, 2)
                        df_styles.at[idx, "R"] = capped
                        df_styles.at[idx, "Arace"] = capped
                        logger.info(
                            f"  → ML Guard: {horse_name} (ML={ml_str}) "
                            f"capped {r_val:.2f}→{capped:.2f}"
                        )
                    # FAVORITE FLOOR: ≤5/2 horses can't rate below avg - 1.5
                    elif ml_dec <= 3.5 and r_val < field_avg - 1.5:
                        floored = round(field_avg - 1.5, 2)
                        df_styles.at[idx, "R"] = floored
                        df_styles.at[idx, "Arace"] = floored
                        logger.info(
                            f"  → ML Guard: {horse_name} (ML={ml_str}) "
                            f"floored {r_val:.2f}→{floored:.2f}"
                        )

        # ═══════════════════════════════════════════════════════════════════════
        # BRIDGE ARCHITECTURE (Feb 18, 2026):
        # Instead of returning early, fall through to the traditional Tier 2
        # bonus computation. The unified engine computed the 6 core components
        # (Class, Form, Speed, Pace, Style, Post) — those are KEPT as-is.
        # The traditional tier2 bonuses add 20 additional angles/adjustments
        # (trainer, layoff, career futility, pace supremacy, weekly bias, etc.)
        # that the unified engine doesn't compute.
        # The 5 overlapping bonuses (SPI, surface switch, surface stats,
        # workouts, class movement) are SKIPPED to avoid double-counting.
        # ═══════════════════════════════════════════════════════════════════════
        logger.info(
            "🌉 BRIDGE: Applying traditional Tier 2 bonuses on unified engine ratings"
        )

    # Ensure class and form columns present
    if "Cclass" not in df_styles.columns:
        df_styles = df_styles.copy()
        df_styles["Cclass"] = 0.0  # Default Cclass if missing
    if "Cform" not in df_styles.columns:
        df_styles = df_styles.copy()
        df_styles["Cform"] = 0.0  # Default Cform if missing

    # Derive per-horse pace tailwind from PPI
    ppi_map = compute_ppi(df_styles).get("by_horse", {})

    # ======================== Phase 1: Parse Tier 2 Enhancements ========================
    impact_values = parse_track_bias_impact_values(pp_text) if pp_text else {}
    spi_values = parse_pedigree_spi(pp_text) if pp_text else {}
    surface_stats = parse_pedigree_surface_stats(pp_text) if pp_text else {}
    awd_analysis = parse_awd_analysis(pp_text) if pp_text else {}

    # RACE AUDIT ENHANCEMENT: Parse weekly post-position bias data
    weekly_post_impacts = parse_weekly_post_bias(pp_text) if pp_text else {}

    # CRITICAL FIX (Feb 20, 2026): Store impact_values in session_state so
    # cross-function code (e.g., P-style bias check at form_cycle L4945)
    # can access weekly bias data. Previously this key was READ but NEVER WRITTEN,
    # making the P-style bias fix dead code.
    if impact_values:
        st.session_state["weekly_bias_impacts"] = impact_values

    # NEW: Parse comprehensive Track Bias Stats (Wire%, Speed Bias%, WnrAvgBL, %Races Won)
    track_bias_stats = parse_track_bias_stats(pp_text) if pp_text else {}
    if track_bias_stats:
        st.session_state["track_bias_stats"] = track_bias_stats

    # NEW: Parse Race Summary ranking tables (Speed Last Race, Current Class, etc.)
    race_summary_rankings = parse_race_summary_rankings(pp_text) if pp_text else {}
    if race_summary_rankings:
        st.session_state["race_summary_rankings"] = race_summary_rankings

    # NEW: Parse BRIS Pace & Speed Pars (E1, E2/Late, SPD)
    # These are RACE-LEVEL pars (same for all horses) representing the average
    # pace/speed ratings of the leader/winner at today's class/distance.
    pace_speed_pars = parse_pace_speed_pars(pp_text) if pp_text else {}
    if pace_speed_pars:
        st.session_state["pace_speed_pars"] = pace_speed_pars

    # RACE AUDIT ENHANCEMENT: Calculate weekly bias amplifier multiplier
    # When extreme biases detected (e.g., E=2.05, S=0.32), amplify track_bias_mult
    weekly_bias_amplifier = calculate_weekly_bias_amplifier(impact_values)

    # Build per-horse PP block lookup for tier-2 (searches must use horse block, NOT full pp_text)
    _horse_pp_blocks = (
        {
            chunk_name: block_text
            for _post, chunk_name, block_text in split_into_horse_chunks(pp_text)
        }
        if pp_text
        else {}
    )

    # DIAGNOSTIC (Feb 13, 2026): Detect horses in df_styles that have NO PP block match.
    # When split_into_horse_chunks fails to match a horse (e.g., regex edge cases with
    # commas in trainer names, or "Previously trained by" blocks), that horse gets
    # _horse_block = full pp_text as fallback, which causes ALL component parsers to
    # return wrong values or zeros. Log these mismatches so they surface immediately.
    if _horse_pp_blocks and df_styles is not None and not df_styles.empty:
        _matched_names = set(_horse_pp_blocks.keys())
        for _, _diag_row in df_styles.iterrows():
            _diag_name = str(_diag_row.get("Horse", ""))
            if _diag_name and _diag_name not in _matched_names:
                logger.warning(
                    f"⚠️ ZERO-SCORE RISK: Horse '{_diag_name}' (post {_diag_row.get('Post', '?')}) "
                    f"has no PP block match in split_into_horse_chunks. "
                    f"Falling back to full pp_text — component scores may be incorrect. "
                    f"Matched blocks: {sorted(_matched_names)}"
                )

    # ======================== TRACK PATTERN LEARNING: Fetch learned patterns ========================
    _track_patterns: dict = {}
    try:
        # Use the global gold_db instance
        _gold_db = globals().get("gold_db")
        if _gold_db and hasattr(_gold_db, "get_track_patterns"):
            _track_patterns = _gold_db.get_track_patterns(
                track_code=track_name,
                surface=surface_type,
                distance=distance_txt,
            )
            if _track_patterns:
                logger.info(
                    f"📊 Loaded {len(_track_patterns)} track patterns for "
                    f"{track_name} {surface_type} {distance_txt}"
                )
    except Exception as tp_err:
        logger.debug(f"Track pattern fetch skipped: {tp_err}")

    # Calculate speed component from figs_df
    speed_map = {}
    if figs_df is not None and not figs_df.empty and "AvgTop2" in figs_df.columns:
        race_avg_fig = figs_df["AvgTop2"].mean()
        # Pre-compute field last-race average for recency floor check
        _field_last_avg = None
        if "LastFig" in figs_df.columns:
            _last_figs_valid = figs_df.loc[figs_df["LastFig"] > 0, "LastFig"]
            if not _last_figs_valid.empty:
                _field_last_avg = _last_figs_valid.mean()

        for _, fig_row in figs_df.iterrows():
            horse_name = fig_row["Horse"]
            horse_fig = fig_row["AvgTop2"]
            # Normalize to race average: positive means faster than average
            raw_speed = (horse_fig - race_avg_fig) * MODEL_CONFIG["speed_fig_weight"]

            # ═══════════════════════════════════════════════════════════
            # CONDITION MODIFIER: Dampen speed figures on off-track days
            # Speed figures are less predictive on sloppy/muddy/heavy
            # surfaces because track-variant adjustments are unreliable.
            # condition_modifiers: fast=1.0, sloppy=1.10, heavy=1.10
            # Dampening: (1.10-1.0)*1.5 = 15% reduction in speed weight
            # ═══════════════════════════════════════════════════════════
            cond_mod = condition_modifiers.get(condition_txt.lower(), 1.0)
            if cond_mod > 1.0:
                dampening = 1.0 - (cond_mod - 1.0) * 1.5
                raw_speed *= max(dampening, 0.60)  # Floor at 60%

            # ═══════════════════════════════════════════════════════════
            # SPEED RECENCY FLOOR (Feb 13, 2026 — TAM R7 Post-Race Fix)
            # If a horse's last figure is >25 below the field's last-race
            # average, its speed component is hard-capped. Prevents stale
            # avg_top2 from propping up severely declining horses.
            # Example: Island Spirit avg_top2=75 but last_fig=48 → cap.
            # ═══════════════════════════════════════════════════════════
            if _field_last_avg is not None and "LastFig" in fig_row:
                _horse_last = fig_row.get("LastFig", 0)
                if _horse_last > 0:
                    _recency_deficit = _field_last_avg - _horse_last
                    if _recency_deficit > 25:
                        # Severe recent collapse → hard cap at -1.5
                        raw_speed = min(raw_speed, -1.5)
                    elif _recency_deficit > 18:
                        # Significant decline → moderate cap at -0.8
                        raw_speed = min(raw_speed, -0.8)

            speed_map[horse_name] = raw_speed

    rows = []
    _race_class_shown = False  # BUG 7 FIX: show Race Classification expander only once

    # Convert single values to lists for uniform processing
    style_biases = (
        running_style_bias
        if isinstance(running_style_bias, list)
        else [running_style_bias]
    )
    post_biases_list = (
        post_bias_pick if isinstance(post_bias_pick, list) else [post_bias_pick]
    )

    # ENHANCEMENT 6 PRE-COMPUTE: Build field-wide E1 lookup for pace supremacy bonus
    # Each horse's best E1 (from up to 3 most recent races) is stored so the tier-2
    # loop can compare any horse's E1 to the entire field.
    _field_e1_values: dict[str, float] = {}
    for _fe_name, _fe_block in _horse_pp_blocks.items():
        try:
            _fe_pace = parse_e1_e2_lp_values(_fe_block)
            if _fe_pace and _fe_pace.get("e1"):
                _field_e1_values[_fe_name] = max(_fe_pace["e1"][:3])
        except BaseException:
            pass

    for _, row in df_styles.iterrows():
        post = str(row.get("Post", row.get("#", "")))
        name = str(row.get("Horse"))
        style = _style_norm(
            row.get("Style") or row.get("OverrideStyle") or row.get("DetectedStyle")
        )
        quirin = row.get("Quirin", np.nan)  # Keep as potential NaN

        if use_unified_engine:
            # BRIDGE: Use unified engine's pre-computed core components
            cstyle = float(row.get("Cstyle", 0.0))
            cpost = float(row.get("Cpost", 0.0))
            cpace = float(row.get("Cpace", 0.0))
            cspeed = float(row.get("Cspeed", 0.0))
            c_class = float(row.get("Cclass", 0.0))
            c_form = float(row.get("Cform", 0.0))
            a_track = float(row.get("Atrack", 0.0))
            # Store pre-bridge R for later delta calculation
            _pre_bridge_R = float(row.get("R", 0.0))
        else:
            # TRADITIONAL: Compute core components from Section A inputs
            # Use multi-bias functions to aggregate bonuses from ALL selected biases
            cstyle = style_match_score_multi(style_biases, style, quirin)
            cpost = post_bias_score_multi(post_biases_list, post)
            cpace = float(ppi_map.get(name, 0.0))
            cspeed = float(speed_map.get(name, 0.0))  # Speed component from figures

            # Track bias with dynamic weight multiplier
            dw = dynamic_weights or {}
            track_bias_mult = dw.get("track_bias", 1.0)
            # RACE AUDIT ENHANCEMENT 1: Amplify track_bias_mult when extreme weekly biases detected
            track_bias_mult *= weekly_bias_amplifier
            a_track = (
                _get_track_bias_delta(
                    track_name, surface_type, distance_txt, style, post
                )
                * track_bias_mult
            )

            # Get pre-computed Cclass and Cform from df_styles (calculated in Section A)
            c_class = float(row.get("Cclass", 0.0))
            c_form = float(row.get("Cform", 0.0))

        # ======================== Tier 2 Bonuses ========================
        tier2_bonus = 0.0

        # Per-horse PP block for tier-2 searches (falls back to full pp_text only if block not found)
        _horse_block = _horse_pp_blocks.get(name, pp_text)

        # Parse structured race history ONCE for this horse (reused by distance, class, form, surface)
        horse_race_history = parse_race_history_from_block(_horse_block)

        # ---- DISTANCE SPECIALIST BONUS (Race 4 Oaklawn tuning) ----
        # Winnemac Avenue: 9-4-2-0 at distance = 44% wins, yet got 0 credit
        # Parse 'Dis (XXX) starts wins - places - shows' from PP header
        try:
            dis_match = re.search(
                r"Dis\s*\(\d+\)\s+(\d+)\s+(\d+)\s*-\s*(\d+)\s*-\s*(\d+)", _horse_block
            )
            if dis_match:
                dis_starts = int(dis_match.group(1))
                dis_wins = int(dis_match.group(2))
                dis_places = int(dis_match.group(3))
                dis_shows = int(dis_match.group(4))
                if dis_starts >= 4:
                    dis_win_pct = dis_wins / dis_starts
                    dis_itm_pct = (dis_wins + dis_places + dis_shows) / dis_starts
                    if dis_win_pct >= 0.35:
                        tier2_bonus += 0.8  # Elite distance specialist
                    elif dis_win_pct >= 0.25:
                        tier2_bonus += 0.5  # Strong at distance
                    elif dis_itm_pct >= 0.50:
                        tier2_bonus += 0.3  # Consistently competitive at distance
        except BaseException:
            pass

        # ---- TRACK SPECIALIST BONUS (Race 4 Oaklawn tuning) ----
        # Parse track-specific record: 'OP starts wins - places - shows'
        try:
            # Extract track code from track_name (e.g., 'Oaklawn Park' -> 'OP')
            track_code_for_search = ""
            track_code_map = {
                "oaklawn": "OP",
                "churchill": "CD",
                "saratoga": "Sar",
                "belmont": "Bel",
                "gulfstream": "GP",
                "keeneland": "Kee",
                "santa anita": "SA",
                "del mar": "Dmr",
                "fair grounds": "FG",
                "aqueduct": "Aqu",
                "laurel": "Lrl",
                "pimlico": "Pim",
                "remington": "RP",
                "turf paradise": "TuP",
                "tampa bay": "Tam",
                "parx": "Prx",
                "monmouth": "Mth",
                "woodbine": "WO",
            }
            for key, code in track_code_map.items():
                if key in (track_name or "").lower():
                    track_code_for_search = code
                    break

            if track_code_for_search:
                trk_pattern = rf"{re.escape(track_code_for_search)}\s+(\d+)\s+(\d+)\s*-\s*(\d+)\s*-\s*(\d+)"
                trk_match = re.search(trk_pattern, _horse_block)
                if trk_match:
                    trk_starts = int(trk_match.group(1))
                    trk_wins = int(trk_match.group(2))
                    trk_places = int(trk_match.group(3))
                    trk_shows = int(trk_match.group(4))
                    if trk_starts >= 4:
                        trk_win_pct = trk_wins / trk_starts
                        trk_itm_pct = (trk_wins + trk_places + trk_shows) / trk_starts
                        if trk_win_pct >= 0.30:
                            tier2_bonus += 0.6  # Track specialist
                        elif trk_win_pct >= 0.20:
                            tier2_bonus += 0.35  # Solid track record
                        elif trk_itm_pct >= 0.45:
                            tier2_bonus += 0.2  # Competitive at track
        except BaseException:
            pass

        # ---- P-STYLE ROUTE BONUS (Race 4 Oaklawn tuning) ----
        # P (presser/stalker) styles dominated Race 4 (top 3 all P)
        # In route races, P styles save ground and have tactical advantage
        try:
            race_furlongs = 8.0  # Default
            dist_lower = (distance_txt or "").lower()
            if "f" in dist_lower:
                race_furlongs = float(dist_lower.replace("f", "").strip())
            elif "mile" in dist_lower:
                race_furlongs = 8.0
                if "1/4" in dist_lower or "1.25" in dist_lower:
                    race_furlongs = 10.0
                elif "1/8" in dist_lower or "1.125" in dist_lower:
                    race_furlongs = 9.0
                elif "1/16" in dist_lower:
                    race_furlongs = 8.5
                elif "1/2" in dist_lower or "1.5" in dist_lower:
                    race_furlongs = 12.0
            if race_furlongs >= 8.0 and style == "P":
                tier2_bonus += 0.25  # Route tactical advantage for pressers
        except BaseException:
            pass

        # ELITE: Weather Impact
        weather_data = st.session_state.get("weather_data", None)
        if weather_data:
            tier2_bonus += calculate_weather_impact(weather_data, style, distance_txt)

        # 13. FIRST-TIME BLINKERS BONUS (TUP R6 Feb 19, 2026)
        # Stormylux: "1stTimeBlinkers 3 33% 33% +42.67" — 33% win rate, +42.67 ROI
        # Ranked 6th, WON at 3/1. This angle was completely invisible to the model.
        # First-time blinkers is one of the strongest trainer angles in handicapping.
        try:
            blinkers_match = re.search(
                r"1stTimeBlinkers\s+(\d+)\s+(\d+)%\s+(\d+)%(?:\s+([+-]?\d+\.?\d*))?",
                _horse_block,
            )
            if blinkers_match:
                blnk_starts = int(blinkers_match.group(1))
                blnk_win_pct = int(blinkers_match.group(2)) / 100.0
                blnk_itm_pct = int(blinkers_match.group(3)) / 100.0
                blnk_roi = (
                    float(blinkers_match.group(4)) if blinkers_match.group(4) else None
                )

                if blnk_starts >= 3:  # Meaningful sample size
                    if blnk_win_pct >= 0.30:  # 30%+ win rate
                        tier2_bonus += 1.0  # Elite blinkers angle
                    elif blnk_win_pct >= 0.20:  # 20-29%
                        tier2_bonus += 0.6  # Strong blinkers angle
                    elif blnk_itm_pct >= 0.50:  # 50%+ ITM
                        tier2_bonus += 0.3  # Moderate blinkers angle

                    # ROI-positive kicker
                    if blnk_roi is not None and blnk_roi > 0.0:
                        tier2_bonus += 0.4  # Positive ROI is extremely rare & valuable
        except BaseException:
            pass

        # ELITE: Jockey/Trainer Performance Impact
        tier2_bonus += calculate_jockey_trainer_impact(name, pp_text)

        # PHASES 2 & 3: Class Movement & Form Cycle — upgraded Feb 20, 2026
        # OLD CODE had two critical bugs:
        #   1. Class hack used undefined `claiming_price` (always NameError → silent skip)
        #   2. Form hack searched full `pp_text` (ALL horses) instead of per-horse block
        # Now uses the proper dead-function implementations with structured race history.
        try:
            # Reuse race history already parsed at top of tier2 section
            if horse_race_history:
                # --- CLASS MOVEMENT (analyze_class_movement) ---
                # Uses structured race_type hierarchy instead of fragile claiming regex
                class_result = analyze_class_movement(
                    horse_race_history,
                    race_type,
                    st.session_state.get("purse_val", 20000),
                )
                tier2_bonus += class_result.get("bonus", 0.0)

                # --- FORM CYCLE (analyze_form_cycle) ---
                # Uses per-horse finish positions + speed figures (not mixed pp_text)
                form_result = analyze_form_cycle(horse_race_history)
                tier2_bonus += form_result.get("bonus", 0.0)
        except Exception:
            pass

        # ======================== REPEAT CONTENDER BONUS (SA R8 audit Feb 20, 2026) ========================
        # Lino's Angel ran 2nd in THIS EXACT condition (OC50k/n1x) 28 days ago → model missed her.
        # If a horse placed (1st-3rd) in same race type at same purse level recently, reward.
        try:
            if horse_race_history and len(horse_race_history) >= 1:
                _today_rt_lower = (race_type or "").lower().replace(" ", "")
                _today_purse_val = st.session_state.get("purse_val", 0)
                _repeat_bonus = 0.0
                for _rr in horse_race_history[:3]:  # Last 3 races
                    _past_rt = (_rr.get("race_type", "") or "").lower().replace(" ", "")
                    _past_finish = _rr.get("finish_pos", 99)
                    # Check if same condition family (OC/ALW at similar level)
                    _same_family = False
                    if _today_rt_lower and _past_rt:
                        # Match OC/ALW condition codes (e.g., "oc50k" in both)
                        for _cond_key in [
                            "oc50",
                            "oc20",
                            "oc40",
                            "oc62",
                            "oc80",
                            "alw",
                            "aoc",
                            "stk",
                            "msw",
                        ]:
                            if _cond_key in _today_rt_lower and _cond_key in _past_rt:
                                _same_family = True
                                break
                    if _same_family and 1 <= _past_finish <= 3:
                        if _past_finish == 1:
                            _repeat_bonus = max(
                                _repeat_bonus, 0.25
                            )  # Won same condition
                        elif _past_finish == 2:
                            _repeat_bonus = max(
                                _repeat_bonus, 0.18
                            )  # 2nd in same condition
                        elif _past_finish == 3:
                            _repeat_bonus = max(
                                _repeat_bonus, 0.12
                            )  # 3rd in same condition
                tier2_bonus += _repeat_bonus
        except BaseException:
            pass

        # ELITE: Track Condition Granularity
        track_info = st.session_state.get("track_condition_detail", None)
        if track_info:
            tier2_bonus += calculate_track_condition_granular(track_info, style, post)

        # ======================== CLAIMING RACE PACE CAP (TUP R7 Feb 2026) ========================
        # Failed pick #3 Forest Acclamation: +0.84 pace × 1.5 = +1.26 rating boost
        # BUT: 0% trainer + 3/50 career wins + worst Prime Power = fatal flaws ignored
        # In claiming races: Talent/form/trainer > pace tactics
        # Cap pace bonus to prevent over-reliance on pace scenarios in cheap races

        # ======================== DISTANCE CATEGORY DETECTION ========================
        is_marathon = is_marathon_distance(distance_txt)
        is_sprint = is_sprint_distance(distance_txt)

        # Parse distance as furlongs for numeric comparisons
        race_furlongs = 8.0  # Default assumption
        try:
            if "f" in distance_txt.lower():
                race_furlongs = float(distance_txt.lower().replace("f", "").strip())
        except BaseException:
            pass

        # ======================== DISTANCE MOVEMENT ANALYSIS (V2 — Feb 10, 2026) ========================
        # Uses ACTUAL per-race distance data parsed from running lines (not starved anymore)
        try:
            if horse_race_history:
                # Use the NEW bridge function that feeds real distance data
                distance_analysis = analyze_distance_from_history(
                    horse_race_history, distance_txt
                )
                distance_bonus = distance_analysis.get("bonus", 0.0)

                # Additional bonus for experience at distance
                experience_count = distance_analysis.get("experience_count", 0)
                if experience_count >= 3:
                    distance_bonus += 0.10  # Proven at distance
                elif experience_count == 0:
                    distance_bonus -= 0.08  # Never raced this distance

                # Stretch out / cut back adjustments
                distance_change = distance_analysis.get("distance_change", "unknown")
                if distance_change == "cut_back" and style in ["E", "E/P"]:
                    distance_bonus += 0.05  # Speed horse cutting back

                tier2_bonus += distance_bonus
        except BaseException:
            pass

        # ======================== SURFACE SWITCH DETECTION (Feb 10, 2026) ========================
        # JWB BUG: Last 2 races on DIRT (6.5f, 5f), today's race TURF 1 mile
        # System gave 69.1% win prob — should have been heavily penalized for
        # Dirt→Turf switch with zero turf experience.
        # ENHANCED (Feb 13, 2026): Now passes pedigree data for mud/turf pedigree integration
        # BRIDGE: Skip surface switch & workout if unified engine already handled
        if not use_unified_engine:
            try:
                if horse_race_history:
                    # Build pedigree data dict for surface switch evaluation
                    _ped = (
                        pedigree_per_horse.get(name, {}) if pedigree_per_horse else {}
                    )
                    _ped_data = {}
                    if _ped:
                        _mud = _ped.get("sire_mud_pct", np.nan)
                        if pd.notna(_mud):
                            _ped_data["sire_mud_pct"] = float(_mud)
                        _off = _ped.get("pedigree_off", np.nan)
                        if pd.notna(_off):
                            _ped_data["pedigree_off"] = float(_off)
                        _turf = _ped.get("pedigree_turf", np.nan)
                        if pd.notna(_turf):
                            _ped_data["pedigree_turf"] = float(_turf)

                    surface_result = detect_surface_switch(
                        horse_race_history, surface_type, _ped_data or None
                    )
                    surface_bonus = surface_result.get("bonus", 0.0)
                    tier2_bonus += surface_bonus
            except BaseException:
                pass

            # ======================== WORKOUT QUALITY SCORING (Feb 10, 2026) ========================
            # JWB BUG: Workout rank 43/43 (dead last) was invisible to system.
            # Now scores actual workout rankings as a quality signal.
            try:
                workout_quality = score_workout_quality(_horse_block)
                workout_bonus = workout_quality.get("quality_bonus", 0.0)
                tier2_bonus += workout_bonus
            except BaseException:
                pass

        # ======================== TRACK PATTERN LEARNING BONUS ========================
        # Apply learned historical patterns at this track/surface/distance.
        # Rewards horses matching characteristics of past winners; penalizes mismatches.
        try:
            if _track_patterns:
                # Extract horse-level attributes for pattern matching
                _h_best_beyer = int(
                    row.get("best_beyer", row.get("Best_Beyer", 0)) or 0
                )
                _h_class_rat = float(row.get("Cclass", row.get("class_rating", 0)) or 0)
                _h_dsl = int(
                    row.get("days_since_last", row.get("Days_Since_Last", 0)) or 0
                )
                _h_pp = float(row.get("prime_power", row.get("Prime_Power", 0)) or 0)
                _h_workout = "Steady"
                try:
                    wq = score_workout_quality(_horse_block)
                    wq_b = wq.get("quality_bonus", 0)
                    if wq_b >= 0.08:
                        _h_workout = "Sharp"
                    elif wq_b <= -0.05:
                        _h_workout = "Sparse"
                except BaseException:
                    pass

                tp_result = apply_track_pattern_bonus(
                    track_patterns=_track_patterns,
                    post_position=int(post) if str(post).isdigit() else 0,
                    running_style=style,
                    best_beyer=_h_best_beyer,
                    class_rating=_h_class_rat,
                    days_since_last=_h_dsl,
                    workout_pattern=_h_workout,
                    prime_power=_h_pp,
                    jockey=str(row.get("jockey", row.get("Jockey", ""))),
                    trainer=str(row.get("trainer", row.get("Trainer", ""))),
                )
                tp_bonus = tp_result.get("bonus", 0.0)
                _track_pattern_detail = tp_result.get(
                    "details", []
                )  # Reserved for future logging
                tier2_bonus += tp_bonus
        except BaseException:
            pass

        # ======================== MARATHON CALIBRATION (12f+) ========================
        if is_marathon:
            # Layoff adjustment (freshening can help at marathons)
            # NOTE: Using default 49 days (optimal marathon freshening range)
            # Could extract actual days from PP text in future enhancement
            tier2_bonus += calculate_layoff_bonus(49, is_marathon)

            # Lightly-raced improver bonus (fresh legs at marathons)
            life_pattern = r"Life:\s+(\d+)\s+"
            life_match = re.search(life_pattern, _horse_block)
            if life_match:
                career_starts = int(life_match.group(1))
                tier2_bonus += calculate_experience_bonus(career_starts, is_marathon)

        # ======================== SPRINT CALIBRATION (≤6.5f) ========================
        elif is_sprint:
            # Post position CRITICAL at turf sprints (inside good, outside death!)
            try:
                post_num = int(post)
                tier2_bonus += calculate_sprint_post_position_bonus(
                    post_num, race_furlongs, surface_type
                )
            except BaseException:
                pass

            # Running style bias (early speed 2.05 impact!)
            tier2_bonus += calculate_sprint_running_style_bonus(style, race_furlongs)

        # 1. Track Bias Impact Value bonus
        if style in impact_values:
            impact_val = impact_values[style]
            if impact_val >= 1.5:
                tier2_bonus += 0.15
            elif impact_val >= 1.2:
                tier2_bonus += 0.10

        # 2. SPI (Sire Performance Index) bonus - BRIDGE: skip if unified engine handled
        if not use_unified_engine:
            if name in spi_values:
                tier2_bonus += calculate_spi_bonus(spi_values[name])

        # 3. Surface Specialty bonus - BRIDGE: skip if unified engine handled
        if not use_unified_engine:
            if name in surface_stats:
                stats = surface_stats[name]
                if surface_type.lower() == "tur" and "turf_pct" in stats:
                    tier2_bonus += calculate_surface_specialty_bonus(
                        stats["turf_pct"], "tur"
                    )
                elif (
                    surface_type.lower() in ["aw", "all-weather", "synthetic"]
                    and "aw_pct" in stats
                ):
                    tier2_bonus += calculate_surface_specialty_bonus(
                        stats["aw_pct"], "aw"
                    )

        # 4. AWD (Distance Mismatch) penalty
        if name in awd_analysis:
            tier2_bonus += calculate_awd_mismatch_penalty(awd_analysis[name])

        # 5. CAREER FUTILITY PENALTY (TUP R7 Feb 2026)
        # #3 Forest Acclamation: 3 wins in 50 starts (6%) finished 4th despite model picking to win
        # Chronic losers rarely break through regardless of pace/class advantages
        career_win_pct = 0.0
        try:
            # Parse career record: "Life: 50 3 - 1 - 7 $53,234 79"
            career_match = re.search(
                r"Life:\s+(\d+)\s+(\d+)\s+-\s+\d+\s+-\s+\d+", _horse_block
            )
            if career_match:
                career_starts = int(career_match.group(1))
                career_wins = int(career_match.group(2))
                if career_starts > 0:
                    career_win_pct = career_wins / career_starts

                    # Apply penalties for chronic losers
                    if career_starts >= 20:  # Only penalize with meaningful sample size
                        if career_win_pct < 0.05:  # Less than 5% win rate
                            tier2_bonus -= 2.0  # Massive penalty
                        elif career_win_pct < 0.10:  # Less than 10% win rate
                            tier2_bonus -= 1.0  # Significant penalty
        except BaseException:
            pass

        # 6. 2ND OFF LAYOFF BONUS (TUP R7 Feb 2026, enhanced TUP R4 tuning)
        # Winner #4 If You Want It: 2nd start after layoff, improved from Speed 62
        # TUP R4: Capital Heat 2nd at 22/1 had 22%/61%/+0.11 ROI — massively undervalued
        # BRISNET shows: "2nd off layoff 65 22% 60%" (22% win rate, 60% ITM)
        is_second_off_layoff = False
        try:
            # Check for "2nd off layoff" stat with high win%
            # Extended regex to optionally capture ROI: "2nd off layoff 65 22% 61% +0.11"
            layoff_match = re.search(
                r"2nd off layoff\s+(\d+)\s+(\d+)%\s+(\d+)%(?:\s+([+-]?\d+\.?\d*))?",
                _horse_block,
            )
            if layoff_match:
                layoff_starts = int(layoff_match.group(1))
                layoff_win_pct = int(layoff_match.group(2)) / 100.0
                layoff_itm_pct = int(layoff_match.group(3)) / 100.0
                layoff_roi = (
                    float(layoff_match.group(4)) if layoff_match.group(4) else None
                )

                if layoff_starts >= 3:  # Meaningful sample size
                    # --- Base bonus by win% tier ---
                    if layoff_win_pct >= 0.20:  # 20%+ win rate
                        tier2_bonus += 1.2  # Strong bonus (was 0.8)
                        is_second_off_layoff = True
                    elif layoff_win_pct >= 0.15:  # 15-19% win rate
                        tier2_bonus += 0.7  # Moderate bonus (was 0.5)
                        is_second_off_layoff = True
                    elif layoff_itm_pct >= 0.50:  # 50%+ ITM even if lower win%
                        tier2_bonus += 0.5  # Small bonus (was 0.3)
                        is_second_off_layoff = True

                    # --- ROI-positive multiplier (extremely rare & valuable) ---
                    if layoff_roi is not None and layoff_roi > 0.0:
                        tier2_bonus += 0.4  # Positive ROI is a strong signal
                        is_second_off_layoff = True

                    # --- High ITM kicker (60%+) ---
                    if layoff_itm_pct >= 0.60 and is_second_off_layoff:
                        tier2_bonus += 0.3  # Elite ITM rate kicker
        except BaseException:
            pass

        # 7. BEST SPEED AT DISTANCE BONUS (TUP R7 Feb 2026)
        # Winner #4 If You Want It: Tied #1 with 80 Best Speed at Distance
        # 2nd place #9 English Danger: Tied #1 with 80 Best Speed at Distance
        # This metric predicted 1-2 finish but model ignored it
        best_speed_at_dist = 0
        try:
            # Parse "Best Speed at Dist" from PP text header
            # Format: "# Best Speed at Dist\n4 If You Want It 80"
            best_speed_pattern = rf"{re.escape(name)}\s+(\d+)"
            # Look in "Best Speed at Dist" section
            best_speed_section = re.search(
                r"# Best Speed at Dist(.{0,500})", pp_text, re.DOTALL
            )  # NOTE: section header is race-level, search per-horse by name below
            if best_speed_section:
                section_text = best_speed_section.group(1)
                horse_best = re.search(best_speed_pattern, section_text)
                if horse_best:
                    best_speed_at_dist = int(horse_best.group(1))

                    # Find all horses' best speeds to determine ranking
                    all_best_speeds = re.findall(
                        r"\d+\s+[A-Za-z\s\']+\s+(\d+)", section_text
                    )
                    if all_best_speeds:
                        speeds = [int(s) for s in all_best_speeds]
                        speeds_sorted = sorted(speeds, reverse=True)

                        if best_speed_at_dist > 0:
                            rank = (
                                speeds_sorted.index(best_speed_at_dist) + 1
                                if best_speed_at_dist in speeds_sorted
                                else 999
                            )

                            # Bonus for top 3 best speeds at distance
                            if rank == 1:
                                tier2_bonus += 0.5  # Best speed at distance
                            elif rank == 2:
                                tier2_bonus += 0.3  # 2nd best
                            elif rank == 3:
                                tier2_bonus += 0.2  # 3rd best
        except BaseException:
            pass

        # 8. HOT TRAINER BONUS (TUP R6 + R7 Feb 2026 - Winner had 22% trainer, 4-0-0 L14, 2nd time Lasix 33%)
        # Extract trainer win % from PP text
        trainer_win_pct = 0.0
        trainer_starts = 0
        is_hot_l14 = False
        is_2nd_lasix_high_pct = False
        try:
            # Parse trainer win %: "Trnr: Eikleberry Kevin (50 11-7-4 22%)"
            trainer_match = re.search(
                r"Trnr:.*?\((\d+)\s+\d+-\d+-\d+\s+(\d+)%\)", _horse_block
            )
            if trainer_match:
                trainer_starts = int(trainer_match.group(1))
                trainer_win_pct = int(trainer_match.group(2)) / 100.0

            # Check for hot trainer in last 14 days (look for patterns like "9 4-0-0")
            hot_l14_match = re.search(
                r"Hot Trainer in last 14 days \((\d+) (\d+)-\d+-\d+\)", _horse_block
            )
            if hot_l14_match and int(hot_l14_match.group(2)) >= 4:
                is_hot_l14 = True

            # Check for 2nd time Lasix with high % angle
            if "2nd time Lasix" in _horse_block and trainer_win_pct >= 0.25:
                is_2nd_lasix_high_pct = True

            tier2_bonus += calculate_hot_trainer_bonus(
                trainer_win_pct, is_hot_l14, is_2nd_lasix_high_pct, trainer_starts
            )
        except BaseException:
            pass

        # 9. RACE AUDIT ENHANCEMENT 3: 1st-After-Claim Trainer Boost
        # Oaklawn R1: Trainer Ashford had 28% win rate on 1st-after-claim but got zero credit
        with contextlib.suppress(BaseException):
            tier2_bonus += calculate_first_after_claim_bonus(_horse_block)

        # 10. RACE AUDIT ENHANCEMENT 4: Style vs Weekly Bias Dynamic Penalty/Bonus
        # Oaklawn R1: Stalker impact 0.32 but model ranked S-type Tiffany Twist #1
        # E-type Tell Me When (impact 2.05) was ranked dead last
        with contextlib.suppress(BaseException):
            tier2_bonus += calculate_style_vs_weekly_bias_bonus(style, impact_values)

        # 11. RACE AUDIT ENHANCEMENT 5: Post Position Bias Multiplier
        # Oaklawn R1: Rail posts 1-3 had 2.59 weekly impact, Sombra Dorada (post 1)
        # finished 2nd but model had her 5th
        with contextlib.suppress(BaseException):
            tier2_bonus += calculate_post_position_bias_bonus(post, weekly_post_impacts)

        # 12. RACE AUDIT ENHANCEMENT 6: E1/E2 Pace Supremacy Bonus
        # Oaklawn R9: She's Storming had best E1 (91) matching par (91) but got no credit.
        # At speed-biased tracks, the horse with the fastest E1 in the field has a tactical
        # advantage. Combined with E/P style + inside post, this is a MAJOR signal.
        with contextlib.suppress(BaseException):
            tier2_bonus += calculate_pace_supremacy_bonus(
                name, _horse_block, _field_e1_values, impact_values
            )

        # 13. TRACK BIAS STATS INTEGRATION (Feb 20, 2026)
        # Uses %Wire, Speed Bias%, WnrAvgBL, and %Races Won from parsed Track Bias Stats.
        # These fields are NOT captured by Impact Values alone.
        try:
            if track_bias_stats and style:
                style_upper = str(style).upper().strip()

                # --- %Wire bonus: When wire-to-wire win rate is high (>50%), E horses dominate ---
                pct_wire = track_bias_stats.get("pct_wire", 0)
                if pct_wire >= 70 and style_upper == "E":
                    tier2_bonus += 0.5  # Very high wire rate = E horses cruise
                elif pct_wire >= 50 and style_upper == "E":
                    tier2_bonus += 0.3  # High wire rate
                elif pct_wire >= 70 and style_upper == "S":
                    tier2_bonus -= 0.4  # High wire crushes closers

                # --- Speed Bias % confirmation: When >80% races won by E/EP, amplify E styles ---
                speed_bias = track_bias_stats.get("speed_bias_pct", 50)
                if speed_bias >= 85:
                    if style_upper in ("E", "E/P"):
                        tier2_bonus += 0.3  # Extreme speed bias track
                    elif style_upper == "S":
                        tier2_bonus -= 0.3  # Closers nearly shut out
                elif speed_bias <= 30:
                    if style_upper in ("P", "S"):
                        tier2_bonus += 0.3  # Pace-collapsing track favors closers
                    elif style_upper == "E":
                        tier2_bonus -= 0.3  # Speed types can't sustain

                # --- WnrAvgBL: Pace contestation indicator ---
                # Low WnrAvgBL at 1st call = winners ARE on the lead (speed favored)
                # High WnrAvgBL = winners rally from behind (closers favored)
                wnr_bl_1st = track_bias_stats.get("wnr_avg_bl_1st", 0)
                if wnr_bl_1st > 0:
                    if wnr_bl_1st <= 1.0 and style_upper in ("E", "E/P"):
                        tier2_bonus += 0.2  # Winners are close to lead at 1st call
                    elif wnr_bl_1st >= 4.0 and style_upper in ("P", "S"):
                        tier2_bonus += 0.3  # Winners rally from way back
                    elif wnr_bl_1st >= 4.0 and style_upper == "E":
                        tier2_bonus -= 0.2  # Speed horses collapse here

                # --- %Races Won validation: Cross-check with Impact Values ---
                pct_won = track_bias_stats.get("pct_races_won", {})
                if pct_won:
                    style_pct = pct_won.get(style_upper, 0)
                    if style_upper == "E/P":
                        style_pct = max(pct_won.get("E/P", 0), pct_won.get("E", 0) / 2)
                    if style_pct >= 50:
                        tier2_bonus += 0.3  # This runstyle wins majority of races
                    elif style_pct <= 5 and style_pct > 0:
                        tier2_bonus -= 0.2  # This runstyle almost never wins
        except BaseException:
            pass

        # 14. RACE SUMMARY RANKING INTEGRATION (Feb 20, 2026)
        # Uses parsed Race Summary ranking tables to identify top-ranked horses
        # across multiple Brisnet metrics (Speed Last Race, Current Class, etc.)
        try:
            if race_summary_rankings and name:
                _summary_bonus = 0.0
                _top_count = 0  # Count how many categories this horse leads

                for _cat_name, _cat_data in race_summary_rankings.items():
                    if not _cat_data:
                        continue
                    horse_val = _cat_data.get(name)
                    if horse_val is None:
                        continue

                    # Determine rank position (1st, 2nd, 3rd...)
                    sorted_vals = sorted(_cat_data.values(), reverse=True)
                    try:
                        _rank = sorted_vals.index(horse_val) + 1
                    except ValueError:
                        continue

                    # Graduated bonus by rank position
                    if _rank == 1:
                        _summary_bonus += 0.25
                        _top_count += 1
                    elif _rank == 2:
                        _summary_bonus += 0.15
                    elif _rank == 3:
                        _summary_bonus += 0.08
                    # Bottom 2 in any category = penalty
                    elif _rank >= len(sorted_vals) - 1:
                        _summary_bonus -= 0.10

                # Multi-category leader bonus: horse that leads 3+ categories
                # is almost certainly the class of the field
                if _top_count >= 4:
                    _summary_bonus += 0.5  # Dominant across metrics
                elif _top_count >= 3:
                    _summary_bonus += 0.3  # Multi-category leader
                elif _top_count >= 2:
                    _summary_bonus += 0.15  # Double leader

                tier2_bonus += np.clip(_summary_bonus, -1.0, 2.0)
        except BaseException:
            pass

        # 15. QUICKPLAY COMMENTS INTEGRATION (Feb 20, 2026)
        # Parse Brisnet's pre-analyzed positive (star) and negative (bullet) signals
        # per horse. These are high-value curated handicapping angles.
        try:
            if _horse_block:
                _qp_comments = parse_quickplay_comments(_horse_block)
                if _qp_comments.get("positive") or _qp_comments.get("negative"):
                    tier2_bonus += score_quickplay_comments(_qp_comments)
        except BaseException:
            pass

        # 16. BRIS RR/CR INTEGRATION (Feb 20, 2026)
        # Parse BRIS Race Rating and Class Rating from past race lines.
        # Use RR trend to detect improving/declining form, and CR to validate class.
        try:
            if _horse_block:
                _rr_cr_data = parse_bris_rr_cr_per_race(_horse_block)
                if len(_rr_cr_data) >= 2:
                    # Check RR trend (improving = recent > older)
                    recent_rr = _rr_cr_data[0].get("rr", 0)
                    older_rr = sum(r.get("rr", 0) for r in _rr_cr_data[1:]) / len(
                        _rr_cr_data[1:]
                    )
                    if recent_rr > 0 and older_rr > 0:
                        rr_trend = (recent_rr - older_rr) / max(older_rr, 1)
                        if rr_trend > 0.05:  # Improving by 5%+
                            tier2_bonus += min(rr_trend * 2, 0.5)  # Max +0.5
                        elif rr_trend < -0.05:  # Declining by 5%+
                            tier2_bonus += max(rr_trend * 1.5, -0.3)  # Max -0.3

                    # CR validation: compare most recent CR to race avg CR
                    recent_cr = _rr_cr_data[0].get("cr", 0)
                    avg_cr = sum(r.get("cr", 0) for r in _rr_cr_data) / len(_rr_cr_data)
                    if recent_cr > avg_cr * 1.1:
                        tier2_bonus += 0.15  # Stepping up in class AND performing
        except BaseException:
            pass

        # ======================== End Tier 2 Bonuses ========================

        # ════════════════════════════════════════════════════════════════
        # FIX E: SPEED QUALITY FLOOR GATE (TUP R6 Feb 19, 2026)
        # Top Review had worst speed in field (C-Speed = -0.50) but
        # ranked #1 through +3.88 in stacked bonuses. If C-Speed is
        # bottom-2 in field, bonuses shouldn't fully compensate.
        # Apply 0.75 multiplier to tier2 to dampen bonus inflation.
        # ════════════════════════════════════════════════════════════════
        if tier2_bonus > 0 and df_styles is not None and not df_styles.empty:
            try:
                _all_speeds = []
                for _, _sr in df_styles.iterrows():
                    _sv = float(_sr.get("Cspeed", 0.0))
                    _all_speeds.append(_sv)
                if len(_all_speeds) >= 4:
                    _all_speeds_sorted = sorted(_all_speeds)
                    # Bottom 2 in field
                    if cspeed <= _all_speeds_sorted[1]:
                        tier2_bonus *= 0.75
            except BaseException:
                pass

        # ════════════════════════════════════════════════════════════════
        # FIX D: TRAINER PENALTY REDUCED FOR CLASS DROPPERS (TUP R6 Feb 19, 2026)
        # Stormylux: Kemper 3% got -0.90 penalty, but horse won at OC20k
        # level and dropped to C4500. Talent overcomes poor trainer when
        # slumming. If horse drops 2+ class levels, halve trainer penalty.
        # ════════════════════════════════════════════════════════════════
        if tier2_bonus < 0:
            try:
                # Detect major class drop from PP data
                _class_levels = {
                    "g1": 11,
                    "g2": 10,
                    "g3": 9,
                    "stk": 8,
                    "aoc": 7,
                    "oc": 7,
                    "alw": 6,
                    "clm": 4,
                    "msw": 3,
                    "mc": 2,
                    "mdn": 1,
                }
                _past_class = 0
                _today_class = 0
                # Today's class from race_type parameter
                _rt_lower = (race_type or "").lower().replace(" ", "")
                for _ck, _cv in _class_levels.items():
                    if _ck in _rt_lower:
                        _today_class = _cv
                        break
                # Past class from last race in PP block
                _past_race_types = re.findall(
                    r"(?:OC|Oc|oc)(\d+)k|(?:Alw|ALW|alw)|(?:Stk|STK|stk)",
                    _horse_block[:500] if _horse_block else "",
                )
                if _past_race_types:
                    # If we found OC in past races, that's class 7+
                    _past_class = 7
                elif re.search(r"™C(\d+)", _horse_block[:300] if _horse_block else ""):
                    _past_claim_match = re.search(
                        r"™C(\d+)", _horse_block[:300] if _horse_block else ""
                    )
                    if _past_claim_match:
                        _past_claim_val = int(_past_claim_match.group(1))
                        _today_claim_match = re.search(r"(\d+)", _rt_lower)
                        _today_claim_val = (
                            int(_today_claim_match.group(1))
                            if _today_claim_match
                            else 0
                        )
                        if (
                            _today_claim_val > 0
                            and _past_claim_val >= _today_claim_val * 2
                        ):
                            _past_class = 6  # Major claiming drop

                if (_past_class >= _today_class + 2) or (
                    _past_class >= 7 and _today_class <= 4
                ):
                    # Major class drop: halve the negative tier2 penalty
                    tier2_bonus *= 0.5
            except BaseException:
                pass

        # CAP tier2_bonus: Bonuses should supplement, not dominate, core ratings
        # FIX C: RACE-TYPE-AWARE TIER2 CAP (TUP R6 Feb 19, 2026)
        # A $4,500 claimer shouldn't generate +3.88 in bonuses.
        # Claiming races: [-3.0, 3.0]. Allowance: [-3.5, 4.0]. Stakes: [-4.0, 5.0].
        _rt_for_cap = (race_type or "").lower()
        if "claim" in _rt_for_cap or "clm" in _rt_for_cap:
            tier2_bonus = np.clip(tier2_bonus, -3.0, 3.0)
        elif "allow" in _rt_for_cap or "aoc" in _rt_for_cap or "oc" in _rt_for_cap:
            tier2_bonus = np.clip(tier2_bonus, -3.5, 4.0)
        else:
            tier2_bonus = np.clip(tier2_bonus, -4.0, 5.0)

        # ═══════════════════════════════════════════════════════════════
        # BRIDGE MARKER: When unified engine is active, compute bridge R
        # by wiring tier2 bonuses directly onto engine's pre-computed R.
        # The HYBRID MODEL section below still runs (harmless) but its
        # result is overridden after it completes.
        # ═══════════════════════════════════════════════════════════════
        _bridge_R = None
        if use_unified_engine:
            _bridge_R = _pre_bridge_R + tier2_bonus

        # HYBRID MODEL: Surface-Adaptive + Maiden-Aware PP Weight (SA R8 + GP R1 + GP R2 - Feb 2026)
        #
        # THREE-RACE VALIDATION:
        #
        # SA R8 (6F Dirt Sprint - Experienced Field):
        #   Top 3 PP horses = top 3 finishers (PP correlation -0.831)
        #   Result: 92% PP weight = 100% accuracy ✓
        #
        # GP R1 (1M Turf Route):
        #   PP 138.3, 137.4, 136.0 DNF top 5 - NO PP correlation!
        #   Winner #8 (PP 130.6) beat #4 (PP 138.3) by 8 points
        #   Result: 0% PP weight (components only) = Optimal ✓
        #
        # GP R2 (6F Dirt Sprint - MAIDEN with 6 first-timers):
        #   Highest PP #5 (126.5) finished 5th
        #   Winner #2 (PP 125.4, only 1.1 pts behind) had pace advantage
        #   Top pick #6 (first-timer, NO PP) placed 2nd ✓
        #   3 of top 4 finishers = first-time starters (no PP data)
        #   Result: 92% PP weight = 33% accuracy (Mixed results) ⚠️
        #
        # KEY INSIGHTS:
        # 1. Prime Power predicts RAW SPEED on DIRT when horses have racing history
        # 2. Turf: tactics/position >> speed (DISABLE PP, use components)
        # 3. Maiden races: Many first-timers lack PP data, components predict debut quality
        # 4. Small PP differences (1-2 points) NOT decisive in maiden races
        # 5. Race Quality (Stakes/Allowance/Claiming): Higher quality = more reliable PP
        # 6. Purse Amount: Higher purse = better horses = more consistent performance
        #
        # DYNAMIC WEIGHTING SYSTEM (Race Quality + Surface + Experience):
        #
        # RACE QUALITY TIERS (by purse and type):
        #   ELITE (Stakes G1-G3, $200k+):      PP reliability: HIGHEST (top horses, consistent)
        #   HIGH (Listed Stakes, Allowance):   PP reliability: HIGH (quality horses)
        #   MID (AOC, Starter):                PP reliability: MODERATE (mixed quality)
        #   LOW (Claiming, Maiden Claiming):   PP reliability: VARIABLE (cheaper horses)
        #
        # DIRT (Adjusted by quality):
        #   Elite Stakes: 95/5 (top horses, PP is king)
        #   Allowance: 90/10 (quality horses, PP very reliable)
        #   Claiming: 85/15 (cheaper horses, more variance)
        #   Maiden: 50/50 to 70/30 (experience-dependent)
        #
        # TURF: 0% PP / 100% Components (all quality levels - validated)
        # SYNTHETIC: 75% PP / 25% Components (all quality levels)
        #
        # This creates an adaptive intelligence that learns race-to-race patterns

        # ═══════════════════════════════════════════════════════════════════════
        # CALCULATE WEIGHTED COMPONENTS (must happen BEFORE Prime Power check)
        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL FIX: weighted_components must be calculated before the
        # if prime_power_raw > 0 block, otherwise the else branch will fail
        # with UnboundLocalError when trying to use it

        # ═══════════════════════════════════════════════════════════════════════
        # CRITICAL: Use race_class_parser weight OR fallback to legacy weights
        # ═══════════════════════════════════════════════════════════════════════
        # The race_class_parser provides INDUSTRY-STANDARD hierarchy weights (1-10 scale)
        # that properly handle ALL race types (G1=10.0, Handicap=7.0, Claiming=2.0, etc.)
        #
        # Legacy system used hardcoded 2.0-3.0 weights which severely under-weighted
        # elite races, causing G1 races to use 3.0 instead of proper 10.0 weight

        # Default component weights (used if parser unavailable)
        speed_multiplier = 1.8
        class_weight = 3.0  # Legacy default
        form_weight = 1.8

        # RACE CLASS PARSER: Get industry-standard weight from Section A
        # The comprehensive_class_rating already includes parser weight, BUT
        # we need the MULTIPLIER to scale c_class in final rating calculation
        race_quality = "mid"  # Default for legacy path

        # Check if we have race_class_parser data from PP text
        parser_class_weight = None
        if RACE_CLASS_PARSER_AVAILABLE and pp_text:
            try:
                race_class_data = parse_and_calculate_class(pp_text)
                parser_class_weight = race_class_data["weight"]["class_weight"]
                # Get quality and normalize to lowercase + map Medium -> mid
                quality_raw = race_class_data["summary"][
                    "quality"
                ]  # 'Elite'/'High'/'Medium'/'Low'
                quality_map = {
                    "Elite": "elite",
                    "High": "high",
                    "Medium": "mid",
                    "Low": "low",
                    "Minimal": "low",
                }
                race_quality = quality_map.get(quality_raw, "mid")
                # Parser status logged silently (removed green banner from Section B)
                logger.info(
                    f"Parser: {race_class_data['summary']['class_type']} | Level {race_class_data['hierarchy']['final_level']} | Weight {parser_class_weight:.2f} | Quality: {race_quality}"
                )
            except Exception as e:
                logger.warning(f"Parser failed in compute_bias_ratings: {e}")
                pass

        # If parser unavailable, fall back to legacy race quality detection
        if parser_class_weight is None:
            try:
                race_metadata = extract_race_metadata_from_pp_text(pp_text)
                race_type_clean = race_metadata.get("race_type_clean", "")
                purse_amount = race_metadata.get("purse_amount", 0)

                # Map to quality tier
                if race_type_clean == "stakes_graded" or (
                    race_type_clean == "stakes" and purse_amount >= 200000
                ):
                    race_quality = "elite"
                elif race_type_clean in ["allowance", "allowance_optional"]:
                    race_quality = "high"
                elif race_type_clean == "claiming":
                    race_quality = "low"
                elif race_type_clean == "maiden_claiming":
                    race_quality = "low-maiden"
                elif purse_amount >= 500000:
                    race_quality = "elite"
            except BaseException:
                pass

        # Set component weights based on race quality
        if parser_class_weight is not None:
            # USE RACE_CLASS_PARSER WEIGHT (industry-standard 1-10 scale)
            # This properly handles G1 (10.0), Handicap (7.0), Claiming (2.0), etc.
            class_weight = parser_class_weight

            # Adjust speed/form multipliers based on quality tier
            if race_quality == "elite":
                speed_multiplier = 1.8  # Elite races: skill matters more than speed
                form_weight = 2.2  # Form very important in stakes
            elif race_quality == "high":
                speed_multiplier = 2.0
                form_weight = 2.0
            elif race_quality == "low":
                speed_multiplier = 2.5  # Speed figures matter more in claiming
                form_weight = 1.8
            else:  # mid
                speed_multiplier = 2.2
                form_weight = 2.0
        else:
            # LEGACY FALLBACK (if parser unavailable)
            if race_quality == "low" or race_quality == "low-maiden":
                speed_multiplier = 2.5
                class_weight = 2.0
                form_weight = 1.8
            elif (
                race_quality == "mid"
                or race_quality == "mid-maiden"
                or race_quality == "high"
            ):
                speed_multiplier = 2.2
                class_weight = 2.5
                form_weight = 2.0
            else:  # elite/stakes
                speed_multiplier = 1.8
                class_weight = (
                    3.0  # Still wrong for G1, but best we can do without parser
                )
                form_weight = 1.8

        # Apply pace component with claiming race cap
        pace_contribution = cpace * 1.5
        if race_quality == "low" or race_quality == "low-maiden":
            if pace_contribution > 0.75:
                pace_contribution = 0.75

        # ═══════════════════════════════════════════════════════════════════════
        # DYNAMIC WEIGHTING: Apply user's race parameter-based weights
        # These weights are calculated from: surface, distance, condition, race type, purse
        # ═══════════════════════════════════════════════════════════════════════
        dw = dynamic_weights or {}
        class_form_mult = dw.get("class_form", 1.0)
        pace_speed_mult = dw.get("pace_speed", 1.0)
        style_post_mult = dw.get("style_post", 1.0)
        track_bias_mult = dw.get("track_bias", 1.0)
        # RACE AUDIT ENHANCEMENT 1: Amplify track_bias_mult when extreme weekly biases detected
        track_bias_mult *= weekly_bias_amplifier

        # Calculate weighted components with dynamic multipliers applied
        weighted_components = (
            c_class * class_weight * class_form_mult
            + c_form * form_weight * class_form_mult
            + cspeed * speed_multiplier * pace_speed_mult
            + pace_contribution * pace_speed_mult
            + cstyle * 1.2 * style_post_mult
            + cpost * 0.8 * style_post_mult
        )

        # Now check if Prime Power is available
        prime_power_raw = safe_float(row.get("Prime Power", 0.0), 0.0)
        if prime_power_raw > 0:  # noqa: E712  # race_class_shown guard below
            # Normalize Prime Power (typical range: 110-130, clip outliers to 0-2 scale)
            pp_normalized = np.clip(
                (prime_power_raw - 110) / 20, 0, 2
            )  # 0 to 2 scale (allows up to 150)
            pp_contribution = pp_normalized * 10  # Scale to match component range

            # Determine optimal PP weight based on surface and distance
            # Parse distance for route vs sprint classification
            distance_furlongs = 8.0  # Default
            try:
                if "f" in distance_txt.lower():
                    distance_furlongs = float(
                        distance_txt.lower().replace("f", "").strip()
                    )
                elif "mile" in distance_txt.lower():
                    if "1mile" in distance_txt.lower().replace(" ", ""):
                        distance_furlongs = 8.0
                    elif "1.5" in distance_txt or "1 1/2" in distance_txt:
                        distance_furlongs = 12.0
                    elif "1.25" in distance_txt or "1 1/4" in distance_txt:
                        distance_furlongs = 10.0
                    elif "1.125" in distance_txt or "1 1/8" in distance_txt:
                        distance_furlongs = 9.0
            except BaseException:
                pass

            # ═══════════════════════════════════════════════════════════════════════
            # UNIVERSAL RACE QUALITY DETECTION (All Tracks, All Types, All Purses)
            # ═══════════════════════════════════════════════════════════════════════

            # STEP 1: Extract race metadata from PP text (PREFERRED - most accurate)
            race_metadata = extract_race_metadata_from_pp_text(pp_text)

            # STEP 2: Try legacy race_type extraction if PP text detection failed
            purse_amount = race_metadata["purse_amount"]
            race_type_clean = race_metadata["race_type_clean"]
            detection_confidence = race_metadata["confidence"]

            if purse_amount == 0 and race_type:
                # Fallback: Try to infer from race_type parameter
                inferred_purse = infer_purse_from_race_type(race_type)
                if inferred_purse and inferred_purse > 0:
                    purse_amount = inferred_purse
                    detection_confidence = 0.7

            if race_type_clean == "unknown" and race_type:
                # Fallback: Parse race_type parameter
                race_type_lower = str(race_type).lower()
                if (
                    "g1" in race_type_lower
                    or "g2" in race_type_lower
                    or "g3" in race_type_lower
                ):
                    race_type_clean = "stakes_graded"
                elif "stake" in race_type_lower or "stk" in race_type_lower:
                    race_type_clean = "stakes"
                elif "allowance" in race_type_lower or "alw" in race_type_lower:
                    race_type_clean = "allowance"
                elif "aoc" in race_type_lower or "optional" in race_type_lower:
                    race_type_clean = "allowance_optional"
                elif "maiden claiming" in race_type_lower or "mcl" in race_type_lower:
                    race_type_clean = "maiden_claiming"
                elif "claiming" in race_type_lower or "clm" in race_type_lower:
                    race_type_clean = "claiming"
                elif "maiden" in race_type_lower or "msw" in race_type_lower:
                    race_type_clean = "maiden_special_weight"
                detection_confidence = 0.6

            # ═══════════════════════════════════════════════════════════════════════
            # ENHANCED RACE QUALITY DETECTION (When Prime Power IS available)
            # ═══════════════════════════════════════════════════════════════════════
            # When PP is present, we can refine the race quality detection with purse details
            # (Uses race_metadata already extracted above — no re-extraction needed)

            # STEP 2: Override/refine race_quality if we have better purse/type info
            _pre_refine_quality = race_quality
            if purse_amount >= 500000:
                race_quality = "elite"
            elif purse_amount >= 150000 and race_quality not in ["elite", "high"]:
                race_quality = "high"
            elif race_type_clean == "stakes_graded" and race_quality != "elite":
                race_quality = "elite"

            # If quality was upgraded, recalculate component weights to stay consistent
            if race_quality != _pre_refine_quality:
                if parser_class_weight is not None:
                    if race_quality == "elite":
                        speed_multiplier = 1.8
                        form_weight = 2.2
                    elif race_quality == "high":
                        speed_multiplier = 2.0
                        form_weight = 2.0
                    elif race_quality == "low":
                        speed_multiplier = 2.5
                        form_weight = 1.8
                    else:
                        speed_multiplier = 2.2
                        form_weight = 2.0
                else:
                    if race_quality in ("low", "low-maiden"):
                        speed_multiplier = 2.5
                        class_weight = 2.0
                        form_weight = 1.8
                    elif race_quality in ("mid", "mid-maiden", "high"):
                        speed_multiplier = 2.2
                        class_weight = 2.5
                        form_weight = 2.0
                    else:
                        speed_multiplier = 1.8
                        class_weight = 3.0
                        form_weight = 1.8

            # STEP 3: Display detected metadata for user validation (ONCE per race, not per horse)
            if not _race_class_shown:
                _race_class_shown = True
                with st.expander(
                    "🔍 Race Classification & Detection Details", expanded=False
                ):
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric(
                            "Detected Purse",
                            f"${purse_amount:,}" if purse_amount > 0 else "Unknown",
                        )
                    with col2:
                        st.metric(
                            "Race Type", race_type_clean.replace("_", " ").title()
                        )
                    with col3:
                        st.metric("Quality Tier", race_quality.upper())
                    with col4:
                        confidence_color = (
                            "🟢"
                            if detection_confidence >= 0.8
                            else "🟡"
                            if detection_confidence >= 0.5
                            else "🔴"
                        )
                        st.metric(
                            "Confidence",
                            f"{confidence_color} {detection_confidence:.0%}",
                        )

                    st.caption(
                        f"**Detection Method:** {race_metadata['detection_method'].replace('_', ' ').title()}"
                    )
                    st.caption(
                        f"**Raw Race Type:** {race_metadata['race_type'] or race_type or 'Not detected'}"
                    )

                    if detection_confidence < 0.5:
                        st.warning(
                            "⚠️ **Low confidence detection** - Verify race type is correctly classified"
                        )
                    if purse_amount == 0:
                        st.info(
                            "ℹ️ Purse not detected - Using race type classification only"
                        )

            # Apply pace component with claiming race cap (TUP R7)
            pace_contribution = cpace * 1.5

            # Cap pace bonus in claiming races to prevent over-reliance
            if race_quality == "low" or race_quality == "low-maiden":
                if pace_contribution > 0.75:
                    pace_contribution = 0.75  # Cap at +0.75 in claiming
                    # pace_cap_applied flag removed (was unused — F841)

            # Apply dynamic weights (same as non-PP path)
            dw = dynamic_weights or {}
            class_form_mult = dw.get("class_form", 1.0)
            pace_speed_mult = dw.get("pace_speed", 1.0)
            style_post_mult = dw.get("style_post", 1.0)
            track_bias_mult = dw.get("track_bias", 1.0)
            # RACE AUDIT ENHANCEMENT 1: Amplify track_bias_mult when extreme weekly biases detected
            track_bias_mult *= weekly_bias_amplifier

            weighted_components = (
                c_class * class_weight * class_form_mult
                + c_form * form_weight * class_form_mult
                + cspeed
                * speed_multiplier
                * pace_speed_mult  # Boost speed in claiming/allowance
                + pace_contribution * pace_speed_mult  # Capped in claiming races
                + cstyle * 1.2 * style_post_mult
                + cpost * 0.8 * style_post_mult
            )

            # ═══════════════════════════════════════════════════════════════════════
            # Surface-adaptive ratio selection (with quality adjustment)
            # ═══════════════════════════════════════════════════════════════════════
            surface_lower = (surface_type or "dirt").lower()

            if (
                "turf" in surface_lower
                or "tur" in surface_lower
                or "grass" in surface_lower
            ):
                # Turf racing: PP has no predictive value (GP R1: highest PP horses lost)
                # Use pure component model (pace, form, style, jockey are what matter)
                pp_weight, comp_weight = 0.0, 1.0
            elif (
                "synth" in surface_lower
                or "aw" in surface_lower
                or "all-weather" in surface_lower
                or "tapeta" in surface_lower
            ):
                # Synthetic: Consistent surface, speed matters but not as much as dirt
                pp_weight, comp_weight = 0.75, 0.25
            else:  # Dirt (default)
                # MAIDEN RACE DETECTION: Adjust PP weight based on field experience
                # GP R2 Learning: Maiden races with many first-timers need balanced weighting
                # because PP data is sparse and components predict debut quality better

                is_maiden = False
                if race_type:
                    race_type_lower = str(race_type).lower()
                    is_maiden = any(
                        keyword in race_type_lower
                        for keyword in [
                            "maiden",
                            "mdn",
                            "msw",
                            "mcl",
                            "mdn sp wt",
                            "maiden sp wt",
                        ]
                    )

                if is_maiden:
                    # Count horses with valid PP data in this race
                    # For maiden races, check if field is mostly first-timers or experienced
                    horses_with_pp = 0
                    total_horses = len(df_styles) if df_styles is not None else 0

                    if df_styles is not None and not df_styles.empty:
                        for _, h_row in df_styles.iterrows():
                            h_pp = safe_float(h_row.get("Prime Power", 0.0), 0.0)
                            if h_pp > 0:
                                horses_with_pp += 1

                    horses_without_pp = total_horses - horses_with_pp

                    # Adjust PP weight based on field composition
                    if horses_without_pp >= horses_with_pp and horses_without_pp > 0:
                        # Majority are first-timers: Equal weight to PP and components
                        # GP R2: 6 first-timers, winner had PP + pace, top pick (first-timer) placed 2nd
                        if distance_furlongs <= 7.0:
                            pp_weight, comp_weight = (
                                0.50,
                                0.50,
                            )  # Sprint: Balanced approach
                        else:
                            pp_weight, comp_weight = (
                                0.40,
                                0.60,
                            )  # Route: Components slightly favored
                    else:
                        # Majority have racing experience: Favor PP but less than non-maiden
                        # Small PP differences not as decisive in maiden races
                        if distance_furlongs <= 7.0:
                            pp_weight, comp_weight = (
                                0.70,
                                0.30,
                            )  # Sprint: Still favor PP
                        else:
                            pp_weight, comp_weight = 0.60, 0.40  # Route: More balanced
                else:
                    # Non-maiden race: Apply RACE QUALITY + CLASS DROPPER analysis
                    # Higher quality races = more reliable PP (better horses, consistent performance)

                    # Check for class dropper scenario (SA R6)
                    class_spread = 0.0
                    if (
                        df_styles is not None
                        and not df_styles.empty
                        and "Class Rating" in df_styles.columns
                    ):
                        class_ratings = []
                        for _, h_row in df_styles.iterrows():
                            cr = safe_float(h_row.get("Class Rating", 0.0), 0.0)
                            if cr > 0:
                                class_ratings.append(cr)

                        if len(class_ratings) >= 2:
                            class_spread = max(class_ratings) - min(class_ratings)

                    # ═══════════════════════════════════════════════════════════════
                    # DYNAMIC WEIGHTING: Quality + Distance + Class Dropper
                    # ═══════════════════════════════════════════════════════════════

                    if distance_furlongs <= 7.0:  # Sprint
                        # Base weights by race quality
                        if race_quality == "elite":
                            # Elite Stakes: Top horses, PP extremely reliable
                            base_pp, base_comp = 0.95, 0.05
                        elif race_quality == "high":
                            # Allowance: Quality horses, PP very reliable
                            base_pp, base_comp = 0.90, 0.10
                        elif race_quality == "mid":
                            # Mid-tier: Standard reliability
                            base_pp, base_comp = 0.88, 0.12
                        else:  # "low" (Claiming)
                            # CLAIMING SPRINT CALIBRATION (Feb 2026)
                            # Previous 62/38 split allowed biases to override PP too much
                            # Claiming races have chaos, but PP still primary predictor
                            # Components (pace, tactics, smart money) provide important nuance
                            base_pp, base_comp = (
                                0.75,
                                0.25,
                            )  # PP-dominant, components as tiebreakers

                        # Class dropper adjustment
                        if class_spread > 1.5:
                            # Significant class advantage: Shift toward components
                            pp_weight = base_pp - 0.07  # Reduce PP weight
                            comp_weight = base_comp + 0.07  # Increase component weight
                        else:
                            pp_weight, comp_weight = base_pp, base_comp

                    else:  # Route
                        # Base weights by race quality (routes need more stamina/pace analysis)
                        if race_quality == "elite":
                            base_pp, base_comp = 0.88, 0.12
                        elif race_quality == "high":
                            base_pp, base_comp = 0.82, 0.18
                        elif race_quality == "mid":
                            base_pp, base_comp = 0.78, 0.22
                        else:  # "low" (Claiming)
                            # CLAIMING ROUTE CALIBRATION (TUP R7 Feb 10, 2026 - REAL RESULTS)
                            # Winner #1 My Munnings: PP 105.7 (4th best) - System buried at -3.10 with 55/45 split
                            # Old 55/45 split allowed biases to overwhelm solid PP horses
                            # REAL WORLD PROOF: Even cheap claiming routes, PP matters ~80%
                            base_pp, base_comp = (
                                0.80,
                                0.20,
                            )  # PP-dominant, bonuses provide nuance

                        # Class dropper adjustment (routes)
                        if class_spread > 2.0:
                            pp_weight = base_pp - 0.10
                            comp_weight = base_comp + 0.10
                        else:
                            pp_weight, comp_weight = base_pp, base_comp

            # ═══════════════════════════════════════════════════════════════════════
            # PP RECENCY DAMPENING (TUP R4 tuning - Feb 2026)
            # When a horse has high PP but poor recent speed+form, PP may be stale.
            # Union Coach: PP 112.5 (#1) but last SPD 65, cspeed -0.30 → PP misleading.
            # Reduce PP weight when both speed and form components are negative,
            # indicating the horse's recent performances don't support its historical PP.
            # ═══════════════════════════════════════════════════════════════════════
            if cspeed < -0.2 and c_form < 0:
                recency_discount = min(0.15, abs(cspeed) * 0.3)  # Max 0.15 reduction
                pp_weight = max(0.40, pp_weight - recency_discount)
                comp_weight = 1.0 - pp_weight

            # Apply surface-adaptive hybrid model
            # ALL secondary factors (components + track + bonuses) at component weight
            # Prime Power at PP weight (dominant on dirt, disabled on turf)
            components_with_bonuses = weighted_components + a_track + tier2_bonus
            arace = comp_weight * components_with_bonuses + pp_weight * pp_contribution
        else:
            # No Prime Power available - use traditional component model
            arace = weighted_components + a_track + tier2_bonus

        R = arace

        # ═══════════════════════════════════════════════════════════════
        # BRIDGE OVERRIDE: Replace traditional R with engine R + tier2
        # The HYBRID MODEL above computed arace using traditional logic,
        # but the bridge path uses the engine's pre-computed rating instead.
        # ═══════════════════════════════════════════════════════════════
        if _bridge_R is not None:
            R = _bridge_R
            arace = R

        # ═══════════════════════════════════════════════════════════════
        # FTS (FIRST-TIME STARTER) ADJUSTMENT - TRADITIONAL PATH
        # ═══════════════════════════════════════════════════════════════
        # Apply conservative multiplier for debut horses in MSW races.
        # This mirrors the FTS logic in unified_rating_engine.py for consistency.
        # CRITICAL FIX (Feb 14, 2026): Initialize is_fts BEFORE try block so
        # the NA Running Style section can always reference it safely.
        is_fts = False
        try:
            # Detect FTS: zero starts AND MSW race
            horse_starts = 0
            if "CStarts" in row:
                horse_starts = safe_int(row.get("CStarts", 0), 0)
            elif "Starts" in row:
                horse_starts = safe_int(row.get("Starts", 0), 0)

            race_type_upper = str(race_type).upper()
            is_msw = any(
                pattern in race_type_upper
                for pattern in ["MSW", "MAIDEN SPECIAL WEIGHT", "MD SP WT", "MDN SP WT"]
            )
            is_fts = (horse_starts == 0) and is_msw

            if is_fts:
                # Check if trainer is elite
                trainer_name = str(row.get("Trainer", ""))
                is_elite_trainer = trainer_name in ELITE_TRAINERS

                # Apply FTS multiplier
                base_mult = FTS_PARAMS["base_multiplier"]  # 0.75
                elite_mult = FTS_PARAMS["elite_trainer_multiplier"]  # 1.2

                if is_elite_trainer:
                    fts_multiplier = base_mult * elite_mult  # 0.90
                else:
                    fts_multiplier = base_mult  # 0.75

                R = R * fts_multiplier
        except Exception:
            pass  # Fail gracefully if FTS detection fails

        # ═══════════════════════════════════════════════════════════════
        # NA RUNNING STYLE ADJUSTMENT - TRADITIONAL PATH (Feb 11, 2026)
        # ═══════════════════════════════════════════════════════════════
        # NA horses have unknown running style (insufficient BRISNET data at
        # this distance/surface). Apply dampener + QSP-based confidence scaling.
        # FTS horses get their own multiplier; non-FTS NA horses get this one.
        try:
            _na_style = _style_norm(
                row.get("Style") or row.get("OverrideStyle") or row.get("DetectedStyle")
            )
            _na_quirin = row.get("Quirin", np.nan)
            _na_starts = 0
            if "CStarts" in row:
                _na_starts = safe_int(row.get("CStarts", 0), 0)
            elif "Starts" in row:
                _na_starts = safe_int(row.get("Starts", 0), 0)

            if _na_style == "NA" and _na_starts > 0:
                # Non-FTS horse with unknown style: apply dampener
                na_dampener = NA_STYLE_PARAMS["rating_dampener"]  # 0.85
                # Scale dampener with QSP: higher QSP = less dampening
                try:
                    _na_q = float(_na_quirin)
                    if pd.notna(_na_q) and _na_q > 0:
                        # QSP 8 -> dampener ~0.925, QSP 0 -> dampener 0.85
                        qsp_recovery = (_na_q / 8.0) * (1.0 - na_dampener)  # up to 0.15
                        na_dampener = na_dampener + qsp_recovery * 0.5  # half recovery
                except Exception:
                    pass
                R = R * na_dampener
            elif _na_style == "NA" and _na_starts == 0 and not is_fts:
                # Edge case: NA + zero starts but not MSW (e.g., MCL debut)
                R = R * NA_STYLE_PARAMS["fts_na_dampener"]  # 0.92
        except Exception:
            pass  # Fail gracefully

        # ═══════════════════════════════════════════════════════════════
        # PACE SCENARIO BONUS: Detect speed duels favoring closers
        # ═══════════════════════════════════════════════════════════════
        # TUP R5: 7 E/EP types created speed duel, P runner won from back
        # Count E/EP types in field to detect likely speed duels
        pace_scenario_bonus = 0.0
        if df_styles is not None and not df_styles.empty:
            speed_types = 0
            for _, h_row in df_styles.iterrows():
                h_style = str(h_row.get("Style", "")).upper()
                if "E" in h_style:  # E or E/P types
                    speed_types += 1

            # If 6+ speed horses and this horse is P or S, boost rating
            if speed_types >= 6 and (
                style in ["P", "S"] or "P" in style or "S" in style
            ):
                pace_scenario_bonus = 0.75  # Moderate boost for closers in speed duels

        R += pace_scenario_bonus

        # ELITE: Single-pass outlier clipping with data quality monitoring
        # Typical racing range: -3 to +15. Values beyond suggest parsing errors or unrealistic bonuses
        if R > 20 or R < -5:
            R = np.clip(R, -3, 15)

        # Ensure Quirin is formatted correctly for display (handle NaN)
        quirin_display = quirin if pd.notna(quirin) else None

        # ═══════════════════════════════════════════════════════════════
        # BRIDGE WRITEBACK: Write final adjusted R back to df_styles
        # and skip rows.append — bridge returns df_styles directly.
        # ═══════════════════════════════════════════════════════════════
        if use_unified_engine:
            df_styles.at[row.name, "R"] = round(R, 2)
            df_styles.at[row.name, "Arace"] = round(arace, 2)
            df_styles.at[row.name, "Tier2_Bonus"] = round(tier2_bonus, 2)
            continue  # Skip rows.append — bridge returns df_styles

        rows.append(
            {
                "#": post,
                "Post": post,
                "Horse": name,
                "Style": style,
                "Quirin": quirin_display,
                "Cstyle": round(cstyle, 2),
                "Cpost": round(cpost, 2),
                "Cpace": round(cpace, 2),
                "Cspeed": round(cspeed, 2),
                "Cclass": round(c_class, 2),
                "Cform": round(c_form, 2),
                "Atrack": round(a_track, 2),
                "Tier2_Bonus": round(tier2_bonus, 2),
                "Arace": round(arace, 2),
                "R": round(R, 2),
            }
        )
    # ═══════════════════════════════════════════════════════════════
    # BRIDGE RETURN: When unified engine is active, all horses were
    # written back to df_styles via the continue path above.
    # Return df_styles directly (already has all columns).
    # ═══════════════════════════════════════════════════════════════
    if use_unified_engine:
        return df_styles.sort_values(by="R", ascending=False)

    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(by="R", ascending=False)




def fair_probs_from_ratings(
    ratings_df: pd.DataFrame, ml_odds_dict: dict[str, float] | None = None
) -> dict[str, float]:
    """
    GOLD STANDARD probability calculation with comprehensive validation and ML odds reality check.

    Guarantees:
    1. Always returns valid probability distribution
    2. Probabilities sum to exactly 1.0
    3. No NaN/Inf in output
    4. Thread-safe (no inplace modifications)
    """
    # VALIDATION: Input checks
    if ratings_df is None or ratings_df.empty:
        return {}
    if "R" not in ratings_df.columns or "Horse" not in ratings_df.columns:
        return {}

    # SAFETY: Work on copy to avoid side effects (only if we modify)
    df = ratings_df.copy()  # Keep copy since we add R_numeric column

    # VALIDATION: Ensure 'R' is numeric
    df["R_numeric"] = pd.to_numeric(df["R"], errors="coerce")

    # GOLD STANDARD: Handle NaN with intelligent fallback
    median_r = df["R_numeric"].median()
    if pd.isna(median_r) or not np.isfinite(median_r):
        # All ratings are invalid - use mean as fallback, or 0
        mean_r = df["R_numeric"].mean()
        median_r = mean_r if np.isfinite(mean_r) else 0.0
    df["R_numeric"] = df["R_numeric"].fillna(median_r)

    # EXTRACT: Ratings array
    r = df["R_numeric"].values
    if len(r) == 0:
        return {}

    # VALIDATION: Final check for invalid values
    if not np.all(np.isfinite(r)):
        # Replace any remaining non-finite with median
        finite_mask = np.isfinite(r)
        if np.any(finite_mask):
            median_finite = np.median(r[finite_mask])
        else:
            median_finite = 0.0
        r[~finite_mask] = median_finite

    # COMPUTE: Softmax probabilities (using gold-standard function)
    p = softmax_from_rating(r)

    # VALIDATION: Ensure we have valid probabilities
    if len(p) != len(r):
        # Should never happen, but fallback to uniform
        p = np.ones(len(r)) / len(r)

    # GOLD STANDARD: Build horse->probability mapping with validation
    horses = df["Horse"].values
    result = {}
    for i, h in enumerate(horses):
        if i < len(p):
            prob = float(p[i])
            # Ensure probability is valid
            if not np.isfinite(prob) or prob < 0 or prob > 1:
                prob = 1.0 / len(horses)  # Fallback to uniform
            result[h] = prob

    # FINAL VALIDATION: Ensure probabilities sum to 1.0
    total_prob = sum(result.values())
    if total_prob > 0 and abs(total_prob - 1.0) > 1e-6:
        # Normalize to exactly 1.0
        result = {h: p / total_prob for h, p in result.items()}

    # ML ODDS REALITY CHECK: Blend model probabilities with market wisdom
    # The market odds reflect real money and decades of handicapping experience
    # CALIBRATED (Feb 7, 2026): Softened caps to reduce Rating/Fair% disconnect.
    # Previously, heavy favorites got Floor=15% even if model rated them 7th,
    # causing Fair% to be #1 while Rating was #6 (confusing to users).
    if ml_odds_dict:
        adjusted = False
        for horse, prob in result.items():
            ml_odds = ml_odds_dict.get(horse, 5.0)

            # --- LONGSHOT CAPS: Prevent longshots from getting unrealistic probabilities ---
            # FIX (Feb 10, 2026): Tightened caps. A 12/1 shot getting 69% probability
            # passed unchecked previously. Now cap based on realistic ML-odds ranges.
            # Historical data: TuP CLM 8500 — 83% of winners at <5/1, only 4% at >10/1.
            if ml_odds >= 30.0 and prob > 0.08:
                result[horse] = 0.08
                adjusted = True
            elif ml_odds >= 20.0 and prob > 0.12:
                result[horse] = 0.12
                adjusted = True
            elif ml_odds >= 10.0 and prob > 0.20:
                result[horse] = 0.20  # 10/1+ shots capped at 20%
                adjusted = True
            elif ml_odds >= 6.0 and prob > 0.30:
                result[horse] = 0.30  # 6/1+ shots capped at 30%
                adjusted = True

            # --- FAVORITE FLOORS: Prevent strong favorites from being crushed ---
            # SOFTENED: Only apply to very strong favorites, lower floors
            # A 1/1 or lower (even money) should never be below ~10%
            elif ml_odds <= 1.0 and prob < 0.10:
                result[horse] = max(prob, 0.10)
                adjusted = True
            elif ml_odds <= 2.0 and prob < 0.08:
                result[horse] = max(prob, 0.08)
                adjusted = True

        # If we adjusted any probabilities, renormalize
        if adjusted:
            total_prob = sum(result.values())
            if total_prob > 0:
                result = {h: p / total_prob for h, p in result.items()}

    return result

