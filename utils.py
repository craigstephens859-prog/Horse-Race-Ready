# utils.py
# Horse Race Ready — Pure Utility Functions
# Extracted from app.py (Phase 2 module split).
# All functions are side-effect-free with NO Streamlit dependencies.

from __future__ import annotations

import math
import re

import numpy as np

from config import TRACK_ALIASES

# ===================== Distance Utilities =====================


def _distance_bucket_from_text(distance_txt: str) -> str:
    """
    Buckets into ≤6f, 6.5–7f, or 8f+ (routes).
    """
    d = (distance_txt or "").strip().lower()

    # Try to extract numeric furlongs from format like "6f", "6.5f", "5 1/2f"
    if "f" in d:
        s = d.replace("½", ".5").replace(" 1/2", ".5")
        m = re.search(r"(\d+(?:\.\d+)?)", s)
        if m:
            val = float(m.group(1))
            if val <= 6.0:
                return "≤6f"
            if val < 8.0:
                return "6.5–7f"
            return "8f+"

    # Miles
    if "mile" in d or "m" in d:
        if "70" in d and "yard" in d:
            return "8f+"
        fracs = {
            "1/16": 1 / 16,
            "1/8": 1 / 8,
            "3/16": 3 / 16,
            "1/4": 1 / 4,
            "5/16": 5 / 16,
            "3/8": 3 / 8,
            "7/16": 7 / 16,
            "1/2": 0.5,
        }
        base = 0.0
        m0 = re.search(r"(\d+)\s*mile", d)
        if m0:
            base = float(m0.group(1))
        extra = 0.0
        for f, v in fracs.items():
            if f in d:
                extra = v
                break
        total_mi = base + extra
        total_f = total_mi * 8.0
        if total_f < 6.5:
            return "≤6f"
        if total_f < 8.0:
            return "6.5–7f"
        return "8f+"
    return "8f+"


def distance_to_furlongs(dist_str: str) -> float:
    """Convert distance string to furlongs (module-level utility)."""
    dist_str = (dist_str or "").lower().strip()
    if "f" in dist_str:
        try:
            return float(
                dist_str.replace("f", "")
                .replace("furlongs", "")
                .replace("furlong", "")
                .strip()
            )
        except ValueError:
            return 6.0
    elif "m" in dist_str or "mile" in dist_str:
        if "1/16" in dist_str:
            return 8.5
        elif "1/8" in dist_str:
            return 9.0
        elif "3/16" in dist_str:
            return 9.5
        elif "1/4" in dist_str:
            return 10.0
        elif "1/2" in dist_str:
            return 12.0
        else:
            return 8.0
    return 6.0


def distance_bucket(distance_txt: str) -> str:
    try:
        return _distance_bucket_from_text(distance_txt)
    except Exception:
        return "8f+"


def is_marathon_distance(distance_txt: str) -> bool:
    """
    Detect if race is marathon distance (1½ miles+ / 12 furlongs+).
    Marathon distances require different weighting strategy.

    Based on McKnight G3 post-race analysis where standard weights failed.
    """
    if not distance_txt:
        return False

    # Convert to furlongs for comparison
    distance_lower = distance_txt.lower().strip()

    # Direct furlong matches
    if "f" in distance_lower:
        try:
            furlongs = float(distance_lower.replace("f", "").strip())
            return furlongs >= 12.0
        except Exception:
            pass

    # Mile conversions
    if "mile" in distance_lower or "m" in distance_lower:
        # 1½ miles = 12f, 1⅝ = 13f, 1¾ = 14f, 2 miles = 16f
        if "½" in distance_txt or "1/2" in distance_txt or "1.5" in distance_txt:
            return True
        if "⅝" in distance_txt or "5/8" in distance_txt or "1.625" in distance_txt:
            return True
        if "¾" in distance_txt or "3/4" in distance_txt or "1.75" in distance_txt:
            return True
        if "2" in distance_txt and "mile" in distance_lower:
            return True

    return False


def is_sprint_distance(distance_txt: str) -> bool:
    """Detect sprint distances (≤6.5f)"""
    if not distance_txt:
        return False
    try:
        if "f" in distance_txt.lower():
            furlongs = float(distance_txt.lower().replace("f", "").strip())
            return furlongs <= 6.5
    except Exception:
        pass
    return False


def _auto_distance_label(s: str) -> str:
    m = re.search(r"(?i)\b(\d+(?:\s*1/2|½)?\s*furlongs?)\b", s)
    if m:
        return m.group(1).title().replace("1/2", "½")
    if re.search(r"(?i)\b8\s*1/2\s*furlongs?\b", s):
        return "8 1/2 Furlongs"
    if re.search(r"(?i)\b8\s*furlongs?\b", s):
        return "8 Furlongs"
    if re.search(r"(?i)\b9\s*furlongs?\b", s):
        return "9 Furlongs"
    if re.search(r"(?i)\b1\s*mile\b", s):
        return "1 Mile"
    if re.search(r"(?i)\b1\s*1/16\b", s):
        return "1 1/16 Miles"
    if re.search(r"(?i)\b7\s*furlongs?\b", s):
        return "7 Furlongs"
    return "6 Furlongs"


# ===================== Track Utilities =====================


def _canonical_track(track_name: str) -> str:
    t = (track_name or "").strip().lower()
    for canon, toks in TRACK_ALIASES.items():
        if t == canon.lower() or t in toks:
            return canon
    for canon, toks in TRACK_ALIASES.items():
        for tok in toks:
            if tok in t:
                return canon
    return (track_name or "").strip()


# ===================== Style Normalization =====================


def _style_norm(style: str) -> str:
    s = (style or "NA").upper()
    return "E/P" if s in ("EP", "E/P") else s


def _normalize_style(tok: str) -> str:
    """Normalize running style token to canonical form.

    Handles all case variations: EP, E/P, ep, e/p, Ep, etc.
    """
    t = (tok or "").upper().strip()
    # Handle all E/P variations
    if t in ("EP", "E/P", "E-P"):
        return "E/P"
    # Return uppercase version for other styles
    return t


# ===================== Name & Value Conversion =====================


def normalize_horse_name(name):
    """Normalize horse name for matching: remove apostrophes, extra spaces, lowercase.

    Use this function consistently throughout the codebase for horse name matching
    to avoid scratched horses appearing or missing matches.

    Args:
        name: Horse name to normalize

    Returns:
        str: Normalized name (lowercase, no apostrophes/backticks, single spaces)
    """
    return " ".join(str(name).replace("'", "").replace("`", "").lower().split())


def safe_float(value, default=0.0):
    """
    Convert value to float, handling:
    - Percentage strings like '75.6%'
    - American odds like '+150', '-200'
    - Regular numbers

    Args:
        value: Value to convert (string, int, float, etc.)
        default: Default value if conversion fails

    Returns:
        float: Converted value or default
    """
    try:
        if isinstance(value, str):
            # Remove % symbol and any whitespace
            value = value.strip().rstrip("%")
            # Remove + symbol from odds like '+150'
            if value.startswith("+"):
                value = value[1:]
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default


def safe_int(value, default=0):
    """
    Convert value to int safely, handling strings, floats, None, NaN.

    CRITICAL FIX (Feb 14, 2026): This function was called 4 times in the
    FTS and NA Running Style adjustment sections but was never defined,
    causing both sections to silently crash on every horse. This meant:
    - FTS debut multiplier (0.75x) was NEVER applied
    - NA style dampener (0.85x) was NEVER applied

    Args:
        value: Value to convert (string, int, float, None, NaN, etc.)
        default: Default value if conversion fails

    Returns:
        int: Converted value or default
    """
    try:
        if value is None:
            return default
        if isinstance(value, float) and (np.isnan(value) or np.isinf(value)):
            return default
        return int(float(value))
    except (ValueError, TypeError, AttributeError):
        return default


# ===================== Odds Utilities =====================


def odds_to_decimal(odds_str: str) -> float:
    """
    Convert various odds formats to decimal for comparison.
    Handles: fractional (5/2), decimal (3.5), American (+250, -150)
    Returns decimal odds (e.g., 5/2 → 3.5)
    """
    if not odds_str or odds_str == "?":
        return 0.0

    odds_str = str(odds_str).strip()

    try:
        # Fractional format: "5/2", "7/1"
        if "/" in odds_str:
            parts = odds_str.split("/")
            numerator = float(parts[0])
            denominator = float(parts[1])
            return (numerator / denominator) + 1.0

        # American format: "+250", "-150"
        elif odds_str.startswith(("+", "-")):
            american = float(odds_str)
            if american > 0:
                return (american / 100) + 1.0
            else:
                return (100 / abs(american)) + 1.0

        # Decimal format: "3.5", "2.0"
        else:
            decimal = float(odds_str)
            # If already in decimal format (> 1), return as-is
            if decimal >= 1.0:
                return decimal
            # If looks like fractional without slash (0.5 = 1/2)
            else:
                return decimal + 1.0

    except (ValueError, ZeroDivisionError, IndexError):
        return 0.0


def fair_to_american(p: float) -> float:
    if p <= 0:
        return math.inf
    if p >= 1:
        return 0.0
    dec = 1.0 / p
    return round((dec - 1) * 100, 0) if dec >= 2 else round(-100 / (dec - 1), 0)


def fair_to_american_str(p: float) -> str:
    v = fair_to_american(p)
    if math.isinf(v):
        return "N/A"
    return f"+{int(v)}" if v > 0 else f"{int(v)}"


# ===================== Form Analysis =====================


def calculate_form_trend(recent_finishes: list[int]) -> float:
    """OPTIMIZED Feb 9 2026: Form trend with calibrated momentum.

    Returns: trend factor (-1.0 to +2.0). Form trend is a MODIFIER, not a dominator.
    Key changes: won-last-2 = +2.0 (was +4.0), declining = -0.5 (was -1.2).
    Validated on Oaklawn R9 (Air of Defiance #2).
    """
    if len(recent_finishes) < 1:
        return 0.0
    if recent_finishes[0] == 1:
        if len(recent_finishes) >= 2 and recent_finishes[1] == 1:
            return 2.0  # Was 4.0 — too dominant
        return 1.5  # Was 2.5
    elif recent_finishes[0] in [2, 3]:
        return 0.7  # Was 1.0
    if len(recent_finishes) < 2:
        return 0.0
    weights = [0.4, 0.3, 0.2, 0.1][: len(recent_finishes)]
    weighted_avg = sum(
        f * w for f, w in zip(recent_finishes, weights, strict=False)
    ) / sum(weights)
    if len(recent_finishes) >= 3:
        r3 = recent_finishes[:3]
        if r3[0] < r3[1] < r3[2]:
            return 1.5
        elif r3[0] > r3[1] > r3[2]:
            return (
                -0.5
            )  # Was -1.2 — declining form often reflects class/distance shifts
    if weighted_avg <= 1.5:
        return 1.2
    elif weighted_avg <= 3.0:
        return 0.8
    elif weighted_avg <= 5.0:
        return 0.0
    elif weighted_avg <= 7.0:
        return -0.5
    else:
        return -1.0
