# app.py
# Horse Race Ready — IQ Mode (Full, final version)
# - Robust PP parsing (incl. NA firsters)
# - Auto track + race-type detection mapped to constant base_class_bias keys
# - Track-bias integration (by track/surface/distance bucket + style/post)
# - Per-horse angles + pedigree tweaks folded into class math
# - Speed Figure parsing and integration
# - Centralized for easy tuning
# - Robust error handling with st.warning
# - Advanced "common sense" A/B/C/D strategy builder w/ budgeting, field size logic, straight/box examples
# - Super High 5 exotic support
# - Classic bullet-style report with download buttons
# - Resilient to older/newer Streamlit rerun APIs

import os
import re
import math
import time
import logging
from typing import Dict, List, Tuple, Optional, Any
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try importing ML Engine for Phase 2 functionality
try:
    from ml_engine import MLCalibrator, RaceDatabase
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    MLCalibrator = None
    RaceDatabase = None

# Historical Data System - LAZY LOADED for fast boot times
# PyTorch is 2-3GB and slows down Render deploys significantly
# We load it only when Section E is accessed
HISTORICAL_DATA_AVAILABLE = None  # None = not yet loaded, True/False after load attempt
HistoricalDataBuilder = None
convert_to_ml_format = None
HISTORICAL_IMPORT_ERROR = None

# ULTRATHINK INTEGRATION: Import optimized 8-angle system
try:
    from horse_angles8 import compute_eight_angles
    ANGLES_AVAILABLE = True
except ImportError:
    ANGLES_AVAILABLE = False
    compute_eight_angles = None

# RACE CLASS PARSER: Comprehensive race type and purse analysis
try:
    from race_class_parser import (
        parse_and_calculate_class,
        parse_race_conditions,
        get_hierarchy_level,
        calculate_class_weight,
        CLASS_MAP,
        LEVEL_MAP
    )
    RACE_CLASS_PARSER_AVAILABLE = True
except ImportError:
    RACE_CLASS_PARSER_AVAILABLE = False
    parse_and_calculate_class = None
    CLASS_MAP = {}
    LEVEL_MAP = {}

# ULTRATHINK INTEGRATION: Import comprehensive rating system
try:
    from parser_integration import ParserToRatingBridge
    COMPREHENSIVE_RATING_AVAILABLE = True
except ImportError:
    COMPREHENSIVE_RATING_AVAILABLE = False
    ParserToRatingBridge = None

# ULTRATHINK V2: Gold-standard parser (94% accuracy, 50+ edge cases)
try:
    from elite_parser_v2_gold import GoldStandardBRISNETParser
    ELITE_PARSER_AVAILABLE = True
except ImportError:
    ELITE_PARSER_AVAILABLE = False
    GoldStandardBRISNETParser = None

# ULTRATHINK V2: Unified rating engine (consolidates all systems)
try:
    from unified_rating_engine import UnifiedRatingEngine
    UNIFIED_ENGINE_AVAILABLE = True
except ImportError:
    UNIFIED_ENGINE_AVAILABLE = False
    UnifiedRatingEngine = None

# ULTRATHINK V3: Gold High-IQ Database (optimized for ML retraining)
try:
    from gold_database_manager import GoldHighIQDatabase
    gold_db = GoldHighIQDatabase("gold_high_iq.db")
    GOLD_DB_AVAILABLE = True
except Exception as e:
    GOLD_DB_AVAILABLE = False
    gold_db = None
    print(f"Gold database initialization error: {e}")

# SECURITY: Import input validation and protection utilities
try:
    from security_validators import (
        sanitize_pp_text,
        validate_track_name,
        validate_distance_string,
        sanitize_race_metadata,
        RateLimiter
    )
    SECURITY_VALIDATORS_AVAILABLE = True
except ImportError as e:
    SECURITY_VALIDATORS_AVAILABLE = False
    print(f"Security validators not available: {e}")

# ===================== Page / Model Settings =====================

st.set_page_config(page_title="Horse Race Ready — IQ Mode", page_icon="🏇", layout="wide")
st.title("🏇  Horse Race Ready — IQ Mode")

# ---------- Durable state ----------
if "parsed" not in st.session_state:
    st.session_state["parsed"] = False
if "pp_text_cache" not in st.session_state:
    st.session_state["pp_text_cache"] = ""

# Defaults for Race Info
if 'track_name' not in st.session_state:
    st.session_state['track_name'] = "Unknown Track"
if 'surface_type' not in st.session_state:
    st.session_state['surface_type'] = "Dirt"
if 'condition_txt' not in st.session_state:
    st.session_state['condition_txt'] = "fast"
if 'distance_txt' not in st.session_state:
    st.session_state['distance_txt'] = "6 Furlongs"
if 'purse_val' not in st.session_state:
    st.session_state['purse_val'] = 80000

# ---------- Version-safe rerun ----------


def _safe_rerun():
    """Works on both old and new Streamlit versions."""
    rr = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rr is not None:
        rr()
    else:
        st.warning("Rerun API not available in this Streamlit build. Adjust any widget to refresh.")
        st.stop()


# ---------- OpenAI (for narrative report only) ----------
MODEL = os.getenv("OPENAI_MODEL", "gpt-5")
TEMP = float(os.getenv("OPENAI_TEMPERATURE", "0.5"))
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
use_sdk_v1 = True
client = None
openai = None
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    use_sdk_v1 = False
try:
    import openai
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except Exception:
    openai = None

# SECURITY: Rate limiter for OpenAI API calls (10 calls per 60 seconds)
if SECURITY_VALIDATORS_AVAILABLE:
    openai_rate_limiter = RateLimiter(max_calls=10, time_window=60)
else:
    openai_rate_limiter = None


def model_supports_temperature(model_name: str) -> bool:
    m = (model_name or "").lower()
    return not (m.startswith("gpt-5") or m.startswith("o4") or m.startswith("o3"))


def call_openai_messages(messages: List[Dict]) -> str:
    # SECURITY: Check rate limit before making API call
    if openai_rate_limiter is not None and not openai_rate_limiter.allow_call():
        return "(Rate limit exceeded. Please wait before generating another report. Max 10 reports per minute.)"

    if not OPENAI_API_KEY or (client is None and openai is None):
        return "(Narrative generation disabled — set OPENAI_API_KEY as environment variable)"
    if use_sdk_v1 and client is not None:
        try:
            kwargs = {"model": MODEL, "messages": messages}
            if model_supports_temperature(MODEL):
                kwargs["temperature"] = TEMP
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
                resp = client.chat.completions.create(model=MODEL, messages=messages)
                return resp.choices[0].message.content
            return f"(Error in OpenAI call: {e})"
    else:
        try:
            kwargs = {"model": MODEL, "messages": messages}
            if model_supports_temperature(MODEL):
                kwargs["temperature"] = TEMP
            resp = openai.ChatCompletion.create(**kwargs)
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
                resp = openai.ChatCompletion.create(model=MODEL, messages=messages)
                return resp["choices"][0]["message"]["content"]
            return f"(Error in OpenAI call: {e})"

# ===================== Model Config & Tuning =====================


MODEL_CONFIG = {
    # --- Rating Model ---
    "softmax_tau": 0.85,  # Controls win prob "sharpness". Lower = more spread out.
    "speed_fig_weight": 0.05,  # (Fig - Avg) * Weight. 0.05 = 10 fig points = 0.5 bonus.
    "first_timer_fig_default": 50,  # Assumed speed fig for a 1st-time starter.

    # --- Pace & Style Model ---
    "ppi_multiplier": 1.5,  # Overall impact of the Pace Pressure Index (PPI).
    "ppi_tailwind_factor": 0.6,  # How much of the PPI value is given to E/EP or S horses.
    "style_strength_weights": {  # Multiplier for pace tailwind based on strength.
        "Strong": 1.0,
        "Solid": 0.8,
        "Slight": 0.5,
        "Weak": 0.3
    },

    # --- Manual Bias Model (Section B) ---
    "style_match_table": {
        "favoring": {"E": 0.70, "E/P": 0.50, "P": -0.20, "S": -0.50},
        "closer favoring": {"E": -0.50, "E/P": -0.20, "P": 0.25, "S": 0.50},
        "fair/neutral": {"E": 0.0, "E/P": 0.0, "P": 0.0, "S": 0.0},
    },
    "style_quirin_threshold": 6,  # Quirin score needed for "strong" style bonus.
    "style_quirin_bonus": 0.10,  # Bonus for strong style (e.g., E w/ Q>=6).
    "post_bias_rail_bonus": 0.40,
    "post_bias_inner_bonus": 0.25,
    "post_bias_mid_bonus": 0.15,
    "post_bias_outside_bonus": 0.25,

    # --- Pedigree & Angle Tweaks (Cclass) ---
    "ped_dist_bonus": 0.06,
    "ped_dist_penalty": -0.04,  # Note: This should be negative
    "ped_dist_neutral_bonus": 0.03,
    "ped_first_pct_threshold": 14,  # Sire/Damsire 1st-time-win %
    "ped_first_pct_bonus": 0.02,
    "angle_debut_msw_bonus": 0.05,
    "angle_debut_other_bonus": 0.03,
    "angle_debut_sprint_bonus": 0.01,
    "angle_second_career_bonus": 0.03,
    "angle_surface_switch_bonus": 0.02,
    "angle_blinkers_on_bonus": 0.02,
    "angle_blinkers_off_bonus": 0.005,
    "angle_shipper_bonus": 0.01,
    "angle_off_track_route_bonus": 0.01,
    "angle_roi_pos_max_bonus": 0.06,  # Max bonus from positive ROI angles
    "angle_roi_pos_per_bonus": 0.01,  # Bonus per positive ROI angle
    "angle_roi_neg_max_penalty": 0.03,  # Max penalty from negative ROI angles (applied as -)
    "angle_roi_neg_per_penalty": 0.005,  # Penalty per negative ROI angle (applied as -)
    "angle_tweak_min_clip": -0.12,  # Min/Max total adjustment from all angles
    "angle_tweak_max_clip": 0.12,

    # --- Exotics & Strategy ---
    "exotic_bias_weights": (1.30, 1.15, 1.05, 1.03),  # (1st, 2nd, 3rd, 4th) Harville bias
    "strategy_confident": {  # Placeholders, not used by new strategy builder
        "ex_max": 4, "ex_min_prob": 0.020,
        "tri_max": 6, "tri_min_prob": 0.010,
        "sup_max": 8, "sup_min_prob": 0.008,
    },
    "strategy_value": {  # Placeholders, not used by new strategy builder
        "ex_max": 6, "ex_min_prob": 0.015,
        "tri_max": 10, "tri_min_prob": 0.008,
        "sup_max": 12, "sup_min_prob": 0.006,
    }
}

# =========================
# Track parsing, race-type, distance options, and track-bias integration
# =========================

# -------- Distance options (UI) --------
DISTANCE_OPTIONS = [
    # Short sprints
    "4 Furlongs", "4 1/2 Furlongs", "4.5 Furlongs",
    "5 Furlongs", "5 1/2 Furlongs", "5.5 Furlongs",
    "6 Furlongs", "6 1/2 Furlongs", "6.5 Furlongs", "7 Furlongs",
    # Routes & variants
    "1 Mile", "1 Mile 70 Yards",
    "1 1/16 Miles", "1 1/8 Miles", "1 3/16 Miles", "1 1/4 Miles",
    "1 5/16 Miles", "1 3/8 Miles", "1 7/16 Miles", "1 1/2 Miles",
    "1 9/16 Miles", "1 5/8 Miles", "1 3/4 Miles", "1 7/8 Miles", "2 Miles"
]


def _distance_bucket_from_text(distance_txt: str) -> str:
    """
    Buckets into ≤6f, 6.5–7f, or 8f+ (routes).
    """
    d = (distance_txt or "").strip().lower()
    # Furlongs
    if "furlong" in d:
        s = d.replace("½", ".5").replace(" 1/2", ".5")
        m = re.search(r'(\d+(?:\.\d+)?)', s)
        if m:
            val = float(m.group(1))
            if val <= 6.0:
                return "≤6"
            if val < 8.0:
                return "6.5–7"
            return "8f+"
    # Miles
    if "mile" in d:
        if "70" in d and "yard" in d:
            return "8f+"
        fracs = {"1/16": 1 / 16, "1/8": 1 / 8, "3/16": 3 / 16, "1/4": 1 / 4,
                 "5/16": 5 / 16, "3/8": 3 / 8, "7/16": 7 / 16, "1/2": 0.5}
        base = 0.0
        m0 = re.search(r'(\d+)\s*mile', d)
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
            return "≤6"
        if total_f < 8.0:
            return "6.5–7"
        return "8f+"
    return "8f+"


def distance_bucket(distance_txt: str) -> str:
    try:
        return _distance_bucket_from_text(distance_txt)
    except Exception:
        return "8f+"


# -------- Canonical track names + aliases --------
TRACK_ALIASES = {
    "Del Mar": ["del mar", "dmr"],
    "Keeneland": ["keeneland", "kee"],
    "Churchill Downs": ["churchill downs", "cd", "churchill"],
    "Kentucky Downs": ["kentucky downs", "kd"],
    "Saratoga": ["saratoga", "sar"],
    "Santa Anita": ["santa anita", "sa", "santa anita park"],
    "Mountaineer": ["mountaineer", "mnr"],
    "Charles Town": ["charlestown", "charles town", "ct"],
    "Gulfstream": ["gulfstream", "gulfstream park", "gp"],
    "Tampa Bay Downs": ["tampa", "tampa bay downs", "tam"],
    "Turf Paradise": ["turf paradise", "tup", "turf"],
    "Belmont Park": ["belmont", "belmont park", "bel", "aqueduct at belmont", "belmont at aqueduct", "big a"],
    "Horseshoe Indianapolis": ["horseshoe indianapolis", "indiana grand", "ind", "indy"],
    "Penn National": ["penn national", "pen"],
    "Presque Isle Downs": ["presque isle", "presque isle downs", "pid"],
    "Woodbine": ["woodbine", "wo"],
    "Evangeline Downs": ["evangeline", "evangeline downs", "evd"],
    "Fairmount Park": ["fairmount park", "fanduel fairmount", "cah", "collinsville"],
    "Finger Lakes": ["finger lakes", "fl"]
}
_CANON_BY_TOKEN = {}
for canon, toks in TRACK_ALIASES.items():
    for t in toks:
        _CANON_BY_TOKEN[t] = canon


def parse_track_name_from_pp(pp_text: str) -> str:
    """
    Parse track name from BRISNET PP text.
    Checks both:
    1. Header/title text for full track names
    2. Race history lines for track abbreviations (e.g., 29Dec25Tup, 08Nov25Aquª)
    """
    text = (pp_text or "")[:2000].lower()  # Increased to capture race history

    # First, check for track abbreviations in race history date lines
    # Pattern: DDMmmYYTrk (e.g., 29Dec25Tup, 08Nov25Aquª)
    date_line_pattern = r'\d{2}[A-Za-z]{3}\d{2}([A-Za-z]{2,4})'
    for match in re.finditer(date_line_pattern, text):
        track_code = match.group(1).lower()
        if track_code in _CANON_BY_TOKEN:
            return _CANON_BY_TOKEN[track_code]

    # Second, check for full track names in header
    for token, canon in _CANON_BY_TOKEN.items():
        if re.search(rf'\b{re.escape(token)}\b', text):
            return canon

    # Third, check for multi-word track names
    for canon, toks in TRACK_ALIASES.items():
        for t in toks:
            t_words = [w for w in t.split() if len(w) > 2]
            if t_words and all(re.search(rf'\b{re.escape(w)}\b', text) for w in t_words):
                return canon

    return ""


def detect_race_number(pp_text: str) -> Optional[int]:
    """Extract race number from PP text header (e.g., 'Race 6')."""
    s = pp_text or ""
    # Look for "Race N" pattern in first few lines
    m = re.search(r'(?mi)\bRace\s+(\d+)\b', s[:500])
    if m:
        try:
            return int(m.group(1))
        except (ValueError, AttributeError):
            # Regex group missing or invalid
            pass
    return None


def parse_brisnet_race_header(pp_text: str) -> Dict[str, Any]:
    """
    Parse comprehensive BRISNET race header information.

    Expected format:
    Ultimate PP's w/ QuickPlay Comments | Track Name | Race Type Purse | Distance | Age/Sex | Date | Race #

    Example:
    Ultimate PP's w/ QuickPlay Comments | Turf Paradise | ©Hcp 50000 | 6 Furlongs | 3yo Fillies | Monday, February 02, 2026 | Race 8

    Returns dict with all extracted fields:
    - track_name, race_number, race_type, purse_amount, distance, age_restriction, sex_restriction, race_date, day_of_week
    """
    if not pp_text:
        return {}

    result = {
        'track_name': '',
        'race_number': 0,
        'race_type': '',
        'purse_amount': 0,
        'distance': '',
        'age_restriction': '',
        'sex_restriction': '',
        'race_date': '',
        'day_of_week': ''
    }

    # Get first line (header)
    header_line = pp_text.strip().split('\n')[0].strip()

    # Split by pipe delimiter
    parts = [p.strip() for p in header_line.split('|')]

    if len(parts) < 2:
        # No pipes - return empty (will fall back to individual parsers)
        return result

    # Parse each section
    for i, part in enumerate(parts):
        part_lower = part.lower()

        # Skip "Ultimate PP's w/ QuickPlay Comments"
        if 'ultimate' in part_lower or 'quickplay' in part_lower:
            continue

        # Track name (first non-Ultimate part)
        if not result['track_name'] and i > 0:
            # Check if this looks like a track name
            if not re.search(r'\d+\s*(?:furlong|mile|yard)', part_lower) and \
               not re.search(r'(clm|alw|stk|hcp|mdn|msw|aoc)', part_lower) and \
               not re.search(r'race\s+\d+', part_lower) and \
               not re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', part_lower):
                result['track_name'] = part
                continue

        # Race type + purse (e.g., "©Hcp 50000", "Clm 4000")
        race_type_match = re.search(r'([©¨§]?[A-Za-z]+)\s+(\d+)', part)
        if race_type_match and not result['race_type']:
            result['race_type'] = race_type_match.group(1)
            try:
                result['purse_amount'] = int(race_type_match.group(2))
            except BaseException:
                pass
            continue

        # Distance (e.g., "6 Furlongs", "1 Mile", "1„ Mile")
        if re.search(r'\d+\s*(?:furlong|mile|yard|f\b)', part_lower):
            result['distance'] = part
            continue

        # Age/Sex restrictions (e.g., "3yo Fillies", "4&up", "F&M")
        if re.search(r'(?:\d+yo|f&m|fillies|mares|colts|geldings|4&up|3&up)', part_lower):
            age_match = re.search(r'(\d+yo|4&up|3&up)', part_lower)
            if age_match:
                result['age_restriction'] = age_match.group(1)

            sex_match = re.search(r'(fillies?|mares?|colts?|geldings?|f&m)', part_lower, re.IGNORECASE)
            if sex_match:
                result['sex_restriction'] = sex_match.group(1).title()
            continue

        # Date (e.g., "Monday, February 02, 2026")
        date_match = re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[,\s]+(.+\d{4})', part_lower)
        if date_match:
            result['day_of_week'] = date_match.group(1).title()
            result['race_date'] = date_match.group(2).strip()
            continue

        # Race number (e.g., "Race 8")
        race_num_match = re.search(r'race\s+(\d+)', part_lower)
        if race_num_match:
            try:
                result['race_number'] = int(race_num_match.group(1))
            except BaseException:
                pass
            continue

    return result


# -------- Race-type constants + detection --------
# This dictionary is our constant. It measures the "reliability" of the race type.
base_class_bias = {
    "stakes (g1)": 0.90,  # PEGASUS TUNING: G1 class weight reduced in rating engine (10.0→6.0)
    "stakes (g2)": 0.92,
    "stakes (g3)": 0.93,
    "stakes (listed)": 0.95,
    "stakes": 0.95,
    "allowance optional claiming (aoc)": 0.96,
    "maiden special weight": 0.97,
    "allowance": 0.99,
    "starter handicap": 1.02,
    "starter allowance": 1.03,
    "waiver claiming": 1.07,
    "claiming": 1.12,
    "maiden claiming": 1.15,
}

condition_modifiers = {
    "fast": 1.0, "firm": 1.0, "good": 1.03, "yielding": 1.04,
    "muddy": 1.08, "sloppy": 1.10, "heavy": 1.10,
}


def detect_race_type(pp_text: str) -> str:
    """
    Normalize many wordings into the exact key set used by base_class_bias.
    """
    s = (pp_text or "")[:1000].lower()

    # Graded stakes first
    if re.search(r'\b(g1|grade\s*i)\b', s):
        return "stakes (g1)"
    if re.search(r'\b(g2|grade\s*ii)\b', s):
        return "stakes (g2)"
    if re.search(r'\b(g3|grade\s*iii)\b', s):
        return "stakes (g3)"

    # Listed / generic stakes
    if "listed" in s:
        return "stakes (listed)"
    if re.search(r'\bstakes?\b', s):
        return "stakes"

    # Maiden
    if re.search(r'\b(mdn|maiden)\b', s):
        if re.search(r'(mcl|mdn\s*clm|maiden\s*claim)', s):
            return "maiden claiming"
        if re.search(r'(msw|maiden\s*special|maiden\s*sp\s*wt)', s):
            return "maiden special weight"
        return "maiden special weight"

    # AOC
    if re.search(r'\b(oc|aoc|optional\s*claim)\b', s):
        return "allowance optional claiming (aoc)"

    # Starter
    if re.search(r'\bstarter\s*allow', s):
        return "starter allowance"
    if re.search(r'\bstarter\s*h(andi)?cap\b', s):
        return "starter handicap"

    # Waiver Claiming
    if re.search(r'\b(waiver|wcl|w\s*clm)\b', s):
        return "waiver claiming"

    # Claiming
    if re.search(r'\bclm|claiming\b', s):
        return "claiming"

    # Allowance last
    if re.search(r'\ballow(ance)?\b', s):
        return "allowance"

    return "allowance"


# -------- Track bias profiles (additive deltas; conservative magnitude) --------
TRACK_BIAS_PROFILES = {
    "Keeneland": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.35, "E/P": 0.20, "P": -0.10, "S": -0.25},
                    "post": {"rail": 0.20, "inner": 0.10, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.15, "E/P": 0.10, "P": 0.00, "S": -0.10},
                       "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Del Mar": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Churchill Downs": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.15, "E/P": 0.05, "P": 0.00, "S": -0.10},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Kentucky Downs": {
        "Turf": {
            "≤6f": {"runstyle": {"E": -0.05, "E/P": 0.00, "P": 0.10, "S": 0.15},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}},
            "6.5–7f": {"runstyle": {"E": -0.05, "E/P": 0.00, "P": 0.10, "S": 0.15},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}},
            "8f+": {"runstyle": {"E": -0.10, "E/P": 0.00, "P": 0.10, "S": 0.20},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}}
        }
    },
    "Saratoga": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Santa Anita": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Mountaineer": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        }
    },
    "Charles Town": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.45, "E/P": 0.25, "P": -0.15, "S": -0.35},
                    "post": {"rail": 0.25, "inner": 0.15, "mid": -0.05, "outside": -0.10}},
            "6.5–7f": {"runstyle": {"E": 0.30, "E/P": 0.20, "P": -0.10, "S": -0.25},
                       "post": {"rail": 0.15, "inner": 0.10, "mid": -0.05, "outside": -0.10}},
            "8f+": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": 0.00, "S": -0.10},
                    "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05}}
        }
    },
    "Gulfstream": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Synthetic": {
            "≤6f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        }
    },
    "Tampa Bay Downs": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Belmont Park": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.15, "E/P": 0.05, "P": 0.00, "S": -0.10},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Horseshoe Indianapolis": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}}
        }
    },
    "Penn National": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        }
    },
    "Presque Isle Downs": {
        "Synthetic": {
            "≤6f": {"runstyle": {"E": 0.15, "E/P": 0.10, "P": 0.00, "S": -0.10},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.05, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}}
        }
    },
    "Woodbine": {
        "Synthetic": {
            "≤6f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Evangeline Downs": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}}
        }
    },
    "Fairmount Park": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}}
        }
    },
    "Finger Lakes": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                    "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+": {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                    "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}}
        }
    },
    # Default fallback profile for tracks not specifically listed
    "_DEFAULT": {
        "Dirt": {
            "≤6f": {"runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                    "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.08, "E/P": 0.05, "P": 0.00, "S": -0.05},
                       "post": {"rail": 0.03, "inner": 0.03, "mid": 0.00, "outside": -0.03}},
            "8f+": {"runstyle": {"E": 0.03, "E/P": 0.03, "P": 0.03, "S": -0.03},
                    "post": {"rail": 0.03, "inner": 0.03, "mid": 0.00, "outside": -0.03}}
        },
        "Turf": {
            "≤6f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.08},
                    "post": {"rail": 0.00, "inner": 0.03, "mid": 0.00, "outside": -0.03}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.03, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": -0.03}},
            "8f+": {"runstyle": {"E": 0.00, "E/P": 0.03, "P": 0.05, "S": -0.03},
                    "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": -0.03}}
        },
        "Synthetic": {
            "≤6f": {"runstyle": {"E": 0.08, "E/P": 0.05, "P": 0.03, "S": -0.05},
                    "post": {"rail": 0.00, "inner": 0.00, "mid": 0.03, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.03, "S": -0.05},
                       "post": {"rail": 0.00, "inner": 0.00, "mid": 0.03, "outside": 0.00}},
            "8f+": {"runstyle": {"E": 0.03, "E/P": 0.05, "P": 0.05, "S": -0.03},
                    "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": 0.00}}
        }
    }
}


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


def _post_bucket(post_str: str) -> str:
    try:
        post = int(re.sub(r"[^\d]", "", str(post_str)))
    except Exception as e:
        st.warning(f"Failed to parse post number: '{post_str}'. Error: {e}")
        post = None  # Default to None and let it become 'mid'
    if post is None:
        return "mid"
    if post == 1:
        return "rail"
    if 2 <= post <= 3:
        return "inner"
    if 4 <= post <= 7:
        return "mid"
    return "outside"


def _style_norm(style: str) -> str:
    s = (style or "NA").upper()
    return "E/P" if s in ("EP", "E/P") else s


def _get_track_bias_delta(track_name: str, surface_type: str, distance_txt: str,
                          style: str, post_str: str) -> float:
    canon = _canonical_track(track_name)
    surf = (surface_type or "Dirt").strip().title()
    buck = distance_bucket(distance_txt)  # ≤6f / 6.5–7f / 8f+

    # Try to get specific track profile first
    cfg = (TRACK_BIAS_PROFILES.get(canon, {})
           .get(surf, {})
           .get(buck, {}))

    # If no specific profile found, use default fallback
    if not cfg:
        cfg = (TRACK_BIAS_PROFILES.get("_DEFAULT", {})
                                  .get(surf, {})
                                  .get(buck, {}))

    # If still no config (shouldn't happen with default), return 0
    if not cfg:
        return 0.0

    s_norm = _style_norm(style)
    runstyle_delta = float((cfg.get("runstyle", {}) or {}).get(s_norm, 0.0))
    post_delta = float((cfg.get("post", {}) or {}).get(_post_bucket(post_str), 0.0))
    return float(np.clip(runstyle_delta + post_delta, -1.0, 1.0))

# ===================== Core Helpers =====================


def detect_valid_race_headers(pp_text: str):
    toks = ("purse", "furlong", "mile", "clm", "allow", "stake", "pars", "post time")
    headers = []
    for m in re.finditer(r"(?mi)^\s*Race\s+(\d+)\b", pp_text or ""):
        win = (pp_text[m.end():m.end() + 250] or "").lower()
        if any(t in win for t in toks):
            headers.append(int(m.group(1)))
    return headers


HORSE_HDR_RE = re.compile(
    r"""(?mi)^\s*
    (\d+)              # post/program
    \s+([A-Za-z0-9'.\-\s&]+?)   # horse name
    \s*\(\s*
    (E\/P|EP|E|P|S|NA)      # style
    (?:\s+(\d+))?           # optional quirin
    \s*\)\s*$              #
    """, re.VERBOSE
)


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


def calculate_style_strength(style: str, quirin: float) -> str:
    s = (style or "NA").upper()
    try:
        q = float(quirin)
    except Exception:
        return "Solid"
    if pd.isna(q):
        return "Solid"
    if s in ("E", "E/P"):
        if q >= 7:
            return "Strong"
        if q >= 5:
            return "Solid"
        if q >= 3:
            return "Slight"
        return "Weak"
    if s in ("P", "S"):
        if q >= 5:
            return "Slight"
        if q >= 3:
            return "Solid"
        return "Strong"
    return "Solid"


def split_into_horse_chunks(pp_text: str) -> List[tuple]:
    chunks = []
    matches = list(HORSE_HDR_RE.finditer(pp_text or ""))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(pp_text)
        post = m.group(1).strip()
        name = m.group(2).strip()
        block = pp_text[start:end]
        chunks.append((post, name, block))
    return chunks


def extract_horses_and_styles(pp_text: str) -> pd.DataFrame:
    rows = []
    for m in HORSE_HDR_RE.finditer(pp_text or ""):
        post = m.group(1).strip()
        name = m.group(2).strip()
        style = _normalize_style(m.group(3))
        qpts = m.group(4)
        quirin = int(qpts) if qpts else np.nan
        auto_strength = calculate_style_strength(style, quirin)
        rows.append({
            "#": post, "Post": post, "Horse": name, "DetectedStyle": style,
            "Quirin": quirin, "AutoStrength": auto_strength,
            "OverrideStyle": "", "StyleStrength": auto_strength
        })
    seen = set()
    uniq = []
    for r in rows:
        key = (r["#"], r["Horse"].lower())
        if key not in seen:
            seen.add(key)
            uniq.append(r)
    df = pd.DataFrame(uniq)
    if not df.empty:
        df["Quirin"] = df["Quirin"].clip(lower=0, upper=8)
    return df


# === FIX 2: Re-ordered regex to catch fractions first ===
_ODDS_TOKEN = r"(\d+\s*/\s*\d+|\d+\s*-\s*\d+|[+-]?\d+(?:\.\d+)?)"
# ========================================================


def extract_morning_line_by_horse(pp_text: str) -> Dict[str, str]:
    ml = {}
    blocks = {name: block for _, name, block in split_into_horse_chunks(pp_text)}
    for name, block in blocks.items():
        if name in ml:
            continue
        m_start = re.search(rf"(?mi)^\s*{_ODDS_TOKEN}", block or "")
        if m_start:
            ml[name.strip()] = m_start.group(1).replace(" ", "")
            continue
        m_labeled = re.search(rf"(?mi)^.*?\b(?:M/?L|Morning\s*Line|ML)\b.*?{_ODDS_TOKEN}", block or "")
        if m_labeled:
            ml[name.strip()] = m_labeled.group(1).replace(" ", "")
    return ml


ANGLE_LINE_RE = re.compile(
    r'(?mi)^\s*(\d{4}\s+)?(1st\s*time\s*str|Debut\s*MdnSpWt|Maiden\s*Sp\s*Wt|2nd\s*career\s*race|Turf\s*to\s*Dirt|Dirt\s*to\s*Turf|Shipper|Blinkers\s*(?:on|off)|(?:\d+(?:-\d+)?)\s*days?Away|JKYw/\s*Sprints|JKYw/\s*Trn\s*L(?:30|45|60)\b|JKYw/\s*[EPS]|JKYw/\s*NA\s*types)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$'
)


def parse_angles_for_block(block) -> pd.DataFrame:
    rows = []
    if not block:
        return pd.DataFrame(rows)
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in ANGLE_LINE_RE.finditer(block_str):
        _yr, cat, starts, win, itm, roi = m.groups()
        rows.append({"Category": re.sub(r"\s+", " ", cat.strip()),
                     "Starts": int(starts), "Win%": float(win),
                     "ITM%": float(itm), "ROI": float(roi)})
    return pd.DataFrame(rows)


def parse_pedigree_snips(block) -> dict:
    out = {"sire_awd": np.nan, "sire_1st": np.nan,
           "damsire_awd": np.nan, "damsire_1st": np.nan,
           "dam_dpi": np.nan}
    if not block:
        return out
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    s = re.search(r'(?mi)^\s*Sire\s*Stats:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi', block_str)
    if s:
        out["sire_awd"] = float(s.group(1))
        out["sire_1st"] = float(s.group(3))
    ds = re.search(r'(?mi)^\s*Dam\'s Sire:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi', block_str)
    if ds:
        out["damsire_awd"] = float(ds.group(1))
        out["damsire_1st"] = float(ds.group(3))
    d = re.search(r'(?mi)^\s*Dam:\s*DPI\s*(\d+(?:\.\d+)?)\s+(\d+)%', block_str)
    if d:
        out["dam_dpi"] = float(d.group(1))
    return out

# ========== SAVANT-LEVEL ENHANCEMENTS (Jan 2026) ==========


def parse_claiming_prices(block) -> List[int]:
    """Extract claiming prices from race lines. Returns list of prices (most recent first)."""
    prices = []
    if not block:
        return prices
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in re.finditer(r'Clm\s+(\d+)', block_str):
        try:
            prices.append(int(m.group(1)))
        except BaseException:
            pass
    return prices[:5]


def analyze_claiming_price_movement(recent_prices: List[int], today_price: int) -> float:
    """SAVANT ANGLE: Claiming price class movement. Returns bonus from -0.10 to +0.15"""
    bonus = 0.0
    if not recent_prices or today_price <= 0:
        return bonus
    avg_recent = np.mean(recent_prices[:3]) if len(recent_prices) >= 3 else recent_prices[0]
    if avg_recent > today_price * 1.3:
        bonus += 0.15  # Big drop = intent to win
    elif avg_recent > today_price * 1.15:
        bonus += 0.08
    elif today_price > avg_recent * 1.3:
        bonus -= 0.10  # Rising in class
    elif today_price > avg_recent * 1.15:
        bonus -= 0.05
    return bonus


def detect_lasix_change(block) -> float:
    """SAVANT ANGLE: Lasix/medication changes. Returns bonus from -0.12 to +0.18"""
    bonus = 0.0
    if not block:
        return bonus
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    race_lines = [line for line in block_str.split('\n') if re.search(r'\d{2}[A-Za-z]{3}\d{2}', line)]
    lasix_pattern = []
    for line in race_lines[:5]:
        if re.search(r'\s+L\s+\d+\.\d+\s*$', line):
            lasix_pattern.append(True)
        else:
            lasix_pattern.append(False)
    # CRITICAL FIX: Validate list has at least 2 elements before accessing indices
    if len(lasix_pattern) >= 2:
        if lasix_pattern[0] and not any(lasix_pattern[1:]):
            bonus += 0.18  # First-time Lasix = major boost
        elif not lasix_pattern[0] and lasix_pattern[1]:
            bonus -= 0.12  # Lasix off = red flag
        elif lasix_pattern[0] and sum(lasix_pattern) >= 3:
            bonus += 0.02  # Consistent user
    return bonus


def parse_fractional_positions(block) -> List[List[int]]:
    """Extract running positions: PP, Start, 1C, 2C, Stretch, Finish."""
    positions = []
    if not block:
        return positions
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    # More flexible pattern with unicode range for position markers (¹²³ª etc)
    pattern = r'(\d{2}[A-Za-z]{3}\d{2}).*?(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})'
    for m in re.finditer(pattern, block_str, re.MULTILINE):
        try:
            # Validate each group exists and contains digits before conversion
            pos = []
            for i in range(2, 8):
                group_val = m.group(i)
                if group_val and group_val.isdigit():
                    pos.append(int(group_val))
                else:
                    # Invalid group - skip this match
                    raise ValueError(f"Invalid position group {i}: {group_val}")
            if len(pos) == 6:  # Ensure we got all 6 positions
                positions.append(pos)
        except (ValueError, IndexError) as e:
            # Log parsing failures for debugging (silent failures hide bugs)
            pass  # Could add logging here if needed
    return positions[:5]


def calculate_trip_quality(positions: List[List[int]], field_size: int = 10) -> float:
    """SAVANT ANGLE: Trip handicapping. Returns bonus from -0.04 to +0.12"""
    bonus = 0.0
    # CRITICAL FIX: Check positions exists AND has elements before accessing
    if not positions or len(positions) == 0 or len(positions[0]) < 6:
        return bonus
    pp, st, c1, c2, stretch, finish = positions[0]
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


def parse_e1_e2_lp_values(block) -> dict:
    """Extract E1, E2, and LP pace figures."""
    e1_vals, e2_vals, lp_vals = [], [], []
    if not block:
        return {'e1': e1_vals, 'e2': e2_vals, 'lp': lp_vals}
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in re.finditer(r'(\d{2,3})\s+(\d{2,3})/\s*(\d{2,3})', block_str):
        try:
            e1_vals.append(int(m.group(1)))
            e2_vals.append(int(m.group(2)))
            lp_vals.append(int(m.group(3)))
        except (ValueError, AttributeError, IndexError):
            # Regex group missing or conversion failed
            pass
    return {'e1': e1_vals[:5], 'e2': e2_vals[:5], 'lp': lp_vals[:5]}


def analyze_pace_figures(e1_vals: List[int], e2_vals: List[int], lp_vals: List[int]) -> float:
    """SAVANT ANGLE: E1/E2/LP pace analysis. Returns bonus from -0.05 to +0.07"""
    bonus = 0.0
    if len(e1_vals) < 3 or len(lp_vals) < 3:
        return bonus
    avg_e1 = np.mean(e1_vals[:3])
    avg_lp = np.mean(lp_vals[:3])
    if avg_lp > avg_e1 + 5:
        bonus += 0.07  # Closer with gas
    if avg_e1 >= 95 and avg_lp >= 85:
        bonus += 0.06  # Speed + stamina
    if avg_e1 >= 90 and avg_lp < 75:
        bonus -= 0.05  # Speed no stamina
    if len(e2_vals) >= 3:
        avg_e2 = np.mean(e2_vals[:3])
        if abs(avg_e1 - avg_e2) <= 3 and abs(avg_e2 - avg_lp) <= 3:
            bonus += 0.04  # Balanced energy
    return bonus


def detect_bounce_risk(speed_figs: List[int]) -> float:
    """SAVANT ANGLE: Bounce detection. Returns penalty/bonus from -0.09 to +0.07"""
    penalty = 0.0
    if len(speed_figs) < 3:
        return penalty
    last_three = speed_figs[:3]
    career_best = max(speed_figs) if speed_figs else 0
    if last_three[0] == career_best and len(speed_figs) > 3:
        if last_three[1] < last_three[0] - 8:
            penalty -= 0.09  # Bounce pattern
        elif last_three[1] < last_three[0] - 5:
            penalty -= 0.05
    if len(speed_figs) >= 4 and last_three[0] >= career_best - 2 and last_three[1] >= career_best - 2:
        penalty += 0.07  # Peak form maintained
    if last_three[0] > last_three[1] > last_three[2]:
        penalty += 0.06  # Improving
    if last_three[0] < last_three[1] < last_three[2]:
        penalty -= 0.05  # Declining
    if max(last_three) - min(last_three) <= 5:
        penalty += 0.03  # Consistent
    return penalty

# ========== END SAVANT ENHANCEMENTS ==========


SPEED_FIG_RE = re.compile(
    # CRITICAL FIX: Match the actual BRISNET format with E1 E2/ LP +calls SPEED_FIG
    # Example: "88 81/ 77 +4 +1 70"  where 70 is the speed figure
    r"(?mi)"
    r"\s+(\d{2,3})"                    # E1 pace figure (88)
    r"\s+(\d{2,3})\s*/\s*(\d{2,3})"   # E2/LP figures (81/ 77)
    r"\s+[+-]\d+"                      # Call position (+4)
    r"\s+[+-]\d+"                      # Call position (+1)
    r"\s+(\d{2,3})"                    # SPEED FIGURE (70) - THIS IS WHAT WE WANT!
    r"(?:\s|$)"                        # End with space or end of line
)


def parse_speed_figures_for_block(block) -> List[int]:
    """
    Parses a horse's PP text block and extracts all main speed figures.
    CRITICAL: Speed figure is in the 4th capture group after E1, E2, LP
    """
    figs = []
    if not block:
        return figs

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    for m in SPEED_FIG_RE.finditer(block_str):
        try:
            # The speed figure is the FOURTH capture group (after E1, E2, LP)
            fig_val = int(m.group(4))
            # Basic sanity check for a realistic speed figure
            if 40 < fig_val < 130:
                figs.append(fig_val)
        except (ValueError, AttributeError, IndexError):
            # Ignore if regex group missing or conversion fails
            pass

    # We only care about the most recent figs, e.g., last 10
    return figs[:10]

# ---------- Helper Functions ----------


def normalize_horse_name(name):
    """Normalize horse name for matching: remove apostrophes, extra spaces, lowercase.

    Use this function consistently throughout the codebase for horse name matching
    to avoid scratched horses appearing or missing matches.

    Args:
        name: Horse name to normalize

    Returns:
        str: Normalized name (lowercase, no apostrophes/backticks, single spaces)
    """
    return ' '.join(str(name).replace("'", "").replace("`", "").lower().split())


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
            value = value.strip().rstrip('%')
            # Remove + symbol from odds like '+150'
            if value.startswith('+'):
                value = value[1:]
        return float(value)
    except (ValueError, TypeError, AttributeError):
        return default

# ---------- GOLD-STANDARD Probability helpers with mathematical rigor ----------


def softmax_from_rating(ratings: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
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
    _tau = tau if tau is not None else MODEL_CONFIG['softmax_tau']
    _tau = max(_tau, 1e-6)  # Prevent division by zero
    _tau = min(_tau, 1e6)   # Prevent numerical instability

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
        stl = row.get("Style") or row.get("OverrideStyle") or row.get("DetectedStyle") or ""
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
    ppi_multiplier = MODEL_CONFIG.get('ppi_multiplier', 1.0)
    ppi_numerator = (counts["E"] + counts["E/P"] - counts["P"] - counts["S"])
    ppi_val = (ppi_numerator * ppi_multiplier) / total

    # VALIDATION: Ensure PPI is finite
    if not np.isfinite(ppi_val):
        ppi_val = 0.0

    # COMPUTE: Per-horse tailwinds with validation
    by_horse = {}
    strength_weights = MODEL_CONFIG.get('style_strength_weights', {})
    tailwind_factor = MODEL_CONFIG.get('ppi_tailwind_factor', 1.0)

    # VALIDATION: Ensure factors are finite
    if not np.isfinite(tailwind_factor):
        tailwind_factor = 1.0

    for stl, nm, strength in zip(styles, names, strengths):
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


def apply_enhancements_and_figs(ratings_df: pd.DataFrame, pp_text: str, processed_weights: Dict[str, float],
                                chaos_index: float, track_name: str, surface_type: str,
                                distance_txt: str, race_type: str,
                                angles_per_horse: Dict[str, pd.DataFrame],
                                pedigree_per_horse: Dict[str, dict], figs_df: pd.DataFrame) -> pd.DataFrame:
    """
    Applies speed figure enhancements (R_ENHANCE_ADJ) to the base ratings.
    """
    if ratings_df is None or ratings_df.empty:
        return ratings_df

    df = ratings_df.copy()

    # --- SPEED FIGURE LOGIC ---
    if figs_df.empty or "AvgTop2" not in figs_df.columns:
        # No figures were parsed or dataframe is empty
        st.caption("No speed figures parsed. R_ENHANCE_ADJ set to 0.")
        df["R_ENHANCE_ADJ"] = 0.0
    else:
        # 1. Merge the figures into the main ratings dataframe
        df = df.merge(figs_df[["Horse", "AvgTop2"]], on="Horse", how="left")

        # 2. Calculate the average "AvgTop2" for all horses *in this race*
        # We fillna with a low value for first-timers
        df["AvgTop2"].fillna(MODEL_CONFIG['first_timer_fig_default'], inplace=True)
        race_avg_fig = df["AvgTop2"].mean()

        # 3. Define the enhancement (R_ENHANCE_ADJ)
        SPEED_FIG_WEIGHT = MODEL_CONFIG['speed_fig_weight']

        df["R_ENHANCE_ADJ"] = (df["AvgTop2"] - race_avg_fig) * SPEED_FIG_WEIGHT

        # 4. Clean up the temporary column
        df.drop(columns=["AvgTop2"], inplace=True)

    # --- END SPEED FIGURE LOGIC ---

    # --- ANGLES BONUS LOGIC ---
    # Add bonus for horses with positive angles
    ANGLE_BONUS = 0.10  # Bonus per positive angle
    df["AngleBonus"] = 0.0
    for horse, angles_df in angles_per_horse.items():
        if angles_df is not None and not angles_df.empty:
            # Count positive angles (rows in the dataframe)
            num_angles = len(angles_df)
            if num_angles > 0:
                bonus = num_angles * ANGLE_BONUS
                df.loc[df["Horse"] == horse, "AngleBonus"] = bonus

    df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"] + df["AngleBonus"]
    df.drop(columns=["AngleBonus"], inplace=True)
    # --- END ANGLES LOGIC ---

    # --- SAVANT ENHANCEMENT LOGIC (Jan 2026) ---
    # NEW: Lasix, Trip Handicapping, Workout Patterns, Pace Figures, Bounce Detection
    df["SavantBonus"] = 0.0

    # Extract horse blocks for savant analysis
    horse_blocks = split_into_horse_chunks(pp_text)
    block_map = {}
    for i, name in enumerate(df["Horse"]):
        if i < len(horse_blocks):
            block_map[name] = horse_blocks[i]

    for horse_name, block in block_map.items():
        savant_bonus = 0.0

        # 1. Lasix detection
        savant_bonus += detect_lasix_change(block)

        # 2. Trip handicapping
        positions = parse_fractional_positions(block)
        savant_bonus += calculate_trip_quality(positions, field_size=len(df))

        # 3. E1/E2/LP pace analysis
        pace_data = parse_e1_e2_lp_values(block)
        savant_bonus += analyze_pace_figures(pace_data['e1'], pace_data['e2'], pace_data['lp'])

        # 4. Bounce detection
        speed_figs = parse_speed_figures_for_block(block)
        savant_bonus += detect_bounce_risk(speed_figs)

        # 5. Workout pattern bonus (from enhanced workout parsing)
        workout_data = parse_workout_data(block)
        savant_bonus += workout_data.get('pattern_bonus', 0.0)

        df.loc[df["Horse"] == horse_name, "SavantBonus"] = savant_bonus

    df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"] + df["SavantBonus"]
    df.drop(columns=["SavantBonus"], inplace=True)
    # --- END SAVANT LOGIC ---

    # Apply the final adjustment
    df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"].fillna(0.0)  # Ensure no NaNs
    df["R"] = (df["R"].astype(float) + df["R_ENHANCE_ADJ"].astype(float)).round(2)

    return df

# ---------- Odds helpers ----------


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


def str_to_decimal_odds(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s:
        return None
    try:
        if re.fullmatch(r'[+-]?\d+(\.\d+)?', s):
            v = float(s)
            return max(v, 1.01)
        if re.fullmatch(r'\+\d+', s) or re.fullmatch(r'-\d+', s):
            return 1 + (float(s) / 100.0 if float(s) > 0 else 100.0 / abs(float(s)))
        if "-" in s:
            a, b = s.split("-", 1)
            return float(a) / float(b) + 1.0
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b) + 1.0
    except Exception as e:
        st.warning(f"Could not parse odds string: '{s}'. Error: {e}")
        return None
    return None


def overlay_table(fair_probs: Dict[str, float], offered: Dict[str, float]) -> pd.DataFrame:
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

        rows.append({
            "Horse": h,
            "Fair %": round(p * 100, 2),
            "Fair (AM)": fair_to_american_str(p),
            "Board (dec)": round(off_dec, 3),
            "Board %": round(off_prob * 100, 2),
            "Edge (pp)": round((p - off_prob) * 100, 2),
            "EV per $1": round(ev, 3),
            "Overlay?": "YES" if off_prob < p else "NO"
        })

    return pd.DataFrame(rows)

# ---------- Exotics (Harville + bias anchors) ----------


def calculate_exotics_biased(fair_probs: Dict[str, float],
                             anchor_first: Optional[str] = None,
                             anchor_second: Optional[str] = None,
                             pool_third: Optional[set] = None,
                             pool_fourth: Optional[set] = None,
                             weights=MODEL_CONFIG['exotic_bias_weights'],
                             top_n: int = 50) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    horses = list(fair_probs.keys())
    probs = np.array([fair_probs[h] for h in horses])
    n = len(horses)
    if n < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def w_first(h): return weights[0] if anchor_first and h == anchor_first else 1.0
    def w_second(h): return weights[1] if anchor_second and h == anchor_second else 1.0
    def w_third(h): return weights[2] if pool_third and h in pool_third else 1.0
    def w_fourth(h): return weights[3] if pool_fourth and h in pool_fourth else 1.0
    # Add a 5th weight for SH5, re-using 4th
    def w_fifth(h): return weights[3] if pool_fourth and h in pool_fourth else 1.0

    # EXACTA
    ex_rows = []
    for i, j in product(range(n), range(n)):
        if i == j:
            continue
        denom_ex = 1.0 - probs[i]
        if denom_ex <= 1e-9:
            continue
        prob = probs[i] * (probs[j] / denom_ex)
        prob *= w_first(horses[i]) * w_second(horses[j])
        ex_rows.append({"Ticket": f"{horses[i]} → {horses[j]}", "Prob": prob})

    # CRITICAL FIX: Validate probability sum before normalization
    ex_total = sum(r["Prob"] for r in ex_rows)
    if ex_total <= 1e-9:  # If sum is essentially zero, return empty DataFrame
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()
    for r in ex_rows:
        r["Prob"] = r["Prob"] / ex_total
    for r in ex_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0 / r["Prob"]) - 1
        else:
            r["Fair Odds"] = float('inf')
    df_ex = pd.DataFrame(ex_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 3:
        return df_ex, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # TRIFECTA
    tri_rows = []
    top8 = np.argsort(-probs)[:min(n, 8)]
    for i, j, k in product(top8, top8, top8):
        if len({i, j, k}) != 3:
            continue
        denom_ij = 1.0 - probs[i]
        denom_ijk = 1.0 - probs[i] - probs[j]
        if denom_ij <= 1e-9 or denom_ijk <= 1e-9:
            continue

        p_ij = probs[i] * (probs[j] / denom_ij)
        prob_ijk = p_ij * (probs[k] / denom_ijk)
        prob_ijk *= w_first(horses[i]) * w_second(horses[j]) * w_third(horses[k])
        tri_rows.append({"Ticket": f"{horses[i]} → {horses[j]} → {horses[k]}", "Prob": prob_ijk})

    tri_total = sum(r["Prob"] for r in tri_rows) or 1.0
    for r in tri_rows:
        r["Prob"] = r["Prob"] / tri_total
    for r in tri_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0 / r["Prob"]) - 1
        else:
            r["Fair Odds"] = float('inf')
    df_tri = pd.DataFrame(tri_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 4:
        return df_ex, df_tri, pd.DataFrame(), pd.DataFrame()

    # SUPERFECTA
    super_rows = []
    top6 = np.argsort(-probs)[:min(n, 6)]  # Keep this at top6 for performance
    for i, j, k, l in product(top6, top6, top6, top6):
        if len({i, j, k, l}) != 4:
            continue
        denom_ij = 1.0 - probs[i]
        denom_ijk = 1.0 - probs[i] - probs[j]
        denom_ijkl = 1.0 - probs[i] - probs[j] - probs[k]
        if denom_ij <= 1e-9 or denom_ijk <= 1e-9 or denom_ijkl <= 1e-9:
            continue

        p_ij = probs[i] * (probs[j] / denom_ij)
        p_ijk = p_ij * (probs[k] / denom_ijk)
        prob_ijkl = p_ijk * (probs[l] / denom_ijkl)
        prob_ijkl *= w_first(horses[i]) * w_second(horses[j]) * w_third(horses[k]) * w_fourth(horses[l])
        super_rows.append({"Ticket": f"{horses[i]} → {horses[j]} → {horses[k]} → {horses[l]}", "Prob": prob_ijkl})

    super_total = sum(r["Prob"] for r in super_rows) or 1.0
    for r in super_rows:
        r["Prob"] = r["Prob"] / super_total
    for r in super_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0 / r["Prob"]) - 1
        else:
            r["Fair Odds"] = float('inf')
    df_super = pd.DataFrame(super_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 5:
        return df_ex, df_tri, df_super, pd.DataFrame()

    # --- NEW: SUPER HIGH 5 ---
    sh5_rows = []
    top7 = np.argsort(-probs)[:min(n, 7)]  # Use Top 7 for SH5
    for i, j, k, l, m in product(top7, top7, top7, top7, top7):
        if len({i, j, k, l, m}) != 5:
            continue
        denom_ij = 1.0 - probs[i]
        denom_ijk = 1.0 - probs[i] - probs[j]
        denom_ijkl = 1.0 - probs[i] - probs[j] - probs[k]
        denom_ijklm = 1.0 - probs[i] - probs[j] - probs[k] - probs[l]
        if denom_ij <= 1e-9 or denom_ijk <= 1e-9 or denom_ijkl <= 1e-9 or denom_ijklm <= 1e-9:
            continue

        p_ij = probs[i] * (probs[j] / denom_ij)
        p_ijk = p_ij * (probs[k] / denom_ijk)
        p_ijkl = p_ijk * (probs[l] / denom_ijkl)
        prob_ijklm = p_ijkl * (probs[m] / denom_ijklm)

        prob_ijklm *= (w_first(horses[i]) *
                       w_second(horses[j]) *
                       w_third(horses[k]) *
                       w_fourth(horses[l]) *
                       w_fifth(horses[m]))

        sh5_rows.append(
            {
                "Ticket": f"{horses[i]} → {horses[j]} → {horses[k]} → {horses[l]} → {horses[m]}",
                "Prob": prob_ijklm})

    sh5_total = sum(r["Prob"] for r in sh5_rows) or 1.0
    for r in sh5_rows:
        r["Prob"] = r["Prob"] / sh5_total
    for r in sh5_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0 / r["Prob"]) - 1
        else:
            r["Fair Odds"] = float('inf')
    df_super_hi_5 = pd.DataFrame(sh5_rows).sort_values(by="Prob", ascending=False).head(top_n)

    return df_ex, df_tri, df_super, df_super_hi_5


def format_exotics_for_prompt(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"**{title} (Model-Derived)**\nNone.\n"
    df = df.copy()
    if "Prob %" not in df.columns:
        df["Prob %"] = (df["Prob"] * 100).round(2)
    # Format Fair Odds to handle potential infinity
    df["Fair Odds"] = df["Fair Odds"].apply(lambda x: f"{x:.2f}" if np.isfinite(x) else "in")
    md = df[["Ticket", "Prob %", "Fair Odds"]].to_markdown(index=False)
    return f"**{title} (Model-Derived)**\n{md}\n"

# -------- Class + suitability model --------


def calculate_final_rating(race_type,
                           race_surface,
                           race_distance_category,
                           race_surface_condition,
                           horse_surface_pref,
                           horse_distance_pref):
    """
    Calculates a final "rating" scalar (lower is better) by combining:
    base class bias × surface fit × distance fit × condition variance.
    """
    base_bias = base_class_bias.get(str(race_type).strip().lower(), 1.10)

    # Surface fit
    surface_modifier = 1.0
    if str(horse_surface_pref).lower() == "any":
        surface_modifier = 1.01
    elif race_surface.lower() != str(horse_surface_pref).lower():
        surface_modifier = 1.12

    # Distance fit
    distance_modifier = 1.0
    if str(horse_distance_pref).lower() == "any":
        distance_modifier = 1.02
    elif race_distance_category != horse_distance_pref:
        distance_modifier = 1.15

    # Condition variance (confidence)
    condition_modifier = condition_modifiers.get(str(race_surface_condition).lower(), 1.0)

    final_score = base_bias * surface_modifier * distance_modifier * condition_modifier
    return round(final_score, 4)

# ===================== Form Cycle & Recency Analysis =====================


def parse_recent_races_detailed(block) -> List[dict]:
    """
    Extract detailed recent race history with dates, finishes, beaten lengths.
    Returns list of dicts with date, finish, beaten_lengths, days_ago
    """
    races = []
    if not block:
        return races
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    # Pattern: date, finish position, beaten lengths
    # Example: "23Dec23 Aqu 3rd 2¼"
    pattern = r'(\d{2}[A-Za-z]{3}\d{2})\s+\w+.*?(\d+)(?:st|nd|rd|th)\s*(\d+)?'

    today = datetime.now()

    for match in re.finditer(pattern, block_str):
        date_str = match.group(1)
        finish = match.group(2)

        try:
            # Parse date
            race_date = datetime.strptime(date_str, '%d%b%y')
            days_ago = (today - race_date).days

            races.append({
                'date': race_date,
                'days_ago': days_ago,
                'finish': int(finish) if finish.isdigit() else 99
            })
        except Exception:
            pass

    return sorted(races, key=lambda x: x['days_ago'])[:6]  # Last 6 races, most recent first


def calculate_layoff_factor(days_since_last: int) -> float:
    """
    Layoff impact on performance.
    Returns: adjustment factor (-5.0 to +0.5)
    PEGASUS WC TUNING: Increased penalties for long layoffs (146-day layoff issue)
    """
    if days_since_last <= 14:  # Racing frequently (good)
        return 0.5
    elif days_since_last <= 30:  # Fresh, ideal
        return 0.3
    elif days_since_last <= 45:  # Standard rest
        return 0.0
    elif days_since_last <= 60:  # Slight concern
        return -0.3
    elif days_since_last <= 90:  # Moderate layoff
        return -0.8
    elif days_since_last <= 120:  # Long layoff - TUNED
        return -1.5
    elif days_since_last <= 180:  # Extended layoff - TUNED
        return -3.0
    else:  # Very long absence - TUNED
        return -5.0


def calculate_form_trend(recent_finishes: List[int]) -> float:
    """
    Analyze finish positions for improvement/decline trend.
    Returns: trend factor (-1.5 to +4.0)
    PEGASUS WC TUNING: Added win momentum bonus (Skippylongstocking won last race)
    """
    if len(recent_finishes) < 1:
        return 0.0

    # PEGASUS TUNING: Win momentum bonus
    if len(recent_finishes) >= 1 and recent_finishes[0] == 1:
        # Won most recent race
        if len(recent_finishes) >= 2 and recent_finishes[1] == 1:
            # Won last 2 races - hot streak
            return 4.0
        else:
            # Won last race only
            return 2.5
    elif len(recent_finishes) >= 1 and recent_finishes[0] in [2, 3]:
        # Place/Show last race - positive form
        return 1.0

    if len(recent_finishes) < 2:
        return 0.0

    # Weight recent races more heavily: [0.4, 0.3, 0.2, 0.1]
    weights = [0.4, 0.3, 0.2, 0.1][:len(recent_finishes)]

    # Calculate weighted average of recent finishes
    weighted_avg = sum(f * w for f, w in zip(recent_finishes, weights)) / sum(weights)

    # Check for improvement pattern (finishes getting better = lower numbers)
    if len(recent_finishes) >= 3:
        recent_3 = recent_finishes[:3]
        # CRITICAL FIX: Verify list has exactly 3 elements before comparison
        if len(recent_3) == 3:
            if recent_3[0] < recent_3[1] < recent_3[2]:  # Improving (3rd, 4th, 5th → getting better)
                return 1.5  # Strong improvement
            elif recent_3[0] > recent_3[1] > recent_3[2]:  # Declining
                return -1.2  # Declining form

    # Weighted average scoring
    if weighted_avg <= 1.5:  # Consistently winning/placing
        return 1.2
    elif weighted_avg <= 3.0:  # In the money regularly
        return 0.8
    elif weighted_avg <= 5.0:  # Mid-pack
        return 0.0
    elif weighted_avg <= 7.0:  # Back half
        return -0.5
    else:  # Consistently poor
        return -1.0


def parse_workout_data(block) -> dict:
    """
    ENHANCED: Extract workout information with pattern analysis.
    Returns dict with best_time, num_works, recency, pattern_bonus
    """
    workouts = {
        'best_time': None,
        'num_recent': 0,
        'days_since_last': 999,
        'pattern_bonus': 0.0
    }

    if not block:
        return workouts

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    # Enhanced pattern: captures bullet (×), distance, time, grade, rank
    pattern = r'([×]?)(\d{1,2}[A-Z][a-z]{2})\s+\w+\s+(\d+)f\s+\w+\s+([\d:.«©ª¬®¯°¨]+)\s+([HBG]g?)(?:\s+(\d+)/(\d+))?'

    work_details = []
    for match in re.finditer(pattern, block_str):
        try:
            bullet = match.group(1) == '×'
            distance = int(match.group(3))
            time_str = match.group(4)
            grade = match.group(5)
            rank = int(match.group(6)) if match.group(6) else None
            total = int(match.group(7)) if match.group(7) else None

            time_clean = re.sub(r'[«©ª¬®¯°¨]', '', time_str)
            # CRITICAL FIX: Wrap float conversions in try-except to prevent ValueError
            try:
                if ':' in time_clean:
                    parts = time_clean.split(':')
                    if len(parts) == 2:
                        time_seconds = float(parts[0]) * 60 + float(parts[1])
                    elif len(parts) > 0:
                        time_seconds = float(parts[-1])  # Use last part if malformed
                    else:
                        continue  # Skip this workout if no valid parts
                else:
                    time_seconds = float(time_clean)
            except (ValueError, IndexError):
                continue  # Skip workout if conversion fails

            # CRITICAL FIX: Validate distance > 0 before division
            normalized_time = time_seconds * (4.0 / distance) if distance > 0 else 999

            work_details.append({
                'bullet': bullet,
                'distance': distance,
                'time': normalized_time,
                'grade': grade,
                'rank': rank,
                'total': total
            })
        except BaseException:
            pass

    workouts['num_recent'] = len(work_details)

    if work_details:
        workouts['best_time'] = min(w['time'] for w in work_details)

        # SAVANT: Workout pattern analysis
        bonus = 0.0
        if len(work_details) >= 3:
            times = [w['time'] for w in work_details[:3]]
            if times[0] < times[1] < times[2]:
                bonus += 0.08  # Sharp pattern
            elif times[0] > times[1] > times[2]:
                bonus -= 0.06  # Dull pattern

        if work_details[0]['bullet']:
            bonus += 0.03  # Recent bullet

        # CRITICAL FIX: Validate total > 0 before division to prevent ZeroDivisionError
        if work_details[0]['rank'] and work_details[0]['total'] and work_details[0]['total'] > 0:
            percentile = work_details[0]['rank'] / work_details[0]['total']
            if percentile <= 0.2:
                bonus += 0.05  # Top 20% of field

        if 'g' in work_details[0]['grade'].lower():
            bonus += 0.03  # Gate work

        if work_details[0]['rank'] and work_details[0]['total']:
            percentile = work_details[0]['rank'] / work_details[0]['total']
            if percentile <= 0.25:
                bonus += 0.04  # Elite work
            elif percentile <= 0.50:
                bonus += 0.02

        bullet_count = sum(1 for w in work_details[:5] if w['bullet'])
        if bullet_count >= 2:
            bonus += 0.05  # Consistent quality

        workouts['pattern_bonus'] = bonus

    return workouts


def evaluate_first_time_starter(
    pedigree: dict,
    angles_df: pd.DataFrame,
    workout_data: dict,
    horse_block: str
) -> float:
    """
    Comprehensive first-time starter evaluation.
    Returns: debut rating from -2.0 to +3.5
    """
    debut_rating = 0.0

    # 1. PEDIGREE QUALITY (weight: heavy for debuts)
    sire_spi = pedigree.get('sire_spi')
    damsire_spi = pedigree.get('damsire_spi')

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
    sire_1st = pedigree.get('sire_1st')
    damsire_1st = pedigree.get('damsire_1st')

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
    if workout_data['num_recent'] >= 3:  # Well-prepared
        debut_rating += 0.4

        if workout_data['best_time'] is not None:
            # Fast workouts indicate readiness
            if workout_data['best_time'] < 48.0:  # Blazing 4f equivalent
                debut_rating += 0.6
            elif workout_data['best_time'] < 49.5:  # Very good
                debut_rating += 0.3
            elif workout_data['best_time'] < 51.0:  # Solid
                debut_rating += 0.1
    elif workout_data['num_recent'] < 2:  # Underprepared
        debut_rating -= 0.5

    # 3. TRAINER DEBUT ANGLES
    if angles_df is not None and not angles_df.empty:
        angle_text = ' '.join(angles_df['Category'].astype(str)).lower()

        if '1st time str' in angle_text or 'debut' in angle_text:
            debut_rating += 0.5  # Trainer pattern recognition

        if 'maiden sp wt' in angle_text or 'maiden special weight' in angle_text:
            debut_rating += 0.3  # MSW debut angle

        # High ROI trainer debut pattern
        if angles_df is not None and 'ROI' in angles_df.columns:
            debut_angles = angles_df[angles_df['Category'].str.contains('debut|1st time', case=False, na=False)]
            if not debut_angles.empty:
                avg_roi = debut_angles['ROI'].mean()
                if avg_roi > 1.5:  # Strong positive ROI
                    debut_rating += 0.8
                elif avg_roi > 1.0:
                    debut_rating += 0.4

    # 4. RACE TYPE CONTEXT
    if 'maiden special weight' in horse_block.lower():
        debut_rating += 0.2  # MSW is better spot than MCL for debuts
    elif 'maiden claiming' in horse_block.lower():
        debut_rating -= 0.3  # MCL debut is tougher

    return float(np.clip(debut_rating, -2.0, 3.5))


def calculate_form_cycle_rating(
    horse_block: str,
    pedigree: dict,
    angles_df: pd.DataFrame
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
        return evaluate_first_time_starter(pedigree, angles_df, workout_data, horse_block)

    # EXPERIENCED HORSE - analyze form cycle
    form_rating = 0.0

    # 1. Layoff factor
    days_since_last = recent_races[0]['days_ago'] if recent_races else 999
    # Use calculate_layoff_factor (defined earlier in file)
    layoff_adj = calculate_layoff_factor(days_since_last)
    form_rating += layoff_adj

    # 2. Form trend
    recent_finishes = [r['finish'] for r in recent_races]
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

    return float(np.clip(form_rating, -3.0, 3.0))

# ===================== Odds Conversion Utilities =====================


def odds_to_decimal(odds_str: str) -> float:
    """
    Convert various odds formats to decimal for comparison.
    Handles: fractional (5/2), decimal (3.5), American (+250, -150)
    Returns decimal odds (e.g., 5/2 → 3.5)
    """
    if not odds_str or odds_str == '?':
        return 0.0

    odds_str = str(odds_str).strip()

    try:
        # Fractional format: "5/2", "7/1"
        if '/' in odds_str:
            parts = odds_str.split('/')
            numerator = float(parts[0])
            denominator = float(parts[1])
            return (numerator / denominator) + 1.0

        # American format: "+250", "-150"
        elif odds_str.startswith(('+', '-')):
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

# ===================== Class Rating Calculator (Comprehensive) =====================


def extract_race_metadata_from_pp(pp_text: str) -> Dict[str, Any]:
    """
    🎯 ELITE EXTRACTION: Parse race type and purse from BRISNET PP text header.

    CRITICAL FOR CALIBRATION: All TUP R4/R5/R6/R7 fixes depend on correct race_quality.
    - Claiming: Speed 2.5x, pace cap +0.75
    - Allowance: Speed 2.2x, Class 2.5x
    - Stakes: Speed 1.8x, Class 3.0x

    BRISNET HEADER FORMATS:
    1. "PURSE $25,000. Claiming. For Three Year Olds..."
    2. "6th Race. Santa Anita. $50,000 Maiden Special Weight"
    3. "Race 4 - Clm25000n2L" (embedded in race type)
    4. "Turf Paradise Race 7 - $6,250 Claiming"

    Returns:
        dict: {
            'purse': int,
            'race_type_raw': str,  # Original text
            'race_type_normalized': str,  # "claiming", "allowance", etc.
            'confidence': float,  # 0.0-1.0
            'source': str  # Where data came from
        }
    """
    result = {
        'purse': 0,
        'race_type_raw': '',
        'race_type_normalized': 'unknown',
        'confidence': 0.0,
        'source': 'none'
    }

    if not pp_text or len(pp_text) < 50:
        return result

    # Extract first 500 chars (header section)
    header = pp_text[:500]

    # ═══════════════ PATTERN 1: "PURSE $X,XXX. Race Type." ═══════════════
    purse_match = re.search(r'PURSE\s+\$([\d,]+)', header, re.IGNORECASE)
    if purse_match:
        try:
            result['purse'] = int(purse_match.group(1).replace(',', ''))
            result['confidence'] += 0.5
            result['source'] = 'PURSE header'
        except ValueError:
            pass

    # Extract race type after PURSE line
    if purse_match:
        after_purse = header[purse_match.end():purse_match.end() + 100]
        # Look for race type keywords
        if re.search(r'\bClaiming\b', after_purse, re.IGNORECASE):
            result['race_type_raw'] = 'Claiming'
            result['race_type_normalized'] = 'claiming'
            result['confidence'] += 0.5
        elif re.search(r'\bAllowance\b', after_purse, re.IGNORECASE):
            result['race_type_raw'] = 'Allowance'
            result['race_type_normalized'] = 'allowance'
            result['confidence'] += 0.5
        elif re.search(r'\bMaiden\s+Special\s+Weight\b', after_purse, re.IGNORECASE):
            result['race_type_raw'] = 'Maiden Special Weight'
            result['race_type_normalized'] = 'maiden special weight'
            result['confidence'] += 0.5
        elif re.search(r'\bMaiden\s+Claiming\b', after_purse, re.IGNORECASE):
            result['race_type_raw'] = 'Maiden Claiming'
            result['race_type_normalized'] = 'maiden claiming'
            result['confidence'] += 0.5
        elif re.search(r'\b(Stakes?|G[123]|Grade)\b', after_purse, re.IGNORECASE):
            result['race_type_raw'] = 'Stakes'
            result['race_type_normalized'] = 'stakes'
            result['confidence'] += 0.5

    # ═══════════════ PATTERN 2: "$X,XXX Race Type" in any line ═══════════════
    if result['purse'] == 0:
        money_race_match = re.search(
            r'\$(\d{1,3}(?:,\d{3})*)\s+(Claiming|Allowance|Maiden|Stakes?)',
            header,
            re.IGNORECASE)
        if money_race_match:
            try:
                result['purse'] = int(money_race_match.group(1).replace(',', ''))
                result['race_type_raw'] = money_race_match.group(2)
                result['race_type_normalized'] = money_race_match.group(2).lower()
                result['confidence'] = 0.8
                result['source'] = '$X Race Type'
            except ValueError:
                pass

    # ═══════════════ PATTERN 3: Embedded in race type code ═══════════════
    if result['purse'] == 0:
        # Look for patterns like "Clm25000n2L", "MC50000", "Alw28000"
        embedded_match = re.search(r'(Clm|MC|Alw|OC)(\d{4,6})', header, re.IGNORECASE)
        if embedded_match:
            race_code = embedded_match.group(1).upper()
            purse_num = embedded_match.group(2)
            try:
                result['purse'] = int(purse_num)
                result['race_type_raw'] = f"{race_code}{purse_num}"
                result['source'] = 'embedded code'
                result['confidence'] = 0.6

                # Decode race type
                if race_code in ['CLM', 'MC']:
                    result['race_type_normalized'] = 'claiming'
                elif race_code in ['ALW', 'OC']:
                    result['race_type_normalized'] = 'allowance'
            except ValueError:
                pass

    # ═══════════════ PATTERN 4: "Race X - Race Type" ═══════════════
    if result['race_type_normalized'] == 'unknown':
        race_type_line = re.search(r'Race\s+\d+\s*-\s*([A-Za-z\s]+)', header, re.IGNORECASE)
        if race_type_line:
            race_text = race_type_line.group(1).strip().lower()
            if 'claim' in race_text:
                result['race_type_normalized'] = 'claiming'
                result['race_type_raw'] = race_type_line.group(1).strip()
                result['confidence'] += 0.3
            elif 'allow' in race_text:
                result['race_type_normalized'] = 'allowance'
                result['race_type_raw'] = race_type_line.group(1).strip()
                result['confidence'] += 0.3
            elif 'maiden' in race_text:
                result['race_type_normalized'] = 'maiden special weight'
                result['race_type_raw'] = race_type_line.group(1).strip()
                result['confidence'] += 0.3

    return result


def extract_race_metadata_from_pp_text(pp_text: str) -> Dict[str, Any]:
    """
    UNIVERSAL: Extract race type and purse from BRISNET PP text headers.
    Works across ALL tracks, ALL purse levels, ALL race types.

    Returns dict with:
    - purse_amount: int (extracted purse)
    - race_type: str (extracted race type)
    - race_type_clean: str (normalized: 'claiming', 'allowance', 'stakes', etc.)
    - confidence: float (0.0-1.0)
    - detection_method: str (how it was detected)

    Examples of BRISNET headers:
    - "PURSE $6,250. Claiming. For Three Year Olds..."
    - "Purse: $50,000 Allowance Optional Claiming"
    - "$100,000 Grade 2 Stakes"
    - "Race 7: Clm6250n3L"
    """
    if not pp_text or len(pp_text.strip()) < 50:
        return {
            'purse_amount': 0,
            'race_type': '',
            'race_type_clean': 'unknown',
            'confidence': 0.0,
            'detection_method': 'no_text'}

    result = {
        'purse_amount': 0,
        'race_type': '',
        'race_type_clean': 'unknown',
        'confidence': 0.0,
        'detection_method': 'none'
    }

    # Get first 800 characters (header region)
    header = pp_text[:800]

    # ========== PURSE EXTRACTION (Multi-Pattern) ==========
    purse_patterns = [
        r'PURSE\s+\$([\d,]+)',  # "PURSE $6,250"
        r'Purse:\s+\$([\d,]+)',  # "Purse: $50,000"
        r'\$([\d,]+)\s+(?:Grade|Stakes|Allowance|Claiming)',  # "$100,000 Grade 2"
        r'\$([\d,]+)',  # Any dollar amount in header
    ]

    for pattern in purse_patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            try:
                result['purse_amount'] = int(match.group(1).replace(',', ''))
                result['detection_method'] = 'purse_text'
                result['confidence'] = 0.9
                break
            except BaseException:
                pass

    # ========== RACE TYPE EXTRACTION (Multi-Pattern) ==========
    race_type_patterns = [
        # Graded Stakes
        (r'Grade\s+(I{1,3}|[123])', 'stakes_graded', 1.0),
        (r'G([123])', 'stakes_graded', 1.0),
        # Stakes
        (r'Stakes', 'stakes', 0.95),
        (r'Handicap', 'stakes', 0.9),
        (r'Listed', 'stakes', 0.9),
        # Allowance
        (r'Allowance\s+Optional\s+Claiming', 'allowance_optional', 0.95),
        (r'Optional\s+Claiming', 'allowance_optional', 0.95),
        (r'Allowance', 'allowance', 0.95),
        (r'\bAOC\b', 'allowance_optional', 0.9),
        (r'\bAlw\b', 'allowance', 0.9),
        # Claiming
        (r'Maiden\s+Claiming', 'maiden_claiming', 0.95),
        (r'Maiden\s+Clm', 'maiden_claiming', 0.9),
        (r'\bMCL\b', 'maiden_claiming', 0.9),
        (r'Claiming', 'claiming', 0.95),
        (r'\bClm\b', 'claiming', 0.9),
        (r'\bMC\b', 'claiming', 0.85),
        # Maiden
        (r'Maiden\s+Special\s+Weight', 'maiden_special_weight', 0.95),
        (r'\bMSW\b', 'maiden_special_weight', 0.9),
        (r'Maiden', 'maiden', 0.85),
        # Starter
        (r'Starter\s+Allowance', 'starter_allowance', 0.9),
        (r'Starter\s+Handicap', 'starter_handicap', 0.9),
        # Waiver
        (r'Waiver', 'waiver_claiming', 0.85),
    ]

    for pattern, race_type_clean, confidence in race_type_patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            result['race_type'] = match.group(0)
            result['race_type_clean'] = race_type_clean
            result['confidence'] = max(result['confidence'], confidence)
            if result['detection_method'] == 'none':
                result['detection_method'] = 'text_pattern'
            break

    # ========== EMBEDDED RACE TYPE (Clm25000n2L format) ==========
    embedded_pattern = r'\b(Clm|MC|OC|Alw|Mdn|MSW|Stk|G[123])([\d]+[kK]?)'
    embedded_match = re.search(embedded_pattern, header, re.IGNORECASE)
    if embedded_match:
        prefix = embedded_match.group(1).lower()
        amount_str = embedded_match.group(2)

        # Extract purse from embedded format if not already found
        if result['purse_amount'] == 0:
            try:
                if 'k' in amount_str.lower():
                    result['purse_amount'] = int(amount_str[:-1]) * 1000
                else:
                    result['purse_amount'] = int(amount_str)
                result['detection_method'] = 'embedded_format'
            except BaseException:
                pass

        # Map prefix to race type if not already found
        if result['race_type_clean'] == 'unknown':
            prefix_map = {
                'clm': 'claiming',
                'mc': 'claiming',
                'oc': 'allowance_optional',
                'alw': 'allowance',
                'mdn': 'maiden',
                'msw': 'maiden_special_weight',
                'stk': 'stakes',
                'g1': 'stakes_graded',
                'g2': 'stakes_graded',
                'g3': 'stakes_graded',
            }
            result['race_type_clean'] = prefix_map.get(prefix, 'unknown')
            result['race_type'] = embedded_match.group(0)
            result['confidence'] = 0.85
            if result['detection_method'] == 'none':
                result['detection_method'] = 'embedded_format'

    return result


def infer_purse_from_race_type(race_type: str) -> Optional[int]:
    """
    LEGACY: Infer purse from race type names like 'Clm25000n2L' or 'MC50000'.
    BRISNET embeds purse values in race type strings.

    NOTE: Use extract_race_metadata_from_pp_text() for comprehensive detection.
    This function is kept for backward compatibility.

    Examples:
    - 'Clm25000n2L' → $25,000
    - 'MC50000' → $50,000
    - 'OC20k' → $20,000
    - 'Alw28000' → $28,000
    """
    if not race_type:
        return None

    # Pattern 1: Direct numbers (Clm25000, MC50000, Alw28000)
    match = re.search(r'(\d{4,6})', race_type)
    if match:
        return int(match.group(1))

    # Pattern 2: With 'k' suffix (OC20k, Alw50k)
    match = re.search(r'(\d+)k', race_type, re.IGNORECASE)
    if match:
        return int(match.group(1)) * 1000

    # Pattern 3: Common defaults by type
    race_lower = race_type.lower()
    if 'maiden' in race_lower or 'mdn' in race_lower or 'md sp wt' in race_lower:
        return 50000  # Typical maiden special weight
    elif 'claiming' in race_lower or 'clm' in race_lower or 'mc' in race_lower:
        return 25000  # Typical claiming level
    elif 'allowance' in race_lower or 'alw' in race_lower:
        return 50000  # Typical allowance
    elif 'stake' in race_lower or 'stk' in race_lower or 'g1' in race_lower or 'g2' in race_lower or 'g3' in race_lower:
        return 100000  # Stakes minimum

    return None


def parse_recent_class_levels(block) -> List[dict]:
    """
    Parse recent races to extract class progression data.
    CRITICAL FIX: Infers purses from race type names since BRISNET embeds them.
    Returns list of dicts with purse, race_type, finish_position
    """
    races = []
    if not block:
        return races
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    # Enhanced pattern to match BRISNET format
    # Example: "11Jan26SAª 6½ ft :21ª :44¨1:09« 1:16© ¡ ¨¨¨ Clm25000n2L ¨¨©"
    lines = block_str.split('\n')
    for line in lines:
        # Updated pattern to capture embedded race types
        race_match = re.search(
            r'(\d{2}[A-Za-z]{3}\d{2})\w+\s+[\d½]+[f]?\s+.*?'
            r'([A-Z][a-z]{2,}\d+[a-zA-Z0-9\-]*|MC\d+|OC\d+k?|Alw\d+|Stk|G[123]|Hcp)',
            line
        )

        if race_match:
            race_type = race_match.group(2)

            # CRITICAL: Infer purse from race type name
            inferred_purse = infer_purse_from_race_type(race_type)

            # Extract finish position from same line (look for FIN column)
            # Pattern: FIN followed by position like "1st", "2nd", "3©", "4¬", etc.
            finish_match = re.search(r'FIN\s+(\d{1,2})[ƒ®«ª³©¨°¬²‚±\s]', line)
            finish_pos = int(finish_match.group(1)) if finish_match else 0

            try:
                races.append({
                    'race_type': race_type,
                    'purse': inferred_purse if inferred_purse else 0,
                    'finish_pos': finish_pos
                })
            except Exception:
                pass

    return races[:5]  # Last 5 races


def calculate_comprehensive_class_rating(
    today_purse: int,
    today_race_type: str,
    horse_block: str,
    pedigree: dict,
    angles_df: pd.DataFrame,
    pp_text: str = "",  # NEW: Full PP text for race class parser
    distance_furlongs: float = 0.0,  # NEW: Race distance
    surface_type: str = "Dirt"  # NEW: Surface type
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

    # DEBUG: Show parser status
    if not RACE_CLASS_PARSER_AVAILABLE:
        st.error("❌ Race Class Parser NOT AVAILABLE - using legacy weights")
    elif not pp_text:
        st.warning("⚠️ No PP text provided - parser cannot run")

    if RACE_CLASS_PARSER_AVAILABLE and pp_text:
        try:
            st.info("🔄 Attempting to parse race class...")
            race_class_data = parse_and_calculate_class(pp_text)
            today_class_weight = race_class_data['weight']['class_weight']
            today_hierarchy_level = race_class_data['hierarchy']['final_level']

            # Log parsed race info for debugging
            st.success(f"✅ Race Class Analysis: {race_class_data['summary']['class_type']} "
                       f"(Level {today_hierarchy_level}, Weight {today_class_weight:.2f})")

            # Show breakdown
            st.caption(f"📊 Breakdown: Hierarchy={race_class_data['weight']['breakdown']['hierarchy_score']:.2f}, "
                       f"Purse={race_class_data['weight']['breakdown']['purse_score']:.3f}, "
                       f"Stakes={race_class_data['weight']['breakdown']['stakes_bonus']:.2f}")

        except Exception as e:
            st.error(f"❌ Race class parser FAILED: {e}")
            import traceback
            st.code(traceback.format_exc())
            # Fall back to legacy method
            pass

    # Race type hierarchy scoring (ENHANCED with race_class_parser data)
    # Based on industry-standard US horse racing hierarchy (1-7 scale)
    race_type_scores = {
        # Maiden (Level 1)
        'msw': 1, 'maiden special weight': 1, 'md sp wt': 1,
        'mdn': 1, 'md': 1, 'maiden': 1,
        'moc': 1, 'maiden optional claiming': 1,
        'mcl': 1, 'maiden claiming': 1, 'md cl': 1, 'mdnclm': 1, 'mdc': 1,
        'msc': 1, 'maiden starter claiming': 1,

        # Claiming (Level 2)
        'clm': 2, 'claiming': 2, 'cl': 2, 'clg': 2, 'c': 2,
        'wcl': 2, 'waiver claiming': 2, 'waiver': 2,
        'clh': 2, 'claiming handicap': 2,
        'cst': 2, 'claiming stakes': 2, 'clmstk': 2,  # Low-end

        # Starter Allowance (Level 3)
        'sta': 3, 'starter allowance': 3, 'str': 3, 'starter': 3,
        'shp': 3, 'starter handicap': 3,
        'stb': 3, 'statebred': 3, 'state bred': 3,

        # Allowance / Trial (Level 4)
        'alw': 4, 'allowance': 4, 'a': 4,
        'n1x': 4, 'nw1': 4, 'allowance non-winners of 1': 4,
        'n2x': 4, 'nw2': 4, 'allowance non-winners of 2': 4,
        'n3x': 4, 'nw3': 4, 'allowance non-winners of 3': 4,
        'n1l': 4, 'n2l': 4, 'n3l': 4,  # Lifetime conditions
        'opt': 4, 'optional': 4,
        'trl': 4, 'trial': 4,

        # Optional Claiming / High Allowance (Level 5)
        'oc': 5, 'ocl': 5, 'optional claiming': 5,
        'aoc': 5, 'allowance optional claiming': 5, 'ao': 5,
        'soc': 5, 'starter optional claiming': 5,

        # Handicap (Level 6)
        'hcp': 6, 'handicap': 6, 'h': 6, '©hcp': 6, '©': 6,
        'och': 6, 'optional claiming handicap': 6,

        # Stakes (Level 7) - Graded, Listed, Non-Graded, Specialty
        'stk': 7, 'stakes': 7, 's': 7,
        'sst': 7, 'starter stakes': 7,
        'n': 7, 'non-graded stakes': 7,
        'l': 7, 'lr': 7, 'listed': 7, 'listed stakes': 7,
        'g3': 7, 'grade 3': 7, 'gr3': 7, 'grade3': 7,
        'g2': 7, 'grade 2': 7, 'gr2': 7, 'grade2': 7,
        'g1': 7, 'grade 1': 7, 'gr1': 7, 'grade1': 7,
        'fut': 7, 'futurity': 7, 'ftr': 7,
        'der': 7, 'derby': 7, 'dby': 7,
        'invit': 7, 'invitational': 7, 'inv': 7,

        # Special (Level 0)
        'mat': 0, 'match': 0, 'match race': 0,
        'tr': 0, 'training': 0, 'training race': 0,
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
        finishes = [r.get('finish_pos', 0) for r in recent_races[:3] if r.get('finish_pos', 0) > 0]
        if finishes:
            # Consider competitive if finished in top 3 in any of last 3 races
            recent_top3_count = sum(1 for f in finishes if f <= 3)
            was_competitive = recent_top3_count >= 1

    # 1. PURSE COMPARISON (weight: heavy) - FORM-ADJUSTED
    if recent_races and today_purse > 0:
        recent_purses = [r['purse'] for r in recent_races if r['purse'] > 0]
        if recent_purses:
            avg_recent_purse = np.mean(recent_purses)
            purse_ratio = today_purse / avg_recent_purse if avg_recent_purse > 0 else 1.0

            # ENHANCED: Use race class weight from parser to adjust purse impact
            purse_weight_multiplier = 1.0
            if race_class_data:
                # Higher quality races = purse difference matters more
                quality = race_class_data['summary']['quality']
                if quality == 'Elite':
                    purse_weight_multiplier = 1.5  # G1/G2 - purse differences critical
                elif quality == 'High':
                    purse_weight_multiplier = 1.2  # G3/Stakes - purse matters
                elif quality == 'Medium':
                    purse_weight_multiplier = 1.0  # Standard weight
                else:
                    purse_weight_multiplier = 0.8  # Low class - purse less predictive

            # Purse movement scoring (ADJUSTED by race quality)
            if purse_ratio >= 1.5:  # Major step up
                class_rating -= (1.2 * purse_weight_multiplier)
            elif purse_ratio >= 1.2:  # Moderate step up
                class_rating -= (0.6 * purse_weight_multiplier)
            elif purse_ratio >= 0.8 and purse_ratio <= 1.2:  # Same class
                class_rating += (0.8 * purse_weight_multiplier)
            elif purse_ratio >= 0.6:  # Class drop (FORM-ADJUSTED)
                if was_competitive:
                    class_rating += (0.8 * purse_weight_multiplier)  # Legitimate drop
                else:
                    class_rating += (0.2 * purse_weight_multiplier)  # Minimal bonus
            else:  # Major drop
                if was_competitive:
                    class_rating += (1.0 * purse_weight_multiplier)  # Should dominate
                else:
                    class_rating -= (0.3 * purse_weight_multiplier)  # Warning flag

    # 2. RACE TYPE PROGRESSION
    if recent_races:
        recent_types = [r['race_type'].lower() for r in recent_races]
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
    spi = pedigree.get('sire_spi') or pedigree.get('damsire_spi')
    if pd.notna(spi):
        spi_val = float(spi)
        pedigree_multiplier = 1.0

        # Pedigree matters MORE in graded stakes
        if race_class_data and race_class_data['summary']['is_graded']:
            pedigree_multiplier = 1.5
        elif race_class_data and race_class_data['summary']['is_stakes']:
            pedigree_multiplier = 1.2

        if spi_val >= 110:
            class_rating += (0.4 * pedigree_multiplier)
        elif spi_val >= 100:
            class_rating += (0.2 * pedigree_multiplier)

    # 4. ANGLE-BASED CLASS INDICATORS
    if angles_df is not None and not angles_df.empty:
        angle_text = ' '.join(angles_df['Category'].astype(str)).lower()

        # Class rise angles
        if 'first time starter' in angle_text or '1st time str' in angle_text:
            class_rating += 0.5  # Debut = unknown class, slight boost
        if '2nd career' in angle_text:
            class_rating += 0.3
        if 'shipper' in angle_text:
            class_rating += 0.2  # Shipper often means stepping up

    # 5. CLAIMING PRICE GRANULARITY (if claiming race)
    claiming_bonus = 0.0
    if 'clm' in today_type_norm or 'claiming' in today_type_norm:
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

    # 6. ABSOLUTE PURSE LEVEL BASELINE
    # High purse races = better horses overall
    if today_purse >= 100000:
        purse_baseline = 1.0
    elif today_purse >= 50000:
        purse_baseline = 0.5
    elif today_purse >= 25000:
        purse_baseline = 0.0
    else:
        purse_baseline = -0.5

    class_rating += claiming_bonus

    class_rating += purse_baseline

    # Clip to reasonable range
    return float(np.clip(class_rating, -3.0, 6.0))

# ===================== 1. Paste PPs & Parse (durable) =====================


# Workflow Progress Indicator
step1_done = st.session_state.get("parsed", False)
step2_done = 'primary_d' in st.session_state and 'primary_probs' in st.session_state
step3_done = st.session_state.get('classic_report_generated', False)

progress_col1, progress_col2, progress_col3, progress_col4 = st.columns(4)
with progress_col1:
    if step1_done:
        st.success("✅ Step 1: Parsed")
    else:
        st.info("⏳ Step 1: Parse Race")
with progress_col2:
    if step2_done:
        st.success("✅ Step 2: Rated")
    else:
        st.info("⏳ Step 2: Set Biases")
with progress_col3:
    if step3_done:
        st.success("✅ Step 3: Analyzed")
    else:
        st.info("⏳ Step 3: Generate Report")
with progress_col4:
    if GOLD_DB_AVAILABLE:
        try:
            stats = gold_db.get_accuracy_stats()
            saved_count = stats.get('total_races', 0) + len(gold_db.get_pending_races(limit=1000))
            if saved_count > 0:
                st.success(f"💾 {saved_count} Saved")
            else:
                st.info("💾 No Races Yet")
        except BaseException:
            st.info("💾 Database Ready")
    else:
        st.info("💾 Database Ready")

st.markdown("---")
st.header("1. Paste PPs & Parse")

pp_text_widget = st.text_area(
    "BRIS PPs text:",
    value=st.session_state["pp_text_cache"],
    height=300,
    key="pp_text_input",
    help="Paste the text from a BRIS Ultimate Past Performances PDF. Minimum 100 characters required.",
    disabled=st.session_state["parsed"]
)

# VALIDATION: Check minimum text length
if pp_text_widget and len(pp_text_widget.strip()) < 100:
    st.warning("⚠️ PP text too short. Please paste complete Past Performance data (minimum 100 characters).")

col_parse, col_reset = st.columns([1, 1])
with col_parse:
    parse_clicked = st.button("Parse PPs", type="primary")
with col_reset:
    reset_clicked = st.button("Reset / Parse New", help="Clear parsed state to paste another race")

if reset_clicked:
    st.session_state["parsed"] = False
    st.session_state["pp_text_cache"] = ""
    # Clear Classic Report from previous race
    st.session_state.pop('classic_report_generated', None)
    st.session_state.pop('classic_report', None)
    _safe_rerun()

if parse_clicked:
    text_now = (st.session_state.get("pp_text_input") or "").strip()
    # CRITICAL FIX: Validate text length before parsing
    if not text_now:
        st.warning("Paste PPs text first.")
    elif len(text_now) < 100:
        st.error("❌ PP text too short. Please paste complete Past Performance data (minimum 100 characters).")
    else:
        # PERFORMANCE: Add progress indicator for parsing operation
        with st.spinner("🔍 Parsing Past Performances..."):
            # SECURITY: Validate PP text before processing
            if SECURITY_VALIDATORS_AVAILABLE:
                try:
                    text_now = sanitize_pp_text(text_now)
                except ValueError as e:
                    st.error(f"Invalid PP text: {e}")
                    st.stop()

            st.session_state["pp_text_cache"] = text_now
            st.session_state["parsed"] = True
            # Clear Classic Report from previous race
            st.session_state.pop('classic_report_generated', None)
            st.session_state.pop('classic_report', None)
        _safe_rerun()

if not st.session_state["parsed"]:
    st.info("Paste your PPs and click **Parse PPs** to continue.")
    st.stop()

pp_text = st.session_state["pp_text_cache"]

# ===================== 2. Race Info (Confirm) =====================

st.header("2. Race Info (Confirm)")
first_line = (pp_text.split("\n", 1)[0] or "").strip()

# Parse comprehensive header information
header_info = parse_brisnet_race_header(pp_text)

# Display extracted header info if available
if header_info and any(header_info.values()):
    with st.expander("📋 Extracted Header Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if header_info.get('track_name'):
                st.caption(f"**Track:** {header_info['track_name']}")
            if header_info.get('race_number'):
                st.caption(f"**Race:** {header_info['race_number']}")
            if header_info.get('race_type'):
                st.caption(f"**Type:** {header_info['race_type']}")
            if header_info.get('purse_amount'):
                st.caption(f"**Purse:** ${header_info['purse_amount']:,}")
        with col2:
            if header_info.get('distance'):
                st.caption(f"**Distance:** {header_info['distance']}")
            if header_info.get('age_restriction') or header_info.get('sex_restriction'):
                restrictions = []
                if header_info.get('age_restriction'):
                    restrictions.append(header_info['age_restriction'])
                if header_info.get('sex_restriction'):
                    restrictions.append(header_info['sex_restriction'])
                st.caption(f"**Restrictions:** {' '.join(restrictions)}")
            if header_info.get('day_of_week') and header_info.get('race_date'):
                st.caption(f"**Date:** {header_info['day_of_week']}, {header_info['race_date']}")

# Track (use parsed value from comprehensive parser if available, otherwise fallback)
parsed_track = header_info.get('track_name') or parse_track_name_from_pp(pp_text)
track_name = st.text_input("Track:", value=(parsed_track or st.session_state['track_name']))

# SECURITY: Validate track name if validators available
if SECURITY_VALIDATORS_AVAILABLE and track_name:
    try:
        track_name = validate_track_name(track_name)
    except ValueError as e:
        st.warning(f"Invalid track name (using default): {e}")
        track_name = "Unknown Track"

st.session_state['track_name'] = track_name

# Race Number (use parsed value if available)
if 'race_num' not in st.session_state:
    st.session_state['race_num'] = 1
auto_race_num = header_info.get('race_number') or detect_race_number(pp_text)
default_race_num = int(auto_race_num) if auto_race_num else st.session_state['race_num']
# CRITICAL FIX: Increase max_value to 20 (some tracks have 16+ races)
race_num = st.number_input("Race Number:", min_value=1, max_value=20, step=1, value=default_race_num)
st.session_state['race_num'] = race_num

# Surface auto from header, but allow override
default_surface = st.session_state['surface_type']
if re.search(r'(?i)\bturf|trf\b', first_line):
    default_surface = "Turf"
if re.search(r'(?i)\baw|tap|synth|poly\b', first_line):
    default_surface = "Synthetic"
surface_type = st.selectbox(
    "Surface:", [
        "Dirt", "Turf", "Synthetic"], index=[
            "Dirt", "Turf", "Synthetic"].index(default_surface) if default_surface in [
                "Dirt", "Turf", "Synthetic"] else 0)
st.session_state['surface_type'] = surface_type

# Condition
conditions = ["fast", "good", "wet-fast", "muddy", "sloppy", "firm", "yielding", "soft", "heavy"]
cond_found = None
for cond in conditions:
    if re.search(rf'(?i)\b{cond}\b', first_line):
        cond_found = cond
        break
default_condition = cond_found if cond_found else st.session_state['condition_txt']
condition_txt = st.selectbox("Condition:", conditions,
                             index=conditions.index(default_condition) if default_condition in conditions else 0)
st.session_state['condition_txt'] = condition_txt

# Distance (auto + dropdown)


def _auto_distance_label(s: str) -> str:
    m = re.search(r'(?i)\b(\d+(?:\s*1/2|½)?\s*furlongs?)\b', s)
    if m:
        return m.group(1).title().replace("1/2", "½")
    if re.search(r'(?i)\b8\s*1/2\s*furlongs?\b', s):
        return "8 1/2 Furlongs"
    if re.search(r'(?i)\b8\s*furlongs?\b', s):
        return "8 Furlongs"
    if re.search(r'(?i)\b9\s*furlongs?\b', s):
        return "9 Furlongs"
    if re.search(r'(?i)\b1\s*mile\b', s):
        return "1 Mile"
    if re.search(r'(?i)\b1\s*1/16\b', s):
        return "1 1/16 Miles"
    if re.search(r'(?i)\b7\s*furlongs?\b', s):
        return "7 Furlongs"
    return "6 Furlongs"


auto_distance = _auto_distance_label(first_line)
# try to map to option variants
preferred = (auto_distance or "").replace("½", "1/2").replace(" 1/2", " 1/2")
idx = DISTANCE_OPTIONS.index("6 Furlongs") if "6 Furlongs" in DISTANCE_OPTIONS else 0
for opt in (preferred, auto_distance, st.session_state['distance_txt']):
    if opt in DISTANCE_OPTIONS:
        idx = DISTANCE_OPTIONS.index(opt)
        break
distance_txt = st.selectbox("Distance:", DISTANCE_OPTIONS, index=idx)
st.session_state['distance_txt'] = distance_txt

# Purse


def detect_purse_amount(pp_text: str) -> Optional[int]:
    s = pp_text or ""
    m = re.search(r'(?mi)\bPurse\b[^$\n\r]*\$\s*([\d,]+)', s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except BaseException:
            pass
    m = re.search(r'(?mi)\b(?:Added|Value)\b[^$\n\r]*\$\s*([\d,]+)', s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except BaseException:
            pass
    m = re.search(
        r'(?mi)\b(Mdn|Maiden|Allowance|Alw|Claiming|Clm|Starter|Stake|Stakes)\b[^:\n\r]{0,50}\b(\d{2,4})\s*[Kk]\b',
        s)
    if m:
        try:
            return int(m.group(2)) * 1000
        except BaseException:
            pass
    m = re.search(r'(?m)\$\s*([\d,]{5,})', s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except BaseException:
            pass
    return None


auto_purse = detect_purse_amount(pp_text)
default_purse = int(auto_purse) if auto_purse else st.session_state['purse_val']
purse_val = st.number_input("Purse ($)", min_value=0, step=5000, value=default_purse)
st.session_state['purse_val'] = purse_val

# Race type detection + override list (constant keys only)
race_type_detected = detect_race_type(pp_text)
st.caption(f"Detected race type: **{race_type_detected}**")
# Ensure base_class_bias is not empty before proceeding
if not base_class_bias:
    st.error("Race type definitions (base_class_bias) are missing or failed to load.")
    st.stop()
try:
    race_type_index = list(
        base_class_bias.keys()).index(race_type_detected) if race_type_detected in base_class_bias else list(
        base_class_bias.keys()).index("allowance")
except ValueError:
    st.warning("Default race type 'allowance' not found in bias definitions. Using first available type.")
    race_type_index = 0  # Default to the first item if 'allowance' isn't found

race_type_manual = st.selectbox(
    "Race Type (override):",
    options=list(base_class_bias.keys()),
    index=race_type_index,
)
race_type = race_type_manual or race_type_detected
race_type_detected = race_type  # lock in constant key
st.session_state['race_type'] = race_type_detected  # Store for Classic Report

# ===================== A. Race Setup: Scratches, ML & Styles =====================

st.markdown("---")
st.header("A. Race Setup: Scratches, ML & Live Odds, Styles")
st.caption("📝 Review auto-detected data below. Edit any horse's ML odds or style, then check 'Scratch?' to remove horses.")

df_styles = extract_horses_and_styles(pp_text)
if df_styles.empty:
    st.error("No horses found. Check your PP text paste.")
    st.stop()

ml_map_raw = extract_morning_line_by_horse(pp_text)
df_styles["ML"] = df_styles["Horse"].map(lambda h: ml_map_raw.get(h, "")) if "Horse" in df_styles else ""
df_styles["Live Odds"] = df_styles["ML"].where(df_styles["ML"].astype(str).str.len() > 0, "")
df_styles["Scratched"] = False

col_cfg = {
    "Post": st.column_config.TextColumn(
        "Post",
        width="small",
        disabled=True),
    "Horse": st.column_config.TextColumn(
        "Horse",
        width="medium",
        disabled=True),
    "ML": st.column_config.TextColumn(
        "ML",
        width="small",
        help="Parsed Morning Line from PPs"),
    "Live Odds": st.column_config.TextColumn(
        "Live Odds",
        width="small",
        help="Enter current odds (e.g., '5/2', '3.5', '+250')"),
    "DetectedStyle": st.column_config.TextColumn(
        "BRIS Style",
        width="small",
        disabled=True),
    "Quirin": st.column_config.NumberColumn(
        "Quirin",
        width="small",
        disabled=True),
    "AutoStrength": st.column_config.TextColumn(
        "Auto-Strength",
        width="small",
        disabled=True),
    "OverrideStyle": st.column_config.SelectboxColumn(
        "Override Style",
        width="small",
        options=[
            "",
            "E",
            "E/P",
            "P",
            "S"]),
    "Scratched": st.column_config.CheckboxColumn(
        "Scratched?",
        width="small")}
df_editor = st.data_editor(df_styles, use_container_width=True, column_config=col_cfg)

# ===================== B. Angle Parsing / Pedigree / Figs =====================

angles_per_horse: Dict[str, pd.DataFrame] = {}
pedigree_per_horse: Dict[str, dict] = {}
figs_per_horse: Dict[str, List[int]] = {}

# Try to use elite parser first for better accuracy
elite_parser_used = False
if ELITE_PARSER_AVAILABLE and GoldStandardBRISNETParser is not None and len(pp_text.strip()) > 100:
    try:
        parser = GoldStandardBRISNETParser()
        horses = parser.parse_full_pp(pp_text, debug=False)
        validation = parser.validate_parsed_data(horses, min_confidence=0.5)

        if validation.get('overall_confidence', 0.0) >= 0.6:
            # Use elite parser data
            elite_parser_used = True
            for horse_name, horse_obj in horses.items():
                if horse_name in df_editor["Horse"].values:
                    # Store speed figures
                    if horse_obj.speed_figures and len(horse_obj.speed_figures) > 0:
                        figs_per_horse[horse_name] = horse_obj.speed_figures

                    # Note: angles and pedigree can still use traditional parsing
                    # as elite parser may not have all that data structured the same way
    except Exception as e:
        st.caption(f"Elite parser unavailable: {str(e)[:50]}")
        pass

# Fall back to traditional parsing if elite parser wasn't used or for missing data
for _post, name, block in split_into_horse_chunks(pp_text):
    if name in df_editor["Horse"].values:
        # Always parse angles and pedigree traditionally
        angles_per_horse[name] = parse_angles_for_block(block)
        pedigree_per_horse[name] = parse_pedigree_snips(block)

        # Only parse figs if not already parsed by elite parser
        if name not in figs_per_horse:
            figs_per_horse[name] = parse_speed_figures_for_block(block)

# Create the figs_df
figs_data = []
for name, fig_list in figs_per_horse.items():
    if fig_list:  # Only add horses that have figures
        figs_data.append({
            "Horse": name,
            "Figures": fig_list,  # The list of parsed figs
            "BestFig": max(fig_list),
            "AvgTop2": round(np.mean(sorted(fig_list, reverse=True)[:2]), 1)
        })
figs_df = pd.DataFrame(figs_data)  # <--- THIS IS THE NEW FIGS DATAFRAME

# CRITICAL: Validate df_editor before operations
if df_editor is None or df_editor.empty:
    st.error("❌ No horse data available. Please enter horses in Section A.")
    st.stop()

# CRITICAL: Explicit False check to handle potential NaN values from data_editor
df_final_field = df_editor[df_editor["Scratched"].fillna(False) == False].copy()
if df_final_field.empty:
    st.warning("⚠️ All horses are scratched.")
    st.stop()

# Store in session state for later access
st.session_state['df_final_field'] = df_final_field

# Ensure StyleStrength and Style exist
df_final_field["StyleStrength"] = df_final_field.apply(lambda row: calculate_style_strength(
    row["OverrideStyle"] if row["OverrideStyle"] else row["DetectedStyle"], row["Quirin"]), axis=1)
df_final_field["Style"] = df_final_field.apply(
    lambda r: _normalize_style(r["OverrideStyle"] if r["OverrideStyle"] else r["DetectedStyle"]), axis=1
)
if "#" not in df_final_field.columns:
    df_final_field["#"] = df_final_field["Post"].astype(str)

# PPI
ppi_results = compute_ppi(df_final_field)
ppi_val = ppi_results.get("ppi", 0.0)
ppi_map_by_horse = ppi_results.get("by_horse", {})
st.session_state['ppi_val'] = ppi_val  # Store for Classic Report

# ===================== Class build per horse (angles+pedigree in background) =====================


def _infer_horse_surface_pref(name: str, ped: dict, ang_df: Optional[pd.DataFrame], race_surface: str) -> str:
    cats = " ".join(
        ang_df["Category"].astype(str).tolist()).lower() if (
        ang_df is not None and not ang_df.empty) else ""
    if "dirt to tur" in cats:
        return "Tur"
    if "turf to dirt" in cats:
        return "Dirt"
    # If nothing clear, use race surface (neutral) to avoid over-penalizing
    return race_surface


def _infer_horse_distance_pref(ped: dict) -> str:
    awds = [x for x in [ped.get("sire_awd"), ped.get("damsire_awd")] if pd.notna(x)]
    if not awds:
        return "any"
    m = float(np.nanmean(awds))
    if m <= 6.5:
        return "≤6"
    if m >= 7.5:
        return "8f+"
    return "6.5–7"


def _angles_pedigree_tweak(name: str, race_surface: str, race_bucket: str, race_cond: str) -> float:
    """
    Small, capped additive tweak that folds pedigree + common angles into Cclass.
    Positive values help; negatives hurt.
    """
    ped = (pedigree_per_horse.get(name, {}) or {})
    ang = angles_per_horse.get(name)
    tweak = 0.0

    # 1) Pedigree AWD vs today's distance bucket
    awds = [x for x in [ped.get("sire_awd"), ped.get("damsire_awd")] if pd.notna(x)]
    # Validate AWD data exists and is valid BEFORE using
    if awds:
        awd_mean = float(np.nanmean(awds))
        if pd.notna(awd_mean):  # Use pandas notna for clarity
            if race_bucket == "≤6f":
                if awd_mean <= 6.5:
                    tweak += MODEL_CONFIG['ped_dist_bonus']
                elif awd_mean >= 7.5:
                    tweak += MODEL_CONFIG['ped_dist_penalty']
            elif race_bucket == "8f+":
                if awd_mean >= 7.5:
                    tweak += MODEL_CONFIG['ped_dist_bonus']
                elif awd_mean <= 6.5:
                    tweak += MODEL_CONFIG['ped_dist_penalty']
            else:  # 6.5-7f bucket
                if 6.3 <= awd_mean <= 7.7:
                    tweak += MODEL_CONFIG['ped_dist_neutral_bonus']

    # Sprint/debut pop from 1st% in true sprints
    if race_bucket == "≤6":
        for v in [ped.get("sire_1st"), ped.get("damsire_1st")]:
            if pd.notna(v):
                if float(v) >= MODEL_CONFIG['ped_first_pct_threshold']:
                    tweak += MODEL_CONFIG['ped_first_pct_bonus']

    # 2) Angles
    if ang is not None and not ang.empty:
        cats = " ".join(ang["Category"].astype(str).tolist()).lower()
        if "1st time str" in cats or "debut mdnspwt" in cats or "maiden sp wt" in cats:
            tweak += MODEL_CONFIG['angle_debut_msw_bonus'] if race_type_detected == "maiden special weight" else MODEL_CONFIG['angle_debut_other_bonus']
            if race_bucket == "≤6f":
                tweak += MODEL_CONFIG['angle_debut_sprint_bonus']
        if "2nd career" in cats:
            tweak += MODEL_CONFIG['angle_second_career_bonus']
        if ("turf to dirt" in cats and race_surface.lower() == "dirt"):
            tweak += MODEL_CONFIG['angle_surface_switch_bonus']
        if ("dirt to tur" in cats and race_surface.lower() == "turf"):
            tweak += MODEL_CONFIG['angle_surface_switch_bonus']
        if "blinkers on" in cats:
            tweak += MODEL_CONFIG['angle_blinkers_on_bonus']
        if "blinkers off" in cats:
            tweak += MODEL_CONFIG['angle_blinkers_off_bonus']
        if "shipper" in cats:
            tweak += MODEL_CONFIG['angle_shipper_bonus']
        try:
            if 'ROI' in ang.columns:
                pos_ct = int((ang["ROI"] > 0).sum())
                neg_ct = int((ang["ROI"] < 0).sum())
                tweak += min(MODEL_CONFIG['angle_roi_pos_max_bonus'], MODEL_CONFIG['angle_roi_pos_per_bonus'] * pos_ct)
                tweak -= min(MODEL_CONFIG['angle_roi_neg_max_penalty'],
                             MODEL_CONFIG['angle_roi_neg_per_penalty'] * neg_ct)
        except Exception as e:
            st.warning(f"Error during angle ROI calculation for horse '{name}'. Error: {e}")
            pass

    # 3) Condition nuance
    if race_cond in {"muddy", "sloppy", "heavy"} and awd_mean == awd_mean:  # Check if not NaN
        if race_bucket != "≤6f" and awd_mean >= 7.5:
            tweak += MODEL_CONFIG['angle_off_track_route_bonus']

    return float(np.clip(round(tweak, 3), MODEL_CONFIG['angle_tweak_min_clip'], MODEL_CONFIG['angle_tweak_max_clip']))
    # Build Cclass and Cform using comprehensive analysis
    race_surface = surface_type
    race_cond = condition_txt
    race_bucket = distance_bucket(distance_txt)

    Cclass_vals = []
    Cform_vals = []
    for _, r in df_final_field.iterrows():
        name = r["Horse"]
        ped = pedigree_per_horse.get(name, {}) or {}
        ang = angles_per_horse.get(name)

        # Get horse's PP block for analysis
        horse_block = ""
        for _, h_name, block in split_into_horse_chunks(pp_text):
            if h_name == name:
                horse_block = block
                break

        # Calculate comprehensive class rating
        # NOW includes PP text for race class parser to properly understand race acronyms
        comprehensive_class = calculate_comprehensive_class_rating(
            today_purse=purse_val,
            today_race_type=race_type_detected,
            horse_block=horse_block,
            pedigree=ped,
            angles_df=ang if ang is not None else pd.DataFrame(),
            pp_text=pp_text,  # NEW: Full PP for race analysis
            distance_furlongs=distance_to_furlongs(distance_txt),  # NEW: Distance
            surface_type=race_surface  # NEW: Surface type
        )

        # Add pedigree/angle tweaks on top of class
        tweak = _angles_pedigree_tweak(name, race_surface, race_bucket, race_cond)
        cclass_total = comprehensive_class + tweak

        # Calculate form cycle rating
        form_rating = calculate_form_cycle_rating(
            horse_block=horse_block,
            pedigree=ped,
            angles_df=ang if ang is not None else pd.DataFrame()
        )

        # Track bias style adjustments
        style_adjustment = 0.0
        horse_style = r.get('Style', 'NA')

        dist_bucket = distance_bucket(distance_txt)
        track_cfg = TRACK_BIAS_PROFILES.get(canon_track, {}).get(race_surface, {}).get(dist_bucket, {})
        runstyle_biases = track_cfg.get('runstyle', {})

        # Strong stalker track detected if S style has > 0.3 advantage
        stalker_impact = runstyle_biases.get('S', 0.0)
        early_speed_impact = runstyle_biases.get('E', 0.0)

        if stalker_impact > 0.3:  # Strong stalker-favoring track
            # Apply penalties/bonuses based on horse's style vs track bias
            if horse_style == 'E':
                style_adjustment -= 1.5  # Heavy penalty for early speed
            elif horse_style == 'E/P':
                style_adjustment -= 0.8  # Moderate penalty
            elif horse_style == 'S':
                style_adjustment += 1.2  # Strong bonus for stalkers
        elif early_speed_impact > 0.3:  # Speed-favoring track
            if horse_style == 'E':
                style_adjustment += 1.2  # Bonus for early speed
            elif horse_style == 'S':
                style_adjustment -= 0.8  # Penalty for closers

        form_rating += style_adjustment

        Cclass_vals.append(round(cclass_total, 3))
        Cform_vals.append(round(form_rating, 3))

    df_final_field["Cclass"] = Cclass_vals
    df_final_field["Cform"] = Cform_vals

# ===================== B. Bias-Adjusted Ratings =====================


st.markdown("---")
st.header("B. Bias-Adjusted Ratings")
st.caption("⚙️ Select your strategy profile and bias preferences. Ratings calculate automatically based on your selections.")
b_col1, b_col2, b_col3 = st.columns(3)
with b_col1:
    strategy_profile = st.selectbox(
        "Select Strategy Profile:",
        options=["Confident", "Value Hunter"],
        index=0, key="strategy_profile"
    )
with b_col2:
    running_style_biases = st.multiselect(
        "Select Running Style Biases:",
        options=["E", "E/P", "P", "S"],
        default=["E"], key="style_biases"
    )
with b_col3:
    post_biases = st.multiselect(
        "Select Post Position Biases:",
        options=[
            "no significant post bias",
            "favors rail (1)",
            "favors inner (1-3)",
            "favors mid (4-7)",
            "favors outside (8+)"],
        default=["no significant post bias"],
        key="post_biases")

if not running_style_biases or not post_biases:
    st.info("Pick at least one **Style** bias and one **Post** bias.")
    st.stop()


def _style_bias_label_from_choice(choice: str) -> str:
    # Map single-letter selection into our style_match table buckets
    up = (choice or "").upper()
    if up in ("E", "E/P"):
        return "speed favoring"
    if up in ("P", "S"):
        return "closer favoring"
    return "fair/neutral"


def style_match_score(running_style_bias: str, style: str, quirin: float) -> float:
    # running_style_bias is already mapped to a label (speed/closer/fair)
    bias = (running_style_bias or "").strip().lower()
    stl = (style or "NA").upper()

    table = MODEL_CONFIG['style_match_table']
    base = table.get(bias, table["fair/neutral"]).get(stl, 0.0)

    try:
        q = float(quirin)
    except Exception:
        q = np.nan

    if stl in ("E", "E/P") and pd.notna(q) and q >= MODEL_CONFIG['style_quirin_threshold']:
        base += MODEL_CONFIG['style_quirin_bonus']
    return float(np.clip(base, -1.0, 1.0))


def style_match_score_multi(running_style_biases: list, style: str, quirin: float) -> float:
    """Calculate style match score from multiple selected running style biases (aggregates bonuses)"""
    if not running_style_biases:
        return 0.0
    
    stl = (style or "NA").upper()
    table = MODEL_CONFIG['style_match_table']
    
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
        
        bonus = table.get(bias_lower, table["fair/neutral"]).get(stl, 0.0)
        
        # Add Quirin bonus if applicable
        if stl in ("E", "E/P") and pd.notna(q) and q >= MODEL_CONFIG['style_quirin_threshold']:
            bonus += MODEL_CONFIG['style_quirin_bonus']
        
        if bonus > 0:  # Only add positive bonuses
            total_bonus += bonus
    
    return float(np.clip(total_bonus, -1.0, 1.0))

# ======================== Phase 1: Enhanced Parsing Functions ========================


def parse_track_bias_impact_values(pp_text: str) -> Dict[str, float]:
    """Extract Track Bias Impact Values from '9b. Track Bias (Numerical)' section"""
    impact_values = {}

    # Find the Track Bias section
    bias_match = re.search(r'9b\.\s*Track Bias.*?\n(.*?)(?=\n\d+[a-z]?\.|$)', pp_text, re.DOTALL | re.IGNORECASE)
    if not bias_match:
        return impact_values

    bias_text = bias_match.group(1)

    # Parse Impact Value lines (e.g., "- E (Early Speed): Impact Value = 1.8")
    for match in re.finditer(r'-\s*([A-Z/]+)\s*\([^)]+\):\s*Impact Value\s*=\s*([\d.]+)', bias_text):
        style_code = match.group(1).strip()
        impact_val = float(match.group(2))
        impact_values[style_code] = impact_val

    return impact_values


def parse_pedigree_spi(pp_text: str) -> Dict[str, Optional[int]]:
    """Extract SPI (Sire Performance Index) from pedigree sections"""
    spi_values = {}

    # Look for pattern like "Sire: Hard Spun (SPI: 1.30)" in Section 4
    horse_sections = re.split(r'\n(?=\d+\.\s+Horse:)', pp_text)

    for section in horse_sections:
        horse_match = re.search(r'Horse:\s*(.+?)(?=\s*\(#|\n)', section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        # Look for SPI in Sire line
        spi_match = re.search(r'Sire:.*?SPI:\s*([\d.]+)', section, re.IGNORECASE)
        if spi_match:
            try:
                spi = int(float(spi_match.group(1)) * 100)  # Convert 1.30 to 130
                spi_values[horse_name] = spi
            except Exception:
                spi_values[horse_name] = None
        else:
            spi_values[horse_name] = None

    return spi_values


def parse_pedigree_surface_stats(pp_text: str) -> Dict[str, Dict[str, any]]:
    """Extract surface statistics (Turf/AW win%) from pedigree sections"""
    surface_stats = {}

    horse_sections = re.split(r'\n(?=\d+\.\s+Horse:)', pp_text)

    for section in horse_sections:
        horse_match = re.search(r'Horse:\s*(.+?)(?=\s*\(#|\n)', section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        stats = {}
        # Look for "Turf: 12% (class-adj)" or similar
        turf_match = re.search(r'Turf:\s*([\d.]+)%', section, re.IGNORECASE)
        if turf_match:
            stats['turf_pct'] = float(turf_match.group(1))

        # Look for "AW: 8% (class-adj)" or similar
        aw_match = re.search(r'(?:AW|All-Weather):\s*([\d.]+)%', section, re.IGNORECASE)
        if aw_match:
            stats['aw_pct'] = float(aw_match.group(1))

        if stats:
            surface_stats[horse_name] = stats

    return surface_stats


def parse_awd_analysis(pp_text: str) -> Dict[str, str]:
    """Extract AWD (Avg Winning Distance) analysis from pedigree sections"""
    awd_data = {}

    horse_sections = re.split(r'\n(?=\d+\.\s+Horse:)', pp_text)

    for section in horse_sections:
        horse_match = re.search(r'Horse:\s*(.+?)(?=\s*\(#|\n)', section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        # Look for "✔ AWD Match" or "⚠ Distance Mismatch"
        if '✔ AWD Match' in section or 'AWD Match' in section:
            awd_data[horse_name] = 'match'
        elif '⚠ Distance Mismatch' in section or 'Distance Mismatch' in section:
            awd_data[horse_name] = 'mismatch'

    return awd_data


def calculate_spi_bonus(spi: Optional[int]) -> float:
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


def calculate_surface_specialty_bonus(surface_pct: Optional[float], surface_type: str) -> float:
    """Calculate bonus for surface specialty (Turf or AW)"""
    if surface_pct is None:
        return 0.0

    if surface_type.lower() == 'tur':
        if surface_pct >= 15:
            return 0.20
        elif surface_pct >= 12:
            return 0.15
        elif surface_pct >= 10:
            return 0.10
        elif surface_pct < 5:
            return -0.10
    elif surface_type.lower() in ['aw', 'all-weather', 'synthetic']:
        if surface_pct >= 12:
            return 0.15
        elif surface_pct >= 10:
            return 0.10
        elif surface_pct < 5:
            return -0.10

    return 0.0


def calculate_awd_mismatch_penalty(awd_status: Optional[str]) -> float:
    """Calculate penalty for distance mismatch"""
    if awd_status == 'mismatch':
        return -0.10
    return 0.0

# ======================== ELITE ENHANCEMENTS ========================


def _parse_post_number(post_str: str) -> Optional[int]:
    """ELITE: Fast post number extraction without regex (3x faster)"""
    try:
        # Method 1: Try direct int conversion (fastest)
        return int(str(post_str).strip())
    except ValueError:
        try:
            # Method 2: Extract digits with string comprehension (2x faster than regex)
            digits = ''.join(c for c in str(post_str) if c.isdigit())
            return int(digits) if digits else None
        except (ValueError, TypeError):
            # Log parsing failures to session state (not UI - avoids warning spam)
            if 'post_parse_failures' not in st.session_state:
                st.session_state.post_parse_failures = []
            st.session_state.post_parse_failures.append(str(post_str))
            return None


def calculate_weather_impact(weather_data: Dict[str, Any], style: str, distance_txt: str) -> float:
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
    track_condition = str(weather_data.get('condition', 'Fast')).lower()
    wind_mph = float(weather_data.get('wind_mph', 0))
    temp_f = float(weather_data.get('temp_f', 70))

    # 1. Track Condition Impact
    if 'mud' in track_condition or 'slop' in track_condition:
        # Mud favors early speed that can secure position
        if style in ['E', 'E/P']:
            bonus += 0.20  # Early speed advantage in mud
        elif style == 'S':
            bonus -= 0.15  # Closers struggle in mud

    elif 'wet' in track_condition or 'yield' in track_condition:
        # Wet track: Moderate advantage to stalkers
        if style == 'E/P':
            bonus += 0.10

    # 2. Wind Impact (>12 mph affects race significantly)
    if wind_mph >= 15:
        # Strong headwinds favor closers (pace collapses)
        if style == 'S':
            bonus += 0.12
        elif style == 'E':
            bonus -= 0.08  # Early speed tires faster

    # 3. Temperature Impact on Stamina (distance races)
    is_route = any(d in distance_txt.upper() for d in ['M', 'MILE']) if distance_txt else False
    if is_route:
        if temp_f >= 85:
            # Hot weather: Stamina drop for routes
            bonus -= 0.08
        elif temp_f <= 35:
            # Cold weather: Slight advantage to closers (slower pace)
            if style == 'S':
                bonus += 0.05

    return float(np.clip(bonus, -0.30, 0.30))


def calculate_jockey_trainer_impact(horse_name: str, pp_text: str) -> float:
    """
    ELITE: Calculate impact of jockey/trainer performance based on BRISNET PP stats.

    BRISNET Format: "Jockey: J. Castellano (15-3-2-2)" = 15 starts, 3 wins, 2 places, 2 shows
    """
    if not pp_text or not horse_name:
        return 0.0

    bonus = 0.0

    # Extract jockey stats from PP text for this horse
    # Pattern: Horse section followed by "Jockey:" then stats
    import re

    # Find horse section and extract jockey/trainer stats
    # Case-insensitive pattern that handles multi-word names with apostrophes/periods
    jockey_pattern = r'Jockey:?\s*([A-Za-z][A-Za-z\s\.\']+?)\s*\((\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\)'
    trainer_pattern = r'Trainer:?\s*([A-Za-z][A-Za-z\s\.\']+?)\s*\((\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\)'

    # Search within reasonable window after horse name
    horse_section_start = pp_text.find(horse_name)
    if horse_section_start != -1:
        # Search next 500 chars for jockey/trainer stats
        section = pp_text[horse_section_start:horse_section_start + 500]

        jockey_match = re.search(jockey_pattern, section)
        if jockey_match:
            starts, wins, places, shows = map(int, jockey_match.groups())

            if starts >= 10:  # Minimum sample size
                win_pct = wins / starts
                itm_pct = (wins + places + shows) / starts  # In-the-money %

                # OPTIMIZED: Elite jockey bonuses tripled (SA R8: 20% jockey won but only got +0.10)
                # Elite jockey (>25% win rate) = +0.35 bonus (was +0.15)
                if win_pct >= 0.25:
                    bonus += 0.35
                # Strong jockey (>20% win rate) = +0.25 bonus (was +0.10)
                elif win_pct >= 0.20:
                    bonus += 0.25
                # Good jockey (>15% win rate) = +0.15 bonus (NEW)
                elif win_pct >= 0.15:
                    bonus += 0.15

                # Hot jockey (>60% ITM) = additional +0.10 (was +0.05)
                if itm_pct >= 0.60:
                    bonus += 0.10
                # Solid ITM (>50%) = +0.05 (NEW)
                elif itm_pct >= 0.50:
                    bonus += 0.05

                # Store jockey win% for combo bonus check
                jockey_win_rate = win_pct
            else:
                jockey_win_rate = 0.0
        else:
            jockey_win_rate = 0.0

        trainer_match = re.search(trainer_pattern, section)
        trainer_win_rate = 0.0
        if trainer_match:
            t_starts, t_wins, t_places, t_shows = map(int, trainer_match.groups())

            if t_starts >= 20:
                t_win_pct = t_wins / t_starts
                trainer_win_rate = t_win_pct

                # Elite trainer (>28% win rate) = +0.12 bonus
                if t_win_pct >= 0.28:
                    bonus += 0.12
                elif t_win_pct >= 0.22:
                    bonus += 0.08

        # ELITE CONNECTIONS COMBO BONUS (SA R8 enhancement)
        # When both jockey AND trainer are elite, add significant combo bonus
        # SA R8 winner: 22% jockey + 18% trainer = elite connections
        if jockey_win_rate >= 0.18 and trainer_win_rate >= 0.15:
            # Both connections are strong/elite - powerful combination
            bonus += 0.25  # Elite combo bonus
        elif jockey_win_rate >= 0.15 and trainer_win_rate >= 0.12:
            # Both connections are good - moderate combo
            bonus += 0.15  # Good combo bonus

    return float(np.clip(bonus, 0, 0.50))  # Increased cap from 0.35 to 0.50


def calculate_track_condition_granular(track_info: Dict[str, Any], style: str, post: int) -> float:
    """
    ELITE: Track condition analysis beyond basic Fast/Muddy.

    Args:
        track_info: Dict with 'condition', 'rail_position', 'moisture_level'
        style: Running style
        post: Post position

    Returns:
        Adjustment to rating
    """
    if not track_info:
        return 0.0

    bonus = 0.0

    # 1. Rail Position Impact
    rail_position = str(track_info.get('rail_position', 'normal')).lower()
    if '10ft' in rail_position or '15ft' in rail_position or 'out' in rail_position:
        # Rail is out - outside posts gain advantage
        if post >= 6:
            bonus += 0.25
        elif post <= 2:
            bonus -= 0.15  # Inside posts penalized

    # 2. Track Seal/Harrowing
    condition_detail = str(track_info.get('condition', '')).lower()
    if 'sealed' in condition_detail or 'harrowed' in condition_detail:
        # Sealed track favors early speed
        if style in ['E', 'E/P']:
            bonus += 0.18

    elif 'cuppy' in condition_detail or 'tiring' in condition_detail:
        # Cuppy/tiring track favors closers
        if style == 'S':
            bonus += 0.15
        elif style == 'E':
            bonus -= 0.10

    # 3. Moisture Content (even on "Fast" tracks)
    moisture = str(track_info.get('moisture_level', 'normal')).lower()
    if 'tacky' in moisture or 'holding' in moisture:
        # Tacky track: Grip advantage to stalkers
        if style == 'E/P':
            bonus += 0.08

    return float(np.clip(bonus, -0.25, 0.25))


# ======================== MARATHON CALIBRATION (McKnight G3 Learning) ========================

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
    if 'f' in distance_lower:
        try:
            furlongs = float(distance_lower.replace('f', '').strip())
            return furlongs >= 12.0
        except BaseException:
            pass

    # Mile conversions
    if 'mile' in distance_lower or 'm' in distance_lower:
        # 1½ miles = 12f, 1⅝ = 13f, 1¾ = 14f, 2 miles = 16f
        if '½' in distance_txt or '1/2' in distance_txt or '1.5' in distance_txt:
            return True
        if '⅝' in distance_txt or '5/8' in distance_txt or '1.625' in distance_txt:
            return True
        if '¾' in distance_txt or '3/4' in distance_txt or '1.75' in distance_txt:
            return True
        if '2' in distance_txt and 'mile' in distance_lower:
            return True

    return False


def calculate_workout_bonus_v2(workout_data: Dict[str, Any], is_marathon: bool = False) -> float:
    """
    CALIBRATED: Improved workout bonus emphasizing percentile rankings.

    Key Learning from McKnight G3:
    - Layabout (WINNER): Bullet 1/2 (Top 50%)
    - Summer Cause (4th): Bullet 15/16 (Top 94%)
    - Zverev (5th): Bullet 30/44 (Top 68%)

    Percentile matters MORE than just having a bullet work!
    """
    bonus = 0.0

    if not workout_data:
        return 0.0

    # Calculate percentile if rank/total available
    percentile = workout_data.get('percentile', 100)
    if percentile is None and workout_data.get('work_rank') and workout_data.get('work_total'):
        try:
            rank = int(workout_data['work_rank'])
            total = int(workout_data['work_total'])
            percentile = (rank / total) * 100
        except BaseException:
            percentile = 100

    # ELITE PERCENTILE BONUSES (from post-race analysis)
    if percentile <= 10:
        bonus += 0.25  # TOP 10% = elite work
    elif percentile <= 25:
        bonus += 0.15  # TOP 25% = strong work
    elif percentile <= 50:
        bonus += 0.10  # TOP 50% = solid work (WINNER range!)
    elif percentile <= 75:
        bonus += 0.05  # Top 75%
    else:
        bonus += 0.02  # Below 75%

    # Additional bullet work bonus (but smaller than before)
    quality = workout_data.get('quality', 'none')
    if quality == 'bullet':
        bonus += 0.05  # Reduced from 0.10
    elif quality == 'handily':
        bonus += 0.03  # Reduced from 0.05

    # Marathon bonus for recent sharp works
    if is_marathon:
        days_since = workout_data.get('days_since', 999)
        if days_since <= 7 and percentile <= 50:
            bonus += 0.05  # Very recent sharp work

    return float(np.clip(bonus, 0.0, 0.35))


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
            return 0.0   # Neutral
        elif days_off > 90:
            return -0.15  # Too long away
        else:
            return 0.0   # Recent race (no bonus)
    else:
        # Standard distances prefer recent racing
        if days_off <= 14:
            return +0.05  # Sharp form
        elif days_off <= 30:
            return 0.0   # Acceptable
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
            return 0.0   # Normal experience
        else:
            return -0.03  # Lots of wear and tear
    else:
        # Standard distances
        if career_starts <= 10:
            return +0.05  # Improving
        elif career_starts <= 20:
            return 0.0   # Normal
        else:
            return 0.0   # Experienced

    return 0.0


def calculate_hot_trainer_bonus(
        trainer_win_pct: float,
        is_hot_l14: bool = False,
        is_2nd_lasix_high_pct: bool = False) -> float:
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

    Returns: Bonus/penalty to add to rating_final
    """
    bonus = 0.0

    # CRITICAL: 0% trainer = massive penalty (TUP R7 lesson)
    if trainer_win_pct == 0.0:
        return -2.5  # Eliminate from contention in most scenarios

    # Very low % trainer penalty (1-5%)
    if trainer_win_pct > 0.0 and trainer_win_pct < 0.05:
        bonus -= 1.0  # Significant penalty
    elif trainer_win_pct >= 0.05 and trainer_win_pct < 0.10:
        bonus -= 0.5  # Moderate penalty

    # High % trainer baseline
    if trainer_win_pct >= 0.25:  # Elite trainer (25%+)
        bonus += 0.3
    elif trainer_win_pct >= 0.20:  # Hot trainer (20-24%)
        bonus += 0.5
    elif trainer_win_pct >= 0.15:  # Above average (15-19%)
        bonus += 0.2

    # Hot trainer in last 14 days (4+ wins)
    if is_hot_l14:
        bonus += 0.3

    # High % trainer angle (2nd time Lasix 33%+)
    if is_2nd_lasix_high_pct:
        bonus += 0.8  # HUGE angle

    return bonus


# ======================== ROUTE & SPRINT CALIBRATION ========================
# Based on Pegasus WC G1 (9f route) and GP Turf Sprint (5f sprint) validation

def is_sprint_distance(distance_txt: str) -> bool:
    """Detect sprint distances (≤6.5f)"""
    if not distance_txt:
        return False
    try:
        if 'f' in distance_txt.lower():
            furlongs = float(distance_txt.lower().replace('f', '').strip())
            return furlongs <= 6.5
    except BaseException:
        pass
    return False


def calculate_sprint_post_position_bonus(post: int, distance: float, surface: str) -> float:
    """Inside posts dominate turf sprints (Rail 1.65 impact, Outside 8+ 0.44 death!)"""
    if distance > 6.5:
        return 0.0
    bonus = 0.0
    if surface.lower() in ['turf', 'tur', 't'] and distance <= 6.0:
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
    style_upper = style.upper() if style else ''
    if distance <= 5.5:
        if 'E' in style_upper and '/' not in style_upper:
            bonus += 0.20
        elif 'E' in style_upper and 'P' in style_upper:
            bonus += 0.10
        elif 'P' in style_upper:
            bonus -= 0.10
        elif 'S' in style_upper:
            bonus -= 0.20
    return bonus


def calculate_hot_combo_bonus(trainer_pct: float, jockey_pct: float, combo_pct: float) -> float:
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


def analyze_class_movement(past_races: List[Dict], today_class: str, today_purse: int) -> Dict[str, Any]:
    """
    COMPREHENSIVE: Analyze if horse is stepping up or down in class.

    Returns:
    - class_change: 'up', 'down', 'same', 'unknown'
    - class_delta: numeric change estimate
    - pattern: 'rising', 'dropping', 'stable'
    - bonus: rating adjustment
    """
    if not past_races or len(past_races) < 2:
        return {'class_change': 'unknown', 'class_delta': 0, 'pattern': 'unknown', 'bonus': 0.0}

    # Class hierarchy (higher = better class)
    class_hierarchy = {
        'Msw': 1,    # Maiden special weight
        'Mcl': 2,    # Maiden claiming
        'Clm': 3,    # Claiming
        'Str': 4,    # Starter allowance
        'Aoc': 5,    # Allowance optional claiming
        'Alw': 6,    # Allowance
        'Stk': 8,    # Stakes
        'G3': 9,     # Grade 3
        'G2': 10,    # Grade 2
        'G1': 11     # Grade 1
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
        race_class = race.get('class', '')
        for key in class_hierarchy:
            if key.lower() in race_class.lower():
                recent_levels.append(class_hierarchy[key])
                break

    if not recent_levels:
        return {'class_change': 'unknown', 'class_delta': 0, 'pattern': 'unknown', 'bonus': 0.0}

    avg_recent = sum(recent_levels) / len(recent_levels)
    class_delta = today_level - avg_recent

    # Determine change
    if class_delta > 1:
        class_change = 'up'
        bonus = -0.10  # Stepping up = tougher
    elif class_delta < -1:
        class_change = 'down'
        bonus = +0.12  # Dropping down = easier
    else:
        class_change = 'same'
        bonus = 0.0

    # Determine pattern (last 3 races)
    pattern = 'stable'
    if len(recent_levels) >= 3:
        if recent_levels[0] > recent_levels[1] > recent_levels[2]:
            pattern = 'rising'
            bonus += 0.05  # Positive progression
        elif recent_levels[0] < recent_levels[1] < recent_levels[2]:
            pattern = 'dropping'
            bonus += 0.08  # Finding easier spots

    return {
        'class_change': class_change,
        'class_delta': class_delta,
        'pattern': pattern,
        'bonus': float(np.clip(bonus, -0.15, 0.20))
    }


def analyze_distance_pattern(past_races: List[Dict], today_distance: str) -> Dict[str, Any]:
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
        return {'distance_change': 'unknown', 'distance_delta': 0, 'bonus': 0.0}

    # Convert distance to furlongs
    def distance_to_furlongs(dist_str: str) -> float:
        dist_str = dist_str.lower().strip()
        if 'f' in dist_str:
            return float(dist_str.replace('f', '').strip())
        elif 'm' in dist_str:
            # 1M = 8F, 1 1/16M = 8.5F, 1 1/8M = 9F
            if '1/16' in dist_str:
                return 8.5
            elif '1/8' in dist_str:
                return 9.0
            elif '3/16' in dist_str:
                return 9.5
            elif '1/4' in dist_str:
                return 10.0
            else:
                return 8.0  # Default 1 mile
        # Unrecognized format - warn and use default
        if dist_str and dist_str != '6.0':
            st.warning(f"⚠️ Unrecognized distance format '{dist_str}' - using 6f default")
        return 6.0  # Default

    today_furlongs = distance_to_furlongs(today_distance)

    # Analyze past distances
    past_distances = []
    experience_count = 0
    figs_at_distance = []

    for race in past_races[:8]:  # Last 8 races
        race_dist = race.get('distance', '')
        race_furlongs = distance_to_furlongs(race_dist)
        past_distances.append(race_furlongs)

        # Count experience at today's distance (within 0.5F)
        if abs(race_furlongs - today_furlongs) <= 0.5:
            experience_count += 1
            if race.get('speed_fig', 0) > 0:
                figs_at_distance.append(race['speed_fig'])

    if not past_distances:
        return {'distance_change': 'unknown', 'distance_delta': 0, 'bonus': 0.0}

    avg_past_dist = sum(past_distances) / len(past_distances)
    distance_delta = today_furlongs - avg_past_dist

    # Determine change type
    bonus = 0.0
    if distance_delta > 1.5:
        distance_change = 'stretch_out'
        # Stretching out 2+ furlongs
        if experience_count >= 2:
            bonus = +0.05  # Has experience at distance
        else:
            bonus = -0.08  # Unproven at distance

    elif distance_delta < -1.5:
        distance_change = 'cut_back'
        # Cutting back in distance usually helps speed horses
        bonus = +0.08

    else:
        distance_change = 'same'
        if experience_count >= 3:
            bonus = +0.05  # Comfortable at distance

    best_fig = max(figs_at_distance) if figs_at_distance else 0

    return {
        'distance_change': distance_change,
        'distance_delta': distance_delta,
        'today_furlongs': today_furlongs,
        'experience_count': experience_count,
        'best_fig_at_distance': best_fig,
        'bonus': float(np.clip(bonus, -0.10, 0.10))
    }


def analyze_form_cycle(past_races: List[Dict]) -> Dict[str, Any]:
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
        return {'cycle': 'unknown', 'trend_score': 0.0, 'bonus': 0.0}

    # Get last 3-5 races
    recent = past_races[:5]
    finishes = [r.get('finish_pos', 0) for r in recent if r.get('finish_pos', 0) > 0]
    figs = [r.get('speed_fig', 0) for r in recent if r.get('speed_fig', 0) > 0]

    if len(finishes) < 3 or len(figs) < 3:
        return {'cycle': 'unknown', 'trend_score': 0.0, 'bonus': 0.0}

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
    cycle = 'stable'
    bonus = 0.0

    if trend_score >= 0.75:
        cycle = 'improving'
        bonus = +0.15  # Strong positive form
    elif trend_score >= 0.25:
        cycle = 'peaking'
        bonus = +0.10  # Solid form
    elif trend_score <= -0.75:
        cycle = 'declining'
        bonus = -0.15  # Poor form
    elif trend_score <= -0.25:
        cycle = 'bottoming'
        bonus = -0.08  # Weak form

    # Bonus for consistency (all recent finishes in top 3)
    if all(f <= 3 for f in finishes[:3]):
        bonus += 0.05  # Consistent runner

    return {
        'cycle': cycle,
        'trend_score': trend_score,
        'last_3_finishes': finishes[:3],
        'last_3_figs': figs[:3],
        'bonus': float(np.clip(bonus, -0.20, 0.20))
    }


def calculate_workout_bonus(workout_data: Dict[str, Any]) -> float:
    """
    COMPREHENSIVE: Bonus for workout quality and recency.
    """
    if not workout_data:
        return 0.0

    bonus = 0.0

    # Rating bonus (B = bullet work = fastest of day)
    rating = workout_data.get('rating', '')
    if 'B' in rating or 'b' in rating:
        bonus += 0.10  # Bullet work
    elif 'H' in rating or 'h' in rating:
        bonus += 0.05  # Handily

    # Rank/percentile bonus (top 20% of works)
    percentile = workout_data.get('percentile', 0.5)
    if percentile <= 0.20:
        bonus += 0.08  # Top 20% work
    elif percentile <= 0.40:
        bonus += 0.05  # Top 40% work

    return float(np.clip(bonus, 0, 0.15))

# ======================== End ELITE ENHANCEMENTS ========================


def post_bias_score(post_bias_pick: str, post_str: str) -> float:
    pick = (post_bias_pick or "").strip().lower()
    # ELITE: Use optimized post parser (3x faster than regex)
    post = _parse_post_number(post_str)

    table = {
        "favors rail (1)": lambda p: MODEL_CONFIG['post_bias_rail_bonus'] if p == 1 else 0.0,
        "favors inner (1-3)": lambda p: MODEL_CONFIG['post_bias_inner_bonus'] if p and 1 <= p <= 3 else 0.0,
        "favors mid (4-7)": lambda p: MODEL_CONFIG['post_bias_mid_bonus'] if p and 4 <= p <= 7 else 0.0,
        "favors outside (8+)": lambda p: MODEL_CONFIG['post_bias_outside_bonus'] if p and p >= 8 else 0.0,
        "no significant post bias": lambda p: 0.0
    }
    fn = table.get(pick, table["no significant post bias"])
    return float(np.clip(fn(post), -0.5, 0.5))


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
        "favors rail (1)": lambda p: MODEL_CONFIG['post_bias_rail_bonus'] if p == 1 else 0.0,
        "favors inner (1-3)": lambda p: MODEL_CONFIG['post_bias_inner_bonus'] if p and 1 <= p <= 3 else 0.0,
        "favors mid (4-7)": lambda p: MODEL_CONFIG['post_bias_mid_bonus'] if p and 4 <= p <= 7 else 0.0,
        "favors outside (8+)": lambda p: MODEL_CONFIG['post_bias_outside_bonus'] if p and p >= 8 else 0.0,
        "no significant post bias": lambda p: 0.0
    }
    
    # Aggregate bonuses from all selected post biases
    for pick in post_bias_picks:
        pick_lower = (pick or "").strip().lower()
        fn = table.get(pick_lower, table["no significant post bias"])
        bonus = fn(post)
        if bonus > 0:  # Only add positive bonuses (horse matches this bias category)
            total_bonus += bonus
    
    return float(np.clip(total_bonus, -0.5, 0.5))


def compute_bias_ratings(df_styles: pd.DataFrame,
                         surface_type: str,
                         distance_txt: str,
                         condition_txt: str,
                         race_type: str,
                         running_style_bias,  # Can be list or str
                         post_bias_pick,  # Can be list or str
                         ppi_value: float = 0.0,  # ppi_value arg seems unused, ppi_map is recalculated
                         pedigree_per_horse: Optional[Dict[str, dict]] = None,
                         track_name: str = "",
                         pp_text: str = "",
                         figs_df: pd.DataFrame = None,
                         dynamic_weights: dict = None) -> pd.DataFrame:
    """
    Reads 'Cclass' and 'Cform' from df_styles (pre-built), adds Cstyle/Cpost/Cpace/Cspeed (+Atrack),
    plus Tier 2 bonuses (Impact Values, SPI, Surface Stats, AWD),
    sums to Arace and R. Returns rating table.

    ULTRATHINK V2: Can use unified rating engine if available and PP text provided.
    
    UPDATED: Now accepts lists of biases to aggregate bonuses from ALL selected biases.
    """
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
        "Arace",
        "R"]
    if df_styles is None or df_styles.empty:
        return pd.DataFrame(columns=cols)

    # ===== ULTRATHINK V2: Try unified engine first if available =====
    use_unified_engine = False
    if UNIFIED_ENGINE_AVAILABLE and ELITE_PARSER_AVAILABLE and pp_text and len(pp_text.strip()) > 100:
        try:
            # Parse with elite parser
            parser = GoldStandardBRISNETParser()
            horses = parser.parse_full_pp(pp_text, debug=False)

            # Validate parsing quality
            validation = parser.validate_parsed_data(horses, min_confidence=0.5)
            avg_confidence = validation.get('overall_confidence', 0.0)

            if avg_confidence >= 0.15 and validation.get('horses_parsed', 0) > 0:
                # High quality parse - use unified engine
                engine = UnifiedRatingEngine(softmax_tau=3.0)

                # Extract race metadata using elite parser's race header (most accurate)
                today_purse = 0
                final_distance = distance_txt
                extracted_race_type = race_type

                if hasattr(parser, 'race_header') and parser.race_header:
                    race_header = parser.race_header
                    today_purse = race_header.get('purse', 0)
                    if race_header.get('distance'):
                        final_distance = race_header.get('distance')
                    if race_header.get('race_type_normalized'):
                        extracted_race_type = race_header.get('race_type_normalized')

                # Fallback to comprehensive extraction if parser didn't get purse
                if today_purse == 0:
                    race_metadata = extract_race_metadata_from_pp_text(pp_text)
                    today_purse = race_metadata.get('purse_amount', 0)

                # Final fallback to session state
                if today_purse == 0:
                    today_purse = st.session_state.get('purse_val', 20000)

                # Get predictions with extracted metadata
                results_df = engine.predict_race(
                    pp_text=pp_text,
                    today_purse=today_purse,
                    today_race_type=extracted_race_type,
                    track_name=track_name,
                    surface_type=surface_type,
                    distance_txt=final_distance
                )

                if not results_df.empty:
                    # Filter to only include horses that are NOT scratched (i.e., in df_styles)
                    # Use normalized names for matching to handle apostrophes and spacing differences
                    # (normalize_horse_name is defined at top of file for consistency)

                    # Build mapping of normalized names to original names from Section A
                    section_a_names = {normalize_horse_name(h): h for h in df_styles['Horse'].tolist()}

                    # Filter and rename horses to match Section A names
                    results_df_filtered = results_df.copy()
                    matched_horses = []
                    for idx, row in results_df_filtered.iterrows():
                        parsed_name = row['Horse']
                        normalized = normalize_horse_name(parsed_name)
                        if normalized in section_a_names:
                            # Use the exact name from Section A
                            results_df_filtered.at[idx, 'Horse'] = section_a_names[normalized]
                            matched_horses.append(idx)

                    # Keep only matched horses
                    results_df_filtered = results_df_filtered.loc[matched_horses].copy()

                    if results_df_filtered.empty:
                        # All horses from unified engine were scratched, fall back to traditional
                        st.warning("⚠️ All unified engine horses are scratched (using fallback)")
                    else:
                        # Build figs_df from elite parser speed figures
                        figs_data = []
                        for _, row in results_df_filtered.iterrows():
                            horse_name = row['Horse']
                            # Get speed figures from elite parser's horses dict
                            if hasattr(parser, 'horses') and horse_name in parser.horses:
                                horse_obj = parser.horses[horse_name]
                                if horse_obj.speed_figures and len(horse_obj.speed_figures) > 0:
                                    figs_data.append({
                                        "Horse": horse_name,
                                        "Figures": horse_obj.speed_figures,
                                        "BestFig": max(horse_obj.speed_figures),
                                        "AvgTop2": horse_obj.avg_top2
                                    })
                        figs_df = pd.DataFrame(figs_data) if figs_data else pd.DataFrame()

                        # Convert unified engine output to app.py format
                        unified_ratings = pd.DataFrame({
                            "#": range(1, len(results_df_filtered) + 1),
                            "Post": results_df_filtered['Post'].astype(str),
                            "Horse": results_df_filtered['Horse'],
                            "Style": results_df_filtered.get('Pace_Style', 'NA'),
                            "Quirin": results_df_filtered.get('Quirin', 0.0),
                            "Cstyle": results_df_filtered.get('Cstyle', 0.0),
                            "Cpost": results_df_filtered.get('Cpost', 0.0),
                            "Cpace": results_df_filtered.get('Cpace', 0.0),
                            "Cspeed": results_df_filtered.get('Cspeed', 0.0),
                            "Cclass": results_df_filtered.get('Cclass', 0.0),
                            "Cform": results_df_filtered.get('Cform', 0.0),
                            "Atrack": results_df_filtered.get('A_Track', 0.0),  # Use actual track advantage
                            "Arace": results_df_filtered['Rating'],
                            "R": results_df_filtered['Rating'],
                            "Parsing_Confidence": results_df_filtered.get('Parsing_Confidence', avg_confidence)
                        })

                        # Add success message
                        scratched_count = len(results_df) - len(results_df_filtered)
                        if scratched_count > 0:
                            st.info(f"🎯 Using Unified Rating Engine (Elite Parser confidence: {avg_confidence:.1%}) - {scratched_count} scratched horse(s) excluded")
                        else:
                            st.info(f"🎯 Using Unified Rating Engine (Elite Parser confidence: {avg_confidence:.1%})")

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
        return df_styles

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

    # Calculate speed component from figs_df
    speed_map = {}
    if figs_df is not None and not figs_df.empty and "AvgTop2" in figs_df.columns:
        race_avg_fig = figs_df["AvgTop2"].mean()
        for _, fig_row in figs_df.iterrows():
            horse_name = fig_row["Horse"]
            horse_fig = fig_row["AvgTop2"]
            # Normalize to race average: positive means faster than average
            speed_map[horse_name] = (horse_fig - race_avg_fig) * MODEL_CONFIG['speed_fig_weight']

    rows = []
    
    # Convert single values to lists for uniform processing
    style_biases = running_style_bias if isinstance(running_style_bias, list) else [running_style_bias]
    post_biases_list = post_bias_pick if isinstance(post_bias_pick, list) else [post_bias_pick]
    
    for _, row in df_styles.iterrows():
        post = str(row.get("Post", row.get("#", "")))
        name = str(row.get("Horse"))
        style = _style_norm(row.get("Style") or row.get("OverrideStyle") or row.get("DetectedStyle"))
        quirin = row.get("Quirin", np.nan)  # Keep as potential NaN

        # Use multi-bias functions to aggregate bonuses from ALL selected biases
        cstyle = style_match_score_multi(style_biases, style, quirin)
        cpost = post_bias_score_multi(post_biases_list, post)
        cpace = float(ppi_map.get(name, 0.0))
        cspeed = float(speed_map.get(name, 0.0))  # Speed component from figures

        # Track bias with dynamic weight multiplier
        dw = dynamic_weights or {}
        track_bias_mult = dw.get('track_bias', 1.0)
        a_track = _get_track_bias_delta(track_name, surface_type, distance_txt, style, post) * track_bias_mult

        # Get pre-computed Cclass and Cform from df_styles (calculated in Section A)
        c_class = float(row.get("Cclass", 0.0))
        c_form = float(row.get("Cform", 0.0))

        # ======================== Tier 2 Bonuses ========================
        tier2_bonus = 0.0

        # ELITE: Weather Impact
        weather_data = st.session_state.get('weather_data', None)
        if weather_data:
            tier2_bonus += calculate_weather_impact(weather_data, style, distance_txt)

        # ELITE: Jockey/Trainer Performance Impact
        tier2_bonus += calculate_jockey_trainer_impact(name, pp_text)

        # ELITE: Track Condition Granularity
        track_info = st.session_state.get('track_condition_detail', None)
        if track_info:
            tier2_bonus += calculate_track_condition_granular(track_info, style, post)

        # ======================== CLAIMING RACE PACE CAP (TUP R7 Feb 2026) ========================
        # Failed pick #3 Forest Acclamation: +0.84 pace × 1.5 = +1.26 rating boost
        # BUT: 0% trainer + 3/50 career wins + worst Prime Power = fatal flaws ignored
        # In claiming races: Talent/form/trainer > pace tactics
        # Cap pace bonus to prevent over-reliance on pace scenarios in cheap races
        pace_cap_applied = False

        # ======================== DISTANCE CATEGORY DETECTION ========================
        is_marathon = is_marathon_distance(distance_txt)
        is_sprint = is_sprint_distance(distance_txt)

        # Parse distance as furlongs for numeric comparisons
        race_furlongs = 8.0  # Default assumption
        try:
            if 'f' in distance_txt.lower():
                race_furlongs = float(distance_txt.lower().replace('f', '').strip())
        except BaseException:
            pass

        # ======================== DISTANCE MOVEMENT ANALYSIS (NEW) ========================
        # Integrates analyze_distance_pattern() into tier2_bonus calculations
        # Rewards horses proven at today's distance, penalizes unproven stretching out
        try:
            # Parse past races from PP text for this horse
            past_races = parse_recent_class_levels(pp_text)  # Reuse existing parser
            if past_races:
                distance_analysis = analyze_distance_pattern(past_races, distance_txt)
                distance_bonus = distance_analysis.get('bonus', 0.0)
                
                # Additional bonus for experience at distance
                experience_count = distance_analysis.get('experience_count', 0)
                if experience_count >= 3:
                    distance_bonus += 0.10  # Proven at distance
                elif experience_count == 0:
                    distance_bonus -= 0.08  # Never raced this distance
                
                # Stretch out / cut back adjustments
                distance_change = distance_analysis.get('distance_change', 'unknown')
                if distance_change == 'stretch_out' and not is_marathon:
                    # Check if horse has good closing ability (LP figures)
                    # Horses stretching out need strong late pace
                    pass  # Already handled in analyze_distance_pattern
                elif distance_change == 'cut_back':
                    # Cutting back often positive for speed horses
                    if style in ['E', 'E/P']:
                        distance_bonus += 0.05  # Speed horse cutting back
                
                tier2_bonus += distance_bonus
        except BaseException:
            pass  # Fail silently if distance analysis fails

        # COMPREHENSIVE: Workout Analysis (V2 with marathon awareness)
        # FUTURE ENHANCEMENT: Full workout parsing from BRISNET PP text
        # Currently using basic workout detection. Advanced parsing would extract:
        # - Workout date, distance, time, track condition, rank/total
        # - Would feed calculate_workout_bonus_v2() for more precise adjustments

        # COMPREHENSIVE: Race History & Pattern Analysis
        # FUTURE ENHANCEMENT: Full race history parsing for pattern detection
        # Currently using ratings from Section A. Advanced parsing would analyze:
        # - Class movement patterns (up/down class trends)
        # - Distance preferences and performance patterns
        # - Form cycles (improving vs declining form)

        # ======================== MARATHON CALIBRATION (12f+) ========================
        if is_marathon:
            # Layoff adjustment (freshening can help at marathons)
            # NOTE: Using default 49 days (optimal marathon freshening range)
            # Could extract actual days from PP text in future enhancement
            tier2_bonus += calculate_layoff_bonus(49, is_marathon)

            # Lightly-raced improver bonus (fresh legs at marathons)
            life_pattern = r'Life:\s+(\d+)\s+'
            life_match = re.search(life_pattern, pp_text)
            if life_match:
                career_starts = int(life_match.group(1))
                tier2_bonus += calculate_experience_bonus(career_starts, is_marathon)

        # ======================== SPRINT CALIBRATION (≤6.5f) ========================
        elif is_sprint:
            # Post position CRITICAL at turf sprints (inside good, outside death!)
            try:
                post_num = int(post)
                tier2_bonus += calculate_sprint_post_position_bonus(post_num, race_furlongs, surface_type)
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

        # 2. SPI (Sire Performance Index) bonus
        if name in spi_values:
            tier2_bonus += calculate_spi_bonus(spi_values[name])

        # 3. Surface Specialty bonus
        if name in surface_stats:
            stats = surface_stats[name]
            if surface_type.lower() == 'tur' and 'turf_pct' in stats:
                tier2_bonus += calculate_surface_specialty_bonus(stats['turf_pct'], 'tur')
            elif surface_type.lower() in ['aw', 'all-weather', 'synthetic'] and 'aw_pct' in stats:
                tier2_bonus += calculate_surface_specialty_bonus(stats['aw_pct'], 'aw')

        # 4. AWD (Distance Mismatch) penalty
        if name in awd_analysis:
            tier2_bonus += calculate_awd_mismatch_penalty(awd_analysis[name])

        # 5. CAREER FUTILITY PENALTY (TUP R7 Feb 2026)
        # #3 Forest Acclamation: 3 wins in 50 starts (6%) finished 4th despite model picking to win
        # Chronic losers rarely break through regardless of pace/class advantages
        career_win_pct = 0.0
        try:
            # Parse career record: "Life: 50 3 - 1 - 7 $53,234 79"
            career_match = re.search(r'Life:\s+(\d+)\s+(\d+)\s+-\s+\d+\s+-\s+\d+', pp_text)
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

        # 6. 2ND OFF LAYOFF BONUS (TUP R7 Feb 2026)
        # Winner #4 If You Want It: 2nd start after layoff, improved from Speed 62
        # BRISNET shows: "2nd off layoff 65 22% 60%" (22% win rate, 60% ITM)
        is_second_off_layoff = False
        try:
            # Check for "2nd off layoff" stat with high win%
            layoff_match = re.search(r'2nd off layoff\s+(\d+)\s+(\d+)%\s+(\d+)%', pp_text)
            if layoff_match:
                layoff_starts = int(layoff_match.group(1))
                layoff_win_pct = int(layoff_match.group(2)) / 100.0
                layoff_itm_pct = int(layoff_match.group(3)) / 100.0

                if layoff_starts >= 3:  # Meaningful sample size
                    if layoff_win_pct >= 0.20:  # 20%+ win rate
                        tier2_bonus += 0.8  # Strong bonus
                        is_second_off_layoff = True
                    elif layoff_win_pct >= 0.15:  # 15-19% win rate
                        tier2_bonus += 0.5  # Moderate bonus
                        is_second_off_layoff = True
                    elif layoff_itm_pct >= 0.50:  # 50%+ ITM even if lower win%
                        tier2_bonus += 0.3  # Small bonus
                        is_second_off_layoff = True
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
            best_speed_pattern = rf'{re.escape(name)}\s+(\d+)'
            # Look in "Best Speed at Dist" section
            best_speed_section = re.search(r'# Best Speed at Dist(.{0,500})', pp_text, re.DOTALL)
            if best_speed_section:
                section_text = best_speed_section.group(1)
                horse_best = re.search(best_speed_pattern, section_text)
                if horse_best:
                    best_speed_at_dist = int(horse_best.group(1))

                    # Find all horses' best speeds to determine ranking
                    all_best_speeds = re.findall(r'\d+\s+[A-Za-z\s\']+\s+(\d+)', section_text)
                    if all_best_speeds:
                        speeds = [int(s) for s in all_best_speeds]
                        speeds_sorted = sorted(speeds, reverse=True)

                        if best_speed_at_dist > 0:
                            rank = speeds_sorted.index(best_speed_at_dist) + \
                                1 if best_speed_at_dist in speeds_sorted else 999

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
        is_hot_l14 = False
        is_2nd_lasix_high_pct = False
        try:
            # Parse trainer win %: "Trnr: Eikleberry Kevin (50 11-7-4 22%)"
            trainer_match = re.search(r'Trnr:.*?\(\d+\s+\d+-\d+-\d+\s+(\d+)%\)', pp_text)
            if trainer_match:
                trainer_win_pct = int(trainer_match.group(1)) / 100.0

            # Check for hot trainer in last 14 days (look for patterns like "9 4-0-0")
            hot_l14_match = re.search(r'Hot Trainer in last 14 days \((\d+) (\d+)-\d+-\d+\)', pp_text)
            if hot_l14_match and int(hot_l14_match.group(2)) >= 4:
                is_hot_l14 = True

            # Check for 2nd time Lasix with high % angle
            if '2nd time Lasix' in pp_text and trainer_win_pct >= 0.25:
                is_2nd_lasix_high_pct = True

            tier2_bonus += calculate_hot_trainer_bonus(trainer_win_pct, is_hot_l14, is_2nd_lasix_high_pct)
        except BaseException:
            pass

        # ======================== End Tier 2 Bonuses ========================

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
                parser_class_weight = race_class_data['weight']['class_weight']
                # Get quality and normalize to lowercase + map Medium -> mid
                quality_raw = race_class_data['summary']['quality']  # 'Elite'/'High'/'Medium'/'Low'
                quality_map = {'Elite': 'elite', 'High': 'high', 'Medium': 'mid', 'Low': 'low', 'Minimal': 'low'}
                race_quality = quality_map.get(quality_raw, 'mid')
                st.success(f"✅ Parser: {race_class_data['summary']['class_type']} | Level {race_class_data['hierarchy']['final_level']} | Weight {parser_class_weight:.2f} | Quality: {race_quality}")
            except Exception as e:
                st.error(f"❌ Parser failed in compute_bias_ratings: {e}")
                import traceback
                st.code(traceback.format_exc())
                pass

        # If parser unavailable, fall back to legacy race quality detection
        if parser_class_weight is None:
            try:
                race_metadata = extract_race_metadata_from_pp_text(pp_text)
                race_type_clean = race_metadata.get('race_type_clean', '')
                purse_amount = race_metadata.get('purse_amount', 0)

                # Map to quality tier
                if race_type_clean == 'stakes_graded':
                    race_quality = "elite"
                elif race_type_clean == 'stakes' and purse_amount >= 200000:
                    race_quality = "elite"
                elif race_type_clean in ['allowance', 'allowance_optional']:
                    race_quality = "high"
                elif race_type_clean == 'claiming':
                    race_quality = "low"
                elif race_type_clean == 'maiden_claiming':
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
            elif race_quality == "mid" or race_quality == "mid-maiden" or race_quality == "high":
                speed_multiplier = 2.2
                class_weight = 2.5
                form_weight = 2.0
            else:  # elite/stakes
                speed_multiplier = 1.8
                class_weight = 3.0  # Still wrong for G1, but best we can do without parser
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
        class_form_mult = dw.get('class_form', 1.0)
        pace_speed_mult = dw.get('pace_speed', 1.0)
        style_post_mult = dw.get('style_post', 1.0)
        track_bias_mult = dw.get('track_bias', 1.0)

        # Calculate weighted components with dynamic multipliers applied
        weighted_components = (
            c_class * class_weight * class_form_mult +
            c_form * form_weight * class_form_mult +
            cspeed * speed_multiplier * pace_speed_mult +
            pace_contribution * pace_speed_mult +
            cstyle * 1.2 * style_post_mult +
            cpost * 0.8 * style_post_mult
        )

        # Now check if Prime Power is available
        prime_power_raw = safe_float(row.get('Prime Power', 0.0), 0.0)
        if prime_power_raw > 0:
            # Normalize Prime Power (typical range: 110-130, clip outliers to 0-2 scale)
            pp_normalized = np.clip((prime_power_raw - 110) / 20, 0, 2)  # 0 to 2 scale (allows up to 150)
            pp_contribution = pp_normalized * 10  # Scale to match component range

            # Determine optimal PP weight based on surface and distance
            # Parse distance for route vs sprint classification
            distance_furlongs = 8.0  # Default
            try:
                if 'f' in distance_txt.lower():
                    distance_furlongs = float(distance_txt.lower().replace('f', '').strip())
                elif 'mile' in distance_txt.lower():
                    if '1mile' in distance_txt.lower().replace(' ', ''):
                        distance_furlongs = 8.0
                    elif '1.5' in distance_txt or '1 1/2' in distance_txt:
                        distance_furlongs = 12.0
                    elif '1.25' in distance_txt or '1 1/4' in distance_txt:
                        distance_furlongs = 10.0
                    elif '1.125' in distance_txt or '1 1/8' in distance_txt:
                        distance_furlongs = 9.0
            except BaseException:
                pass

            # ═══════════════════════════════════════════════════════════════════════
            # UNIVERSAL RACE QUALITY DETECTION (All Tracks, All Types, All Purses)
            # ═══════════════════════════════════════════════════════════════════════

            # STEP 1: Extract race metadata from PP text (PREFERRED - most accurate)
            race_metadata = extract_race_metadata_from_pp_text(pp_text)

            # STEP 2: Try legacy race_type extraction if PP text detection failed
            purse_amount = race_metadata['purse_amount']
            race_type_clean = race_metadata['race_type_clean']
            detection_confidence = race_metadata['confidence']

            if purse_amount == 0 and race_type:
                # Fallback: Try to infer from race_type parameter
                inferred_purse = infer_purse_from_race_type(race_type)
                if inferred_purse and inferred_purse > 0:
                    purse_amount = inferred_purse
                    detection_confidence = 0.7

            if race_type_clean == 'unknown' and race_type:
                # Fallback: Parse race_type parameter
                race_type_lower = str(race_type).lower()
                if 'g1' in race_type_lower or 'g2' in race_type_lower or 'g3' in race_type_lower:
                    race_type_clean = 'stakes_graded'
                elif 'stake' in race_type_lower or 'stk' in race_type_lower:
                    race_type_clean = 'stakes'
                elif 'allowance' in race_type_lower or 'alw' in race_type_lower:
                    race_type_clean = 'allowance'
                elif 'aoc' in race_type_lower or 'optional' in race_type_lower:
                    race_type_clean = 'allowance_optional'
                elif 'maiden claiming' in race_type_lower or 'mcl' in race_type_lower:
                    race_type_clean = 'maiden_claiming'
                elif 'claiming' in race_type_lower or 'clm' in race_type_lower:
                    race_type_clean = 'claiming'
                elif 'maiden' in race_type_lower or 'msw' in race_type_lower:
                    race_type_clean = 'maiden_special_weight'
                detection_confidence = 0.6

            # ═══════════════════════════════════════════════════════════════════════
            # ENHANCED RACE QUALITY DETECTION (When Prime Power IS available)
            # ═══════════════════════════════════════════════════════════════════════
            # When PP is present, we can refine the race quality detection with purse details

            # STEP 1: Extract detailed race metadata from PP text
            race_metadata = extract_race_metadata_from_pp_text(pp_text)
            purse_amount = race_metadata['purse_amount']
            race_type_clean = race_metadata['race_type_clean']
            detection_confidence = race_metadata['confidence']

            # STEP 2: Override/refine race_quality if we have better purse/type info
            if purse_amount >= 500000:
                race_quality = "elite"
            elif purse_amount >= 150000 and race_quality not in ["elite", "high"]:
                race_quality = "high"
            elif race_type_clean == 'stakes_graded' and race_quality != "elite":
                race_quality = "elite"

            # STEP 3: Display detected metadata for user validation
            with st.expander("🔍 Race Classification & Detection Details", expanded=False):
                col1, col2, col3, col4 = st.columns(4)
                with col1:
                    st.metric("Detected Purse", f"${purse_amount:,}" if purse_amount > 0 else "Unknown")
                with col2:
                    st.metric("Race Type", race_type_clean.replace('_', ' ').title())
                with col3:
                    st.metric("Quality Tier", race_quality.upper())
                with col4:
                    confidence_color = "🟢" if detection_confidence >= 0.8 else "🟡" if detection_confidence >= 0.5 else "🔴"
                    st.metric("Confidence", f"{confidence_color} {detection_confidence:.0%}")

                st.caption(f"**Detection Method:** {race_metadata['detection_method'].replace('_', ' ').title()}")
                st.caption(f"**Raw Race Type:** {race_metadata['race_type'] or race_type or 'Not detected'}")

                if detection_confidence < 0.5:
                    st.warning("⚠️ **Low confidence detection** - Verify race type is correctly classified")
                if purse_amount == 0:
                    st.info("ℹ️ Purse not detected - Using race type classification only")

            # Apply pace component with claiming race cap (TUP R7)
            pace_contribution = cpace * 1.5

            # Cap pace bonus in claiming races to prevent over-reliance
            if race_quality == "low" or race_quality == "low-maiden":
                if pace_contribution > 0.75:
                    pace_contribution = 0.75  # Cap at +0.75 in claiming
                    pace_cap_applied = True

            # Apply dynamic weights (same as non-PP path)
            dw = dynamic_weights or {}
            class_form_mult = dw.get('class_form', 1.0)
            pace_speed_mult = dw.get('pace_speed', 1.0)
            style_post_mult = dw.get('style_post', 1.0)
            track_bias_mult = dw.get('track_bias', 1.0)

            weighted_components = (
                c_class * class_weight * class_form_mult +
                c_form * form_weight * class_form_mult +
                cspeed * speed_multiplier * pace_speed_mult +  # Boost speed in claiming/allowance
                pace_contribution * pace_speed_mult +  # Capped in claiming races
                cstyle * 1.2 * style_post_mult +
                cpost * 0.8 * style_post_mult
            )

            # ═══════════════════════════════════════════════════════════════════════
            # Surface-adaptive ratio selection (with quality adjustment)
            # ═══════════════════════════════════════════════════════════════════════
            surface_lower = (surface_type or "dirt").lower()

            if 'turf' in surface_lower or 'tur' in surface_lower or 'grass' in surface_lower:
                # Turf racing: PP has no predictive value (GP R1: highest PP horses lost)
                # Use pure component model (pace, form, style, jockey are what matter)
                pp_weight, comp_weight = 0.0, 1.0
            elif 'synth' in surface_lower or 'aw' in surface_lower or 'all-weather' in surface_lower or 'tapeta' in surface_lower:
                # Synthetic: Consistent surface, speed matters but not as much as dirt
                pp_weight, comp_weight = 0.75, 0.25
            else:  # Dirt (default)
                # MAIDEN RACE DETECTION: Adjust PP weight based on field experience
                # GP R2 Learning: Maiden races with many first-timers need balanced weighting
                # because PP data is sparse and components predict debut quality better

                is_maiden = False
                if race_type:
                    race_type_lower = str(race_type).lower()
                    is_maiden = any(keyword in race_type_lower for keyword in
                                    ['maiden', 'mdn', 'msw', 'mcl', 'mdn sp wt', 'maiden sp wt'])

                if is_maiden:
                    # Count horses with valid PP data in this race
                    # For maiden races, check if field is mostly first-timers or experienced
                    horses_with_pp = 0
                    total_horses = len(df_styles) if df_styles is not None else 0

                    if df_styles is not None and not df_styles.empty:
                        for _, h_row in df_styles.iterrows():
                            h_pp = safe_float(h_row.get('Prime Power', 0.0), 0.0)
                            if h_pp > 0:
                                horses_with_pp += 1

                    horses_without_pp = total_horses - horses_with_pp

                    # Adjust PP weight based on field composition
                    if horses_without_pp >= horses_with_pp and horses_without_pp > 0:
                        # Majority are first-timers: Equal weight to PP and components
                        # GP R2: 6 first-timers, winner had PP + pace, top pick (first-timer) placed 2nd
                        if distance_furlongs <= 7.0:
                            pp_weight, comp_weight = 0.50, 0.50  # Sprint: Balanced approach
                        else:
                            pp_weight, comp_weight = 0.40, 0.60  # Route: Components slightly favored
                    else:
                        # Majority have racing experience: Favor PP but less than non-maiden
                        # Small PP differences not as decisive in maiden races
                        if distance_furlongs <= 7.0:
                            pp_weight, comp_weight = 0.70, 0.30  # Sprint: Still favor PP
                        else:
                            pp_weight, comp_weight = 0.60, 0.40  # Route: More balanced
                else:
                    # Non-maiden race: Apply RACE QUALITY + CLASS DROPPER analysis
                    # Higher quality races = more reliable PP (better horses, consistent performance)

                    # Check for class dropper scenario (SA R6)
                    class_spread = 0.0
                    if df_styles is not None and not df_styles.empty and 'Class Rating' in df_styles.columns:
                        class_ratings = []
                        for _, h_row in df_styles.iterrows():
                            cr = safe_float(h_row.get('Class Rating', 0.0), 0.0)
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
                            # CLAIMING FIX: TUP R4/R5 proved 85% PP weight produces INVERSE correlation
                            # Winner #3 in R5 had HIGHEST Speed LR (82) and was P style (survived speed duel)
                            # Winner in R4 had LOWEST PP (76.8) but had Smart Money + post bias
                            # Claiming races = chaos: Speed LR, tactics, Smart Money matter MORE than PP
                            base_pp, base_comp = 0.62, 0.38  # Dramatically lower PP, boost components

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
                            # CLAIMING ROUTE FIX: Even lower PP weight for claiming routes
                            # Pace stamina and tactics dominate in cheap route races
                            base_pp, base_comp = 0.55, 0.45  # Favor components heavily

                        # Class dropper adjustment (routes)
                        if class_spread > 2.0:
                            pp_weight = base_pp - 0.10
                            comp_weight = base_comp + 0.10
                        else:
                            pp_weight, comp_weight = base_pp, base_comp

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
        # PACE SCENARIO BONUS: Detect speed duels favoring closers
        # ═══════════════════════════════════════════════════════════════
        # TUP R5: 7 E/EP types created speed duel, P runner won from back
        # Count E/EP types in field to detect likely speed duels
        pace_scenario_bonus = 0.0
        if df_styles is not None and not df_styles.empty:
            speed_types = 0
            for _, h_row in df_styles.iterrows():
                h_style = str(h_row.get('Style', '')).upper()
                if 'E' in h_style:  # E or E/P types
                    speed_types += 1

            # If 6+ speed horses and this horse is P or S, boost rating
            if speed_types >= 6 and (style in ['P', 'S'] or 'P' in style or 'S' in style):
                pace_scenario_bonus = 1.5  # Significant boost for closers in speed duels

        R += pace_scenario_bonus

        # ELITE: Single-pass outlier clipping with data quality monitoring
        # Typical racing range: -5 to +20. Values beyond suggest parsing errors or unrealistic bonuses
        if R > 30 or R < -10:
            R = np.clip(R, -5, 20)

        # Ensure Quirin is formatted correctly for display (handle NaN)
        quirin_display = quirin if pd.notna(quirin) else None

        rows.append({
            "#": post, "Post": post, "Horse": name, "Style": style, "Quirin": quirin_display,
            "Cstyle": round(cstyle, 2), "Cpost": round(cpost, 2), "Cpace": round(cpace, 2),
            "Cspeed": round(cspeed, 2), "Cclass": round(c_class, 2), "Cform": round(c_form, 2),
            "Atrack": round(a_track, 2), "Arace": round(arace, 2), "R": round(R, 2)
        })
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(by="R", ascending=False)


def fair_probs_from_ratings(ratings_df: pd.DataFrame,
                            ml_odds_dict: Optional[Dict[str, float]] = None) -> Dict[str, float]:
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
    df['R_numeric'] = pd.to_numeric(df['R'], errors='coerce')

    # GOLD STANDARD: Handle NaN with intelligent fallback
    median_r = df['R_numeric'].median()
    if pd.isna(median_r) or not np.isfinite(median_r):
        # All ratings are invalid - use mean as fallback, or 0
        mean_r = df['R_numeric'].mean()
        median_r = mean_r if np.isfinite(mean_r) else 0.0
    df['R_numeric'].fillna(median_r, inplace=True)

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

    # ML ODDS REALITY CHECK: Prevent longshots from getting unrealistic probabilities
    # This adds common sense - the market odds reflect real race experience
    if ml_odds_dict:
        adjusted = False
        for horse, prob in result.items():
            ml_odds = ml_odds_dict.get(horse, 5.0)

            # Progressive caps based on ML odds:
            # 10/1 or more: cap at 25%
            # 15/1 or more: cap at 20%
            # 20/1 or more: cap at 15%
            # 30/1 or more: cap at 10%
            if ml_odds >= 30.0 and prob > 0.10:
                result[horse] = 0.10
                adjusted = True
            elif ml_odds >= 20.0 and prob > 0.15:
                result[horse] = 0.15
                adjusted = True
            elif ml_odds >= 15.0 and prob > 0.20:
                result[horse] = 0.20
                adjusted = True
            elif ml_odds >= 10.0 and prob > 0.25:
                result[horse] = 0.25
                adjusted = True

        # If we adjusted any probabilities, renormalize
        if adjusted:
            total_prob = sum(result.values())
            if total_prob > 0:
                result = {h: p / total_prob for h, p in result.items()}

    return result


# Build scenarios - UPDATED: Create unified scenario using ALL selected biases
# Instead of cartesian product, we aggregate bonuses from all selections
scenarios = [("COMBINED", "COMBINED")]  # Single unified scenario
tabs = st.tabs(["Combined Bias Analysis"])
all_scenario_ratings = {}

# ============ DYNAMIC WEIGHT FUNCTIONS (Race Parameter Integration) ============
# These functions dynamically adjust factor weights based on race conditions


def get_weight_preset(surface: str, distance: str) -> dict:
    """
    Generate base weights based on surface type and distance.
    
    Surface Logic:
    - Dirt: Speed figures and pace more predictive
    - Turf: Class/form and pedigree more predictive
    - Synthetic: Balanced approach
    
    Distance Logic:
    - Sprint (≤7f): Pace/speed dominates
    - Route (≥8f): Class/stamina more important
    """
    surf = (surface or "Dirt").strip().lower()
    dist_bucket = distance_bucket(distance) if distance else "8f+"
    
    base = {
        "class_form": 1.0,
        "pace_speed": 1.0,
        "style_post": 1.0,
        "track_bias": 1.0,
        "trs_jky": 1.0
    }
    
    # Surface adjustments
    if "turf" in surf:
        base["class_form"] = 1.25  # Class more predictive on turf
        base["pace_speed"] = 0.85  # Pace less dominant on turf
        base["track_bias"] = 1.15  # Turf biases can be strong
    elif "synth" in surf or "all-weather" in surf:
        base["class_form"] = 1.10
        base["pace_speed"] = 0.95
    else:  # Dirt
        base["pace_speed"] = 1.15  # Speed/pace more dominant on dirt
        base["class_form"] = 1.0
    
    # Distance adjustments
    if dist_bucket == "≤6f":  # Sprint
        base["pace_speed"] *= 1.20  # Pace critical in sprints
        base["class_form"] *= 0.90  # Class less predictive in sprints
    elif dist_bucket == "6.5–7f":  # Middle distance
        base["pace_speed"] *= 1.05
        base["class_form"] *= 1.05
    else:  # Route (8f+)
        base["pace_speed"] *= 0.85  # Pace less dominant in routes
        base["class_form"] *= 1.20  # Class/stamina key in routes
    
    return base


def apply_strategy_profile_to_weights(weights: dict, profile: str) -> dict:
    """
    Adjust weights based on user's strategy profile.
    
    Confident: Favor top-rated horses (class/form emphasis)
    Value Hunter: Look for overlays (pace/track bias emphasis)
    """
    if not weights:
        return {"class_form": 1.0, "trs_jky": 1.0}
    
    w = weights.copy()
    profile_lower = (profile or "").lower()
    
    if "value" in profile_lower:
        # Value hunters look for pace/bias edges
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.15
        w["track_bias"] = w.get("track_bias", 1.0) * 1.20
        w["class_form"] = w.get("class_form", 1.0) * 0.90
    else:  # Confident
        # Confident players trust class/form
        w["class_form"] = w.get("class_form", 1.0) * 1.15
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.95
    
    return w


def adjust_by_race_type(weights: dict, race_type: str) -> dict:
    """
    Adjust weights based on race type/class.
    
    Stakes (G1-G3): Class separation critical, pace less dominant
    Allowance: Balanced
    Claiming: Form/recent races more important than class
    Maiden: Pedigree/debut angles more important
    """
    if not weights:
        return {"class_form": 1.0, "trs_jky": 1.0}
    
    w = weights.copy()
    rt = (race_type or "").lower()
    
    if "g1" in rt or "g2" in rt:
        # Elite stakes - class differences narrow, form is key
        w["class_form"] = w.get("class_form", 1.0) * 1.30
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.85
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.15  # Top jockeys matter
    elif "g3" in rt or "stakes" in rt:
        w["class_form"] = w.get("class_form", 1.0) * 1.20
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.90
    elif "allowance" in rt:
        # Balanced approach for allowance
        w["class_form"] = w.get("class_form", 1.0) * 1.05
    elif "claiming" in rt or "clm" in rt:
        # Claiming - form/recent performance key, class less predictive
        w["class_form"] = w.get("class_form", 1.0) * 0.80
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.15
    elif "maiden" in rt:
        # Maiden - limited data, pedigree/angles matter
        w["class_form"] = w.get("class_form", 1.0) * 0.90
        w["track_bias"] = w.get("track_bias", 1.0) * 1.10
    
    return w


def apply_purse_scaling(weights: dict, purse: int) -> dict:
    """
    Scale weights based on purse amount (proxy for overall race quality).
    
    Higher purse = more reliable data, tighter competition
    Lower purse = more variance, pace/bias edges more exploitable
    """
    if not weights:
        return {"class_form": 1.0, "trs_jky": 1.0}
    
    w = weights.copy()
    purse_val = purse or 0
    
    if purse_val >= 500000:  # Major stakes ($500k+)
        w["class_form"] = w.get("class_form", 1.0) * 1.25
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.90
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.20
    elif purse_val >= 100000:  # Quality stakes/allowance
        w["class_form"] = w.get("class_form", 1.0) * 1.10
        w["trs_jky"] = w.get("trs_jky", 1.0) * 1.10
    elif purse_val >= 50000:  # Mid-level
        pass  # Use base weights
    elif purse_val >= 20000:  # Lower claiming
        w["class_form"] = w.get("class_form", 1.0) * 0.90
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.10
        w["track_bias"] = w.get("track_bias", 1.0) * 1.15
    else:  # Bottom level
        w["class_form"] = w.get("class_form", 1.0) * 0.80
        w["pace_speed"] = w.get("pace_speed", 1.0) * 1.15
        w["track_bias"] = w.get("track_bias", 1.0) * 1.20
    
    return w


def apply_condition_adjustment(weights: dict, condition: str) -> dict:
    """
    Adjust weights based on track condition.
    
    Fast/Firm: Standard weights
    Good/Yielding: Stamina/class slightly more important
    Muddy/Sloppy/Heavy: Off-track specialists, pace less predictive
    """
    if not weights:
        return {"class_form": 1.0, "trs_jky": 1.0}
    
    w = weights.copy()
    cond = (condition or "fast").lower()
    
    if "mud" in cond or "slop" in cond or "heavy" in cond:
        # Off-track - pace scenarios disrupted
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.80
        w["class_form"] = w.get("class_form", 1.0) * 1.15
        w["track_bias"] = w.get("track_bias", 1.0) * 1.30  # Rail/post matters more
    elif "good" in cond or "yield" in cond:
        w["pace_speed"] = w.get("pace_speed", 1.0) * 0.95
        w["class_form"] = w.get("class_form", 1.0) * 1.05
    # Fast/Firm = no adjustment
    
    return w


base_weights = get_weight_preset(surface_type, distance_txt)
profiled_weights = apply_strategy_profile_to_weights(base_weights, strategy_profile)
racetype_weights = adjust_by_race_type(profiled_weights, race_type_detected)
purse_weights = apply_purse_scaling(racetype_weights, purse_val)
final_weights = apply_condition_adjustment(purse_weights, condition_txt)

# Display active dynamic weights for transparency
st.caption(f"⚙️ **Dynamic Weights Applied:** Surface: {surface_type} | Distance: {distance_txt} | Condition: {condition_txt} | Race Type: {race_type_detected} | Purse: ${purse_val:,}")
weights_display = " | ".join([f"{k}: {v:.2f}" for k, v in final_weights.items()])
st.caption(f"📊 **Factor Weights:** {weights_display}")

for i, (rbias, pbias) in enumerate(scenarios):
    with tabs[i]:
        # Debug info for track bias detection
        canon_track = _canonical_track(track_name)
        dist_bucket = distance_bucket(distance_txt)
        style_display = ", ".join(running_style_biases)
        post_display = ", ".join(post_biases)
        st.caption(f"🔍 Track Bias Detection: {canon_track} • {surface_type} • {dist_bucket}")
        st.caption(f"📊 Selected Biases - Running Styles: {style_display} | Post Positions: {post_display}")

        ratings_df = compute_bias_ratings(
            df_styles=df_final_field.copy(),  # Pass a copy to avoid modifying original
            surface_type=surface_type,
            distance_txt=distance_txt,
            condition_txt=condition_txt,
            race_type=race_type_detected,
            running_style_bias=running_style_biases,  # Pass full list
            post_bias_pick=post_biases,  # Pass full list
            # ppi_value=ppi_val, # Removed as it's recalculated inside
            pedigree_per_horse=pedigree_per_horse,
            track_name=track_name,
            pp_text=pp_text,
            figs_df=figs_df,  # Pass speed figures
            dynamic_weights=final_weights  # Pass dynamic weights based on race parameters
        )
        ratings_df = apply_enhancements_and_figs(
            ratings_df=ratings_df,
            pp_text=pp_text,
            processed_weights=final_weights,
            chaos_index=0.0,
            track_name=track_name,
            surface_type=surface_type,
            distance_txt=distance_txt,
            race_type=race_type_detected,
            angles_per_horse=angles_per_horse,
            pedigree_per_horse=pedigree_per_horse,
            figs_df=figs_df  # <--- PASS THE REAL FIGS_DF
        )
        # Build ML odds dict from df_final_field for reality check
        # Prefer Live Odds (from Section A user input), fall back to ML (parsed morning line)
        ml_odds_dict = {}
        for _, row in df_final_field.iterrows():
            horse_name = row.get('Horse')
            # Try Live Odds first (user input), then ML (parsed)
            live_odds = row.get('Live Odds', '')
            ml_odds = row.get('ML', '')
            odds_str = live_odds if live_odds else ml_odds

            if horse_name and odds_str:
                try:
                    # Parse odds string (e.g., "30/1", "5/2", "3.5")
                    odds_str = str(odds_str).strip()
                    if '/' in odds_str:
                        parts = odds_str.split('/')
                        ml_odds_dict[horse_name] = float(parts[0]) / float(parts[1])
                    else:
                        ml_odds_dict[horse_name] = float(odds_str)
                except (ValueError, TypeError, IndexError, ZeroDivisionError):
                    # Invalid odds format - use default
                    ml_odds_dict[horse_name] = 5.0
            else:
                ml_odds_dict[horse_name] = 5.0

        fair_probs = fair_probs_from_ratings(ratings_df, ml_odds_dict)
        if 'Horse' in ratings_df.columns:
            ratings_df["Fair %"] = ratings_df["Horse"].map(lambda h: f"{fair_probs.get(h, 0) * 100:.1f}%")
            ratings_df["Fair Odds"] = ratings_df["Horse"].map(lambda h: fair_to_american_str(fair_probs.get(h, 0)))
        else:
            ratings_df["Fair %"] = ""
            ratings_df["Fair Odds"] = ""
        all_scenario_ratings[(rbias, pbias)] = (ratings_df.copy(), fair_probs)  # Store copy and probs

        disp = ratings_df.sort_values(by="R", ascending=False)
        if "R_ENHANCE_ADJ" in disp.columns:
            disp = disp.drop(columns=["R_ENHANCE_ADJ"])
        st.dataframe(
            disp,
            use_container_width=True, hide_index=True,
            column_config={
                "R": st.column_config.NumberColumn("Rating", format="%.2f"),
                "Cstyle": st.column_config.NumberColumn("C-Style", format="%.2f"),
                "Cpost": st.column_config.NumberColumn("C-Post", format="%.2f"),
                "Cpace": st.column_config.NumberColumn("C-Pace", format="%.2f"),
                "Cspeed": st.column_config.NumberColumn("C-Speed", format="%.2f"),
                "Cclass": st.column_config.NumberColumn("C-Class", format="%.2f"),
                "Cform": st.column_config.NumberColumn("C-Form", format="%.2f", help="Form cycle: layoff, trend, consistency"),
                "Atrack": st.column_config.NumberColumn("A-Track", format="%.2f"),
                "Arace": st.column_config.NumberColumn("A-Race", format="%.2f"),
                # Format Quirin to show integer or be blank
                "Quirin": st.column_config.NumberColumn("Quirin", format="%d", help="BRIS Pace Points"),
            }
        )

# Ensure primary key exists before accessing
if scenarios:
    primary_key = scenarios[0]
    if primary_key in all_scenario_ratings:
        primary_df, primary_probs = all_scenario_ratings[primary_key]

        # Store in session state for button access
        st.session_state['primary_d'] = primary_df
        st.session_state['primary_probs'] = primary_probs

        # Display ALL selected biases, not just the first scenario
        style_biases_display = ", ".join(running_style_biases)
        post_biases_display = ", ".join(post_biases)
        st.info(f"**Combined Scenario Analysis** • Profile: `{strategy_profile}` • PPI: {ppi_val:+.2f}")
        st.caption(f"📊 **Active Biases:** Running Styles: `{style_biases_display}` | Post Positions: `{post_biases_display}` (All bonuses aggregated)")
    else:
        st.error("Primary scenario ratings not found. Check calculations.")
        primary_df, primary_probs = pd.DataFrame(), {}  # Assign defaults
        st.stop()
else:
    st.error("No scenarios generated. Check bias selections.")
    primary_df, primary_probs = pd.DataFrame(), {}  # Assign defaults
    st.stop()


# ===================== C. Overlays & Betting Strategy =====================

st.markdown("---")
st.header("C. Overlays Table")
st.caption("💰 Overlays = horses whose fair odds are better than morning line odds. These represent betting value.")

# Offered odds map
offered_odds_map = {}
for _, r in df_final_field.iterrows():
    odds_str = str(r.get("Live Odds", "")).strip() or str(r.get("ML", "")).strip()
    dec = str_to_decimal_odds(odds_str)
    if dec:
        offered_odds_map[r["Horse"]] = dec

# Overlay table vs fair line
df_ol = overlay_table(fair_probs=primary_probs, offered=offered_odds_map)
st.session_state['df_ol'] = df_ol  # Store for Classic Report
st.dataframe(
    df_ol,
    use_container_width=True, hide_index=True,
    column_config={
        "EV per $1": st.column_config.NumberColumn("EV per $1", format="$%.3"),
        "Edge (pp)": st.column_config.NumberColumn("Edge (pp)")
    }
)

# All ticketing strategy UI has been removed from this section
# It is now generated "behind the scenes" in Section D.

# ===================== D. Strategy Builder & Classic Report =====================


def build_component_breakdown(primary_df, name_to_post, name_to_odds, pp_text=""):
    """
    GOLD STANDARD: Build detailed component breakdown with complete mathematical transparency.

    Shows exactly what the rating system sees in each horse, with:
    - All component values with proper weighting (using race_class_parser if available)
    - Calculated weighted contributions
    - Full traceability of rating calculation
    - Robust error handling for missing/invalid data

    Args:
        pp_text: Full BRISNET PP text for race_class_parser analysis

    Returns: Markdown formatted component breakdown for top 5 horses
    """
    # VALIDATION: Input checks
    if primary_df is None or primary_df.empty:
        return "No component data available."

    if 'Horse' not in primary_df.columns or 'R' not in primary_df.columns:
        return "Invalid component data structure."

    # EXTRACTION: Get top 5 horses by rating
    try:
        top_horses = primary_df.nlargest(5, 'R')
    except BaseException:
        # Fallback if rating column has issues
        top_horses = primary_df.head(5)

    if top_horses.empty:
        return "No horses to analyze."

    # COMPONENT WEIGHTS: Display weights for breakdown
    # NOTE: These are DEFAULT weights for display purposes only
    # ACTUAL weights used in rating calculation are determined by race_class_parser
    # and vary by race type (G1=10.0, Handicap=7.0, Claiming=2.0, etc.)
    WEIGHTS = {
        'Cclass': 3.0,   # Default - overridden by parser (1.0-10.0 scale)
        'Cspeed': 1.8,   # Speed figures - raw ability
        'Cform': 1.8,    # Form cycle - current condition
        'Cpace': 1.5,    # Pace advantage - tactical fit
        'Cstyle': 1.2,   # Running style - bias fit
        'Cpost': 0.8     # Post position - track bias
    }

    # IMPORTANT: Try to get actual class weight used in calculation from parser
    # This ensures breakdown shows true weights, not defaults
    if RACE_CLASS_PARSER_AVAILABLE and pp_text:
        try:
            race_class_data = parse_and_calculate_class(pp_text)
            actual_class_weight = race_class_data['weight']['class_weight']
            WEIGHTS['Cclass'] = actual_class_weight  # Use actual weight from parser
        except BaseException:
            pass  # Fall back to default

    breakdown = "### Component Breakdown (Top 5 Horses)\n"
    breakdown += "_Mathematical transparency: Shows exactly what the system sees in each horse_\n\n"

    for idx, row in top_horses.iterrows():
        horse_name = row.get('Horse', 'Unknown')
        post = name_to_post.get(horse_name, '?')
        ml = name_to_odds.get(horse_name, '?')

        # SAFE EXTRACTION: Get rating with error handling
        try:
            final_rating = float(row.get('R', 0))
        except BaseException:
            final_rating = 0.0

        breakdown += f"**#{post} {horse_name}** (ML {ml}) - **Rating: {final_rating:.2f}**\n"

        # CORE COMPONENTS: Extract with validation
        components = {}
        component_descriptions = {
            'Cclass': 'Purse earnings, race level history',
            'Cform': 'Recent performance trend, consistency',
            'Cspeed': 'Speed figures relative to field average',
            'Cpace': 'Pace advantage/disadvantage vs projected pace',
            'Cstyle': 'Running style fit for pace scenario',
            'Cpost': 'Post position bias for this track/distance'
        }

        weighted_sum = 0.0
        for comp_name, weight in WEIGHTS.items():
            try:
                comp_value = float(row.get(comp_name, 0))
            except BaseException:
                comp_value = 0.0

            components[comp_name] = comp_value
            weighted_contribution = comp_value * weight
            weighted_sum += weighted_contribution

            description = component_descriptions.get(comp_name, '')
            breakdown += f"- **{comp_name[1:]}:** {comp_value:+.2f} (×{weight} weight = {weighted_contribution:+.2f}) - {description}\n"

        # TRACK BIAS: Additional component
        try:
            atrack = float(row.get('Atrack', 0))
        except BaseException:
            atrack = 0.0
        breakdown += f"- **Track Bias:** {atrack:+.2f} - Track-specific advantages (style + post combo)\n"

        # TRANSPARENCY: Show weighted total
        breakdown += f"- **Weighted Core Total:** {weighted_sum:.2f}\n"

        # QUIRIN POINTS: BRIS pace rating
        quirin = row.get('Quirin', 'N/A')
        if quirin != 'N/A':
            try:
                quirin = int(float(quirin))
            except BaseException:
                quirin = 'N/A'
        breakdown += f"- **Quirin Points:** {quirin} - BRISNET early pace points\n"

        # FINAL RATING: Includes all angles and bonuses
        breakdown += f"- **Final Rating:** {final_rating:.2f} (includes 8 elite angles + tier 2 bonuses + track bias)\n\n"

    breakdown += "_Note: Positive values = advantages, negative = disadvantages. Weighted contributions show impact on final rating._\n"

    return breakdown


def build_betting_strategy(primary_df: pd.DataFrame, df_ol: pd.DataFrame,
                           strategy_profile: str, name_to_post: Dict[str, str],
                           name_to_odds: Dict[str, str], field_size: int, ppi_val: float,
                           smart_money_horses: List[Dict] = None, name_to_ml: Dict[str, str] = None) -> str:
    """
    Builds elite strategy report with finishing order predictions, component transparency,
    A/B/C/D grouping, and $50 bankroll optimization.

    Args:
        smart_money_horses: List of horses with significant ML→Live odds drops for Smart Money Alert
        name_to_ml: Dictionary mapping horse names to ML odds (for comparison with live odds)
    """

    import numpy as np
    from itertools import combinations

    # Handle None defaults for mutable default arguments
    if smart_money_horses is None:
        smart_money_horses = []
    if name_to_ml is None:
        name_to_ml = {}

    # --- ELITE: Calculate Most Likely Finishing Order (Sequential Selection Algorithm) ---
    def calculate_most_likely_finishing_order(df: pd.DataFrame, top_n: int = 5) -> List[Tuple[str, float]]:
        """
        GOLD STANDARD: Calculate most likely finishing order using mathematically sound sequential selection.

        Algorithm Guarantees:
        1. Each horse appears EXACTLY ONCE in finishing order (no duplicates)
        2. Probabilities are properly renormalized after each selection
        3. Later positions reflect conditional probabilities given earlier selections
        4. Handles edge cases (empty df, invalid probabilities, small fields)

        Mathematical Approach:
        - Position 1: Horse with highest base probability wins
        - Position 2: Remove winner, renormalize remaining field, select highest
        - Position 3-5: Continue sequential removal and selection

        Returns: List of (horse_name, conditional_probability) for positions 1-N
        """
        # VALIDATION: Input checks
        if df is None or df.empty:
            return []

        if 'Horse' not in df.columns or 'Fair %' not in df.columns:
            return []

        horses = df['Horse'].tolist()
        if len(horses) == 0:
            return []

        # EXTRACT: Base win probabilities with robust error handling
        win_probs = []
        for horse in horses:
            horse_df = df[df['Horse'] == horse]
            if horse_df.empty:
                win_probs.append(1.0 / len(horses))  # Fallback to uniform
                continue

            prob_str = horse_df['Fair %'].iloc[0]
            try:
                # Handle various formats: "25.5%", "0.255", 25.5
                if isinstance(prob_str, str):
                    prob = float(prob_str.strip('%').strip()) / (100.0 if '%' in str(prob_str) else 1.0)
                else:
                    prob = float(prob_str)
                    if prob > 1.0:  # Assume percentage
                        prob = prob / 100.0
            except BaseException:
                prob = 1.0 / len(horses)

            # SANITY CHECK: Probability bounds
            prob = max(0.0, min(1.0, prob))
            win_probs.append(prob)

        win_probs = np.array(win_probs, dtype=np.float64)

        # NORMALIZATION: Ensure probabilities sum to 1.0
        if win_probs.sum() > 0:
            win_probs = win_probs / win_probs.sum()
        else:
            # All probabilities were invalid - use uniform distribution
            win_probs = np.ones(len(horses), dtype=np.float64) / len(horses)

        # CONFIDENCE CAP (TUP R6 Feb 2026): Prevent over-confident predictions
        # Failed pick #2 Ez Cowboy had 94.8% probability but finished 4th
        # Cap maximum win probability at 65% for competitive fields (6+ horses)
        if len(horses) >= 6:
            max_prob = 0.65
            for i in range(len(win_probs)):
                if win_probs[i] > max_prob:
                    excess = win_probs[i] - max_prob
                    win_probs[i] = max_prob
                    # Redistribute excess to other horses proportionally
                    other_indices = [j for j in range(len(win_probs)) if j != i]
                    if other_indices:
                        redistribution = excess / len(other_indices)
                        for j in other_indices:
                            win_probs[j] += redistribution
            # Re-normalize after capping
            if win_probs.sum() > 0:
                win_probs = win_probs / win_probs.sum()

        # SEQUENTIAL SELECTION: Build finishing order one position at a time
        finishing_order = []
        remaining_indices = list(range(len(horses)))
        remaining_probs = win_probs.copy()

        for position in range(min(top_n, len(horses))):
            # VALIDATION: Check we have horses remaining
            if len(remaining_indices) == 0:
                break

            # RENORMALIZATION: Ensure remaining probabilities sum to 1.0
            prob_sum = remaining_probs.sum()
            if prob_sum > 0:
                remaining_probs = remaining_probs / prob_sum
            else:
                # Fallback to uniform for remaining horses
                remaining_probs = np.ones(len(remaining_indices), dtype=np.float64) / len(remaining_indices)

            # SELECTION: Horse with highest conditional probability
            best_relative_idx = np.argmax(remaining_probs)

            # BOUNDS CHECK: Validate index before accessing (prevents crash)
            if best_relative_idx >= len(remaining_indices):
                # Should never happen, but safety first
                break

            selected_horse_idx = remaining_indices[best_relative_idx]
            selected_prob = remaining_probs[best_relative_idx]

            # RECORD: Add to finishing order
            finishing_order.append((horses[selected_horse_idx], float(selected_prob)))

            # REMOVAL: Eliminate selected horse from remaining pool
            remaining_indices.pop(best_relative_idx)
            remaining_probs = np.delete(remaining_probs, best_relative_idx)

        return finishing_order

    # EXECUTE: Calculate most likely finishing order (ensures mathematical validity)
    finishing_order = calculate_most_likely_finishing_order(primary_df, top_n=5)

    # BUILD: Alternative horses for each position (for display purposes)
    # Shows top 3 most likely horses for each position, excluding already-selected horses
    most_likely = {}
    selected_horses = {horse for horse, _ in finishing_order}

    for pos_idx, (primary_horse, primary_prob) in enumerate(finishing_order, start=1):
        # ALTERNATIVE CANDIDATES: Get probabilities for all horses at this position
        # Exclude horses already selected for earlier positions
        alternatives = []
        for h in primary_df['Horse'].tolist():
            # Skip horses already selected for earlier positions (except the primary for this position)
            if h in selected_horses and h != primary_horse:
                continue

            h_prob_str = primary_df[primary_df['Horse'] == h]['Fair %'].iloc[0]
            try:
                if isinstance(h_prob_str, str):
                    h_prob = float(h_prob_str.strip('%').strip()) / (100.0 if '%' in str(h_prob_str) else 1.0)
                else:
                    h_prob = float(h_prob_str)
                    if h_prob > 1.0:
                        h_prob = h_prob / 100.0
                h_prob = max(0.0, min(1.0, h_prob))
            except BaseException:
                h_prob = 0.0

            alternatives.append((h, h_prob))

        # RANKING: Sort by probability (highest first) and take top 3
        alternatives.sort(key=lambda x: x[1], reverse=True)
        most_likely[pos_idx] = alternatives[:3]

    # --- 1. Helper Functions ---
    def format_horse_list(horse_names: List[str]) -> str:
        """Creates a bulleted list of horses with post, name, and odds (shows ML → Live if different)."""
        if not horse_names:
            return "* None"
        lines = []
        # Sort horses by post number before displaying
        sorted_horses = sorted(horse_names, key=lambda name: int(name_to_post.get(name, '999')))
        for name in sorted_horses:
            post = name_to_post.get(name, '??')
            current_odds = name_to_odds.get(name, 'N/A')
            ml_odds = name_to_ml.get(name, '')

            # Show ML → Live format when they differ, otherwise just show current odds
            if ml_odds and current_odds != ml_odds and current_odds != 'N/A':
                odds_display = f"ML {ml_odds} → Live {current_odds}"
            else:
                odds_display = current_odds

            lines.append(f"* **#{post} - {name}** ({odds_display})")
        return "\n".join(lines)

    def get_bet_cost(base: float, num_combos: int) -> str:
        """Calculates and formats simple bet cost."""
        cost = base * num_combos
        base_str = f"${base:.2f}"
        return f"{num_combos} combos = **${cost:.2f}** (at {base_str} base)"

    def get_box_combos(num_horses: int, box_size: int) -> int:
        """Calculates combinations for a box bet."""
        import math
        if num_horses < box_size:
            return 0
        return math.factorial(num_horses) // math.factorial(num_horses - box_size)

    # === FIX 1: Defined the missing helper function ===
    def get_min_cost_str(base: float, *legs) -> str:
        """Calculates part-wheel combos and cost from leg sizes."""
        final_combos = 1
        leg_counts = [l for l in legs if l > 0]  # Filter out 0s
        if not leg_counts:
            final_combos = 0
        else:
            for l in leg_counts:
                final_combos *= l

        cost = base * final_combos
        base_str = f"${base:.2f}"
        return f"{final_combos} combos = **${cost:.2f}** (at {base_str} base)"
    # ===================================================

    # --- 2. A/B/C/D Grouping Logic (with Odds Drift Gate) ---
    A_group, B_group, C_group, D_group = [], [], [], []

    all_horses = primary_df['Horse'].tolist()
    pos_ev_horses = set(df_ol[df_ol["EV per $1"] > 0.05]['Horse'].tolist()) if not df_ol.empty else set()

    # ═══════════════════════════════════════════════════════════════
    # PEGASUS 2026 TUNING: Block horses with massive odds drift from A-Group
    # British Isles 20/1 → 50/1 should NEVER have been in A-Group
    # ═══════════════════════════════════════════════════════════════
    blocked_from_A = set()
    dumb_money_horses = st.session_state.get('dumb_money_horses', [])
    if dumb_money_horses:
        for h in dumb_money_horses:
            if h.get('ratio', 1.0) > 2.0:  # 2x+ drift = blocked from A-Group
                blocked_from_A.add(h['name'])
    
    # Filter out blocked horses from A-group consideration
    eligible_for_A = [h for h in all_horses if h not in blocked_from_A]

    if strategy_profile == "Confident":
        A_group = [eligible_for_A[0]] if eligible_for_A else [all_horses[0]] if all_horses else []
        if len(eligible_for_A) > 1:
            second_horse = eligible_for_A[1]
            first_rating = primary_df[primary_df['Horse'] == eligible_for_A[0]]['R'].iloc[0] if eligible_for_A else 0
            second_rating = primary_df[primary_df['Horse'] == second_horse]['R'].iloc[0]
            if second_rating > (first_rating * 0.90):  # Only add #2 if very close
                A_group.append(second_horse)
    else:  # Value Hunter - Prioritize top pick + overlays (excluding blocked horses)
        eligible_overlays = pos_ev_horses - blocked_from_A
        A_group = list(set([eligible_for_A[0]] if eligible_for_A else []) | eligible_overlays)
        if len(A_group) > 4:  # Cap A group size for Value Hunter
            A_group = sorted(A_group, key=lambda h: primary_df[primary_df['Horse'] == h].index[0])[:4]

    B_group = [h for h in all_horses if h not in A_group][:3]
    C_group = [h for h in all_horses if h not in A_group and h not in B_group][:4]
    D_group = [h for h in all_horses if h not in A_group and h not in B_group and h not in C_group]

    nA, nB, nC, nD = len(A_group), len(B_group), len(C_group), len(D_group)
    nAll = field_size  # Total runners

    # --- 3. Build Pace Projection ---
    pace_report = "### Pace Projection\n"
    if ppi_val > 0.5:
        pace_report += f"* **Fast Pace Likely (PPI {ppi_val:+.2f}):** Favors horses that can press or stalk ('E/P', 'P') and potentially closers ('S') if leaders tire. Pure speed ('E') might fade.\n"
    elif ppi_val < -0.5:
        pace_report += f"* **Slow Pace Likely (PPI {ppi_val:+.2f}):** Favors horses near the lead ('E', 'E/P'). Closers ('P', 'S') may find it hard to catch up.\n"
    else:
        pace_report += f"* **Moderate Pace Expected (PPI {ppi_val:+.2f}):** Fair for most styles. Tactical speed ('E/P') is often useful.\n"

    # --- 4. Build Contender Analysis (Simplified) ---
    contender_report = "### Contender Analysis\n"
    contender_report += "**Key Win Contenders (A-Group):**\n" + format_horse_list(A_group) + "\n"
    contender_report += "* _Primary win threats based on model rank and/or betting value. Use these horses ON TOP in exactas, trifectas, etc._\n\n"

    contender_report += "**Primary Challengers (B-Group):**\n" + format_horse_list(B_group) + "\n"
    contender_report += "* _Logical contenders expected to finish 2nd or 3rd. Use directly underneath A-Group horses._\n"

    top_rated_horse = all_horses[0]
    is_overlay = top_rated_horse in pos_ev_horses
    top_ml_str = name_to_odds.get(top_rated_horse, '100')
    top_ml_dec = str_to_decimal_odds(top_ml_str) or 101
    is_underlay = not is_overlay and (
        primary_probs_dict.get(
            top_rated_horse, 0) > (
            1 / top_ml_dec)) and top_ml_dec < 4  # Define underlay as < 3/1

    if is_overlay:
        contender_report += f"\n**Value Note:** Top pick **#{name_to_post.get(top_rated_horse)} - {top_rated_horse}** looks like a good value bet (Overlay).\n"
    elif is_underlay:
        contender_report += f"\n**Value Note:** Top pick **#{name_to_post.get(top_rated_horse)} - {top_rated_horse}** might be overbet (Underlay at {top_ml_str}). Consider using more underneath than on top.\n"

    # --- 5. Build Simplified Blueprint Section ---
    blueprint_report = "### Betting Strategy Blueprints (Scale Base Bets to Budget: Max ~$100 Recommended)\n"
    blueprint_report += "_Costs are examples using minimum base bets ($0.50 Tri, $0.10 Super/SH5). Adjust base amount ($0.10, $0.50, $1.00+) per ticket to fit your total budget for this race._\n"

    # --- Field Size Logic ---
    if field_size <= 6:
        blueprint_report += "\n**Note:** With a small field (<=6 runners), Superfecta and Super High 5 payouts are often very low. Focus on Win, Exacta, and Trifecta bets.\n"
        # Generate only Win/Ex/Tri for small fields
        blueprint_report += f"\n#### {strategy_profile} Profile Plan (Small Field)\n"
        if strategy_profile == "Value Hunter":
            blueprint_report += "* **Win Bets:** Consider betting all **A-Group** horses.\n"
        else:  # Confident
            blueprint_report += "* **Win Bet:** Focus on top **A-Group** horse(s).\n"

        # Exacta Examples
        blueprint_report += f"* **Exacta Part-Wheel:** `A / B,C` ({nA}x{nB + nC}) - {get_min_cost_str(1.00, nA, nB + nC)}\n"
        if nA >= 2:
            ex_box_combos = get_box_combos(nA, 2)
            blueprint_report += f"* **Exacta Box (A-Group):** `{', '.join(map(str, [int(name_to_post.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(1.00, ex_box_combos)}\n"

        # Trifecta Examples
        blueprint_report += f"* **Trifecta Part-Wheel:** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        if nA >= 3:
            tri_box_combos = get_box_combos(nA, 3)
            blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str, [int(name_to_post.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"

        blueprint_report += "_Structure: Use A-horses on top, spread underneath with B and C._\n"

    else:  # Standard logic for fields > 6
        # --- Confident Blueprint ---
        blueprint_report += "\n#### Confident Profile Plan\n"
        blueprint_report += "_Focus: Key A-Group horses ON TOP._\n"
        blueprint_report += "* **Win Bet:** Focus on top **A-Group** horse(s).\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B` ({nA}x{nB}) - {get_min_cost_str(1.00, nA, nB)}\n"
        if nA >= 1 and nB >= 1 and nC >= 1:  # Check if groups have members for straight example
            blueprint_report += "* **Straight Trifecta:** `Top A / Top B / Top C` (1 combo) - Consider at higher base (e.g., $1 or $2).\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B / C / D` ({nA}x{nB}x{nC}x{nD}) - {get_min_cost_str(0.10, nA, nB, nC, nD)}\n"
        if field_size >= 7:  # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B / C / D / ALL` ({nA}x{nB}x{nC}x{nD}x{nAll}) - {get_min_cost_str(0.10, nA, nB, nC, nD, nAll)}\n"

        # --- Value-Hunter Blueprint ---
        blueprint_report += "\n#### Value-Hunter Profile Plan\n"
        blueprint_report += "_Focus: Use A-Group (includes overlays) ON TOP, spread wider underneath._\n"
        blueprint_report += "* **Win Bets:** Consider betting all **A-Group** horses.\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B,C` ({nA}x{nB + nC}) - {get_min_cost_str(1.00, nA, nB + nC)}\n"
        if nA >= 3:  # Example box if A group is large enough
            tri_box_combos = get_box_combos(nA, 3)
            blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str, [int(name_to_post.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B,C / B,C,D` ({nA}x{nB + nC}x{nB + nC + nD}) - {get_min_cost_str(0.50, nA, nB + nC, nB + nC + nD)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B,C / B,C,D / ALL` ({nA}x{nB + nC}x{nB + nC + nD}x{nAll}) - {get_min_cost_str(0.10, nA, nB + nC, nB + nC + nD, nAll)}\n"
        if field_size >= 7:  # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B,C / B,C,D / ALL / ALL` ({nA}x{nB + nC}x{nB + nC + nD}x{nAll}x{nAll}) - {get_min_cost_str(0.10, nA, nB + nC, nB + nC + nD, nAll, nAll)}\n"

    # --- GOLD STANDARD: Build probability dictionary with validation ---
    primary_probs_dict = {}
    for horse in primary_df['Horse'].tolist():
        horse_df = primary_df[primary_df['Horse'] == horse]
        if horse_df.empty:
            primary_probs_dict[horse] = 1.0 / max(len(primary_df), 1)
            continue

        prob_str = horse_df['Fair %'].iloc[0]
        try:
            # Handle multiple formats: "25.5%", "0.255", 25.5
            if isinstance(prob_str, str):
                prob = float(prob_str.strip('%').strip()) / (100.0 if '%' in str(prob_str) else 1.0)
            else:
                prob = float(prob_str)
                if prob > 1.0:  # Assume percentage format
                    prob = prob / 100.0

            # VALIDATION: Probability bounds [0, 1]
            prob = max(0.0, min(1.0, prob))
        except BaseException:
            prob = 1.0 / max(len(primary_df), 1)

        primary_probs_dict[horse] = prob

    # NORMALIZATION: Ensure probabilities sum to exactly 1.0
    total_prob = sum(primary_probs_dict.values())
    if total_prob > 0:
        primary_probs_dict = {h: p / total_prob for h, p in primary_probs_dict.items()}

    detailed_breakdown = build_component_breakdown(primary_df, name_to_post, name_to_odds, pp_text=pp_text)

    component_report = "### What Our System Sees in Top Contenders\n\n"
    component_report += detailed_breakdown + "\n"

    # --- GOLD STANDARD: Build Finishing Order Predictions (NO DUPLICATES) ---
    # Use sequential selection algorithm - each horse appears EXACTLY ONCE
    finishing_order = calculate_most_likely_finishing_order(primary_df, top_n=5)

    finishing_order_report = "### Most Likely Finishing Order\n\n"
    finishing_order_report += "**Algorithm:** Sequential selection ensuring each horse appears EXACTLY ONCE. The percentage indicates conditional probability (e.g., for 2nd place: probability of finishing 2nd given not finishing 1st).\n\n"

    position_names = {1: "🥇 Win (1st)", 2: "🥈 Place (2nd)", 3: "🥉 Show (3rd)", 4: "4th", 5: "5th"}

    for pos, (horse, prob) in enumerate(finishing_order, 1):
        post = name_to_post.get(horse, '?')
        odds = name_to_odds.get(horse, '?')
        finishing_order_report += f"* **{position_names[pos]} • #{post} {horse}** (Odds: {odds}) — {prob * 100:.1f}% conditional probability\n"

    finishing_order_report += "\n💡 **Use These Rankings:** Build your exotic tickets using this exact finishing order for optimal probability-based coverage.\n\n"

    # --- ELITE: Build $50 Bankroll Optimization ---
    bankroll_report = "### $50 Bankroll Structure\n\n"

    if strategy_profile == "Value Hunter":
        bankroll_report += "**Strategy:** Value Hunter - Focus on overlays with wider coverage\n\n"

        # Win bets on A-Group overlays
        win_cost = min(len([h for h in A_group if h in pos_ev_horses]), 3) * 8
        bankroll_report += f"* **Win Bets** (${win_cost}): $8 each on top {min(len([h for h in A_group if h in pos_ev_horses]), 3)} overlay(s) from A-Group\n"

        # Exacta part-wheel
        ex_combos = nA * (nB + min(nC, 2))
        ex_cost = min(int(ex_combos * 0.50), 14)
        bankroll_report += f"* **Exacta** (${ex_cost}): A / B,C (top 2 from C) - ${ex_cost / ex_combos:.2f} base × {ex_combos} combos\n"

        # Trifecta
        tri_combos = nA * (nB + min(nC, 2)) * (nB + nC + min(nD, 2))
        tri_cost = min(int(tri_combos * 0.30), 12)
        bankroll_report += f"* **Trifecta** (${tri_cost}): A / B,C / B,C,D - ${tri_cost / tri_combos:.2f} base × {tri_combos} combos\n"

        # Superfecta
        super_cost = 50 - win_cost - ex_cost - tri_cost
        super_cost = max(super_cost, 8)
        bankroll_report += f"* **Superfecta** (${super_cost}): A / B,C / B,C,D / Top 5 from D+All\n"

    else:  # Confident
        bankroll_report += "**Strategy:** Confident - Focus on top pick with deeper coverage\n\n"

        # Win bet on top pick
        bankroll_report += f"* **Win Bet** ($20): $20 on #{name_to_post.get(all_horses[0], '?')} {all_horses[0]}\n"

        # Exacta
        ex_combos = nA * nB
        bankroll_report += f"* **Exacta** ($10): A / B - ${10 / max(ex_combos, 1):.2f} base × {ex_combos} combos\n"

        # Trifecta
        tri_combos = nA * nB * nC
        bankroll_report += f"* **Trifecta** ($12): A / B / C - ${12 / max(tri_combos, 1):.2f} base × {tri_combos} combos\n"

        # Superfecta
        bankroll_report += f"* **Superfecta** ($8): A / B / C / D - scaled to fit budget\n"

    bankroll_report += f"\n**Total Investment:** $50 (optimized)\n"
    bankroll_report += f"**Risk Level:** {strategy_profile} approach - {'Wider coverage, value-based' if strategy_profile == 'Value Hunter' else 'Concentrated on top selection'}\n"
    bankroll_report += f"\n💡 **Use Finishing Order Predictions:** The probability rankings above show the most likely finishers for each position. Build your tickets using horses with highest probabilities for each slot.\n"

    # --- Smart Money Alert Section ---
    smart_money_report = ""
    if smart_money_horses:
        smart_money_report = "### 🚨 Smart Money Alert: Significant Odds Movement Detected\n\n"
        smart_money_report += "The following horses show **major public support** with Live odds dropping significantly from Morning Line. This typically indicates:\n"
        smart_money_report += "* Trainer/connections betting\n"
        smart_money_report += "* Informed money from sharp handicappers\n"
        smart_money_report += "* Positive workout reports or stable buzz\n"
        smart_money_report += "* Hidden form improvement\n\n"

        # Sort by biggest movement percentage
        smart_money_horses.sort(key=lambda x: x['movement_pct'], reverse=True)

        for horse_data in smart_money_horses:
            name = horse_data['name']
            post = horse_data['post']
            ml = horse_data['ml']
            live = horse_data['live']
            movement_pct = horse_data['movement_pct']

            smart_money_report += f"* **🔥 #{post} {name}** - ML {ml} → Live {live} (📉 {movement_pct:.0f}% drop)\n"

        smart_money_report += "\n💡 **Action:** These horses are getting **heavy public support**. Consider them seriously even if model doesn't rank them #1. Sharp money often spots angles the numbers miss.\n\n"
        smart_money_report += "---\n"

    # --- 6. Build Final Report String (OPTIMIZED ORDER: Most Important First) ---
    final_report = f"""
{finishing_order_report}
---
{smart_money_report}{bankroll_report}
---
{component_report}
---
{contender_report}
---
### A/B/C/D Contender Groups for Tickets

**A-Group (Key Win Contenders - Use ON TOP)**
{format_horse_list(A_group)}

**B-Group (Primary Challengers - Use 2nd/3rd)**
{format_horse_list(B_group)}

**C-Group (Underneath Keys - Use 2nd/3rd/4th)**
{format_horse_list(C_group)}

**D-Group (Exotic Fillers - Use 3rd/4th/5th)**
{format_horse_list(D_group)}

---
{pace_report}
---
{blueprint_report}
---
### Bankroll & Strategy Notes
* **Budget:** Decide your total wager amount for this race (e.g., $20, $50, up to ~$100 recommended max).
* **Scale Base Bets:** Adjust the base bet amount ($0.10, $0.50, $1.00+) on the blueprint tickets to match your budget. Use an online wager calculator or your betting platform's tools to confirm costs.
* **Confidence:** Bet more confidently when A/B groups look strong. Reduce base bets or narrow the C/D groups in tickets if less confident.
* **Small Fields (<=6):** Focus on Win/Exacta/Trifecta as complex exotics pay less.
* **Play SH5** mainly on mandatory payout days or when you have a very strong opinion & budget allows.
"""
    return final_report


st.markdown("---")
st.header("D. Classic Report")
st.caption("📊 Generate a comprehensive handicapping analysis with pace projections, contender groups, and betting strategy blueprints.")

# Only show button if race has been parsed
if not st.session_state.get("parsed", False):
    st.warning("⚠️ Please parse a race first in Section A before analyzing.")
elif 'primary_d' not in st.session_state or 'primary_probs' not in st.session_state:
    st.error("❌ Rating data not available.")
    st.info("📋 **Next Steps:** Scroll up to **Section B: Bias-Adjusted Ratings** and make sure you:\n1. Select a Strategy Profile (Confident or Value Hunter)\n2. Select at least one Running Style Bias (E, E/P, P, or S)\n3. Select at least one Post Position Bias\n\nOnce you make your selections, the ratings will calculate automatically and the 'Analyze This Race' button will appear below.")
else:
    # Display existing Classic Report if it exists (before the button)
    if st.session_state.get('classic_report_generated', False):
        st.success("✅ Classic Report Generated")
        st.markdown(st.session_state.get('classic_report', ''))

        # Show download buttons for existing report
        if 'classic_report' in st.session_state:
            report_str = st.session_state['classic_report']
            analysis_bytes = report_str.encode("utf-8")
            df_ol = st.session_state.get('df_ol', pd.DataFrame())
            overlays_bytes = df_ol.to_csv(index=False).encode("utf-8-sig") if isinstance(df_ol, pd.DataFrame) else b""

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "⬇️ Download Full Analysis (.txt)",
                    data=analysis_bytes,
                    file_name="analysis.txt",
                    mime="text/plain",
                    key="dl_analysis_persistent")
            with col2:
                st.download_button(
                    "⬇️ Download Overlays (CSV)",
                    data=overlays_bytes,
                    file_name="overlays.csv",
                    mime="text/csv",
                    key="dl_overlays_persistent")
            with col3:
                # Create strategy_report_md from session state if available
                if 'strategy_report' in st.session_state:
                    tickets_bytes = st.session_state['strategy_report'].encode("utf-8")
                    st.download_button(
                        "⬇️ Download Strategy Detail (.txt)",
                        data=tickets_bytes,
                        file_name="strategy_detail.txt",
                        mime="text/plain",
                        key="dl_strategy_persistent")

        st.info("💡 To generate a new report, click 'Analyze This Race' again below.")

    if st.button("Analyze This Race", type="primary", key="analyze_button"):
        # Clear previous Classic Report before generating new one
        st.session_state.pop('classic_report_generated', None)
        st.session_state.pop('classic_report', None)

        with st.spinner("Handicapping Race..."):
            try:
                # ============================================================
                # GOLD STANDARD: SEQUENTIAL VALIDATION & DATA INTEGRITY CHECK
                # ============================================================

                # STAGE 1: Critical Data Retrieval with Validation
                primary_df = st.session_state.get('primary_d')
                primary_probs = st.session_state.get('primary_probs')
                df_final_field = st.session_state.get('df_final_field')
                df_ol = st.session_state.get('df_ol', pd.DataFrame())

                # VALIDATION: Check primary data exists and is valid
                if primary_df is None or not isinstance(primary_df, pd.DataFrame):
                    st.error("❌ CRITICAL ERROR: Primary ratings dataframe is missing. Please regenerate ratings.")
                    st.stop()

                if primary_df.empty:
                    st.error("❌ CRITICAL ERROR: Primary ratings dataframe is empty. Check field entries.")
                    st.stop()

                # GOLD STANDARD FIX: Ensure Post and ML columns are in primary_df for classic report
                if df_final_field is not None and not df_final_field.empty:
                    if 'Post' not in primary_df.columns or 'ML' not in primary_df.columns:
                        # Merge Post and ML from df_final_field into primary_df
                        post_ml_data = df_final_field[['Horse', 'Post', 'ML']].copy()
                        primary_df = primary_df.merge(post_ml_data, on='Horse', how='left')
                        # Update session state with enriched primary_df
                        st.session_state['primary_d'] = primary_df

                # VALIDATION: Check required columns exist
                required_cols = ['Horse', 'R', 'Fair %', 'Fair Odds']
                missing_cols = [col for col in required_cols if col not in primary_df.columns]
                if missing_cols:
                    st.error(f"❌ CRITICAL ERROR: Missing required columns: {missing_cols}")
                    st.error("Required columns: Horse, R, Fair %, Fair Odds")
                    st.stop()

                # VALIDATION: Check Horse column has valid names
                if primary_df['Horse'].isna().any():
                    st.error("❌ CRITICAL ERROR: Some horses have missing names")
                    st.stop()

                # VALIDATION: Check R (ratings) column is numeric and finite
                try:
                    primary_df['R_test'] = pd.to_numeric(primary_df['R'], errors='coerce')
                    if primary_df['R_test'].isna().all():
                        st.error("❌ CRITICAL ERROR: All ratings are invalid (non-numeric)")
                        st.stop()
                    if not np.all(np.isfinite(primary_df['R_test'].dropna())):
                        st.error("❌ CRITICAL ERROR: Ratings contain infinite values")
                        st.stop()
                    primary_df.drop(columns=['R_test'], inplace=True)
                except Exception as e:
                    st.error(f"❌ CRITICAL ERROR: Rating validation failed: {e}")
                    st.stop()

                # VALIDATION: Check Fair % exists and is valid
                if primary_df['Fair %'].isna().all():
                    st.error("❌ CRITICAL ERROR: No fair probabilities calculated")
                    st.stop()

                # STAGE 2: Context Data Retrieval with Safe Defaults
                strategy_profile = st.session_state.get('strategy_profile', 'Balanced')
                ppi_val = st.session_state.get('ppi_val', 0.0)
                track_name = st.session_state.get('track_name', '')
                surface_type = st.session_state.get('surface_type', 'Dirt')
                condition_txt = st.session_state.get('condition_txt', '')
                distance_txt = st.session_state.get('distance_txt', '')
                race_type_detected = st.session_state.get('race_type', '')
                purse_val = st.session_state.get('purse_val', 0)

                # VALIDATION: Ensure numeric values are finite
                if not np.isfinite(ppi_val):
                    ppi_val = 0.0
                if not np.isfinite(purse_val) or purse_val < 0:
                    purse_val = 0

                # VALIDATION: Ensure text fields are strings
                track_name = str(track_name) if track_name else 'Unknown'
                surface_type = str(surface_type) if surface_type else 'Dirt'
                condition_txt = str(condition_txt) if condition_txt else 'Fast'
                distance_txt = str(distance_txt) if distance_txt else '6F'
                race_type_detected = str(race_type_detected) if race_type_detected else 'Unknown'

                # STAGE 3: Sort and Build Mappings with Validation
                primary_sorted = primary_df.sort_values(by="R", ascending=False)

                # VALIDATION: Ensure final field exists
                if df_final_field is None or df_final_field.empty:
                    st.error("❌ CRITICAL ERROR: Final field dataframe missing")
                    st.stop()

                # GOLD STANDARD: Build safe mappings with validation
                try:
                    name_to_post = pd.Series(
                        df_final_field["Post"].values,
                        index=df_final_field["Horse"]
                    ).to_dict()

                    # CRITICAL FIX: Prioritize Live Odds over ML odds (just like Fair % calculation does)
                    name_to_odds = {}
                    smart_money_horses = []  # Track horses with significant odds movement
                    name_to_ml = {}  # Store ML odds separately for comparison

                    for _, row in df_final_field.iterrows():
                        horse_name = row.get('Horse')
                        live_odds = row.get('Live Odds', '')
                        ml_odds = row.get('ML', '')

                        # Store ML for comparison
                        name_to_ml[horse_name] = ml_odds

                        # Use Live Odds if entered by user, otherwise fall back to ML
                        name_to_odds[horse_name] = live_odds if live_odds else ml_odds

                        # SMART MONEY DETECTION: Flag horses with significant odds drops
                        if live_odds and ml_odds:
                            try:
                                live_decimal = odds_to_decimal(live_odds)
                                ml_decimal = odds_to_decimal(ml_odds)

                                # Smart money = Live odds < 60% of ML odds
                                # Example: ML 30/1 → Live 17/1 is 17/30 = 56.7% (ALERT!)
                                if live_decimal > 0 and ml_decimal > 0:
                                    ratio = live_decimal / ml_decimal
                                    if ratio < 0.6:  # Live odds dropped to <60% of ML
                                        movement_pct = (1 - ratio) * 100
                                        smart_money_horses.append({
                                            'name': horse_name,
                                            'post': row.get('Post'),
                                            'ml': ml_odds,
                                            'live': live_odds,
                                            'ratio': ratio,
                                            'movement_pct': movement_pct
                                        })
                            except Exception:
                                pass  # Ignore conversion errors

                    # VALIDATION: Ensure all horses in primary_df have post/odds mappings
                    missing_horses = []
                    for horse in primary_df["Horse"]:
                        if horse not in name_to_post:
                            missing_horses.append(horse)

                    if missing_horses:
                        st.error(f"❌ CRITICAL ERROR: Horse name mismatch between ratings and Section A")
                        st.error(f"Missing horses: {', '.join(missing_horses)}")
                        st.error("This usually means horse names were modified after Section A")
                        st.stop()

                except KeyError as e:
                    st.error(f"❌ CRITICAL ERROR: Missing required column in final field: {e}")
                    st.stop()

                field_size = len(primary_df)

                # VALIDATION: Field size sanity check
                if field_size < 2:
                    st.error("❌ CRITICAL ERROR: Field must have at least 2 horses")
                    st.stop()
                if field_size > 20:
                    st.warning("⚠️ WARNING: Unusually large field size (>20 horses)")

                # ============================================================
                # SEQUENTIAL EXECUTION: All validations passed, proceed safely
                # ============================================================

                # ═══════════════════════════════════════════════════════════════
                # SMART MONEY BONUS: Apply rating boost to horses with sharp money
                # ═══════════════════════════════════════════════════════════════
                # TUP R4: #6 Your Call had ML 5/1 → Live 3/1 (40% drop) and WON
                # TUP R5: #8 Naval Escort had ML 5/1 → Live 2/1 (60% drop) detected
                # System detected smart money but didn't boost ratings - FIX IT

                # ═══════════════════════════════════════════════════════════════
                # PEGASUS 2026 TUNING: Odds Drift OUT Penalty
                # British Isles 20/1 → 50/1 = money LEAVING (penalize heavily)
                # White Abarrio 4/1 → 9/2 = money STAYING (existing smart money bonus)
                # ═══════════════════════════════════════════════════════════════
                
                dumb_money_horses = []  # Horses with odds drifting OUT (money leaving)
                
                for _, row in df_final_field.iterrows():
                    horse_name = row.get('Horse')
                    live_odds = row.get('Live Odds', '')
                    ml_odds = row.get('ML', '')
                    
                    if live_odds and ml_odds:
                        try:
                            live_decimal = odds_to_decimal(live_odds)
                            ml_decimal = odds_to_decimal(ml_odds)
                            
                            if live_decimal > 0 and ml_decimal > 0:
                                ratio = live_decimal / ml_decimal
                                
                                # DUMB MONEY: Odds drifting OUT significantly
                                # British Isles: 20/1 → 50/1 = ratio 2.5 = major red flag
                                if ratio > 1.5:  # Odds up 50%+ = money leaving
                                    drift_pct = (ratio - 1) * 100
                                    dumb_money_horses.append({
                                        'name': horse_name,
                                        'post': row.get('Post'),
                                        'ml': ml_odds,
                                        'live': live_odds,
                                        'ratio': ratio,
                                        'drift_pct': drift_pct
                                    })
                        except Exception:
                            pass

                # Apply PENALTY to horses with money leaving (odds drifting out)
                if dumb_money_horses:
                    for idx, row in primary_df.iterrows():
                        horse_drift = next((h for h in dumb_money_horses if h['name'] == row['Horse']), None)
                        if horse_drift:
                            ratio = horse_drift['ratio']
                            # Scaled penalty: bigger drift = bigger penalty
                            if ratio > 2.0:
                                penalty = -3.0  # Massive penalty for 2x+ drift (British Isles case)
                            elif ratio > 1.5:
                                penalty = -1.5  # Moderate penalty for 50%+ drift
                            else:
                                penalty = 0.0
                            
                            primary_df.at[idx, 'R'] = row['R'] + penalty

                # Store dumb_money_horses for A-Group gate
                st.session_state['dumb_money_horses'] = dumb_money_horses

                # Apply BONUS to horses with money coming IN (smart money)
                if smart_money_horses:
                    smart_money_names = [h['name'] for h in smart_money_horses]
                    smart_money_bonus = 2.5  # Significant boost for sharp action

                    # Apply bonus to primary_df ratings
                    for idx, row in primary_df.iterrows():
                        if row['Horse'] in smart_money_names:
                            primary_df.at[idx, 'R'] = row['R'] + smart_money_bonus

                # Re-sort after applying all odds-based adjustments
                primary_sorted = primary_df.sort_values(by="R", ascending=False)

                top_table = primary_sorted[['Horse', 'R', 'Fair %', 'Fair Odds']].head(5).to_markdown(index=False)

                overlay_pos = df_ol[df_ol["EV per $1"] > 0] if not df_ol.empty else pd.DataFrame()
                overlay_table_md = (overlay_pos[['Horse', 'Fair %', 'Fair (AM)', 'Board (dec)', 'EV per $1']].to_markdown(
                    index=False) if not overlay_pos.empty else "None.")

                # --- 2. NEW: Generate Simplified A/B/C/D Strategy Report ---
                strategy_report_md = build_betting_strategy(
                    primary_df,
                    df_ol,
                    strategy_profile,
                    name_to_post,
                    name_to_odds,
                    field_size,
                    ppi_val,
                    smart_money_horses,
                    name_to_ml)

                # Store strategy report in session state
                st.session_state['strategy_report'] = strategy_report_md

                # --- 3. Update the LLM Prompt ---
                race_num = st.session_state.get('race_num', 1)
                prompt = f"""
Act as a professional horse racing analyst writing a clear, concise, and actionable betting report suitable for handicappers of all levels.

--- RACE CONTEXT ---
- Track: {track_name}
- Race Number: {race_num}
- Surface: {surface_type} ({condition_txt}) • Distance: {distance_txt}
- Race Type: {race_type_detected}
- Purse: ${purse_val:,}
- Strategy Profile Selected: {strategy_profile}
- Field Size: {field_size} horses

--- KEY MODEL OUTPUTS ---
Top 5 Rated Horses:
{top_table}

Horses Offering Potential Value (Overlays):
{overlay_table_md}

--- FULL ANALYSIS & BETTING PLAN ---
{strategy_report_md}

--- TASK: WRITE ELITE CLASSIC REPORT ---
Your goal is to present a sophisticated yet clear analysis. Structure your report as follows:

**1. Race Overview (3-5 sentences)**
- Summarize race conditions: track, surface, distance, purse
- Pace projection based on PPI and field composition
- Key race dynamics and angles to focus on

**2. Top Contenders & Betting Focus**
- A-Group: Key win contenders with their dominant strengths (use post #, name, current odds)
- B-Group: Primary challengers and why they're threats
- Highlight any value opportunities or underlays from model analysis
- Note any Smart Money Alert horses with significant ML→Live odds movement

**3. Recommended Betting Approach**
- State the optimal strategy profile for this race
- Explain which ticket structures make most sense given field dynamics
- Reference the specific blueprints from the betting strategy section above
- Mention bankroll allocation and risk management considerations

**STYLE GUIDE:**
- Be direct and analytical (no fluff like "buckle up" or "folks")
- Use racing terminology appropriately
- Focus on ACTIONABLE insights: Why these horses? Which tickets? How to play the race?
- DO NOT restate the component breakdown, finishing order, or bankroll structure already displayed above
- Instead, SYNTHESIZE the data into a clear narrative: pace scenario → contender strengths → betting approach
- Use horse names with post numbers (#) for clarity
- Keep it concise: ~150-200 words total
- **Tone:** Confident, professional, accessible to all handicappers
"""
                report = call_openai_messages(messages=[{"role": "user", "content": prompt}])

                # ===== PHASE 3: ADVANCED PROBABILITY ANALYSIS =====
                try:
                    from phase3_probability_engine import Phase3ProbabilityEngine, format_phase3_report
                    
                    # Get win probabilities from primary_df
                    if 'Win%' in primary_df.columns:
                        win_probs = primary_df['Win%'].values / 100.0  # Convert percentage to decimal
                        horse_names = primary_df['Horse'].values if 'Horse' in primary_df.columns else None
                        
                        # Run Phase 3 analysis
                        phase3_engine = Phase3ProbabilityEngine(bankroll=50.0)
                        phase3_results = phase3_engine.analyze_race_comprehensive(
                            win_probs=win_probs,
                            horse_names=horse_names,
                            confidence_level=0.95
                        )
                        
                        # Format Phase 3 report
                        phase3_report = format_phase3_report(phase3_results)
                        
                    else:
                        phase3_report = "Phase 3 analysis unavailable (missing win probabilities)"
                except Exception as e:
                    phase3_report = f"Phase 3 analysis error: {str(e)}"
                
                # Store Classic Report in session state so it persists across reruns
                st.session_state['classic_report'] = report
                st.session_state['phase3_report'] = phase3_report
                st.session_state['classic_report_generated'] = True

                st.success("✅ Analysis Complete! Thank you for contributing to our community database.")

                # Wrap ENTIRE report in consistent font styling (includes strategy_report_md, LLM output, AND Phase 3)
                styled_report = f"""<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; font-size: 15px; line-height: 1.6; color: #1f2937;">

{strategy_report_md}

---

{report}

---

<pre style="background-color: #f3f4f6; padding: 15px; border-radius: 5px; font-family: 'Courier New', monospace; font-size: 13px; line-height: 1.4; overflow-x: auto;">
{phase3_report}
</pre>

</div>"""
                st.markdown(styled_report, unsafe_allow_html=True)

                # ---- Save to disk (optional) ----
                report_str = report if isinstance(report, str) else str(report)
                with open("analysis.txt", "w", encoding="utf-8", errors="replace") as f:
                    f.write(report_str)

                if isinstance(df_ol, pd.DataFrame):
                    df_ol.to_csv("overlays.csv", index=False, encoding="utf-8-sig")
                else:
                    pd.DataFrame().to_csv("overlays.csv", index=False, encoding="utf-8-sig")

                # --- Create a tickets.txt from the strategy report ---
                with open("tickets.txt", "w", encoding="utf-8", errors="replace") as f:
                    f.write(strategy_report_md)  # Save the raw strategy markdown
                tickets_bytes = strategy_report_md.encode("utf-8")

                # ---- Download buttons (browser) ----
                analysis_bytes = report_str.encode("utf-8")
                overlays_bytes = df_ol.to_csv(index=False).encode(
                    "utf-8-sig") if isinstance(df_ol, pd.DataFrame) else b""

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "⬇️ Download Full Analysis (.txt)",
                        data=analysis_bytes,
                        file_name="analysis.txt",
                        mime="text/plain",
                        key="dl_analysis_new")
                with col2:
                    st.download_button(
                        "⬇️ Download Overlays (CSV)",
                        data=overlays_bytes,
                        file_name="overlays.csv",
                        mime="text/csv",
                        key="dl_overlays_new")
                with col3:
                    st.download_button(
                        "⬇️ Download Strategy Detail (.txt)",
                        data=tickets_bytes,
                        file_name="strategy_detail.txt",
                        mime="text/plain",
                        key="dl_strategy_new")

                # ============================================================
                # AUTO-SAVE TO GOLD HIGH-IQ DATABASE
                # ============================================================
                if GOLD_DB_AVAILABLE and gold_db is not None and primary_df is not None:
                    try:
                        # safe_float is now defined at top of file as a global helper

                        # Generate race ID using ACTUAL race date from PP text (not current date)
                        # This ensures correct race identification even when analyzing historical/future races
                        race_date_str = None
                        pp_text_raw = st.session_state.get('pp_text_cache', '')

                        # Extract race date from BRISNET text (format: "Saturday, February 1, 2026")
                        import re
                        date_match = re.search(
                            r'(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})', pp_text_raw)
                        if date_match:
                            month_str = date_match.group(2)
                            day_str = date_match.group(3).zfill(2)
                            year_str = date_match.group(4)

                            # Convert month name to number
                            months = {'January': '01', 'February': '02', 'March': '03', 'April': '04',
                                      'May': '05', 'June': '06', 'July': '07', 'August': '08',
                                      'September': '09', 'October': '10', 'November': '11', 'December': '12'}
                            month_num = months.get(month_str, '01')
                            race_date_str = f"{year_str}{month_num}{day_str}"  # YYYYMMDD format

                        # Fallback to current date if extraction fails
                        if not race_date_str:
                            race_date_str = datetime.now().strftime('%Y%m%d')
                            st.warning("⚠️ Could not extract race date from PP text, using current date")

                        race_date = race_date_str  # Use extracted date
                        race_id = f"{track_name}_{race_date}_R{st.session_state.get('race_num', 1)}"

                        # Calculate race bucket for track bias lookup
                        race_bucket = distance_bucket(distance_txt)

                        # Prepare COMPREHENSIVE race metadata with all context
                        race_metadata = {
                            'track': track_name,
                            'date': race_date,
                            'race_num': st.session_state.get('race_num', 1),
                            'race_type': race_type_detected,
                            'surface': surface_type,
                            'distance': distance_txt,
                            'condition': condition_txt,
                            'purse': purse_val,
                            'field_size': len(primary_df),
                            # ADDITIONAL INTELLIGENT FEATURES
                            'ppi_race_wide': ppi_val,  # Pace Pressure Index
                            'track_bias_config': TRACK_BIAS_PROFILES.get(track_name, {}).get(surface_type, {}).get(race_bucket, {}),
                            'early_speed_count': len([r for r in primary_df.iterrows() if r[1].get('E1_Style') in ['E', 'EP']]),
                            'presser_count': len([r for r in primary_df.iterrows() if r[1].get('E1_Style') in ['P', 'EP']]),
                            'closer_count': len([r for r in primary_df.iterrows() if r[1].get('E1_Style') == 'S']),
                            'avg_field_beyer': safe_float(primary_df.get('Best Beyer', pd.Series([0])).mean()),
                            'top3_beyer_avg': safe_float(primary_df.nlargest(3, 'Best Beyer', keep='first').get('Best Beyer', pd.Series([0])).mean() if 'Best Beyer' in primary_df.columns else 0),
                            'avg_field_days_off': safe_float(primary_df.get('Days Since', pd.Series([0])).mean()),
                            'chaos_index': 0.0,  # Field unpredictability metric (0 = predictable, higher = chaotic)
                            'race_bucket': race_bucket,  # Distance category: ≤6f, 6.5-7f, or 8f+
                            'is_maiden': 'maiden' in race_type_detected.lower() or 'mdn' in race_type_detected.lower(),
                            'is_stakes': 'stakes' in race_type_detected.lower() or 'stk' in race_type_detected.lower() or any(g in race_type_detected.lower() for g in ['g1', 'g2', 'g3']),
                            'is_turf': surface_type.lower() == 'turf',
                            'is_synthetic': surface_type.lower() in ['synthetic', 'tapeta', 'polytrack']
                        }

                        # Prepare horses data with ALL AVAILABLE FEATURES for maximum ML intelligence
                        horses_data = []
                        for idx, row in primary_df.iterrows():
                            horse_name = str(row.get('Horse', f'Horse_{idx + 1}'))

                            # Extract Fair % with percentage handling
                            fair_pct_raw = row.get('Fair %', 0.0)
                            fair_pct_value = safe_float(fair_pct_raw) / 100.0  # Convert to probability (0-1)

                            # Get individual horse angles and pedigree data
                            angles_df = angles_per_horse.get(horse_name)
                            pedigree_data = pedigree_per_horse.get(horse_name, {})

                            # Parse individual angle categories for granular ML features
                            angle_early_speed = 0.0
                            angle_class_move = 0.0
                            angle_recency = 0.0
                            angle_workout = 0.0
                            angle_connections = 0.0
                            angle_surface_switch = 0.0
                            angle_distance_switch = 0.0
                            angle_debut = 0.0

                            if angles_df is not None and not angles_df.empty:
                                cats_lower = " ".join(angles_df["Category"].astype(str).tolist()).lower()
                                # Extract specific angle values for intelligent pattern recognition
                                if "early speed" in cats_lower:
                                    angle_early_speed = 1.0
                                if "class" in cats_lower or "up in class" in cats_lower or "down in class" in cats_lower:
                                    angle_class_move = 1.0
                                if "last out" in cats_lower or "recent" in cats_lower or "30 days" in cats_lower:
                                    angle_recency = 1.0
                                if "workout" in cats_lower or "work" in cats_lower:
                                    angle_workout = 1.0
                                if "trainer" in cats_lower or "jockey" in cats_lower or "combo" in cats_lower:
                                    angle_connections = 1.0
                                if "turf to dirt" in cats_lower or "dirt to turf" in cats_lower:
                                    angle_surface_switch = 1.0
                                if "distance" in cats_lower or "sprint" in cats_lower or "route" in cats_lower:
                                    angle_distance_switch = 1.0
                                if "debut" in cats_lower or "1st time" in cats_lower or "maiden sp wt" in cats_lower:
                                    angle_debut = 1.0

                            horse_dict = {
                                'program_number': int(safe_float(name_to_post.get(horse_name, idx + 1), idx + 1)),
                                'horse_name': horse_name,
                                'post_position': int(safe_float(name_to_post.get(horse_name, idx + 1), idx + 1)),
                                'morning_line_odds': safe_float(name_to_odds.get(horse_name, '99'), 99.0),
                                'jockey': str(row.get('Jockey', '')),
                                'trainer': str(row.get('Trainer', '')),
                                'owner': str(row.get('Owner', '')),
                                'running_style': str(row.get('E1_Style', 'P')),
                                'prime_power': safe_float(row.get('Prime Power', 0.0)),
                                'best_beyer': int(safe_float(row.get('Best Beyer', 0))),
                                'last_beyer': int(safe_float(row.get('Last Beyer', 0))),
                                'avg_beyer_3': safe_float(row.get('Avg Beyer (3)', 0.0)),
                                'e1_pace': safe_float(row.get('E1', 0.0)),
                                'e2_pace': safe_float(row.get('E2', 0.0)),
                                'late_pace': safe_float(row.get('Late', 0.0)),
                                'days_since_last': int(safe_float(row.get('Days Since', 0))),
                                'class_rating': safe_float(row.get('Class Rating', 0.0)),
                                'form_rating': safe_float(row.get('Form Rating', 0.0)),
                                'speed_rating': safe_float(row.get('Speed Rating', 0.0)),
                                'pace_rating': safe_float(row.get('Pace Rating', 0.0)),
                                'style_rating': safe_float(row.get('Style Rating', 0.0)),
                                'post_rating': safe_float(row.get('Post Rating', 0.0)),
                                'angles_total': safe_float(row.get('Angles Total', 0.0)),
                                'rating_final': safe_float(row.get('R', 0.0)),
                                'predicted_probability': fair_pct_value,
                                'predicted_rank': int(idx + 1),
                                'fair_odds': safe_float(row.get('Fair Odds', 99.0), 99.0),
                                # PhD enhancements if available
                                'rating_confidence': safe_float(row.get('Confidence', 0.5), 0.5),
                                'form_decay_score': safe_float(row.get('Form Decay', 0.0)),
                                'pace_esp_score': safe_float(row.get('Pace ESP', 0.0)),
                                'mud_adjustment': safe_float(row.get('Mud Adj', 0.0)),
                                # INDIVIDUAL ANGLE FEATURES (8 key angles for ML pattern learning)
                                'angle_early_speed': angle_early_speed,
                                'angle_class': angle_class_move,
                                'angle_recency': angle_recency,
                                'angle_work_pattern': angle_workout,
                                'angle_connections': angle_connections,
                                'angle_pedigree': safe_float(pedigree_data.get('sire_1st', 0.0)) / 100.0 if pedigree_data else 0.0,
                                'angle_runstyle_bias': 1.0 if row.get('E1_Style') in ['E', 'EP'] else 0.0,
                                'angle_post': safe_float(row.get('Post Rating', 0.0)),
                                # PEDIGREE FEATURES (critical for surface/distance/mud breeding)
                                'pedigree_sire_awd': safe_float(pedigree_data.get('sire_awd', 7.0)) if pedigree_data else 7.0,
                                'pedigree_sire_1st_pct': safe_float(pedigree_data.get('sire_1st', 0.0)) if pedigree_data else 0.0,
                                'pedigree_damsire_awd': safe_float(pedigree_data.get('damsire_awd', 7.0)) if pedigree_data else 7.0,
                                'pedigree_damsire_1st_pct': safe_float(pedigree_data.get('damsire_1st', 0.0)) if pedigree_data else 0.0,
                                'pedigree_dam_dpi': safe_float(pedigree_data.get('dam_dpi', 1.0)) if pedigree_data else 1.0,
                                # ADDITIONAL CONTEXTUAL FEATURES
                                'quirin_points': int(safe_float(row.get('Quirin', 0))),
                                'style_strength': safe_float(row.get('StyleStrength', 0.0)),
                                'ppi_individual': safe_float(ppi_map_by_horse.get(horse_name, 0.0)),
                                'field_size_context': len(primary_df),
                                'post_position_bias': safe_float(row.get('Post', 0)) / max(len(primary_df), 1),
                                # LIFETIME STATS (if available from PP text)
                                'starts_lifetime': int(safe_float(row.get('Starts', 0))),
                                'wins_lifetime': int(safe_float(row.get('Wins', 0))),
                                'win_pct': safe_float(row.get('Win%', 0.0)),
                                'earnings_lifetime': safe_float(row.get('Earnings', 0.0))
                            }
                            horses_data.append(horse_dict)

                        # Save to gold database
                        pp_text_raw = st.session_state.get('pp_text_cache', '')
                        success = gold_db.save_analyzed_race(
                            race_id=race_id,
                            race_metadata=race_metadata,
                            horses_data=horses_data,
                            pp_text_raw=pp_text_raw
                        )

                        if success:
                            st.success(f"💾 **Auto-saved to gold database:** `{race_id}`")
                            st.info(f"📊 Saved {len(horses_data)} horses with {len([k for k in horses_data[0].keys() if k.startswith('rating_') or k.startswith('angle_')])} ML features each")
                            st.session_state['last_saved_race_id'] = race_id
                            st.info("🏁 After race completes, submit actual top 5 finishers in **Section E** below!")
                        else:
                            st.error(f"❌ Failed to save race {race_id} to database")

                    except Exception as save_error:
                        st.warning(f"Could not auto-save race: {save_error}")
                        # Don't fail the entire analysis if save fails
                # ============================================================

            except Exception as e:
                st.error(f"Error generating report: {e}")
                import traceback
                st.error(traceback.format_exc())

# ===================== E. GOLD HIGH-IQ SYSTEM 🏆 (Real Data → 90%) =====================

st.markdown("---")

# Show database info prominently before the header
if GOLD_DB_AVAILABLE and gold_db is not None:
    try:
        # Get quick stats for header display
        stats = gold_db.get_accuracy_stats()
        pending_races = gold_db.get_pending_races(limit=1000)  # Get all pending
        total_saved = stats.get('total_races', 0) + len(pending_races)

        st.header(f"E. Gold High-IQ System 🏆 - {total_saved} Races Saved")
        if total_saved > 0:
            st.success(f"💾 **Database Active:** {stats.get('total_races', 0)} races with results, {len(pending_races)} pending results")
    except BaseException:
        st.header("E. Gold High-IQ System 🏆 (Real Data → 90% Accuracy)")
else:
    st.header("E. Gold High-IQ System 🏆 (Real Data → 90% Accuracy)")

if not GOLD_DB_AVAILABLE or gold_db is None:
    st.error("❌ Gold High-IQ Database not available. Check initialization.")
    st.warning("""
    **Troubleshooting:**
    1. Ensure `gold_database_manager.py` exists
    2. Check that SQLite3 is working
    3. Restart the Streamlit app

    Files needed:
    - gold_database_manager.py
    - gold_database_schema.sql
    - retrain_model.py
    """)
else:
    # VERIFY DATABASE PERSISTENCE ON STARTUP
    try:
        import os
        db_path = gold_db.db_path
        if os.path.exists(db_path):
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            st.success(f"✅ **Database Verified:** {db_path} ({db_size_mb:.2f} MB) - Data persists across sessions!")
        else:
            st.info(f"📁 New database will be created: {db_path}")
    except Exception as verify_error:
        st.warning(f"Could not verify database: {verify_error}")

    # Get stats from gold database
    try:
        stats = gold_db.get_accuracy_stats()
        pending_races = gold_db.get_pending_races(limit=20)

        # Create tabs for Gold High-IQ System
        tab_overview, tab_results, tab_calibration, tab_retrain = st.tabs(
            ["📊 Dashboard", "🏁 Submit Actual Top 5", "🤖 Auto-Calibration Monitor", "🚀 Retrain Model"]
        )

        # Tab 1: Dashboard
        with tab_overview:
            st.markdown("""
            ### Real Data Learning System

            Every time you click "Analyze This Race", the system **auto-saves**:
            - All horse features (speed, class, pace, angles, PhD enhancements)
            - Model predictions (probabilities, ratings, confidence)
            - Race metadata (track, conditions, purse, etc.)

            After the race completes, submit the actual top 5 finishers below.
            The system learns from real outcomes to reach 90%+ accuracy.

            🔒 **Data Persistence:** All analyzed races are permanently saved to the database.
            You can safely close your browser and return anytime - your data persists!

            📁 **Database Location:** `gold_high_iq.db` (stored in your project folder)
            """)

            # Calculate total analyzed races (completed + pending)
            completed_races = stats.get('total_races', 0)
            total_analyzed = completed_races + len(pending_races)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyzed", total_analyzed, help="All races saved (completed + pending results)")
            with col2:
                st.metric(
                    "With Results",
                    completed_races,
                    help="Races with actual results entered - used for accuracy tracking")
            with col3:
                st.metric("Pending Results", len(pending_races), help="Races analyzed but awaiting actual finishers")
            with col4:
                if completed_races > 0:
                    st.metric("Winner Accuracy", f"{stats.get('winner_accuracy', 0.0):.1%}")
                else:
                    st.metric("Winner Accuracy", "N/A", help="Submit results to track accuracy")

            # Training readiness indicator
            st.markdown("#### Training Readiness")
            ready = completed_races >= 50
            if ready:
                st.success(f"✅ Ready to retrain! You have {completed_races} completed races.")
            else:
                st.info(f"⏳ Need 50 completed races to retrain model. Current progress: {completed_races}/50")

            # Progress bars
            st.markdown("#### Progress to Milestones")

            milestones = [
                (50, "First Retrain", "70-75%"),
                (100, "Second Retrain", "75-80%"),
                (500, "Major Improvement", "85-87%"),
                (1000, "Gold Standard", "90%+")
            ]

            for target, label, expected_acc in milestones:
                progress = min(completed_races / target, 1.0)
                st.progress(
                    progress,
                    text=f"{label} ({expected_acc} expected): {completed_races}/{target} races"
                )

            # System performance
            if stats.get('total_races', 0) > 0:
                st.markdown("#### System Performance")
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Winner Accuracy", f"{stats.get('winner_accuracy', 0.0):.1%}")
                with col2:
                    st.metric("Top-3 Accuracy", f"{stats.get('top3_accuracy', 0.0):.1%}")
                with col3:
                    st.metric("Top-5 Accuracy", f"{stats.get('top5_accuracy', 0.0):.1%}")

        # Tab 2: Submit Actual Top 5
        with tab_results:
            # Show success message if just saved
            if st.session_state.get('last_save_success'):
                race_id = st.session_state.get('last_save_race_id', 'Unknown')
                actual_winner = st.session_state.get('last_save_winner', 'Unknown')
                predicted_winner = st.session_state.get('last_save_predicted', 'Unknown')

                if predicted_winner == actual_winner:
                    st.success(f"🎯 **Prediction Correct!** Winner: {actual_winner}")
                else:
                    st.info(f"📊 Predicted: {predicted_winner} | Actual Winner: {actual_winner}")

                st.success(f"✅ Results successfully saved for {race_id}")
                st.info("🚀 Go to 'Retrain Model' tab when you have 50+ completed races!")

                # Clear the flag
                st.session_state['last_save_success'] = False
                st.markdown("---")

            st.markdown("""
            ### Submit Actual Top 5 Finishers

            After a race completes, enter the actual finishing order here.
            **Only the top 5 positions are required** for high-quality ML training.
            """)

            if not pending_races:
                st.success("✅ No pending races! All analyzed races have results entered.")
                st.info("💡 Analyze more races in Sections 1-4 to build training data.")
            else:
                st.info(f"📋 {len(pending_races)} races awaiting results")

                # Select race
                race_options = [
                    f"{r[1]} R{r[3]} on {r[2]} ({r[4]} horses)"
                    for r in pending_races
                ]
                selected_idx = st.selectbox(
                    "Select Race to Enter Results:",
                    range(len(race_options)),
                    format_func=lambda i: race_options[i],
                    key="select_pending_race"
                )

                if selected_idx is not None:
                    selected_race = pending_races[selected_idx]
                    race_id, track, date, race_num, field_size = selected_race

                    st.markdown(f"#### 🏇 {race_id}")
                    st.caption(f"{field_size} horses ran in this race")

                    # Get horses for this race
                    horses = gold_db.get_horses_for_race(race_id)

                    if not horses:
                        st.error("No horses found for this race.")
                    else:
                        # Display horses in clean table
                        st.markdown("**Horses in this race:**")

                        horses_df = pd.DataFrame(horses)
                        display_df = horses_df[['program_number', 'horse_name', 'post_position',
                                               'predicted_probability', 'fair_odds']].copy()
                        # Format values BEFORE renaming columns
                        display_df['predicted_probability'] = (
                            display_df['predicted_probability'] * 100).round(1).astype(str) + '%'
                        display_df['fair_odds'] = display_df['fair_odds'].round(2)
                        # Now rename columns for display
                        display_df.columns = ['#', 'Horse Name', 'Post', 'Predicted Win %', 'Fair Odds']

                        st.dataframe(display_df, use_container_width=True, hide_index=True)

                        # Enter top 5
                        st.markdown("---")
                        st.markdown("### 🏆 Enter Actual Top 5 Finishers")
                        st.caption("Enter program numbers separated by commas and press ENTER to auto-save")

                        # Get horse data for validation and display
                        program_numbers = sorted([int(h['program_number']) for h in horses])
                        horse_names_dict = {int(h['program_number']): h['horse_name'] for h in horses}

                        # Text input for finish order - NO callback (simpler approach)
                        finish_input = st.text_input(
                            "Finishing order (1st through 5th) - Press ENTER to save",
                            placeholder="Example: 5,12,1,8,7",
                            key=f"finish_input_{race_id}",
                            help="Type the program numbers in order from 1st to 5th, separated by commas, then press ENTER")

                        # Process the input if provided
                        if finish_input and finish_input.strip():
                            finish_order = []

                            try:
                                # Parse comma-separated values
                                raw_values = [x.strip() for x in finish_input.split(',')]
                                finish_order = [int(x) for x in raw_values if x]

                                # Validation checks
                                if len(finish_order) < 5:
                                    st.error(f"❌ Need 5 horses, only got {len(finish_order)}")
                                elif len(finish_order) > 5:
                                    st.warning(f"⚠️ Too many horses ({len(finish_order)}), using first 5")
                                    finish_order = finish_order[:5]

                                # Check for duplicates
                                elif len(finish_order) != len(set(finish_order)):
                                    st.error("❌ Cannot use same horse in multiple positions")
                                    finish_order = []

                                # Check all are valid program numbers
                                else:
                                    invalid = [x for x in finish_order if x not in program_numbers]
                                    if invalid:
                                        st.error(f"❌ Invalid program numbers: {', '.join(map(str, invalid))}")
                                        finish_order = []
                                    else:
                                        # Valid input - show preview and save button
                                        preview_parts = []
                                        for i, pos in enumerate(finish_order[:5]):
                                            horse_name = horse_names_dict.get(pos, 'Unknown')
                                            medals = ["🥇", "🥈", "🥉", "4th", "5th"]
                                            preview_parts.append(f"{medals[i]} #{pos} {horse_name}")
                                        st.success(" → ".join(preview_parts))

                                        # Auto-save button
                                        if st.button("💾 Save Results", key=f"save_btn_{race_id}", type="primary"):
                                            with st.spinner("Saving to database..."):
                                                try:
                                                    success = gold_db.submit_race_results(
                                                        race_id=race_id,
                                                        finish_order_programs=finish_order
                                                    )

                                                    if success:
                                                        # Store success info
                                                        st.session_state['last_save_success'] = True
                                                        st.session_state['last_save_race_id'] = race_id
                                                        st.session_state['last_save_winner'] = horse_names_dict[finish_order[0]]

                                                        # Check if we predicted correctly
                                                        predicted_winner_row = horses_df[horses_df['predicted_rank'] == 1]
                                                        predicted_winner = predicted_winner_row['horse_name'].values[
                                                            0] if not predicted_winner_row.empty else 'Unknown'
                                                        st.session_state['last_save_predicted'] = predicted_winner

                                                        st.toast(
                                                            f"🏁 Results saved! Winner: #{finish_order[0]} {horse_names_dict[finish_order[0]]}", icon="✅")

                                                        # AUTO-CALIBRATION: Learn from result
                                                        try:
                                                            from auto_calibration_engine import auto_calibrate_on_result_submission
                                                            calibration_result = auto_calibrate_on_result_submission(
                                                                gold_db.db_path)

                                                            if calibration_result.get('status') != 'skipped':
                                                                accuracy = calibration_result.get(
                                                                    'winner_accuracy', 0) * 100
                                                                st.toast(
                                                                    f"🤖 Model auto-calibrated! Winner accuracy: {accuracy:.1f}%", icon="🧠")
                                                        except Exception as cal_err:
                                                            logger.warning(f"Auto-calibration failed: {cal_err}")

                                                        # Clear the input
                                                        st.session_state[f"finish_input_{race_id}"] = ""

                                                        time.sleep(0.5)
                                                        _safe_rerun()
                                                    else:
                                                        st.error("❌ Failed to save to database")

                                                except Exception as e:
                                                    st.error(f"❌ Error: {str(e)}")
                                                    import traceback
                                                    st.code(traceback.format_exc(), language='python')

                            except ValueError:
                                st.error("❌ Invalid format - use numbers separated by commas (e.g., 5,12,1,8,7)")
                        else:
                            st.info("💡 Example: Type **5,12,1,8,7** then click Save Results")

                        st.markdown("---")

                # Show recently saved results for verification
                st.markdown("---")
                st.markdown("### 📊 Database Integrity Verification")

                try:
                    import sqlite3
                    import os
                    # CRITICAL FIX: Add timeout to prevent database lock hangs
                    conn = sqlite3.connect(gold_db.db_path, timeout=10.0)
                    cursor = conn.cursor()

                    # COMPREHENSIVE DATABASE CHECK
                    # 1. Total race count
                    cursor.execute("SELECT COUNT(DISTINCT race_id) FROM races_analyzed")
                    total_analyzed = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(DISTINCT race_id) FROM gold_high_iq")
                    total_with_results = cursor.fetchone()[0]

                    cursor.execute("SELECT COUNT(*) FROM gold_high_iq")
                    total_horses_saved = cursor.fetchone()[0]

                    # 2. Database file persistence
                    db_size = os.path.getsize(gold_db.db_path) / (1024 * 1024)  # MB

                    # 3. Show verification metrics
                    col1, col2, col3, col4 = st.columns(4)
                    with col1:
                        st.metric("DB Size", f"{db_size:.2f} MB",
                                  help="Physical database file size - persists across sessions")
                    with col2:
                        st.metric("Races Analyzed", total_analyzed, help="All races saved automatically")
                    with col3:
                        st.metric("With Results", total_with_results, help="Races ready for ML training")
                    with col4:
                        st.metric(
                            "Horses Saved",
                            total_horses_saved,
                            help="Total training examples (5 per completed race)")

                    st.success(f"🔒 **Data Persistence Verified:** All data stored in `{gold_db.db_path}` - survives browser close/reopen!")

                    # 4. Get last 10 saved results
                    cursor.execute("""
                        SELECT
                            race_id,
                            horse_name,
                            actual_finish_position,
                            predicted_rank,
                            prediction_error,
                            result_entered_timestamp
                        FROM gold_high_iq
                        ORDER BY result_entered_timestamp DESC
                        LIMIT 10
                    """)

                    recent_results = cursor.fetchall()
                    conn.close()

                    if recent_results:
                        st.markdown("#### 🏇 Last 10 Training Examples Saved")

                        verify_df = pd.DataFrame(recent_results, columns=[
                            'Race ID', 'Horse', 'Actual Pos', 'Predicted Pos',
                            'Error', 'Saved At'
                        ])

                        # Format the dataframe
                        verify_df['Error'] = verify_df['Error'].round(1)
                        verify_df['Saved At'] = pd.to_datetime(verify_df['Saved At']).dt.strftime('%m/%d %I:%M %p')

                        st.dataframe(verify_df, use_container_width=True, hide_index=True)

                        st.caption(
                            "💡 Each completed race saves 5 entries (top 5 finishers). Database automatically commits and persists all data.")
                    else:
                        st.info("📋 No results in database yet. Submit your first race above to begin ML training data collection!")

                except Exception as e:
                    st.error(f"❌ Database verification failed: {str(e)}")
                    st.warning("⚠️ This may indicate database corruption or file permission issues.")

        # Tab 3: Auto-Calibration Monitor
        with tab_calibration:
            st.markdown("""
            ### 🤖 Real-Time Model Learning Monitor
            
            This dashboard shows **proof** that your model is automatically learning from each race result.
            After every result submission, the system adjusts component weights using gradient descent.
            """)
            
            # Load current weights from unified_rating_engine
            try:
                import importlib
                import unified_rating_engine
                importlib.reload(unified_rating_engine)
                
                current_weights = unified_rating_engine.BASE_WEIGHTS.copy()
                
                st.markdown("#### 🎯 Current Component Weights")
                st.caption("These weights determine how much each factor influences the final rating")
                
                # Display weights as metrics
                cols = st.columns(6)
                weight_names = ['class', 'speed', 'form', 'pace', 'style', 'post']
                for idx, weight_name in enumerate(weight_names):
                    with cols[idx]:
                        weight_val = current_weights.get(weight_name, 0.0)
                        st.metric(
                            weight_name.capitalize(),
                            f"{weight_val:.3f}",
                            help=f"Current emphasis on {weight_name} factor"
                        )
                
                st.markdown("---")
                
                # Load calibration history
                import os
                import json
                history_file = "calibration_history.json"
                
                if os.path.exists(history_file):
                    with open(history_file, 'r') as f:
                        calibration_history = json.load(f)
                    
                    if calibration_history:
                        st.markdown("#### 📈 Recent Calibration Events")
                        st.caption(f"Showing last {min(len(calibration_history), 20)} auto-calibration updates")
                        
                        # Create DataFrame from history
                        history_records = []
                        for event in calibration_history[-20:]:  # Last 20 events
                            record = {
                                'Timestamp': event.get('timestamp', 'N/A'),
                                'Races Used': event.get('num_races', 0),
                                'Winner Acc': f"{event.get('winner_accuracy', 0) * 100:.1f}%",
                                'Avg Loss': f"{event.get('avg_loss', 0):.4f}",
                            }
                            
                            # Add weight changes
                            weight_updates = event.get('weight_updates', {})
                            for w_name in weight_names:
                                if w_name in weight_updates:
                                    delta = weight_updates[w_name]
                                    record[f'{w_name.capitalize()} Δ'] = f"{delta:+.4f}"
                            
                            history_records.append(record)
                        
                        if history_records:
                            history_df = pd.DataFrame(history_records)
                            st.dataframe(history_df, use_container_width=True, hide_index=True)
                            
                            # Show learning progress
                            st.markdown("#### 📊 Learning Progress")
                            
                            # Calculate total calibrations
                            total_calibrations = len(calibration_history)
                            recent_accuracy = calibration_history[-1].get('winner_accuracy', 0) * 100 if calibration_history else 0
                            
                            col1, col2, col3 = st.columns(3)
                            with col1:
                                st.metric(
                                    "Total Calibrations",
                                    total_calibrations,
                                    help="Number of times the model has auto-adjusted"
                                )
                            with col2:
                                st.metric(
                                    "Latest Winner Accuracy",
                                    f"{recent_accuracy:.1f}%",
                                    help="Most recent winner prediction accuracy"
                                )
                            with col3:
                                # Calculate average loss trend
                                if len(calibration_history) >= 2:
                                    recent_loss = calibration_history[-1].get('avg_loss', 0)
                                    previous_loss = calibration_history[-2].get('avg_loss', 0)
                                    loss_delta = recent_loss - previous_loss
                                    st.metric(
                                        "Loss Change",
                                        f"{recent_loss:.4f}",
                                        delta=f"{loss_delta:.4f}",
                                        delta_color="inverse",
                                        help="Lower is better - model is learning to minimize prediction error"
                                    )
                                else:
                                    st.metric("Loss Change", "N/A", help="Need 2+ calibrations to show trend")
                            
                            st.success("✅ **Auto-Calibration Active:** Model updates after every race result submission")
                            
                            # Show weight evolution chart (if enough data)
                            if len(calibration_history) >= 5:
                                st.markdown("#### 📉 Weight Evolution Over Time")
                                
                                # Extract weight history
                                weight_history = {w: [] for w in weight_names}
                                timestamps = []
                                
                                for event in calibration_history[-20:]:
                                    timestamps.append(event.get('timestamp', ''))
                                    weights = event.get('current_weights', {})
                                    for w_name in weight_names:
                                        weight_history[w_name].append(weights.get(w_name, 0))
                                
                                # Create line chart data
                                chart_data = pd.DataFrame(weight_history, index=timestamps)
                                st.line_chart(chart_data)
                                st.caption("📌 Watch how weights adjust based on race outcomes - evidence of real-time learning!")
                        else:
                            st.info("📋 No calibration events recorded yet. Submit race results to trigger auto-learning!")
                    else:
                        st.info("📋 No calibration history yet. The model will auto-calibrate after you submit race results.")
                else:
                    st.info(f"📋 Calibration history file not found: `{history_file}`")
                    st.caption("The system will create this file automatically after the first auto-calibration event.")
                
            except Exception as e:
                st.error(f"❌ Error loading calibration data: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

        # Tab 4: Retrain Model
        with tab_retrain:
            st.markdown("""
            ### Retrain ML Model with Real Data

            Once you have **50+ completed races**, retrain the model to learn from real outcomes.
            The model uses PyTorch with Plackett-Luce ranking loss for optimal accuracy.
            """)

            # Check if ready
            ready_to_train = stats.get('total_races', 0) >= 50

            if not ready_to_train:
                st.warning(f"⏳ Need at least 50 completed races. Currently: {stats.get('total_races', 0)}")
                st.info("💡 Complete more races in the 'Submit Actual Top 5' tab.")
            else:
                st.success(f"✅ Ready to train! {stats.get('total_races', 0)} races available.")

                # Training parameters
                col1, col2, col3 = st.columns(3)
                with col1:
                    epochs = st.number_input("Epochs", min_value=10, max_value=200, value=50)
                with col2:
                    learning_rate = st.select_slider(
                        "Learning Rate",
                        options=[0.0001, 0.0005, 0.001, 0.005, 0.01],
                        value=0.001
                    )
                with col3:
                    batch_size = st.selectbox("Batch Size", [4, 8, 16, 32], index=1)

                # Train button
                if st.button("🚀 Start Retraining", type="primary", key="retrain_btn"):
                    with st.spinner(f"Training model on {stats.get('total_races', 0)} races... This may take 2-5 minutes..."):
                        try:
                            from retrain_model import retrain_model

                            results = retrain_model(
                                db_path=gold_db.db_path,
                                epochs=epochs,
                                learning_rate=learning_rate,
                                batch_size=batch_size,
                                min_races=50
                            )

                            if 'error' in results:
                                st.error(f"❌ Training failed: {results['error']}")
                            else:
                                st.success("✅ Training complete!")

                                # Display results
                                col1, col2, col3 = st.columns(3)
                                with col1:
                                    st.metric(
                                        "Winner Accuracy",
                                        f"{results['metrics']['winner_accuracy']:.1%}"
                                    )
                                with col2:
                                    st.metric(
                                        "Top-3 Accuracy",
                                        f"{results['metrics']['top3_accuracy']:.1%}"
                                    )
                                with col3:
                                    st.metric(
                                        "Top-5 Accuracy",
                                        f"{results['metrics']['top5_accuracy']:.1%}"
                                    )

                                st.info(f"⏱️ Training time: {results.get('duration', 0):.1f} seconds")
                                st.info(f"💾 Model saved: {results.get('model_path', 'N/A')}")

                                st.balloons()

                        except Exception as e:
                            st.error(f"Training error: {e}")
                            import traceback
                            st.code(traceback.format_exc())

                # Training history
                st.markdown("---")
                st.markdown("### Training History")

                try:
                    import sqlite3
                    # CRITICAL FIX: Add timeout to prevent database lock hangs
                    conn = sqlite3.connect(gold_db.db_path, timeout=10.0)
                    history_df = pd.read_sql_query("""
                        SELECT
                            retrain_timestamp,
                            total_races_used,
                            val_winner_accuracy,
                            val_top3_accuracy,
                            val_top5_accuracy,
                            training_duration_seconds
                        FROM retraining_history
                        ORDER BY retrain_timestamp DESC
                        LIMIT 10
                    """, conn)
                    conn.close()

                    if not history_df.empty:
                        history_df.columns = [
                            'Timestamp', 'Races Used', 'Winner Acc',
                            'Top-3 Acc', 'Top-5 Acc', 'Duration (s)'
                        ]
                        history_df['Winner Acc'] = (history_df['Winner Acc'] * 100).round(1).astype(str) + '%'
                        history_df['Top-3 Acc'] = (history_df['Top-3 Acc'] * 100).round(1).astype(str) + '%'
                        history_df['Top-5 Acc'] = (history_df['Top-5 Acc'] * 100).round(1).astype(str) + '%'
                        history_df['Duration (s)'] = history_df['Duration (s)'].round(1)

                        st.dataframe(history_df, use_container_width=True, hide_index=True)
                    else:
                        st.info("No training history yet. Train the model to see results here.")
                except Exception as e:
                    st.warning("Could not load training history.")

    except Exception as e:
        st.error(f"Error in Gold High-IQ System: {e}")
        import traceback
        st.code(traceback.format_exc())

# End of Section E

st.markdown("---")
st.caption("Horse Race Ready - IQ Mode | Advanced Track Bias Analysis with ML Probability Calibration & Real Data Training")
