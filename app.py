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
from typing import Dict, List, Tuple, Optional
from datetime import datetime
from itertools import product

import numpy as np
import pandas as pd
import streamlit as st

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
    st.session_state['track_name'] = "Keeneland"
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
    "speed_fig_weight": 0.05, # (Fig - Avg) * Weight. 0.05 = 10 fig points = 0.5 bonus.
    "first_timer_fig_default": 50, # Assumed speed fig for a 1st-time starter.

    # --- Pace & Style Model ---
    "ppi_multiplier": 1.5, # Overall impact of the Pace Pressure Index (PPI).
    "ppi_tailwind_factor": 0.6, # How much of the PPI value is given to E/EP or S horses.
    "style_strength_weights": { # Multiplier for pace tailwind based on strength.
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
    "style_quirin_threshold": 6, # Quirin score needed for "strong" style bonus.
    "style_quirin_bonus": 0.10, # Bonus for strong style (e.g., E w/ Q>=6).
    "post_bias_rail_bonus": 0.40,
    "post_bias_inner_bonus": 0.25,
    "post_bias_mid_bonus": 0.15,
    "post_bias_outside_bonus": 0.25,

    # --- Pedigree & Angle Tweaks (Cclass) ---
    "ped_dist_bonus": 0.06,
    "ped_dist_penalty": -0.04, # Note: This should be negative
    "ped_dist_neutral_bonus": 0.03,
    "ped_first_pct_threshold": 14, # Sire/Damsire 1st-time-win %
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
    "angle_roi_pos_max_bonus": 0.06, # Max bonus from positive ROI angles
    "angle_roi_pos_per_bonus": 0.01, # Bonus per positive ROI angle
    "angle_roi_neg_max_penalty": 0.03, # Max penalty from negative ROI angles (applied as -)
    "angle_roi_neg_per_penalty": 0.005, # Penalty per negative ROI angle (applied as -)
    "angle_tweak_min_clip": -0.12, # Min/Max total adjustment from all angles
    "angle_tweak_max_clip": 0.12,

    # --- Exotics & Strategy ---
    "exotic_bias_weights": (1.30, 1.15, 1.05, 1.03), # (1st, 2nd, 3rd, 4th) Harville bias
    "strategy_confident": { # Placeholders, not used by new strategy builder
        "ex_max": 4, "ex_min_prob": 0.020,
        "tri_max": 6, "tri_min_prob": 0.010,
        "sup_max": 8, "sup_min_prob": 0.008,
    },
    "strategy_value": { # Placeholders, not used by new strategy builder
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
            if val <= 6.0:  return "≤6"
            if val < 8.0:   return "6.5–7"
            return "8f+"
    # Miles
    if "mile" in d:
        if "70" in d and "yard" in d:
            return "8f+"
        fracs = {"1/16": 1/16, "1/8": 1/8, "3/16": 3/16, "1/4": 1/4,
                 "5/16": 5/16, "3/8": 3/8, "7/16": 7/16, "1/2": 0.5}
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
        if total_f < 6.5: return "≤6"
        if total_f < 8.0: return "6.5–7"
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
    head = (pp_text or "")[:800].lower()
    for token, canon in _CANON_BY_TOKEN.items():
        if re.search(rf'\b{re.escape(token)}\b', head):
            return canon
    for canon, toks in TRACK_ALIASES.items():
        for t in toks:
            t_words = [w for w in t.split() if len(w) > 2]
            if t_words and all(re.search(rf'\b{re.escape(w)}\b', head) for w in t_words):
                return canon
    return ""

# -------- Race-type constants + detection --------
# This dictionary is our constant. It measures the "reliability" of the race type.
base_class_bias = {
    "stakes (g1)": 0.90,
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
    if re.search(r'\b(g1|grade\s*i)\b', s): return "stakes (g1)"
    if re.search(r'\b(g2|grade\s*ii)\b', s): return "stakes (g2)"
    if re.search(r'\b(g3|grade\s*iii)\b', s): return "stakes (g3)"

    # Listed / generic stakes
    if "listed" in s: return "stakes (listed)"
    if re.search(r'\bstakes?\b', s): return "stakes"

    # Maiden
    if re.search(r'\b(mdn|maiden)\b', s):
        if re.search(r'(mcl|mdn\s*clm|maiden\s*claim)', s): return "maiden claiming"
        if re.search(r'(msw|maiden\s*special|maiden\s*sp\s*wt)', s): return "maiden special weight"
        return "maiden special weight"

    # AOC
    if re.search(r'\b(oc|aoc|optional\s*claim)\b', s): return "allowance optional claiming (aoc)"

    # Starter
    if re.search(r'\bstarter\s*allow', s): return "starter allowance"
    if re.search(r'\bstarter\s*h(andi)?cap\b', s): return "starter handicap"

    # Waiver Claiming
    if re.search(r'\b(waiver|wcl|w\s*clm)\b', s): return "waiver claiming"

    # Claiming
    if re.search(r'\bclm|claiming\b', s): return "claiming"

    # Allowance last
    if re.search(r'\ballow(ance)?\b', s): return "allowance"

    return "allowance"

# -------- Track bias profiles (additive deltas; conservative magnitude) --------
TRACK_BIAS_PROFILES = {
    "Keeneland": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.35, "E/P": 0.20, "P": -0.10, "S": -0.25},
                         "post":     {"rail": 0.20, "inner": 0.10, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.15, "E/P": 0.10, "P": 0.00, "S": -0.10},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Del Mar": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Churchill Downs": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.15, "E/P": 0.05, "P": 0.00, "S": -0.10},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Kentucky Downs": {
        "Turf": {
            "≤6f":    {"runstyle": {"E": -0.05, "E/P": 0.00, "P": 0.10, "S": 0.15},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}},
            "6.5–7f": {"runstyle": {"E": -0.05, "E/P": 0.00, "P": 0.10, "S": 0.15},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}},
            "8f+":    {"runstyle": {"E": -0.10, "E/P": 0.00, "P": 0.10, "S": 0.20},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}}
        }
    },
    "Saratoga": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Santa Anita": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Mountaineer": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        }
    },
    "Charles Town": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.45, "E/P": 0.25, "P": -0.15, "S": -0.35},
                         "post":     {"rail": 0.25, "inner": 0.15, "mid": -0.05, "outside": -0.10}},
            "6.5–7f": {"runstyle": {"E": 0.30, "E/P": 0.20, "P": -0.10, "S": -0.25},
                         "post":     {"rail": 0.15, "inner": 0.10, "mid": -0.05, "outside": -0.10}},
            "8f+":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": 0.00, "S": -0.10},
                         "post":     {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05}}
        }
    },
    "Gulfstream": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Synthetic": {
            "≤6f":    {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        }
    },
    "Tampa Bay Downs": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Belmont Park": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.15, "E/P": 0.05, "P": 0.00, "S": -0.10},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Horseshoe Indianapolis": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05}}
        }
    },
    "Penn National": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                         "post":     {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00}}
        }
    },
    "Presque Isle Downs": {
        "Synthetic": {
            "≤6f":    {"runstyle": {"E": 0.15, "E/P": 0.10, "P": 0.00, "S": -0.10},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05}}
        }
    },
    "Woodbine": {
        "Synthetic": {
            "≤6f":    {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.00}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.00}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        },
        "Turf": {
            "≤6f":    {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05}}
        }
    },
    "Evangeline Downs": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}}
        }
    },
    "Fairmount Park": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}}
        }
    },
    "Finger Lakes": {
        "Dirt": {
            "≤6f":    {"runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                         "post":     {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05}},
            "6.5–7f": {"runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                         "post":     {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}},
            "8f+":    {"runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                         "post":     {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05}}
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
        post = None # Default to None and let it become 'mid'
    if post is None:
        return "mid"
    if post == 1:       return "rail"
    if 2 <= post <= 3:  return "inner"
    if 4 <= post <= 7:  return "mid"
    return "outside"

def _style_norm(style: str) -> str:
    s = (style or "NA").upper()
    return "E/P" if s in ("EP", "E/P") else s

def _get_track_bias_delta(track_name: str, surface_type: str, distance_txt: str,
                          style: str, post_str: str) -> float:
    canon = _canonical_track(track_name)
    surf  = (surface_type or "Dirt").strip().title()
    buck  = distance_bucket(distance_txt)  # ≤6f / 6.5–7f / 8f+
    cfg   = (TRACK_BIAS_PROFILES.get(canon, {})
                                .get(surf, {})
                                .get(buck, {}))
    if not cfg:
        return 0.0
    s_norm = _style_norm(style)
    runstyle_delta = float((cfg.get("runstyle", {}) or {}).get(s_norm, 0.0))
    post_delta     = float((cfg.get("post", {}) or {}).get(_post_bucket(post_str), 0.0))
    return float(np.clip(runstyle_delta + post_delta, -1.0, 1.0))

# ===================== Core Helpers =====================

def detect_valid_race_headers(pp_text: str):
    toks = ("purse", "furlong", "mile", "clm", "allow", "stake", "pars", "post time")
    headers = []
    for m in re.finditer(r"(?mi)^\s*Race\s+(\d+)\b", pp_text or ""):
        win = (pp_text[m.end():m.end()+250] or "").lower()
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
    t = (tok or "").upper().strip()
    return "E/P" if t in ("EP", "E/P") else t

def calculate_style_strength(style: str, quirin: float) -> str:
    s = (style or "NA").upper()
    try:
        q = float(quirin)
    except Exception:
        return "Solid"
    if pd.isna(q): return "Solid"
    if s in ("E", "E/P"):
        if q >= 7: return "Strong"
        if q >= 5: return "Solid"
        if q >= 3: return "Slight"
        return "Weak"
    if s in ("P", "S"):
        if q >= 5: return "Slight"
        if q >= 3: return "Solid"
        return "Strong"
    return "Solid"

def split_into_horse_chunks(pp_text: str) -> List[tuple]:
    chunks = []
    matches = list(HORSE_HDR_RE.finditer(pp_text or ""))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(pp_text)
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
        if name in ml: continue
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
        out["sire_awd"] = float(s.group(1)); out["sire_1st"] = float(s.group(3))
    ds = re.search(r'(?mi)^\s*Dam\'s Sire:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi', block_str)
    if ds:
        out["damsire_awd"] = float(ds.group(1)); out["damsire_1st"] = float(ds.group(3))
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
        except:
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
    pattern = r'(\d{2}[A-Za-z]{3}\d{2}).*?\s+(\d{1,2})\s+(\d{1,2})\s+(\d{1,2})[ªƒ²³¨«¬©°±´‚]*\s+(\d{1,2})[ªƒ²³¨«¬©°±´‚]*\s+(\d{1,2})[ªƒ²³¨«¬©°±´‚]*\s+(\d{1,2})[ªƒ²³¨«¬©°±´‚]*'
    for m in re.finditer(pattern, block_str, re.MULTILINE):
        try:
            pos = [int(m.group(i)) for i in range(2, 8)]
            positions.append(pos)
        except:
            pass
    return positions[:5]

def calculate_trip_quality(positions: List[List[int]], field_size: int = 10) -> float:
    """SAVANT ANGLE: Trip handicapping. Returns bonus from -0.04 to +0.12"""
    bonus = 0.0
    if not positions or len(positions[0]) < 6:
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
        except:
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
    # Matches a date, track, etc., then a race type, then captures the first fig
    r"(?mi)^\s*(\d{2}[A-Za-z]{3}\d{2})\s+.*?" # Date (e.g., 23Sep23)
    r"\b(Clm|Mdn|Md Sp Wt|Alw|OC|G1|G2|G3|Stk|Hcp)\b" # A race type keyword
    r".*?\s+(\d{2,3})\s+" # The first 2-3 digit number after the type
)

def parse_speed_figures_for_block(block) -> List[int]:
    """
    Parses a horse's PP text block and extracts all main speed figures.
    """
    figs = []
    if not block:
        return figs
    
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    for m in SPEED_FIG_RE.finditer(block_str):
        try:
            # The speed figure is the third capture group
            fig_val = int(m.group(3))
            # Basic sanity check for a realistic speed figure
            if 40 < fig_val < 130:
                figs.append(fig_val)
        except Exception:
            pass # Ignore if conversion fails

    # We only care about the most recent figs, e.g., last 10
    return figs[:10]

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
            horse_name = f"Unknown_{len(names)+1}"
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

def apply_enhancements_and_figs(ratings_df: pd.DataFrame, pp_text: str, processed_weights: Dict[str,float],
                                chaos_index: float, track_name: str, surface_type: str,
                                distance_txt: str, race_type: str,
                                angles_per_horse: Dict[str,pd.DataFrame],
                                pedigree_per_horse: Dict[str,dict], figs_df: pd.DataFrame) -> pd.DataFrame:
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
    df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"].fillna(0.0) # Ensure no NaNs
    df["R"] = (df["R"].astype(float) + df["R_ENHANCE_ADJ"].astype(float)).round(2)

    return df

# ---------- Odds helpers ----------
def fair_to_american(p: float) -> float:
    if p <= 0: return math.inf
    if p >= 1: return 0.0
    dec = 1.0/p
    return round((dec-1)*100,0) if dec>=2 else round(-100/(dec-1),0)

def fair_to_american_str(p: float) -> str:
    v = fair_to_american(p)
    if math.isinf(v): return "N/A"
    return f"+{int(v)}" if v > 0 else f"{int(v)}"

def str_to_decimal_odds(s: str) -> Optional[float]:
    s = (s or "").strip()
    if not s: return None
    try:
        if re.fullmatch(r'[+-]?\d+(\.\d+)?', s):
            v = float(s); return max(v, 1.01)
        if re.fullmatch(r'\+\d+', s) or re.fullmatch(r'-\d+', s):
            return 1 + (float(s)/100.0 if float(s)>0 else 100.0/abs(float(s)))
        if "-" in s:
            a,b = s.split("-",1)
            return float(a)/float(b) + 1.0
        if "/" in s:
            a,b = s.split("/",1)
            return float(a)/float(b) + 1.0
    except Exception as e:
        st.warning(f"Could not parse odds string: '{s}'. Error: {e}")
        return None
    return None

def overlay_table(fair_probs: Dict[str,float], offered: Dict[str,float]) -> pd.DataFrame:
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
def calculate_exotics_biased(fair_probs: Dict[str,float],
                             anchor_first: Optional[str] = None,
                             anchor_second: Optional[str] = None,
                             pool_third: Optional[set] = None,
                             pool_fourth: Optional[set] = None,
                             weights=MODEL_CONFIG['exotic_bias_weights'],
                             top_n: int = 50) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame,pd.DataFrame]:

    horses = list(fair_probs.keys())
    probs = np.array([fair_probs[h] for h in horses])
    n = len(horses)
    if n < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def w_first(h):  return weights[0] if anchor_first and h == anchor_first else 1.0
    def w_second(h): return weights[1] if anchor_second and h == anchor_second else 1.0
    def w_third(h):  return weights[2] if pool_third and h in pool_third else 1.0
    def w_fourth(h): return weights[3] if pool_fourth and h in pool_fourth else 1.0
    # Add a 5th weight for SH5, re-using 4th
    def w_fifth(h):  return weights[3] if pool_fourth and h in pool_fourth else 1.0


    # EXACTA
    ex_rows = []
    for i,j in product(range(n), range(n)):
        if i == j: continue
        denom_ex = 1.0 - probs[i]
        if denom_ex <= 1e-9: continue
        prob = probs[i] * (probs[j] / denom_ex)
        prob *= w_first(horses[i]) * w_second(horses[j])
        ex_rows.append({"Ticket":f"{horses[i]} → {horses[j]}", "Prob":prob})

    ex_total = sum(r["Prob"] for r in ex_rows) or 1.0
    for r in ex_rows: r["Prob"] = r["Prob"]/ex_total
    for r in ex_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0/r["Prob"])-1
        else:
            r["Fair Odds"] = float('inf')
    df_ex = pd.DataFrame(ex_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 3:
        return df_ex, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # TRIFECTA
    tri_rows = []
    top8 = np.argsort(-probs)[:min(n,8)]
    for i,j,k in product(top8, top8, top8):
        if len({i,j,k}) != 3: continue
        denom_ij = 1.0 - probs[i]
        denom_ijk = 1.0 - probs[i] - probs[j]
        if denom_ij <= 1e-9 or denom_ijk <= 1e-9: continue

        p_ij = probs[i]*(probs[j]/denom_ij)
        prob_ijk = p_ij*(probs[k]/denom_ijk)
        prob_ijk *= w_first(horses[i]) * w_second(horses[j]) * w_third(horses[k])
        tri_rows.append({"Ticket":f"{horses[i]} → {horses[j]} → {horses[k]}", "Prob":prob_ijk})

    tri_total = sum(r["Prob"] for r in tri_rows) or 1.0
    for r in tri_rows: r["Prob"] = r["Prob"]/tri_total
    for r in tri_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0/r["Prob"])-1
        else:
            r["Fair Odds"] = float('inf')
    df_tri = pd.DataFrame(tri_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 4:
        return df_ex, df_tri, pd.DataFrame(), pd.DataFrame()

    # SUPERFECTA
    super_rows = []
    top6 = np.argsort(-probs)[:min(n,6)] # Keep this at top6 for performance
    for i,j,k,l in product(top6, top6, top6, top6):
        if len({i,j,k,l}) != 4: continue
        denom_ij = 1.0 - probs[i]
        denom_ijk = 1.0 - probs[i] - probs[j]
        denom_ijkl = 1.0 - probs[i] - probs[j] - probs[k]
        if denom_ij <= 1e-9 or denom_ijk <= 1e-9 or denom_ijkl <= 1e-9: continue

        p_ij = probs[i]*(probs[j]/denom_ij)
        p_ijk = p_ij*(probs[k]/denom_ijk)
        prob_ijkl = p_ijk*(probs[l]/denom_ijkl)
        prob_ijkl *= w_first(horses[i]) * w_second(horses[j]) * w_third(horses[k]) * w_fourth(horses[l])
        super_rows.append({"Ticket":f"{horses[i]} → {horses[j]} → {horses[k]} → {horses[l]}", "Prob":prob_ijkl})

    super_total = sum(r["Prob"] for r in super_rows) or 1.0
    for r in super_rows: r["Prob"] = r["Prob"]/super_total
    for r in super_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0/r["Prob"])-1
        else:
            r["Fair Odds"] = float('inf')
    df_super = pd.DataFrame(super_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 5:
        return df_ex, df_tri, df_super, pd.DataFrame()

    # --- NEW: SUPER HIGH 5 ---
    sh5_rows = []
    top7 = np.argsort(-probs)[:min(n,7)] # Use Top 7 for SH5
    for i,j,k,l,m in product(top7, top7, top7, top7, top7):
        if len({i,j,k,l,m}) != 5: continue
        denom_ij = 1.0 - probs[i]
        denom_ijk = 1.0 - probs[i] - probs[j]
        denom_ijkl = 1.0 - probs[i] - probs[j] - probs[k]
        denom_ijklm = 1.0 - probs[i] - probs[j] - probs[k] - probs[l]
        if denom_ij <= 1e-9 or denom_ijk <= 1e-9 or denom_ijkl <= 1e-9 or denom_ijklm <= 1e-9: continue

        p_ij = probs[i]*(probs[j]/denom_ij)
        p_ijk = p_ij*(probs[k]/denom_ijk)
        p_ijkl = p_ijk*(probs[l]/denom_ijkl)
        prob_ijklm = p_ijkl*(probs[m]/denom_ijklm)

        prob_ijklm *= (w_first(horses[i]) * w_second(horses[j]) * w_third(horses[k]) * w_fourth(horses[l]) * w_fifth(horses[m]))

        sh5_rows.append({"Ticket":f"{horses[i]} → {horses[j]} → {horses[k]} → {horses[l]} → {horses[m]}", "Prob":prob_ijklm})

    sh5_total = sum(r["Prob"] for r in sh5_rows) or 1.0
    for r in sh5_rows: r["Prob"] = r["Prob"]/sh5_total
    for r in sh5_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0/r["Prob"])-1
        else:
            r["Fair Odds"] = float('inf')
    df_super_hi_5 = pd.DataFrame(sh5_rows).sort_values(by="Prob", ascending=False).head(top_n)

    return df_ex, df_tri, df_super, df_super_hi_5


def format_exotics_for_prompt(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"**{title} (Model-Derived)**\nNone.\n"
    df = df.copy()
    if "Prob %" not in df.columns:
        df["Prob %"] = (df["Prob"]*100).round(2)
    # Format Fair Odds to handle potential infinity
    df["Fair Odds"] = df["Fair Odds"].apply(lambda x: f"{x:.2f}" if np.isfinite(x) else "in")
    md = df[["Ticket","Prob %","Fair Odds"]].to_markdown(index=False)
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
    Returns: adjustment factor (-2.0 to +0.5)
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
    elif days_since_last <= 180:  # Long layoff
        return -1.5
    else:  # Extended absence
        return -2.0

def calculate_form_trend(recent_finishes: List[int]) -> float:
    """
    Analyze finish positions for improvement/decline trend.
    Returns: trend factor (-1.5 to +1.5)
    """
    if len(recent_finishes) < 2:
        return 0.0

    # Weight recent races more heavily: [0.4, 0.3, 0.2, 0.1]
    weights = [0.4, 0.3, 0.2, 0.1][:len(recent_finishes)]

    # Calculate weighted average of recent finishes
    weighted_avg = sum(f * w for f, w in zip(recent_finishes, weights)) / sum(weights)

    # Check for improvement pattern (finishes getting better = lower numbers)
    if len(recent_finishes) >= 3:
        recent_3 = recent_finishes[:3]
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
            if ':' in time_clean:
                parts = time_clean.split(':')
                time_seconds = float(parts[0]) * 60 + float(parts[1]) if len(parts) == 2 else float(parts[1])
            else:
                time_seconds = float(time_clean)
            
            normalized_time = time_seconds * (4.0 / distance) if distance > 0 else 999
            
            work_details.append({
                'bullet': bullet,
                'distance': distance,
                'time': normalized_time,
                'grade': grade,
                'rank': rank,
                'total': total
            })
        except:
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

# ===================== Class Rating Calculator (Comprehensive) =====================

def parse_recent_class_levels(block) -> List[dict]:
    """
    Parse recent races to extract class progression data.
    Returns list of dicts with purse, race_type, finish_position
    """
    races = []
    if not block:
        return races
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    # Pattern: date track race_type purse  (e.g., "23Sep23 Bel Alw 85000")
    pattern = r'(\d{2}[A-Za-z]{3}\d{2})\s+\w+\s+(Clm|Md Sp Wt|Mdn|Alw|OC|Stk|G1|G2|G3|Hcp)\s+(\d+)'

    for match in re.finditer(pattern, block_str):
        race_type = match.group(2)
        purse_str = match.group(3)
        try:
            races.append({
                'race_type': race_type,
                'purse': int(purse_str) if purse_str.isdigit() else 0
            })
        except Exception:
            pass

    return races[:5]  # Last 5 races

def calculate_comprehensive_class_rating(
    today_purse: int,
    today_race_type: str,
    horse_block: str,
    pedigree: dict,
    angles_df: pd.DataFrame
) -> float:
    """
    Comprehensive class rating considering:
    1. Today's purse vs recent purse levels
    2. Race type hierarchy (MCL < CLM < MSW < ALW < STK < GRD)
    3. Class movement trend
    4. Pedigree quality indicators
    5. Angle-based class boosts

    Returns: Class rating from -3.0 to +6.0
    """

    # Race type hierarchy scoring
    race_type_scores = {
        'mcl': 1, 'maiden claiming': 1, 'md cl': 1,
        'clm': 2, 'claiming': 2, 'cl': 2,
        'mdn': 3, 'md sp wt': 3, 'maiden special weight': 3, 'msw': 3,
        'alw': 4, 'allowance': 4,
        'oc': 4.5, 'optional claiming': 4.5,
        'stk': 5, 'stakes': 5,
        'hcp': 5.5, 'handicap': 5.5,
        'g3': 6, 'grade 3': 6,
        'g2': 7, 'grade 2': 7,
        'g1': 8, 'grade 1': 8
    }

    today_type_norm = str(today_race_type).strip().lower()
    today_score = race_type_scores.get(today_type_norm, 3.5)

    # Parse recent races
    recent_races = parse_recent_class_levels(horse_block)

    class_rating = 0.0

    # 1. PURSE COMPARISON (weight: heavy)
    if recent_races and today_purse > 0:
        recent_purses = [r['purse'] for r in recent_races if r['purse'] > 0]
        if recent_purses:
            avg_recent_purse = np.mean(recent_purses)
            purse_ratio = today_purse / avg_recent_purse if avg_recent_purse > 0 else 1.0

            # Purse movement scoring
            if purse_ratio >= 1.5:  # Major step up
                class_rating -= 1.2
            elif purse_ratio >= 1.2:  # Moderate step up
                class_rating -= 0.6
            elif purse_ratio >= 0.8 and purse_ratio <= 1.2:  # Same class
                class_rating += 0.8
            elif purse_ratio >= 0.6:  # Slight drop
                class_rating += 1.5
            else:  # Major drop (class relief)
                class_rating += 2.5

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

    # 3. PEDIGREE QUALITY BOOST
    spi = pedigree.get('sire_spi') or pedigree.get('damsire_spi')
    if pd.notna(spi):
        spi_val = float(spi)
        if spi_val >= 110:
            class_rating += 0.4
        elif spi_val >= 100:
            class_rating += 0.2

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

st.header("1. Paste PPs & Parse")

pp_text_widget = st.text_area(
    "BRIS PPs text:",
    value=st.session_state["pp_text_cache"],
    height=300,
    key="pp_text_input",
    help="Paste the text from a BRIS Ultimate Past Performances PDF.",
    disabled=st.session_state["parsed"]
)

col_parse, col_reset = st.columns([1,1])
with col_parse:
    parse_clicked = st.button("Parse PPs", type="primary")
with col_reset:
    reset_clicked = st.button("Reset / Parse New", help="Clear parsed state to paste another race")

if reset_clicked:
    st.session_state["parsed"] = False
    st.session_state["pp_text_cache"] = ""
    _safe_rerun()

if parse_clicked:
    text_now = (st.session_state.get("pp_text_input") or "").strip()
    if not text_now:
        st.warning("Paste PPs text first.")
    else:
        # SECURITY: Validate PP text before processing
        if SECURITY_VALIDATORS_AVAILABLE:
            try:
                text_now = sanitize_pp_text(text_now)
            except ValueError as e:
                st.error(f"Invalid PP text: {e}")
                st.stop()
        
        st.session_state["pp_text_cache"] = text_now
        st.session_state["parsed"] = True
        _safe_rerun()

if not st.session_state["parsed"]:
    st.info("Paste your PPs and click **Parse PPs** to continue.")
    st.stop()

pp_text = st.session_state["pp_text_cache"]

# ===================== 2. Race Info (Confirm) =====================

st.header("2. Race Info (Confirm)")
first_line = (pp_text.split("\n",1)[0] or "").strip()

# Track
parsed_track = parse_track_name_from_pp(pp_text)
track_name = st.text_input("Track:", value=(parsed_track or st.session_state['track_name']))

# SECURITY: Validate track name if validators available
if SECURITY_VALIDATORS_AVAILABLE and track_name:
    try:
        track_name = validate_track_name(track_name)
    except ValueError as e:
        st.warning(f"Invalid track name (using default): {e}")
        track_name = "Unknown Track"

st.session_state['track_name'] = track_name

# Surface auto from header, but allow override
default_surface = st.session_state['surface_type']
if re.search(r'(?i)\bturf|trf\b', first_line): default_surface = "Turf"
if re.search(r'(?i)\baw|tap|synth|poly\b', first_line): default_surface = "Synthetic"
surface_type = st.selectbox("Surface:", ["Dirt","Turf","Synthetic"],
                            index=["Dirt","Turf","Synthetic"].index(default_surface) if default_surface in ["Dirt", "Turf", "Synthetic"] else 0)
st.session_state['surface_type'] = surface_type

# Condition
conditions = ["fast","good","wet-fast","muddy","sloppy","firm","yielding","soft","heavy"]
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
    if m: return m.group(1).title().replace("1/2","½")
    if re.search(r'(?i)\b8\s*1/2\s*furlongs?\b', s): return "8 1/2 Furlongs"
    if re.search(r'(?i)\b8\s*furlongs?\b', s): return "8 Furlongs"
    if re.search(r'(?i)\b9\s*furlongs?\b', s): return "9 Furlongs"
    if re.search(r'(?i)\b1\s*mile\b', s): return "1 Mile"
    if re.search(r'(?i)\b1\s*1/16\b', s): return "1 1/16 Miles"
    if re.search(r'(?i)\b7\s*furlongs?\b', s): return "7 Furlongs"
    return "6 Furlongs"

auto_distance = _auto_distance_label(first_line)
# try to map to option variants
preferred = (auto_distance or "").replace("½","1/2").replace(" 1/2"," 1/2")
idx = DISTANCE_OPTIONS.index("6 Furlongs") if "6 Furlongs" in DISTANCE_OPTIONS else 0
for opt in (preferred, auto_distance, st.session_state['distance_txt']):
    if opt in DISTANCE_OPTIONS:
        idx = DISTANCE_OPTIONS.index(opt); break
distance_txt = st.selectbox("Distance:", DISTANCE_OPTIONS, index=idx)
st.session_state['distance_txt'] = distance_txt

# Purse
def detect_purse_amount(pp_text: str) -> Optional[int]:
    s = pp_text or ""
    m = re.search(r'(?mi)\bPurse\b[^$\n\r]*\$\s*([\d,]+)', s)
    if m:
        try: return int(m.group(1).replace(",", ""))
        except: pass
    m = re.search(r'(?mi)\b(?:Added|Value)\b[^$\n\r]*\$\s*([\d,]+)', s)
    if m:
        try: return int(m.group(1).replace(",", ""))
        except: pass
    m = re.search(r'(?mi)\b(Mdn|Maiden|Allowance|Alw|Claiming|Clm|Starter|Stake|Stakes)\b[^:\n\r]{0,50}\b(\d{2,4})\s*[Kk]\b', s)
    if m:
        try: return int(m.group(2)) * 1000
        except: pass
    m = re.search(r'(?m)\$\s*([\d,]{5,})', s)
    if m:
        try: return int(m.group(1).replace(",", ""))
        except: pass
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
    race_type_index = list(base_class_bias.keys()).index(race_type_detected) if race_type_detected in base_class_bias else list(base_class_bias.keys()).index("allowance")
except ValueError:
     st.warning("Default race type 'allowance' not found in bias definitions. Using first available type.")
     race_type_index = 0 # Default to the first item if 'allowance' isn't found

race_type_manual = st.selectbox(
    "Race Type (override):",
    options=list(base_class_bias.keys()),
    index=race_type_index,
)
race_type = race_type_manual or race_type_detected
race_type_detected = race_type  # lock in constant key
st.session_state['race_type'] = race_type_detected  # Store for Classic Report

# ===================== A. Race Setup: Scratches, ML & Styles =====================

st.header("A. Race Setup: Scratches, ML & Live Odds, Styles")

df_styles = extract_horses_and_styles(pp_text)
if df_styles.empty:
    st.error("No horses found. Check your PP text paste.")
    st.stop()

ml_map_raw = extract_morning_line_by_horse(pp_text)
df_styles["ML"] = df_styles["Horse"].map(lambda h: ml_map_raw.get(h,"")) if "Horse" in df_styles else ""
df_styles["Live Odds"] = df_styles["ML"].where(df_styles["ML"].astype(str).str.len()>0,"")
df_styles["Scratched"] = False

col_cfg = {
    "Post": st.column_config.TextColumn("Post", width="small", disabled=True),
    "Horse": st.column_config.TextColumn("Horse", width="medium", disabled=True),
    "ML": st.column_config.TextColumn("ML", width="small", help="Parsed Morning Line from PPs"),
    "Live Odds": st.column_config.TextColumn("Live Odds", width="small", help="Enter current odds (e.g., '5/2', '3.5', '+250')"),
    "DetectedStyle": st.column_config.TextColumn("BRIS Style", width="small", disabled=True),
    "Quirin": st.column_config.NumberColumn("Quirin", width="small", disabled=True),
    "AutoStrength": st.column_config.TextColumn("Auto-Strength", width="small", disabled=True),
    "OverrideStyle": st.column_config.SelectboxColumn("Override Style", width="small", options=["", "E", "E/P", "P", "S"]),
    "Scratched": st.column_config.CheckboxColumn("Scratched?", width="small")
}
df_editor = st.data_editor(df_styles, use_container_width=True, column_config=col_cfg)

# ===================== B. Angle Parsing / Pedigree / Figs =====================

angles_per_horse: Dict[str, pd.DataFrame] = {}
pedigree_per_horse: Dict[str, dict] = {}
figs_per_horse: Dict[str, List[int]] = {}

# Try to use elite parser first for better accuracy
elite_parser_used = False
if ELITE_PARSER_AVAILABLE and len(pp_text.strip()) > 100:
    try:
        from elite_parser import GoldStandardBRISNETParser
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
    if fig_list: # Only add horses that have figures
        figs_data.append({
            "Horse": name,
            "Figures": fig_list, # The list of parsed figs
            "BestFig": max(fig_list),
            "AvgTop2": round(np.mean(sorted(fig_list, reverse=True)[:2]), 1)
        })
figs_df = pd.DataFrame(figs_data) # <--- THIS IS THE NEW FIGS DATAFRAME

df_final_field = df_editor[df_editor["Scratched"]==False].copy()
if df_final_field.empty:
    st.warning("All horses are scratched.")
    st.stop()

# Store in session state for later access
st.session_state['df_final_field'] = df_final_field

# Ensure StyleStrength and Style exist
df_final_field["StyleStrength"] = df_final_field.apply(
    lambda row: calculate_style_strength(row["OverrideStyle"] if row["OverrideStyle"] else row["DetectedStyle"], row["Quirin"]), axis=1
)
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
    cats = " ".join(ang_df["Category"].astype(str).tolist()).lower() if (ang_df is not None and not ang_df.empty) else ""
    if "dirt to tur" in cats: return "Tur"
    if "turf to dirt" in cats: return "Dirt"
    # If nothing clear, use race surface (neutral) to avoid over-penalizing
    return race_surface

def _infer_horse_distance_pref(ped: dict) -> str:
    awds = [x for x in [ped.get("sire_awd"), ped.get("damsire_awd")] if pd.notna(x)]
    if not awds:
        return "any"
    m = float(np.nanmean(awds))
    if m <= 6.5: return "≤6"
    if m >= 7.5: return "8f+"
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
    awd_mean = float(np.nanmean(awds)) if awds else np.nan
    if awd_mean == awd_mean: # Check if not NaN
        if race_bucket == "≤6f":
            if awd_mean <= 6.5: tweak += MODEL_CONFIG['ped_dist_bonus']
            elif awd_mean >= 7.5: tweak += MODEL_CONFIG['ped_dist_penalty']
        elif race_bucket == "8f+":
            if awd_mean >= 7.5: tweak += MODEL_CONFIG['ped_dist_bonus']
            elif awd_mean <= 6.5: tweak += MODEL_CONFIG['ped_dist_penalty']
        else: # 6.5-7f bucket
            if 6.3 <= awd_mean <= 7.7: tweak += MODEL_CONFIG['ped_dist_neutral_bonus']

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
            if race_bucket == "≤6f": tweak += MODEL_CONFIG['angle_debut_sprint_bonus']
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
                tweak -= min(MODEL_CONFIG['angle_roi_neg_max_penalty'], MODEL_CONFIG['angle_roi_neg_per_penalty'] * neg_ct)
        except Exception as e:
            st.warning(f"Error during angle ROI calculation for horse '{name}'. Error: {e}")
            pass

    # 3) Condition nuance
    if race_cond in {"muddy","sloppy","heavy"} and awd_mean == awd_mean: # Check if not NaN
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
    ped  = pedigree_per_horse.get(name, {}) or {}
    ang  = angles_per_horse.get(name)

    # Get horse's PP block for analysis
    horse_block = ""
    for _, h_name, block in split_into_horse_chunks(pp_text):
        if h_name == name:
            horse_block = block
            break

    # Calculate comprehensive class rating
    comprehensive_class = calculate_comprehensive_class_rating(
        today_purse=purse_val,
        today_race_type=race_type_detected,
        horse_block=horse_block,
        pedigree=ped,
        angles_df=ang if ang is not None else pd.DataFrame()
    )

    # Add pedigree/angle tweaks on top of class
    tweak = _angles_pedigree_tweak(name, race_surface, race_bucket, race_cond)
    cclass_total = comprehensive_class + tweak

    # Calculate form cycle rating (includes first-time starter evaluation)
    form_rating = calculate_form_cycle_rating(
        horse_block=horse_block,
        pedigree=ped,
        angles_df=ang if ang is not None else pd.DataFrame()
    )

    Cclass_vals.append(round(cclass_total, 3))
    Cform_vals.append(round(form_rating, 3))

df_final_field["Cclass"] = Cclass_vals
df_final_field["Cform"] = Cform_vals

# ===================== B. Bias-Adjusted Ratings =====================

st.header("B. Bias-Adjusted Ratings")
b_col1, b_col2, b_col3 = st.columns(3)
with b_col1:
    strategy_profile = st.selectbox(
        "Select Strategy Profile:",
        options=["Confident","Value Hunter"],
        index=0, key="strategy_profile"
    )
with b_col2:
    running_style_biases = st.multiselect(
        "Select Running Style Biases:",
        options=["E","E/P","P","S"],
        default=["E"], key="style_biases"
    )
with b_col3:
    post_biases = st.multiselect(
        "Select Post Position Biases:",
        options=["no significant post bias", "favors rail (1)", "favors inner (1-3)", "favors mid (4-7)", "favors outside (8+)"],
        default=["no significant post bias"],
        key="post_biases"
    )

if not running_style_biases or not post_biases:
    st.info("Pick at least one **Style** bias and one **Post** bias.")
    st.stop()

def _style_bias_label_from_choice(choice: str) -> str:
    # Map single-letter selection into our style_match table buckets
    up = (choice or "").upper()
    if up in ("E", "E/P"): return "speed favoring"
    if up in ("P", "S"):   return "closer favoring"
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

    if stl in ("E","E/P") and pd.notna(q) and q >= MODEL_CONFIG['style_quirin_threshold']:
        base += MODEL_CONFIG['style_quirin_bonus']
    return float(np.clip(base, -1.0, 1.0))

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

# ======================== End Phase 1 Functions ========================

def post_bias_score(post_bias_pick: str, post_str: str) -> float:
    pick = (post_bias_pick or "").strip().lower()
    try:
        post = int(re.sub(r"[^\d]", "", str(post_str)))
    except Exception as e:
        st.warning(f"Failed to parse post number: '{post_str}'. Error: {e}")
        post = None

    table = {
        "favors rail (1)": lambda p: MODEL_CONFIG['post_bias_rail_bonus'] if p == 1 else 0.0,
        "favors inner (1-3)": lambda p: MODEL_CONFIG['post_bias_inner_bonus'] if p and 1 <= p <= 3 else 0.0,
        "favors mid (4-7)": lambda p: MODEL_CONFIG['post_bias_mid_bonus'] if p and 4 <= p <= 7 else 0.0,
        "favors outside (8+)": lambda p: MODEL_CONFIG['post_bias_outside_bonus'] if p and p >= 8 else 0.0,
        "no significant post bias": lambda p: 0.0
    }
    fn = table.get(pick, table["no significant post bias"])
    return float(np.clip(fn(post), -0.5, 0.5))

def compute_bias_ratings(df_styles: pd.DataFrame,
                         surface_type: str,
                         distance_txt: str,
                         condition_txt: str,
                         race_type: str,
                         running_style_bias: str,
                         post_bias_pick: str,
                         ppi_value: float = 0.0, # ppi_value arg seems unused, ppi_map is recalculated
                         pedigree_per_horse: Optional[Dict[str,dict]] = None,
                         track_name: str = "",
                         pp_text: str = "",
                         figs_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Reads 'Cclass' and 'Cform' from df_styles (pre-built), adds Cstyle/Cpost/Cpace/Cspeed (+Atrack),
    plus Tier 2 bonuses (Impact Values, SPI, Surface Stats, AWD),
    sums to Arace and R. Returns rating table.

    ULTRATHINK V2: Can use unified rating engine if available and PP text provided.
    """
    cols = ["#", "Post", "Horse", "Style", "Quirin", "Cstyle", "Cpost", "Cpace", "Cspeed", "Cclass", "Cform", "Atrack", "Arace", "R"]
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

            if avg_confidence >= 0.6 and validation.get('horses_parsed', 0) > 0:
                # High quality parse - use unified engine
                engine = UnifiedRatingEngine(softmax_tau=3.0)

                # Extract purse value
                import re
                purse_match = re.search(r'\$(\d+(?:,\d+)*)', pp_text[:500])
                today_purse = int(purse_match.group(1).replace(',', '')) if purse_match else 20000

                # Get predictions
                results_df = engine.predict_race(
                    pp_text=pp_text,
                    today_purse=today_purse,
                    today_race_type=race_type,
                    track_name=track_name,
                    surface_type=surface_type,
                    distance_txt=distance_txt
                )

                if not results_df.empty:
                    # Filter to only include horses that are NOT scratched (i.e., in df_styles)
                    horses_in_field = set(df_styles['Horse'].tolist())
                    results_df_filtered = results_df[results_df['Horse'].isin(horses_in_field)].copy()
                    
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
        df_styles["Cclass"] = 0.0 # Default Cclass if missing
    if "Cform" not in df_styles.columns:
        df_styles = df_styles.copy()
        df_styles["Cform"] = 0.0 # Default Cform if missing

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
    mapped_bias = _style_bias_label_from_choice(running_style_bias)
    for _, row in df_styles.iterrows():
        post = str(row.get("Post", row.get("#", "")))
        name = str(row.get("Horse"))
        style = _style_norm(row.get("Style") or row.get("OverrideStyle") or row.get("DetectedStyle"))
        quirin = row.get("Quirin", np.nan) # Keep as potential NaN

        cstyle = style_match_score(mapped_bias, style, quirin) # Pass potential NaN
        cpost  = post_bias_score(post_bias_pick, post)
        cpace  = float(ppi_map.get(name, 0.0))
        cspeed = float(speed_map.get(name, 0.0))  # Speed component from figures

        a_track = _get_track_bias_delta(track_name, surface_type, distance_txt, style, post)

        # Get pre-computed Cclass and Cform from df_styles (calculated in Section A)
        c_class = float(row.get("Cclass", 0.0))
        c_form = float(row.get("Cform", 0.0))

        # ======================== Tier 2 Bonuses ========================
        tier2_bonus = 0.0

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

        # ======================== End Tier 2 Bonuses ========================

        # Apply component weights: Class×2.5, Form×1.8, Speed×2.0, Pace×1.5, Style×1.2, Post×0.8
        weighted_components = (
            c_class * 2.5 +
            c_form * 1.8 +
            cspeed * 2.0 +
            cpace * 1.5 +
            cstyle * 1.2 +
            cpost * 0.8
        )
        arace = weighted_components + a_track + tier2_bonus
        R     = arace

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


def fair_probs_from_ratings(ratings_df: pd.DataFrame) -> Dict[str, float]:
    """
    GOLD STANDARD probability calculation with comprehensive validation.
    
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
    
    # SAFETY: Work on copy to avoid side effects
    df = ratings_df.copy()
    
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
    
    return result


# Build scenarios
scenarios = [(s, p) for s in running_style_biases for p in post_biases]
tabs = st.tabs([f"S: {s} | P: {p}" for s,p in scenarios])
all_scenario_ratings = {}

# Simple weight presets (placeholders retained)
def get_weight_preset(surface, distance): return {"class_form":1.0, "trs_jky":1.0}
def apply_strategy_profile_to_weights(w, profile): return w
def adjust_by_race_type(w, rt): return w
def apply_purse_scaling(w, purse): return w if w else {}

base_weights = get_weight_preset(surface_type, distance_txt)
profiled_weights = apply_strategy_profile_to_weights(base_weights, strategy_profile)
racetype_weights = adjust_by_race_type(profiled_weights, race_type_detected)
final_weights = apply_purse_scaling(racetype_weights, purse_val)

for i, (rbias, pbias) in enumerate(scenarios):
    with tabs[i]:
        ratings_df = compute_bias_ratings(
            df_styles=df_final_field.copy(), # Pass a copy to avoid modifying original
            surface_type=surface_type,
            distance_txt=distance_txt,
            condition_txt=condition_txt,
            race_type=race_type_detected,
            running_style_bias=rbias,
            post_bias_pick=pbias,
            # ppi_value=ppi_val, # Removed as it's recalculated inside
            pedigree_per_horse=pedigree_per_horse,
            track_name=track_name,
            pp_text=pp_text,
            figs_df=figs_df  # Pass speed figures
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
            figs_df=figs_df # <--- PASS THE REAL FIGS_DF
        )
        fair_probs = fair_probs_from_ratings(ratings_df)
        if 'Horse' in ratings_df.columns:
            ratings_df["Fair %"] = ratings_df["Horse"].map(lambda h: f"{fair_probs.get(h,0)*100:.1f}%")
            ratings_df["Fair Odds"] = ratings_df["Horse"].map(lambda h: fair_to_american_str(fair_probs.get(h,0)))
        else:
            ratings_df["Fair %"] = ""
            ratings_df["Fair Odds"] = ""
        all_scenario_ratings[(rbias,pbias)] = (ratings_df.copy(), fair_probs) # Store copy and probs

        disp = ratings_df.sort_values(by="R", ascending=False)
        if "R_ENHANCE_ADJ" in disp.columns:
            disp = disp.drop(columns=["R_ENHANCE_ADJ"])
        st.dataframe(
            disp,
            use_container_width=True, hide_index=True,
            column_config={
                "R": st.column_config.NumberColumn("Rating", format="%.2"),
                "Cstyle": st.column_config.NumberColumn("C-Style", format="%.2"),
                "Cpost": st.column_config.NumberColumn("C-Post", format="%.2"),
                "Cpace": st.column_config.NumberColumn("C-Pace", format="%.2"),
                "Cspeed": st.column_config.NumberColumn("C-Speed", format="%.2"),
                "Cclass": st.column_config.NumberColumn("C-Class", format="%.2"),
                "Cform": st.column_config.NumberColumn("C-Form", format="%.2", help="Form cycle: layoff, trend, consistency"),
                "Atrack": st.column_config.NumberColumn("A-Track", format="%.2"),
                "Arace": st.column_config.NumberColumn("A-Race", format="%.2"),
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

        st.info(f"**Primary Scenario:** S: `{primary_key[0]}` • P: `{primary_key[1]}` • Profile: `{strategy_profile}`  • PPI: {ppi_val:+.2f}")
    else:
        st.error("Primary scenario ratings not found. Check calculations.")
        primary_df, primary_probs = pd.DataFrame(), {} # Assign defaults
        st.stop()
else:
    st.error("No scenarios generated. Check bias selections.")
    primary_df, primary_probs = pd.DataFrame(), {} # Assign defaults
    st.stop()


# ===================== C. Overlays & Betting Strategy =====================

st.header("C. Overlays Table")

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

def build_betting_strategy(primary_df: pd.DataFrame, df_ol: pd.DataFrame,
                           strategy_profile: str, name_to_post: Dict[str, str],
                           name_to_ml: Dict[str, str], field_size: int, ppi_val: float) -> str:
    """
    Builds a clearer, simplified betting strategy report using A/B/C/D grouping,
    minimum base bet examples, field size logic, and specific bet types (straight/box).
    """

    # --- 1. Helper Functions ---
    def format_horse_list(horse_names: List[str]) -> str:
        """Creates a bulleted list of horses with post, name, and ML."""
        if not horse_names:
            return "* None"
        lines = []
        # Sort horses by post number before displaying
        sorted_horses = sorted(horse_names, key=lambda name: int(name_to_post.get(name, '999')))
        for name in sorted_horses:
            post = name_to_post.get(name, '??')
            ml = name_to_ml.get(name, 'N/A')
            lines.append(f"* **#{post} - {name}** (ML: {ml})")
        return "\n".join(lines)

    def get_bet_cost(base: float, num_combos: int) -> str:
        """Calculates and formats simple bet cost."""
        cost = base * num_combos
        base_str = f"${base:.2f}"
        return f"{num_combos} combos = **${cost:.2f}** (at {base_str} base)"

    def get_box_combos(num_horses: int, box_size: int) -> int:
        """Calculates combinations for a box bet."""
        import math
        if num_horses < box_size: return 0
        return math.factorial(num_horses) // math.factorial(num_horses - box_size)

    # === FIX 1: Defined the missing helper function ===
    def get_min_cost_str(base: float, *legs) -> str:
        """Calculates part-wheel combos and cost from leg sizes."""
        final_combos = 1
        leg_counts = [l for l in legs if l > 0] # Filter out 0s
        if not leg_counts:
            final_combos = 0
        else:
            for l in leg_counts:
                final_combos *= l

        cost = base * final_combos
        base_str = f"${base:.2f}"
        return f"{final_combos} combos = **${cost:.2f}** (at {base_str} base)"
    # ===================================================

    # --- 2. A/B/C/D Grouping Logic (Simplified) ---
    A_group, B_group, C_group, D_group = [], [], [], []

    all_horses = primary_df['Horse'].tolist()
    pos_ev_horses = set(df_ol[df_ol["EV per $1"] > 0.05]['Horse'].tolist()) if not df_ol.empty else set()

    if strategy_profile == "Confident":
        A_group = all_horses[:1]
        if len(all_horses) > 1 and primary_df.iloc[1]['R'] > (primary_df.iloc[0]['R'] * 0.90): # Only add #2 if very close
             A_group.append(all_horses[1])
    else: # Value Hunter - Prioritize top pick + overlays
        A_group = list(set([all_horses[0]]) | pos_ev_horses)
        if len(A_group) > 4: # Cap A group size for Value Hunter
             A_group = sorted(A_group, key=lambda h: primary_df[primary_df['Horse'] == h].index[0])[:4] # Keep top 4 ranked from the value pool

    B_group = [h for h in all_horses if h not in A_group][:3]
    C_group = [h for h in all_horses if h not in A_group and h not in B_group][:4]
    D_group = [h for h in all_horses if h not in A_group and h not in B_group and h not in C_group]

    nA, nB, nC, nD = len(A_group), len(B_group), len(C_group), len(D_group)
    nAll = field_size # Total runners

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
    top_ml_str = name_to_ml.get(top_rated_horse, '100')
    top_ml_dec = str_to_decimal_odds(top_ml_str) or 101
    is_underlay = not is_overlay and (primary_probs.get(top_rated_horse, 0) > (1 / top_ml_dec)) and top_ml_dec < 4 # Define underlay as < 3/1

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
        else: # Confident
             blueprint_report += "* **Win Bet:** Focus on top **A-Group** horse(s).\n"

        # Exacta Examples
        blueprint_report += f"* **Exacta Part-Wheel:** `A / B,C` ({nA}x{nB+nC}) - {get_min_cost_str(1.00, nA, nB+nC)}\n"
        if nA >= 2:
            ex_box_combos = get_box_combos(nA, 2)
            blueprint_report += f"* **Exacta Box (A-Group):** `{', '.join(map(str,[int(name_to_post.get(h,'0')) for h in A_group]))}` BOX - {get_bet_cost(1.00, ex_box_combos)}\n"

        # Trifecta Examples
        blueprint_report += f"* **Trifecta Part-Wheel:** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        if nA >= 3:
            tri_box_combos = get_box_combos(nA, 3)
            blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str,[int(name_to_post.get(h,'0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"

        blueprint_report += "_Structure: Use A-horses on top, spread underneath with B and C._\n"

    else: # Standard logic for fields > 6
        # --- Confident Blueprint ---
        blueprint_report += "\n#### Confident Profile Plan\n"
        blueprint_report += "_Focus: Key A-Group horses ON TOP._\n"
        blueprint_report += "* **Win Bet:** Focus on top **A-Group** horse(s).\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B` ({nA}x{nB}) - {get_min_cost_str(1.00, nA, nB)}\n"
        if nA >=1 and nB >= 1 and nC >=1: # Check if groups have members for straight example
             blueprint_report += "* **Straight Trifecta:** `Top A / Top B / Top C` (1 combo) - Consider at higher base (e.g., $1 or $2).\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B / C / D` ({nA}x{nB}x{nC}x{nD}) - {get_min_cost_str(0.10, nA, nB, nC, nD)}\n"
        if field_size >= 7: # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B / C / D / ALL` ({nA}x{nB}x{nC}x{nD}x{nAll}) - {get_min_cost_str(0.10, nA, nB, nC, nD, nAll)}\n"

        # --- Value-Hunter Blueprint ---
        blueprint_report += "\n#### Value-Hunter Profile Plan\n"
        blueprint_report += "_Focus: Use A-Group (includes overlays) ON TOP, spread wider underneath._\n"
        blueprint_report += "* **Win Bets:** Consider betting all **A-Group** horses.\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B,C` ({nA}x{nB+nC}) - {get_min_cost_str(1.00, nA, nB + nC)}\n"
        if nA >= 3: # Example box if A group is large enough
             tri_box_combos = get_box_combos(nA, 3)
             blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str,[int(name_to_post.get(h,'0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B,C / B,C,D` ({nA}x{nB+nC}x{nB+nC+nD}) - {get_min_cost_str(0.50, nA, nB + nC, nB + nC + nD)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B,C / B,C,D / ALL` ({nA}x{nB+nC}x{nB+nC+nD}x{nAll}) - {get_min_cost_str(0.10, nA, nB + nC, nB + nC + nD, nAll)}\n"
        if field_size >= 7: # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B,C / B,C,D / ALL / ALL` ({nA}x{nB+nC}x{nB+nC+nD}x{nAll}x{nAll}) - {get_min_cost_str(0.10, nA, nB + nC, nB + nC + nD, nAll, nAll)}\n"

    # --- 6. Build Final Report String ---
    final_report = """
{pace_report}
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


st.header("D. Classic Report")

# Only show button if race has been parsed
if not st.session_state.get("parsed", False):
    st.warning("⚠️ Please parse a race first in Section A before analyzing.")
elif 'primary_d' not in st.session_state or 'primary_probs' not in st.session_state:
    st.error("❌ Rating data not available. Please ensure Section C completed successfully.")
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
                st.download_button("⬇️ Download Full Analysis (.txt)", data=analysis_bytes, file_name="analysis.txt", mime="text/plain", key="dl_analysis_persistent")
            with col2:
                st.download_button("⬇️ Download Overlays (CSV)", data=overlays_bytes, file_name="overlays.csv", mime="text/csv", key="dl_overlays_persistent")
            with col3:
                # Create strategy_report_md from session state if available
                if 'strategy_report' in st.session_state:
                    tickets_bytes = st.session_state['strategy_report'].encode("utf-8")
                    st.download_button("⬇️ Download Strategy Detail (.txt)", data=tickets_bytes, file_name="strategy_detail.txt", mime="text/plain", key="dl_strategy_persistent")
        
        st.info("💡 To generate a new report, click 'Analyze This Race' again below.")
    
    if st.button("Analyze This Race", type="primary", key="analyze_button"):
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
                    name_to_ml = pd.Series(
                        df_final_field["ML"].values,
                        index=df_final_field["Horse"]
                    ).to_dict()
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

                top_table = primary_sorted[['Horse','R','Fair %','Fair Odds']].head(5).to_markdown(index=False)

                overlay_pos = df_ol[df_ol["EV per $1"] > 0] if not df_ol.empty else pd.DataFrame()
                overlay_table_md = (overlay_pos[['Horse','Fair %','Fair (AM)','Board (dec)','EV per $1']].to_markdown(index=False)
                                    if not overlay_pos.empty else "None.")

                # --- 2. NEW: Generate Simplified A/B/C/D Strategy Report ---
                strategy_report_md = build_betting_strategy(
                    primary_df, df_ol, strategy_profile, name_to_post, name_to_ml, field_size, ppi_val
                )
                
                # Store strategy report in session state
                st.session_state['strategy_report'] = strategy_report_md

                # --- 3. Update the LLM Prompt ---
                prompt = f"""
Act as a professional horse racing analyst writing a clear, concise, and actionable betting report suitable for handicappers of all levels.

--- RACE CONTEXT ---
- Track: {track_name}
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

--- TASK: WRITE CLASSIC REPORT (Simplified & Clear) ---
Your goal is to present the information from the "FULL ANALYSIS & BETTING PLAN" section clearly.
- **Race Summary:** 6-8 sentences about the race conditions.
- **Pace Projection:** Use the "Pace Projection" section provided. Explain briefly what it means for different running styles.
- **Contender Analysis:** - Summarize the **A-Group** (Key Win Contenders) and **B-Group** (Primary Challengers). Use their names and post numbers. Briefly explain *why* they are contenders (e.g., "Top rated," "Good value overlay," "Logical threat").
    - Mention the simple **Value Note** about the top-rated horse if provided.
    - Keep this section focused on the top ~4 contenders overall (A + top B).
- **Betting Strategy:**
    - Clearly state the selected **Strategy Profile** ({strategy_profile}).
    - Present the **A/B/C/D Contender Groups** exactly as listed (with names, posts, MLs).
    - Present the **Betting Strategy Blueprints** for the selected profile ({strategy_profile}). Show the example ticket structures (e.g., "Trifecta: A / B / C", "Exacta Box: A-Group") and their calculated minimum costs.
    - IMPORTANT: Emphasize that these are *blueprints* and the user should **scale the base bet amounts** ($0.10, $0.50, etc.) to fit their own budget per race (mentioning ~$100 max recommended).
    - Include the final **Bankroll & Strategy Notes**.
- **Tone:** Be informative, direct, and easy to understand. Avoid overly complex jargon. Use horse names and post numbers (#) frequently.
"""
                report = call_openai_messages(messages=[{"role":"user","content":prompt}])
                
                # Store Classic Report in session state so it persists across reruns
                st.session_state['classic_report'] = report
                st.session_state['classic_report_generated'] = True
                
                st.markdown(report)

                # ---- Save to disk (optional) ----
                report_str = report if isinstance(report, str) else str(report)
                with open("analysis.txt","w", encoding="utf-8", errors="replace") as f:
                    f.write(report_str)

                if isinstance(df_ol, pd.DataFrame):
                    df_ol.to_csv("overlays.csv", index=False, encoding="utf-8-sig")
                else:
                    pd.DataFrame().to_csv("overlays.csv", index=False, encoding="utf-8-sig")

                # --- Create a tickets.txt from the strategy report ---
                with open("tickets.txt","w", encoding="utf-8", errors="replace") as f:
                    f.write(strategy_report_md) # Save the raw strategy markdown
                tickets_bytes = strategy_report_md.encode("utf-8")

                # ---- Download buttons (browser) ----
                analysis_bytes = report_str.encode("utf-8")
                overlays_bytes = df_ol.to_csv(index=False).encode("utf-8-sig") if isinstance(df_ol, pd.DataFrame) else b""

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button("⬇️ Download Full Analysis (.txt)", data=analysis_bytes, file_name="analysis.txt", mime="text/plain", key="dl_analysis_new")
                with col2:
                    st.download_button("⬇️ Download Overlays (CSV)", data=overlays_bytes, file_name="overlays.csv", mime="text/csv", key="dl_overlays_new")
                with col3:
                    st.download_button("⬇️ Download Strategy Detail (.txt)", data=tickets_bytes, file_name="strategy_detail.txt", mime="text/plain", key="dl_strategy_new")

                # ============================================================
                # AUTO-SAVE TO GOLD HIGH-IQ DATABASE
                # ============================================================
                if GOLD_DB_AVAILABLE and gold_db is not None and primary_df is not None:
                    try:
                        # Helper function to safely convert percentage strings and odds
                        def safe_float(value, default=0.0):
                            """
                            Convert value to float, handling:
                            - Percentage strings like '75.6%' 
                            - American odds like '+150', '-200'
                            - Regular numbers
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
                        
                        # Generate race ID
                        race_date = datetime.now().strftime('%Y%m%d')
                        race_id = f"{track_name}_{race_date}_R{st.session_state.get('race_num', 1)}"
                        
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
                            horse_name = str(row.get('Horse', f'Horse_{idx+1}'))
                            
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
                                'morning_line_odds': safe_float(name_to_ml.get(horse_name, '99'), 99.0),
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
                            st.success(f"💾 **Auto-saved to gold database:** {race_id}")
                            st.session_state['last_saved_race_id'] = race_id
                            st.info("🏁 After race completes, submit actual top 5 finishers in **Section E** below!")
                        
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
    try:
        # Get stats from gold database
        stats = gold_db.get_accuracy_stats()
        pending_races = gold_db.get_pending_races(limit=20)
        
        # Create tabs for Gold High-IQ System
        tab_overview, tab_results, tab_retrain = st.tabs(
            ["📊 Dashboard", "🏁 Submit Actual Top 5", "🚀 Retrain Model"]
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
            """)
            
            # Calculate total analyzed races (completed + pending)
            completed_races = stats.get('total_races', 0)
            total_analyzed = completed_races + len(pending_races)
            
            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Total Analyzed", total_analyzed, help="All races saved (completed + pending results)")
            with col2:
                st.metric("With Results", completed_races, help="Races with actual results entered - used for accuracy tracking")
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
                        display_df['predicted_probability'] = (display_df['predicted_probability'] * 100).round(1).astype(str) + '%'
                        display_df['fair_odds'] = display_df['fair_odds'].round(2)
                        # Now rename columns for display
                        display_df.columns = ['#', 'Horse Name', 'Post', 'Predicted Win %', 'Fair Odds']
                        
                        st.dataframe(display_df, use_container_width=True, hide_index=True)
                        
                        # Enter top 5
                        st.markdown("---")
                        st.markdown("### 🏆 Enter Actual Top 5 Finishers")
                        st.caption("Select the program numbers that finished 1st through 5th")
                        
                        # Create clean input grid
                        col1, col2, col3, col4, col5 = st.columns(5)
                        
                        program_numbers = [h['program_number'] for h in horses]
                        horse_names_dict = {h['program_number']: h['horse_name'] for h in horses}
                        
                        with col1:
                            st.markdown("**🥇 1st Place**")
                            pos1 = st.selectbox(
                                "Winner",
                                program_numbers,
                                key=f"pos1_{race_id}",
                                format_func=lambda x: f"#{x} - {horse_names_dict[x][:20]}"
                            )
                        
                        with col2:
                            st.markdown("**🥈 2nd Place**")
                            pos2 = st.selectbox(
                                "Second",
                                program_numbers,
                                key=f"pos2_{race_id}",
                                index=min(1, len(program_numbers)-1),
                                format_func=lambda x: f"#{x} - {horse_names_dict[x][:20]}"
                            )
                        
                        with col3:
                            st.markdown("**🥉 3rd Place**")
                            pos3 = st.selectbox(
                                "Third",
                                program_numbers,
                                key=f"pos3_{race_id}",
                                index=min(2, len(program_numbers)-1),
                                format_func=lambda x: f"#{x} - {horse_names_dict[x][:20]}"
                            )
                        
                        with col4:
                            st.markdown("**4th Place**")
                            pos4 = st.selectbox(
                                "Fourth",
                                program_numbers,
                                key=f"pos4_{race_id}",
                                index=min(3, len(program_numbers)-1),
                                format_func=lambda x: f"#{x} - {horse_names_dict[x][:20]}"
                            )
                        
                        with col5:
                            st.markdown("**5th Place**")
                            pos5 = st.selectbox(
                                "Fifth",
                                program_numbers,
                                key=f"pos5_{race_id}",
                                index=min(4, len(program_numbers)-1),
                                format_func=lambda x: f"#{x} - {horse_names_dict[x][:20]}"
                            )
                        
                        # Validation and submit
                        finish_order = [pos1, pos2, pos3, pos4, pos5]
                        
                        # Show preview
                        st.markdown("---")
                        st.markdown("**Preview:**")
                        preview_parts = []
                        for i, pos in enumerate(finish_order):
                            if i == 0:
                                preview_parts.append(f"🥇 {horse_names_dict[pos]}")
                            elif i == 1:
                                preview_parts.append(f"🥈 {horse_names_dict[pos]}")
                            elif i == 2:
                                preview_parts.append(f"🥉 {horse_names_dict[pos]}")
                            elif i == 3:
                                preview_parts.append(f"4th {horse_names_dict[pos]}")
                            else:
                                preview_parts.append(f"5th {horse_names_dict[pos]}")
                        preview_text = " → ".join(preview_parts)
                        st.info(preview_text)
                        
                        # Validation check
                        if len(set(finish_order)) != 5:
                            st.error("❌ Each position must be unique! Please select 5 different horses.")
                        
                        # Submit button (always show, but validate before submitting)
                        if st.button("✅ Submit Top 5 Results", type="primary", key=f"submit_{race_id}"):
                            # Re-validate on submit
                            if len(set(finish_order)) != 5:
                                st.error("❌ Cannot submit: Each position must be unique!")
                            else:
                                with st.spinner("Saving results..."):
                                    success = gold_db.submit_race_results(
                                        race_id=race_id,
                                        finish_order_programs=finish_order
                                    )
                                    
                                    if success:
                                        st.success(f"✅ Results saved for {race_id}!")
                                        st.balloons()
                                        
                                        # Show accuracy feedback
                                        predicted_winner_row = horses_df[horses_df['predicted_rank'] == 1]
                                        predicted_winner = predicted_winner_row['horse_name'].values[0] if not predicted_winner_row.empty else 'Unknown'
                                        
                                        actual_winner = horse_names_dict[pos1]
                                        
                                        if predicted_winner == actual_winner:
                                            st.success(f"🎯 Predicted winner correctly: {actual_winner}")
                                        else:
                                            st.info(f"📊 Predicted: {predicted_winner} | Actual: {actual_winner}")
                                        
                                        st.info("🚀 Go to 'Retrain Model' tab to update predictions with real data!")
                                        
                                        time.sleep(2)
                                        _safe_rerun()
                                    else:
                                        st.error("❌ Error saving results. Please try again.")
        
        # Tab 3: Retrain Model
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
                    conn = sqlite3.connect(gold_db.db_path)
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