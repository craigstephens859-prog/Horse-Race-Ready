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

import os, re, json, math
from typing import Dict, List, Tuple, Optional
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st
from itertools import product

# Try importing ML Engine for Phase 2 functionality
try:
    from ml_engine import MLCalibrator, RaceDatabase
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False
    MLCalibrator = None
    RaceDatabase = None

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

def model_supports_temperature(model_name: str) -> bool:
    m = (model_name or "").lower()
    return not (m.startswith("gpt-5") or m.startswith("o4") or m.startswith("o3"))

def call_openai_messages(messages: List[Dict]) -> str:
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
            if val <= 6.0:  return "≤6f"
            if val < 8.0:   return "6.5–7f"
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
        if total_f < 6.5: return "≤6f"
        if total_f < 8.0: return "6.5–7f"
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
def parse_angles_for_block(block: str) -> pd.DataFrame:
    rows = []
    for m in ANGLE_LINE_RE.finditer(block or ""):
        _yr, cat, starts, win, itm, roi = m.groups()
        rows.append({"Category": re.sub(r"\s+", " ", cat.strip()),
                     "Starts": int(starts), "Win%": float(win),
                     "ITM%": float(itm), "ROI": float(roi)})
    return pd.DataFrame(rows)

def parse_pedigree_snips(block: str) -> dict:
    out = {"sire_awd": np.nan, "sire_1st": np.nan,
           "damsire_awd": np.nan, "damsire_1st": np.nan,
           "dam_dpi": np.nan}
    s = re.search(r'(?mi)^\s*Sire\s*Stats:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi', block or "")
    if s:
        out["sire_awd"] = float(s.group(1)); out["sire_1st"] = float(s.group(3))
    ds = re.search(r'(?mi)^\s*Dam\'s Sire:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi', block or "")
    if ds:
        out["damsire_awd"] = float(ds.group(1)); out["damsire_1st"] = float(ds.group(3))
    d = re.search(r'(?mi)^\s*Dam:\s*DPI\s*(\d+(?:\.\d+)?)\s+(\d+)%', block or "")
    if d:
        out["dam_dpi"] = float(d.group(1))
    return out

SPEED_FIG_RE = re.compile(
    # Matches a date, track, etc., then a race type, then captures the first fig
    r"(?mi)^\s*(\d{2}[A-Za-z]{3}\d{2})\s+.*?" # Date (e.g., 23Sep23)
    r"\b(Clm|Mdn|Md Sp Wt|Alw|OC|G1|G2|G3|Stk|Hcp)\b" # A race type keyword
    r".*?\s+(\d{2,3})\s+" # The first 2-3 digit number after the type
)

def parse_speed_figures_for_block(block: str) -> List[int]:
    """
    Parses a horse's PP text block and extracts all main speed figures.
    """
    figs = []
    if not block:
        return figs
    
    for m in SPEED_FIG_RE.finditer(block):
        try:
            # The speed figure is the second capture group
            fig_val = int(m.group(2))
            # Basic sanity check for a realistic speed figure
            if 40 < fig_val < 130:
                figs.append(fig_val)
        except Exception:
            pass # Ignore if conversion fails
            
    # We only care about the most recent figs, e.g., last 10
    return figs[:10]

# ---------- Probability helpers ----------
def softmax_from_rating(ratings: np.ndarray, tau: Optional[float] = None) -> np.ndarray:
    if ratings.size == 0:
        return ratings
    _tau = tau if tau is not None else MODEL_CONFIG['softmax_tau']
    x = np.array(ratings, dtype=float) / max(_tau, 1e-6)
    x = x - np.max(x)
    ex = np.exp(x)
    p = ex / np.sum(ex)
    return p

def compute_ppi(df_styles: pd.DataFrame) -> dict:
    """PPI that works with Detected/Override styles. Returns {'ppi':float, 'by_horse':{name:tailwind}}"""
    if df_styles is None or df_styles.empty:
        return {"ppi": 0.0, "by_horse": {}}
    styles, names, strengths = [], [], []
    for _, row in df_styles.iterrows():
        stl = row.get("Style") or row.get("OverrideStyle") or row.get("DetectedStyle") or ""
        stl = _normalize_style(stl)
        styles.append(stl)
        names.append(row.get("Horse", ""))
        strengths.append(row.get("StyleStrength", "Solid"))
    counts = {"E": 0, "E/P": 0, "P": 0, "S": 0}
    for stl in styles:
        if stl in counts:
            counts[stl] += 1
    total = sum(counts.values()) or 1
    
    ppi_val = (counts["E"] + counts["E/P"] - counts["P"] - counts["S"]) * MODEL_CONFIG['ppi_multiplier'] / total

    by_horse = {}
    strength_weights = MODEL_CONFIG['style_strength_weights']
    tailwind_factor = MODEL_CONFIG['ppi_tailwind_factor']
    
    for stl, nm, strength in zip(styles, names, strengths):
        wt = strength_weights.get(str(strength), 0.8)
        if stl in ("E", "E/P"):
            by_horse[nm] = round(tailwind_factor * wt * ppi_val, 3)
        elif stl == "S":
            by_horse[nm] = round(-tailwind_factor * wt * ppi_val, 3)
        else:
            by_horse[nm] = 0.0
    return {"ppi": round(ppi_val,3), "by_horse": by_horse}

def_and_figs(ratings_df: pd.DataFrame, pp_text: str, processed_weights: Dict[str,float],
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

    # Apply the final adjustment
    df["R_ENHANCE_ADJ"] = df["R_ENHANCE_ADJ"].fillna(0.0) # Ensure no NaNs
    df["R"] = (df["R"].astype(float) + df["R_ENHANCE_ADJ"].astype(float)).round(2)
    
    return df

# ---------- Odds helpers ----------
def fair_to_american(p: float) -> float:
    if p <= 0: return float("inf")
    if p >= 1: return 0.0
    dec = 1.0/p
    return round((dec-1)*100,0) if dec>=2 else round(-100/(dec-1),0)

def fair_to_american_str(p: float) -> str:
    v = fair_to_american(p)
    if v == float("inf"): return "N/A"
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
    rows = []
    for h, p in fair_probs.items():
        off_dec = offered.get(h)
        if off_dec is None: continue
        off_prob = 1.0/off_dec if off_dec>0 else 0.0
        ev = (off_dec-1)*p - (1-p)
        rows.append({"Horse":h, "Fair %":round(p*100,2),
                     "Fair (AM)":fair_to_american_str(p),
                     "Board (dec)":round(off_dec,3),
                     "Board %":round(off_prob*100,2),
                     "Edge (pp)":round((p-off_prob)*100,2),
                     "EV per $1":round(ev,3), "Overlay?":"YES" if off_prob < p else "NO"})
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
    df["Fair Odds"] = df["Fair Odds"].apply(lambda x: f"{x:.2f}" if np.isfinite(x) else "inf")
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
st.session_state['track_name'] = track_name

# Surface auto from header, but allow override
default_surface = st.session_state['surface_type']
if re.search(r'(?i)\bturf|trf\b', first_line): default_surface = "Turf"
if re.search(r'(?i)\baw|tap|synth|poly\b', first_line): default_surface = "Synthetic"
surface_type = st.selectbox("Surface:", ["Dirt","Turf","Synthetic"],
                            index=["Dirt","Turf","Synthetic"].index(default_surface) if default_surface in ["Dirt", "Turf", "Synthetic"] else 0) # Added check
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

for _post, name, block in split_into_horse_chunks(pp_text):
    if name in df_editor["Horse"].values:
        angles_per_horse[name] = parse_angles_for_block(block)
        pedigree_per_horse[name] = parse_pedigree_snips(block)
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

# ===================== Class build per horse (angles+pedigree in background) =====================

def _infer_horse_surface_pref(name: str, ped: dict, ang_df: Optional[pd.DataFrame], race_surface: str) -> str:
    cats = " ".join(ang_df["Category"].astype(str).tolist()).lower() if (ang_df is not None and not ang_df.empty) else ""
    if "dirt to turf" in cats: return "Turf"
    if "turf to dirt" in cats: return "Dirt"
    # If nothing clear, use race surface (neutral) to avoid over-penalizing
    return race_surface

def _infer_horse_distance_pref(ped: dict) -> str:
    awds = [x for x in [ped.get("sire_awd"), ped.get("damsire_awd")] if pd.notna(x)]
    if not awds:
        return "any"
    m = float(np.nanmean(awds))
    if m <= 6.5: return "≤6f"
    if m >= 7.5: return "8f+"
    return "6.5–7f"

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
    if race_bucket == "≤6f":
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
        if ("dirt to turf" in cats and race_surface.lower() == "turf"):
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

# Build Cclass as additive bonus (invert calculate_final_rating lower-is-better)
race_surface = surface_type
race_cond = condition_txt
race_bucket = distance_bucket(distance_txt)

Cclass_vals = []
for _, r in df_final_field.iterrows():
    name = r["Horse"]
    ped  = pedigree_per_horse.get(name, {}) or {}
    ang  = angles_per_horse.get(name) # Can be None
    if ang is None:
         ang = pd.DataFrame() # Ensure ang is DataFrame for _infer_horse_surface_pref
    surf_pref = _infer_horse_surface_pref(name, ped, ang, race_surface)
    dist_pref = _infer_horse_distance_pref(ped)

    base_scalar = calculate_final_rating(
        race_type=race_type_detected,
        race_surface=race_surface,
        race_distance_category=race_bucket,
        race_surface_condition=race_cond,
        horse_surface_pref=surf_pref,
        horse_distance_pref=dist_pref
    )
    # Convert to additive bonus; more reliable (lower scalar) -> bigger bonus
    cclass_add = round(1.00 - base_scalar, 3)
    cclass_add += _angles_pedigree_tweak(name, race_surface, race_bucket, race_cond)
    Cclass_vals.append(cclass_add)

df_final_field["Cclass"] = Cclass_vals

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
            except:
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
    
    if surface_type.lower() == 'turf':
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
                         pp_text: str = "") -> pd.DataFrame:
    """
    Reads 'Cclass' from df_styles (pre-built), adds Cstyle/Cpost/Cpace (+Atrack),
    plus Tier 2 bonuses (Impact Values, SPI, Surface Stats, AWD),
    sums to Arace and R. Returns rating table.
    """
    cols = ["#", "Post", "Horse", "Style", "Quirin", "Cstyle", "Cpost", "Cpace", "Cclass", "Atrack", "Arace", "R"]
    if df_styles is None or df_styles.empty:
        return pd.DataFrame(columns=cols)

    # Ensure class column present
    if "Cclass" not in df_styles.columns:
        df_styles = df_styles.copy()
        df_styles["Cclass"] = 0.0 # Default Cclass if missing

    # Derive per-horse pace tailwind from PPI
    ppi_map = compute_ppi(df_styles).get("by_horse", {})
    
    # ======================== Phase 1: Parse Tier 2 Enhancements ========================
    impact_values = parse_track_bias_impact_values(pp_text) if pp_text else {}
    spi_values = parse_pedigree_spi(pp_text) if pp_text else {}
    surface_stats = parse_pedigree_surface_stats(pp_text) if pp_text else {}
    awd_analysis = parse_awd_analysis(pp_text) if pp_text else {}

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

        a_track = _get_track_bias_delta(track_name, surface_type, distance_txt, style, post)

        c_class = float(row.get("Cclass", 0.0))
        
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
            if surface_type.lower() == 'turf' and 'turf_pct' in stats:
                tier2_bonus += calculate_surface_specialty_bonus(stats['turf_pct'], 'turf')
            elif surface_type.lower() in ['aw', 'all-weather', 'synthetic'] and 'aw_pct' in stats:
                tier2_bonus += calculate_surface_specialty_bonus(stats['aw_pct'], 'aw')
        
        # 4. AWD (Distance Mismatch) penalty
        if name in awd_analysis:
            tier2_bonus += calculate_awd_mismatch_penalty(awd_analysis[name])
        
        # ======================== End Tier 2 Bonuses ========================

        arace = c_class + cstyle + cpost + cpace + a_track + tier2_bonus
        R     = arace

        # Ensure Quirin is formatted correctly for display (handle NaN)
        quirin_display = quirin if pd.notna(quirin) else None

        rows.append({
            "#": post, "Post": post, "Horse": name, "Style": style, "Quirin": quirin_display,
            "Cstyle": round(cstyle, 2), "Cpost": round(cpost, 2), "Cpace": round(cpace, 2),
            "Cclass": round(c_class, 2), "Atrack": round(a_track, 2), "Arace": round(arace, 2), "R": round(R, 2)
        })
    out = pd.DataFrame(rows, columns=cols)
    return out.sort_values(by="R", ascending=False)


def fair_probs_from_ratings(ratings_df: pd.DataFrame) -> Dict[str, float]:
    if ratings_df is None or ratings_df.empty or "R" not in ratings_df.columns or "Horse" not in ratings_df.columns:
        return {}
    # Ensure 'R' is numeric, coercing errors to NaN, then fill NaN with a default (e.g., median or 0)
    ratings_df['R_numeric'] = pd.to_numeric(ratings_df['R'], errors='coerce')
    median_r = ratings_df['R_numeric'].median()
    if pd.isna(median_r): median_r = 0 # Handle case where all are NaN
    ratings_df['R_numeric'].fillna(median_r, inplace=True)

    r = ratings_df["R_numeric"].values
    if len(r) == 0: return {}

    p = softmax_from_rating(r) # tau will be pulled from MODEL_CONFIG

    # Clean up temporary column
    ratings_df.drop(columns=['R_numeric'], inplace=True)

    # Ensure probabilities sum to 1 (or very close)
    p_sum = np.sum(p)
    if p_sum > 0:
      p = p / p_sum

    return {h: p[i] for i, h in enumerate(ratings_df["Horse"].values)}


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
            pp_text=pp_text
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
                "R": st.column_config.NumberColumn("Rating", format="%.2f"),
                "Cstyle": st.column_config.NumberColumn("C-Style", format="%.2f"),
                "Cpost": st.column_config.NumberColumn("C-Post", format="%.2f"),
                "Cpace": st.column_config.NumberColumn("C-Pace", format="%.2f"),
                "Cclass": st.column_config.NumberColumn("C-Class", format="%.2f"),
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
st.dataframe(
    df_ol,
    use_container_width=True, hide_index=True,
    column_config={
        "EV per $1": st.column_config.NumberColumn("EV per $1", format="$%.3f"),
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
             blueprint_report += f"* **Win Bets:** Consider betting all **A-Group** horses.\n"
        else: # Confident
             blueprint_report += f"* **Win Bet:** Focus on top **A-Group** horse(s).\n"
        
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
        blueprint_report += f"\n#### Confident Profile Plan\n"
        blueprint_report += "_Focus: Key A-Group horses ON TOP._\n"
        blueprint_report += f"* **Win Bet:** Focus on top **A-Group** horse(s).\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B` ({nA}x{nB}) - {get_min_cost_str(1.00, nA, nB)}\n"
        if nA >=1 and nB >= 1 and nC >=1: # Check if groups have members for straight example
             blueprint_report += f"* **Straight Trifecta:** `Top A / Top B / Top C` (1 combo) - Consider at higher base (e.g., $1 or $2).\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B / C / D` ({nA}x{nB}x{nC}x{nD}) - {get_min_cost_str(0.10, nA, nB, nC, nD)}\n"
        if field_size >= 7: # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B / C / D / ALL` ({nA}x{nB}x{nC}x{nD}x{nAll}) - {get_min_cost_str(0.10, nA, nB, nC, nD, nAll)}\n"
        
        # --- Value-Hunter Blueprint ---
        blueprint_report += f"\n#### Value-Hunter Profile Plan\n"
        blueprint_report += "_Focus: Use A-Group (includes overlays) ON TOP, spread wider underneath._\n"
        blueprint_report += f"* **Win Bets:** Consider betting all **A-Group** horses.\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B,C` ({nA}x{nB+nC}) - {get_min_cost_str(1.00, nA, nB + nC)}\n"
        if nA >= 3: # Example box if A group is large enough
             tri_box_combos = get_box_combos(nA, 3)
             blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str,[int(name_to_post.get(h,'0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B,C / B,C,D` ({nA}x{nB+nC}x{nB+nC+nD}) - {get_min_cost_str(0.50, nA, nB + nC, nB + nC + nD)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B,C / B,C,D / ALL` ({nA}x{nB+nC}x{nB+nC+nD}x{nAll}) - {get_min_cost_str(0.10, nA, nB + nC, nB + nC + nD, nAll)}\n"
        if field_size >= 7: # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B,C / B,C,D / ALL / ALL` ({nA}x{nB+nC}x{nB+nC+nD}x{nAll}x{nAll}) - {get_min_cost_str(0.10, nA, nB + nC, nB + nC + nD, nAll, nAll)}\n"

    # --- 6. Build Final Report String ---
    final_report = f"""
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
if st.button("Analyze This Race", type="primary", key="analyze_button"):
    with st.spinner("Handicapping Race..."):
        try:
            # --- 1. Build Data for Strategy & Prompt ---
            if primary_df.empty or not all(col in primary_df.columns for col in ['Horse', 'R', 'Fair %', 'Fair Odds']):
                st.error("Primary ratings data is incomplete for report generation.")
                st.stop()

            primary_sorted = primary_df.sort_values(by="R", ascending=False)
            name_to_post = pd.Series(df_final_field["Post"].values,
                                     index=df_final_field["Horse"]).to_dict()
            name_to_ml = pd.Series(df_final_field["ML"].values, 
                                   index=df_final_field["Horse"]).to_dict()
            field_size = len(primary_df)
            
            top_table = primary_sorted[['Horse','R','Fair %','Fair Odds']].head(5).to_markdown(index=False)

            overlay_pos = df_ol[df_ol["EV per $1"] > 0] if not df_ol.empty else pd.DataFrame()
            overlay_table_md = (overlay_pos[['Horse','Fair %','Fair (AM)','Board (dec)','EV per $1']].to_markdown(index=False)
                                if not overlay_pos.empty else "None.")

            # --- 2. NEW: Generate Simplified A/B/C/D Strategy Report ---
            strategy_report_md = build_betting_strategy(
                primary_df, df_ol, strategy_profile, name_to_post, name_to_ml, field_size, ppi_val
            )

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

            st.download_button("⬇️ Download Full Analysis (.txt)", data=analysis_bytes, file_name="analysis.txt", mime="text/plain")
            st.download_button("⬇️ Download Overlays (CSV)", data=overlays_bytes, file_name="overlays.csv", mime="text/csv")
            st.download_button("⬇️ Download Strategy Detail (.txt)", data=tickets_bytes, file_name="strategy_detail.txt", mime="text/plain") # Renamed for clarity

        except Exception as e:
            st.error(f"Error generating report: {e}")
            import traceback
            st.error(traceback.format_exc())

# ===================== E. ML System & Results Tracking =====================

st.header("E. ML System & Results Tracking")

if ML_AVAILABLE:
    try:
        db = RaceDatabase()
        calibrator = MLCalibrator(db)
        
        tab_results, tab_history, tab_train, tab_predict = st.tabs(["📝 Enter Results", "📊 Race History", "🤖 Train Model", "🎯 Get Predictions"])
        
        # Tab 1: Enter Race Results
        with tab_results:
            st.subheader("Enter Race Results After Completion")
            
            if st.session_state.get("parsed", False):
                # Pre-fill race info from current analysis
                result_track = st.text_input("Track Name", value=st.session_state.get('track_name', ''))
                result_date = st.date_input("Race Date", value=datetime.now())
                result_race_num = st.number_input("Race Number", min_value=1, value=1)
                result_distance = st.text_input("Distance", value=st.session_state.get('distance_txt', ''))
                result_surface = st.selectbox("Surface", ["Dirt", "Turf", "Synthetic"], 
                                             index=["Dirt", "Turf", "Synthetic"].index(st.session_state.get('surface_type', 'Dirt')))
                
                st.markdown("#### Enter Horse Results")
                
                # Get horses from current analysis
                if 'df_final_field' in locals() and df_final_field is not None:
                    horses = df_final_field['Horse'].tolist()
                else:
                    horses = []
                
                if horses:
                    winner = st.selectbox("Winner", horses)
                    finish_positions = {}
                    final_odds = {}
                    
                    for horse in horses:
                        col1, col2 = st.columns(2)
                        with col1:
                            finish_positions[horse] = st.number_input(f"{horse} - Finish Position", 
                                                                     min_value=1, 
                                                                     value=horses.index(horse) + 1,
                                                                     key=f"pos_{horse}")
                        with col2:
                            final_odds[horse] = st.number_input(f"{horse} - Final Odds (decimal)", 
                                                                min_value=1.01, 
                                                                value=5.0,
                                                                key=f"odds_{horse}")
                    
                    if st.button("💾 Save Race Results"):
                        try:
                            # Save race
                            race_id = db.save_race(
                                track=result_track,
                                date=result_date.strftime('%Y-%m-%d'),
                                race_number=result_race_num,
                                distance=result_distance,
                                surface=result_surface
                            )
                            
                            # Save horses
                            for horse in horses:
                                db.save_horse(
                                    race_id=race_id,
                                    name=horse,
                                    post_position=horses.index(horse) + 1,
                                    final_odds=final_odds[horse],
                                    finish_position=finish_positions[horse],
                                    won=(horse == winner)
                                )
                            
                            st.success(f"✅ Results saved! Race ID: {race_id}")
                        except Exception as e:
                            st.error(f"Error saving results: {e}")
                else:
                    st.info("📋 Parse a race first in Section A to see horses here")
            else:
                st.info("📋 Parse a race first in Section A, then return here to enter results")
        
        # Tab 2: Race History
        with tab_history:
            st.subheader("Historical Race Data")
            
            summary = db.get_race_summary()
            if summary:
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("Total Races", summary['total_races'])
                with col2:
                    st.metric("Total Horses", summary['total_horses'])
                with col3:
                    avg_field = summary['total_horses'] / summary['total_races'] if summary['total_races'] > 0 else 0
                    st.metric("Avg Field Size", f"{avg_field:.1f}")
                
                # Show recent races
                st.markdown("#### Recent Races")
                races = db.get_races(limit=20)
                if races:
                    race_df = pd.DataFrame(races)
                    st.dataframe(race_df, use_container_width=True)
                else:
                    st.info("No races recorded yet")
            else:
                st.info("📊 No race data yet. Enter results in the '📝 Enter Results' tab after races complete.")
        
        # Tab 3: Train Model
        with tab_train:
            st.subheader("Train ML Probability Calibration Model")
            
            summary = db.get_race_summary()
            if summary and summary['total_races'] >= 10:
                st.info(f"📚 {summary['total_races']} races available for training")
                
                use_pytorch = st.checkbox("Use PyTorch Neural Network (if available)", value=True)
                epochs = st.slider("Training Epochs", min_value=10, max_value=500, value=100)
                
                if st.button("🚀 Train Model"):
                    with st.spinner("Training model..."):
                        try:
                            results = calibrator.train(use_neural_net=use_pytorch, epochs=epochs)
                            
                            st.success(f"✅ Model trained successfully!")
                            st.json(results)
                            
                            # Show loss plot if available
                            if 'loss_history' in results and results['loss_history']:
                                loss_df = pd.DataFrame({'Epoch': range(len(results['loss_history'])), 
                                                       'Loss': results['loss_history']})
                                st.line_chart(loss_df.set_index('Epoch'))
                        except Exception as e:
                            st.error(f"Training error: {e}")
                            import traceback
                            st.error(traceback.format_exc())
            else:
                needed = 10 - (summary['total_races'] if summary else 0)
                st.warning(f"⚠️ Need at least 10 races to train model. Enter {needed} more race results.")
        
        # Tab 4: Get Predictions
        with tab_predict:
            st.subheader("ML-Enhanced Probability Predictions")
            
            if calibrator.is_trained():
                st.success("✅ Model is trained and ready")
                
                if st.session_state.get("parsed", False) and 'ratings_df' in locals():
                    st.markdown("#### Current Race Predictions")
                    
                    # Get raw probabilities from ratings
                    if 'all_scenario_ratings' in locals() and all_scenario_ratings:
                        # Use first scenario's ratings
                        first_key = list(all_scenario_ratings.keys())[0]
                        rating_probs = fair_probs_from_ratings(all_scenario_ratings[first_key])
                        
                        if rating_probs:
                            # Get ML-calibrated probabilities
                            horses_list = list(rating_probs.keys())
                            raw_probs = [rating_probs[h] for h in horses_list]
                            
                            try:
                                calibrated = calibrator.predict(raw_probs)
                                
                                # Create comparison dataframe
                                pred_df = pd.DataFrame({
                                    'Horse': horses_list,
                                    'Raw Probability': [f"{p:.1%}" for p in raw_probs],
                                    'ML Calibrated': [f"{p:.1%}" for p in calibrated],
                                    'Adjustment': [f"{(c-r)*100:+.1f}%" for r, c in zip(raw_probs, calibrated)]
                                })
                                
                                st.dataframe(pred_df, use_container_width=True)
                                st.info("💡 ML model adjusts probabilities based on historical performance")
                            except Exception as e:
                                st.error(f"Prediction error: {e}")
                    else:
                        st.info("Run 'Analyze This Race' first to see predictions")
                else:
                    st.info("📋 Parse and analyze a race first to get ML predictions")
            else:
                st.warning("⚠️ Model not trained yet. Train it in the '🤖 Train Model' tab first (requires 10+ races).")
    
    except Exception as e:
        st.error(f"Error initializing ML system: {e}")
        import traceback
        st.error(traceback.format_exc())
else:
    st.warning("""
    ⚠️ **ML System Not Available**
    
    The ML Engine could not be loaded. This is usually because required dependencies are missing.
    
    **To enable Section E (Results Tracking & ML Training):**
    1. Install requirements: `pip install -r requirements-ml.txt`
    2. Or install manually: `pip install scikit-learn torch numpy`
    3. Restart the Streamlit app
    
    The core rating system (Sections A-D) works fine without ML.
    """)

st.markdown("---")
st.caption("Horse Race Ready - IQ Mode | Advanced Track Bias Analysis with ML Probability Calibration")