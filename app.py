# app.py
# Horse Race Ready — IQ Mode (Full, final version)
# - Robust PP parsing (incl. NA firsters)
# - Auto track + race-type detection mapped to constant base_class_bias keys
# - Track-bias integration (by track/surface/distance bucket + style/post)
# - Per-horse angles + pedigree tweaks folded into class math
# - Speed Figure parsing and integration
# - Centralized MODEL_CONFIG for easy tuning
# - Robust error handling with st.warning
# - Advanced "common sense" A/B/C/D strategy builder w/ budgeting, field size logic, straight/box examples
# - Super High 5 exotic support
# - Classic bullet-style report with download buttons
# - Resilient to older/newer Streamlit rerun APIs

import os, re, json, math
from typing import Dict, List, Tuple, Optional
from collections import Counter

import numpy as np
import pandas as pd
import streamlit as st
from itertools import product, permutations

# PuLP for ticket optimization
try:
    import pulp
    PULP_AVAILABLE = True
except ImportError:
    PULP_AVAILABLE = False
    st.warning("⚠️ PuLP not installed. Ticket optimization disabled. Install with: pip install pulp")

# ML imports (optional - graceful fallback if not installed)
try:
    from sklearn.ensemble import RandomForestRegressor
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    st.warning("⚠️ scikit-learn not installed. ML adjustment disabled. Install with: pip install scikit-learn")

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
    "softmax_tau": 0.55, "speed_fig_weight": 0.05, "first_timer_fig_default": 50,
    "ppi_multiplier": 1.0, "ppi_tailwind_factor": 0.3,
    "prime_power_weight": 0.09, "late_pace_weight": 0.07, "trainer_meet_bonus": 0.08,
    "jockey_upgrade_bonus": 0.07, "layoff_bullet_bonus": 0.10, "class_drop_bonus": 0.12,
    "front_bandages_bonus": 0.05, "lasix_off_penalty": -0.08, "back_fig_par_bonus": 0.08,
    "quirin_lp_sneak_bonus": 0.11, "frac_pace_bonus": 0.09, "dam_sire_turf_bonus": 0.10,
    "dam_sire_route_bonus": 0.09, "trainer_pattern_roi_scale": 0.02, "bad_trip_bonus": 0.06,
    "fig_trend_bonus": 0.11, "equip_bonus": 0.07, "bounce_penalty": -0.09,
    "sire_mud_bonus": 0.08, "owner_roi_bonus": 0.06, "odds_drift_bonus": 0.05,
    
    # --- Pace & Style Model ---
    "style_strength_weights": { # Multiplier for pace tailwind based on strength.
        "Strong": 1.0, 
        "Solid": 0.8, 
        "Slight": 0.5, 
        "Weak": 0.3
    },
    
    # --- Manual Bias Model (Section B) ---
    "style_match_table": {
        "speed favoring": {"E": 0.70, "E/P": 0.50, "P": -0.20, "S": -0.50},
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
    
    # --- Trainer Intent Signals ---
    "trainer_intent_class_drop_bonus": 0.12,  # Big drop = win now
    "trainer_intent_jky_switch_bonus": 0.09,  # To top jky = intent
    "trainer_intent_equip_bonus": 0.08,       # Blinkers on = focus
    "trainer_intent_layoff_works_bonus": 0.10,# Sharp works post-layoff
    "trainer_intent_ship_bonus": 0.07,        # From better track
    "trainer_intent_roi_threshold": 1.5,      # Positive ROI angle multiplier
    
    # --- Exotics & Strategy ---
    "exotic_bias_weights": (1.30, 1.15, 1.05, 1.03), # (1st, 2nd, 3rd, 4th) Harville bias
    "positions_to_sim": 5,  # For SH5
    "top_for_pos": [2, 2, 3, 3, 3],  # 1st:2, 2nd:2, 3rd:3, 4th:3, 5th:3
    "closer_bias_high_ppi": 0.35,  # Probability boost for closers when PPI > 0.5
    "strategy_confident": { # Placeholders, not used by new strategy builder
        "ex_max": 4, "ex_min_prob": 0.020,
        "tri_max": 6, "tri_min_prob": 0.010,
        "sup_max": 8, "sup_min_prob": 0.008,
    },
    "strategy_value": { # Placeholders, not used by new strategy builder
        "ex_max": 6, "ex_min_prob": 0.015,
        "tri_max": 10, "tri_min_prob": 0.008,
        "sup_max": 12, "sup_min_prob": 0.006,
    },
    
    # --- Jockey & Trainer Intent Bonuses ---
    "intent_max_bonus": 0.20,
    "trainer_roi_bonus_per": 0.05,
    "jock_win_bonus_per": 0.60,
    "jock_upgrade_bonus_per": 0.04,
    
    # --- Figure Trend Analysis (NEW) ---
    "fig_uptrend_bonus": 0.12,  # Last 3 figs improving
    "fig_downtrend_penalty": -0.10,  # Last 3 figs declining
    
    # --- Recency Weighting (NEW) ---
    "recency_decay_rate": 0.15,  # Exponential decay: recent races weighted heavier
    
    # --- Distance Consistency (NEW) ---
    "distance_specialist_threshold": 0.22,
    "distance_specialist_bonus": 0.08,
    "distance_poor_threshold": 0.10,
    "distance_poor_penalty": -0.06,
    
    # --- Bounce Detection (NEW) ---
    "bounce_fig_drop_threshold": 6,
    "bounce_penalty_aggressive": -0.10,
    
    # --- Class Transition (NEW) ---
    "class_rise_penalty": -0.07,
    "blinkers_off_penalty": -0.06,
    
    # --- Dynamic Exotic Probabilities ---
    "positions_to_sim": 5,  # For SH5
    "top_for_pos": [2, 2, 3, 3, 3],  # 1st:2, 2nd:2, 3rd:3, 4th:3, 5th:3
    "closer_bias_high_ppi": 0.35,  # Probability of boosting closer in high-PPI scenarios
    
    # --- TIER 1: BRISNET Pedigree Ratings (NEW) ---
    "bris_ped_fast_bonus": 0.06,  # Fast track specialist
    "bris_ped_off_bonus": 0.05,  # Off-track specialist (mud/sloppy)
    "bris_ped_distance_bonus": 0.07,  # Distance specialist
    "bris_ped_turf_bonus": 0.06,  # Turf specialist
    "bris_ped_rating_threshold": 85,  # Bonus applied if rating >= threshold
    
    # --- TIER 1: Dam Production Index Bonus (NEW) ---
    "dpi_bonus_threshold": 1.5,  # Apply bonus if DPI > 1.5 (dam produces above-average earners)
    "dpi_bonus": 0.05,  # Bonus for high DPI
    
    # --- TIER 1: Surface-Specific Record Penalty (NEW) ---
    "surface_mismatch_off_penalty": -0.06,  # Penalize poor off-track record on mud
    "surface_mismatch_dirt_penalty": -0.05,  # Penalize poor dirt record on dirt
    "surface_specialist_threshold_good": 0.40,  # Strong record threshold
    "surface_specialist_threshold_poor": 0.20,  # Poor record threshold
    
    # --- PART 2 ENHANCEMENT: CR/RR Performance Ratio (NEW) ---
    "cr_rr_outperform_threshold": 0.95,  # Ratio >= 0.95 = performing close to field RR
    "cr_rr_excellent_threshold": 1.00,   # Ratio >= 1.00 = performing above field
    "cr_rr_outperform_bonus": 0.06,      # Bonus if ratio >= 0.95
    "cr_rr_excellent_bonus": 0.10,       # Bonus if ratio >= 1.00
    "cr_rr_consistency_bonus": 0.04,     # Bonus for consistent CR performances
    
    # --- TIER 2: PEDIGREE SPI (Sire Production Index) ---
    "spi_strong_threshold": 1.0,         # SPI >= 1.0 = strong sire
    "spi_strong_bonus": 0.06,            # Bonus for strong sire
    "spi_weak_threshold": 0.5,           # SPI <= 0.5 = weak sire
    "spi_weak_penalty": -0.05,           # Penalty for weak sire
    
    # --- TIER 2: SURFACE SPECIALTY STATISTICS (Sire %Mud, %Turf) ---
    "mud_specialist_threshold": 25,      # Sire %Mud >= 25% = specialist
    "mud_specialist_bonus": 0.08,        # Bonus for mud specialist
    "mud_moderate_threshold": 15,        # Sire %Mud >= 15% = moderate
    "mud_moderate_bonus": 0.04,          # Bonus for moderate mud record
    "turf_specialist_threshold": 30,     # Sire %Turf >= 30% = specialist
    "turf_specialist_bonus": 0.08,       # Bonus for turf specialist
    "turf_moderate_threshold": 20,       # Sire %Turf >= 20% = moderate
    "turf_moderate_bonus": 0.04,         # Bonus for moderate turf record
    
    # --- TIER 2: AWD (Average Winning Distance) MISMATCH PENALTIES ---
    "awd_large_mismatch_penalty": -0.08,     # 2.0f+ mismatch
    "awd_moderate_mismatch_penalty": -0.04,  # 1.0-2.0f mismatch
    "awd_small_mismatch_penalty": -0.02,     # 0.5-1.0f mismatch
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

def apex_enhance(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["APEX"] = 0.0
    
    # Ensure AvgTop2 column exists
    if "AvgTop2" not in df.columns:
        df["AvgTop2"] = MODEL_CONFIG['first_timer_fig_default']
    
    # Safe calculations with fallbacks if data is missing
    max_prime = df["Prime"].max() if not df["Prime"].isna().all() else 0
    if pd.isna(max_prime):
        max_prime = 0
    
    lp_values = [np.mean(a["lp"] or [50]) for a in all_angles_per_horse.values() if a.get("lp")]
    avg_lp = np.nanmean(lp_values) if lp_values else 50
    
    frac_values = [np.mean([f[0] for f in a["frac"]][:3]) for a in all_angles_per_horse.values() if a.get("frac")]
    best_frac = min(frac_values) if frac_values else 99
    
    race_avg_avgtop2 = df["AvgTop2"].mean()
    
    for i, r in df.iterrows():
        h = r["Horse"]
        if h not in all_angles_per_horse:
            continue
        a = all_angles_per_horse[h]
        adj = (r["AvgTop2"] - race_avg_avgtop2) * MODEL_CONFIG["speed_fig_weight"]
        
        # Safe Prime adjustment (handle NaN)
        prime_val = r["Prime"] if not pd.isna(r["Prime"]) else 0
        adj += (prime_val - max_prime) * 0.09
        
        # Safe LP adjustment
        adj += (np.mean(a.get("lp") or [50]) - avg_lp) * 0.07
        adj += 0.08 if a.get("trainer_win", 0) >= 23 else 0
        adj += 0.07 if any(j in a.get("jockey", "") for j in ["Irad Ortiz Jr","Flavien Prat","Jose Ortiz","Joel Rosario","John Velazquez","Tyler Gaffalione"]) else 0
        adj += 0.10 if 45 <= a.get("layoff", 0) <= 180 and a.get("bullets", 0) >= 3 else 0
        # Pattern bonuses are already numeric, summed later - skip string check
        adj += 0.05 if "Front Bandages On" in a.get("equip", "") else 0
        adj -= 0.08 if "Lasix Off" in a.get("equip", "") else 0
        figs_dict = figs_per_horse.get(h, {})
        figs_list = figs_dict.get('SPD', []) if isinstance(figs_dict, dict) else []
        recent_figs = figs_list[1:4] if figs_list and len(figs_list) > 1 else []
        adj += 0.08 if recent_figs and max(recent_figs) >= today_par + 8 else 0
        adj += 0.11 if r["Style"] not in ("E","E/P") and np.mean(a.get("lp") or [50]) >= avg_lp + 8 else 0
        
        # Safe frac calculation
        horse_fracs = a.get("frac", [])
        if horse_fracs:
            horse_avg_frac = np.mean([f[0] for f in horse_fracs[:3]])
            adj += 0.09 if horse_avg_frac <= best_frac + 2 else 0
        dam_sire = a.get("dam_sire", (0, 0))
        adj += 0.10 if surface_type=="Turf" and dam_sire[0] >= 19 else 0
        adj += 0.09 if distance_bucket(distance_txt)=="8f+" and dam_sire[1] >= 22 else 0
        patterns = a.get("patterns", [])
        adj += min(sum(p for p in patterns if isinstance(p, (int, float)) and p > 0) * 0.02, 0.12)
        adj += a.get("trips", 0) * 0.06 if a.get("trips", 0) >= 2 else 0
        adj += 0.11 if figs_list and len(figs_list) >= 3 and figs_list[0] > figs_list[1] > figs_list[2] else 0
        equip = a.get("equip", "")
        adj += 0.07 if equip and "Lasix Off" not in equip else 0
        adj -= 0.09 if a.get("bounce", False) else 0
        adj += 0.08 if condition_txt in ("muddy","sloppy") and a.get("sire_mud", 0) >= 18 else 0
        adj += 0.06 if a.get("owner_roi", 0) > 0 else 0
        
        # Trainer Intent Signals
        intent = trainer_intent_per_horse.get(h, {})
        adj += MODEL_CONFIG["trainer_intent_class_drop_bonus"] if intent.get("class_drop_pct", 0) >= 30 else 0
        adj += MODEL_CONFIG["trainer_intent_jky_switch_bonus"] if intent.get("jky_switch", False) and a.get("trainer_win", 0) >= 20 else 0
        adj += MODEL_CONFIG["trainer_intent_equip_bonus"] if intent.get("equip_change") == "blink_on" else 0
        adj += MODEL_CONFIG["trainer_intent_layoff_works_bonus"] if a.get("layoff", 0) > 45 and intent.get("layoff_works", 0) >= 3 else 0
        adj += MODEL_CONFIG["trainer_intent_ship_bonus"] if intent.get("ship_from", "") in ["SAR", "CD", "BEL"] else 0  # Elite tracks
        adj += (intent.get("roi_angles", 0) / MODEL_CONFIG["trainer_intent_roi_threshold"]) * 0.04  # Scaled ROI boost
        
        # Post-Time Odds Drift Tracker (Live Odds vs ML Delta)
        ml_dec  = str_to_decimal_odds(df_final_field.loc[df_final_field["Horse"]==h, "ML"].iloc[0]) if h in df_final_field["Horse"].values else None
        live_dec = str_to_decimal_odds(df_final_field.loc[df_final_field["Horse"]==h, "Live Odds"].iloc[0]) if h in df_final_field["Horse"].values else None
        if ml_dec and live_dec:
            live_dec = live_dec or ml_dec
        elif ml_dec:
            live_dec = ml_dec
        else:
            ml_dec = live_dec = None
        
        if ml_dec and live_dec and ml_dec > 0:
            drift_pct = max(0, (ml_dec - live_dec) / ml_dec)  # only positive drift rewarded
            adj += drift_pct * 0.50  # 50 basis points per 100% drift
        
        df.loc[i, "APEX"] = round(adj, 3)
    df["R"] = (df["R"] + df["APEX"]).round(2)
    return df

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
                          style: str, post_str: str,
                          impact_values: Optional[dict] = None) -> float:
    """
    Calculate track bias delta using Impact Values if available, otherwise fall back to generic profiles.
    
    Args:
        track_name: Name of the racetrack
        surface_type: Dirt/Turf/Synthetic
        distance_txt: Race distance (e.g., "6 Furlongs")
        style: Running style (E, E/P, P, S)
        post_str: Post position as string
        impact_values: Dict with structure {'running_style': {...}, 'post_position': {...}}
    
    Returns:
        float: Track bias delta bonus/penalty clipped to [-1.0, 1.0]
    """
    
    # PRIORITY 1: Use Impact Values if provided and present in data
    if impact_values and isinstance(impact_values, dict):
        impact_values_data = impact_values.get('running_style', {})
        post_impact_data = impact_values.get('post_position', {})
        
        # Try to extract Impact Values
        s_norm = _style_norm(style)
        rs_impact = impact_values_data.get(s_norm, None)
        
        post_bucket = _post_bucket(post_str)
        post_impact = post_impact_data.get(post_bucket, None)
        
        # If we have at least one Impact Value, use them (convert from percentage to bonus)
        # Impact Value 1.41 means +41% edge -> convert to +0.41 bonus
        if rs_impact is not None or post_impact is not None:
            rs_delta = 0.0
            post_delta = 0.0
            
            # Running Style Impact: 1.30 means +30% edge = +0.30 bonus
            if rs_impact is not None:
                rs_delta = float(rs_impact - 1.0) if rs_impact > 0 else 0.0
            
            # Post Position Impact: 1.41 means +41% edge = +0.41 bonus
            if post_impact is not None:
                post_delta = float(post_impact - 1.0) if post_impact > 0 else 0.0
            
            # Use Impact Values (data-driven)
            combined_delta = rs_delta + post_delta
            return float(np.clip(combined_delta, -1.0, 1.0))
    
    # FALLBACK: Use generic TRACK_BIAS_PROFILES if Impact Values not available
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

def calculate_spi_bonus(spi: float, dam_sire_spi: float) -> float:
    """
    Calculate bonus/penalty based on Sire Production Index (SPI).
    SPI < 1.0 = weak sire, SPI > 1.0 = strong sire
    Typical range: 0.3-0.7 (mostly weak to moderate)
    
    Returns: bonus/penalty float
    """
    bonus = 0.0
    
    # Sire SPI bonus/penalty
    if pd.notna(spi):
        if spi >= MODEL_CONFIG.get('spi_strong_threshold', 1.0):
            bonus += MODEL_CONFIG.get('spi_strong_bonus', 0.06)
        elif spi <= MODEL_CONFIG.get('spi_weak_threshold', 0.5):
            bonus += MODEL_CONFIG.get('spi_weak_penalty', -0.05)
    
    # Dam-Sire SPI (lesser impact, typically worth 50% of sire impact)
    if pd.notna(dam_sire_spi):
        if dam_sire_spi >= MODEL_CONFIG.get('spi_strong_threshold', 1.0):
            bonus += MODEL_CONFIG.get('spi_strong_bonus', 0.06) * 0.5
        elif dam_sire_spi <= MODEL_CONFIG.get('spi_weak_threshold', 0.5):
            bonus += MODEL_CONFIG.get('spi_weak_penalty', -0.05) * 0.5
    
    return float(np.clip(bonus, -0.12, 0.12))

def calculate_surface_specialty_bonus(sire_mud_pct: float, sire_turf_pct: float, 
                                     dam_sire_mud_pct: float, race_condition: str,
                                     race_surface: str) -> float:
    """
    Calculate bonus based on sire surface specialty statistics.
    High %Mud on off-track races, high %Turf on turf races.
    
    Returns: bonus/penalty float
    """
    bonus = 0.0
    
    # Off-track/Muddy condition bonuses
    if race_condition.lower() in ("muddy", "sloppy", "heavy", "wet-fast"):
        if pd.notna(sire_mud_pct):
            if sire_mud_pct >= MODEL_CONFIG.get('mud_specialist_threshold', 25):
                bonus += MODEL_CONFIG.get('mud_specialist_bonus', 0.08)
            elif sire_mud_pct >= MODEL_CONFIG.get('mud_moderate_threshold', 15):
                bonus += MODEL_CONFIG.get('mud_moderate_bonus', 0.04)
        
        if pd.notna(dam_sire_mud_pct):
            if dam_sire_mud_pct >= MODEL_CONFIG.get('mud_specialist_threshold', 25):
                bonus += MODEL_CONFIG.get('mud_specialist_bonus', 0.08) * 0.3
    
    # Turf condition bonus
    if race_surface.lower() == "turf" and pd.notna(sire_turf_pct):
        if sire_turf_pct >= MODEL_CONFIG.get('turf_specialist_threshold', 30):
            bonus += MODEL_CONFIG.get('turf_specialist_bonus', 0.08)
        elif sire_turf_pct >= MODEL_CONFIG.get('turf_moderate_threshold', 20):
            bonus += MODEL_CONFIG.get('turf_moderate_bonus', 0.04)
    
    return float(np.clip(bonus, -0.12, 0.12))

def calculate_awd_mismatch_penalty(sire_awd: float, dam_sire_awd: float, 
                                   race_distance: str) -> float:
    """
    Calculate penalty for Average Winning Distance (AWD) mismatch.
    Horse with shorter AWD may struggle at longer distances and vice versa.
    
    Example: Race is 5.5f, but Sire AWD is 7.5f = 2.0f mismatch = significant penalty
    
    Returns: penalty float (typically negative)
    """
    penalty = 0.0
    
    try:
        # Parse race distance in furlongs
        race_dist_match = re.search(r'(\d+\.?\d*)\s*(?:Furlongs?|f)?', str(race_distance))
        if not race_dist_match:
            return 0.0
        
        race_dist_f = float(race_dist_match.group(1))
        
        # Sire AWD mismatch
        if pd.notna(sire_awd):
            awd_diff = abs(race_dist_f - sire_awd)
            
            # Penalty scales with mismatch magnitude
            # 0.5f mismatch = small penalty (-0.02)
            # 1.0f mismatch = moderate penalty (-0.04)
            # 2.0f+ mismatch = large penalty (-0.08)
            if awd_diff >= 2.0:
                penalty += MODEL_CONFIG.get('awd_large_mismatch_penalty', -0.08)
            elif awd_diff >= 1.0:
                penalty += MODEL_CONFIG.get('awd_moderate_mismatch_penalty', -0.04)
            elif awd_diff >= 0.5:
                penalty += MODEL_CONFIG.get('awd_small_mismatch_penalty', -0.02)
        
        # Dam-Sire AWD (lesser weight, typically 30% of sire impact)
        if pd.notna(dam_sire_awd):
            ds_awd_diff = abs(race_dist_f - dam_sire_awd)
            
            if ds_awd_diff >= 2.0:
                penalty += MODEL_CONFIG.get('awd_large_mismatch_penalty', -0.08) * 0.3
            elif ds_awd_diff >= 1.0:
                penalty += MODEL_CONFIG.get('awd_moderate_mismatch_penalty', -0.04) * 0.3
            elif ds_awd_diff >= 0.5:
                penalty += MODEL_CONFIG.get('awd_small_mismatch_penalty', -0.02) * 0.3
    except:
        pass
    
    return float(np.clip(penalty, -0.15, 0.0))


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

def parse_equip_lasix(block: str) -> Tuple[str, str]:
    # Blinkers: Look for B or b in equipment line (BRIS standard)
    equip_line = re.search(r"(?i)Equipment.*?([A-Za-z]+)", block or "")
    blink = "on" if equip_line and "B" in equip_line.group(1).upper() else "off"
    if re.search(r"Blinkers\s+On", block or "", re.I): blink = "on"
    if re.search(r"Blinkers\s+Off", block or "", re.I): blink = "off"
    
    # Lasix: L = first, L# = repeat, no L = off
    lasix = "off"
    if re.search(r"\bL\b", block or ""): lasix = "first"
    elif re.search(r"\bL\d+\b", block or ""): lasix = "repeat"
    elif re.search(r"Lasix", block or "", re.I): lasix = "on"
    
    return blink, lasix

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

def parse_all_angles(block: str) -> dict:
    """Parse BRISNET Ultimate PP format for advanced handicapping angles"""
    # Prime Power: 101.5 (4th)
    prime_match = re.search(r"Prime Power:\s*(\d+\.?\d*)", block or "")
    prime_val = float(prime_match.group(1)) if prime_match else np.nan
    
    # Extract E2/LP (late pace) values from past performance lines
    # Format: "21Sep25Mnr® 5½ ft :22© :46« :59« 1:06« ¦ ¨§¯ ™C4000 ¨§® 87 72/ 74 +5 +1 56"
    # The pattern is: E1 E2/ LP where LP is after the slash
    lp_values = []
    for m in re.finditer(r"(?m)^\d{2}[A-Za-z]{3}\d{2}.*?\s+(\d{2,3})\s+(\d{2,3})/\s*(\d{2,3})", block or ""):
        lp_values.append(int(m.group(3)))  # LP is the 3rd capture group
    
    # Extract fractional positions (1C, 2C columns) from past performances
    # Format: "56 4 6 6ªƒ 8 8¨§ 6¨§" where positions are like "6ªƒ" (6 with superscript)
    frac_positions = []
    for m in re.finditer(r"(?m)^\d{2}[A-Za-z]{3}\d{2}.*?\s+\d+\s+\d+\s+(\d+)[ªƒ²³¨«¬©°±´‚]*\s+(\d+)[ªƒ²³¨«¬©°±´‚]*", block or ""):
        try:
            pos1 = int(m.group(1))
            pos2 = int(m.group(2))
            frac_positions.append((pos1, pos2, (pos1+pos2)//2))  # Store as tuple
        except:
            pass
    
    # Trainer win% from: "Trnr: Cluley Denis (120 23-17-14 19%)"
    trainer_match = re.search(r"Trnr:.*?\([^\)]+\s+(\d+)%\)", block or "")
    trainer_win = float(trainer_match.group(1)) if trainer_match else 0
    
    # Jockey from QuickPlay comments or race lines
    jockey_match = re.search(r"(?m)^\d{2}[A-Za-z]{3}\d{2}.*?([A-Z][a-z]+[A-Z][a-z]*?\d*[ª©§¨]*)\s+(?:Lb?f?|L)\s+[\d\.*]+", block or "")
    jockey_name = jockey_match.group(1) if jockey_match else "NA"
    
    # Days since last race - calculate from most recent race date
    layoff = 0
    recent_race = re.search(r"(\d{2})[A-Za-z]{3}(\d{2})", block or "")
    if recent_race:
        try:
            day = int(recent_race.group(1))
            year = int(recent_race.group(2)) + 2000
            # Simplified: assume recent if within 90 days
            layoff = 30  # Default placeholder
        except:
            pass
    
    # Bullet works (marked with "B" or "Bg" in workout lines)
    bullets = len(re.findall(r"(?m)^\d{2}[A-Za-z]{3}.*?\sB(?:g|\b)", block or ""))
    
    # Equipment changes from QuickPlay comments
    equip = ""
    if re.search(r"(?i)blinkers?\s+on", block or ""):
        equip = "Blinkers On"
    elif re.search(r"(?i)lasix\s+off", block or ""):
        equip = "Lasix Off"
    elif re.search(r"(?i)front\s+bandages", block or ""):
        equip = "Front Bandages On"
    
    # Dam's Sire stats: "Dam'sSire: AWD 6.1 18%Mud"
    dam_sire_match = re.search(r"Dam'sSire:\s*AWD\s*([\d.]+)\s*(\d+)%Mud", block or "")
    dam_sire = (float(dam_sire_match.group(2)), float(dam_sire_match.group(1))) if dam_sire_match else (0, 0)
    
    # Sire mud stats: "Sire Stats: AWD 7.5 13%Mud"
    sire_mud_match = re.search(r"Sire Stats:\s*AWD\s*[\d.]+\s*(\d+)%Mud", block or "")
    sire_mud = float(sire_mud_match.group(1)) if sire_mud_match else 0
    
    # Pattern recognition from QuickPlay comments (× and ñ markers)
    patterns = []
    if re.search(r"(?i)ñ.*drops?\s+in\s+class", block or ""):
        patterns.append(5.0)  # Class drop bonus
    if re.search(r"(?i)ñ.*first\s+time", block or ""):
        patterns.append(3.0)
    if re.search(r"(?i)ñ.*high.*speed", block or ""):
        patterns.append(2.0)
    
    # Trip comments (trouble in running)
    trouble_phrases = ["bumped", "steadied", "blocked", "altered course", "hung", "weakened", "off slow", "wide"]
    trips = min(sum(1 for phrase in trouble_phrases if phrase in block.lower()), 4)
    
    # Owner ROI (not commonly in PPs, default to 0)
    owner_roi = 0
    
    d = {
        "prime": prime_val,
        "lp": lp_values[:10],
        "frac": frac_positions[:8],
        "trainer_win": trainer_win,
        "jockey": jockey_name,
        "layoff": layoff,
        "bullets": bullets,
        "equip": equip,
        "dam_sire": dam_sire,
        "sire_mud": sire_mud,
        "patterns": patterns,
        "trips": trips,
        "owner_roi": owner_roi,
        "bounce": False  # Calculated in apex_enhance where figs_per_horse is available
    }
    return d

def parse_trainer_intent(block: str) -> dict:
    """Parse trainer intent signals from BRISNET Ultimate PPs"""
    # Class drop from QuickPlay comments: "ñ Drops in class today"
    class_drop = 0
    if re.search(r"(?i)ñ.*drops?\s+in\s+class", block or ""):
        class_drop = 50  # Assume significant drop if noted
    
    # Jockey switch - compare recent jockeys
    recent_jockeys = re.findall(r"(?m)^\d{2}[A-Za-z]{3}\d{2}.*?([A-Z][a-z]+[A-Z][a-z]*?\d*)[ª©§¨]*\s+(?:Lb?f?|L)", block or "")
    jky_switch = len(set(recent_jockeys[:3])) > 1 if len(recent_jockeys) >= 2 else False
    
    # Equipment change: "ñ" indicates positive, check for blinkers
    equip_change = "none"
    if re.search(r"(?i)blinkers?\s+on", block or ""):
        equip_change = "blink_on"
    elif re.search(r"(?i)blinkers?\s+off", block or ""):
        equip_change = "blink_off"
    
    # Layoff works: count bullet workouts (marked with "B")
    # Format: "28Oct Mnr 4f ft :47ª B"
    workout_matches = re.findall(r"(?m)^\d{2}[A-Za-z]{3}.*?\d+f.*?:(\d{2}).*?B", block or "")
    layoff_works = sum(1 for time in workout_matches if int(time) <= 50)  # Fast works under 50 seconds
    
    # Shipper: "Previously trained by" or ship indicators
    ship_from = ""
    if re.search(r"Previously trained by", block or ""):
        ship_from = "SHIP"
    shipper_match = re.search(r"(?i)ñ.*shipper", block or "")
    if shipper_match:
        ship_from = "SHIP"
    
    # ROI angles: look for positive trainer stats
    # Format: "JKYw/ Trn L60 31 16% 45% -0.37" - last number is ROI
    roi_total = 0
    for m in re.finditer(r"(?m)^[+]?[\w\s/]+\s+\d+\s+\d+%\s+\d+%\s+([+-]?\d+\.\d+)", block or ""):
        roi_val = float(m.group(1))
        if roi_val > 0:
            roi_total += roi_val
    
    d = {
        "class_drop_pct": class_drop,
        "jky_switch": jky_switch,
        "equip_change": equip_change,
        "layoff_works": layoff_works,
        "ship_from": ship_from,
        "roi_angles": roi_total
    }
    return d

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

def parse_jock_train_for_block(block: str) -> Dict:
    """Parse jockey win%, trainer ROI, and situational stats from BRISNET PP block"""
    out = {"jock_win_pct": np.nan, "jock_roi": np.nan, "trainer_roi": np.nan, "trainer_situational": {}}
    
    # Jockey regex: Jky: (starts wins-2nds-3rds .win $ROI)
    jock_re = re.compile(r'(?mi)Jky:\s*\((\d+)\s*(\d+)-(\d+)-(\d+)\s*\.(\d+)\s*\$\s*([\d.]+)\)')
    m = jock_re.search(block)
    if m:
        out["jock_win_pct"] = float(m.group(5)) / 100
        out["jock_roi"] = float(m.group(6))
    
    # Trainer regex similar
    train_re = re.compile(r'(?mi)Trn:\s*\((\d+)\s*(\d+)-(\d+)-(\d+)\s*\.(\d+)\s*\$\s*([\d.]+)\)')
    m = train_re.search(block)
    if m:
        out["trainer_roi"] = float(m.group(6))
    
    # Situationals: e.g., Shipper 52 15% 35% +0.23
    sit_re = re.compile(r'(?mi)([A-Za-z]+)\s*(\d+)\s*(\d+)%\s*(\d+)%\s*([+-][\d.]+)')
    for m in sit_re.finditer(block):
        cat, starts, win_pct, itm_pct, roi = m.groups()
        out["trainer_situational"][cat] = {"win_pct": int(win_pct), "roi": float(roi)}
    
    # Last race purse for class drop (e.g., Purse $50,000)
    last_purse_re = re.compile(r'(?mi)Purse\s*\$\s*([\d,]+)')
    m = last_purse_re.search(block)  # Assume first match is last race
    last_purse = int(m.group(1).replace(",", "")) if m else np.nan
    out["last_purse"] = last_purse
    
    return out

def parse_jockey_trainer_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses jockey and trainer names from a horse's PP text block.
    
    BRISNET Format:
    - Jockey: "KIMURA KAZUSHI (0 0-0-0 0%)"
    - Trainer: "Trnr: DAmato Philip (0 0-0-0 0%)"
    
    Returns dict with keys: 'jockey', 'trainer'
    """
    result = {
        'jockey': '',
        'trainer': ''
    }
    
    if not block:
        return result
    
    # Parse jockey - appears on a line by itself in ALL CAPS before "Trnr:"
    # Format: "KIMURA KAZUSHI (0 0-0-0 0%)" or "RISPOLI UMBERTO (0 0-0-0 0%)"
    jockey_match = re.search(r'^([A-Z][A-Z\s\'.-]+?)\s*\([\d\s-]+%\)', block, re.MULTILINE)
    if jockey_match:
        jockey_name = jockey_match.group(1).strip()
        # Clean up extra spaces and convert to title case for readability
        jockey_name = ' '.join(jockey_name.split())
        result['jockey'] = jockey_name
        
        if debug:
            print(f"  Jockey found: '{jockey_name}'")
    
    # Parse trainer - appears on line starting with "Trnr:"
    # Format: "Trnr: DAmato Philip (0 0-0-0 0%)"
    trainer_match = re.search(r'Trnr:\s*([A-Za-z][A-Za-z\s,\'.-]+?)\s*\([\d\s-]+%\)', block, re.MULTILINE)
    if trainer_match:
        trainer_name = trainer_match.group(1).strip()
        # Clean up extra spaces
        trainer_name = ' '.join(trainer_name.split())
        result['trainer'] = trainer_name
        
        if debug:
            print(f"  Trainer found: '{trainer_name}'")
    
    return result

def parse_running_style_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses running style from a horse's PP text block header.
    
    BRISNET Format in header line:
    - "1 Omnipontet (S 1)" - S = Sustained
    - "2 Nay V Belle (E/P 3)" - E/P = Early/Presser
    - "4 Queen Maxima (P 1)" - P = Presser
    - "5 Sunglow (E 8)" - E = Early
    - "7 Puro Magic (NA 0)" - NA = Not Available
    
    Returns dict with key: 'running_style'
    """
    result = {
        'running_style': ''
    }
    
    if not block:
        return result
    
    # Parse running style from header - appears in parentheses after horse name
    # Format: "Post# HorseName (STYLE #)"
    # Style can be: E, E/P, P, S, or NA
    style_match = re.search(r'^\s*\d+\s+[A-Za-z\s=\'-]+\s+\(([A-Z/]+)\s+\d+\)', block, re.MULTILINE)
    if style_match:
        running_style = style_match.group(1).strip()
        result['running_style'] = running_style
        
        if debug:
            print(f"  Running Style found: '{running_style}'")
    
    return result

def parse_quickplay_comments_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses QuickPlay handicapping comments from a horse's PP text block.
    
    BRISNET Format:
    - Positive: "ñ 21% trainer: NonGraded Stk"
    - Negative: "× Has not raced for more than 3 months"
    
    Returns dict with keys: 'positive_comments' (list), 'negative_comments' (list)
    """
    result = {
        'positive_comments': [],
        'negative_comments': []
    }
    
    if not block:
        return result
    
    # Parse positive comments - lines starting with ñ
    positive_matches = re.findall(r'^ñ\s*(.+)$', block, re.MULTILINE)
    result['positive_comments'] = [comment.strip() for comment in positive_matches]
    
    # Parse negative comments - lines starting with ×
    negative_matches = re.findall(r'^×\s*(.+)$', block, re.MULTILINE)
    result['negative_comments'] = [comment.strip() for comment in negative_matches]
    
    if debug:
        if result['positive_comments']:
            print(f"  Positive comments ({len(result['positive_comments'])}):")
            for comment in result['positive_comments']:
                print(f"    ñ {comment}")
        if result['negative_comments']:
            print(f"  Negative comments ({len(result['negative_comments'])}):")
            for comment in result['negative_comments']:
                print(f"    × {comment}")
    
    return result

def parse_recent_workout_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses the most recent workout from a horse's PP text block.
    
    BRISNET Format (workout lines appear at bottom of horse block):
    - "25Oct SA 4f ft :47« H 12/62"
    - "18Oct SA 5f ft 1:02© Hg 37/48"
    
    Format: Date Track Distance Surface Time Grade Rank/Total
    
    Returns dict with keys: 'workout_date', 'workout_track', 'workout_distance', 
                           'workout_time', 'workout_rank', 'workout_total'
    """
    result = {
        'workout_date': '',
        'workout_track': '',
        'workout_distance': '',
        'workout_time': '',
        'workout_rank': '',
        'workout_total': ''
    }
    
    if not block:
        return result
    
    # Workout lines typically start with date pattern like "25Oct" or "18Oct"
    # Format: DDMmmYY Track Distance Surface Time Grade Rank/Total
    # Example: "25Oct SA 4f ft :47« H 12/62"
    # Example with bullet: "×26Oct SA 4f ft :47¨ H 1/11" (× char 215 indicates best workout)
    # Look for lines with this pattern after the race data ends
    # Time can include special chars: « (char 171), © (169), ª (170), ¬ (172), ® (174), ¯ (175), ° (176), ¨ (168)
    workout_pattern = r'×?(\d{1,2}[A-Z][a-z]{2})\s+([A-Z][A-Za-z]{1,3})\s+(\d+f?)\s+(?:ft|gd|sy|sl|fm|hy|my|tr\.t|˜)\s+([\d:\.«©ª¬®¯°¨]+)\s+[A-Z]?g?\s+(\d+)/(\d+)'
    
    matches = re.findall(workout_pattern, block, re.MULTILINE)
    
    if matches:
        # Take the first (most recent) workout
        first_workout = matches[0]
        result['workout_date'] = first_workout[0]
        result['workout_track'] = first_workout[1]
        result['workout_distance'] = first_workout[2]
        result['workout_time'] = first_workout[3]
        result['workout_rank'] = first_workout[4]
        result['workout_total'] = first_workout[5]
        
        if debug:
            print(f"  Recent workout: {result['workout_date']} {result['workout_track']} {result['workout_distance']} {result['workout_time']} (#{result['workout_rank']}/{result['workout_total']})")
    
    return result

def parse_prime_power_for_block(block, debug=False):
    """
    Parse Prime Power rating and rank from a BRISNET horse block.
    
    Format: "Prime Power: 131.9 (7th)"
    
    Returns dict with:
    - prime_power: float rating value (e.g., 131.9)
    - prime_power_rank: str rank (e.g., "7th")
    """
    result = {
        'prime_power': None,
        'prime_power_rank': None
    }
    
    # Pattern: "Prime Power: 131.9 (7th)"
    # Captures the decimal number and the rank in parentheses
    prime_power_pattern = r'Prime Power:\s+([\d.]+)\s+\((\d+(?:st|nd|rd|th))\)'
    
    match = re.search(prime_power_pattern, block)
    
    if match:
        try:
            result['prime_power'] = float(match.group(1))
            result['prime_power_rank'] = match.group(2)
            
            if debug:
                print(f"  Prime Power: {result['prime_power']} (rank: {result['prime_power_rank']})")
        except ValueError:
            pass
    
    return result

# BRISNET Speed Figure Regex - Match the number pattern directly
# Format: "| ¨¨¬ OC50k/n1x-N ¨¨ 92 101/ 93 +4 +4 98 2 4©..."
# Key insight: The "pipe" is actually broken bar ¦ (char 166) not | (char 124)!
# After broken bar, we'll eventually see this number sequence:
# RR (2-3 digits) CR/ (2-3 digits with slash) E1 (2-3 digits) +/-# +/-# SPD (2-3 digits) PP (1 digit)
# The post position (PP) that follows SPD is always a SINGLE digit
# So we can use that as confirmation: ...SPD followed by space and single digit
SPEED_FIG_RE = re.compile(
    r"[|\¦I]"  # Pipe separator - match | (124), ¦ (166), or I (73)
    r"(?:[^\d]|\d(?!/\s*\d{2,3}\s+[+-]))*?"  # Skip non-rating content (lazy match)
    r"(\d{2,3})\s+"  # RR (Race Rating) - group 1
    r"(\d{2,3})/?\s+"  # CR (Class Rating) with optional slash - group 2
    r"(\d{2,3})\s+"  # E1 (Early pace) - group 3
    r"[+-]?\d+\s+"  # First position indicator
    r"[+-]?\d+\s+"  # Second position indicator
    r"(\d{2,3})"  # SPD (Speed rating) - group 4
    r"\s+\d(?:\s|©|ª|«|¬)"  # Followed by single-digit post position
)

def parse_speed_figures_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses a horse's PP text block and extracts BRIS speed/pace figures.
    Returns dict with keys: 'SPD', 'E1', 'E2', 'LP', 'RR', 'CR'
    Each contains a list of recent values (most recent first, up to 10 races).
    
    BRISNET Format: "92 101/ 93 +4 +4 98 2 4© 4© 3¯"
    Where: RR=92, CR=101, E1=93, E2=98 (after position indicators)
    SPD is not in this section - may need to extract from elsewhere
    """
    result = {
        'SPD': [],  # Final speed ratings (placeholder - not in this section)
        'E1': [],   # Early pace ratings  
        'E2': [],   # Mid pace ratings (what appears after position indicators)
        'LP': [],   # Late pace ratings (placeholder)
        'RR': [],   # Race ratings (field quality)
        'CR': []    # Class ratings (horse performance)
    }
    
    if not block:
        return result
    
    if debug:
        # Show lines that look like running lines (start with date pattern)
        st.write("**Looking for running lines (lines starting with DDMmmYY):**")
        potential_lines = [line for line in block.split('\n') if re.match(r'^\s*\d{2}[A-Za-z]{3}\d{2}', line)]
        if potential_lines:
            st.code("\n".join(potential_lines[:5]), language="text")  # Show first 5
            # DIAGNOSTIC: Show character codes around where pipe should be
            first_line = potential_lines[0]
            # Look for any character that might be a pipe or vertical bar
            for pipe_char in ['|', '¦', '│', 'I', 'l']:
                pipe_idx = first_line.find(pipe_char)
                if pipe_idx >= 0:
                    st.write(f"**🔍 Found '{pipe_char}' (code {ord(pipe_char)}) at position {pipe_idx}:**")
                    snippet = first_line[max(0, pipe_idx-5):min(pipe_idx+30, len(first_line))]
                    char_info = " ".join([f"{c}({ord(c)})" for c in snippet[:20]])
                    st.caption(f"Chars around it: {char_info}")
                    break
            else:
                st.warning(f"⚠️ No pipe character found! First 60 chars: {first_line[:60]}")
                char_codes = " ".join([f"{c}({ord(c)})" for c in first_line[:60]])
                st.caption(f"Character codes: {char_codes}")
        else:
            st.warning("❌ No lines found starting with date pattern (e.g., '23Sep23')")
    
    matches_found = 0
    for m in SPEED_FIG_RE.finditer(block):
        try:
            matches_found += 1
            rr = int(m.group(1))   # Race Rating
            cr_text = m.group(2).replace('/', '').strip()  # Class Rating (remove slash)
            cr = int(cr_text)
            e1 = int(m.group(3))   # Early pace
            e2 = int(m.group(4))   # What appears after positions (might be E2 or SPD)
            
            if debug:
                st.caption(f"🔍 Match {matches_found}: RR={rr}, CR={cr}, E1={e1}, E2={e2}")
                st.caption(f"    Full match: {m.group(0)[:100]}...")
            
            # Sanity checks for realistic BRIS figures (typically 40-130 range)
            if 40 <= e2 <= 130:
                # Using E2 as SPD for now since true SPD not in regex
                result['SPD'].append(e2)
                result['E2'].append(e2)
            if 40 <= e1 <= 130:
                result['E1'].append(e1)
            if 50 <= rr <= 140:  # RR can be slightly higher
                result['RR'].append(rr)
            if 40 <= cr <= 130:
                result['CR'].append(cr)
                
        except (ValueError, IndexError) as e:
            if debug:
                st.warning(f"⚠️ Failed to parse match {matches_found}: {e}")
            pass  # Ignore if conversion fails
    
    if debug:
        if matches_found == 0:
            st.error(f"❌ No regex matches found in block. First 500 chars:\n{block[:500]}")
        else:
            st.success(f"✅ Found {matches_found} running lines with speed figures!")
    
    # Keep only the most recent 10 races for each metric
    for key in result:
        result[key] = result[key][:10]
    
    return result

def parse_prime_power_for_block(block: str) -> float:
    """Extract Prime Power bonus from BRISNET PP block"""
    prime_re = re.compile(r'(?mi)Prime\s*Power:\s*(\d+\.\d+)')
    m = prime_re.search(block)
    return float(m.group(1)) if m else np.nan

def parse_cr_rr_history(figs_dict: dict) -> dict:
    """
    Calculate CR/RR performance metrics from parsed speed figures.
    Returns dict with: avg_cr, avg_rr, cr_rr_ratio, consistency_score
    """
    out = {"avg_cr": np.nan, "avg_rr": np.nan, "cr_rr_ratio": np.nan, "consistency": 0.0}
    
    try:
        cr_list = figs_dict.get("CR", [])
        rr_list = figs_dict.get("RR", [])
        
        if len(cr_list) > 0 and len(rr_list) > 0:
            # Average CR and RR (most recent 5 races)
            avg_cr = np.mean(cr_list[:5]) if len(cr_list) >= 1 else np.nan
            avg_rr = np.mean(rr_list[:5]) if len(rr_list) >= 1 else np.nan
            
            out["avg_cr"] = avg_cr
            out["avg_rr"] = avg_rr
            
            # CR/RR ratio: how much does horse outperform the field?
            # Ratio > 1.0 means horse performs better than field quality
            # Ratio >= 0.95 = very good (performing close to or above field RR)
            if pd.notna(avg_rr) and avg_rr > 0:
                cr_rr_ratio = avg_cr / avg_rr
                out["cr_rr_ratio"] = cr_rr_ratio
            
            # Consistency score: Do CR performances vary widely or stay consistent?
            # Low std dev = reliable performer
            if len(cr_list) >= 3:
                cr_std = np.std(cr_list[:5])
                # Normalize to 0-1 scale (lower std = higher consistency)
                consistency = max(0.0, 1.0 - (cr_std / 20.0))  # 20 point std dev = 0 consistency
                out["consistency"] = consistency
    except:
        pass
    
    return out

def parse_ep_lp_trip_for_block(block: str) -> Dict:
    """Parse Early Pace/Late Pace and trip excuse comments from BRISNET PP block"""
    out = {"avg_lp": np.nan, "excuse_count": 0}
    
    # EP/LP: Collect last 3 races' LP values
    # Format: CR E1/E2 LP from speed figure lines
    pace_re = re.compile(r'(?mi)(\d+)\s+(\d+)/\s*(\d+)\s+(\d+)')  # CR E1/E2 LP SPD
    lps = []
    for m in pace_re.finditer(block):
        try:
            lp = int(m.group(3))  # LP is group 3
            if 40 < lp < 130:
                lps.append(lp)
        except (ValueError, IndexError):
            pass
    out["avg_lp"] = np.mean(lps[:3]) if lps else np.nan
    
    # Trip notes: Count keywords in comments for excuse tracking
    trip_keywords = ["blocked", "wide", "steadied", "checked", "bumped", "altered course"]
    comment_text = block.lower()
    excuse_count = sum(comment_text.count(k) for k in trip_keywords)
    out["excuse_count"] = min(excuse_count, 3)  # Cap at 3 for last 3 races
    
    return out

def parse_expanded_ped_work_layoff(block: str) -> Dict:
    """Extract surface win%, bullet workouts, and layoff days from PP block"""
    out = {"turf_win_pct": np.nan, "mud_win_pct": np.nan, "bullet_count": 0, "days_off": np.nan}
    
    # Pedigree surface: Turf|Wet|Fast (rating) starts wins-2nds-3rds $earnings
    surf_re = re.compile(r'(?mi)(Turf|Wet|Fast)\s*\((\d+)\)\s*(\d+)\s+(\d+)\s*-\s*(\d+)\s*-\s*(\d+)\s*\$\s*([\d,]+)')
    for m in surf_re.finditer(block or ""):
        try:
            surf, rating, starts, wins = m.group(1), m.group(2), m.group(3), m.group(4)
            if int(starts) > 0:
                win_pct = int(wins) / int(starts)
                if surf == "Turf":
                    out["turf_win_pct"] = win_pct
                elif surf in ("Wet", "Fast"):
                    out["mud_win_pct"] = win_pct
        except:
            pass
    
    # Workouts: Count bullet workouts (marked with "B" in workout lines)
    work_re = re.compile(r'(?mi)^\d{2}[A-Za-z]{3}\d{2}.*?\s+B(?:g)?\s')
    out["bullet_count"] = len(work_re.findall(block or ""))
    
    # Layoff: Days since last race via angle line format or "days Away"
    lay_re = re.compile(r'(?mi)(\d+)\s*days?\s*Away')
    m = lay_re.search(block or "")
    if m:
        out["days_off"] = int(m.group(1))
    
    return out

def parse_bris_pedigree_ratings(block: str) -> dict:
    """
    Extract BRISNET Pedigree Ratings (Fast/Off/Distance/Turf) from PP block.
    Format: "Pedigree: Fast 89, Off 84, Distance 92, Turf 88"
    Returns dict with keys: fast_ped, off_ped, distance_ped, turf_ped
    """
    out = {"fast_ped": np.nan, "off_ped": np.nan, "distance_ped": np.nan, "turf_ped": np.nan}
    
    if not block:
        return out
    
    # Parse BRISNET pedigree ratings: Fast, Off, Distance, Turf
    # Format variations:
    # "Pedigree: Fast 89, Off 84, Distance 92, Turf 88"
    # "Fast Ped 89  Off Ped 84  Dist Ped 92  Turf Ped 88"
    
    fast_match = re.search(r'(?mi)(?:Fast\s+(?:Ped)?|Pedigree:?\s*Fast)\s*(\d+)', block)
    if fast_match:
        out["fast_ped"] = float(fast_match.group(1))
    
    off_match = re.search(r'(?mi)(?:Off\s+(?:Ped)?|Off\s+Surface)\s*(\d+)', block)
    if off_match:
        out["off_ped"] = float(off_match.group(1))
    
    dist_match = re.search(r'(?mi)(?:Distance\s+(?:Ped)?|Dist(?:ance)?\s+Ped)\s*(\d+)', block)
    if dist_match:
        out["distance_ped"] = float(dist_match.group(1))
    
    turf_match = re.search(r'(?mi)(?:Turf\s+(?:Ped)?|Turf\s+Pedigree)\s*(\d+)', block)
    if turf_match:
        out["turf_ped"] = float(turf_match.group(1))
    
    return out

def parse_surface_specific_record(block: str) -> dict:
    """
    Extract surface-specific record (dirt/turf/off-track) from PP history.
    Returns dict with keys: dirt_record, turf_record, off_track_record (each as dict with 'wins', 'itmr', 'starts')
    """
    out = {
        "dirt_record": {"wins": 0, "itmr": 0, "starts": 0, "win_pct": 0.0},
        "turf_record": {"wins": 0, "itmr": 0, "starts": 0, "win_pct": 0.0},
        "off_track_record": {"wins": 0, "itmr": 0, "starts": 0, "win_pct": 0.0},
    }
    
    if not block:
        return out
    
    try:
        # Parse surface records from summary lines (typical BRISNET format)
        # Format: "On Dirt: 5 wins-8 2nd-4 3rd (23 starts) or 22% Win, 57% ITM"
        # Look for surface-specific lines with win/ITM/starts info
        
        dirt_match = re.search(r'(?mi)(?:on\s+)?dirt:?\s*(\d+)\s+(?:wins?|w)\s*-?\s*(\d+)\s+(?:2nds?|2)\s*-?\s*(\d+)\s+(?:3rds?|3)\s*\((\d+)\s+(?:starts?|st)\)', block)
        if dirt_match:
            wins, seconds, thirds, starts = map(int, dirt_match.groups())
            itmr = wins + seconds + thirds
            out["dirt_record"] = {"wins": wins, "itmr": itmr, "starts": starts, "win_pct": wins/starts if starts > 0 else 0.0}
        
        turf_match = re.search(r'(?mi)(?:on\s+)?turf:?\s*(\d+)\s+(?:wins?|w)\s*-?\s*(\d+)\s+(?:2nds?|2)\s*-?\s*(\d+)\s+(?:3rds?|3)\s*\((\d+)\s+(?:starts?|st)\)', block)
        if turf_match:
            wins, seconds, thirds, starts = map(int, turf_match.groups())
            itmr = wins + seconds + thirds
            out["turf_record"] = {"wins": wins, "itmr": itmr, "starts": starts, "win_pct": wins/starts if starts > 0 else 0.0}
        
        off_match = re.search(r'(?mi)(?:on\s+)?(?:off-?track|muddy|sloppy):?\s*(\d+)\s+(?:wins?|w)\s*-?\s*(\d+)\s+(?:2nds?|2)\s*-?\s*(\d+)\s+(?:3rds?|3)\s*\((\d+)\s+(?:starts?|st)\)', block)
        if off_match:
            wins, seconds, thirds, starts = map(int, off_match.groups())
            itmr = wins + seconds + thirds
            out["off_track_record"] = {"wins": wins, "itmr": itmr, "starts": starts, "win_pct": wins/starts if starts > 0 else 0.0}
    except:
        pass
    
    return out

def parse_pedigree_spi(block: str) -> dict:
    """
    Extract Sire Production Index (SPI) from pedigree section.
    Format: "Sire: {SireNameHere} {SPI_value}"
    Example: "Sire: Way of Appeal SPI .36"
    Returns dict with key 'spi' containing float value or NaN
    """
    out = {"spi": np.nan, "dam_sire_spi": np.nan}
    
    if not block:
        return out
    
    try:
        # Look for SPI (Sire Production Index) - typically decimal 0.3-0.7 range
        # Format variations:
        # "Sire: {name} SPI .36"
        # "Sire Production Index: 0.36"
        # "SPI .36"
        spi_match = re.search(r'(?mi)(?:Sire[^:]*:|SPI)\s*\.?(\d\.\d+|\d+\.\d+)', block)
        if spi_match:
            spi_text = spi_match.group(1)
            # Handle cases like ".36" -> 0.36 or "0.36"
            spi_val = float(spi_text) if '.' in spi_text else float(spi_text) / 100.0
            out["spi"] = spi_val
        
        # Dam-Sire SPI (production index of dam's sire)
        dam_sire_match = re.search(r'(?mi)Dam[\s-]*Sire.*?SPI\s*\.?(\d\.\d+|\d+\.\d+)', block)
        if dam_sire_match:
            dam_sire_text = dam_sire_match.group(1)
            dam_sire_val = float(dam_sire_text) if '.' in dam_sire_text else float(dam_sire_text) / 100.0
            out["dam_sire_spi"] = dam_sire_val
    except:
        pass
    
    return out

def parse_pedigree_surface_stats(block: str) -> dict:
    """
    Extract Sire surface specialty statistics (%Mud, %Turf, Dam-Sire %Mud, etc.)
    Format: "Sire: {name} Mud 23% Turf 19%"
    Returns dict with keys: sire_mud_pct, sire_turf_pct, dam_sire_mud_pct
    """
    out = {"sire_mud_pct": np.nan, "sire_turf_pct": np.nan, "dam_sire_mud_pct": np.nan}
    
    if not block:
        return out
    
    try:
        # Extract Sire % statistics (Mud, Turf)
        # Format: "Mud 23%" "Turf 19%"
        sire_section = re.search(r'(?mi)Sire[^D]*?(?=Dam|$)', block)
        if sire_section:
            section_text = sire_section.group(0)
            
            # Sire Mud %
            mud_match = re.search(r'(?mi)Mud\s+(\d+)%', section_text)
            if mud_match:
                out["sire_mud_pct"] = float(mud_match.group(1))
            
            # Sire Turf %
            turf_match = re.search(r'(?mi)Turf\s+(\d+)%', section_text)
            if turf_match:
                out["sire_turf_pct"] = float(turf_match.group(1))
        
        # Extract Dam-Sire % statistics (primarily Mud)
        dam_sire_section = re.search(r'(?mi)Dam[\s-]*Sire[^\\n]*', block)
        if dam_sire_section:
            ds_text = dam_sire_section.group(0)
            
            # Dam-Sire Mud %
            ds_mud_match = re.search(r'(?mi)Mud\s+(\d+)%', ds_text)
            if ds_mud_match:
                out["dam_sire_mud_pct"] = float(ds_mud_match.group(1))
    except:
        pass
    
    return out

def parse_awd_analysis(block: str) -> dict:
    """
    Extract Sire and Dam-Sire Average Winning Distance (AWD) from pedigree section.
    Format: "Sire: {name} AWD 6.2f"
    Returns dict with keys: sire_awd, dam_sire_awd (both in furlongs as float)
    """
    out = {"sire_awd": np.nan, "dam_sire_awd": np.nan}
    
    if not block:
        return out
    
    try:
        # Extract Sire AWD (Average Winning Distance)
        # Format: "AWD 6.2f" or "AWD 7.5f"
        sire_section = re.search(r'(?mi)Sire[^D]*?(?=Dam|$)', block)
        if sire_section:
            section_text = sire_section.group(0)
            awd_match = re.search(r'(?mi)AWD\s+(\d+\.?\d*)f', section_text)
            if awd_match:
                out["sire_awd"] = float(awd_match.group(1))
        
        # Extract Dam-Sire AWD
        dam_sire_section = re.search(r'(?mi)Dam[\s-]*Sire[^\\n]*', block)
        if dam_sire_section:
            ds_text = dam_sire_section.group(0)
            ds_awd_match = re.search(r'(?mi)AWD\s+(\d+\.?\d*)f', ds_text)
            if ds_awd_match:
                out["dam_sire_awd"] = float(ds_awd_match.group(1))
    except:
        pass
    
    return out

def parse_track_bias_impact_values(block: str) -> dict:
    """
    Extract Track Bias Impact Values from the Track Bias section.
    Format: "Running Style Impact: E 1.30, E/P 1.20, P 0.95, S 0.80"
    Format: "Post Position Impact: Rail 1.41, 1-3 1.10, 4-7 1.05, 8+ 0.95"
    Returns dict with structure:
    {
        'running_style': {'E': 1.30, 'E/P': 1.20, 'P': 0.95, 'S': 0.80},
        'post_position': {'rail': 1.41, 'inner': 1.10, 'mid': 1.05, 'outside': 0.95}
    }
    """
    out = {
        'running_style': {'E': np.nan, 'E/P': np.nan, 'P': np.nan, 'S': np.nan},
        'post_position': {'rail': np.nan, 'inner': np.nan, 'mid': np.nan, 'outside': np.nan}
    }
    
    if not block:
        return out
    
    try:
        # Extract Running Style Impact Values
        # Pattern: "E 1.30" or "E/P 1.20" etc (as part of line like "E 1.30, E/P 1.20, P 0.95, S 0.80")
        rs_section = re.search(r'(?mi)Running\s+Style.*?Impact', block)
        if rs_section:
            # Look for values after the section header
            section_end = block.find('\n', rs_section.end())
            if section_end == -1:
                section_end = len(block)
            section_text = block[rs_section.end():section_end]
            
            e_match = re.search(r'(?mi)\bE\s+(\d\.?\d*)', section_text)
            if e_match:
                out['running_style']['E'] = float(e_match.group(1))
            
            ep_match = re.search(r'(?mi)E/P\s+(\d\.?\d*)', section_text)
            if ep_match:
                out['running_style']['E/P'] = float(ep_match.group(1))
            
            p_match = re.search(r'(?mi)\bP\s+(\d\.?\d*)', section_text)
            if p_match:
                out['running_style']['P'] = float(p_match.group(1))
            
            s_match = re.search(r'(?mi)\bS\s+(\d\.?\d*)', section_text)
            if s_match:
                out['running_style']['S'] = float(s_match.group(1))
        
        # Extract Post Position Impact Values
        # Pattern: "Rail 1.41" or "1-3 1.10" or "4-7 1.05" or "8+ 0.95"
        pp_section = re.search(r'(?mi)Post.*?Position.*?Impact', block)
        if pp_section:
            section_end = block.find('\n', pp_section.end())
            if section_end == -1:
                section_end = len(block)
            section_text = block[pp_section.end():section_end]
            
            rail_match = re.search(r'(?mi)Rail\s+(\d\.?\d*)', section_text)
            if rail_match:
                out['post_position']['rail'] = float(rail_match.group(1))
            
            inner_match = re.search(r'(?mi)(?:1-3|Inner)\s+(\d\.?\d*)', section_text)
            if inner_match:
                out['post_position']['inner'] = float(inner_match.group(1))
            
            mid_match = re.search(r'(?mi)(?:4-7|Mid)\s+(\d\.?\d*)', section_text)
            if mid_match:
                out['post_position']['mid'] = float(mid_match.group(1))
            
            outside_match = re.search(r'(?mi)(?:8\+|Outside)\s+(\d\.?\d*)', section_text)
            if outside_match:
                out['post_position']['outside'] = float(outside_match.group(1))
    except:
        pass
    
    return out

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

def calculate_figure_trend(figs: List[float]) -> Tuple[float, str]:
    """
    Analyze figure trend: uptrend (+bonus), downtrend (-bonus), flat (0).
    Returns (trend_bonus, trend_label)
    """
    if not figs or len(figs) < 2:
        return 0.0, "insufficient"
    
    recent_3 = figs[:3]  # Most recent 3 figs
    
    # Uptrend: each fig higher than previous
    if len(recent_3) >= 2 and recent_3[0] > recent_3[1]:
        if len(recent_3) >= 3 and recent_3[1] > recent_3[2]:
            return 0.11, "strong_uptrend"
        return 0.07, "uptrend"
    
    # Downtrend: each fig lower than previous
    if len(recent_3) >= 2 and recent_3[0] < recent_3[1]:
        if len(recent_3) >= 3 and recent_3[1] < recent_3[2]:
            return -0.11, "strong_downtrend"
        return -0.07, "downtrend"
    
    return 0.0, "flat"

def calculate_distance_record(block: str, target_distance: str) -> Tuple[float, int, int]:
    """
    Extract win% at specific distance from PP block.
    Returns (win_pct, wins_at_distance, starts_at_distance)
    """
    if not block:
        return np.nan, 0, 0
    
    try:
        # Parse distance from each race line and track performance
        # Format: "23Sep25Mnr 5½ ft :22© :46« :59« 1:06« ¦ ¨§¯ ™C4000 ¨§® 87 72/ 74 +5 +1 56 1 1©"
        race_lines = re.findall(r'(?m)^\d{2}[A-Za-z]{3}\d{2}[A-Za-z]{3}\s+([\d½]+)', block or "")
        
        # Count races at this distance
        distance_races = 0
        distance_wins = 0
        
        for line in re.finditer(r'(?m)^\d{2}[A-Za-z]{3}\d{2}[A-Za-z]{3}\s+([\d½]+)[^0-9]*(\d+)\s+(\d+)/\s*(\d+)', block or ""):
            race_dist = line.group(1)
            position = int(line.group(3))  # Finishing position
            
            # Normalize distance
            if "½" in str(target_distance):
                target_norm = float(target_distance.replace("½", ".5"))
            else:
                target_norm = float(target_distance) if target_distance else 0
            
            if "½" in race_dist:
                race_dist_norm = float(race_dist.replace("½", ".5"))
            else:
                race_dist_norm = float(race_dist) if race_dist else 0
            
            # Check if within 0.1 furlongs (close match)
            if abs(race_dist_norm - target_norm) < 0.2:
                distance_races += 1
                if position == 1:
                    distance_wins += 1
        
        if distance_races > 0:
            win_pct = distance_wins / distance_races
            return win_pct, distance_wins, distance_races
    except:
        pass
    
    return np.nan, 0, 0

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

    # --- MERGE SPEED FIGURES FIRST ---
    if figs_df.empty or "AvgTop2" not in figs_df.columns:
        # No figures were parsed or dataframe is empty
        st.caption("No speed figures parsed. R_ENHANCE_ADJ set to 0.")
        df["AvgTop2"] = MODEL_CONFIG['first_timer_fig_default']
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

    # --- ADD PRIME POWER AND APEX ENHANCEMENT (after AvgTop2 is available) ---
    df["Prime"] = df["Horse"].map(lambda h: all_angles_per_horse.get(h, {}).get("prime", np.nan))
    st.caption(f"🔍 Added Prime column: {df['Prime'].notna().sum()} horses with values, sample={df['Prime'].head(3).tolist()}")
    df = apex_enhance(df)  # apex_enhance already adds APEX to R internally
    st.caption(f"🔍 After apex_enhance: APEX column exists={('APEX' in df.columns)}, R column sample={df['R'].head(3).tolist()}")

    # Clean up the temporary AvgTop2 column if it exists
    if "AvgTop2" in df.columns:
        df.drop(columns=["AvgTop2"], inplace=True)

    # --- END SPEED FIGURE LOGIC ---
    # NOTE: R already includes R_ENHANCE_ADJ + APEX from apex_enhance()
    # Clean up the helper column
    if "R_ENHANCE_ADJ" in df.columns:
        df.drop(columns=["R_ENHANCE_ADJ"], inplace=True)
    
    return df

# ---------- ML Adjustment (RandomForest) ----------
def ml_adjust(df: pd.DataFrame, trainer_intent_data: Dict[str, dict]) -> pd.DataFrame:
    """
    Applies machine learning adjustment using RandomForest to refine ratings.
    Incorporates trainer intent features (class drop %, ROI angles) along with
    existing rating components.
    
    Returns df with additional ML_ADJ column added to R rating.
    """
    if not SKLEARN_AVAILABLE:
        st.caption("⚠️ ML adjustment skipped - scikit-learn not available")
        return df
    
    if df is None or df.empty or len(df) < 3:
        return df
    
    df = df.copy()
    
    # Build feature matrix
    features = pd.DataFrame(index=df.index)
    
    # Base rating components
    features["R"] = df.get("R", 0)
    features["Cclass"] = df.get("Cclass", 0)
    features["Cstyle"] = df.get("Cstyle", 0)
    features["Cpost"] = df.get("Cpost", 0)
    features["Cpace"] = df.get("Cpace", 0)
    features["Atrack"] = df.get("Atrack", 0)
    features["Quirin"] = df.get("Quirin", 0).fillna(0)
    features["LastFig"] = df.get("LastFig", 0).fillna(0)
    
    # APEX component
    features["APEX"] = df.get("APEX", 0).fillna(0)
    
    # Add trainer intent features
    features["Intent_ClassDrop"] = df.index.map(lambda idx: trainer_intent_data.get(df.loc[idx, "Horse"], {}).get("class_drop_pct", 0))
    features["Intent_ROI"] = df.index.map(lambda idx: trainer_intent_data.get(df.loc[idx, "Horse"], {}).get("roi_angles", 0))
    
    # Fill any remaining NaNs
    features = features.fillna(0)

    # Create synthetic target (use current R as proxy for training)
    # In production, you'd train on historical race results
    y = df["R"].fillna(df["R"].mean()).values

    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(features)    # Train RandomForest
    rf = RandomForestRegressor(
        n_estimators=50,
        max_depth=5,
        min_samples_split=2,
        random_state=42,
        n_jobs=-1
    )
    
    try:
        rf.fit(X_scaled, y)
        predictions = rf.predict(X_scaled)
        
        # ML adjustment is the difference between prediction and current rating
        # Scale it down to avoid over-correction
        ml_adjustment = (predictions - y) * 0.3  # 30% weight on ML delta
        
        df["ML_ADJ"] = ml_adjustment.round(3)
        df["R"] = (df["R"] + df["ML_ADJ"]).round(2)
        
        # Feature importance (top 5)
        importance = pd.DataFrame({
            'feature': features.columns,
            'importance': rf.feature_importances_
        }).sort_values('importance', ascending=False).head(5)
        
        st.caption(f"✅ ML adjustment applied. Top features: {', '.join(importance['feature'].tolist())}")
        
    except Exception as e:
        st.warning(f"⚠️ ML adjustment failed: {e}")
        df["ML_ADJ"] = 0.0
    
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

# ---------- Safe market/ML adjust helpers (NaN-proof) ----------
def _implied_prob_from_str(odds_str: str) -> float:
    dec = str_to_decimal_odds(str(odds_str).strip())
    if dec is None or dec <= 0: return np.nan
    return 1.0 / dec

def _safe_normalize_prob_map(prob_map: Dict[str, float]) -> Dict[str, float]:
    keys = list(prob_map.keys())
    vals = np.array([max(float(v), 0.0) for v in prob_map.values()], dtype=float)
    s = float(vals.sum())
    if s <= 0:
        n = max(len(keys), 1)
        return {k: 1.0/n for k in keys}
    return {k: float(v)/s for k, v in zip(keys, vals)}

def _build_market_table(df_field: pd.DataFrame, fair_probs: Dict[str, float]) -> pd.DataFrame:
    rows = []
    for _, r in df_field.iterrows():
        h = str(r.get("Horse","")).strip()
        if not h: continue
        ml_s   = str(r.get("ML","") or "").strip()
        live_s = str(r.get("Live Odds","") or "").strip()
        rows.append({
            "Horse": h,
            "fair_prob": float(fair_probs.get(h, 0.0)),
            "ml_prob": _implied_prob_from_str(ml_s) if ml_s else np.nan,
            "live_prob": _implied_prob_from_str(live_s) if live_s else np.nan
        })
    return pd.DataFrame(rows)

def _alpha_from_race_type(race_type: str) -> float:
    base = base_class_bias.get((race_type or "").strip().lower(), 1.02)  # 0.90..1.15 constants
    alpha = (base - 0.90) / (1.15 - 0.90) * (0.55 - 0.15) + 0.15
    return float(np.clip(alpha, 0.15, 0.55))

def adjust_probs_with_market(df_market: pd.DataFrame, race_type: str) -> Dict[str, float]:
    if df_market is None or df_market.empty:
        return {}
    fair = {row["Horse"]: float(row.get("fair_prob", 0.0)) for _, row in df_market.iterrows()}
    obs  = {}
    for _, row in df_market.iterrows():
        h = row["Horse"]; live_p = row.get("live_prob", np.nan); ml_p = row.get("ml_prob", np.nan)
        obs[h] = live_p if (live_p == live_p) else (ml_p if (ml_p == ml_p) else np.nan)
    if all((v != v) for v in obs.values()):  # all NaN
        return _safe_normalize_prob_map(fair)
    obs_filled = {h: (fair[h] if (v != v) else float(v)) for h, v in obs.items()}
    fair_n = _safe_normalize_prob_map(fair)
    obs_n  = _safe_normalize_prob_map(obs_filled)
    alpha  = _alpha_from_race_type(race_type)
    blended = {h: (1.0 - alpha)*fair_n[h] + alpha*obs_n[h] for h in fair_n}
    return _safe_normalize_prob_map(blended)

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

# Define today's par figure based on race type (rough estimates)
today_par = {
    "maiden special weight": 75, "maiden claiming": 65, "claiming": 70,
    "starter allowance": 75, "allowance": 80, "allowance optional claiming": 85,
    "listed stakes": 90, "graded stakes": 95, "grade 3": 95, "grade 2": 100, "grade 1": 105
}.get(race_type_detected, 75)

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
ep_lp_trip_per_horse: Dict[str, Dict] = {}
expanded_ped_per_horse: Dict[str, Dict] = {}
figs_per_horse: Dict[str, dict] = {}  # Changed from List[int] to dict
jockey_trainer_per_horse: Dict[str, dict] = {}
jock_train_per_horse: Dict[str, Dict] = {}
running_style_per_horse: Dict[str, dict] = {}
quickplay_per_horse: Dict[str, dict] = {}
workout_per_horse: Dict[str, dict] = {}
prime_power_per_horse: Dict[str, dict] = {}
equip_lasix_per_horse: Dict[str, Tuple[str, str]] = {}
all_angles_per_horse: Dict[str, dict] = {}
trainer_intent_per_horse: Dict[str, dict] = {}
bris_ped_ratings_per_horse: Dict[str, dict] = {}  # NEW: BRISNET Pedigree Ratings
surface_record_per_horse: Dict[str, dict] = {}  # NEW: Surface-specific records
cr_rr_per_horse: Dict[str, dict] = {}  # NEW: CR/RR performance ratio history
pedigree_spi_per_horse: Dict[str, dict] = {}  # NEW: Sire Production Index
pedigree_surface_stats_per_horse: Dict[str, dict] = {}  # NEW: Sire surface specialty stats
awd_analysis_per_horse: Dict[str, dict] = {}  # NEW: Average Winning Distance analysis
track_bias_impact_per_horse: Dict[str, dict] = {}  # NEW: Track Bias Impact Values
blocks: Dict[str, str] = {}

for _post, name, block in split_into_horse_chunks(pp_text):
    if name in df_editor["Horse"].values:
        blocks[name] = block
        angles_per_horse[name] = parse_angles_for_block(block)
        pedigree_per_horse[name] = parse_pedigree_snips(block)
        ep_lp_trip_per_horse[name] = parse_ep_lp_trip_for_block(block)
        expanded_ped_per_horse[name] = parse_expanded_ped_work_layoff(block)
        bris_ped_ratings_per_horse[name] = parse_bris_pedigree_ratings(block)  # NEW
        surface_record_per_horse[name] = parse_surface_specific_record(block)  # NEW
        pedigree_spi_per_horse[name] = parse_pedigree_spi(block)  # NEW: SPI
        pedigree_surface_stats_per_horse[name] = parse_pedigree_surface_stats(block)  # NEW: Mud/Turf %
        awd_analysis_per_horse[name] = parse_awd_analysis(block)  # NEW: AWD
        track_bias_impact_per_horse[name] = parse_track_bias_impact_values(block)  # NEW: Impact Values
        jockey_trainer_per_horse[name] = parse_jockey_trainer_for_block(block, debug=False)
        jock_train_per_horse[name] = parse_jock_train_for_block(block)
        running_style_per_horse[name] = parse_running_style_for_block(block, debug=False)
        quickplay_per_horse[name] = parse_quickplay_comments_for_block(block, debug=False)
        workout_per_horse[name] = parse_recent_workout_for_block(block, debug=False)
        prime_power_per_horse[name] = parse_prime_power_for_block(block, debug=False)
        figs_per_horse[name] = parse_speed_figures_for_block(block, debug=False)
        cr_rr_per_horse[name] = parse_cr_rr_history(figs_per_horse[name])  # NEW: Parse CR/RR metrics
        equip_lasix_per_horse[name] = parse_equip_lasix(block)
        all_angles_per_horse[name] = parse_all_angles(block)
        trainer_intent_per_horse[name] = parse_trainer_intent(block)

# Debug: Check what was parsed
st.caption(f"🔍 Parsed all_angles_per_horse for {len(all_angles_per_horse)} horses")
if all_angles_per_horse:
    sample_horse = list(all_angles_per_horse.keys())[0]
    sample_data = all_angles_per_horse[sample_horse]
    st.caption(f"✓ Sample: {sample_horse} → prime={sample_data.get('prime', 'N/A')}, trainer_win={sample_data.get('trainer_win', 'N/A')}, lp count={len(sample_data.get('lp', []))}")
else:
    st.error("❌ all_angles_per_horse is EMPTY - parsing failed!")
    
st.caption(f"🔍 Parsed trainer_intent_per_horse for {len(trainer_intent_per_horse)} horses")
if trainer_intent_per_horse:
    sample_horse = list(trainer_intent_per_horse.keys())[0]
    st.caption(f"✓ Sample intent: {sample_horse} → {trainer_intent_per_horse[sample_horse]}")
else:
    st.error("❌ trainer_intent_per_horse is EMPTY - parsing failed!")

# Create the figs_df
figs_data = []
for name, fig_dict in figs_per_horse.items():
    spd_list = fig_dict.get("SPD", [])
    if spd_list:  # Only add horses that have SPD figures
        figs_data.append({
            "Horse": name,
            "Figures": spd_list,  # The list of SPD figures
            "BestFig": max(spd_list),
            "AvgTop2": round(np.mean(sorted(spd_list, reverse=True)[:2]), 1)
        })
    else:
        st.caption(f"⚠️ Horse '{name}' has no SPD figures parsed")

figs_df = pd.DataFrame(figs_data)
st.caption(f"📊 figs_df created with {len(figs_df)} horses (out of {len(figs_per_horse)} total)")

df_final_field = df_editor[df_editor["Scratched"]==False].copy()
if df_final_field.empty:
    st.warning("All horses are scratched.")
    st.stop()

# Add LastFig column - most recent speed figure for each horse
df_final_field['LastFig'] = df_final_field['Horse'].map(
    lambda h: figs_per_horse.get(h, {}).get('SPD', [np.nan])[0] if figs_per_horse.get(h, {}).get('SPD') else np.nan
)

# Add other speed rating columns
df_final_field['E1'] = df_final_field['Horse'].map(
    lambda h: figs_per_horse.get(h, {}).get('E1', [np.nan])[0] if figs_per_horse.get(h, {}).get('E1') else np.nan
)
df_final_field['E2'] = df_final_field['Horse'].map(
    lambda h: figs_per_horse.get(h, {}).get('E2', [np.nan])[0] if figs_per_horse.get(h, {}).get('E2') else np.nan
)
df_final_field['RR'] = df_final_field['Horse'].map(
    lambda h: figs_per_horse.get(h, {}).get('RR', [np.nan])[0] if figs_per_horse.get(h, {}).get('RR') else np.nan
)
df_final_field['CR'] = df_final_field['Horse'].map(
    lambda h: figs_per_horse.get(h, {}).get('CR', [np.nan])[0] if figs_per_horse.get(h, {}).get('CR') else np.nan
)

st.caption(f"✅ Added speed figure columns: LastFig, E1, E2, RR, CR to df_final_field")

# Add Jockey and Trainer columns
df_final_field['Jockey'] = df_final_field['Horse'].map(
    lambda h: jockey_trainer_per_horse.get(h, {}).get('jockey', '')
)
df_final_field['Trainer'] = df_final_field['Horse'].map(
    lambda h: jockey_trainer_per_horse.get(h, {}).get('trainer', '')
)

st.caption(f"✅ Added jockey/trainer columns: Jockey, Trainer to df_final_field")

# Add Running Style column from BRISNET
df_final_field['RunningStyle'] = df_final_field['Horse'].map(
    lambda h: running_style_per_horse.get(h, {}).get('running_style', '')
)

st.caption(f"✅ Added running style column: RunningStyle to df_final_field")

# Add QuickPlay comment columns
df_final_field['QuickPlayPositive'] = df_final_field['Horse'].map(
    lambda h: '; '.join(quickplay_per_horse.get(h, {}).get('positive_comments', []))
)
df_final_field['QuickPlayNegative'] = df_final_field['Horse'].map(
    lambda h: '; '.join(quickplay_per_horse.get(h, {}).get('negative_comments', []))
)

st.caption(f"✅ Added QuickPlay columns: QuickPlayPositive, QuickPlayNegative to df_final_field")

# Add Recent Workout columns
df_final_field['WorkoutDate'] = df_final_field['Horse'].map(
    lambda h: workout_per_horse.get(h, {}).get('workout_date', '')
)
df_final_field['WorkoutTrack'] = df_final_field['Horse'].map(
    lambda h: workout_per_horse.get(h, {}).get('workout_track', '')
)
df_final_field['WorkoutDistance'] = df_final_field['Horse'].map(
    lambda h: workout_per_horse.get(h, {}).get('workout_distance', '')
)
df_final_field['WorkoutTime'] = df_final_field['Horse'].map(
    lambda h: workout_per_horse.get(h, {}).get('workout_time', '')
)
df_final_field['WorkoutRank'] = df_final_field['Horse'].map(
    lambda h: workout_per_horse.get(h, {}).get('workout_rank', '')
)
df_final_field['WorkoutTotal'] = df_final_field['Horse'].map(
    lambda h: workout_per_horse.get(h, {}).get('workout_total', '')
)

st.caption(f"✅ Added workout columns: WorkoutDate, WorkoutTrack, WorkoutDistance, WorkoutTime, WorkoutRank, WorkoutTotal to df_final_field")

# Add Prime Power columns
df_final_field['PrimePower'] = df_final_field['Horse'].map(
    lambda h: prime_power_per_horse.get(h, {}).get('prime_power', None)
)
df_final_field['PrimePowerRank'] = df_final_field['Horse'].map(
    lambda h: prime_power_per_horse.get(h, {}).get('prime_power_rank', '')
)

st.caption(f"✅ Added Prime Power columns: PrimePower, PrimePowerRank to df_final_field")

# Add Equipment and Lasix columns
df_final_field['Blinkers'] = df_final_field['Horse'].map(
    lambda h: equip_lasix_per_horse.get(h, ('', ''))[0]
)
df_final_field['Lasix'] = df_final_field['Horse'].map(
    lambda h: equip_lasix_per_horse.get(h, ('', ''))[1]
)

st.caption(f"✅ Added Equipment/Lasix columns: Blinkers, Lasix to df_final_field")

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

    # 4) Expanded pedigree/work/layoff
    exp_ped = expanded_ped_per_horse.get(name, {})
    jt_data = jock_train_per_horse.get(name, {})
    
    # Turf surface bonus
    if race_surface.lower() == "turf" and pd.notna(exp_ped.get("turf_win_pct")):
        if exp_ped.get("turf_win_pct", 0) > 0.15:
            tweak += MODEL_CONFIG['ped_dist_bonus']
    
    # Mud/sloppy bonus
    if race_cond in {"muddy", "sloppy"} and pd.notna(exp_ped.get("mud_win_pct")):
        if exp_ped.get("mud_win_pct", 0) > 0.15:
            tweak += MODEL_CONFIG['ped_dist_bonus']
    
    # Bullet workout bonus
    work_bonus = min(exp_ped.get("bullet_count", 0) * 0.03, 0.09)
    tweak += work_bonus
    
    # Layoff + trainer situational bonus (45-90 days ideal)
    days_off = exp_ped.get("days_off", np.nan)
    if pd.notna(days_off) and 45 <= days_off <= 90:
        trainer_lay_data = jt_data.get("trainer_situational", {}).get("Layoff", {})
        if trainer_lay_data.get("win_pct", 0) > 18:
            tweak += 0.04

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
    
    # Equipment & Lasix adjustments
    blink, lasix = equip_lasix_per_horse.get(name, ("off", "off"))
    cclass_add += 0.08 if blink == "on" else 0.03 if blink == "off" else 0
    cclass_add += 0.09 if lasix == "first" else -0.10 if lasix == "off" else -0.04 if lasix == "repeat" else 0
    
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
                         track_name: str = "") -> pd.DataFrame:
    """
    Reads 'Cclass' from df_styles (pre-built), adds Cstyle/Cpost/Cpace (+Atrack),
    sums to Arace and R. Returns rating table.
    """
    cols = ["#", "Post", "Horse", "Style", "Quirin", "Cstyle", "Cpost", "Cpace", "Cclass", "Atrack", "Arace", "R", "LastFig", "E1", "E2", "RR", "CR"]
    if df_styles is None or df_styles.empty:
        return pd.DataFrame(columns=cols)

    # Ensure class column present
    if "Cclass" not in df_styles.columns:
        df_styles = df_styles.copy()
        df_styles["Cclass"] = 0.0 # Default Cclass if missing

    # Derive per-horse pace tailwind from PPI
    ppi_map = compute_ppi(df_styles).get("by_horse", {})

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
        
        # Jockey/Trainer Intent Bonus
        jt_data = jock_train_per_horse.get(name, {})  # Assume stored like pedigree_per_horse
        last_purse = jt_data.get("last_purse", np.nan)
        drop_pct = (purse_val - last_purse) / last_purse if last_purse > 0 else 0
        drop_bonus = MODEL_CONFIG['class_drop_bonus'] if drop_pct < -0.20 else 0  # Negative for drop
        
        jock_bonus = max(0, (jt_data.get("jock_win_pct", 0) - 0.15) * MODEL_CONFIG['jock_win_bonus_per'])
        trainer_bonus = max(0, (jt_data.get("trainer_roi", 1.0) - 1.0) * MODEL_CONFIG['trainer_roi_bonus_per'])
        
        # Jock upgrade: Compare to avg jock_win (compute field avg)
        field_avg_jock_win = np.nanmean([d.get("jock_win_pct", np.nan) for d in jock_train_per_horse.values()])
        upgrade_diff = jt_data.get("jock_win_pct", 0) - field_avg_jock_win
        upgrade_bonus = max(0, upgrade_diff * MODEL_CONFIG['jock_upgrade_bonus_per'])
        
        intent_bonus = min(MODEL_CONFIG['intent_max_bonus'], drop_bonus + jock_bonus + trainer_bonus + upgrade_bonus)

        # Prime Power Bonus - calculate field average once
        if _ == 0:
            all_primes = [parse_prime_power_for_block(blocks.get(row_inner["Horse"], "")) 
                         for _, row_inner in df_styles.iterrows()]
            field_avg_prime = np.nanmean([p for p in all_primes if not np.isnan(p)])
        
        prime = parse_prime_power_for_block(blocks.get(name, ""))
        prime_bonus = (prime - field_avg_prime) * 0.005 if not np.isnan(prime) and not np.isnan(field_avg_prime) else 0

        # EP/LP/Trip Enhancement Bonus
        pace_data = ep_lp_trip_per_horse.get(name, {})
        field_avg_lp = np.nanmean([d.get("avg_lp", np.nan) for d in ep_lp_trip_per_horse.values()])
        lp_bonus = 0.06 if pace_data.get("avg_lp", 0) > field_avg_lp + 10 and ppi_map.get(name, 0) > 0.5 else 0
        excuse_bonus = min(pace_data.get("excuse_count", 0) * 0.03, 0.09)
        pace_enh_bonus = lp_bonus + excuse_bonus
        cpace += pace_enh_bonus

        # === 5 NEW HIGH-ROI PREDICTIVE FEATURES ===
        
        # 1. FIGURE TREND ANALYSIS (Uptrend/Downtrend bonus/penalty)
        figs_list = figs_per_horse.get(name, {}).get('SPD', [])
        trend_bonus, trend_label = calculate_figure_trend(figs_list)
        
        # 2. RECENCY WEIGHTING (Recent races weighted heavier)
        # Apply decay to older figures - most recent get full weight, older get less
        recency_decay = 1.0 if len(figs_list) <= 1 else min(0.15 * (len(figs_list) - 1), 0.4)
        cpace *= (1.0 - recency_decay * 0.3)  # Slightly reduce pace if older
        
        # 3. DISTANCE-SPECIFIC CONSISTENCY (Win% at THIS distance)
        block = blocks.get(name, "")
        dist_win_pct, dist_wins, dist_starts = calculate_distance_record(block, distance_txt)
        distance_bonus = 0.0
        if pd.notna(dist_win_pct):
            if dist_win_pct > MODEL_CONFIG['distance_specialist_threshold']:
                distance_bonus = MODEL_CONFIG['distance_specialist_bonus']  # 0.08
            elif dist_win_pct < MODEL_CONFIG['distance_poor_threshold']:
                distance_bonus = MODEL_CONFIG['distance_poor_penalty']  # -0.06
        
        # 4. BOUNCE DETECTION (Sharp drop in speed figures)
        bounce_penalty = 0.0
        if len(figs_list) >= 2:
            fig_drop = figs_list[1] - figs_list[0]  # Older - Recent (positive = drop)
            if fig_drop > MODEL_CONFIG['bounce_fig_drop_threshold']:  # >6 point drop
                bounce_penalty = MODEL_CONFIG['bounce_penalty_aggressive']  # -0.10
        
        # 5. CLASS TRANSITION & EQUIPMENT CHANGES
        class_trans_penalty = 0.0
        equip_penalty = 0.0
        
        # Class rise penalty: if moving UP in class, apply penalty
        last_purse = jt_data.get("last_purse", np.nan)
        if pd.notna(last_purse) and last_purse > 0:
            class_change_pct = (purse_val - last_purse) / last_purse
            if class_change_pct > 0.20:  # Moving UP in class by >20%
                class_trans_penalty = MODEL_CONFIG['class_rise_penalty']  # -0.07
        
        # Blinkers off penalty
        blink_status = equip_lasix_per_horse.get(name, ('', ''))[0]
        if blink_status == "off" and re.search(r'Blinkers\s+On', block or "", re.I):
            equip_penalty = MODEL_CONFIG['blinkers_off_penalty']  # -0.06
        
        # Sum all new bonuses
        new_features_bonus = trend_bonus + distance_bonus + bounce_penalty + class_trans_penalty + equip_penalty
        
        # === TIER 1: BRISNET PEDIGREE RATINGS (NEW) ===
        bris_ped_bonus = 0.0
        bris_ped = bris_ped_ratings_per_horse.get(name, {})
        
        # Apply bonus if condition matches horse's pedigree strength
        if surface_type.lower() == "dirt" and pd.notna(bris_ped.get("fast_ped")):
            if bris_ped["fast_ped"] >= MODEL_CONFIG['bris_ped_rating_threshold']:
                bris_ped_bonus += MODEL_CONFIG['bris_ped_fast_bonus']
        
        if surface_type.lower() == "turf" and pd.notna(bris_ped.get("turf_ped")):
            if bris_ped["turf_ped"] >= MODEL_CONFIG['bris_ped_rating_threshold']:
                bris_ped_bonus += MODEL_CONFIG['bris_ped_turf_bonus']
        
        if condition_txt.lower() in ("muddy", "sloppy", "heavy") and pd.notna(bris_ped.get("off_ped")):
            if bris_ped["off_ped"] >= MODEL_CONFIG['bris_ped_rating_threshold']:
                bris_ped_bonus += MODEL_CONFIG['bris_ped_off_bonus']
        
        # Distance specialist bonus (applies to any race)
        if pd.notna(bris_ped.get("distance_ped")):
            if bris_ped["distance_ped"] >= MODEL_CONFIG['bris_ped_rating_threshold']:
                bris_ped_bonus += MODEL_CONFIG['bris_ped_distance_bonus']
        
        # === TIER 1: DAM PRODUCTION INDEX BONUS (NEW) ===
        dpi_bonus = 0.0
        ped_data = pedigree_per_horse.get(name, {})
        if pd.notna(ped_data.get("dam_dpi")):
            if ped_data["dam_dpi"] >= MODEL_CONFIG['dpi_bonus_threshold']:
                dpi_bonus = MODEL_CONFIG['dpi_bonus']
        
        # === TIER 1: SURFACE-SPECIFIC RECORD PENALTY (NEW) ===
        surface_record_penalty = 0.0
        surf_record = surface_record_per_horse.get(name, {})
        
        # Penalize if moving to surface where horse has poor record but good record elsewhere
        if surface_type.lower() == "dirt":
            dirt_rec = surf_record.get("dirt_record", {})
            off_rec = surf_record.get("off_track_record", {})
            
            # If dirt record is poor (<20%) but off-track record is good (>40%), penalize
            if dirt_rec.get("win_pct", 0) < MODEL_CONFIG['surface_specialist_threshold_poor'] and \
               off_rec.get("win_pct", 0) > MODEL_CONFIG['surface_specialist_threshold_good']:
                surface_record_penalty = MODEL_CONFIG['surface_mismatch_dirt_penalty']
        
        elif surface_type.lower() == "turf":
            turf_rec = surf_record.get("turf_record", {})
            dirt_rec = surf_record.get("dirt_record", {})
            
            # If turf record is poor but dirt record is good, apply penalty (less likely on turf)
            if turf_rec.get("win_pct", 0) < MODEL_CONFIG['surface_specialist_threshold_poor'] and \
               dirt_rec.get("win_pct", 0) > MODEL_CONFIG['surface_specialist_threshold_good']:
                surface_record_penalty = MODEL_CONFIG['surface_mismatch_dirt_penalty']
        
        # === PART 2 ENHANCEMENT: CR/RR PERFORMANCE RATIO BONUS (NEW) ===
        cr_rr_bonus = 0.0
        cr_rr_data = cr_rr_per_horse.get(name, {})
        cr_rr_ratio = cr_rr_data.get("cr_rr_ratio", np.nan)
        consistency = cr_rr_data.get("consistency", 0.0)
        
        # Bonus if horse consistently performs close to or above field quality
        if pd.notna(cr_rr_ratio):
            if cr_rr_ratio >= MODEL_CONFIG['cr_rr_excellent_threshold']:
                # Horse performing above field quality (CR >= RR)
                cr_rr_bonus += MODEL_CONFIG['cr_rr_excellent_bonus']
            elif cr_rr_ratio >= MODEL_CONFIG['cr_rr_outperform_threshold']:
                # Horse performing close to field quality
                cr_rr_bonus += MODEL_CONFIG['cr_rr_outperform_bonus']
        
        # Additional bonus for consistent performances
        if consistency > 0.7:  # High consistency
            cr_rr_bonus += MODEL_CONFIG['cr_rr_consistency_bonus']
        
        # Combine all three Tier 1 bonuses + CR/RR bonus
        tier1_bonus = bris_ped_bonus + dpi_bonus + surface_record_penalty + cr_rr_bonus

        # === NEW TIER 2: PEDIGREE ENHANCEMENTS (SPI, Surface Stats, AWD) ===
        
        # 1. Sire Production Index (SPI) Bonus/Penalty
        spi_bonus = 0.0
        spi_data = pedigree_spi_per_horse.get(name, {})
        if pd.notna(spi_data.get("spi")) or pd.notna(spi_data.get("dam_sire_spi")):
            spi_bonus = calculate_spi_bonus(spi_data.get("spi"), spi_data.get("dam_sire_spi"))
        
        # 2. Surface Specialty Statistics Bonus (Sire %Mud, %Turf)
        surface_stats_bonus = 0.0
        surf_stats = pedigree_surface_stats_per_horse.get(name, {})
        if any(pd.notna(v) for v in [surf_stats.get("sire_mud_pct"), surf_stats.get("sire_turf_pct"), 
                                       surf_stats.get("dam_sire_mud_pct")]):
            surface_stats_bonus = calculate_surface_specialty_bonus(
                surf_stats.get("sire_mud_pct"),
                surf_stats.get("sire_turf_pct"),
                surf_stats.get("dam_sire_mud_pct"),
                condition_txt,
                surface_type
            )
        
        # 3. Average Winning Distance (AWD) Mismatch Penalty
        awd_penalty = 0.0
        awd_data = awd_analysis_per_horse.get(name, {})
        if pd.notna(awd_data.get("sire_awd")) or pd.notna(awd_data.get("dam_sire_awd")):
            awd_penalty = calculate_awd_mismatch_penalty(
                awd_data.get("sire_awd"),
                awd_data.get("dam_sire_awd"),
                distance_txt
            )
        
        # Sum Tier 2 bonuses
        tier2_bonus = spi_bonus + surface_stats_bonus + awd_penalty

        # === TRACK BIAS WITH IMPACT VALUES (CRITICAL FIX) ===
        # Pass Impact Values to _get_track_bias_delta for data-driven calculation
        impact_values = track_bias_impact_per_horse.get(name, {})
        a_track = _get_track_bias_delta(track_name, surface_type, distance_txt, style, post,
                                       impact_values=impact_values)

        c_class = float(row.get("Cclass", 0.0))

        arace = c_class + cstyle + cpost + cpace + a_track + intent_bonus + prime_bonus + new_features_bonus + tier1_bonus + tier2_bonus
        R     = arace

        # Ensure Quirin is formatted correctly for display (handle NaN)
        quirin_display = quirin if pd.notna(quirin) else None

        rows.append({
            "#": post, "Post": post, "Horse": name, "Style": style, "Quirin": quirin_display,
            "Cstyle": round(cstyle, 2), "Cpost": round(cpost, 2), "Cpace": round(cpace, 2),
            "Cclass": round(c_class, 2), "Atrack": round(a_track, 2), "Arace": round(arace, 2), "R": round(R, 2),
            "LastFig": row.get("LastFig", np.nan),
            "E1": row.get("E1", np.nan),
            "E2": row.get("E2", np.nan),
            "RR": row.get("RR", np.nan),
            "CR": row.get("CR", np.nan)
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

    p = softmax_from_rating(r, tau=0.55)

    # Clean up temporary column
    ratings_df.drop(columns=['R_numeric'], inplace=True)

    # Ensure probabilities sum to 1 (or very close)
    p_sum = np.sum(p)
    if p_sum > 0:
      p = p / p_sum

    return {h: p[i] for i, h in enumerate(ratings_df["Horse"].values)}


def dynamic_exotic_probs(win_probs: Dict[str, float], ppi_val: float, df_styles: pd.DataFrame,
                         positions: int = 5, sims: int = 20000) -> List[List[str]]:
    """
    Simulate exotic race outcomes with common-sense closer bias in high-PPI scenarios.
    Returns list of simulated outcomes (each outcome is a list of horses in finish order).
    """
    horses = list(win_probs.keys())
    probs = np.array(list(win_probs.values()))
    style_map = dict(zip(df_styles["Horse"], df_styles["Style"]))  # For closer bias
    outcomes = []
    
    for _ in range(sims):
        idx = np.random.choice(range(len(horses)), positions, replace=False, p=probs / probs.sum())
        outcome = [horses[i] for i in idx]
        
        # Common-sense bias: In high PPI (>0.5), boost closers (P/S) probability for 3rd/4th/5th
        if ppi_val > 0.5:
            closer_horses = [h for h in outcome if style_map.get(h, '') in ["P", "S"]]
            if closer_horses and np.random.rand() < MODEL_CONFIG['closer_bias_high_ppi']:
                # Randomly promote a closer to 3rd-5th if in top 2
                closer = np.random.choice(closer_horses)
                curr_pos = outcome.index(closer)
                if curr_pos < 2:
                    target_pos = np.random.choice([2, 3, 4])
                    outcome[curr_pos], outcome[target_pos] = outcome[target_pos], outcome[curr_pos]
        
        outcomes.append(outcome)
    
    return outcomes


def extract_probable_positions(outcomes: List[List[str]]) -> Dict[str, List[Tuple[str, int]]]:
    """
    Extract most probable horses for each position from simulated outcomes.
    Returns dict with keys "Pos1", "Pos2", etc., each containing top N horses by frequency.
    """
    pos_counters = [Counter([o[i] for o in outcomes]) for i in range(MODEL_CONFIG['positions_to_sim'])]
    probables = {}
    
    for pos in range(1, MODEL_CONFIG['positions_to_sim'] + 1):
        top_n = MODEL_CONFIG['top_for_pos'][pos-1]
        probables[f"Pos{pos}"] = pos_counters[pos-1].most_common(top_n)
    
    return probables


def generate_bet_outcomes(probables: Dict[str, List[Tuple[str, int]]]) -> str:
    """
    Generate human-readable betting recommendations from probable positions.
    Covers Win, Exacta, Trifecta, Superfecta, and Super High 5.
    """
    report = "### Most Probable Finishing Outcomes (Common-Sense Bets)\n"
    
    # Win: Top 2 for 1st (straight bets, key underlays)
    top1 = [h for h, _ in probables["Pos1"]]
    report += f"**Win (1st Place):** Most likely 2 winners: {top1[0]} (underlay if short odds - key it), {top1[1] if len(top1)>1 else 'None'} (value overlay bet).\n"
    
    # Exacta: Top 2 1st / Top 2 2nd (straight + box)
    top2 = [h for h, _ in probables["Pos2"]]
    report += f"**Exacta (1st-2nd):** Wheel {top1[0]}/{top2[0]},{top2[1]} or box top 2 1st/2nd ({'/'.join(top1 + top2)}).\n"
    
    # Trifecta: Add Top 3 3rd (wheels + part-box)
    top3 = [h for h, _ in probables["Pos3"]]
    report += f"**Trifecta (1st-3rd):** Wheel {top1[0]}/{top2[0]},{top2[1]}/{top3[0]},{top3[1]},{top3[2]} (key underlay on top, spread overlays underneath).\n"
    
    # Superfecta: Add Top 3 4th
    top4 = [h for h, _ in probables["Pos4"]]
    report += f"**Superfecta (1st-4th):** Part-wheel {top1[0]}/{top2[0]},{top2[1]}/{top3[0]},{top3[1]},{top3[2]}/{top4[0]},{top4[1]},{top4[2]} (common-sense: Tight on top if underlay dominant).\n"
    
    # SH5: Add Top 3 5th (wide wheel for value bombs)
    top5 = [h for h, _ in probables["Pos5"]] if "Pos5" in probables else []
    if top5:
        report += f"**Super High 5 (1st-5th):** Wheel {top1[0]}/{top2[0]},{top2[1]}/{top3[0]},{top3[1]},{top3[2]}/{top4[0]},{top4[1]},{top4[2]}/{top5[0]},{top5[1]},{top5[2]} (spread wide underneath for overlays in meltdowns).\n"
    
    return report


def optimize_tickets(outcomes: List[List[str]], probables: Dict[str, List[Tuple[str, int]]],
                     offered_odds: Dict[str, float], budget: float = 100.0, base_bet: float = 0.10) -> str:
    """
    Optimize exotic tickets using PuLP to maximize EV within budget.
    Focuses on high-probability permutations with key underlay bonuses.
    """
    if not PULP_AVAILABLE:
        return "⚠️ Ticket optimization disabled (PuLP not installed)."
    
    # Extract top horses for each position
    top1_h = [h for h, _ in probables["Pos1"]]
    top2_h = [h for h, _ in probables["Pos2"]]
    top3_h = [h for h, _ in probables["Pos3"]]
    top4_h = [h for h, _ in probables["Pos4"]]
    top5_h = [h for h, _ in probables["Pos5"]] if "Pos5" in probables else []
    
    # Generate candidate supers/SH5 (limit to top combos to avoid explosion)
    candidate_perms = []
    try:
        for p1 in permutations(top1_h + top2_h, 1):  # Top-heavy
            for p2 in permutations(top2_h + top3_h, 1):
                for p3 in permutations(top3_h + top4_h, 1):
                    for p4 in permutations(top4_h + top5_h, 1):
                        perm = list(p1 + p2 + p3 + p4)
                        if len(set(perm)) == 4:  # Unique
                            candidate_perms.append(tuple(perm))
        
        if top5_h and len(candidate_perms) > 0:
            # Extend to SH5, but sample to keep feasible
            candidate_perms = [p + (np.random.choice(top5_h),) for p in candidate_perms[:100]]  # Cap for PuLP
    except Exception as e:
        return f"⚠️ Error generating permutations: {e}"
    
    if not candidate_perms:
        return "⚠️ No candidate permutations generated."
    
    # Count frequency of perms in simulations
    perm_len = len(candidate_perms[0]) if candidate_perms else 4
    perm_freq = Counter(tuple(o[:perm_len]) for o in outcomes)
    total_sims = len(outcomes)
    
    # Fair payouts based on simulation frequency
    fair_payouts = {perm: (total_sims / freq) * 2 if freq > 0 else float('inf') 
                    for perm, freq in perm_freq.items() if perm in candidate_perms}
    
    # EV calculation: Rough estimate using offered odds on top horse
    ev_map = {}
    for perm in candidate_perms:
        top_horse_odds = offered_odds.get(perm[0], 5.0)  # Default 4-1
        prob = perm_freq.get(perm, 1) / total_sims if total_sims > 0 else 0
        ev = prob * (top_horse_odds - 1) - (1 - prob) if prob > 0 else -1
        ev_map[perm] = max(ev, 0)  # Positive only
    
    # PuLP optimization: Max sum EV s.t. cost <= budget, prefer underlay-top
    try:
        prob = pulp.LpProblem("TicketOpt", pulp.LpMaximize)
        x = pulp.LpVariable.dicts("select", candidate_perms, cat='Binary')
        
        # Objective: EV + bonus for underlay tops (odds < 3.0)
        prob += pulp.lpSum([
            x[p] * (ev_map.get(p, 0) + (0.1 if offered_odds.get(p[0], 10) < 3.0 else 0))
            for p in candidate_perms
        ])
        
        # Constraint: Total cost <= budget
        prob += pulp.lpSum([x[p] * base_bet for p in candidate_perms]) <= budget
        
        prob.solve(pulp.PULP_CBC_CMD(msg=0))
        
        selected = [p for p in candidate_perms if x[p].value() and x[p].value() > 0.5]
    except Exception as e:
        return f"⚠️ PuLP optimization error: {e}"
    
    # Build report
    report = f"**🎯 Optimized Tickets (${budget:.2f} budget, {len(selected)} combos - EV-focused, key underlays on top):**\n"
    if not selected:
        report += "No profitable combos found within budget constraints.\n"
        return report
    
    for p in sorted(selected, key=lambda pp: ev_map.get(pp, 0), reverse=True)[:10]:  # Top 10 by EV
        fair_payout = fair_payouts.get(p, 0)
        ev_val = ev_map.get(p, 0)
        perm_str = '/'.join(p)
        report += f"* `{perm_str}` (Fair Payout: ${fair_payout:.2f}, Est EV: ${ev_val:.2f})\n"
    
    return report


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
            track_name=track_name
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

        # SAFE market blend (no sklearn, no NaNs)
        market_df = _build_market_table(df_final_field, fair_probs)
        use_probs = adjust_probs_with_market(market_df, race_type_detected)
        if not use_probs:
            use_probs = fair_probs
            st.caption("Market blend: no valid Live/ML odds — using model-only fair probabilities.")
        else:
            valid_rows = int(market_df["live_prob"].notna().sum() + market_df["ml_prob"].notna().sum())
            st.caption(f"Market blend applied using {valid_rows} Live/ML entries (α={_alpha_from_race_type(race_type_detected):.2f}).")

        # Show Fair% / Fair Odds from the probabilities we actually use
        if 'Horse' in ratings_df.columns:
            ratings_df["Fair %"]    = ratings_df["Horse"].map(lambda h: f"{use_probs.get(h,0)*100:.1f}%")
            ratings_df["Fair Odds"] = ratings_df["Horse"].map(lambda h: fair_to_american_str(use_probs.get(h,0)))
        else:
            ratings_df["Fair %"] = ""
            ratings_df["Fair Odds"] = ""

        all_scenario_ratings[(rbias, pbias)] = (ratings_df, use_probs)

        # Table display (guard for helper column)
        disp = ratings_df.sort_values(by="R", ascending=False).copy()
        if "R_ENHANCE_ADJ" in disp.columns:
            disp = disp.drop(columns=["R_ENHANCE_ADJ"])
        
        # Add custom display columns
        disp["Frac1"] = disp["Horse"].map(lambda h: round(np.mean([f[0] for f in all_angles_per_horse.get(h, {}).get("frac",[(99,)])[:3]]),1) if all_angles_per_horse.get(h) else 99)
        disp["ParBeat"] = disp["Horse"].map(lambda h: max([f-today_par for f in figs_per_horse.get(h, {}).get('SPD', [])[1:4]], default=0))
        
        # Add Drift column (odds drift percentage: positive = shortening/coming in, negative = drifting out)
        def calculate_drift(horse_name):
            if horse_name not in df_final_field["Horse"].values:
                return 0.0
            horse_row = df_final_field.loc[df_final_field["Horse"] == horse_name].iloc[0]
            ml_str = str(horse_row.get("ML", "")).strip()
            live_str = str(horse_row.get("Live Odds", "")).strip()
            
            # If no live odds entered, no drift
            if not live_str or live_str == ml_str:
                return 0.0
            
            ml_dec = str_to_decimal_odds(ml_str)
            live_dec = str_to_decimal_odds(live_str)
            
            if not ml_dec or ml_dec <= 1 or not live_dec or live_dec <= 1:
                return 0.0
            
            # Drift % = ((ML - Live) / ML) * 100
            # Positive = odds shortened (more money), Negative = odds drifted (less money)
            drift_pct = ((ml_dec - live_dec) / ml_dec) * 100
            return round(drift_pct, 1)
        
        disp["Drift"] = disp["Horse"].map(calculate_drift)
        
        # Add Intent column (sum of trainer intent numeric signals)
        disp["Intent"] = disp["Horse"].map(lambda h: round(sum([v for k,v in trainer_intent_per_horse.get(h, {}).items() if isinstance(v, (int,float))]), 2))
        
        # Debug: Check what's in disp before cleanup
        st.caption(f"🔍 Display columns before cleanup: {list(disp.columns)}")
        st.caption(f"🔍 Prime in disp: {'Prime' in disp.columns}, APEX in disp: {'APEX' in disp.columns}, R in disp: {'R' in disp.columns}")
        if "Prime" in disp.columns:
            st.caption(f"🔍 Prime sample values: {disp['Prime'].head(3).tolist()}")
        if "APEX" in disp.columns:
            st.caption(f"🔍 APEX sample values: {disp['APEX'].head(3).tolist()}")
        if "R" in disp.columns:
            st.caption(f"🔍 R sample values: {disp['R'].head(3).tolist()}")
        
        # Clean up NaN values for display - convert to numeric first, then fillna
        if "Prime" in disp.columns:
            disp["Prime"] = pd.to_numeric(disp["Prime"], errors='coerce').fillna(0).astype(int)
        if "R" in disp.columns:
            disp["R"] = pd.to_numeric(disp["R"], errors='coerce').fillna(0.0).round(2)
        if "APEX" in disp.columns:
            disp["APEX"] = pd.to_numeric(disp["APEX"], errors='coerce').fillna(0.0).round(3)
        if "Intent" in disp.columns:
            disp["Intent"] = pd.to_numeric(disp["Intent"], errors='coerce').fillna(0.0).round(2)
        
        # Select and reorder columns for display
        display_cols = ["#","Horse","Prime","R","Frac1","ParBeat","Drift","Intent","APEX","Fair %","Fair Odds"]
        # Only include columns that exist
        display_cols = [c for c in display_cols if c in disp.columns]
        st.caption(f"🔍 Final display_cols to show: {display_cols}")
        disp = disp[display_cols]
        
        st.dataframe(
            disp,
            use_container_width=True, hide_index=True,
            column_config={
                "#": st.column_config.TextColumn("#", width="small"),
                "Horse": st.column_config.TextColumn("Horse", width="medium"),
                "Prime": st.column_config.NumberColumn("Prime", format="%.0f"),
                "R": st.column_config.NumberColumn("Rating", format="%.2f"),
                "Frac1": st.column_config.NumberColumn("Frac1", format="%.1f", help="Avg Early Fractional Position (first 3 races)"),
                "ParBeat": st.column_config.NumberColumn("ParBeat", format="%.0f", help="Best figure vs today's par (races 2-4)"),
                "Drift": st.column_config.NumberColumn("Drift", format="%.1f", help="Odds drift % (ML to Live - positive = shortening)"),
                "Intent": st.column_config.NumberColumn("Intent", format="%.2f", help="Trainer intent signal score"),
                "APEX": st.column_config.NumberColumn("APEX", format="%.3f", help="Advanced handicapping adjustment"),
                "Fair %": st.column_config.TextColumn("Fair %", width="small"),
                "Fair Odds": st.column_config.TextColumn("Fair Odds", width="small"),
            }
        )

# Ensure primary key exists before accessing
if scenarios:
    primary_key = scenarios[0]
    primary_df, primary_probs = all_scenario_ratings[primary_key]  # this is use_probs now
    st.info(f"**Primary Scenario:** S: `{primary_key[0]}` • P: `{primary_key[1]}` • Profile: `{strategy_profile}`  • PPI: {ppi_val:+.2f}")
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
                           name_to_ml: Dict[str, str], field_size: int, ppi_val: float,
                           offered_odds_map: Optional[Dict[str, float]] = None) -> str:
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
    
    # FIXED: Sort horses by FINAL BLENDED PROBABILITY (includes live odds), not just R rating
    # This ensures live odds impact the finish order predictions
    primary_probs_sorted = sorted(primary_probs.items(), key=lambda x: x[1], reverse=True)
    all_horses = [h for h, p in primary_probs_sorted]  # Horses ordered by blended probability
    
    pos_ev_horses = set(df_ol[df_ol["EV per $1"] > 0.05]['Horse'].tolist()) if not df_ol.empty else set()
    neg_ev_horses = set(df_ol[df_ol["EV per $1"] < -0.05]['Horse'].tolist()) if not df_ol.empty else set()
    
    if strategy_profile == "Confident":
        # Confident Model: A-Group = UNDERLAYS (favorites with negative EV), C-Group = OVERLAYS
        # Underlays are horses bet BELOW their fair value (low odds, chalk horses)
        A_group = [h for h in all_horses if h in neg_ev_horses][:3]  # Top 3 underlays (favorites)
        if not A_group:  # Fallback if no clear underlays
            A_group = all_horses[:2]  # Use top 2 rated horses
    else: # Value Hunter - Prioritize overlays
        # Value Hunter: A-Group = OVERLAYS (positive EV), C-Group = UNDERLAYS (favorites)
        A_group = [h for h in all_horses if h in pos_ev_horses][:3]  # Top 3 overlays
        if not A_group:  # Fallback if no overlays
            A_group = [all_horses[0]] if all_horses else []  # Use top pick

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
    
    # --- 3a. Most Likely Winner Analysis (NEW) ---
    top_rated_horse = all_horses[0]
    top_prob = primary_probs.get(top_rated_horse, 0)
    is_overlay = top_rated_horse in pos_ev_horses
    is_underlay = top_rated_horse in neg_ev_horses
    top_ml_str = name_to_ml.get(top_rated_horse, '100')
    top_ml_dec = str_to_decimal_odds(top_ml_str) or 101
    top_live_odds = df_final_field[df_final_field["Horse"] == top_rated_horse]["Live Odds"].values
    top_live_dec = str_to_decimal_odds(str(top_live_odds[0])) if len(top_live_odds) > 0 and top_live_odds[0] else top_ml_dec
    
    most_likely_section = f"### 🎯 Most Likely Winner (Model Prediction)\n"
    most_likely_section += f"**#{name_to_post.get(top_rated_horse, '?')} - {top_rated_horse}**\n\n"
    most_likely_section += f"* **Model Probability:** {top_prob*100:.1f}% (highest among all runners)\n"
    most_likely_section += f"* **Morning Line:** {top_ml_str} (dec: {top_ml_dec:.2f})\n"
    most_likely_section += f"* **Current Live Odds:** {top_live_dec:.2f} (dec)\n"
    
    if is_underlay:
        most_likely_section += f"* **Betting Status:** 🔴 **UNDERLAY** - Odds are shorter than fair value. Use primarily under other horses in exotics; careful on win bets.\n"
    elif is_overlay:
        most_likely_section += f"* **Betting Status:** 🟢 **OVERLAY** - Odds are better than fair value. Strong value play; consider win and exacta top.\n"
    else:
        most_likely_section += f"* **Betting Status:** 🟡 **FAIR** - Odds approximately match model probability.\n"
    
    most_likely_section += f"\n**Why This Horse Wins Most Often (Per Model):**\n"
    
    # Build reasoning from top horse's data
    top_horse_row = primary_df[primary_df["Horse"] == top_rated_horse].iloc[0] if not primary_df[primary_df["Horse"] == top_rated_horse].empty else None
    if top_horse_row is not None:
        reasoning = []
        
        # Check R rating components
        if top_horse_row.get("Cclass", 0) > 0:
            reasoning.append(f"✓ Strong class rating ({top_horse_row['Cclass']:.2f}) - fits field well")
        
        if top_horse_row.get("Cstyle", 0) > 0.05:
            reasoning.append(f"✓ Excellent style match ({top_horse_row['Cstyle']:.2f}) - track bias favors running style")
        
        if top_horse_row.get("Cpace", 0) > 0.1:
            reasoning.append(f"✓ Pace tailwind ({top_horse_row['Cpace']:.2f}) - field setup helps this horse")
        
        if not pd.isna(top_horse_row.get("LastFig")) and top_horse_row.get("LastFig", 0) > 80:
            reasoning.append(f"✓ Strong recent speed figure ({top_horse_row['LastFig']:.0f}) - racing in form")
        
        if top_horse_row.get("Atrack", 0) > 0.05:
            reasoning.append(f"✓ Track bias advantage ({top_horse_row['Atrack']:.2f}) - specific track/distance bias")
        
        # Check jockey/trainer data
        jt_data = jock_train_per_horse.get(top_rated_horse, {})
        if jt_data.get("jock_win_pct", 0) > 0.20:
            reasoning.append(f"✓ Elite jockey ({jt_data['jock_win_pct']*100:.0f}% win rate) - top rider")
        
        if jt_data.get("trainer_roi", 1.0) > 1.05:
            reasoning.append(f"✓ Positive trainer ROI ({jt_data['trainer_roi']:.2f}) - trainer adds value")
        
        if reasoning:
            for r in reasoning[:4]:  # Top 4 reasons
                most_likely_section += f"{r}\n"
        else:
            most_likely_section += f"✓ Best overall rating ({primary_df['R'].max():.2f}) among field\n"
    
    most_likely_section += f"\n**Confidence Level:** {'HIGH' if top_prob > 0.25 else 'MEDIUM' if top_prob > 0.15 else 'MODERATE'} ({top_prob*100:.0f}% model probability)\n"
    contender_report += f"\n{most_likely_section}"
    # Add trainer intent note if strong signals detected
    if trainer_intent_per_horse.get(top_rated_horse, {}).get("roi_angles", 0) > 2:
        contender_report += f"\n**Trainer Intent Note:** Strong signals (ROI {trainer_intent_per_horse[top_rated_horse]['roi_angles']:.1f}) indicate win today.\n"

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
    
    # --- 7. Generate Dynamic Exotic Probabilities ---
    exotic_outcomes = dynamic_exotic_probs(primary_probs, ppi_val, primary_df, sims=MODEL_CONFIG['exotic_sims'])
    probables = extract_probable_positions(exotic_outcomes)
    bet_outcomes_report = generate_bet_outcomes(probables)
    final_report += f"\n{bet_outcomes_report}"
    
    # --- 8. Generate Optimized Exotic Tickets ---
    if offered_odds_map:
        ticket_report = optimize_tickets(exotic_outcomes, probables, offered_odds_map)
        final_report += f"\n### Optimized Exotic Tickets\n{ticket_report}\n* **Common Sense:** Hammer underlays (low odds, high prob) on top for exacta/tri; spread overlays (value) for super/SH5 bombs.\n"
    
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
            
            # Build offered odds map from live odds or morning line
            offered_odds_map = {}
            for _, row in primary_df.iterrows():
                h = str(row.get("Horse", ""))
                if h:
                    live_odds_str = str(df_final_field[df_final_field["Horse"] == h]["Live Odds"].values[0] if not df_final_field[df_final_field["Horse"] == h].empty else "")
                    ml_odds_str = name_to_ml.get(h, "")
                    odds_str = live_odds_str if live_odds_str and live_odds_str != "" else ml_odds_str
                    dec_odds = str_to_decimal_odds(odds_str) if odds_str else 5.0
                    offered_odds_map[h] = dec_odds if dec_odds else 5.0

            # --- 2. NEW: Generate Simplified A/B/C/D Strategy Report ---
            strategy_report_md = build_betting_strategy(
                primary_df, df_ol, strategy_profile, name_to_post, name_to_ml, field_size, ppi_val, offered_odds_map
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
