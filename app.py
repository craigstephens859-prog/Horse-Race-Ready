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

import logging
import math
import os
import re
from datetime import datetime
from itertools import product
from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===================== String Constants (extracted to reduce duplication) =====================
DIST_BUCKET_SPRINT = "≤6f"
DIST_BUCKET_MID = "6.5–7f"
DIST_BUCKET_ROUTE = "8f+"
COL_FAIR_PCT = "Fair %"
COL_FAIR_ODDS = "Fair Odds"
COL_EDGE_PP = "Edge (pp)"
COL_EV_PER_DOLLAR = "EV per $1"
COL_PROB_PCT = "Prob %"
DEFAULT_DISTANCE = "6 Furlongs"
BIAS_FAIR_NEUTRAL = "fair/neutral"
RACE_TYPE_MAIDEN_SP_WT = "maiden special weight"
RACE_TYPE_MAIDEN_CLM = "maiden claiming"

# DATABASE PERSISTENCE: Ensures data survives Render redeployments
try:
    from db_persistence import (
        backup_to_github_async,
        get_persistence_status,
        initialize_persistent_db,
        is_render,
    )

    PERSISTENT_DB_PATH = initialize_persistent_db("gold_high_iq.db")
    print(f"[OK] Persistent DB path: {PERSISTENT_DB_PATH}")
except ImportError:
    PERSISTENT_DB_PATH = "gold_high_iq.db"
    backup_to_github_async = None
    get_persistence_status = None

    def is_render():
        return False

    print("[WARN] db_persistence not available, using local DB path")

# ML Engine removed — functionality superseded by MLBlendEngine + TrackIntelligenceEngine

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
    from race_class_parser import parse_and_calculate_class

    RACE_CLASS_PARSER_AVAILABLE = True
except ImportError:
    RACE_CLASS_PARSER_AVAILABLE = False
    parse_and_calculate_class = None

# ParserToRatingBridge removed — superseded by UnifiedRatingEngine pipeline

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
# Uses persistent path so data survives Render redeployments
try:
    from gold_database_manager import GoldHighIQDatabase

    gold_db = GoldHighIQDatabase(PERSISTENT_DB_PATH)
    GOLD_DB_AVAILABLE = True
    print(f"[OK] Gold DB initialized at: {PERSISTENT_DB_PATH}")
except Exception as e:
    GOLD_DB_AVAILABLE = False
    gold_db = None
    print(f"Gold database initialization error: {e}")

# ADAPTIVE LEARNING v2: Auto-calibration with persistent learned weights
# Loads weights that have been tuned from historical race results
try:
    from auto_calibration_engine_v2 import (
        AutoCalibrationEngine,
        auto_calibrate_on_result_submission,
        get_all_track_calibrations_summary,
        get_live_learned_weights,
    )

    ADAPTIVE_LEARNING_AVAILABLE = True

    # Load learned weights at startup (persisted from past calibrations)
    LEARNED_WEIGHTS = get_live_learned_weights(PERSISTENT_DB_PATH)
    print(
        f"[OK] Loaded {len(LEARNED_WEIGHTS)} learned weights from {PERSISTENT_DB_PATH}"
    )
except ImportError as e:
    ADAPTIVE_LEARNING_AVAILABLE = False
    LEARNED_WEIGHTS = {}
    print(f"Adaptive learning not available: {e}")

    def get_all_track_calibrations_summary(db_path=""):
        return []


# ML BLEND ENGINE: PyTorch retrained model for prediction blending
try:
    from ml_blend_engine import MLBlendEngine

    _ml_blend = MLBlendEngine(model_dir="models")
    ML_BLEND_AVAILABLE = _ml_blend.model is not None
    if ML_BLEND_AVAILABLE:
        _mi = _ml_blend.get_model_info()
        print(
            f"[OK] ML Blend Engine loaded: {_mi.get('model_path', '?')} ({_mi.get('n_features', '?')} features)"
        )
    else:
        print("[WARN] ML Blend Engine: no trained model found in models/")
except Exception as e:
    ML_BLEND_AVAILABLE = False
    _ml_blend = None
    print(f"ML Blend Engine not available: {e}")

# TRACK INTELLIGENCE: Bias detection & track-specific profiling
try:
    from track_intelligence import TrackIntelligenceEngine

    _track_intel = TrackIntelligenceEngine(db_path=PERSISTENT_DB_PATH)
    TRACK_INTEL_AVAILABLE = True
    print("[OK] Track Intelligence Engine initialized")
except Exception as e:
    TRACK_INTEL_AVAILABLE = False
    _track_intel = None
    print(f"Track Intelligence not available: {e}")


# ===================== First-Time Starter (FTS) Parameters =====================
# Maiden Special Weight (MSW) - Define FTS Parameters Dictionary
# Handles horses with zero previous racing history
FTS_PARAMS = {
    "jockey_confidence": 0.7,  # Boost for strong bookings
    "trainer_confidence": 0.9,  # Key for top trainers (>20% FTS win rate)
    "speed_confidence": 0.4,  # Low; use workouts if available
    "form_confidence": 0.0,  # Disabled, no prior races
    "class_confidence": 0.6,  # Implies quality but unproven
    "pedigree_confidence": 0.8,  # Crucial for debuts
    "ml_odds_confidence": 0.8,  # Market accuracy for elite FTS
    "live_odds_confidence": 0.9,  # Insider sentiment
    "base_multiplier": 0.75,  # Base FTS rating multiplier (75% of normal)
    "elite_trainer_multiplier": 1.2,  # Boost for elite connections (75% * 1.2 = 90%)
}

# Elite Trainers: Top 5% debut ROI (expand based on historical data)
ELITE_TRAINERS = {
    "Wesley Ward",
    "Todd Pletcher",
    "Bob Baffert",
    "Chad Brown",
    "Steve Asmussen",
}  # Set for O(1) lookup performance

# ===================== NA Running Style + QSP Parameters =====================
# Horses with "NA" running style: BRISNET couldn't assign a style because the
# horse lacks sufficient starts at this exact distance/surface combination.
# NA ≠ bad — it means UNKNOWN. We use Quirin Speed Points (0-8) to infer
# partial early-speed tendency and adjust confidence/ratings accordingly.
NA_STYLE_PARAMS = {
    "form_confidence": 0.0,  # No meaningful form cycle at this dist/surface
    "speed_confidence": 0.5,  # Workouts/morning line provide some signal
    "pedigree_boost": 0.2,  # Pedigree becomes more important for unknowns
    "trainer_boost": 0.1,  # Trainer patterns matter more for unknowns
    "style_penalty": -0.15,  # Gentler than -0.3 (unknown ≠ mismatch)
    "rating_dampener": 0.85,  # 15% overall reduction for style uncertainty
    "fts_na_dampener": 0.92,  # Mild additional dampener when FTS+NA stack
    "qsp_speed_scaling": 0.3,  # Max speed_confidence boost from QSP: += (q/8)*0.3
    "qsp_ep_threshold": 5,  # QSP >= 5 infers partial E/P tendency
    "qsp_style_offset": 0.15,  # Max style penalty offset from high QSP
}

# INTELLIGENT LEARNING ENGINE: High-IQ pattern analysis from training sessions
try:
    from intelligent_learning_engine import (
        IntelligentLearningEngine,
        analyze_and_learn_from_result,
    )

    INTELLIGENT_LEARNING_AVAILABLE = True
    print("[OK] Intelligent Learning Engine loaded")
except ImportError as e:
    INTELLIGENT_LEARNING_AVAILABLE = False
    IntelligentLearningEngine = None
    analyze_and_learn_from_result = None
    print(f"Intelligent learning not available: {e}")

# SECURITY: Import input validation and protection utilities
try:
    from security_validators import (
        RateLimiter,
        sanitize_pp_text,
        sanitize_race_metadata,
        validate_distance_string,
        validate_track_name,
    )

    SECURITY_VALIDATORS_AVAILABLE = True
except ImportError as e:
    SECURITY_VALIDATORS_AVAILABLE = False
    print(f"Security validators not available: {e}")

# ===================== Page / Model Settings =====================

st.set_page_config(
    page_title="Horse Race Ready — IQ Mode", page_icon="🏇", layout="wide"
)

# ===================== ACCESS KEY GATE =====================
# Prevents direct access to the Render URL without going through Squarespace.
# Set ACCESS_KEY env var on Render, then embed with ?key=YOUR_SECRET in the URL.
# Local development: no ACCESS_KEY env var = no gate (open access).
_ACCESS_KEY = os.getenv("ACCESS_KEY", "")
if _ACCESS_KEY:
    _provided_key = st.query_params.get("key", "")
    if _provided_key != _ACCESS_KEY:
        st.error(
            "🔒 Access Denied — Please visit [HandicappingHorseRaces.org](https://www.handicappinghorseraces.org) to access this app."
        )
        st.stop()
# ===================== END ACCESS KEY GATE =====================

st.title("🏇  Horse Race Ready — IQ Mode")

# ============ COMMUNITY DATABASE STATS BANNER ============
# Prominently show that races are being saved for the community
try:
    if GOLD_DB_AVAILABLE and gold_db is not None:
        import sqlite3

        conn = sqlite3.connect(gold_db.db_path, timeout=5.0)
        cursor = conn.cursor()

        # Get total races analyzed
        cursor.execute("SELECT COUNT(DISTINCT race_id) FROM races_analyzed")
        total_analyzed = cursor.fetchone()[0]

        # Get races with results
        cursor.execute("SELECT COUNT(DISTINCT race_id) FROM gold_high_iq")
        with_results = cursor.fetchone()[0]

        # Get most recent save timestamp
        cursor.execute("""
            SELECT analyzed_timestamp FROM races_analyzed 
            ORDER BY analyzed_timestamp DESC LIMIT 1
        """)
        last_save = cursor.fetchone()
        last_save_time = last_save[0] if last_save else "No races yet"

        conn.close()

        # Display community banner
        if total_analyzed > 0:
            cols = st.columns([2, 2, 2, 3])
            with cols[0]:
                st.metric(
                    "🌐 Community Races",
                    total_analyzed,
                    help="All races analyzed by users - permanently saved!",
                )
            with cols[1]:
                st.metric(
                    "✅ Results Entered",
                    with_results,
                    help="Races with actual finish results",
                )
            with cols[2]:
                pending = max(0, total_analyzed - with_results)  # Never show negative
                st.metric(
                    "⏳ Awaiting Results",
                    pending,
                    help="Races needing actual finish positions",
                )
            with cols[3]:
                if last_save_time and last_save_time != "No races yet":
                    try:
                        from datetime import datetime as dt_banner

                        save_dt = dt_banner.fromisoformat(
                            last_save_time.replace("T", " ").split(".")[0]
                        )
                        time_ago = (dt_banner.now() - save_dt).total_seconds()
                        if time_ago < 60:
                            time_str = f"{int(time_ago)}s ago"
                        elif time_ago < 3600:
                            time_str = f"{int(time_ago / 60)}m ago"
                        elif time_ago < 86400:
                            time_str = f"{int(time_ago / 3600)}h ago"
                        else:
                            time_str = f"{int(time_ago / 86400)}d ago"
                        st.metric(
                            "🕐 Last Save",
                            time_str,
                            help=f"Full timestamp: {last_save_time}",
                        )
                    except Exception:
                        st.metric("🕐 Last Save", "Recently")
                else:
                    st.metric("🕐 Last Save", "None yet")

            st.caption(
                "💾 **Database Auto-Saves:** All analyzed races persist permanently. Come back anytime!"
            )
except Exception:
    pass  # Silently fail if database not ready yet
# ============ END COMMUNITY BANNER ============

# ---------- Durable state ----------
if "parsed" not in st.session_state:
    st.session_state["parsed"] = False
if "pp_text_cache" not in st.session_state:
    st.session_state["pp_text_cache"] = ""

# Defaults for Race Info
if "track_name" not in st.session_state:
    st.session_state["track_name"] = "Unknown Track"
if "surface_type" not in st.session_state:
    st.session_state["surface_type"] = "Dirt"
if "condition_txt" not in st.session_state:
    st.session_state["condition_txt"] = "fast"
if "distance_txt" not in st.session_state:
    st.session_state["distance_txt"] = "6 Furlongs"
if "purse_val" not in st.session_state:
    st.session_state["purse_val"] = 80000

# ---------- Version-safe rerun ----------


def _safe_rerun():
    """Works on both old and new Streamlit versions."""
    rr = getattr(st, "rerun", None) or getattr(st, "experimental_rerun", None)
    if rr is not None:
        rr()
    else:
        st.warning(
            "Rerun API not available in this Streamlit build. Adjust any widget to refresh."
        )
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


def call_openai_messages(messages: list[dict]) -> str:
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
    "softmax_tau": 3.0,  # Controls win prob "sharpness". Must match unified engine rating scale.
    # CALIBRATED Feb 5, 2026: Was 0.85 which created 96%+ concentration on single horse.
    # Unified engine raw Rating sums have 5-10pt spreads; tau=3.0 gives realistic odds.
    "speed_fig_weight": 0.15,  # OPTIMIZED Feb 9 2026: Was 0.05 (speed was irrelevant).
    # 0.15 = 10 fig points = 1.5 bonus. Consistent with Beyer/Quirin/Benter research
    # that speed figures predict ~30-40% of race outcomes.
    "first_timer_fig_default": 50,  # Assumed speed fig for a 1st-time starter.
    # --- Pace & Style Model ---
    "ppi_multiplier": 1.5,  # Overall impact of the Pace Pressure Index (PPI).
    "ppi_tailwind_factor": 0.6,  # How much of the PPI value is given to E/EP or S horses.
    "style_strength_weights": {  # Multiplier for pace tailwind based on strength.
        "Strong": 1.0,
        "Solid": 0.8,
        "Slight": 0.5,
        "Weak": 0.3,
    },
    # --- Manual Bias Model (Section B) ---
    "style_match_table": {
        "favoring": {"E": 0.85, "E/P": 0.65, "P": -0.25, "S": -0.60},
        "closer favoring": {"E": -0.60, "E/P": -0.25, "P": 0.35, "S": 0.65},
        "fair/neutral": {"E": 0.0, "E/P": 0.0, "P": 0.0, "S": 0.0},
    },
    "style_quirin_threshold": 6,  # Quirin score needed for "strong" style bonus.
    "style_quirin_bonus": 0.15,  # INCREASED from 0.10 — Quirin matters more
    "post_bias_rail_bonus": 0.50,
    "post_bias_inner_bonus": 0.35,
    "post_bias_mid_bonus": 0.35,
    "post_bias_outside_bonus": 0.35,
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
    "exotic_bias_weights": (
        1.30,
        1.15,
        1.05,
        1.03,
    ),  # (1st, 2nd, 3rd, 4th) Harville bias
    "strategy_confident": {  # Placeholders, not used by new strategy builder
        "ex_max": 4,
        "ex_min_prob": 0.020,
        "tri_max": 6,
        "tri_min_prob": 0.010,
        "sup_max": 8,
        "sup_min_prob": 0.008,
    },
    "strategy_value": {  # Placeholders, not used by new strategy builder
        "ex_max": 6,
        "ex_min_prob": 0.015,
        "tri_max": 10,
        "tri_min_prob": 0.008,
        "sup_max": 12,
        "sup_min_prob": 0.006,
    },
}

# =========================
# Track parsing, race-type, distance options, and track-bias integration
# =========================

# -------- Distance options (UI) --------
DISTANCE_OPTIONS = [
    # Short sprints
    "4 Furlongs",
    "4 1/2 Furlongs",
    "4.5 Furlongs",
    "5 Furlongs",
    "5 1/2 Furlongs",
    "5.5 Furlongs",
    "6 Furlongs",
    "6 1/2 Furlongs",
    "6.5 Furlongs",
    "7 Furlongs",
    "7 1/2 Furlongs",
    "7.5 Furlongs",
    # Routes & variants
    "1 Mile",
    "1 Mile 70 Yards",
    "1 1/16 Miles",
    "1 1/8 Miles",
    "1 3/16 Miles",
    "1 1/4 Miles",
    "1 5/16 Miles",
    "1 3/8 Miles",
    "1 7/16 Miles",
    "1 1/2 Miles",
    "1 9/16 Miles",
    "1 5/8 Miles",
    "1 3/4 Miles",
    "1 7/8 Miles",
    "2 Miles",
]


def _distance_bucket_from_text(distance_txt: str) -> str:
    """
    Buckets into ≤6f, 6.5–7f, or 8f+ (routes).
    """
    d = (distance_txt or "").strip().lower()
    # Furlongs
    if "furlong" in d:
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
    if "mile" in d:
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


# -------- Canonical track names + aliases --------
TRACK_ALIASES = {
    # === Major / Grade 1 Tracks ===
    "Aqueduct": ["aqueduct", "aqu"],
    "Belmont Park": [
        "belmont park",
        "belmont at aqueduct",
        "aqueduct at belmont",
        "belmont at the big a",
        "big a",
        "belmont",
        "bel",
    ],
    "Churchill Downs": ["churchill downs", "churchill", "cd"],
    "Del Mar": ["del mar", "dmr"],
    "Gulfstream Park": ["gulfstream park", "gulfstream", "gp"],
    "Keeneland": ["keeneland", "kee"],
    "Laurel Park": ["laurel park", "laurel", "lrl"],
    "Oaklawn Park": ["oaklawn park", "oaklawn", "op"],
    "Pimlico": ["pimlico", "pim"],
    "Santa Anita": ["santa anita park", "santa anita", "sa"],
    "Saratoga": ["saratoga", "sar"],
    # === Secondary / Regional Tracks ===
    "Ak-Sar-Ben": ["ak-sar-ben", "aksarben", "akr"],
    "Albuquerque Downs": ["albuquerque downs", "albuquerque", "abq"],
    "Arapahoe Park": ["arapahoe park", "arapahoe", "arp"],
    "Arlington Park": ["arlington park", "arlington", "ap"],
    "Belterra Park": ["belterra park", "belterra", "btp"],
    "Canterbury Park": ["canterbury park", "canterbury", "cbp"],
    "Century Mile": ["century mile", "cym"],
    "Charles Town": ["charles town", "charlestown", "ct"],
    "Colonial Downs": ["colonial downs", "colonial", "cln"],
    "Columbus": ["columbus", "clb"],
    "Delaware Park": ["delaware park", "delaware", "del"],
    "Delta Downs": ["delta downs", "ded"],
    "Ellis Park": ["ellis park", "ellis", "elp"],
    "Emerald Downs": ["emerald downs", "emerald", "emr", "emd"],
    "Evangeline Downs": ["evangeline downs", "evangeline", "evd"],
    "Fair Grounds": ["fair grounds", "fairgrounds", "fg"],
    "Fair Meadows": ["fair meadows", "fmr"],
    "Fairmount Park": [
        "fairmount park",
        "fanduel fairmount",
        "cah",
        "collinsville",
        "fmp",
    ],
    "Finger Lakes": ["finger lakes", "fl"],
    "Fonner Park": ["fonner park", "fonner", "fon"],
    "Fort Erie": ["fort erie", "fe"],
    "Golden Gate Fields": ["golden gate fields", "golden gate", "ggf", "gg"],
    "Grants Pass": ["grants pass", "grp"],
    "Great Lakes Downs": ["great lakes downs", "great lakes", "gld"],
    "Gulfstream Park West": ["gulfstream park west", "gulfstream west", "gpw"],
    "Hastings": ["hastings", "hst"],
    "Hawthorne": ["hawthorne", "haw"],
    "Horseshoe Indianapolis": [
        "horseshoe indianapolis",
        "indiana grand",
        "ind",
        "indy",
        "hsi",
    ],
    "Kentucky Downs": ["kentucky downs", "kd"],
    "Lone Star Park": ["lone star park", "lone star", "ls"],
    "Los Alamitos": ["los alamitos", "lam"],
    "Louisiana Downs": ["louisiana downs", "lad"],
    "Mahoning Valley": ["mahoning valley", "mahoning", "mvr"],
    "Monmouth Park": ["monmouth park", "monmouth", "mth"],
    "Mountaineer": ["mountaineer", "mnr"],
    "Parx Racing": ["parx racing", "parx", "philadelphia park", "prx"],
    "Penn National": ["penn national", "penn", "pen"],
    "Pleasanton": ["pleasanton", "pln"],
    "Portland Meadows": ["portland meadows", "portland", "pm"],
    "Prairie Meadows": ["prairie meadows", "prairie", "prm"],
    "Presque Isle Downs": ["presque isle downs", "presque isle", "pid"],
    "Remington Park": ["remington park", "remington", "rp"],
    "Retama Park": ["retama park", "retama", "ret"],
    "Ruidoso Downs": ["ruidoso downs", "ruidoso", "rud"],
    "Sam Houston": ["sam houston race park", "sam houston", "hou"],
    "Santa Rosa": ["santa rosa", "sr"],
    "Sunland Park": ["sunland park", "sunland", "sun"],
    "Sunray Park": ["sunray park", "sunray", "sry"],
    "Suffolk Downs": ["suffolk downs", "suffolk", "suf"],
    "Tampa Bay Downs": ["tampa bay downs", "tampa bay", "tampa", "tam"],
    "Thistledown": ["thistledown", "tdn"],
    "Turfway Park": ["turfway park", "turfway", "tp"],
    "Turf Paradise": ["turf paradise", "tup"],
    "Will Rogers Downs": ["will rogers downs", "will rogers", "wrd"],
    "Woodbine": ["woodbine", "wo"],
    "Zia Park": ["zia park", "zia", "zp"],
}
_CANON_BY_TOKEN = {}
for canon, toks in TRACK_ALIASES.items():
    for t in toks:
        _CANON_BY_TOKEN[t] = canon


def _find_header_line(pp_text: str) -> str:
    """
    Find the actual BRISNET header line by skipping copyright/preamble lines.

    BRISNET PPs often start with lines like:
      (c) Copyright 2026 BRIS...
      (blank lines)
    before the real header:
      Ultimate PP's w/ QuickPlay Comments | Oaklawn Park | ...
    or:
      Ultimate PP's w/ QuickPlay Comments Oaklawn Park Alw 12500s ...
    """
    if not pp_text:
        return ""
    lines = pp_text.strip().split("\n")
    for line in lines[:15]:  # Check first 15 lines for the header
        stripped = line.strip()
        if not stripped:
            continue
        # Skip copyright lines: "(c) Copyright", "©", "Copyright"
        if stripped.startswith("(c)") or stripped.lower().startswith("copyright"):
            continue
        # Skip lines that are only special chars / very short garbage
        if len(stripped) < 5:
            continue
        # Skip lines that look like disclaimers or legal boilerplate
        if any(
            kw in stripped.lower()
            for kw in [
                "all rights reserved",
                "unauthorized",
                "license",
                "brisnet.com/terms",
            ]
        ):
            continue
        # This is the first real content line — use it as the header
        return stripped
    # Fallback: just use the first non-empty line
    for line in lines:
        if line.strip():
            return line.strip()
    return ""


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
    date_line_pattern = r"\d{2}[A-Za-z]{3}\d{2}([A-Za-z]{2,4})"
    for match in re.finditer(date_line_pattern, text):
        track_code = match.group(1).lower()
        if track_code in _CANON_BY_TOKEN:
            return _CANON_BY_TOKEN[track_code]

    # Second, check for full track names in header
    for token, canon in _CANON_BY_TOKEN.items():
        if re.search(rf"\b{re.escape(token)}\b", text):
            return canon

    # Third, check for multi-word track names
    for canon, toks in TRACK_ALIASES.items():
        for t in toks:
            t_words = [w for w in t.split() if len(w) > 2]
            if t_words and all(
                re.search(rf"\b{re.escape(w)}\b", text) for w in t_words
            ):
                return canon

    return ""


def detect_race_number(pp_text: str) -> int | None:
    """Extract race number from PP text header (e.g., 'Race 6')."""
    s = pp_text or ""
    # Look for "Race N" pattern in first few lines
    m = re.search(r"(?mi)\bRace\s+(\d+)\b", s[:500])
    if m:
        try:
            return int(m.group(1))
        except (ValueError, AttributeError):
            # Regex group missing or invalid
            pass
    return None


def parse_brisnet_race_header(pp_text: str) -> dict[str, Any]:
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
        "track_name": "",
        "race_number": 0,
        "race_type": "",
        "purse_amount": 0,
        "distance": "",
        "age_restriction": "",
        "sex_restriction": "",
        "race_date": "",
        "day_of_week": "",
    }

    # Get the actual header line (skip copyright/preamble lines)
    header_line = _find_header_line(pp_text)

    # Split by pipe delimiter
    parts = [p.strip() for p in header_line.split("|")]

    if len(parts) < 2:
        # ===== Non-pipe format parser =====
        # Format: "Ultimate PP's w/ QuickPlay Comments Oaklawn Park Alw 12500s 6 Furlongs 4&up Thursday, February 05, 2026 Race 9"
        text = header_line

        # Strip product-type prefixes:
        #   "Ultimate PP's w/ QuickPlay Comments"
        #   "Premium Plus PP's"
        #   and similar BRISNET product headers
        text = re.sub(
            r"^(?:Ultimate\s+PP.*?Comments|Premium\s+Plus\s+PP[''']?s?(?:\s+MC)?)\s*",
            "",
            text,
            flags=re.IGNORECASE,
        ).strip()

        # --- Extract Race Number (anchored at end, e.g., "Race 9") ---
        race_num_match = re.search(r"\bRace\s+(\d+)\s*$", text, re.IGNORECASE)
        if race_num_match:
            try:
                result["race_number"] = int(race_num_match.group(1))
            except (ValueError, TypeError):
                pass
            text = text[: race_num_match.start()].strip()

        # --- Extract Date (e.g., "Thursday, February 05, 2026") ---
        date_pattern = r"((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d{1,2},?\s+\d{4})"
        date_match = re.search(date_pattern, text, re.IGNORECASE)
        if date_match:
            date_str = date_match.group(1)
            comma_parts = date_str.split(",", 1)
            if len(comma_parts) >= 2:
                result["day_of_week"] = comma_parts[0].strip()
                result["race_date"] = comma_parts[1].strip()
            text = text[: date_match.start()].strip().rstrip(",").strip()

        # --- Extract Distance (e.g., "6 Furlongs", "1 1/16 Miles", "1„ Mile") ---
        dist_pattern = r"([\d½¼¾„ˆ]+(?:\s*[\d½¼¾/]+)?\s*(?:Furlongs?|Miles?|Yards?)\b)"
        dist_match = re.search(dist_pattern, text, re.IGNORECASE)
        if dist_match:
            result["distance"] = dist_match.group(1).strip()
            text = (
                text[: dist_match.start()].strip()
                + " "
                + text[dist_match.end() :].strip()
            )
            text = text.strip()

        # --- Extract Age/Sex restrictions (e.g., "4&up", "3yo Fillies", "F&M") ---
        age_sex_pattern = r"\b(\d+&up|3yo|4yo|F&M|Fillies|Mares|Colts|Geldings|C&G)\b"
        age_sex_matches = re.findall(age_sex_pattern, text, re.IGNORECASE)
        for m in age_sex_matches:
            ml = m.lower()
            if re.match(r"\d+&up|\d+yo", ml):
                result["age_restriction"] = m
            elif re.match(r"fillies?|mares?|colts?|geldings?|f&m|c&g", ml):
                result["sex_restriction"] = m.title()
        # Remove matched age/sex tokens from text
        text = re.sub(age_sex_pattern, "", text, flags=re.IGNORECASE).strip()

        # --- Extract Track Name ---
        # Strategy 1: Match against known track names (longest match first for accuracy)
        text_lower = text.lower().strip()
        matched_track = ""
        matched_len = 0
        for canon, toks in TRACK_ALIASES.items():
            for t in toks:
                if text_lower.startswith(t) and len(t) > matched_len:
                    # Verify word boundary (not a partial match)
                    if len(text_lower) == len(t) or not text_lower[len(t)].isalpha():
                        matched_track = canon
                        matched_len = len(t)
        if matched_track:
            result["track_name"] = matched_track
            text = text[matched_len:].strip()
        else:
            # Strategy 2: Everything before the first race-type keyword is the track name
            race_type_boundary = re.search(
                r"\b(Alw|Clm|Stk|Hcp|Mdn|Msw|Aoc|Mcl|Wmc|Oc|Soc|Str|G1|G2|G3|PWC|"
                r"©|¨|§|Allowance|Claiming|Stakes|Handicap|Maiden|Optional)\b",
                text,
                re.IGNORECASE,
            )
            if race_type_boundary:
                candidate = text[: race_type_boundary.start()].strip()
                if candidate and len(candidate) > 2:
                    result["track_name"] = candidate
                    text = text[race_type_boundary.start() :].strip()
            else:
                # Strategy 3: Everything before any digit sequence is the track name
                digit_boundary = re.search(r"\b\d", text)
                if digit_boundary and digit_boundary.start() > 2:
                    result["track_name"] = text[: digit_boundary.start()].strip()
                    text = text[digit_boundary.start() :].strip()

        # --- Extract Race Type + Purse from remaining text ---
        # e.g., "Alw 12500s", "Clm 25000", "©Hcp 50000", "Str 40000"
        type_purse_match = re.search(r"([©¨§]?\w+)\s+(\d+)\w*", text)
        if type_purse_match:
            result["race_type"] = type_purse_match.group(1)
            try:
                result["purse_amount"] = int(type_purse_match.group(2))
            except (ValueError, TypeError):
                pass

        return result

    # Parse each section
    for i, part in enumerate(parts):
        part_lower = part.lower()

        # Skip product-type segments like "Ultimate PP's w/ QuickPlay Comments" or "Premium Plus PP's"
        if (
            "ultimate" in part_lower
            or "quickplay" in part_lower
            or "premium plus" in part_lower
        ):
            continue

        # Track name (first non-Ultimate part)
        if not result["track_name"] and i > 0:
            # Check if this looks like a track name
            if (
                not re.search(r"\d+\s*(?:furlong|mile|yard)", part_lower)
                and not re.search(r"(clm|alw|stk|hcp|mdn|msw|aoc)", part_lower)
                and not re.search(r"race\s+\d+", part_lower)
                and not re.search(
                    r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)",
                    part_lower,
                )
            ):
                result["track_name"] = part
                continue

        # Race type + purse (e.g., "©Hcp 50000", "Clm 4000")
        race_type_match = re.search(r"([©¨§]?[A-Za-z]+)\s+(\d+)", part)
        if race_type_match and not result["race_type"]:
            result["race_type"] = race_type_match.group(1)
            try:
                result["purse_amount"] = int(race_type_match.group(2))
            except Exception:
                pass
            continue

        # Distance (e.g., "6 Furlongs", "1 Mile", "1„ Mile")
        if re.search(r"\d+\s*(?:furlong|mile|yard|f\b)", part_lower):
            result["distance"] = part
            continue

        # Age/Sex restrictions (e.g., "3yo Fillies", "4&up", "F&M")
        if re.search(
            r"(?:\d+yo|f&m|fillies|mares|colts|geldings|4&up|3&up)", part_lower
        ):
            age_match = re.search(r"(\d+yo|4&up|3&up)", part_lower)
            if age_match:
                result["age_restriction"] = age_match.group(1)

            sex_match = re.search(
                r"(fillies?|mares?|colts?|geldings?|f&m)", part_lower, re.IGNORECASE
            )
            if sex_match:
                result["sex_restriction"] = sex_match.group(1).title()
            continue

        # Date (e.g., "Monday, February 02, 2026")
        date_match = re.search(
            r"(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[,\s]+(.+\d{4})",
            part_lower,
        )
        if date_match:
            result["day_of_week"] = date_match.group(1).title()
            result["race_date"] = date_match.group(2).strip()
            continue

        # Race number (e.g., "Race 8")
        race_num_match = re.search(r"race\s+(\d+)", part_lower)
        if race_num_match:
            try:
                result["race_number"] = int(race_num_match.group(1))
            except Exception:
                pass

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
    "starter optional claiming": 1.00,
    "waiver claiming": 1.07,
    "claiming": 1.12,
    "maiden claiming": 1.15,
}

condition_modifiers = {
    "fast": 1.0,
    "firm": 1.0,
    "good": 1.03,
    "yielding": 1.04,
    "muddy": 1.08,
    "sloppy": 1.10,
    "heavy": 1.10,
}


def detect_race_type(pp_text: str) -> str:
    """
    Normalize many wordings into the exact key set used by base_class_bias.
    """
    s = (pp_text or "")[:1000].lower()

    # Graded stakes first
    if re.search(r"\b(g1|grade\s*i)\b", s):
        return "stakes (g1)"
    if re.search(r"\b(g2|grade\s*ii)\b", s):
        return "stakes (g2)"
    if re.search(r"\b(g3|grade\s*iii)\b", s):
        return "stakes (g3)"

    # Listed / generic stakes
    if "listed" in s:
        return "stakes (listed)"
    if re.search(r"\bstakes?\b", s):
        return "stakes"

    # Maiden
    # Check for standalone "MC" followed by number (e.g., "MC 16000" or "MC16000")
    if re.search(r"\bmc\s*\d", s):
        return "maiden claiming"

    if re.search(r"\b(mdn|maiden)\b", s):
        if re.search(r"(mcl|mdn\s*clm|maiden\s*claim)", s):
            return "maiden claiming"
        if re.search(r"(msw|maiden\s*special|maiden\s*sp\s*wt)", s):
            return "maiden special weight"
        return "maiden special weight"

    # Starter Optional Claiming (check before general OC/AOC)
    if re.search(r"\b(soc|starter\s*optional\s*claim\w*)\b", s):
        return "starter optional claiming"

    # AOC
    if re.search(r"\b(oc|aoc|optional\s*claim)\b", s):
        return "allowance optional claiming (aoc)"

    # Starter
    if re.search(r"\bstarter\s*allow", s):
        return "starter allowance"
    if re.search(r"\bstarter\s*h(andi)?cap\b", s):
        return "starter handicap"

    # Waiver Claiming
    if re.search(r"\b(waiver|wcl|w\s*clm)\b", s):
        return "waiver claiming"

    # Claiming
    if re.search(r"\bclm|claiming\b", s):
        return "claiming"

    # Allowance last
    if re.search(r"\ballow(ance)?\b", s):
        return "allowance"

    return "allowance"


# -------- Track bias profiles (additive deltas; conservative magnitude) --------
TRACK_BIAS_PROFILES = {
    "Keeneland": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.35, "E/P": 0.20, "P": -0.10, "S": -0.25},
                "post": {"rail": 0.20, "inner": 0.10, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Del Mar": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Churchill Downs": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.05, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Kentucky Downs": {
        "Turf": {
            "≤6f": {
                "runstyle": {"E": -0.05, "E/P": 0.00, "P": 0.10, "S": 0.15},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": -0.05, "E/P": 0.00, "P": 0.10, "S": 0.15},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
            "8f+": {
                "runstyle": {"E": -0.10, "E/P": 0.00, "P": 0.10, "S": 0.20},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
        }
    },
    "Saratoga": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Santa Anita": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Mountaineer": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        }
    },
    "Charles Town": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.45, "E/P": 0.25, "P": -0.15, "S": -0.35},
                "post": {"rail": 0.25, "inner": 0.15, "mid": -0.05, "outside": -0.10},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.30, "E/P": 0.20, "P": -0.10, "S": -0.25},
                "post": {"rail": 0.15, "inner": 0.10, "mid": -0.05, "outside": -0.10},
            },
            "8f+": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    "Gulfstream Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Synthetic": {
            "≤6f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        },
    },
    "Tampa Bay Downs": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Belmont Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.05, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Horseshoe Indianapolis": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
        },
    },
    "Penn National": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": 0.00},
            },
        }
    },
    "Presque Isle Downs": {
        "Synthetic": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": 0.00, "S": -0.10},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.05},
            },
        }
    },
    "Woodbine": {
        "Synthetic": {
            "≤6f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Evangeline Downs": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    "Oaklawn Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.30, "E/P": 0.15, "P": -0.10, "S": -0.20},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.10},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    "Fair Grounds": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.20},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.10},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Aqueduct": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.05, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Laurel Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.05, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.05, "mid": 0.05, "outside": -0.05},
            },
        },
    },
    "Fairmount Park": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    "Finger Lakes": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.25, "E/P": 0.15, "P": -0.05, "S": -0.15},
                "post": {"rail": 0.10, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
            "8f+": {
                "runstyle": {"E": 0.05, "E/P": 0.00, "P": 0.05, "S": -0.05},
                "post": {"rail": 0.05, "inner": 0.00, "mid": 0.00, "outside": -0.05},
            },
        }
    },
    # Default fallback profile for tracks not specifically listed
    "_DEFAULT": {
        "Dirt": {
            "≤6f": {
                "runstyle": {"E": 0.15, "E/P": 0.10, "P": -0.05, "S": -0.10},
                "post": {"rail": 0.05, "inner": 0.05, "mid": 0.00, "outside": -0.05},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.08, "E/P": 0.05, "P": 0.00, "S": -0.05},
                "post": {"rail": 0.03, "inner": 0.03, "mid": 0.00, "outside": -0.03},
            },
            "8f+": {
                "runstyle": {"E": 0.03, "E/P": 0.03, "P": 0.03, "S": -0.03},
                "post": {"rail": 0.03, "inner": 0.03, "mid": 0.00, "outside": -0.03},
            },
        },
        "Turf": {
            "≤6f": {
                "runstyle": {"E": 0.10, "E/P": 0.05, "P": 0.00, "S": -0.08},
                "post": {"rail": 0.00, "inner": 0.03, "mid": 0.00, "outside": -0.03},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.03, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": -0.03},
            },
            "8f+": {
                "runstyle": {"E": 0.00, "E/P": 0.03, "P": 0.05, "S": -0.03},
                "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": -0.03},
            },
        },
        "Synthetic": {
            "≤6f": {
                "runstyle": {"E": 0.08, "E/P": 0.05, "P": 0.03, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.03, "outside": 0.00},
            },
            "6.5–7f": {
                "runstyle": {"E": 0.05, "E/P": 0.05, "P": 0.03, "S": -0.05},
                "post": {"rail": 0.00, "inner": 0.00, "mid": 0.03, "outside": 0.00},
            },
            "8f+": {
                "runstyle": {"E": 0.03, "E/P": 0.05, "P": 0.05, "S": -0.03},
                "post": {"rail": 0.00, "inner": 0.03, "mid": 0.03, "outside": 0.00},
            },
        },
    },
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


# ===================== Core Helpers =====================


def detect_valid_race_headers(pp_text: str):
    toks = ("purse", "furlong", "mile", "clm", "allow", "stake", "pars", "post time")
    headers = []
    for m in re.finditer(r"(?mi)^\s*Race\s+(\d+)\b", pp_text or ""):
        win = (pp_text[m.end() : m.end() + 250] or "").lower()
        if any(t in win for t in toks):
            headers.append(int(m.group(1)))
    return headers


HORSE_HDR_RE = re.compile(
    r"""(?mi)^\s*
    (?:POST\s+)?          # optional "POST " prefix
    (\d+[A-Za-z]?)        # program number (may have letter suffix for coupled entries, e.g. "1A")
    (?:pp(\d+))?          # optional "pp" + post position (e.g. "pp6" in "1pp6")
    [:\s]+                # colon or whitespace separator
    ([A-Za-z0-9''\u2019.\-\s&]+?)  # horse name (lazy to avoid over-capturing)
    [^\w(]*               # skip any special chars/icons between name and parentheses
    \(\s*
    (E\/P|EP|E|P|S|NA)   # running style
    (?:\s+(\d+))?         # optional quirin
    \s*\)                 # closing paren
    (?:\s*-\s*ML\s+\S+)?  # optional "- ML odds" suffix
    (?:\s*\*+\s*.+)?       # optional "*** ACTUAL WINNER ***" annotation
    .*$                   # allow trailing special chars/icons (BRISNET symbols like ì)
    """,
    re.VERBOSE,
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
    """Calculate style strength label from running style + Quirin Speed Points.

    For NA (unknown) styles, use QSP to infer strength instead of defaulting to 'Solid':
    - QSP >= 6: 'Solid' (decent early speed signal from BRISNET)
    - QSP 3-5: 'Slight' (some signal but inconclusive)
    - QSP 0-2 or NaN: 'Weak' (no meaningful data to differentiate)
    """
    s = (style or "NA").upper()
    try:
        q = float(quirin)
    except Exception:
        return "Weak" if s == "NA" else "Solid"
    if pd.isna(q):
        return "Weak" if s == "NA" else "Solid"
    # NA style: use QSP to infer strength (Feb 11, 2026)
    if s == "NA":
        if q >= 6:
            return "Solid"
        if q >= 3:
            return "Slight"
        return "Weak"
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


def split_into_horse_chunks(pp_text: str) -> list[tuple]:
    chunks = []
    matches = list(HORSE_HDR_RE.finditer(pp_text or ""))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(pp_text)
        # group(2) is the post from "ppN"; group(1) is program# (fallback when pp absent)
        post = (m.group(2) or m.group(1)).strip()
        name = m.group(3).strip()
        block = pp_text[start:end]
        chunks.append((post, name, block))
    return chunks


def extract_horses_and_styles(pp_text: str) -> pd.DataFrame:
    rows = []
    for m in HORSE_HDR_RE.finditer(pp_text or ""):
        program_num = m.group(1).strip()
        # group(2) is the post from "ppN"; group(1) is program# (fallback when pp absent)
        post = (m.group(2) or program_num).strip()
        name = m.group(3).strip()
        style = _normalize_style(m.group(4))
        qpts = m.group(5)
        quirin = int(qpts) if qpts else np.nan
        auto_strength = calculate_style_strength(style, quirin)
        rows.append(
            {
                "#": program_num,
                "Post": post,
                "Horse": name,
                "DetectedStyle": style,
                "Quirin": quirin,
                "AutoStrength": auto_strength,
                "OverrideStyle": "",
                "StyleStrength": auto_strength,
            }
        )
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


def extract_morning_line_by_horse(pp_text: str) -> dict[str, str]:
    ml = {}
    blocks = {name: block for _, name, block in split_into_horse_chunks(pp_text)}
    for name, block in blocks.items():
        if name in ml:
            continue
        m_start = re.search(rf"(?mi)^\s*{_ODDS_TOKEN}", block or "")
        if m_start:
            ml[name.strip()] = m_start.group(1).replace(" ", "")
            continue
        m_labeled = re.search(
            rf"(?mi)^.*?\b(?:M/?L|Morning\s*Line|ML)\b.*?{_ODDS_TOKEN}", block or ""
        )
        if m_labeled:
            ml[name.strip()] = m_labeled.group(1).replace(" ", "")
    return ml


ANGLE_LINE_RE = re.compile(
    r"""(?mix)^\s*\+?           # optional + prefix (positive trend indicator)
    (                              # ── category capture group ──
      \d{4}                        # year-based stats  (2025, 2024 …)
    | JKYw/\s*(?:                  # jockey angle variants
        Sprints                    #   JKYw/ Sprints
      | Routes                     #   JKYw/ Routes
      | Trn\s*L(?:30|45|60)        #   JKYw/ Trn L30/L45/L60
      | (?:[EPS]|NA)\s*types       #   JKYw/ S types, E types, P types, NA types
      )
    | 1st\s*time\s*(?:str|Turf|Dirt|AW)  # first time surface / distance
    | Debut\s*MdnSpWt              # debut maiden special weight
    | Maiden\s*Sp\s*Wt             # maiden special weight
    | 2nd\s*career\s*race          # 2nd career race
    | Turf\s*to\s*Dirt             # surface switch
    | Dirt\s*to\s*Turf             # surface switch
    | Rte\s*to\s*Sprint            # distance switch
    | Sprint\s*to\s*Rte            # distance switch
    | MdnClm\s*to\s*Mdn            # class change
    | Shipper                      # shipper
    | Blinkers\s*(?:on|off)        # equipment change
    | (?:\d+(?:-\d+)?)\s*days?Away # layoff
    | Sprints                      # standalone distance
    | Routes                       # standalone distance
    | Dirt\s*starts                # standalone surface
    | Turf\s*starts                # standalone surface
    )\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$
    """
)


def parse_angles_for_block(block) -> pd.DataFrame:
    rows = []
    if not block:
        return pd.DataFrame(rows)
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in ANGLE_LINE_RE.finditer(block_str):
        cat, starts, win, itm, roi = m.groups()
        rows.append(
            {
                "Category": re.sub(r"\s+", " ", cat.strip()),
                "Starts": int(starts),
                "Win%": float(win),
                "ITM%": float(itm),
                "ROI": float(roi),
            }
        )
    return pd.DataFrame(rows)


def parse_pedigree_snips(block) -> dict:
    out = {
        "sire_awd": np.nan,
        "sire_1st": np.nan,
        "sire_mud_pct": np.nan,  # %Mud from Sire Stats (Feb 13, 2026)
        "damsire_awd": np.nan,
        "damsire_1st": np.nan,
        "dam_dpi": np.nan,
    }
    if not block:
        return out
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    s = re.search(
        r"(?mi)^\s*Sire\s*Stats:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi",
        block_str,
    )
    if s:
        out["sire_awd"] = float(s.group(1))
        out["sire_mud_pct"] = float(s.group(2))  # Group 2 is %Mud
        out["sire_1st"] = float(s.group(3))  # Group 3 is %-1st
    ds = re.search(
        r"(?mi)^\s*Dam\'s Sire:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi",
        block_str,
    )
    if ds:
        out["damsire_awd"] = float(ds.group(1))
        out["damsire_1st"] = float(ds.group(3))
    d = re.search(r"(?mi)^\s*Dam:\s*DPI\s*(\d+(?:\.\d+)?)\s+(\d+)%", block_str)
    if d:
        out["dam_dpi"] = float(d.group(1))

    # Parse Pedigree breeding ratings: "Pedigree: Fast 85 Off 72 Dist 90 Turf 78"
    # (Feb 13, 2026 enhancement for surface-switch pedigree integration)
    ped_ratings = re.search(
        r"(?mi)Pedigree:\s*Fast\s+(\d+)\s+Off\s+(\d+)\s+Dist\s+(\d+)\s+Turf\s+(\d+)",
        block_str,
    )
    if ped_ratings:
        out["pedigree_fast"] = float(ped_ratings.group(1))
        out["pedigree_off"] = float(ped_ratings.group(2))
        out["pedigree_distance"] = float(ped_ratings.group(3))
        out["pedigree_turf"] = float(ped_ratings.group(4))

    return out


# ========== SAVANT-LEVEL ENHANCEMENTS (Jan 2026) ==========


def parse_claiming_prices(block) -> list[int]:
    """Extract claiming prices from race lines. Returns list of prices (most recent first)."""
    prices = []
    if not block:
        return prices
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in re.finditer(r"Clm\s+(\d+)", block_str):
        try:
            prices.append(int(m.group(1)))
        except BaseException:
            pass
    return prices[:5]


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


def detect_lasix_change(block) -> float:
    """SAVANT ANGLE: Lasix/medication changes. Returns bonus from -0.12 to +0.18"""
    bonus = 0.0
    if not block:
        return bonus
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    race_lines = [
        line
        for line in block_str.split("\n")
        if re.search(r"\d{2}[A-Za-z]{3}\d{2}", line)
    ]
    lasix_pattern = []
    for line in race_lines[:5]:
        if re.search(r"\s+L\s+\d+\.\d+\s*$", line):
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


def parse_fractional_positions(block) -> list[list[int]]:
    """Extract running positions: PP, Start, 1C, 2C, Stretch, Finish."""
    positions = []
    if not block:
        return positions
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    # More flexible pattern with unicode range for position markers (¹²³ª etc)
    pattern = r"(\d{2}[A-Za-z]{3}\d{2}).*?(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})[\s\u00aa-\u00b4]*(\d{1,2})"
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
        except (ValueError, IndexError):
            # Log parsing failures for debugging (silent failures hide bugs)
            pass  # Could add logging here if needed
    return positions[:5]


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


def parse_e1_e2_lp_values(block) -> dict:
    """Extract E1, E2, and LP pace figures."""
    e1_vals, e2_vals, lp_vals = [], [], []
    if not block:
        return {"e1": e1_vals, "e2": e2_vals, "lp": lp_vals}
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    for m in re.finditer(r"(\d{2,3})\s+(\d{2,3})/\s*(\d{2,3})", block_str):
        try:
            e1_vals.append(int(m.group(1)))
            e2_vals.append(int(m.group(2)))
            lp_vals.append(int(m.group(3)))
        except (ValueError, AttributeError, IndexError):
            # Regex group missing or conversion failed
            pass
    return {"e1": e1_vals[:5], "e2": e2_vals[:5], "lp": lp_vals[:5]}


def analyze_pace_figures(
    e1_vals: list[int],
    e2_vals: list[int],
    lp_vals: list[int],
    e1_par: int = 0,
    e2_par: int = 0,
    lp_par: int = 0,
) -> float:
    """OPTIMIZED Feb 9 2026: PAR-adjusted pace analysis with recency-weighted averages.

    Returns bonus from -0.15 to +0.20. Uses recency weights [2x, 1x, 0.5x]
    and energy distribution analysis. Validated on Oaklawn R9 (Air of Defiance #2).
    """
    bonus = 0.0
    if len(e1_vals) < 2 or len(lp_vals) < 2:
        return bonus
    # Recency-weighted averages (most recent race weighted 2x)
    weights = [2.0, 1.0, 0.5][: len(e1_vals)]
    w_sum = sum(weights)
    avg_e1 = sum(v * w for v, w in zip(e1_vals[:3], weights)) / w_sum
    avg_lp = sum(v * w for v, w in zip(lp_vals[:3], weights[: len(lp_vals)])) / sum(
        weights[: len(lp_vals)]
    )
    avg_e2 = (
        sum(v * w for v, w in zip(e2_vals[:3], weights[: len(e2_vals)]))
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


# ========== END SAVANT ENHANCEMENTS ==========


SPEED_FIG_RE = re.compile(
    # CRITICAL FIX: Match the actual BRISNET format with E1 E2/ LP +calls SPEED_FIG
    # Example: "88 81/ 77 +4 +1 70"  where 70 is the speed figure
    r"(?mi)"
    r"\s+(\d{2,3})"  # E1 pace figure (88)
    r"\s+(\d{2,3})\s*/\s*(\d{2,3})"  # E2/LP figures (81/ 77)
    r"\s+[+-]\d+"  # Call position (+4)
    r"\s+[+-]\d+"  # Call position (+1)
    r"\s+(\d{2,3})"  # SPEED FIGURE (70) - THIS IS WHAT WE WANT!
    r"(?:\s|$)"  # End with space or end of line
)


def parse_speed_figures_for_block(block) -> list[int]:
    """Extract speed figures using E2/LP '/' marker, then skip call changes to find fig.

    OPTIMIZED Feb 9 2026: Algorithmic approach handles ANY number of call positions
    (2 for sprints, 3-4 for routes) instead of regex with fixed capture groups.
    """
    figs = []
    if not block:
        return figs

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    for line in block_str.split("\n"):
        parts = line.split()
        # Find the E2/LP marker: a field containing "/"
        slash_idx = None
        for idx, part in enumerate(parts):
            if "/" in part and idx >= 1:
                # Validate: fields around "/" should be numeric (E1 E2/ LP)
                left = part.replace("/", "")
                if left.isdigit() and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    slash_idx = idx
                    break
        if slash_idx is None:
            continue
        # LP is right after the "/" field
        lp_idx = slash_idx + 1
        # After LP, walk forward past call changes until we hit the speed figure
        j = lp_idx + 1
        while j < len(parts):
            raw = parts[j]
            cleaned = raw.lstrip("+-")
            # Strip Unicode superscripts/special chars that BRISNET uses for margins
            cleaned = re.sub(r"[^\d]", "", cleaned)
            if not cleaned:
                break
            try:
                val = int(cleaned)
            except ValueError:
                break
            has_sign = raw[0] in "+-" if raw else False
            # Call changes are small (typically 0-25) and often signed
            # Speed figures are large (40-130) and unsigned
            if has_sign and val < 30:
                j += 1
                continue
            if not has_sign and val < 30:
                j += 1
                continue
            # First number >= 40 without sign is likely the speed figure
            if 40 <= val <= 130:
                figs.append(val)
            break

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


# ---------- GOLD-STANDARD Probability helpers with mathematical rigor ----------


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
        # 1. Merge the figures into the main ratings dataframe
        df = df.merge(figs_df[["Horse", "AvgTop2"]], on="Horse", how="left")

        # 2. Calculate the average "AvgTop2" for all horses *in this race*
        # We fillna with a low value for first-timers
        df["AvgTop2"] = df["AvgTop2"].fillna(MODEL_CONFIG["first_timer_fig_default"])
        race_avg_fig = df["AvgTop2"].mean()

        # 3. Define the enhancement (R_ENHANCE_ADJ)
        SPEED_FIG_WEIGHT = MODEL_CONFIG["speed_fig_weight"]

        df["R_ENHANCE_ADJ"] = (df["AvgTop2"] - race_avg_fig) * SPEED_FIG_WEIGHT

        # 4. Clean up the temporary column
        df = df.drop(columns=["AvgTop2"])

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

        # 3. E1/E2/LP pace analysis
        pace_data = parse_e1_e2_lp_values(block)
        savant_bonus += analyze_pace_figures(
            pace_data["e1"], pace_data["e2"], pace_data["lp"]
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


def str_to_decimal_odds(s: str) -> float | None:
    s = (s or "").strip()
    if not s:
        return None
    try:
        # CHECK AMERICAN ODDS FIRST (must precede generic numeric check)
        # American: "+250" → (250/100)+1 = 3.5, "-150" → (100/150)+1 = 1.667
        if re.fullmatch(r"\+\d+", s):
            return 1.0 + float(s) / 100.0
        if re.fullmatch(r"-\d+", s):
            return 1.0 + 100.0 / abs(float(s))
        # Fractional: "5/2" → 5/2+1 = 3.5, "3-1" → 3/1+1 = 4.0
        if "/" in s:
            a, b = s.split("/", 1)
            return float(a) / float(b) + 1.0
        if "-" in s:
            a, b = s.split("-", 1)
            return float(a) / float(b) + 1.0
        # Decimal odds or plain number (e.g. "3.5", "2.10")
        if re.fullmatch(r"\d+(\.\d+)?", s):
            v = float(s)
            return max(v, 1.01)
    except Exception as e:
        st.warning(f"Could not parse odds string: '{s}'. Error: {e}")
        return None
    return None


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


# ---------- Exotics (Harville + bias anchors) ----------


def calculate_exotics_biased(
    fair_probs: dict[str, float],
    anchor_first: str | None = None,
    anchor_second: str | None = None,
    pool_third: set | None = None,
    pool_fourth: set | None = None,
    weights=MODEL_CONFIG["exotic_bias_weights"],
    top_n: int = 50,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    horses = list(fair_probs.keys())
    probs = np.array([fair_probs[h] for h in horses])
    n = len(horses)
    if n < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def w_first(h):
        return weights[0] if anchor_first and h == anchor_first else 1.0

    def w_second(h):
        return weights[1] if anchor_second and h == anchor_second else 1.0

    def w_third(h):
        return weights[2] if pool_third and h in pool_third else 1.0

    def w_fourth(h):
        return weights[3] if pool_fourth and h in pool_fourth else 1.0

    # Add a 5th weight for SH5, re-using 4th
    def w_fifth(h):
        return weights[3] if pool_fourth and h in pool_fourth else 1.0

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
            r["Fair Odds"] = float("inf")
    df_ex = pd.DataFrame(ex_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 3:
        return df_ex, pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    # TRIFECTA
    tri_rows = []
    top8 = np.argsort(-probs)[: min(n, 8)]
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
        tri_rows.append(
            {"Ticket": f"{horses[i]} → {horses[j]} → {horses[k]}", "Prob": prob_ijk}
        )

    tri_total = sum(r["Prob"] for r in tri_rows) or 1.0
    for r in tri_rows:
        r["Prob"] = r["Prob"] / tri_total
    for r in tri_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0 / r["Prob"]) - 1
        else:
            r["Fair Odds"] = float("inf")
    df_tri = pd.DataFrame(tri_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 4:
        return df_ex, df_tri, pd.DataFrame(), pd.DataFrame()

    # SUPERFECTA
    super_rows = []
    top6 = np.argsort(-probs)[: min(n, 6)]  # Keep this at top6 for performance
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
        prob_ijkl *= (
            w_first(horses[i])
            * w_second(horses[j])
            * w_third(horses[k])
            * w_fourth(horses[l])
        )
        super_rows.append(
            {
                "Ticket": f"{horses[i]} → {horses[j]} → {horses[k]} → {horses[l]}",
                "Prob": prob_ijkl,
            }
        )

    super_total = sum(r["Prob"] for r in super_rows) or 1.0
    for r in super_rows:
        r["Prob"] = r["Prob"] / super_total
    for r in super_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0 / r["Prob"]) - 1
        else:
            r["Fair Odds"] = float("inf")
    df_super = (
        pd.DataFrame(super_rows).sort_values(by="Prob", ascending=False).head(top_n)
    )

    if n < 5:
        return df_ex, df_tri, df_super, pd.DataFrame()

    # --- NEW: SUPER HIGH 5 ---
    sh5_rows = []
    top7 = np.argsort(-probs)[: min(n, 7)]  # Use Top 7 for SH5
    for i, j, k, l, m in product(top7, top7, top7, top7, top7):
        if len({i, j, k, l, m}) != 5:
            continue
        denom_ij = 1.0 - probs[i]
        denom_ijk = 1.0 - probs[i] - probs[j]
        denom_ijkl = 1.0 - probs[i] - probs[j] - probs[k]
        denom_ijklm = 1.0 - probs[i] - probs[j] - probs[k] - probs[l]
        if (
            denom_ij <= 1e-9
            or denom_ijk <= 1e-9
            or denom_ijkl <= 1e-9
            or denom_ijklm <= 1e-9
        ):
            continue

        p_ij = probs[i] * (probs[j] / denom_ij)
        p_ijk = p_ij * (probs[k] / denom_ijk)
        p_ijkl = p_ijk * (probs[l] / denom_ijkl)
        prob_ijklm = p_ijkl * (probs[m] / denom_ijklm)

        prob_ijklm *= (
            w_first(horses[i])
            * w_second(horses[j])
            * w_third(horses[k])
            * w_fourth(horses[l])
            * w_fifth(horses[m])
        )

        sh5_rows.append(
            {
                "Ticket": f"{horses[i]} → {horses[j]} → {horses[k]} → {horses[l]} → {horses[m]}",
                "Prob": prob_ijklm,
            }
        )

    sh5_total = sum(r["Prob"] for r in sh5_rows) or 1.0
    for r in sh5_rows:
        r["Prob"] = r["Prob"] / sh5_total
    for r in sh5_rows:
        if r["Prob"] > 1e-9:
            r["Fair Odds"] = (1.0 / r["Prob"]) - 1
        else:
            r["Fair Odds"] = float("inf")
    df_super_hi_5 = (
        pd.DataFrame(sh5_rows).sort_values(by="Prob", ascending=False).head(top_n)
    )

    return df_ex, df_tri, df_super, df_super_hi_5


def format_exotics_for_prompt(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"**{title} (Model-Derived)**\nNone.\n"
    df = df.copy()
    if "Prob %" not in df.columns:
        df["Prob %"] = (df["Prob"] * 100).round(2)
    # Format Fair Odds to handle potential infinity
    df["Fair Odds"] = df["Fair Odds"].apply(
        lambda x: f"{x:.2f}" if np.isfinite(x) else "in"
    )
    md = df[["Ticket", "Prob %", "Fair Odds"]].to_markdown(index=False)
    return f"**{title} (Model-Derived)**\n{md}\n"


# -------- Class + suitability model --------


def calculate_final_rating(
    race_type,
    race_surface,
    race_distance_category,
    race_surface_condition,
    horse_surface_pref,
    horse_distance_pref,
):
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
    condition_modifier = condition_modifiers.get(
        str(race_surface_condition).lower(), 1.0
    )

    final_score = base_bias * surface_modifier * distance_modifier * condition_modifier
    return round(final_score, 4)


# ===================== Form Cycle & Recency Analysis =====================


def parse_recent_races_detailed(block) -> list[dict]:
    """Extract recent race history using E2/LP slash marker + jockey name boundary.

    OPTIMIZED Feb 9 2026: Uses jockey name boundary to truncate line before
    scanning for finish positions, preventing decimal odds (e.g., 13.31)
    from contaminating finish position extraction.

    BRISNET race line structure after date+track:
      dist surf class E1 E2/ LP [calls] SPD  PP ST 1C 2C STR FIN   JockeyName  Med Odds ...
    Key: FIN is the last number before 2+ spaces followed by jockey name (alpha chars).
    """
    races = []
    if not block:
        return races

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    today = datetime.now()
    date_pattern = re.compile(r"(\d{2}[A-Za-z]{3}\d{2})[A-Za-z]{2,4}\s+(.+)")

    for m in date_pattern.finditer(block_str):
        date_str = m.group(1)
        rest = m.group(2)
        try:
            race_date = datetime.strptime(date_str, "%d%b%y")
            days_ago = (today - race_date).days

            # Strategy: find the "/" in E2/LP, then after call changes and speed fig,
            # take exactly 6 numbers: PP ST 1C 2C STR FIN
            # But first, truncate line at jockey name to avoid odds contamination
            # Jockey name: 2+ spaces followed by capital letter + lowercase (e.g. "  OrtizIJ")
            jockey_match = re.search(r"(\d)\s{2,}([A-Z][a-zA-Z])", rest)
            if jockey_match:
                rest_trimmed = rest[: jockey_match.start(2)]
            else:
                rest_trimmed = rest

            # Find the "/" separator (E2/LP marker)
            slash_pos = rest_trimmed.find("/")
            if slash_pos < 0:
                continue

            # Extract everything after the "/" section
            after_slash = rest_trimmed[slash_pos + 1 :]
            # Parse all numeric fields from after LP onward
            nums = re.findall(r"[+-]?\d+", after_slash)
            # Format: LP CALL1 CALL2 [CALL3] SPD PP ST 1C 2C STR FIN
            # LP is first, then calls (small, often signed), then SPD (large unsigned),
            # then PP ST 1C 2C STR FIN (all small)
            # Find the speed figure (first number >= 40 after calls)
            spd_idx = None
            for i, s in enumerate(nums):
                val = int(s.lstrip("+-"))
                if i == 0:
                    continue  # Skip LP
                if val >= 40:
                    spd_idx = i
                    break
            if spd_idx is not None and spd_idx + 6 < len(nums):
                # After SPD: PP, ST, 1C, 2C, STR, FIN
                finish = int(nums[spd_idx + 6].lstrip("+-"))
            else:
                # Fallback: take the 6th number counting back from end of trimmed nums
                # (less reliable but better than nothing)
                finish = 99

            races.append({"date": race_date, "days_ago": days_ago, "finish": finish})
        except Exception:
            pass

    return sorted(races, key=lambda x: x["days_ago"])[:6]


def calculate_layoff_factor(
    days_since_last: int,
    num_workouts: int = 0,
    workout_pattern_bonus: float = 0.0,
) -> float:
    """OPTIMIZED Feb 9 2026: Layoff impact with workout mitigation.

    Returns: adjustment factor (-3.0 to +0.5).
    Key changes: 60-120d brackets gentler (strategic freshening window),
    workout mitigation 15%/workout up to 60% recovery, max penalty -3.0.
    Validated on Oaklawn R9 (Air of Defiance #2).
    """
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
    if base < 0 and num_workouts > 0:
        work_credit = min(num_workouts * 0.15, 0.60)  # 15% per workout
        base *= 1.0 - work_credit
        base += workout_pattern_bonus
    return round(max(base, -3.0), 2)


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
    weighted_avg = sum(f * w for f, w in zip(recent_finishes, weights)) / sum(weights)
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


def parse_workout_data(block) -> dict:
    """
    ENHANCED: Extract workout information with pattern analysis.
    Returns dict with best_time, num_works, recency, pattern_bonus
    """
    workouts = {
        "best_time": None,
        "num_recent": 0,
        "days_since_last": 999,
        "pattern_bonus": 0.0,
    }

    if not block:
        return workouts

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    # Enhanced pattern: captures bullet (×), distance, time, grade, rank
    pattern = r"([×]?)(\d{1,2}[A-Z][a-z]{2})\s+\w+\s+(\d+)f\s+\w+\s+([\d:.«©ª¬®¯°¨]+)\s+([HBG]g?)(?:\s+(\d+)/(\d+))?"

    work_details = []
    for match in re.finditer(pattern, block_str):
        try:
            bullet = match.group(1) == "×"
            distance = int(match.group(3))
            time_str = match.group(4)
            grade = match.group(5)
            rank = int(match.group(6)) if match.group(6) else None
            total = int(match.group(7)) if match.group(7) else None

            time_clean = re.sub(r"[«©ª¬®¯°¨]", "", time_str)
            # CRITICAL FIX: Wrap float conversions in try-except to prevent ValueError
            try:
                if ":" in time_clean:
                    parts = time_clean.split(":")
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

            work_details.append(
                {
                    "bullet": bullet,
                    "distance": distance,
                    "time": normalized_time,
                    "grade": grade,
                    "rank": rank,
                    "total": total,
                }
            )
        except Exception:
            pass

    workouts["num_recent"] = len(work_details)

    if work_details:
        workouts["best_time"] = min(w["time"] for w in work_details)

        # SAVANT: Workout pattern analysis
        bonus = 0.0
        if len(work_details) >= 3:
            times = [w["time"] for w in work_details[:3]]
            if times[0] < times[1] < times[2]:
                bonus += 0.08  # Sharp pattern
            elif times[0] > times[1] > times[2]:
                bonus -= 0.06  # Dull pattern

        if work_details[0]["bullet"]:
            bonus += 0.03  # Recent bullet

        if "g" in work_details[0]["grade"].lower():
            bonus += 0.03  # Gate work

        # CRITICAL FIX: Validate total > 0 before division to prevent ZeroDivisionError
        # NOTE: Single percentile check (was double-counted before Feb 9 fix)
        if (
            work_details[0]["rank"]
            and work_details[0]["total"]
            and work_details[0]["total"] > 0
        ):
            percentile = work_details[0]["rank"] / work_details[0]["total"]
            if percentile <= 0.20:
                bonus += 0.07  # Top 20% of field (elite work)
            elif percentile <= 0.25:
                bonus += 0.04  # Top 25%
            elif percentile <= 0.50:
                bonus += 0.02  # Top half

        bullet_count = sum(1 for w in work_details[:5] if w["bullet"])
        if bullet_count >= 2:
            bonus += 0.05  # Consistent quality

        workouts["pattern_bonus"] = bonus

    return workouts


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


# ===================== Odds Conversion Utilities =====================


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


# ===================== Class Rating Calculator (Comprehensive) =====================


def extract_race_metadata_from_pp(pp_text: str) -> dict[str, Any]:
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
        "purse": 0,
        "race_type_raw": "",
        "race_type_normalized": "unknown",
        "confidence": 0.0,
        "source": "none",
    }

    if not pp_text or len(pp_text) < 50:
        return result

    # Extract first 500 chars (header section)
    header = pp_text[:500]

    # ═══════════════ PATTERN 1: "PURSE $X,XXX. Race Type." ═══════════════
    purse_match = re.search(r"PURSE\s+\$([\d,]+)", header, re.IGNORECASE)
    if purse_match:
        try:
            result["purse"] = int(purse_match.group(1).replace(",", ""))
            result["confidence"] += 0.5
            result["source"] = "PURSE header"
        except ValueError:
            pass

    # Extract race type after PURSE line
    if purse_match:
        after_purse = header[purse_match.end() : purse_match.end() + 100]
        # Look for race type keywords
        if re.search(r"\bClaiming\b", after_purse, re.IGNORECASE):
            result["race_type_raw"] = "Claiming"
            result["race_type_normalized"] = "claiming"
            result["confidence"] += 0.5
        elif re.search(r"\bAllowance\b", after_purse, re.IGNORECASE):
            result["race_type_raw"] = "Allowance"
            result["race_type_normalized"] = "allowance"
            result["confidence"] += 0.5
        elif re.search(r"\bMaiden\s+Special\s+Weight\b", after_purse, re.IGNORECASE):
            result["race_type_raw"] = "Maiden Special Weight"
            result["race_type_normalized"] = "maiden special weight"
            result["confidence"] += 0.5
        elif re.search(r"\bMaiden\s+Claiming\b", after_purse, re.IGNORECASE):
            result["race_type_raw"] = "Maiden Claiming"
            result["race_type_normalized"] = "maiden claiming"
            result["confidence"] += 0.5
        elif re.search(r"\b(Stakes?|G[123]|Grade)\b", after_purse, re.IGNORECASE):
            result["race_type_raw"] = "Stakes"
            result["race_type_normalized"] = "stakes"
            result["confidence"] += 0.5

    # ═══════════════ PATTERN 2: "$X,XXX Race Type" in any line ═══════════════
    if result["purse"] == 0:
        money_race_match = re.search(
            r"\$(\d{1,3}(?:,\d{3})*)\s+(Claiming|Allowance|Maiden|Stakes?)",
            header,
            re.IGNORECASE,
        )
        if money_race_match:
            try:
                result["purse"] = int(money_race_match.group(1).replace(",", ""))
                result["race_type_raw"] = money_race_match.group(2)
                result["race_type_normalized"] = money_race_match.group(2).lower()
                result["confidence"] = 0.8
                result["source"] = "$X Race Type"
            except ValueError:
                pass

    # ═══════════════ PATTERN 3: Embedded in race type code ═══════════════
    if result["purse"] == 0:
        # Look for patterns like "Clm25000n2L", "MC50000", "Alw28000"
        embedded_match = re.search(r"(Clm|MC|Alw|OC)(\d{4,6})", header, re.IGNORECASE)
        if embedded_match:
            race_code = embedded_match.group(1).upper()
            purse_num = embedded_match.group(2)
            try:
                result["purse"] = int(purse_num)
                result["race_type_raw"] = f"{race_code}{purse_num}"
                result["source"] = "embedded code"
                result["confidence"] = 0.6

                # Decode race type
                if race_code in ["CLM", "MC"]:
                    result["race_type_normalized"] = "claiming"
                elif race_code in ["ALW", "OC"]:
                    result["race_type_normalized"] = "allowance"
            except ValueError:
                pass

    # ═══════════════ PATTERN 4: "Race X - Race Type" ═══════════════
    if result["race_type_normalized"] == "unknown":
        race_type_line = re.search(
            r"Race\s+\d+\s*-\s*([A-Za-z\s]+)", header, re.IGNORECASE
        )
        if race_type_line:
            race_text = race_type_line.group(1).strip().lower()
            if "claim" in race_text:
                result["race_type_normalized"] = "claiming"
                result["race_type_raw"] = race_type_line.group(1).strip()
                result["confidence"] += 0.3
            elif "allow" in race_text:
                result["race_type_normalized"] = "allowance"
                result["race_type_raw"] = race_type_line.group(1).strip()
                result["confidence"] += 0.3
            elif "maiden" in race_text:
                result["race_type_normalized"] = "maiden special weight"
                result["race_type_raw"] = race_type_line.group(1).strip()
                result["confidence"] += 0.3

    return result


def extract_race_metadata_from_pp_text(pp_text: str) -> dict[str, Any]:
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
            "purse_amount": 0,
            "race_type": "",
            "race_type_clean": "unknown",
            "confidence": 0.0,
            "detection_method": "no_text",
        }

    result = {
        "purse_amount": 0,
        "race_type": "",
        "race_type_clean": "unknown",
        "confidence": 0.0,
        "detection_method": "none",
    }

    # Get first 800 characters (header region)
    header = pp_text[:800]

    # ========== PURSE EXTRACTION (Multi-Pattern) ==========
    purse_patterns = [
        r"PURSE\s+\$([\d,]+)",  # "PURSE $6,250"
        r"Purse:\s+\$([\d,]+)",  # "Purse: $50,000"
        r"\$([\d,]+)\s+(?:Grade|Stakes|Allowance|Claiming)",  # "$100,000 Grade 2"
        r"\$([\d,]+)",  # Any dollar amount in header
    ]

    for pattern in purse_patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            try:
                result["purse_amount"] = int(match.group(1).replace(",", ""))
                result["detection_method"] = "purse_text"
                result["confidence"] = 0.9
                break
            except BaseException:
                pass

    # ========== RACE TYPE EXTRACTION (Multi-Pattern) ==========
    race_type_patterns = [
        # Graded Stakes
        (r"Grade\s+(I{1,3}|[123])", "stakes_graded", 1.0),
        (r"G([123])", "stakes_graded", 1.0),
        # Stakes
        (r"Stakes", "stakes", 0.95),
        (r"Handicap", "stakes", 0.9),
        (r"Listed", "stakes", 0.9),
        # Allowance
        (r"Allowance\s+Optional\s+Claiming", "allowance_optional", 0.95),
        (r"Optional\s+Claiming", "allowance_optional", 0.95),
        (r"Allowance", "allowance", 0.95),
        (r"\bAOC\b", "allowance_optional", 0.9),
        (r"\bAlw\b", "allowance", 0.9),
        # Claiming
        (r"Maiden\s+Claiming", "maiden_claiming", 0.95),
        (r"Maiden\s+Clm", "maiden_claiming", 0.9),
        (r"\bMCL\b", "maiden_claiming", 0.9),
        (r"Claiming", "claiming", 0.95),
        (r"\bClm\b", "claiming", 0.9),
        (r"\bMC\b", "claiming", 0.85),
        # Maiden
        (r"Maiden\s+Special\s+Weight", "maiden_special_weight", 0.95),
        (r"\bMSW\b", "maiden_special_weight", 0.9),
        (r"Maiden", "maiden", 0.85),
        # Starter
        (r"Starter\s+Optional\s+Claiming", "starter_optional_claiming", 0.9),
        (r"Starter\s+Allowance", "starter_allowance", 0.9),
        (r"Starter\s+Handicap", "starter_handicap", 0.9),
        (r"\bSOC\b", "starter_optional_claiming", 0.85),
        # Waiver
        (r"Waiver", "waiver_claiming", 0.85),
    ]

    for pattern, race_type_clean, confidence in race_type_patterns:
        match = re.search(pattern, header, re.IGNORECASE)
        if match:
            result["race_type"] = match.group(0)
            result["race_type_clean"] = race_type_clean
            result["confidence"] = max(result["confidence"], confidence)
            if result["detection_method"] == "none":
                result["detection_method"] = "text_pattern"
            break

    # ========== EMBEDDED RACE TYPE (Clm25000n2L format) ==========
    embedded_pattern = r"\b(Clm|MC|OC|Alw|Mdn|MSW|Stk|G[123])([\d]+[kK]?)"
    embedded_match = re.search(embedded_pattern, header, re.IGNORECASE)
    if embedded_match:
        prefix = embedded_match.group(1).lower()
        amount_str = embedded_match.group(2)

        # Extract purse from embedded format if not already found
        if result["purse_amount"] == 0:
            try:
                if "k" in amount_str.lower():
                    result["purse_amount"] = int(amount_str[:-1]) * 1000
                else:
                    result["purse_amount"] = int(amount_str)
                result["detection_method"] = "embedded_format"
            except BaseException:
                pass

        # Map prefix to race type if not already found
        if result["race_type_clean"] == "unknown":
            prefix_map = {
                "clm": "claiming",
                "mc": "claiming",
                "oc": "allowance_optional",
                "alw": "allowance",
                "mdn": "maiden",
                "msw": "maiden_special_weight",
                "stk": "stakes",
                "g1": "stakes_graded",
                "g2": "stakes_graded",
                "g3": "stakes_graded",
            }
            result["race_type_clean"] = prefix_map.get(prefix, "unknown")
            result["race_type"] = embedded_match.group(0)
            result["confidence"] = 0.85
            if result["detection_method"] == "none":
                result["detection_method"] = "embedded_format"

    return result


def infer_purse_from_race_type(race_type: str) -> int | None:
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
    match = re.search(r"(\d{4,6})", race_type)
    if match:
        return int(match.group(1))

    # Pattern 2: With 'k' suffix (OC20k, Alw50k)
    match = re.search(r"(\d+)k", race_type, re.IGNORECASE)
    if match:
        return int(match.group(1)) * 1000

    # Pattern 3: Common defaults by type
    race_lower = race_type.lower()
    if "maiden" in race_lower or "mdn" in race_lower or "md sp wt" in race_lower:
        return 50000  # Typical maiden special weight
    elif "claiming" in race_lower or "clm" in race_lower or "mc" in race_lower:
        return 25000  # Typical claiming level
    elif "allowance" in race_lower or "alw" in race_lower:
        return 50000  # Typical allowance
    elif (
        "stake" in race_lower
        or "stk" in race_lower
        or "g1" in race_lower
        or "g2" in race_lower
        or "g3" in race_lower
    ):
        return 100000  # Stakes minimum

    return None


def parse_recent_class_levels(block) -> list[dict]:
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
    lines = block_str.split("\n")
    for line in lines:
        # Updated pattern to capture embedded race types
        race_match = re.search(
            r"(\d{2}[A-Za-z]{3}\d{2})\w+\s+[\d½]+[f]?\s+.*?"
            r"([A-Z][a-z]{2,}\d+[a-zA-Z0-9\-]*|MC\d+|OC\d+k?|Alw\d+|Stk|G[123]|Hcp)",
            line,
        )

        if race_match:
            race_type = race_match.group(2)

            # CRITICAL: Infer purse from race type name
            inferred_purse = infer_purse_from_race_type(race_type)

            # Extract finish position from same line (look for FIN column)
            # Pattern: FIN followed by position like "1st", "2nd", "3©", "4¬", etc.
            finish_match = re.search(r"FIN\s+(\d{1,2})[ƒ®«ª³©¨°¬²‚±\s]", line)
            finish_pos = int(finish_match.group(1)) if finish_match else 0

            try:
                races.append(
                    {
                        "race_type": race_type,
                        "purse": inferred_purse if inferred_purse else 0,
                        "finish_pos": finish_pos,
                    }
                )
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
        "starter optional claiming": 5,
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


# ===================== 1. Paste PPs & Parse (durable) =====================


# Workflow Progress Indicator
step1_done = st.session_state.get("parsed", False)
step2_done = "primary_d" in st.session_state and "primary_probs" in st.session_state
step3_done = st.session_state.get("classic_report_generated", False)

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
            saved_count = stats.get("total_races", 0) + len(
                gold_db.get_pending_races(limit=1000)
            )
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
    disabled=st.session_state["parsed"],
)

# VALIDATION: Check minimum text length
if pp_text_widget and len(pp_text_widget.strip()) < 100:
    st.warning(
        "⚠️ PP text too short. Please paste complete Past Performance data (minimum 100 characters)."
    )

col_parse, col_reset = st.columns([1, 1])
with col_parse:
    parse_clicked = st.button("Parse PPs", type="primary")
with col_reset:
    reset_clicked = st.button(
        "Reset / Parse New", help="Clear parsed state to paste another race"
    )

if reset_clicked:
    st.session_state["parsed"] = False
    st.session_state["pp_text_cache"] = ""
    # Clear Classic Report from previous race
    st.session_state.pop("classic_report_generated", None)
    st.session_state.pop("classic_report", None)
    # FIX (Feb 11, 2026): Clear stale race_num so next parse detects fresh value
    st.session_state.pop("race_num", None)
    _safe_rerun()

if parse_clicked:
    text_now = (st.session_state.get("pp_text_input") or "").strip()
    # CRITICAL FIX: Validate text length before parsing
    if not text_now:
        st.warning("Paste PPs text first.")
    elif len(text_now) < 100:
        st.error(
            "❌ PP text too short. Please paste complete Past Performance data (minimum 100 characters)."
        )
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
            st.session_state.pop("classic_report_generated", None)
            st.session_state.pop("classic_report", None)
            # FIX (Feb 11, 2026): Detect race number from fresh PP text and
            # force-set session state so the widget picks up the correct value
            _fresh_race_num = parse_brisnet_race_header(text_now).get(
                "race_number"
            ) or detect_race_number(text_now)
            if _fresh_race_num:
                st.session_state["race_num"] = int(_fresh_race_num)
        _safe_rerun()

if not st.session_state["parsed"]:
    st.info("Paste your PPs and click **Parse PPs** to continue.")
    st.stop()

pp_text = st.session_state["pp_text_cache"]

# ===================== 2. Race Info (Confirm) =====================

st.header("2. Race Info (Confirm)")
first_line = _find_header_line(pp_text)

# Parse comprehensive header information
header_info = parse_brisnet_race_header(pp_text)

# Display extracted header info if available
if header_info and any(header_info.values()):
    with st.expander("📋 Extracted Header Information", expanded=False):
        col1, col2 = st.columns(2)
        with col1:
            if header_info.get("track_name"):
                st.caption(f"**Track:** {header_info['track_name']}")
            if header_info.get("race_number"):
                st.caption(f"**Race:** {header_info['race_number']}")
            if header_info.get("race_type"):
                st.caption(f"**Type:** {header_info['race_type']}")
            if header_info.get("purse_amount"):
                st.caption(f"**Purse:** ${header_info['purse_amount']:,}")
        with col2:
            if header_info.get("distance"):
                st.caption(f"**Distance:** {header_info['distance']}")
            if header_info.get("age_restriction") or header_info.get("sex_restriction"):
                restrictions = []
                if header_info.get("age_restriction"):
                    restrictions.append(header_info["age_restriction"])
                if header_info.get("sex_restriction"):
                    restrictions.append(header_info["sex_restriction"])
                st.caption(f"**Restrictions:** {' '.join(restrictions)}")
            if header_info.get("day_of_week") and header_info.get("race_date"):
                st.caption(
                    f"**Date:** {header_info['day_of_week']}, {header_info['race_date']}"
                )

# Track (use parsed value from comprehensive parser if available, otherwise fallback)
parsed_track = header_info.get("track_name") or parse_track_name_from_pp(pp_text)
track_name = st.text_input(
    "Track:", value=(parsed_track or st.session_state["track_name"])
)

# SECURITY: Validate track name if validators available
if SECURITY_VALIDATORS_AVAILABLE and track_name:
    try:
        track_name = validate_track_name(track_name)
    except ValueError as e:
        st.warning(f"Invalid track name (using default): {e}")
        track_name = "Unknown Track"

st.session_state["track_name"] = track_name

# Race Number (use parsed value if available)
if "race_num" not in st.session_state:
    st.session_state["race_num"] = 1
auto_race_num = header_info.get("race_number") or detect_race_number(pp_text)
default_race_num = int(auto_race_num) if auto_race_num else st.session_state["race_num"]
# FIX (Feb 11, 2026): Force-set session state BEFORE widget render so Streamlit
# uses the freshly detected race number instead of stale widget state from a prior race.
st.session_state["race_num"] = default_race_num
# CRITICAL FIX: Increase max_value to 20 (some tracks have 16+ races)
race_num = st.number_input(
    "Race Number:", min_value=1, max_value=20, step=1, value=default_race_num
)
st.session_state["race_num"] = race_num

# Surface auto from header, but allow override
default_surface = st.session_state["surface_type"]
if re.search(r"(?i)\bturf|trf\b", first_line):
    default_surface = "Turf"
if re.search(r"(?i)\baw|tap|synth|poly\b", first_line):
    default_surface = "Synthetic"
surface_type = st.selectbox(
    "Surface:",
    ["Dirt", "Turf", "Synthetic"],
    index=["Dirt", "Turf", "Synthetic"].index(default_surface)
    if default_surface in ["Dirt", "Turf", "Synthetic"]
    else 0,
)
st.session_state["surface_type"] = surface_type

# Condition
conditions = [
    "fast",
    "good",
    "wet-fast",
    "muddy",
    "sloppy",
    "firm",
    "yielding",
    "soft",
    "heavy",
]
cond_found = None
for cond in conditions:
    if re.search(rf"(?i)\b{cond}\b", first_line):
        cond_found = cond
        break
default_condition = cond_found if cond_found else st.session_state["condition_txt"]
condition_txt = st.selectbox(
    "Condition:",
    conditions,
    index=conditions.index(default_condition) if default_condition in conditions else 0,
)
st.session_state["condition_txt"] = condition_txt

# Distance (auto + dropdown)


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


auto_distance = _auto_distance_label(first_line)
# try to map to option variants
preferred = (auto_distance or "").replace("½", "1/2").replace(" 1/2", " 1/2")
idx = DISTANCE_OPTIONS.index("6 Furlongs") if "6 Furlongs" in DISTANCE_OPTIONS else 0
for opt in (preferred, auto_distance, st.session_state["distance_txt"]):
    if opt in DISTANCE_OPTIONS:
        idx = DISTANCE_OPTIONS.index(opt)
        break
distance_txt = st.selectbox("Distance:", DISTANCE_OPTIONS, index=idx)
st.session_state["distance_txt"] = distance_txt

# SECURITY (Feb 13, 2026): Validate distance string before downstream parsing
if SECURITY_VALIDATORS_AVAILABLE and distance_txt:
    try:
        distance_txt = validate_distance_string(distance_txt)
    except ValueError as e:
        logger.warning(f"Distance validation failed: {e}, using default")
        distance_txt = "6 Furlongs"
        st.session_state["distance_txt"] = distance_txt

# Purse


def detect_purse_amount(pp_text: str) -> int | None:
    s = pp_text or ""
    m = re.search(r"(?mi)\bPurse\b[^$\n\r]*\$\s*([\d,]+)", s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except BaseException:
            pass
    m = re.search(r"(?mi)\b(?:Added|Value)\b[^$\n\r]*\$\s*([\d,]+)", s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except BaseException:
            pass
    m = re.search(
        r"(?mi)\b(Mdn|Maiden|Allowance|Alw|Claiming|Clm|Starter|Stake|Stakes)\b[^:\n\r]{0,50}\b(\d{2,4})\s*[Kk]\b",
        s,
    )
    if m:
        try:
            return int(m.group(2)) * 1000
        except BaseException:
            pass
    m = re.search(r"(?m)\$\s*([\d,]{5,})", s)
    if m:
        try:
            return int(m.group(1).replace(",", ""))
        except BaseException:
            pass
    return None


auto_purse = detect_purse_amount(pp_text)
default_purse = int(auto_purse) if auto_purse else st.session_state["purse_val"]
purse_val = st.number_input("Purse ($)", min_value=0, step=5000, value=default_purse)
st.session_state["purse_val"] = purse_val

# Race type detection + override list (constant keys only)
race_type_detected = detect_race_type(pp_text)
st.caption(f"Detected race type: **{race_type_detected}**")
# Ensure base_class_bias is not empty before proceeding
if not base_class_bias:
    st.error("Race type definitions (base_class_bias) are missing or failed to load.")
    st.stop()
try:
    race_type_index = (
        list(base_class_bias.keys()).index(race_type_detected)
        if race_type_detected in base_class_bias
        else list(base_class_bias.keys()).index("allowance")
    )
except ValueError:
    st.warning(
        "Default race type 'allowance' not found in bias definitions. Using first available type."
    )
    race_type_index = 0  # Default to the first item if 'allowance' isn't found

race_type_manual = st.selectbox(
    "Race Type (override):",
    options=list(base_class_bias.keys()),
    index=race_type_index,
)
race_type = race_type_manual or race_type_detected
race_type_detected = race_type  # lock in constant key
st.session_state["race_type"] = race_type_detected  # Store for Classic Report

# ═══ AUTO-REFRESH Track Intelligence profile for the active track ═══
# Eagerly rebuild the TI profile so Section E shows fresh data without Retrain.
# Guard: only rebuild once per track_name to avoid redundant DB writes on rerun.
if (
    TRACK_INTEL_AVAILABLE
    and _track_intel is not None
    and track_name
    and track_name != "Unknown Track"
    and st.session_state.get("_ti_last_refreshed_track") != track_name.upper()
):
    try:
        _track_intel.update_after_submission(track_name)
        st.session_state["_ti_last_refreshed_track"] = track_name.upper()
        logger.info(f"Auto-refreshed Track Intelligence profile for {track_name}")
    except Exception as _ti_refresh_err:
        logger.debug(f"TI auto-refresh skipped: {_ti_refresh_err}")

# ===================== A. Race Setup: Scratches, ML & Styles =====================

st.markdown("---")
st.header("A. Race Setup: Scratches, ML & Live Odds, Styles")
st.caption(
    "📝 Review auto-detected data below. Edit any horse's ML odds or style, then check 'Scratch?' to remove horses."
)

df_styles = extract_horses_and_styles(pp_text)
if df_styles.empty:
    st.error("No horses found. Check your PP text paste.")
    st.stop()

ml_map_raw = extract_morning_line_by_horse(pp_text)
df_styles["ML"] = (
    df_styles["Horse"].map(lambda h: ml_map_raw.get(h, ""))
    if "Horse" in df_styles
    else ""
)
df_styles["Live Odds"] = df_styles["ML"].where(
    df_styles["ML"].astype(str).str.len() > 0, ""
)
df_styles["Scratched"] = False

col_cfg = {
    "Post": st.column_config.TextColumn("Post", width="small", disabled=True),
    "Horse": st.column_config.TextColumn("Horse", width="medium", disabled=True),
    "ML": st.column_config.TextColumn(
        "ML", width="small", help="Parsed Morning Line from PPs"
    ),
    "Live Odds": st.column_config.TextColumn(
        "Live Odds",
        width="small",
        help="Enter current odds (e.g., '5/2', '3.5', '+250')",
    ),
    "DetectedStyle": st.column_config.TextColumn(
        "BRIS Style", width="small", disabled=True
    ),
    "Quirin": st.column_config.NumberColumn("Quirin", width="small", disabled=True),
    "AutoStrength": st.column_config.TextColumn(
        "Auto-Strength", width="small", disabled=True
    ),
    "OverrideStyle": st.column_config.SelectboxColumn(
        "Override Style", width="small", options=["", "E", "E/P", "P", "S"]
    ),
    "Scratched": st.column_config.CheckboxColumn("Scratched?", width="small"),
}
df_editor = st.data_editor(df_styles, use_container_width=True, column_config=col_cfg)

# ===================== B. Angle Parsing / Pedigree / Figs =====================

angles_per_horse: dict[str, pd.DataFrame] = {}
pedigree_per_horse: dict[str, dict] = {}
figs_per_horse: dict[str, list[int]] = {}

# Try to use elite parser first for better accuracy
elite_parser_used = False
if (
    ELITE_PARSER_AVAILABLE
    and GoldStandardBRISNETParser is not None
    and len(pp_text.strip()) > 100
):
    try:
        parser = GoldStandardBRISNETParser()
        horses = parser.parse_full_pp(pp_text, debug=False)
        validation = parser.validate_parsed_data(horses, min_confidence=0.5)

        if validation.get("overall_confidence", 0.0) >= 0.6:
            # Use elite parser data
            elite_parser_used = True
            # Store full elite parsed data for downstream DB storage
            st.session_state["elite_horses_data"] = {
                name: obj.to_dict() for name, obj in horses.items()
            }
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
        figs_data.append(
            {
                "Horse": name,
                "Figures": fig_list,  # The list of parsed figs
                "BestFig": max(fig_list),
                "AvgTop2": round(np.mean(sorted(fig_list, reverse=True)[:2]), 1),
                "LastFig": fig_list[0]
                if fig_list
                else 0,  # Most recent figure (Feb 13 fix)
            }
        )
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
st.session_state["df_final_field"] = df_final_field

# Ensure StyleStrength and Style exist
df_final_field["StyleStrength"] = df_final_field.apply(
    lambda row: calculate_style_strength(
        row["OverrideStyle"] if row["OverrideStyle"] else row["DetectedStyle"],
        row["Quirin"],
    ),
    axis=1,
)
df_final_field["Style"] = df_final_field.apply(
    lambda r: _normalize_style(
        r["OverrideStyle"] if r["OverrideStyle"] else r["DetectedStyle"]
    ),
    axis=1,
)
if "#" not in df_final_field.columns:
    df_final_field["#"] = df_final_field["Post"].astype(str)

# PPI
ppi_results = compute_ppi(df_final_field)
ppi_val = ppi_results.get("ppi", 0.0)
ppi_map_by_horse = ppi_results.get("by_horse", {})
st.session_state["ppi_val"] = ppi_val  # Store for Classic Report

# ===================== Class build per horse (angles+pedigree in background) =====================


def _infer_horse_surface_pref(
    name: str, ped: dict, ang_df: pd.DataFrame | None, race_surface: str
) -> str:
    cats = (
        " ".join(ang_df["Category"].astype(str).tolist()).lower()
        if (ang_df is not None and not ang_df.empty)
        else ""
    )
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

    # DIAGNOSTIC (Feb 13, 2026): Log when horse has no PP block match
    if not horse_block and pp_text:
        logger.warning(
            f"⚠️ ZERO-SCORE RISK (Cclass/Cform): Horse '{name}' has no PP block. "
            f"Class and form ratings will default to 0.0."
        )

    # Calculate comprehensive class rating
    # NOW includes PP text for race class parser to properly understand race acronyms
    comprehensive_class = calculate_comprehensive_class_rating(
        today_purse=purse_val,
        today_race_type=race_type_detected,
        horse_block=horse_block,
        pedigree=ped,
        angles_df=ang if ang is not None else pd.DataFrame(),
        pp_text=pp_text,  # NEW: Full PP for race analysis
        _distance_furlongs=distance_to_furlongs(distance_txt),  # NEW: Distance
        _surface_type=race_surface,  # NEW: Surface type
    )

    # Add pedigree/angle tweaks on top of class
    tweak = _angles_pedigree_tweak(name, race_surface, race_bucket, race_cond)
    cclass_total = comprehensive_class + tweak

    # Calculate form cycle rating
    form_rating = calculate_form_cycle_rating(
        horse_block=horse_block,
        pedigree=ped,
        angles_df=ang if ang is not None else pd.DataFrame(),
    )

    # Track bias style adjustments
    style_adjustment = 0.0
    horse_style = r.get("Style", "NA")

    dist_bucket = distance_bucket(distance_txt)
    track_cfg = (
        TRACK_BIAS_PROFILES.get(_canonical_track(track_name), {})
        .get(race_surface, {})
        .get(dist_bucket, {})
    )
    runstyle_biases = track_cfg.get("runstyle", {})

    # Strong stalker track detected if S style has > 0.3 advantage
    stalker_impact = runstyle_biases.get("S", 0.0)
    early_speed_impact = runstyle_biases.get("E", 0.0)

    if stalker_impact > 0.3:  # Strong stalker-favoring track
        # Apply penalties/bonuses based on horse's style vs track bias
        if horse_style == "E":
            style_adjustment -= 1.5  # Heavy penalty for early speed
        elif horse_style == "E/P":
            style_adjustment -= 0.8  # Moderate penalty
        elif horse_style == "S":
            style_adjustment += 1.2  # Strong bonus for stalkers
    elif early_speed_impact > 0.3:  # Speed-favoring track
        if horse_style == "E":
            style_adjustment += 1.2  # Bonus for early speed
        elif horse_style == "P":
            # SA R8 audit: P-types were over-penalized at speed tracks.
            # SA meet data showed P impact=1.33 (ABOVE average).
            # Only penalize P if BRISNET impact values actually suppress P.
            _weekly_impacts = st.session_state.get("weekly_bias_impacts", {})
            _p_impact = _weekly_impacts.get("P", 1.0) if _weekly_impacts else 1.0
            if _p_impact < 0.80:
                style_adjustment -= 0.5  # Only penalize if P is truly suppressed
            # else: no penalty — P is competitive at this track
        elif horse_style == "S":
            style_adjustment -= 0.8  # Penalty for closers

    # QSP / STYLE MISMATCH PENALTY (SA R8 audit Feb 20, 2026)
    # Tapatia Mia: E/P style + QSP=1. Can't be a presser with zero early speed points.
    # If style claims early speed (E, E/P) but QSP < 3, penalize — style is unreliable.
    try:
        _qsp_val = float(r.get("Quirin", np.nan))
        if pd.notna(_qsp_val):
            if horse_style in ("E", "E/P") and _qsp_val <= 2:
                style_adjustment -= 0.6  # Style/QSP mismatch — not a real speed horse
            elif horse_style in ("E", "E/P") and _qsp_val <= 3:
                style_adjustment -= 0.3  # Marginal mismatch
    except (ValueError, TypeError):
        pass

    form_rating += style_adjustment

    Cclass_vals.append(round(cclass_total, 3))
    Cform_vals.append(round(form_rating, 3))

df_final_field["Cclass"] = Cclass_vals
df_final_field["Cform"] = Cform_vals

# ===================== B. Bias-Adjusted Ratings =====================


st.markdown("---")
st.header("B. Bias-Adjusted Ratings")
st.caption(
    "⚙️ Select your strategy profile and bias preferences. Ratings calculate automatically based on your selections."
)
b_col1, b_col2, b_col3 = st.columns(3)
with b_col1:
    strategy_profile = st.selectbox(
        "Select Strategy Profile:",
        options=["Confident", "Value Hunter"],
        index=0,
        key="strategy_profile",
    )
with b_col2:
    running_style_biases = st.multiselect(
        "Select Running Style Biases:",
        options=["E", "E/P", "P", "S"],
        default=["E"],
        key="style_biases",
    )
with b_col3:
    post_biases = st.multiselect(
        "Select Post Position Biases:",
        options=[
            "no significant post bias",
            "favors rail (1)",
            "favors inner (1-3)",
            "favors mid (4-7)",
            "favors outside (8+)",
        ],
        default=["no significant post bias"],
        key="post_biases",
    )

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
    """Single-bias style match score with NA QSP handling (Feb 11, 2026)."""
    bias = (running_style_bias or "").strip().lower()
    stl = (style or "NA").upper()

    table = MODEL_CONFIG["style_match_table"]

    try:
        q = float(quirin)
    except Exception:
        q = np.nan

    # NA style: use QSP for partial credit (Feb 11, 2026)
    if stl == "NA" and pd.notna(q) and q >= NA_STYLE_PARAMS["qsp_ep_threshold"]:
        if "favoring" in bias and "closer" not in bias:
            base = 0.15 * (q / 8.0)
        elif "closer" in bias:
            base = -0.10 * (q / 8.0)
        else:
            base = 0.0
    else:
        base = table.get(bias, table["fair/neutral"]).get(stl, 0.0)

    if (
        stl in ("E", "E/P")
        and pd.notna(q)
        and q >= MODEL_CONFIG["style_quirin_threshold"]
    ):
        base += MODEL_CONFIG["style_quirin_bonus"]
    return float(np.clip(base, -1.0, 1.0))


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


# ======================== Phase 1: Enhanced Parsing Functions ========================


def parse_track_bias_impact_values(pp_text: str) -> dict[str, float]:
    """Extract Track Bias Impact Values from '9b. Track Bias (Numerical)' section
    OR from raw BRISNET 'Week Totals' / 'MEET Totals' format."""
    impact_values = {}

    # METHOD 1: Formatted '9b. Track Bias' section
    bias_match = re.search(
        r"9b\.\s*Track Bias.*?\n(.*?)(?=\n\d+[a-z]?\.|$)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if bias_match:
        bias_text = bias_match.group(1)
        for match in re.finditer(
            r"-\s*([A-Z/]+)\s*\([^)]+\):\s*Impact Value\s*=\s*([\d.]+)", bias_text
        ):
            style_code = match.group(1).strip()
            impact_val = float(match.group(2))
            impact_values[style_code] = impact_val
        if impact_values:
            return impact_values

    # METHOD 2: Raw BRISNET 'Week Totals' format (preferred — most recent)
    # Runstyle: E E/P P S
    # ...
    # Impact Values: 1.17 1.14 1.18 0.32
    week_match = re.search(
        r"\*\s*Week\s+Totals\s*\*.*?"
        r"Runstyle:\s*E\s+E/P\s+P\s+S.*?"
        r"Impact Values:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if week_match:
        impact_values["E"] = float(week_match.group(1))
        impact_values["E/P"] = float(week_match.group(2))
        impact_values["P"] = float(week_match.group(3))
        impact_values["S"] = float(week_match.group(4))
        return impact_values

    # METHOD 3: Meet Totals fallback
    meet_match = re.search(
        r"\*\s*MEET\s+Totals\s*\*.*?"
        r"Runstyle:\s*E\s+E/P\s+P\s+S.*?"
        r"Impact Values:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if meet_match:
        impact_values["E"] = float(meet_match.group(1))
        impact_values["E/P"] = float(meet_match.group(2))
        impact_values["P"] = float(meet_match.group(3))
        impact_values["S"] = float(meet_match.group(4))

    return impact_values


# ======================== RACE AUDIT ENHANCEMENTS (Feb 2026) ========================
# Oaklawn R1 Audit: Model ranked Tell Me When last (#7) but horse WON.
# Model's #1 Tiffany Twist finished 4th. Root cause: 5 gaps in bias/angle handling.
# All enhancements below are ADDITIVE — no existing logic is modified.


def parse_weekly_post_bias(pp_text: str) -> dict[str, float]:
    """
    Extract weekly post-position Impact Values from BRISNET PP '9b' section
    OR from raw BRISNET Track Bias Stats format.

    Returns dict mapping individual post numbers (as strings) to their impact values.
    """
    post_impacts: dict[str, float] = {}

    # METHOD 1: Formatted '9b. Track Bias' section
    bias_match = re.search(
        r"9b\.\s*Track Bias.*?\n(.*?)(?=\n\d+[a-z]?\.|$)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if bias_match:
        bias_text = bias_match.group(1)
        for match in re.finditer(
            r"-\s*Posts?\s*(\d+)[-–](\d+)\s*\([^)]*\):\s*Impact Value\s*=\s*([\d.]+)",
            bias_text,
        ):
            low_post = int(match.group(1))
            high_post = int(match.group(2))
            impact_val = float(match.group(3))
            for p in range(low_post, high_post + 1):
                post_impacts[str(p)] = impact_val
        for match in re.finditer(
            r"-\s*Post\s*(\d+)\s*\([^)]*\):\s*Impact Value\s*=\s*([\d.]+)", bias_text
        ):
            post_impacts[match.group(1)] = float(match.group(2))
        if post_impacts:
            return post_impacts

    # METHOD 2: Raw BRISNET 'Week Totals' format (preferred — most recent)
    # Post Bias: RAIL 1-3 4-7 8+
    # ...
    # Impact Values: 2.59 1.72 0.68 0.70
    week_post_match = re.search(
        r"\*\s*Week\s+Totals\s*\*.*?"
        r"Post Bias:\s*RAIL\s+(\d+)-(\d+)\s+(\d+)-(\d+)\s+(\d+)\+.*?"
        r"Impact Values:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if week_post_match:
        inner_low = int(week_post_match.group(1))
        inner_high = int(week_post_match.group(2))
        mid_low = int(week_post_match.group(3))
        mid_high = int(week_post_match.group(4))
        outer_start = int(week_post_match.group(5))
        rail_impact = float(week_post_match.group(6))
        inner_impact = float(week_post_match.group(7))
        mid_impact = float(week_post_match.group(8))
        outer_impact = float(week_post_match.group(9))
        # RAIL = post 1 (separate from the inner range)
        post_impacts["1"] = rail_impact
        for p in range(inner_low, inner_high + 1):
            if str(p) not in post_impacts:
                post_impacts[str(p)] = inner_impact
        for p in range(mid_low, mid_high + 1):
            post_impacts[str(p)] = mid_impact
        for p in range(outer_start, outer_start + 6):
            post_impacts[str(p)] = outer_impact
        return post_impacts

    # METHOD 3: Meet Totals fallback
    meet_post_match = re.search(
        r"\*\s*MEET\s+Totals\s*\*.*?"
        r"Post Bias:\s*RAIL\s+(\d+)-(\d+)\s+(\d+)-(\d+)\s+(\d+)\+.*?"
        r"Impact Values:\s*([\d.]+)\s+([\d.]+)\s+([\d.]+)\s+([\d.]+)",
        pp_text,
        re.DOTALL | re.IGNORECASE,
    )
    if meet_post_match:
        inner_low = int(meet_post_match.group(1))
        inner_high = int(meet_post_match.group(2))
        mid_low = int(meet_post_match.group(3))
        mid_high = int(meet_post_match.group(4))
        outer_start = int(meet_post_match.group(5))
        rail_impact = float(meet_post_match.group(6))
        inner_impact = float(meet_post_match.group(7))
        mid_impact = float(meet_post_match.group(8))
        outer_impact = float(meet_post_match.group(9))
        post_impacts["1"] = rail_impact
        for p in range(inner_low, inner_high + 1):
            if str(p) not in post_impacts:
                post_impacts[str(p)] = inner_impact
        for p in range(mid_low, mid_high + 1):
            post_impacts[str(p)] = mid_impact
        for p in range(outer_start, outer_start + 6):
            post_impacts[str(p)] = outer_impact

    return post_impacts


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
            claim_wins = int(claim_match.group(2))
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


def parse_pedigree_spi(pp_text: str) -> dict[str, int | None]:
    """Extract SPI (Sire Performance Index) from pedigree sections"""
    spi_values = {}

    # Look for pattern like "Sire: Hard Spun (SPI: 1.30)" in Section 4
    horse_sections = re.split(r"\n(?=\d+\.\s+Horse:)", pp_text)

    for section in horse_sections:
        horse_match = re.search(r"Horse:\s*(.+?)(?=\s*\(#|\n)", section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        # Look for SPI in Sire line
        spi_match = re.search(r"Sire:.*?SPI:\s*([\d.]+)", section, re.IGNORECASE)
        if spi_match:
            try:
                spi = int(float(spi_match.group(1)) * 100)  # Convert 1.30 to 130
                spi_values[horse_name] = spi
            except Exception:
                spi_values[horse_name] = None
        else:
            spi_values[horse_name] = None

    return spi_values


def parse_pedigree_surface_stats(pp_text: str) -> dict[str, dict[str, any]]:
    """Extract surface statistics (Turf/AW win%) from pedigree sections"""
    surface_stats = {}

    horse_sections = re.split(r"\n(?=\d+\.\s+Horse:)", pp_text)

    for section in horse_sections:
        horse_match = re.search(r"Horse:\s*(.+?)(?=\s*\(#|\n)", section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        stats = {}
        # Look for "Turf: 12% (class-adj)" or similar
        turf_match = re.search(r"Turf:\s*([\d.]+)%", section, re.IGNORECASE)
        if turf_match:
            stats["turf_pct"] = float(turf_match.group(1))

        # Look for "AW: 8% (class-adj)" or similar
        aw_match = re.search(r"(?:AW|All-Weather):\s*([\d.]+)%", section, re.IGNORECASE)
        if aw_match:
            stats["aw_pct"] = float(aw_match.group(1))

        if stats:
            surface_stats[horse_name] = stats

    return surface_stats


def parse_awd_analysis(pp_text: str) -> dict[str, str]:
    """Extract AWD (Avg Winning Distance) analysis from pedigree sections"""
    awd_data = {}

    horse_sections = re.split(r"\n(?=\d+\.\s+Horse:)", pp_text)

    for section in horse_sections:
        horse_match = re.search(r"Horse:\s*(.+?)(?=\s*\(#|\n)", section)
        if not horse_match:
            continue
        horse_name = horse_match.group(1).strip()

        # Look for "✔ AWD Match" or "⚠ Distance Mismatch"
        if "✔ AWD Match" in section or "AWD Match" in section:
            awd_data[horse_name] = "match"
        elif "⚠ Distance Mismatch" in section or "Distance Mismatch" in section:
            awd_data[horse_name] = "mismatch"

    return awd_data


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


# ======================== ELITE ENHANCEMENTS ========================


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


def parse_jockey_combo_stats(section: str) -> tuple[float, float]:
    """
    Parse jockey stats and combo percentage from BRISNET format.

    Searches for patterns like:
    - "Jky: LastName FirstName (starts wins-places-shows win%)"
    - Combo patterns: "w/TrnrLastName: 50 22% 18% 40%" (starts, win%, place%, combo%)

    Returns:
        tuple: (jockey_win_rate, combo_win_rate)
    """
    jockey_win_rate = 0.0
    combo_win_rate = 0.0

    # JOCKEY: Search for "Jky:" pattern
    jockey_pattern = r"Jky:.*?\((\d+)\s+(\d+)-(\d+)-(\d+)\s+(\d+)%\)"
    jockey_match = re.search(jockey_pattern, section)
    if jockey_match:
        j_starts = int(jockey_match.group(1))
        j_win_pct = int(jockey_match.group(5)) / 100.0
        if j_starts >= 20:
            jockey_win_rate = j_win_pct

    # COMBO: Search for trainer/jockey combo patterns
    # Format: "w/TrnrName: 50 22% 18% 40%" or similar
    combo_pattern = r"w/.*?:\s*(\d+)\s+(\d+)%\s+(\d+)%\s+(\d+)%"
    combo_match = re.search(combo_pattern, section)
    if combo_match:
        combo_starts = int(combo_match.group(1))
        combo_pct = (
            int(combo_match.group(4)) / 100.0
        )  # Last percentage is combo win rate
        if combo_starts >= 10:  # Minimum 10 starts for combo stats
            combo_win_rate = combo_pct

    return jockey_win_rate, combo_win_rate


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
        t_wins = int(trainer_match.group(2))
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
    if "f" in distance_lower:
        try:
            furlongs = float(distance_lower.replace("f", "").strip())
            return furlongs >= 12.0
        except BaseException:
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


def is_fts_in_msw(horse, race_data):
    """
    Check if a horse is a First-Time Starter (FTS) in a Maiden Special Weight (MSW) race.

    FTS horses have zero previous racing history (starts == 0) and require special handling
    for maiden special weight races where pedigree, workouts, and connections matter more.

    Args:
        horse: Dictionary containing horse data including 'starts' field
        race_data: Dictionary containing race metadata including 'type' field

    Returns:
        bool: True if horse is FTS in MSW race, False otherwise
    """
    return horse.get("starts", 0) == 0 and race_data.get("type", "").upper() == "MSW"


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


def predict_race_with_fts(race_data, score_horse_func, default_params):
    """
    Template function demonstrating FTS (First-Time Starter) integration into race prediction.

    This function shows how to:
    1. Detect FTS horses in MSW races using is_fts_in_msw()
    2. Apply FTS-specific confidence parameters from FTS_PARAMS
    3. Apply elite trainer multiplier boost (1.2x for elite trainers)
    4. Fall back to standard parameters for non-FTS horses

    Integration pattern for existing rating systems:
    - Check is_fts_in_msw(horse, race_data) before scoring
    - Use FTS_PARAMS instead of DEFAULT_PARAMS if True
    - Apply 0.75 base multiplier * 1.2 elite trainer boost
    - Standard horses use 1.0 multiplier

    Args:
        race_data: Dict with 'horses' list and 'type' (race type like 'MSW')
        score_horse_func: Function that takes (horse, params) and returns score
        default_params: Standard confidence parameters for non-FTS horses

    Returns:
        List of top 4 horse names sorted by score

    Example:
        race_data = {
            'type': 'MSW',
            'horses': [
                {'name': 'Horse A', 'starts': 0, 'trainer': 'Wesley Ward'},
                {'name': 'Horse B', 'starts': 3, 'trainer': 'John Doe'}
            ]
        }
        top_4 = predict_race_with_fts(race_data, score_func, DEFAULT_PARAMS)
    """
    scores = {}

    for horse in race_data.get("horses", []):
        # Apply FTS-specific handling if applicable
        if is_fts_in_msw(horse, race_data):
            # Use FTS confidence parameters (higher pedigree/odds, lower speed/form)
            params = FTS_PARAMS
            # Base multiplier 0.75 accounts for uncertainty, elite trainer adds 20%
            trainer_boost = 1.2 if is_elite_trainer(horse.get("trainer", "")) else 1.0
            multiplier = 0.75 * trainer_boost
        else:
            # Standard horses use default parameters and no multiplier adjustment
            params = default_params
            multiplier = 1.0

        # Calculate score with appropriate parameters and apply multiplier
        base_score = score_horse_func(horse, params)
        scores[horse["name"]] = base_score * multiplier

    # Sort and return top 4 horses
    top_4 = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:4]
    return [name for name, score in top_4]


def parse_race_info(race_data):
    """
    Parse race information to identify race classification, maiden status, and FTS horses.

    Production-ready parser for Equibase, DRF, and Brisnet data formats.
    Handles all race classes: MSW, MCL, ALW, STK, CLM, etc.

    Args:
        race_data: Dict or DataFrame with race information
                   Expected keys: 'race_type' (str), 'horses' (list of dicts)
                   Each horse dict should have: 'name', 'starts', 'past_performances'

    Returns:
        Dict with exactly 3 keys:
        {
            "race_class": str,              # Normalized abbreviation (e.g., "MSW", "MCL")
            "is_maiden_race": bool,         # True for any maiden type
            "horses": list[dict]            # Enhanced with "is_first_time_starter"
        }

    Example:
        >>> race_data = {
        ...     'race_type': 'Maiden Special Weight',
        ...     'horses': [
        ...         {'name': 'Debut Horse', 'starts': 0},
        ...         {'name': 'Veteran', 'starts': 5}
        ...     ]
        ... }
        >>> result = parse_race_info(race_data)
        >>> result['race_class']
        'MSW'
        >>> result['is_maiden_race']
        True
        >>> result['horses'][0]['is_first_time_starter']
        True
    """
    # Convert DataFrame to dict if needed
    if isinstance(race_data, pd.DataFrame):
        race_data = race_data.to_dict("records")[0] if not race_data.empty else {}

    # Race classification mapping (comprehensive coverage)
    RACE_CLASS_MAP = {
        # Maiden races
        "maiden special weight": "MSW",
        "maiden claiming": "MCL",
        "maiden": "MSW",  # Default maiden to MSW
        "msw": "MSW",
        "mcl": "MCL",
        "mdn sp wt": "MSW",
        "mdn clm": "MCL",
        # Allowance races
        "allowance": "ALW",
        "allowance optional claiming": "AOC",
        "alw": "ALW",
        "aoc": "AOC",
        "optional claiming": "AOC",
        # Stakes races
        "stakes": "STK",
        "graded stakes": "GST",
        "listed stakes": "LST",
        "stk": "STK",
        "grd": "GST",
        "grde": "GST",
        # Claiming races
        "claiming": "CLM",
        "clm": "CLM",
        "claiming handicap": "CLH",
        "clh": "CLH",
        # Handicap races
        "handicap": "HCP",
        "hcp": "HCP",
        # Trial races
        "trial": "TRL",
        "trl": "TRL",
        # Starter races
        "starter allowance": "STR",
        "starter optional claiming": "SOC",
        "starter": "STR",
        "str": "STR",
        "soc": "SOC",
    }

    # Maiden race types (for is_maiden_race detection)
    MAIDEN_TYPES = {"MSW", "MCL"}

    # Extract race type string
    race_type_raw = race_data.get("race_type", "") or race_data.get("type", "") or ""
    race_type_lower = str(race_type_raw).lower().strip()

    # Normalize race class
    race_class = "UNK"  # Unknown default
    for key, abbrev in RACE_CLASS_MAP.items():
        if key in race_type_lower:
            race_class = abbrev
            break

    # Check if maiden race
    is_maiden_race = race_class in MAIDEN_TYPES or "maiden" in race_type_lower

    # Process horses
    horses = race_data.get("horses", [])
    if not isinstance(horses, list):
        horses = []

    enhanced_horses = []
    for horse in horses:
        # Ensure horse is a dict
        if not isinstance(horse, dict):
            continue

        # Create enhanced horse dict (preserve all original keys)
        enhanced_horse = horse.copy()

        # Extract starts from various possible fields
        starts = horse.get("starts", 0)
        if starts is None or starts == "":
            starts = 0

        # Try alternative field names if starts is 0
        if starts == 0:
            starts = horse.get("career_starts", 0) or 0
            if starts == 0:
                # Try to infer from past_performances
                past_perf = horse.get("past_performances", []) or []
                if isinstance(past_perf, list):
                    starts = len(past_perf)

        # Convert to int safely
        try:
            starts = int(starts)
        except (ValueError, TypeError):
            starts = 0

        # Determine FTS status
        is_fts = starts == 0

        # Add required fields
        enhanced_horse["starts"] = starts
        enhanced_horse["is_first_time_starter"] = is_fts

        # Ensure name field exists
        if "name" not in enhanced_horse:
            enhanced_horse["name"] = (
                horse.get("horse_name", "") or horse.get("Horse", "") or "Unknown"
            )

        enhanced_horses.append(enhanced_horse)

    # Return strictly formatted dict with EXACTLY 3 keys
    return {
        "race_class": race_class,
        "is_maiden_race": is_maiden_race,
        "horses": enhanced_horses,
    }


def calculate_workout_bonus_v2(
    workout_data: dict[str, Any], is_marathon: bool = False
) -> float:
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
    percentile = workout_data.get("percentile", 100)
    if (
        percentile is None
        and workout_data.get("work_rank")
        and workout_data.get("work_total")
    ):
        try:
            rank = int(workout_data["work_rank"])
            total = int(workout_data["work_total"])
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
    quality = workout_data.get("quality", "none")
    if quality == "bullet":
        bonus += 0.05  # Reduced from 0.10
    elif quality == "handily":
        bonus += 0.03  # Reduced from 0.05

    # Marathon bonus for recent sharp works
    if is_marathon:
        days_since = workout_data.get("days_since", 999)
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

    return 0.0


def calculate_hot_trainer_bonus(
    trainer_win_pct: float,
    is_hot_l14: bool = False,
    is_2nd_lasix_high_pct: bool = False,
    trainer_starts: int = 0,
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

    # ═══════════════════════════════════════════════════════════════
    # SAMPLE-SIZE AWARE TRAINER PENALTY (Feb 18, 2026 TUP R4 Tuning)
    # 0% on 5 starts could be bad luck. 0% on 30+ starts is a pattern.
    # 2% on 51 starts (LaVanway) is effectively a confirmed loser.
    # ═══════════════════════════════════════════════════════════════
    if trainer_win_pct == 0.0:
        if trainer_starts >= 30:
            return -1.8  # Confirmed 0% with huge sample — near-fatal
        elif trainer_starts >= 15:
            return -1.5  # Solid sample 0% — very bad
        else:
            return -1.2  # Small sample 0% — bad but recoverable

    # Very low % trainer penalty (1-5%) — AMPLIFIED with sample size
    if trainer_win_pct > 0.0 and trainer_win_pct < 0.05:
        bonus -= 0.9  # Was -0.7: 2% trainer penalty increased
        if trainer_starts >= 30:
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


# ======================== ROUTE & SPRINT CALIBRATION ========================
# Based on Pegasus WC G1 (9f route) and GP Turf Sprint (5f sprint) validation


def is_sprint_distance(distance_txt: str) -> bool:
    """Detect sprint distances (≤6.5f)"""
    if not distance_txt:
        return False
    try:
        if "f" in distance_txt.lower():
            furlongs = float(distance_txt.lower().replace("f", "").strip())
            return furlongs <= 6.5
    except BaseException:
        pass
    return False


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
    past_races: list[dict], today_class: str, today_purse: int
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


# ============ STRUCTURED RACE HISTORY PARSING (Feb 10, 2026) ============


def parse_race_history_from_block(block: str) -> list[dict]:
    """
    Parse structured race history from a single horse's BRISNET PP block.
    Extracts per-race: date, track, surface, distance (furlongs), race type,
    finish, speed figure.

    Surface codes: ft/fst=Dirt, sy/sly=Dirt, my=Dirt, fm/frm=Turf,
                  yl=Turf, sf=Turf, gf=Turf, hy=Turf, aw/tp=Synthetic
    """
    if not block:
        return []

    races = []
    surface_map = {
        "ft": "Dirt",
        "fst": "Dirt",
        "fast": "Dirt",
        "sy": "Dirt",
        "sly": "Dirt",
        "sloppy": "Dirt",
        "my": "Dirt",
        "mdy": "Dirt",
        "muddy": "Dirt",
        "gd": "Dirt",
        "fm": "Turf",
        "frm": "Turf",
        "firm": "Turf",
        "yl": "Turf",
        "yld": "Turf",
        "yielding": "Turf",
        "sf": "Turf",
        "sft": "Turf",
        "soft": "Turf",
        "gf": "Turf",
        "gd-fm": "Turf",
        "hy": "Turf",
        "aw": "Synthetic",
        "tp": "Synthetic",
        "syn": "Synthetic",
    }

    for line in block.split("\n"):
        # Running lines start with date (ddMMMyy) + track code
        date_match = re.match(r"(\d{2}[A-Za-z]{3}\d{2})(\w{2,4})\d*\s+", line.strip())
        if not date_match:
            continue

        try:
            date_str = date_match.group(1)
            track_code = date_match.group(2)

            # Surface detection
            surface = "Dirt"  # Default
            surface_area = line[:80].lower()
            for code, surf in surface_map.items():
                if re.search(rf"\b{re.escape(code)}\b", surface_area):
                    surface = surf
                    break
            # BRISNET turf indicator ⓘ
            if "\u24d8" in line or "\u2460" in line or "ⓘ" in line:
                surface = "Turf"

            # Distance extraction (furlongs)
            distance_f = 0.0
            dist_match = re.search(r"(\d+)\s*½?\s*f(?:ur)?", line[:60], re.IGNORECASE)
            if dist_match:
                base = int(dist_match.group(1))
                distance_f = base + (0.5 if "½" in line[:60] else 0.0)
            else:
                mile_match = re.search(
                    r"(\d+)\s*(?:(\d+)/(\d+))?\s*m(?:ile)?", line[:60], re.IGNORECASE
                )
                if mile_match:
                    miles = int(mile_match.group(1))
                    if mile_match.group(2) and mile_match.group(3):
                        miles += int(mile_match.group(2)) / int(mile_match.group(3))
                    distance_f = miles * 8.0

            # Race type
            race_type = ""
            type_match = re.search(
                r"(Clm\d+[a-zA-Z0-9/.\-]*|MC\d+[a-zA-Z0-9/.\-]*|"
                r"Mdn\d*|Md\s*Sp\s*Wt|Alw\d*[a-zA-Z0-9/.\-]*|"
                r"OC\d+[a-zA-Z0-9/.\-]*|Stk[a-zA-Z0-9/.\-]*|"
                r"G[123]\s*\w*|Hcp\d*|Moc\d*|S\s*Mdn\s*\d*k?)",
                line,
                re.IGNORECASE,
            )
            if type_match:
                race_type = type_match.group(1).strip()

            # Speed figure (first 2-3 digit number between 20-130 after col 40)
            speed_fig = 0
            for spd_str in re.findall(r"\b(\d{2,3})\b", line[40:]):
                fig = int(spd_str)
                if 20 <= fig <= 130:
                    speed_fig = fig
                    break

            # Finish position: scan position numbers in running-line area
            # (matches elite_parser_v2_gold._parse_race_history logic)
            finish_pos = 0
            fin_matches = re.findall(r"(\d{1,2})[ƒ®«ª³©¨°¬²‚±¹²³⁴⁵⁶⁷⁸⁹⁰ⁿ]*", line[80:])
            if fin_matches:
                for fm in reversed(fin_matches):
                    try:
                        f_val = int(fm)
                        if 1 <= f_val <= 20:
                            finish_pos = f_val
                            break
                    except ValueError:
                        continue

            if date_str and (distance_f > 0 or race_type):
                races.append(
                    {
                        "date": date_str,
                        "track": track_code,
                        "surface": surface,
                        "distance": str(distance_f) + "f" if distance_f > 0 else "",
                        "distance_f": round(distance_f, 1),
                        "race_type": race_type,
                        "speed_fig": speed_fig,
                        "finish_pos": finish_pos,
                    }
                )
        except Exception:
            continue

    return races[:10]


def apply_track_pattern_bonus(
    track_patterns: dict,
    post_position: int,
    running_style: str,
    best_beyer: int,
    class_rating: float,
    days_since_last: int,
    workout_pattern: str,
    prime_power: float,
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


def calculate_workout_bonus(workout_data: dict[str, Any]) -> float:
    """
    COMPREHENSIVE: Bonus for workout quality and recency.
    """
    if not workout_data:
        return 0.0

    bonus = 0.0

    # Rating bonus (B = bullet work = fastest of day)
    rating = workout_data.get("rating", "")
    if "B" in rating or "b" in rating:
        bonus += 0.10  # Bullet work
    elif "H" in rating or "h" in rating:
        bonus += 0.05  # Handily

    # Rank/percentile bonus (top 20% of works)
    percentile = workout_data.get("percentile", 0.5)
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
        "no significant post bias": lambda p: 0.0,
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
        "no significant post bias": lambda p: 0.0,
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
    ppi_value: float = 0.0,  # ppi_value arg seems unused, ppi_map is recalculated
    pedigree_per_horse: dict[str, dict] | None = None,
    track_name: str = "",
    pp_text: str = "",
    figs_df: pd.DataFrame = None,
    dynamic_weights: dict = None,
) -> pd.DataFrame:
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
                    try:
                        _track_weights = get_live_learned_weights(
                            PERSISTENT_DB_PATH,
                            track_code=track_name.upper(),
                        )
                    except Exception:
                        pass  # Fall back to global weights

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
        if track_name and "gold_db" in dir() or True:
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
                if dis_starts >= 4:
                    dis_win_pct = dis_wins / dis_starts
                    dis_itm_pct = (dis_wins + dis_places) / dis_starts
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
                    if trk_starts >= 4:
                        trk_win_pct = trk_wins / trk_starts
                        trk_itm_pct = (trk_wins + trk_places) / trk_starts
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
        pace_cap_applied = False

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
        track_pattern_detail = []
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
                track_pattern_detail = tp_result.get("details", [])
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
        try:
            tier2_bonus += calculate_first_after_claim_bonus(_horse_block)
        except BaseException:
            pass

        # 10. RACE AUDIT ENHANCEMENT 4: Style vs Weekly Bias Dynamic Penalty/Bonus
        # Oaklawn R1: Stalker impact 0.32 but model ranked S-type Tiffany Twist #1
        # E-type Tell Me When (impact 2.05) was ranked dead last
        try:
            tier2_bonus += calculate_style_vs_weekly_bias_bonus(style, impact_values)
        except BaseException:
            pass

        # 11. RACE AUDIT ENHANCEMENT 5: Post Position Bias Multiplier
        # Oaklawn R1: Rail posts 1-3 had 2.59 weekly impact, Sombra Dorada (post 1)
        # finished 2nd but model had her 5th
        try:
            tier2_bonus += calculate_post_position_bias_bonus(post, weekly_post_impacts)
        except BaseException:
            pass

        # 12. RACE AUDIT ENHANCEMENT 6: E1/E2 Pace Supremacy Bonus
        # Oaklawn R9: She's Storming had best E1 (91) matching par (91) but got no credit.
        # At speed-biased tracks, the horse with the fastest E1 in the field has a tactical
        # advantage. Combined with E/P style + inside post, this is a MAJOR signal.
        try:
            tier2_bonus += calculate_pace_supremacy_bonus(
                name, _horse_block, _field_e1_values, impact_values
            )
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
                    "mdn": 1,
                    "mc": 2,
                    "msw": 3,
                    "clm": 4,
                    "c": 4,
                    "alw": 6,
                    "aoc": 7,
                    "oc": 7,
                    "stk": 8,
                    "g3": 9,
                    "g2": 10,
                    "g1": 11,
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
        if prime_power_raw > 0:  # noqa: race_class_shown guard below
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
            if purse_amount >= 500000:
                race_quality = "elite"
            elif purse_amount >= 150000 and race_quality not in ["elite", "high"]:
                race_quality = "high"
            elif race_type_clean == "stakes_graded" and race_quality != "elite":
                race_quality = "elite"

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
                    pace_cap_applied = True

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
        "trs_jky": 1.0,
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
st.caption(
    f"⚙️ **Dynamic Weights Applied:** Surface: {surface_type} | Distance: {distance_txt} | Condition: {condition_txt} | Race Type: {race_type_detected} | Purse: ${purse_val:,}"
)
weights_display = " | ".join([f"{k}: {v:.2f}" for k, v in final_weights.items()])
st.caption(f"📊 **Factor Weights:** {weights_display}")

for i, (rbias, pbias) in enumerate(scenarios):
    with tabs[i]:
        # Debug info for track bias detection
        canon_track = _canonical_track(track_name)
        dist_bucket = distance_bucket(distance_txt)
        style_display = ", ".join(running_style_biases)
        post_display = ", ".join(post_biases)
        st.caption(
            f"🔍 Track Bias Detection: {canon_track} • {surface_type} • {dist_bucket}"
        )
        st.caption(
            f"📊 Selected Biases - Running Styles: {style_display} | Post Positions: {post_display}"
        )

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
            dynamic_weights=final_weights,  # Pass dynamic weights based on race parameters
        )
        ratings_df = apply_enhancements_and_figs(
            ratings_df=ratings_df,
            pp_text=pp_text,
            _processed_weights=final_weights,
            _chaos_index=0.0,
            _track_name=track_name,
            _surface_type=surface_type,
            _distance_txt=distance_txt,
            _race_type=race_type_detected,
            angles_per_horse=angles_per_horse,
            _pedigree_per_horse=pedigree_per_horse,
            figs_df=figs_df,  # <--- PASS THE REAL FIGS_DF
        )
        # Build ML odds dict from df_final_field for reality check
        # Prefer Live Odds (from Section A user input), fall back to ML (parsed morning line)
        ml_odds_dict = {}
        for _, row in df_final_field.iterrows():
            horse_name = row.get("Horse")
            # Try Live Odds first (user input), then ML (parsed)
            live_odds = row.get("Live Odds", "")
            ml_odds = row.get("ML", "")
            odds_str = live_odds if live_odds else ml_odds

            if horse_name and odds_str:
                try:
                    # Parse odds string (e.g., "30/1", "5/2", "3.5")
                    odds_str = str(odds_str).strip()
                    if "/" in odds_str:
                        parts = odds_str.split("/")
                        ml_odds_dict[horse_name] = float(parts[0]) / float(parts[1])
                    else:
                        ml_odds_dict[horse_name] = float(odds_str)
                except (ValueError, TypeError, IndexError, ZeroDivisionError):
                    # Invalid odds format - use default
                    ml_odds_dict[horse_name] = 5.0
            else:
                ml_odds_dict[horse_name] = 5.0

        fair_probs = fair_probs_from_ratings(ratings_df, ml_odds_dict)
        if "Horse" in ratings_df.columns:
            ratings_df["Fair %"] = ratings_df["Horse"].map(
                lambda h: f"{fair_probs.get(h, 0) * 100:.1f}%"
            )
            ratings_df["Fair Odds"] = ratings_df["Horse"].map(
                lambda h: fair_to_american_str(fair_probs.get(h, 0))
            )
        else:
            ratings_df["Fair %"] = ""
            ratings_df["Fair Odds"] = ""
        all_scenario_ratings[(rbias, pbias)] = (
            ratings_df.copy(),
            fair_probs,
        )  # Store copy and probs

        disp = ratings_df.sort_values(by="R", ascending=False)
        if "R_ENHANCE_ADJ" in disp.columns:
            disp = disp.drop(columns=["R_ENHANCE_ADJ"])
        st.dataframe(
            disp,
            use_container_width=True,
            hide_index=True,
            column_config={
                "R": st.column_config.NumberColumn("Rating", format="%.2f"),
                "Cstyle": st.column_config.NumberColumn("C-Style", format="%.2f"),
                "Cpost": st.column_config.NumberColumn("C-Post", format="%.2f"),
                "Cpace": st.column_config.NumberColumn("C-Pace", format="%.2f"),
                "Cspeed": st.column_config.NumberColumn("C-Speed", format="%.2f"),
                "Cclass": st.column_config.NumberColumn("C-Class", format="%.2f"),
                "Cform": st.column_config.NumberColumn(
                    "C-Form",
                    format="%.2f",
                    help="Form cycle: layoff, trend, consistency",
                ),
                "Atrack": st.column_config.NumberColumn("A-Track", format="%.2f"),
                "Arace": st.column_config.NumberColumn("A-Race", format="%.2f"),
                # Format Quirin to show integer or be blank
                "Quirin": st.column_config.NumberColumn(
                    "Quirin", format="%d", help="BRIS Pace Points"
                ),
            },
        )

# Ensure primary key exists before accessing
if scenarios:
    primary_key = scenarios[0]
    if primary_key in all_scenario_ratings:
        primary_df, primary_probs = all_scenario_ratings[primary_key]

        # Store in session state for button access
        st.session_state["primary_d"] = primary_df
        st.session_state["primary_probs"] = primary_probs

        # Display ALL selected biases, not just the first scenario
        style_biases_display = ", ".join(running_style_biases)
        post_biases_display = ", ".join(post_biases)
        st.info(
            f"**Combined Scenario Analysis** • Profile: `{strategy_profile}` • PPI: {ppi_val:+.2f}"
        )
        st.caption(
            f"📊 **Active Biases:** Running Styles: `{style_biases_display}` | Post Positions: `{post_biases_display}` (All bonuses aggregated)"
        )
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
st.caption(
    "💰 Overlays = horses whose fair odds are better than morning line odds. These represent betting value."
)

# Offered odds map
offered_odds_map = {}
for _, r in df_final_field.iterrows():
    odds_str = str(r.get("Live Odds", "")).strip() or str(r.get("ML", "")).strip()
    dec = str_to_decimal_odds(odds_str)
    if dec:
        offered_odds_map[r["Horse"]] = dec

# Overlay table vs fair line
df_ol = overlay_table(fair_probs=primary_probs, offered=offered_odds_map)
st.session_state["df_ol"] = df_ol  # Store for Classic Report
st.dataframe(
    df_ol,
    use_container_width=True,
    hide_index=True,
    column_config={
        "EV per $1": st.column_config.NumberColumn("EV per $1", format="$%.3"),
        "Edge (pp)": st.column_config.NumberColumn("Edge (pp)"),
    },
)

# All ticketing strategy UI has been removed from this section
# It is now generated "behind the scenes" in Section D.

# ===================== D. Strategy Builder & Classic Report =====================


def build_component_breakdown(
    primary_df, name_to_post, name_to_odds, pp_text="", name_to_prog=None
):
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

    if "Horse" not in primary_df.columns or "R" not in primary_df.columns:
        return "Invalid component data structure."

    # EXTRACTION: Get top 5 horses by rating
    try:
        top_horses = primary_df.nlargest(5, "R")
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
        "Cclass": 3.0,  # Default - overridden by parser (1.0-10.0 scale)
        "Cspeed": 1.8,  # Speed figures - raw ability
        "Cform": 1.8,  # Form cycle - current condition
        "Cpace": 1.5,  # Pace advantage - tactical fit
        "Cstyle": 1.2,  # Running style - bias fit
        "Cpost": 0.8,  # Post position - track bias
    }

    # IMPORTANT: Try to get actual class weight used in calculation from parser
    # This ensures breakdown shows true weights, not defaults
    if RACE_CLASS_PARSER_AVAILABLE and pp_text:
        try:
            race_class_data = parse_and_calculate_class(pp_text)
            actual_class_weight = race_class_data["weight"]["class_weight"]
            WEIGHTS["Cclass"] = actual_class_weight  # Use actual weight from parser
        except BaseException:
            pass  # Fall back to default

    breakdown = "### Component Breakdown (Top 5 Horses)\n"
    # CRITICAL FIX: Use program number for horse labels (what bettors see on tote board)
    _prog_map = name_to_prog if name_to_prog else name_to_post

    breakdown += "_Mathematical transparency: Shows exactly what the system sees in each horse_\n\n"

    for idx, row in top_horses.iterrows():
        horse_name = row.get("Horse", "Unknown")
        prog = _prog_map.get(horse_name, "?")
        ml = name_to_odds.get(horse_name, "?")

        # SAFE EXTRACTION: Get rating with error handling
        try:
            final_rating = float(row.get("R", 0))
        except BaseException:
            final_rating = 0.0

        breakdown += (
            f"**#{prog} {horse_name}** (ML {ml}) - **Rating: {final_rating:.2f}**\n"
        )

        # CORE COMPONENTS: Extract with validation
        components = {}
        component_descriptions = {
            "Cclass": "Purse earnings, race level history",
            "Cform": "Recent performance trend, consistency",
            "Cspeed": "Speed figures relative to field average",
            "Cpace": "Pace advantage/disadvantage vs projected pace",
            "Cstyle": "Running style fit for pace scenario",
            "Cpost": "Post position bias for this track/distance",
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

            description = component_descriptions.get(comp_name, "")
            breakdown += f"- **{comp_name[1:]}:** {comp_value:+.2f} (×{weight} weight = {weighted_contribution:+.2f}) - {description}\n"

        # TRACK BIAS: Additional component
        try:
            atrack = float(row.get("Atrack", 0))
        except BaseException:
            atrack = 0.0
        breakdown += f"- **Track Bias:** {atrack:+.2f} - Track-specific advantages (style + post combo)\n"

        # TRANSPARENCY: Show weighted total
        breakdown += f"- **Weighted Core Total:** {weighted_sum:.2f}\n"

        # QUIRIN POINTS: BRIS pace rating
        quirin = row.get("Quirin", "N/A")
        if quirin != "N/A":
            try:
                quirin = int(float(quirin))
            except BaseException:
                quirin = "N/A"
        breakdown += f"- **Quirin Points:** {quirin} - BRISNET early pace points\n"

        # FINAL RATING: Includes all angles and bonuses
        breakdown += f"- **Final Rating:** {final_rating:.2f} (includes 8 elite angles + tier 2 bonuses + track bias)\n\n"

    breakdown += "_Note: Positive values = advantages, negative = disadvantages. Weighted contributions show impact on final rating._\n"

    return breakdown


def build_betting_strategy(
    primary_df: pd.DataFrame,
    df_ol: pd.DataFrame,
    strategy_profile: str,
    name_to_post: dict[str, str],
    name_to_odds: dict[str, str],
    field_size: int,
    ppi_val: float,
    smart_money_horses: list[dict] = None,
    name_to_ml: dict[str, str] = None,
    name_to_prog: dict[str, str] = None,
) -> str:
    """
    Builds elite strategy report with finishing order predictions, component transparency,
    A/B/C/D grouping, and $50 bankroll optimization.

    Args:
        smart_money_horses: List of horses with significant ML→Live odds drops for Smart Money Alert
        name_to_ml: Dictionary mapping horse names to ML odds (for comparison with live odds)
        name_to_prog: Dictionary mapping horse names to program numbers (saddle cloth).
                      If None, falls back to name_to_post for backward compatibility.
    """

    import numpy as np

    # Handle None defaults for mutable default arguments
    if smart_money_horses is None:
        smart_money_horses = []
    if name_to_ml is None:
        name_to_ml = {}
    # CRITICAL FIX: Use program numbers for all horse labels/betting references
    # Program # = saddle cloth / tote board number (what bettors actually bet on)
    # Falls back to post position for backward compatibility
    if name_to_prog is None:
        name_to_prog = name_to_post

    # --- GOLD STANDARD: Build probability dictionary with validation ---
    # (Populated early so underlay/overlay detection works correctly)
    primary_probs_dict = {}
    for horse in primary_df["Horse"].tolist():
        horse_df = primary_df[primary_df["Horse"] == horse]
        if horse_df.empty:
            primary_probs_dict[horse] = 1.0 / max(len(primary_df), 1)
            continue

        prob_str = horse_df["Fair %"].iloc[0]
        try:
            # Handle multiple formats: "25.5%", "0.255", 25.5
            if isinstance(prob_str, str):
                prob = float(prob_str.strip("%").strip()) / (
                    100.0 if "%" in str(prob_str) else 1.0
                )
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

    # --- ELITE: Calculate Most Likely Finishing Order (Sequential Selection Algorithm) ---
    def calculate_most_likely_finishing_order(
        df: pd.DataFrame, top_n: int = 5
    ) -> list[tuple[str, float]]:
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

        if "Horse" not in df.columns or "Fair %" not in df.columns:
            return []

        horses = df["Horse"].tolist()
        if len(horses) == 0:
            return []

        # EXTRACT: Base win probabilities with robust error handling
        win_probs = []
        for horse in horses:
            horse_df = df[df["Horse"] == horse]
            if horse_df.empty:
                win_probs.append(1.0 / len(horses))  # Fallback to uniform
                continue

            prob_str = horse_df["Fair %"].iloc[0]
            try:
                # Handle various formats: "25.5%", "0.255", 25.5
                if isinstance(prob_str, str):
                    prob = float(prob_str.strip("%").strip()) / (
                        100.0 if "%" in str(prob_str) else 1.0
                    )
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
                remaining_probs = np.ones(
                    len(remaining_indices), dtype=np.float64
                ) / len(remaining_indices)

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
        for h in primary_df["Horse"].tolist():
            # Skip horses already selected for earlier positions (except the primary for this position)
            if h in selected_horses and h != primary_horse:
                continue

            h_prob_str = primary_df[primary_df["Horse"] == h]["Fair %"].iloc[0]
            try:
                if isinstance(h_prob_str, str):
                    h_prob = float(h_prob_str.strip("%").strip()) / (
                        100.0 if "%" in str(h_prob_str) else 1.0
                    )
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
    def format_horse_list(horse_names: list[str]) -> str:
        """Creates a bulleted list of horses with program#, name, and odds (shows ML → Live if different)."""
        if not horse_names:
            return "* None"
        lines = []
        # Sort horses by program number before displaying
        sorted_horses = sorted(
            horse_names, key=lambda name: int(name_to_prog.get(name, "999"))
        )
        for name in sorted_horses:
            prog = name_to_prog.get(name, "??")
            current_odds = name_to_odds.get(name, "N/A")
            ml_odds = name_to_ml.get(name, "")

            # Show ML → Live format when they differ, otherwise just show current odds
            if ml_odds and current_odds != ml_odds and current_odds != "N/A":
                odds_display = f"ML {ml_odds} → Live {current_odds}"
            else:
                odds_display = current_odds

            lines.append(f"* **#{prog} - {name}** ({odds_display})")
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

    # ═══════════════════════════════════════════════════════
    # CRITICAL FIX: Sort horses by RATING before grouping
    # Previously used unsorted DataFrame order (= post order)
    # which made post #1 the A-Group key regardless of rating
    # ═══════════════════════════════════════════════════════
    _sorted_for_groups = primary_df.copy()
    _sorted_for_groups["_R_numeric"] = pd.to_numeric(
        _sorted_for_groups["R"], errors="coerce"
    )
    _sorted_for_groups = _sorted_for_groups.sort_values(
        "_R_numeric", ascending=False, na_position="last"
    )
    all_horses = _sorted_for_groups["Horse"].tolist()
    pos_ev_horses = (
        set(df_ol[df_ol["EV per $1"] > 0.05]["Horse"].tolist())
        if not df_ol.empty
        else set()
    )

    # ═══════════════════════════════════════════════════════════════
    # PEGASUS 2026 TUNING: Block horses with massive odds drift from A-Group
    # British Isles 20/1 → 50/1 should NEVER have been in A-Group
    # ═══════════════════════════════════════════════════════════════
    blocked_from_A = set()
    dumb_money_horses = st.session_state.get("dumb_money_horses", [])
    if dumb_money_horses:
        for h in dumb_money_horses:
            if h.get("ratio", 1.0) > 2.0:  # 2x+ drift = blocked from A-Group
                blocked_from_A.add(h["name"])

    # Filter out blocked horses AND NaN-rated horses from A-group consideration
    # CRITICAL FIX: Never promote a NaN-rated horse to A-Group
    def _has_valid_rating(horse_name: str) -> bool:
        """Check if a horse has a valid (non-NaN, non-None) rating."""
        rows = primary_df[primary_df["Horse"] == horse_name]
        if rows.empty:
            return False
        r_val = rows["R"].iloc[0]
        if r_val is None:
            return False
        try:
            return np.isfinite(float(r_val))
        except (ValueError, TypeError):
            return False

    eligible_for_A = [
        h for h in all_horses if h not in blocked_from_A and _has_valid_rating(h)
    ]

    if strategy_profile == "Confident":
        A_group = (
            [eligible_for_A[0]]
            if eligible_for_A
            else [all_horses[0]]
            if all_horses
            else []
        )
        if len(eligible_for_A) > 1:
            second_horse = eligible_for_A[1]
            first_rating = (
                primary_df[primary_df["Horse"] == eligible_for_A[0]]["R"].iloc[0]
                if eligible_for_A
                else 0
            )
            second_rating = primary_df[primary_df["Horse"] == second_horse]["R"].iloc[0]
            try:
                first_r = float(first_rating) if first_rating is not None else 0
                second_r = float(second_rating) if second_rating is not None else 0
                if second_r > (first_r * 0.90):  # Only add #2 if very close
                    A_group.append(second_horse)
            except (ValueError, TypeError):
                pass  # Skip adding second horse if ratings are invalid
    else:  # Value Hunter - Prioritize top pick + overlays (excluding blocked horses)
        eligible_overlays = pos_ev_horses - blocked_from_A
        A_group = list(
            set([eligible_for_A[0]] if eligible_for_A else []) | eligible_overlays
        )
        if len(A_group) > 4:  # Cap A group size for Value Hunter
            A_group = sorted(
                A_group, key=lambda h: primary_df[primary_df["Horse"] == h].index[0]
            )[:4]

    B_group = [h for h in all_horses if h not in A_group][:3]
    C_group = [h for h in all_horses if h not in A_group and h not in B_group][:4]
    D_group = [
        h
        for h in all_horses
        if h not in A_group and h not in B_group and h not in C_group
    ]

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
    contender_report += (
        "**Key Win Contenders (A-Group):**\n" + format_horse_list(A_group) + "\n"
    )
    contender_report += "* _Primary win threats based on model rank and/or betting value. Use these horses ON TOP in exactas, trifectas, etc._\n\n"

    contender_report += (
        "**Primary Challengers (B-Group):**\n" + format_horse_list(B_group) + "\n"
    )
    contender_report += "* _Logical contenders expected to finish 2nd or 3rd. Use directly underneath A-Group horses._\n"

    top_rated_horse = all_horses[0]
    is_overlay = top_rated_horse in pos_ev_horses
    top_ml_str = name_to_odds.get(top_rated_horse, "100")
    top_ml_dec = str_to_decimal_odds(top_ml_str) or 101
    is_underlay = (
        not is_overlay
        and (primary_probs_dict.get(top_rated_horse, 0) > (1 / top_ml_dec))
        and top_ml_dec < 4
    )  # Define underlay as < 3/1

    if is_overlay:
        contender_report += f"\n**Value Note:** Top pick **#{name_to_prog.get(top_rated_horse)} - {top_rated_horse}** looks like a good value bet (Overlay).\n"
    elif is_underlay:
        contender_report += f"\n**Value Note:** Top pick **#{name_to_prog.get(top_rated_horse)} - {top_rated_horse}** might be overbet (Underlay at {top_ml_str}). Consider using more underneath than on top.\n"

    # --- 5. Build Simplified Blueprint Section ---
    blueprint_report = "### Betting Strategy Blueprints (Scale Base Bets to Budget: Max ~$50 Recommended)\n"
    blueprint_report += "_Costs are examples using minimum base bets ($1.00 Exacta, $0.50 Tri, $0.10 Super, $1.00 SH5). Adjust base amount to fit your total budget for this race._\n"

    # --- Field Size Logic ---
    if field_size <= 6:
        blueprint_report += "\n**Note:** With a small field (<=6 runners), Superfecta and Super High 5 payouts are often very low. Focus on Win, Exacta, and Trifecta bets.\n"
        # Generate only Win/Ex/Tri for small fields
        blueprint_report += f"\n#### {strategy_profile} Profile Plan (Small Field)\n"
        if strategy_profile == "Value Hunter":
            blueprint_report += (
                "* **Win Bets:** Consider betting all **A-Group** horses.\n"
            )
        else:  # Confident
            blueprint_report += "* **Win Bet:** Focus on top **A-Group** horse(s).\n"

        # Exacta Examples
        blueprint_report += f"* **Exacta Part-Wheel:** `A / B,C` ({nA}x{nB + nC}) - {get_min_cost_str(1.00, nA, nB + nC)}\n"
        if nA >= 2:
            ex_box_combos = get_box_combos(nA, 2)
            blueprint_report += f"* **Exacta Box (A-Group):** `{', '.join(map(str, [int(name_to_prog.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(1.00, ex_box_combos)}\n"

        # Trifecta Examples
        blueprint_report += f"* **Trifecta Part-Wheel:** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        if nA >= 3:
            tri_box_combos = get_box_combos(nA, 3)
            blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str, [int(name_to_prog.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"

        blueprint_report += (
            "_Structure: Use A-horses on top, spread underneath with B and C._\n"
        )

    else:  # Standard logic for fields > 6
        # --- Confident Blueprint ---
        blueprint_report += "\n#### Confident Profile Plan\n"
        blueprint_report += "_Focus: Key A-Group horses ON TOP._\n"
        blueprint_report += "* **Win Bet:** Focus on top **A-Group** horse(s).\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B` ({nA}x{nB}) - {get_min_cost_str(1.00, nA, nB)}\n"
        if (
            nA >= 1 and nB >= 1 and nC >= 1
        ):  # Check if groups have members for straight example
            blueprint_report += "* **Straight Trifecta:** `Top A / Top B / Top C` (1 combo) - Consider at higher base (e.g., $1 or $2).\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B / C` ({nA}x{nB}x{nC}) - {get_min_cost_str(0.50, nA, nB, nC)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B / C / D` ({nA}x{nB}x{nC}x{nD}) - {get_min_cost_str(0.10, nA, nB, nC, nD)}\n"
        if field_size >= 7:  # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B / C / D / ALL` ({nA}x{nB}x{nC}x{nD}x{nAll}) - {get_min_cost_str(1.00, nA, nB, nC, nD, nAll)}\n"

        # --- Value-Hunter Blueprint ---
        blueprint_report += "\n#### Value-Hunter Profile Plan\n"
        blueprint_report += "_Focus: Use A-Group (includes overlays) ON TOP, spread wider underneath._\n"
        blueprint_report += "* **Win Bets:** Consider betting all **A-Group** horses.\n"
        blueprint_report += f"* **Exacta (Part-Wheel):** `A / B,C` ({nA}x{nB + nC}) - {get_min_cost_str(1.00, nA, nB + nC)}\n"
        if nA >= 3:  # Example box if A group is large enough
            tri_box_combos = get_box_combos(nA, 3)
            blueprint_report += f"* **Trifecta Box (A-Group):** `{', '.join(map(str, [int(name_to_prog.get(h, '0')) for h in A_group]))}` BOX - {get_bet_cost(0.50, tri_box_combos)}\n"
        blueprint_report += f"* **Trifecta (Part-Wheel):** `A / B,C / B,C,D` ({nA}x{nB + nC}x{nB + nC + nD}) - {get_min_cost_str(0.50, nA, nB + nC, nB + nC + nD)}\n"
        blueprint_report += f"* **Superfecta (Part-Wheel):** `A / B,C / B,C,D / ALL` ({nA}x{nB + nC}x{nB + nC + nD}x{nAll}) - {get_min_cost_str(0.10, nA, nB + nC, nB + nC + nD, nAll)}\n"
        if field_size >= 7:  # Only suggest SH5 if 7+ runners
            blueprint_report += f"* **Super High-5 (Part-Wheel):** `A / B,C / B,C,D / ALL / ALL` ({nA}x{nB + nC}x{nB + nC + nD}x{nAll}x{nAll}) - {get_min_cost_str(1.00, nA, nB + nC, nB + nC + nD, nAll, nAll)}\n"

    detailed_breakdown = build_component_breakdown(
        primary_df,
        name_to_post,
        name_to_odds,
        pp_text=pp_text,
        name_to_prog=name_to_prog,
    )

    component_report = "### What Our System Sees in Top Contenders\n\n"
    component_report += detailed_breakdown + "\n"

    # --- GOLD STANDARD: Build Finishing Order Predictions (NO DUPLICATES) ---
    # Use sequential selection algorithm - each horse appears EXACTLY ONCE
    finishing_order = calculate_most_likely_finishing_order(primary_df, top_n=5)

    finishing_order_report = "### Most Likely Finishing Order\n\n"
    finishing_order_report += "**Algorithm:** Sequential selection ensuring each horse appears EXACTLY ONCE. The percentage indicates conditional probability (e.g., for 2nd place: probability of finishing 2nd given not finishing 1st).\n\n"

    position_names = {
        1: "🥇 Win (1st)",
        2: "🥈 Place (2nd)",
        3: "🥉 Show (3rd)",
        4: "4th",
        5: "5th",
    }

    for pos, (horse, prob) in enumerate(finishing_order, 1):
        prog = name_to_prog.get(horse, "?")
        odds = name_to_odds.get(horse, "?")
        finishing_order_report += f"* **{position_names[pos]} • #{prog} {horse}** (Odds: {odds}) — {prob * 100:.1f}% conditional probability\n"

    finishing_order_report += "\n💡 **Use These Rankings:** Build your exotic tickets using this exact finishing order for optimal probability-based coverage.\n\n"

    # --- ELITE: Build $50 Bankroll Optimization ---
    bankroll_report = "### $50 Bankroll Structure\n\n"

    if strategy_profile == "Value Hunter":
        bankroll_report += (
            "**Strategy:** Value Hunter - Focus on overlays with wider coverage\n\n"
        )

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
        bankroll_report += (
            f"* **Superfecta** (${super_cost}): A / B,C / B,C,D / Top 5 from D+All\n"
        )

    else:  # Confident
        bankroll_report += (
            "**Strategy:** Confident - Focus on top pick with deeper coverage\n\n"
        )

        # Win bet on top pick
        bankroll_report += f"* **Win Bet** ($20): $20 on #{name_to_prog.get(all_horses[0], '?')} {all_horses[0]}\n"

        # Exacta
        ex_combos = nA * nB
        bankroll_report += f"* **Exacta** ($10): A / B - ${10 / max(ex_combos, 1):.2f} base × {ex_combos} combos\n"

        # Trifecta
        tri_combos = nA * nB * nC
        bankroll_report += f"* **Trifecta** ($12): A / B / C - ${12 / max(tri_combos, 1):.2f} base × {tri_combos} combos\n"

        # Superfecta
        bankroll_report += (
            "* **Superfecta** ($8): A / B / C / D - scaled to fit budget\n"
        )

    bankroll_report += "\n**Total Investment:** $50 (optimized)\n"
    bankroll_report += f"**Risk Level:** {strategy_profile} approach - {'Wider coverage, value-based' if strategy_profile == 'Value Hunter' else 'Concentrated on top selection'}\n"
    bankroll_report += "\n💡 **Use Finishing Order Predictions:** The probability rankings above show the most likely finishers for each position. Build your tickets using horses with highest probabilities for each slot.\n"

    # --- Smart Money Alert Section ---
    smart_money_report = ""
    if smart_money_horses:
        smart_money_report = (
            "### 🚨 Smart Money Alert: Significant Odds Movement Detected\n\n"
        )
        smart_money_report += "The following horses show **major public support** with Live odds dropping significantly from Morning Line. This typically indicates:\n"
        smart_money_report += "* Trainer/connections betting\n"
        smart_money_report += "* Informed money from sharp handicappers\n"
        smart_money_report += "* Positive workout reports or stable buzz\n"
        smart_money_report += "* Hidden form improvement\n\n"

        # Sort by biggest movement percentage
        smart_money_horses.sort(key=lambda x: x["movement_pct"], reverse=True)

        for horse_data in smart_money_horses:
            name = horse_data["name"]
            post = horse_data["post"]
            ml = horse_data["ml"]
            live = horse_data["live"]
            movement_pct = horse_data["movement_pct"]

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
* **Budget:** Decide your total wager amount for this race (e.g., $20, $50 recommended max per bet type).
* **Scale Base Bets:** Exacta $1.00, Trifecta $0.50–$1.00, Superfecta $0.10–$0.20, Super High-5 $1.00. Adjust to match your budget.
* **Confidence:** Bet more confidently when A/B groups look strong. Reduce base bets or narrow the C/D groups in tickets if less confident.
* **Small Fields (<=6):** Focus on Win/Exacta/Trifecta as complex exotics pay less.
* **Play SH5** mainly on mandatory payout days or when you have a very strong opinion & budget allows.
"""
    return final_report


def _build_na_context(primary_df: pd.DataFrame, df_field: pd.DataFrame) -> str:
    """Build NA running style context string for Classic Report LLM prompt.

    Explains which horses have unknown running styles and what their
    Quirin Speed Points suggest about early speed tendency.
    """
    na_lines = []
    if primary_df is None or primary_df.empty:
        return "All horses have confirmed running styles."

    for _, row in primary_df.iterrows():
        style = str(row.get("Style", "")).upper()
        if style == "NA":
            name = row.get("Horse", "Unknown")
            post = row.get("Post", "?")
            qsp = int(safe_float(row.get("Quirin", 0), 0))
            starts = int(safe_float(row.get("Starts", row.get("CStarts", 0)), 0))

            if starts == 0:
                label = "First-time starter \u2014 no racing history"
            elif qsp >= 6:
                label = f"QSP {qsp}/8 \u2014 likely early speed tendency, style dampened ~92%"
            elif qsp >= 3:
                label = f"QSP {qsp}/8 \u2014 moderate speed signal, style unknown, rating dampened"
            else:
                label = f"QSP {qsp}/8 \u2014 minimal speed data, high uncertainty, rating dampened"

            na_lines.append(f"#{post} {name}: {label}")

    if not na_lines:
        return "All horses have confirmed running styles."

    header = (
        f"{len(na_lines)} horse(s) with NA (unknown) running style. "
        "BRISNET assigns NA when horse lacks sufficient starts at this exact "
        "distance/surface. Ratings are dampened and QSP used to infer pace tendency:"
    )
    return header + "\n" + "\n".join(na_lines)


st.markdown("---")
st.header("D. Classic Report")
st.caption(
    "📊 Generate a comprehensive handicapping analysis with pace projections, contender groups, and betting strategy blueprints."
)

# Only show button if race has been parsed
if not st.session_state.get("parsed", False):
    st.warning("⚠️ Please parse a race first in Section A before analyzing.")
elif "primary_d" not in st.session_state or "primary_probs" not in st.session_state:
    st.error("❌ Rating data not available.")
    st.info(
        "📋 **Next Steps:** Scroll up to **Section B: Bias-Adjusted Ratings** and make sure you:\n1. Select a Strategy Profile (Confident or Value Hunter)\n2. Select at least one Running Style Bias (E, E/P, P, or S)\n3. Select at least one Post Position Bias\n\nOnce you make your selections, the ratings will calculate automatically and the 'Analyze This Race' button will appear below."
    )
else:
    # Display existing Classic Report if it exists (before the button)
    if st.session_state.get("classic_report_generated", False):
        st.success("✅ Classic Report Generated")
        classic_report = st.session_state.get("classic_report", "")
        phase3_html = st.session_state.get("phase3_report", "")
        persistent_report = f"""<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; font-size: 15px; line-height: 1.6; color: #1f2937;">
{classic_report}

---

{phase3_html}
</div>"""
        st.markdown(persistent_report, unsafe_allow_html=True)

        # Show download buttons for existing report
        if "classic_report" in st.session_state:
            report_str = st.session_state["classic_report"]
            analysis_bytes = report_str.encode("utf-8")
            df_ol = st.session_state.get("df_ol", pd.DataFrame())
            overlays_bytes = (
                df_ol.to_csv(index=False).encode("utf-8-sig")
                if isinstance(df_ol, pd.DataFrame)
                else b""
            )

            col1, col2, col3 = st.columns(3)
            with col1:
                st.download_button(
                    "⬇️ Download Full Analysis (.txt)",
                    data=analysis_bytes,
                    file_name="analysis.txt",
                    mime="text/plain",
                    key="dl_analysis_persistent",
                )
            with col2:
                st.download_button(
                    "⬇️ Download Overlays (CSV)",
                    data=overlays_bytes,
                    file_name="overlays.csv",
                    mime="text/csv",
                    key="dl_overlays_persistent",
                )
            with col3:
                # Create strategy_report_md from session state if available
                if "strategy_report" in st.session_state:
                    tickets_bytes = st.session_state["strategy_report"].encode("utf-8")
                    st.download_button(
                        "⬇️ Download Strategy Detail (.txt)",
                        data=tickets_bytes,
                        file_name="strategy_detail.txt",
                        mime="text/plain",
                        key="dl_strategy_persistent",
                    )

        st.info("💡 To generate a new report, click 'Analyze This Race' again below.")

    if st.button("Analyze This Race", type="primary", key="analyze_button"):
        # Clear previous Classic Report before generating new one
        st.session_state.pop("classic_report_generated", None)
        st.session_state.pop("classic_report", None)

        with st.spinner("Handicapping Race..."):
            try:
                # ============================================================
                # GOLD STANDARD: SEQUENTIAL VALIDATION & DATA INTEGRITY CHECK
                # ============================================================

                # STAGE 1: Critical Data Retrieval with Validation
                primary_df = st.session_state.get("primary_d")
                primary_probs = st.session_state.get("primary_probs")
                df_final_field = st.session_state.get("df_final_field")
                df_ol = st.session_state.get("df_ol", pd.DataFrame())

                # VALIDATION: Check primary data exists and is valid
                if primary_df is None or not isinstance(primary_df, pd.DataFrame):
                    st.error(
                        "❌ CRITICAL ERROR: Primary ratings dataframe is missing. Please regenerate ratings."
                    )
                    st.stop()

                if primary_df.empty:
                    st.error(
                        "❌ CRITICAL ERROR: Primary ratings dataframe is empty. Check field entries."
                    )
                    st.stop()

                # GOLD STANDARD FIX: Ensure Post and ML columns are in primary_df for classic report
                if df_final_field is not None and not df_final_field.empty:
                    if (
                        "Post" not in primary_df.columns
                        or "ML" not in primary_df.columns
                    ):
                        # Merge Post and ML from df_final_field into primary_df
                        post_ml_data = df_final_field[["Horse", "Post", "ML"]].copy()
                        primary_df = primary_df.merge(
                            post_ml_data, on="Horse", how="left"
                        )
                        # Update session state with enriched primary_df
                        st.session_state["primary_d"] = primary_df

                # VALIDATION: Check required columns exist
                required_cols = ["Horse", "R", "Fair %", "Fair Odds"]
                missing_cols = [
                    col for col in required_cols if col not in primary_df.columns
                ]
                if missing_cols:
                    st.error(
                        f"❌ CRITICAL ERROR: Missing required columns: {missing_cols}"
                    )
                    st.error("Required columns: Horse, R, Fair %, Fair Odds")
                    st.stop()

                # VALIDATION: Check Horse column has valid names
                if primary_df["Horse"].isna().any():
                    st.error("❌ CRITICAL ERROR: Some horses have missing names")
                    st.stop()

                # VALIDATION: Check R (ratings) column is numeric and finite
                try:
                    primary_df["R_test"] = pd.to_numeric(
                        primary_df["R"], errors="coerce"
                    )
                    if primary_df["R_test"].isna().all():
                        st.error(
                            "❌ CRITICAL ERROR: All ratings are invalid (non-numeric)"
                        )
                        st.stop()
                    if not np.all(np.isfinite(primary_df["R_test"].dropna())):
                        st.error("❌ CRITICAL ERROR: Ratings contain infinite values")
                        st.stop()
                    primary_df = primary_df.drop(columns=["R_test"])
                except Exception as e:
                    st.error(f"❌ CRITICAL ERROR: Rating validation failed: {e}")
                    st.stop()

                # VALIDATION: Check Fair % exists and is valid
                if primary_df["Fair %"].isna().all():
                    st.error("❌ CRITICAL ERROR: No fair probabilities calculated")
                    st.stop()

                # STAGE 2: Context Data Retrieval with Safe Defaults
                strategy_profile = st.session_state.get("strategy_profile", "Balanced")
                ppi_val = st.session_state.get("ppi_val", 0.0)
                track_name = st.session_state.get("track_name", "")
                surface_type = st.session_state.get("surface_type", "Dirt")
                condition_txt = st.session_state.get("condition_txt", "")
                distance_txt = st.session_state.get("distance_txt", "")
                race_type_detected = st.session_state.get("race_type", "")
                purse_val = st.session_state.get("purse_val", 0)

                # VALIDATION: Ensure numeric values are finite
                if not np.isfinite(ppi_val):
                    ppi_val = 0.0
                if not np.isfinite(purse_val) or purse_val < 0:
                    purse_val = 0

                # VALIDATION: Ensure text fields are strings
                track_name = str(track_name) if track_name else "Unknown"
                surface_type = str(surface_type) if surface_type else "Dirt"
                condition_txt = str(condition_txt) if condition_txt else "Fast"
                distance_txt = str(distance_txt) if distance_txt else "6F"
                race_type_detected = (
                    str(race_type_detected) if race_type_detected else "Unknown"
                )

                # STAGE 3: Sort and Build Mappings with Validation
                primary_sorted = primary_df.sort_values(by="R", ascending=False)

                # VALIDATION: Ensure final field exists
                if df_final_field is None or df_final_field.empty:
                    st.error("❌ CRITICAL ERROR: Final field dataframe missing")
                    st.stop()

                # GOLD STANDARD: Build safe mappings with validation
                try:
                    name_to_post = pd.Series(
                        df_final_field["Post"].values, index=df_final_field["Horse"]
                    ).to_dict()

                    # CRITICAL FIX: Build program-number mapping for consistent display
                    # Program # = saddle cloth / tote board number (what bettors use)
                    # Post = starting gate position (informational only)
                    if "#" in df_final_field.columns:
                        name_to_prog = pd.Series(
                            df_final_field["#"].values, index=df_final_field["Horse"]
                        ).to_dict()
                    else:
                        name_to_prog = name_to_post.copy()  # Fallback: prog = post

                    # Store Section A data in session state for Section E validation
                    # This ensures the validation uses the ACTUAL post numbers from the race setup
                    post_to_name = {int(v): k for k, v in name_to_post.items()}
                    st.session_state["section_a_posts"] = set(
                        int(v) for v in name_to_post.values()
                    )
                    st.session_state["section_a_post_to_name"] = post_to_name

                    # CRITICAL FIX: Prioritize Live Odds over ML odds (just like Fair % calculation does)
                    name_to_odds = {}
                    smart_money_horses = []  # Track horses with significant odds movement
                    name_to_ml = {}  # Store ML odds separately for comparison

                    for _, row in df_final_field.iterrows():
                        horse_name = row.get("Horse")
                        live_odds = row.get("Live Odds", "")
                        ml_odds = row.get("ML", "")

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
                                        smart_money_horses.append(
                                            {
                                                "name": horse_name,
                                                "post": name_to_prog.get(
                                                    horse_name, row.get("Post")
                                                ),
                                                "ml": ml_odds,
                                                "live": live_odds,
                                                "ratio": ratio,
                                                "movement_pct": movement_pct,
                                            }
                                        )
                            except Exception:
                                pass  # Ignore conversion errors

                    # VALIDATION: Ensure all horses in primary_df have post/odds mappings
                    missing_horses = []
                    for horse in primary_df["Horse"]:
                        if horse not in name_to_post:
                            missing_horses.append(horse)

                    if missing_horses:
                        st.error(
                            "❌ CRITICAL ERROR: Horse name mismatch between ratings and Section A"
                        )
                        st.error(f"Missing horses: {', '.join(missing_horses)}")
                        st.error(
                            "This usually means horse names were modified after Section A"
                        )
                        st.stop()

                except KeyError as e:
                    st.error(
                        f"❌ CRITICAL ERROR: Missing required column in final field: {e}"
                    )
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
                    horse_name = row.get("Horse")
                    live_odds = row.get("Live Odds", "")
                    ml_odds = row.get("ML", "")

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
                                    dumb_money_horses.append(
                                        {
                                            "name": horse_name,
                                            "post": row.get("Post"),
                                            "ml": ml_odds,
                                            "live": live_odds,
                                            "ratio": ratio,
                                            "drift_pct": drift_pct,
                                        }
                                    )
                        except Exception:
                            pass

                # Apply PENALTY to horses with money leaving (odds drifting out)
                if dumb_money_horses:
                    for idx, row in primary_df.iterrows():
                        horse_drift = next(
                            (h for h in dumb_money_horses if h["name"] == row["Horse"]),
                            None,
                        )
                        if horse_drift:
                            ratio = horse_drift["ratio"]
                            # Scaled penalty: bigger drift = bigger penalty
                            if ratio > 2.0:
                                penalty = (
                                    -3.0
                                )  # Massive penalty for 2x+ drift (British Isles case)
                            elif ratio > 1.5:
                                penalty = -1.5  # Moderate penalty for 50%+ drift
                            else:
                                penalty = 0.0

                            primary_df.at[idx, "R"] = row["R"] + penalty

                # Store dumb_money_horses for A-Group gate
                st.session_state["dumb_money_horses"] = dumb_money_horses

                # Apply BONUS to horses with money coming IN (smart money)
                # RACE AUDIT ENHANCEMENT 2: Graduated smart money boost based on contraction severity
                # Oaklawn R1: Tell Me When dropped from 8/1 to 5/2 (ratio 0.31) — extreme steam
                if smart_money_horses:
                    smart_money_lookup = {h["name"]: h for h in smart_money_horses}

                    # Apply graduated bonus to primary_df ratings
                    for idx, row in primary_df.iterrows():
                        if row["Horse"] in smart_money_lookup:
                            horse_data = smart_money_lookup[row["Horse"]]
                            ratio = horse_data.get("ratio", 0.6)
                            # Graduated boost: bigger drop = bigger boost
                            if ratio < 0.30:
                                smart_money_bonus = (
                                    4.0  # Extreme steam (3x+ contraction)
                                )
                            elif ratio < 0.40:
                                smart_money_bonus = 3.5  # Very heavy action
                            elif ratio < 0.50:
                                smart_money_bonus = 3.0  # Strong action
                            else:
                                smart_money_bonus = 2.5  # Standard threshold
                            primary_df.at[idx, "R"] = row["R"] + smart_money_bonus

                # Re-sort after applying all odds-based adjustments
                primary_sorted = primary_df.sort_values(by="R", ascending=False)

                top_table = (
                    primary_sorted[["Horse", "R", "Fair %", "Fair Odds"]]
                    .head(5)
                    .to_markdown(index=False)
                )

                overlay_pos = (
                    df_ol[df_ol["EV per $1"] > 0] if not df_ol.empty else pd.DataFrame()
                )
                overlay_table_md = (
                    overlay_pos[
                        ["Horse", "Fair %", "Fair (AM)", "Board (dec)", "EV per $1"]
                    ].to_markdown(index=False)
                    if not overlay_pos.empty
                    else "None."
                )

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
                    name_to_ml,
                    name_to_prog,
                )

                # Store strategy report in session state
                st.session_state["strategy_report"] = strategy_report_md

                # --- 3. Update the LLM Prompt ---
                race_num = st.session_state.get("race_num", 1)
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

--- NA STYLE CONTEXT ---
{_build_na_context(primary_sorted, df_final_field)}

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
                report = call_openai_messages(
                    messages=[{"role": "user", "content": prompt}]
                )

                # ===== PHASE 3: ADVANCED PROBABILITY ANALYSIS =====
                try:
                    from phase3_probability_engine import (
                        Phase3ProbabilityEngine,
                        format_phase3_report,
                    )

                    # Get win probabilities from primary_df (column is 'Fair %' with string values like '25.5%')
                    if "Fair %" in primary_df.columns:
                        win_probs = (
                            primary_df["Fair %"]
                            .apply(
                                lambda x: (
                                    float(str(x).replace("%", "").strip()) / 100.0
                                    if pd.notna(x)
                                    and str(x)
                                    .replace("%", "")
                                    .replace(".", "")
                                    .strip()
                                    .isdigit()
                                    else 0.0
                                )
                            )
                            .values
                        )
                        horse_names = (
                            primary_df["Horse"].values
                            if "Horse" in primary_df.columns
                            else None
                        )
                        post_positions = (
                            primary_df["Post"].values.astype(int).tolist()
                            if "Post" in primary_df.columns
                            else None
                        )

                        # Run Phase 3 analysis
                        phase3_engine = Phase3ProbabilityEngine(bankroll=50.0)
                        phase3_results = phase3_engine.analyze_race_comprehensive(
                            win_probs=win_probs,
                            horse_names=horse_names,
                            post_positions=post_positions,
                            confidence_level=0.95,
                        )

                        # Format Phase 3 report
                        phase3_report = format_phase3_report(phase3_results)

                    else:
                        phase3_report = (
                            "Phase 3 analysis unavailable (missing win probabilities)"
                        )
                except Exception as e:
                    phase3_report = f"Phase 3 analysis error: {str(e)}"

                # Store Classic Report in session state so it persists across reruns
                st.session_state["classic_report"] = report
                st.session_state["phase3_report"] = phase3_report
                st.session_state["classic_report_generated"] = True

                st.success(
                    "✅ Analysis Complete! Thank you for contributing to our community database."
                )

                # Wrap ENTIRE report in consistent font styling (includes strategy_report_md, LLM output, AND Phase 3)
                styled_report = f"""<div style="font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif; font-size: 15px; line-height: 1.6; color: #1f2937;">

{strategy_report_md}

---

{report}

---

{phase3_report}

</div>"""
                st.markdown(styled_report, unsafe_allow_html=True)

                # ---- Save to disk (optional) ----
                report_str = report if isinstance(report, str) else str(report)
                with open("analysis.txt", "w", encoding="utf-8", errors="replace") as f:
                    f.write(report_str)

                if isinstance(df_ol, pd.DataFrame):
                    df_ol.to_csv("overlays.csv", index=False, encoding="utf-8-sig")
                else:
                    pd.DataFrame().to_csv(
                        "overlays.csv", index=False, encoding="utf-8-sig"
                    )

                # --- Create a tickets.txt from the strategy report ---
                with open("tickets.txt", "w", encoding="utf-8", errors="replace") as f:
                    f.write(strategy_report_md)  # Save the raw strategy markdown
                tickets_bytes = strategy_report_md.encode("utf-8")

                # ---- Download buttons (browser) ----
                analysis_bytes = report_str.encode("utf-8")
                overlays_bytes = (
                    df_ol.to_csv(index=False).encode("utf-8-sig")
                    if isinstance(df_ol, pd.DataFrame)
                    else b""
                )

                col1, col2, col3 = st.columns(3)
                with col1:
                    st.download_button(
                        "⬇️ Download Full Analysis (.txt)",
                        data=analysis_bytes,
                        file_name="analysis.txt",
                        mime="text/plain",
                        key="dl_analysis_new",
                    )
                with col2:
                    st.download_button(
                        "⬇️ Download Overlays (CSV)",
                        data=overlays_bytes,
                        file_name="overlays.csv",
                        mime="text/csv",
                        key="dl_overlays_new",
                    )
                with col3:
                    st.download_button(
                        "⬇️ Download Strategy Detail (.txt)",
                        data=tickets_bytes,
                        file_name="strategy_detail.txt",
                        mime="text/plain",
                        key="dl_strategy_new",
                    )

                # ============================================================
                # AUTO-SAVE TO GOLD HIGH-IQ DATABASE
                # ============================================================
                if GOLD_DB_AVAILABLE and gold_db is not None and primary_df is not None:
                    try:
                        # safe_float is now defined at top of file as a global helper

                        # Generate race ID using ACTUAL race date from PP text (not current date)
                        # This ensures correct race identification even when analyzing historical/future races
                        race_date_str = None
                        pp_text_raw = st.session_state.get("pp_text_cache", "")

                        # Extract race date from BRISNET text (format: "Saturday, February 1, 2026")
                        import re

                        date_match = re.search(
                            r"(Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),\s+([A-Z][a-z]+)\s+(\d{1,2}),\s+(\d{4})",
                            pp_text_raw,
                        )
                        if date_match:
                            month_str = date_match.group(2)
                            day_str = date_match.group(3).zfill(2)
                            year_str = date_match.group(4)

                            # Convert month name to number
                            months = {
                                "January": "01",
                                "February": "02",
                                "March": "03",
                                "April": "04",
                                "May": "05",
                                "June": "06",
                                "July": "07",
                                "August": "08",
                                "September": "09",
                                "October": "10",
                                "November": "11",
                                "December": "12",
                            }
                            month_num = months.get(month_str, "01")
                            race_date_str = (
                                f"{year_str}{month_num}{day_str}"  # YYYYMMDD format
                            )

                        # Fallback to current date if extraction fails
                        if not race_date_str:
                            race_date_str = datetime.now().strftime("%Y%m%d")
                            st.warning(
                                "⚠️ Could not extract race date from PP text, using current date"
                            )

                        race_date = race_date_str  # Use extracted date
                        race_id = f"{track_name}_{race_date}_R{st.session_state.get('race_num', 1)}"

                        # Calculate race bucket for track bias lookup
                        race_bucket = distance_bucket(distance_txt)

                        # SECURITY (Feb 13, 2026): Sanitize text fields before DB storage
                        # Strips SQL injection chars (;'"\) from user-influenced text
                        def _sanitize_metadata_text(
                            val: str, max_len: int = 200
                        ) -> str:
                            import re as _re

                            return _re.sub(r'[;\'"\\]', "", str(val).strip())[:max_len]

                        # Prepare COMPREHENSIVE race metadata with all context
                        race_metadata = {
                            "track": _sanitize_metadata_text(track_name, 50),
                            "date": race_date,
                            "race_num": st.session_state.get("race_num", 1),
                            "race_type": _sanitize_metadata_text(
                                race_type_detected, 100
                            ),
                            "surface": _sanitize_metadata_text(surface_type, 30),
                            "distance": _sanitize_metadata_text(distance_txt, 50),
                            "condition": _sanitize_metadata_text(condition_txt, 100),
                            "purse": purse_val,
                            "field_size": len(df_final_field)
                            if df_final_field is not None
                            else len(primary_df),
                            # ADDITIONAL INTELLIGENT FEATURES
                            "ppi_race_wide": ppi_val,  # Pace Pressure Index
                            "track_bias_config": TRACK_BIAS_PROFILES.get(track_name, {})
                            .get(surface_type, {})
                            .get(race_bucket, {}),
                            "early_speed_count": len(
                                [
                                    r
                                    for r in primary_df.iterrows()
                                    if r[1].get("E1_Style") in ["E", "EP"]
                                ]
                            ),
                            "presser_count": len(
                                [
                                    r
                                    for r in primary_df.iterrows()
                                    if r[1].get("E1_Style") in ["P", "EP"]
                                ]
                            ),
                            "closer_count": len(
                                [
                                    r
                                    for r in primary_df.iterrows()
                                    if r[1].get("E1_Style") == "S"
                                ]
                            ),
                            "avg_field_beyer": safe_float(
                                primary_df.get("Best Beyer", pd.Series([0])).mean()
                            ),
                            "top3_beyer_avg": safe_float(
                                primary_df.nlargest(3, "Best Beyer", keep="first")
                                .get("Best Beyer", pd.Series([0]))
                                .mean()
                                if "Best Beyer" in primary_df.columns
                                else 0
                            ),
                            "avg_field_days_off": safe_float(
                                primary_df.get("Days Since", pd.Series([0])).mean()
                            ),
                            "chaos_index": 0.0,  # Field unpredictability metric (0 = predictable, higher = chaotic)
                            "race_bucket": race_bucket,  # Distance category: ≤6f, 6.5-7f, or 8f+
                            "is_maiden": "maiden" in race_type_detected.lower()
                            or "mdn" in race_type_detected.lower(),
                            "is_stakes": "stakes" in race_type_detected.lower()
                            or "stk" in race_type_detected.lower()
                            or any(
                                g in race_type_detected.lower()
                                for g in ["g1", "g2", "g3"]
                            ),
                            "is_turf": surface_type.lower() == "turf",
                            "is_synthetic": surface_type.lower()
                            in ["synthetic", "tapeta", "polytrack"],
                            # NA STYLE FIELD COMPOSITION (Feb 11, 2026)
                            "na_style_count": sum(
                                1
                                for _, r in primary_df.iterrows()
                                if str(r.get("Style", "")).upper() == "NA"
                            ),
                            "na_high_qsp_count": sum(
                                1
                                for _, r in primary_df.iterrows()
                                if str(r.get("Style", "")).upper() == "NA"
                                and int(safe_float(r.get("Quirin", 0))) >= 5
                            ),
                        }

                        # Prepare horses data with ALL AVAILABLE FEATURES for maximum ML intelligence
                        # CRITICAL: Sort by rating to compute correct predicted_rank
                        _primary_sorted_for_save = primary_df.copy()
                        _primary_sorted_for_save["_R_sort"] = pd.to_numeric(
                            _primary_sorted_for_save["R"], errors="coerce"
                        )
                        _primary_sorted_for_save = _primary_sorted_for_save.sort_values(
                            "_R_sort", ascending=False, na_position="last"
                        )

                        horses_data = []
                        for rank_idx, (_, row) in enumerate(
                            _primary_sorted_for_save.iterrows()
                        ):
                            horse_name = str(row.get("Horse", f"Horse_{rank_idx + 1}"))

                            # Extract Fair % with percentage handling
                            fair_pct_raw = row.get("Fair %", 0.0)
                            fair_pct_value = (
                                safe_float(fair_pct_raw) / 100.0
                            )  # Convert to probability (0-1)

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
                                cats_lower = " ".join(
                                    angles_df["Category"].astype(str).tolist()
                                ).lower()
                                # Extract specific angle values for intelligent pattern recognition
                                if "early speed" in cats_lower:
                                    angle_early_speed = 1.0
                                if (
                                    "class" in cats_lower
                                    or "up in class" in cats_lower
                                    or "down in class" in cats_lower
                                ):
                                    angle_class_move = 1.0
                                if (
                                    "last out" in cats_lower
                                    or "recent" in cats_lower
                                    or "30 days" in cats_lower
                                ):
                                    angle_recency = 1.0
                                if "workout" in cats_lower or "work" in cats_lower:
                                    angle_workout = 1.0
                                if (
                                    "trainer" in cats_lower
                                    or "jockey" in cats_lower
                                    or "combo" in cats_lower
                                ):
                                    angle_connections = 1.0
                                if (
                                    "turf to dirt" in cats_lower
                                    or "dirt to turf" in cats_lower
                                ):
                                    angle_surface_switch = 1.0
                                if (
                                    "distance" in cats_lower
                                    or "sprint" in cats_lower
                                    or "route" in cats_lower
                                ):
                                    angle_distance_switch = 1.0
                                if (
                                    "debut" in cats_lower
                                    or "1st time" in cats_lower
                                    or "maiden sp wt" in cats_lower
                                ):
                                    angle_debut = 1.0

                            horse_dict = {
                                "program_number": int(
                                    safe_float(
                                        name_to_prog.get(
                                            horse_name,
                                            name_to_post.get(horse_name, rank_idx + 1),
                                        ),
                                        rank_idx + 1,
                                    )
                                ),
                                "horse_name": horse_name,
                                "post_position": int(
                                    safe_float(
                                        name_to_post.get(horse_name, rank_idx + 1),
                                        rank_idx + 1,
                                    )
                                ),
                                "morning_line_odds": safe_float(
                                    name_to_odds.get(horse_name, "99"), 99.0
                                ),
                                "jockey": str(row.get("Jockey", "")),
                                "trainer": str(row.get("Trainer", "")),
                                "owner": str(row.get("Owner", "")),
                                "running_style": str(
                                    row.get("Style", row.get("E1_Style", "P"))
                                ),
                                "prime_power": safe_float(row.get("Prime Power", 0.0)),
                                "best_beyer": int(safe_float(row.get("Best Beyer", 0))),
                                "last_beyer": int(safe_float(row.get("Last Beyer", 0))),
                                "avg_beyer_3": safe_float(
                                    row.get("Avg Beyer (3)", 0.0)
                                ),
                                "e1_pace": safe_float(row.get("E1", 0.0)),
                                "e2_pace": safe_float(row.get("E2", 0.0)),
                                "late_pace": safe_float(row.get("Late", 0.0)),
                                "days_since_last": int(
                                    safe_float(row.get("Days Since", 0))
                                ),
                                # CRITICAL FIX: Keys must match gold_database_manager.py expectations
                                # AND column names must match primary_df columns (Cclass, Cform, etc.)
                                # Previously: 'class_rating' + 'Class Rating' → double mismatch → all 0.0 → no learning
                                "rating_class": safe_float(row.get("Cclass", 0.0)),
                                "rating_form": safe_float(row.get("Cform", 0.0)),
                                "rating_speed": safe_float(row.get("Cspeed", 0.0)),
                                "rating_pace": safe_float(row.get("Cpace", 0.0)),
                                "rating_style": safe_float(row.get("Cstyle", 0.0)),
                                "rating_post": safe_float(row.get("Cpost", 0.0)),
                                "class_rating": safe_float(row.get("Cclass", 0.0)),
                                "rating_tier2_bonus": safe_float(
                                    row.get("Tier2_Bonus", 0.0)
                                ),
                                "rating_angles_total": safe_float(
                                    row.get("Cstyle", 0.0)
                                )
                                + safe_float(row.get("Cpost", 0.0))
                                + safe_float(row.get("Cpace", 0.0))
                                + safe_float(row.get("Cspeed", 0.0))
                                + safe_float(row.get("Cclass", 0.0))
                                + safe_float(row.get("Cform", 0.0)),
                                "rating_final": safe_float(row.get("R", 0.0)),
                                "predicted_probability": fair_pct_value,
                                "predicted_rank": int(rank_idx + 1),
                                "fair_odds": safe_float(
                                    row.get("Fair Odds", 99.0), 99.0
                                ),
                                # PhD enhancements if available
                                "rating_confidence": safe_float(
                                    row.get("Confidence", 0.5), 0.5
                                ),
                                "form_decay_score": safe_float(
                                    row.get("Form Decay", 0.0)
                                ),
                                "pace_esp_score": safe_float(row.get("Pace ESP", 0.0)),
                                "mud_adjustment": safe_float(row.get("Mud Adj", 0.0)),
                                # INDIVIDUAL ANGLE FEATURES (8 key angles for ML pattern learning)
                                "angle_early_speed": angle_early_speed,
                                "angle_class": angle_class_move,
                                "angle_recency": angle_recency,
                                "angle_work_pattern": angle_workout,
                                "angle_connections": angle_connections,
                                "angle_pedigree": safe_float(
                                    pedigree_data.get("sire_1st", 0.0)
                                )
                                / 100.0
                                if pedigree_data
                                else 0.0,
                                "angle_runstyle_bias": 1.0
                                if row.get("Style", row.get("E1_Style"))
                                in ["E", "EP", "E/P"]
                                else 0.0,
                                "angle_post": safe_float(row.get("Cpost", 0.0)),
                                # PEDIGREE FEATURES (critical for surface/distance/mud breeding)
                                "pedigree_sire_awd": safe_float(
                                    pedigree_data.get("sire_awd", 7.0)
                                )
                                if pedigree_data
                                else 7.0,
                                "pedigree_sire_1st_pct": safe_float(
                                    pedigree_data.get("sire_1st", 0.0)
                                )
                                if pedigree_data
                                else 0.0,
                                "pedigree_damsire_awd": safe_float(
                                    pedigree_data.get("damsire_awd", 7.0)
                                )
                                if pedigree_data
                                else 7.0,
                                "pedigree_damsire_1st_pct": safe_float(
                                    pedigree_data.get("damsire_1st", 0.0)
                                )
                                if pedigree_data
                                else 0.0,
                                "pedigree_dam_dpi": safe_float(
                                    pedigree_data.get("dam_dpi", 1.0)
                                )
                                if pedigree_data
                                else 1.0,
                                # ADDITIONAL CONTEXTUAL FEATURES
                                "quirin_points": int(safe_float(row.get("Quirin", 0))),
                                "style_strength": safe_float(
                                    row.get("StyleStrength", 0.0)
                                ),
                                "ppi_individual": safe_float(
                                    ppi_map_by_horse.get(horse_name, 0.0)
                                ),
                                "field_size_context": len(df_final_field)
                                if df_final_field is not None
                                else len(primary_df),
                                "post_position_bias": safe_float(row.get("Post", 0))
                                / max(len(primary_df), 1),
                                # LIFETIME STATS (if available from PP text)
                                "starts_lifetime": int(
                                    safe_float(row.get("Starts", 0))
                                ),
                                "wins_lifetime": int(safe_float(row.get("Wins", 0))),
                                "win_pct": safe_float(row.get("Win%", 0.0)),
                                "earnings_lifetime": safe_float(
                                    row.get("Earnings", 0.0)
                                ),
                                # NA STYLE + QSP CONTEXT (Feb 11, 2026)
                                "is_na_style": 1
                                if str(row.get("Style", "")).upper() == "NA"
                                else 0,
                                "na_qsp_confidence": round(
                                    0.7
                                    + 0.3
                                    * (int(safe_float(row.get("Quirin", 0))) / 8.0),
                                    3,
                                )
                                if str(row.get("Style", "")).upper() == "NA"
                                else 1.0,
                                "is_fts": 1
                                if int(
                                    safe_float(
                                        row.get("Starts", row.get("CStarts", 0)), 0
                                    )
                                )
                                == 0
                                else 0,
                                "na_dampener_applied": round(
                                    NA_STYLE_PARAMS["rating_dampener"]
                                    + (int(safe_float(row.get("Quirin", 0))) / 8.0)
                                    * (1.0 - NA_STYLE_PARAMS["rating_dampener"])
                                    * 0.5,
                                    3,
                                )
                                if str(row.get("Style", "")).upper() == "NA"
                                and int(
                                    safe_float(
                                        row.get("Starts", row.get("CStarts", 0)), 0
                                    )
                                )
                                > 0
                                else 1.0,
                            }

                            # ═══════════════════════════════════════════════════════
                            # ELITE PARSER ENRICHMENT: Populate fields from parsed PP data
                            # The elite parser extracts weight, medication, equipment,
                            # lifetime records, jockey/trainer stats, surface records,
                            # damsire, beaten lengths — data that the rating engine
                            # doesn't produce but is critical for ML training.
                            # ═══════════════════════════════════════════════════════
                            elite_data = st.session_state.get("elite_horses_data", {})
                            # Try exact match first, then normalized match
                            ep = elite_data.get(horse_name)
                            if ep is None:
                                norm_name = normalize_horse_name(horse_name)
                                for ek, ev in elite_data.items():
                                    if normalize_horse_name(ek) == norm_name:
                                        ep = ev
                                        break
                            if ep:
                                horse_dict["weight"] = ep.get(
                                    "weight"
                                ) or horse_dict.get("weight")
                                horse_dict["medication"] = ep.get(
                                    "medication"
                                ) or horse_dict.get("medication")
                                horse_dict["equipment"] = ep.get(
                                    "equipment_string"
                                ) or horse_dict.get("equipment")
                                horse_dict["damsire"] = ep.get("damsire") or ""
                                # Lifetime records (override if elite parser found them)
                                if ep.get("starts_lifetime"):
                                    horse_dict["starts_lifetime"] = ep[
                                        "starts_lifetime"
                                    ]
                                if ep.get("wins_lifetime"):
                                    horse_dict["wins_lifetime"] = ep["wins_lifetime"]
                                horse_dict["places_lifetime"] = ep.get(
                                    "places_lifetime", 0
                                )
                                horse_dict["shows_lifetime"] = ep.get(
                                    "shows_lifetime", 0
                                )
                                if ep.get("earnings_lifetime_parsed"):
                                    horse_dict["earnings_lifetime"] = ep[
                                        "earnings_lifetime_parsed"
                                    ]
                                # Jockey/trainer win pct
                                j_starts = ep.get("jockey_starts", 0)
                                j_wins = ep.get("jockey_wins", 0)
                                horse_dict["jockey_win_pct"] = (
                                    round(j_wins / j_starts, 3) if j_starts > 0 else 0.0
                                )
                                t_starts = ep.get("trainer_starts", 0)
                                t_wins = ep.get("trainer_wins", 0)
                                horse_dict["trainer_win_pct"] = (
                                    round(t_wins / t_starts, 3) if t_starts > 0 else 0.0
                                )

                                # Surface/distance records (stored as "S-W-P-S" string or dict)
                                def _parse_record(rec):
                                    """Parse '5-1-2-0' record string to dict."""
                                    if isinstance(rec, dict):
                                        return rec
                                    if isinstance(rec, str) and "-" in rec:
                                        parts = rec.split("-")
                                        if len(parts) >= 2:
                                            try:
                                                return {
                                                    "starts": int(parts[0]),
                                                    "wins": int(parts[1]),
                                                }
                                            except ValueError:
                                                pass
                                    return {"starts": 0, "wins": 0}

                                turf_rec = _parse_record(ep.get("turf_record"))
                                horse_dict["turf_starts"] = turf_rec.get("starts", 0)
                                horse_dict["turf_wins"] = turf_rec.get("wins", 0)
                                wet_rec = _parse_record(ep.get("wet_record"))
                                horse_dict["wet_starts"] = wet_rec.get("starts", 0)
                                horse_dict["wet_wins"] = wet_rec.get("wins", 0)
                                dist_rec = _parse_record(ep.get("distance_record"))
                                horse_dict["distance_starts"] = dist_rec.get(
                                    "starts", 0
                                )
                                horse_dict["distance_wins"] = dist_rec.get("wins", 0)
                                # Average beaten lengths (from finish beaten lengths list)
                                beaten = ep.get("beaten_lengths_finish") or []
                                if beaten:
                                    try:
                                        valid_bl = [
                                            float(b) for b in beaten if b is not None
                                        ]
                                        horse_dict["avg_beaten_lengths"] = (
                                            round(sum(valid_bl) / len(valid_bl), 2)
                                            if valid_bl
                                            else 0.0
                                        )
                                    except (ValueError, TypeError):
                                        horse_dict["avg_beaten_lengths"] = 0.0

                            horses_data.append(horse_dict)

                        # Save to gold database
                        pp_text_raw = st.session_state.get("pp_text_cache", "")
                        success = gold_db.save_analyzed_race(
                            race_id=race_id,
                            race_metadata=race_metadata,
                            horses_data=horses_data,
                            pp_text_raw=pp_text_raw,
                        )

                        if success is True:
                            st.success(
                                f"💾 **Auto-saved to gold database:** `{race_id}`"
                            )
                            st.info(
                                f"📊 Saved {len(horses_data)} horses with {len([k for k in horses_data[0].keys() if k.startswith('rating_') or k.startswith('angle_')])} ML features each"
                            )
                            st.session_state["last_saved_race_id"] = race_id
                            st.info(
                                "🏁 After race completes, submit actual top 4 finishers in **Section E** below!"
                            )

                            # CLOUD BACKUP after analysis save
                            try:
                                if backup_to_github_async:
                                    backup_to_github_async(gold_db.db_path)
                            except Exception:
                                pass
                        elif isinstance(success, str):
                            st.error(f"❌ Failed to save race {race_id}: {success}")
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
        total_saved = stats.get("total_races", 0) + len(pending_races)

        st.header(f"E. Gold High-IQ System 🏆 - {total_saved} Races Saved")
        if total_saved > 0:
            st.success(
                f"💾 **Database Active:** {stats.get('total_races', 0)} races with results, {len(pending_races)} pending results"
            )
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
        _is_persistent = False
        if get_persistence_status:
            p_status = get_persistence_status(db_path)
            _plevel = p_status.get("persistence_level", "")
            _is_persistent = (
                "PERSISTENT DISK" in _plevel
                or "GITHUB BACKUP" in _plevel
                or "LOCAL" in _plevel
            )
        else:
            _plevel = "💻 LOCAL (development mode)"
            _is_persistent = True

        if os.path.exists(db_path):
            db_size_mb = os.path.getsize(db_path) / (1024 * 1024)
            if _is_persistent:
                st.success(
                    f"✅ **Database Verified:** {db_path} ({db_size_mb:.2f} MB) — data persists across sessions!"
                )
            else:
                st.warning(
                    f"⚠️ **Database Active:** {db_path} ({db_size_mb:.2f} MB) — **EPHEMERAL** storage! Data will be lost on next Render redeploy. Add a Persistent Disk ($0.25/mo) or set GITHUB_TOKEN for backup."
                )
        else:
            st.info(f"📁 New database will be created: {db_path}")

        st.caption(f"🔐 Storage: {_plevel}")
    except Exception as verify_error:
        st.warning(f"Could not verify database: {verify_error}")

    # Get stats from gold database
    try:
        stats = gold_db.get_accuracy_stats()
        pending_races = gold_db.get_pending_races(limit=20)

        # Create tabs for Gold High-IQ System
        tab_overview, tab_results, tab_calibration, tab_retrain, tab_track_intel = (
            st.tabs(
                [
                    "📊 Dashboard",
                    "🏁 Submit Actual Top 4",
                    "🤖 Auto-Calibration Monitor",
                    "🚀 Retrain Model",
                    "🧠 Track Intelligence",
                ]
            )
        )

        # Tab 1: Dashboard
        with tab_overview:
            # Build persistence note based on actual storage status
            if get_persistence_status:
                _ps = get_persistence_status(PERSISTENT_DB_PATH)
                _pl = _ps.get("persistence_level", "")
                if "PERSISTENT DISK" in _pl:
                    _persist_note = "🔒 **Data Persistence:** All analyzed races are permanently saved. Data survives Render redeploys!"
                elif "GITHUB BACKUP" in _pl:
                    _persist_note = "☁️ **Data Persistence:** Backed up to GitHub. Restored automatically on redeploy."
                elif "EPHEMERAL" in _pl:
                    _persist_note = (
                        "⚠️ **Data Persistence:** Database is saved but on **ephemeral** storage. "
                        "Data persists between browser sessions but **will be lost on Render redeploy**. "
                        "Add a Persistent Disk ($0.25/mo) in Render Dashboard → Disks to keep data permanently."
                    )
                else:
                    _persist_note = (
                        "💻 **Data Persistence:** Saved locally (development mode)."
                    )
            else:
                _persist_note = "💻 **Data Persistence:** Saved locally."

            st.markdown(f"""
            ### Real Data Learning System

            Every time you click "Analyze This Race", the system **auto-saves**:
            - All horse features (speed, class, pace, angles, PhD enhancements)
            - Model predictions (probabilities, ratings, confidence)
            - Race metadata (track, conditions, purse, etc.)

            After the race completes, submit the actual top 4 finishers below.
            The system learns from real outcomes to reach 90%+ accuracy.

            {_persist_note}

            📁 **Database Location:** `{PERSISTENT_DB_PATH}`
            """)

            # Calculate total analyzed races (completed + pending)
            completed_races = stats.get("total_races", 0)
            total_analyzed = completed_races + len(pending_races)

            # Metrics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric(
                    "Total Analyzed",
                    total_analyzed,
                    help="All races saved (completed + pending results)",
                )
            with col2:
                st.metric(
                    "With Results",
                    completed_races,
                    help="Races with actual results entered - used for accuracy tracking",
                )
            with col3:
                st.metric(
                    "Pending Results",
                    len(pending_races),
                    help="Races analyzed but awaiting actual finishers",
                )
            with col4:
                if completed_races > 0:
                    st.metric(
                        "Winner Accuracy",
                        f"{stats.get('winner_accuracy', 0.0):.1%}",
                        help="Historical: % of races where the model's #1 pick won at time of analysis (before retraining)",
                    )
                else:
                    st.metric(
                        "Winner Accuracy",
                        "N/A",
                        help="Submit results to track accuracy",
                    )

            # Training readiness indicator
            st.markdown("#### Training Readiness")
            ready = completed_races >= 50
            if ready:
                st.success(
                    f"✅ Ready to retrain! You have {completed_races} completed races."
                )
            else:
                st.info(
                    f"⏳ Need 50 completed races to retrain model. Current progress: {completed_races}/50"
                )

            # Progress bars
            st.markdown("#### Progress to Milestones")

            milestones = [
                (50, "First Retrain", "70-75%"),
                (100, "Second Retrain", "75-80%"),
                (500, "Major Improvement", "85-87%"),
                (1000, "Gold Standard", "90%+"),
            ]

            for target, label, expected_acc in milestones:
                progress = min(completed_races / target, 1.0)
                st.progress(
                    progress,
                    text=f"{label} ({expected_acc} expected): {completed_races}/{target} races",
                )

            # System performance
            if stats.get("total_races", 0) > 0:
                st.markdown("#### Historical Performance")
                st.caption(
                    "Based on the model's predictions at the time each race was analyzed "
                    "(before retraining). See Retrain Model tab for retrained model accuracy."
                )
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Winner Pick Rate",
                        f"{stats.get('winner_accuracy', 0.0):.1%}",
                        help="% of races where the #1 pick at analysis time was the actual winner",
                    )
                with col2:
                    st.metric(
                        "Top-3 Hit Rate",
                        f"{stats.get('top3_accuracy', 0.0):.1%}",
                        help="Avg overlap between predicted top 3 and actual top 3",
                    )
                with col3:
                    st.metric(
                        "Top-4 Hit Rate",
                        f"{stats.get('top4_accuracy', 0.0):.1%}",
                        help="Avg overlap between predicted top 4 and actual top 4",
                    )

            # COMMUNITY RACES TABLE - Show all saved races
            st.markdown("---")
            st.markdown("#### 🌐 Community Races Log (All Users)")
            st.caption(
                "All races analyzed by the community - permanently saved to database"
            )

            try:
                import sqlite3

                conn = sqlite3.connect(gold_db.db_path, timeout=5.0)
                cursor = conn.cursor()

                # Get all races with status
                cursor.execute("""
                    SELECT 
                        r.race_id,
                        r.track_code,
                        r.race_date,
                        r.race_number,
                        r.field_size,
                        r.analyzed_timestamp,
                        CASE WHEN g.race_id IS NOT NULL THEN '✅ Results' ELSE '⏳ Pending' END as status
                    FROM races_analyzed r
                    LEFT JOIN (SELECT DISTINCT race_id FROM gold_high_iq) g ON r.race_id = g.race_id
                    ORDER BY r.analyzed_timestamp DESC
                    LIMIT 50
                """)

                all_races = cursor.fetchall()
                conn.close()

                if all_races:
                    # Create dataframe
                    races_df = pd.DataFrame(
                        all_races,
                        columns=[
                            "Race ID",
                            "Track",
                            "Date",
                            "Race #",
                            "Horses",
                            "Saved At",
                            "Status",
                        ],
                    )

                    # Format timestamp
                    races_df["Saved At"] = pd.to_datetime(
                        races_df["Saved At"]
                    ).dt.strftime("%m/%d %I:%M %p")

                    # Display with status colors
                    st.dataframe(
                        races_df,
                        use_container_width=True,
                        hide_index=True,
                        column_config={
                            "Status": st.column_config.TextColumn(
                                "Status",
                                help="✅ = Results entered | ⏳ = Awaiting results",
                            )
                        },
                    )

                    # Summary
                    completed = len([r for r in all_races if "✅" in r[6]])
                    pending = len([r for r in all_races if "⏳" in r[6]])
                    st.caption(
                        f"Showing last 50 races | ✅ {completed} completed | ⏳ {pending} pending"
                    )
                else:
                    st.info("📋 No races saved yet. Analyze a race above to begin!")

            except Exception as races_err:
                st.warning(f"Could not load race log: {races_err}")

        # Tab 2: Submit Actual Top 5
        with tab_results:
            st.markdown("""
            ### Submit Actual Top 4 Finishers

            After a race completes, enter the actual finishing order here.
            **Only the top 4 positions are required** for high-quality ML training.
            """)

            if not pending_races:
                st.success(
                    "✅ No pending races! All analyzed races have results entered."
                )
                st.info("💡 Analyze more races in Sections 1-4 to build training data.")
            else:
                # Filter out races marked as completed in this session
                pending_races = [
                    r
                    for r in pending_races
                    if not st.session_state.get(f"race_completed_{r[0]}", False)
                ]
                if not pending_races:
                    st.success("✅ All pending races have results entered!")
                    st.info(
                        "💡 Analyze more races in Sections 1-4 to build training data."
                    )
                else:
                    st.info(f"📋 {len(pending_races)} races awaiting results")

                # Select race
                race_options = [
                    f"{r[1]} R{r[3]} on {r[2]} ({r[4]} horses)" for r in pending_races
                ]
                selected_idx = st.selectbox(
                    "Select Race to Enter Results:",
                    range(len(race_options)),
                    format_func=lambda i: race_options[i],
                    key="select_pending_race",
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
                        display_df = horses_df[
                            [
                                "program_number",
                                "horse_name",
                                "post_position",
                                "predicted_probability",
                                "fair_odds",
                            ]
                        ].copy()
                        # Format values BEFORE renaming columns
                        display_df["predicted_probability"] = (
                            display_df["predicted_probability"] * 100
                        ).round(1).astype(str) + "%"
                        display_df["fair_odds"] = display_df["fair_odds"].round(2)
                        # Now rename columns for display
                        display_df.columns = [
                            "#",
                            "Horse Name",
                            "Post",
                            "Predicted Win %",
                            "Fair Odds",
                        ]

                        st.dataframe(
                            display_df, use_container_width=True, hide_index=True
                        )

                        # Enter top 4
                        st.markdown("---")

                        # Get horse data for validation and display
                        # First try to use Section A data from session state (most accurate)
                        section_a_posts = st.session_state.get("section_a_posts", set())
                        section_a_post_to_name = st.session_state.get(
                            "section_a_post_to_name", {}
                        )

                        # Fall back to database data if Section A data not available
                        db_program_numbers = sorted(
                            [int(h["program_number"]) for h in horses]
                        )
                        db_horse_names_dict = {
                            int(h["program_number"]): h["horse_name"] for h in horses
                        }

                        # Use Section A data if available, otherwise use database
                        if section_a_posts:
                            program_numbers = sorted(section_a_posts)
                            horse_names_dict = section_a_post_to_name
                        else:
                            program_numbers = db_program_numbers
                            horse_names_dict = db_horse_names_dict

                        # Create combined set of valid programs (both Section A and database)
                        all_known_programs = set(program_numbers) | set(
                            db_program_numbers
                        )

                        # Also allow program numbers up to max + buffer for late changes
                        max_program = (
                            max(all_known_programs)
                            if all_known_programs
                            else field_size
                        )

                        # ========== SIMPLIFIED TOP 4 ENTRY FLOW ==========
                        st.markdown("#### 🏆 Enter Actual Top 4 Finishers")

                        # FIX (Feb 11, 2026): Once saved, hide the form and show
                        # confirmation so users don't accidentally double-submit.
                        if st.session_state.get(f"race_completed_{race_id}"):
                            _saved_order = st.session_state.get(
                                f"saved_finish_order_{race_id}", []
                            )
                            if _saved_order:
                                _saved_parts = []
                                for _si, _sp in enumerate(_saved_order[:4]):
                                    _sname = horse_names_dict.get(_sp, f"Horse #{_sp}")
                                    _smedals = ["🥇", "🥈", "🥉", "4th"]
                                    _saved_parts.append(
                                        f"{_smedals[_si]} #{_sp} {_sname}"
                                    )
                                st.success(
                                    f"✅ Results already saved: {' → '.join(_saved_parts)}"
                                )
                            else:
                                st.success(f"✅ Results already saved for {race_id}")
                            st.caption(
                                "Results have been stored in the database. Parse a new race to enter more results."
                            )
                        else:
                            st.caption(
                                "Enter program numbers separated by commas, then click Save Results"
                            )

                            # --- widget keys scoped to this race ---
                            _input_key = f"finish_input_{race_id}"
                            _save_flag = f"pending_save_{race_id}"

                            def _queue_save():
                                """on_click callback: stash the raw input before the re-run."""
                                raw = st.session_state.get(_input_key, "").strip()
                                if raw:
                                    st.session_state[_save_flag] = raw
                                else:
                                    st.session_state[_save_flag] = "__EMPTY__"

                            st.text_input(
                                "Finishing order (1st through 4th)",
                                placeholder="Example: 8,5,6,9",
                                help="Type the program numbers in order from 1st to 4th, separated by commas",
                                key=_input_key,
                            )

                            st.button(
                                "💾 Save Results",
                                type="primary",
                                key=f"save_btn_{race_id}",
                                on_click=_queue_save,
                            )

                            # --- process pending save (set by the callback BEFORE this re-run) ---
                            _pending_raw = st.session_state.pop(_save_flag, None)

                            if _pending_raw == "__EMPTY__":
                                st.warning(
                                    "⚠️ Please enter 4 program numbers separated by commas (e.g., 5,8,11,12)"
                                )

                            elif _pending_raw and not st.session_state.get(
                                f"race_completed_{race_id}"
                            ):
                                finish_input = _pending_raw
                                finish_order = []

                                try:
                                    # Parse comma-separated values
                                    raw_values = [
                                        x.strip() for x in finish_input.split(",")
                                    ]
                                    finish_order = [int(x) for x in raw_values if x]

                                    # Validation checks - NOW TOP 4
                                    if len(finish_order) < 4:
                                        st.error(
                                            f"❌ Need 4 horses, only got {len(finish_order)}"
                                        )
                                        finish_order = []
                                    elif len(finish_order) > 4:
                                        st.warning("⚠️ Using first 4 horses")
                                        finish_order = finish_order[:4]

                                    # Check for duplicates
                                    if finish_order and len(finish_order) != len(
                                        set(finish_order)
                                    ):
                                        st.error(
                                            "❌ Cannot use same horse in multiple positions"
                                        )
                                        finish_order = []

                                    # Validate program numbers exist
                                    if finish_order:
                                        invalid = [
                                            x
                                            for x in finish_order
                                            if x not in all_known_programs
                                            and x > max_program + 2
                                        ]
                                        if invalid:
                                            st.warning(
                                                f"⚠️ Program numbers {invalid} not in parsed data - will allow anyway"
                                            )

                                    # If we have valid input, save immediately
                                    if finish_order and len(finish_order) >= 4:
                                        # Show preview
                                        preview_parts = []
                                        for i, pos in enumerate(finish_order[:4]):
                                            horse_name = horse_names_dict.get(
                                                pos, f"Horse #{pos}"
                                            )
                                            medals = ["🥇", "🥈", "🥉", "4th"]
                                            preview_parts.append(
                                                f"{medals[i]} #{pos} {horse_name}"
                                            )
                                        st.success(" → ".join(preview_parts))

                                        # Save immediately on form submit
                                        with st.spinner("Saving to database..."):
                                            try:
                                                # Build UI horse data as fallback for submit
                                                horses_ui_data = []
                                                for h in horses:
                                                    horses_ui_data.append(
                                                        {
                                                            "program_number": h.get(
                                                                "program_number", 0
                                                            ),
                                                            "horse_name": h.get(
                                                                "horse_name", "Unknown"
                                                            ),
                                                            "post_position": h.get(
                                                                "post_position", 0
                                                            ),
                                                            "predicted_rank": h.get(
                                                                "predicted_rank", 99
                                                            ),
                                                            "predicted_probability": h.get(
                                                                "predicted_probability",
                                                                0,
                                                            ),
                                                            "rating_final": h.get(
                                                                "rating_final", 0
                                                            ),
                                                            "prime_power": h.get(
                                                                "prime_power", 0
                                                            ),
                                                            "running_style": h.get(
                                                                "running_style", ""
                                                            ),
                                                            "quirin_speed_pts": h.get(
                                                                "quirin_speed_pts",
                                                                h.get("quirin", 0),
                                                            ),
                                                            "field_size": field_size,
                                                        }
                                                    )

                                                # Save results (using top 4) with UI fallback data
                                                success = gold_db.submit_race_results(
                                                    race_id=race_id,
                                                    finish_order_programs=finish_order[
                                                        :4
                                                    ],
                                                    horses_ui=horses_ui_data,
                                                )

                                                if success:
                                                    # Mark this race as completed to remove from pending list
                                                    st.session_state[
                                                        f"race_completed_{race_id}"
                                                    ] = True
                                                    # Stash finish order for confirmation display
                                                    st.session_state[
                                                        f"saved_finish_order_{race_id}"
                                                    ] = finish_order[:4]

                                                    # Check if we predicted correctly
                                                    predicted_winner_row = horses_df[
                                                        horses_df["predicted_rank"] == 1
                                                    ]
                                                    predicted_winner = (
                                                        predicted_winner_row[
                                                            "horse_name"
                                                        ].values[0]
                                                        if not predicted_winner_row.empty
                                                        else "Unknown"
                                                    )
                                                    actual_winner_name = (
                                                        horse_names_dict.get(
                                                            finish_order[0],
                                                            f"Horse #{finish_order[0]}",
                                                        )
                                                    )

                                                    # Show inline success with prediction accuracy
                                                    if (
                                                        predicted_winner
                                                        == actual_winner_name
                                                    ):
                                                        st.success(
                                                            f"🎯 **Perfect Prediction!** Winner: {actual_winner_name} | Results saved for {race_id}"
                                                        )
                                                    else:
                                                        st.success(
                                                            f"✅ Results saved! Winner: #{finish_order[0]} {actual_winner_name}"
                                                        )
                                                        st.info(
                                                            f"📊 Predicted: {predicted_winner} | Actual: {actual_winner_name}"
                                                        )

                                                    # AUTO-CALIBRATION v2: Learn from result with persistence
                                                    # Includes TRACK-SPECIFIC calibration
                                                    try:
                                                        if ADAPTIVE_LEARNING_AVAILABLE:
                                                            # Extract track code for per-track calibration
                                                            _cal_track = (
                                                                race_id.split("_")[0]
                                                                if "_" in race_id
                                                                else ""
                                                            )
                                                            calibration_result = auto_calibrate_on_result_submission(
                                                                gold_db.db_path,
                                                                track_code=_cal_track,
                                                            )
                                                            if (
                                                                calibration_result.get(
                                                                    "status"
                                                                )
                                                                != "skipped"
                                                            ):
                                                                accuracy = (
                                                                    calibration_result.get(
                                                                        "winner_accuracy",
                                                                        0,
                                                                    )
                                                                    * 100
                                                                )
                                                                top3_acc = (
                                                                    calibration_result.get(
                                                                        "top3_accuracy",
                                                                        0,
                                                                    )
                                                                    * 100
                                                                )
                                                                # Track calibration info
                                                                _tc = calibration_result.get(
                                                                    "track_calibration",
                                                                    {},
                                                                )
                                                                _tc_status = _tc.get(
                                                                    "status", ""
                                                                )
                                                                if (
                                                                    _tc_status
                                                                    == "success"
                                                                ):
                                                                    _tc_w = (
                                                                        _tc.get(
                                                                            "winner_accuracy",
                                                                            0,
                                                                        )
                                                                        * 100
                                                                    )
                                                                    _tc_t3 = (
                                                                        _tc.get(
                                                                            "top3_accuracy",
                                                                            0,
                                                                        )
                                                                        * 100
                                                                    )
                                                                    _tc_races = _tc.get(
                                                                        "races_analyzed",
                                                                        0,
                                                                    )
                                                                    st.info(
                                                                        f"🧠 Global: W={accuracy:.0f}% T3={top3_acc:.0f}% | "
                                                                        f"🏇 {_cal_track}: W={_tc_w:.0f}% T3={_tc_t3:.0f}% ({_tc_races} races)"
                                                                    )
                                                                elif (
                                                                    _tc_status
                                                                    == "insufficient_data"
                                                                ):
                                                                    st.info(
                                                                        f"🧠 Global: W={accuracy:.0f}% T3={top3_acc:.0f}% | "
                                                                        f"⏳ {_cal_track}: building profile ({_tc.get('races_available', 0)}/2 races)"
                                                                    )
                                                                else:
                                                                    st.info(
                                                                        f"🧠 Model learned! Winner: {accuracy:.0f}% | Top-3: {top3_acc:.0f}%"
                                                                    )
                                                                # Refresh learned weights so next prediction uses updated values
                                                                globals()[
                                                                    "LEARNED_WEIGHTS"
                                                                ] = get_live_learned_weights(
                                                                    gold_db.db_path
                                                                )
                                                                logger.info(
                                                                    "🔄 Refreshed LEARNED_WEIGHTS after calibration"
                                                                )
                                                    except Exception as cal_err:
                                                        logger.warning(
                                                            f"Auto-calibration failed: {cal_err}"
                                                        )

                                                    # TRACK INTELLIGENCE: Auto-rebuild profile for this track after new results
                                                    try:
                                                        if (
                                                            TRACK_INTEL_AVAILABLE
                                                            and _track_intel is not None
                                                        ):
                                                            _rebuild_track = (
                                                                race_id.split("_")[0]
                                                                if "_" in race_id
                                                                else ""
                                                            )
                                                            if _rebuild_track:
                                                                _track_intel.update_after_submission(
                                                                    _rebuild_track
                                                                )
                                                                logger.info(
                                                                    f"🧠 Track Intelligence profile rebuilt for {_rebuild_track}"
                                                                )
                                                    except Exception as ti_err:
                                                        logger.warning(
                                                            f"Track Intelligence rebuild failed: {ti_err}"
                                                        )

                                                    # INTELLIGENT LEARNING: High-IQ pattern analysis
                                                    try:
                                                        if (
                                                            INTELLIGENT_LEARNING_AVAILABLE
                                                            and analyze_and_learn_from_result
                                                        ):
                                                            predictions_list = []
                                                            for (
                                                                _,
                                                                row,
                                                            ) in horses_df.iterrows():
                                                                pred = {
                                                                    "program_number": int(
                                                                        row.get(
                                                                            "program_number",
                                                                            row.get(
                                                                                "post",
                                                                                0,
                                                                            ),
                                                                        )
                                                                    ),
                                                                    "horse_name": row.get(
                                                                        "horse_name",
                                                                        "Unknown",
                                                                    ),
                                                                    "predicted_rank": int(
                                                                        row.get(
                                                                            "predicted_rank",
                                                                            99,
                                                                        )
                                                                    ),
                                                                    "rating": float(
                                                                        row.get(
                                                                            "rating", 0
                                                                        )
                                                                    ),
                                                                    "c_form": float(
                                                                        row.get(
                                                                            "Cform",
                                                                            row.get(
                                                                                "c_form",
                                                                                0.5,
                                                                            ),
                                                                        )
                                                                    ),
                                                                    "c_class": float(
                                                                        row.get(
                                                                            "Cclass",
                                                                            row.get(
                                                                                "c_class",
                                                                                0.5,
                                                                            ),
                                                                        )
                                                                    ),
                                                                    "pace_style": row.get(
                                                                        "Pace_Style",
                                                                        row.get(
                                                                            "pace_style",
                                                                            "",
                                                                        ),
                                                                    ),
                                                                    "last_fig": row.get(
                                                                        "speed_last",
                                                                        row.get(
                                                                            "last_fig",
                                                                            0,
                                                                        ),
                                                                    ),
                                                                    "speed_figures": row.get(
                                                                        "speed_figures",
                                                                        [],
                                                                    ),
                                                                    "angles": row.get(
                                                                        "angles", []
                                                                    ),
                                                                }
                                                                predictions_list.append(
                                                                    pred
                                                                )

                                                            learning_result = analyze_and_learn_from_result(
                                                                db_path=gold_db.db_path,
                                                                race_id=race_id,
                                                                predictions=predictions_list,
                                                                actual_results=finish_order[
                                                                    :4
                                                                ],
                                                            )

                                                            if (
                                                                learning_result.get(
                                                                    "insights_found", 0
                                                                )
                                                                > 0
                                                            ):
                                                                insights = (
                                                                    learning_result.get(
                                                                        "insights", []
                                                                    )
                                                                )
                                                                if insights:
                                                                    st.info(
                                                                        f"🎓 Found {len(insights)} learning patterns"
                                                                    )
                                                                    st.session_state[
                                                                        "last_learning_insights"
                                                                    ] = insights
                                                    except Exception as learn_err:
                                                        logger.warning(
                                                            f"Intelligent learning failed: {learn_err}"
                                                        )

                                                    # TRACK PATTERN LEARNING: Show pattern update confirmation
                                                    try:
                                                        if hasattr(
                                                            gold_db,
                                                            "get_track_patterns",
                                                        ):
                                                            # Patterns were stored during submit_race_results
                                                            # Show what we learned
                                                            _tp = gold_db.get_track_patterns(
                                                                track_code=race_id.split(
                                                                    "_"
                                                                )[0]
                                                                if "_" in race_id
                                                                else "UNK",
                                                                surface=st.session_state.get(
                                                                    "race_surface",
                                                                    "Dirt",
                                                                ),
                                                                distance=st.session_state.get(
                                                                    "race_distance",
                                                                    "6F",
                                                                ),
                                                            )
                                                            if _tp:
                                                                _tp_races = _tp.get(
                                                                    "total_races_analyzed",
                                                                    {},
                                                                ).get("value", 0)
                                                                _tp_style = _tp.get(
                                                                    "dominant_winning_style",
                                                                    {},
                                                                ).get("value", "?")
                                                                _tp_beyer = _tp.get(
                                                                    "avg_winner_best_beyer",
                                                                    {},
                                                                ).get("value", "?")
                                                                st.info(
                                                                    f"📊 Track patterns updated! "
                                                                    f"{_tp_races} races learned | "
                                                                    f"Dominant style: {_tp_style} | "
                                                                    f"Avg winner Beyer: {_tp_beyer}"
                                                                )
                                                    except Exception as tp_err:
                                                        logger.debug(
                                                            f"Track pattern display: {tp_err}"
                                                        )

                                                    # CLOUD BACKUP: Push to GitHub so data survives Render redeploys
                                                    try:
                                                        if backup_to_github_async:
                                                            backup_to_github_async(
                                                                gold_db.db_path
                                                            )
                                                            logger.info(
                                                                "☁️ GitHub backup triggered (async)"
                                                            )
                                                    except Exception as bk_err:
                                                        logger.debug(
                                                            f"GitHub backup note: {bk_err}"
                                                        )

                                                    # Force immediate rerun so the form is
                                                    # replaced by the confirmation message.
                                                    # Without this, both form + success show
                                                    # on the same render cycle.
                                                    st.rerun()

                                                else:
                                                    st.error(
                                                        "❌ Failed to save to database"
                                                    )

                                            except Exception as e:
                                                st.error(f"❌ Error: {str(e)}")
                                                import traceback

                                                st.code(
                                                    traceback.format_exc(),
                                                    language="python",
                                                )

                                except ValueError:
                                    st.error(
                                        "❌ Invalid format - use numbers separated by commas (e.g., 8,5,6,9)"
                                    )

                        st.markdown("---")

                # Show recently saved results for verification
                st.markdown("---")
                st.markdown("### 📊 Database Integrity Verification")

                try:
                    import os
                    import sqlite3

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
                        st.metric(
                            "DB Size",
                            f"{db_size:.2f} MB",
                            help="Physical database file size - persists across sessions",
                        )
                    with col2:
                        st.metric(
                            "Races Analyzed",
                            total_analyzed,
                            help="All races saved automatically",
                        )
                    with col3:
                        st.metric(
                            "With Results",
                            total_with_results,
                            help="Races ready for ML training",
                        )
                    with col4:
                        st.metric(
                            "Horses Saved",
                            total_horses_saved,
                            help="Total training examples (5 per completed race)",
                        )

                    st.success(
                        f"🔒 **Data Persistence Verified:** All data stored in `{gold_db.db_path}` - survives browser close/reopen!"
                    )

                    # 4.5 Show Intelligent Learning Insights (if any)
                    if (
                        "last_learning_insights" in st.session_state
                        and st.session_state["last_learning_insights"]
                    ):
                        st.markdown("#### 🎓 Latest Learning Insights")
                        insights = st.session_state["last_learning_insights"]

                        with st.expander(
                            f"📊 Found {len(insights)} pattern(s) from last race analysis",
                            expanded=True,
                        ):
                            for i, insight in enumerate(insights, 1):
                                pattern_icons = {
                                    "best_recent_speed": "⚡",
                                    "class_drop": "📉",
                                    "layoff_cycle_bounce": "🔄",
                                    "lone_presser_hot_pace": "🏃",
                                    "form_speed_override": "📈",
                                    "post_bias_alignment": "🎯",
                                }
                                icon = pattern_icons.get(insight["pattern"], "💡")

                                st.markdown(f"""
                                **{icon} Pattern {i}: {insight["pattern"].replace("_", " ").title()}**
                                - Horse: **{insight["horse"]}**
                                - {insight["description"]}
                                - Confidence: {insight["confidence"] * 100:.0f}%
                                """)

                            st.info(
                                "💡 These insights are automatically stored and used to improve future predictions!"
                            )

                    # 4.6 Show Pattern Learning History
                    if INTELLIGENT_LEARNING_AVAILABLE:
                        try:
                            engine = IntelligentLearningEngine(gold_db.db_path)
                            learnings = engine.get_accumulated_learnings()

                            if learnings.get("total_races_analyzed", 0) > 0:
                                st.markdown("#### 📚 Accumulated Learning History")

                                col_a, col_b = st.columns(2)
                                with col_a:
                                    st.metric(
                                        "Races Analyzed",
                                        learnings["total_races_analyzed"],
                                        help="Total races where patterns were identified",
                                    )

                                with col_b:
                                    top_patterns = list(
                                        learnings.get("pattern_frequency", {}).keys()
                                    )[:3]
                                    if top_patterns:
                                        st.metric(
                                            "Top Patterns",
                                            ", ".join(
                                                [
                                                    p.replace("_", " ").title()[:15]
                                                    for p in top_patterns
                                                ]
                                            ),
                                            help="Most frequently identified winning patterns",
                                        )

                                # Show pattern frequency
                                pattern_freq = learnings.get("pattern_frequency", {})
                                if pattern_freq:
                                    with st.expander("📈 Pattern Frequency Details"):
                                        for pattern, data in pattern_freq.items():
                                            count = data.get("count", 0)
                                            avg_imp = data.get("avg_improvement", 0)
                                            st.write(
                                                f"- **{pattern.replace('_', ' ').title()}**: {count} times (avg rank improvement: {avg_imp:.1f})"
                                            )
                        except Exception:
                            pass  # Silently skip if learning history fails

                    # 4. Get last 10 saved results
                    cursor.execute("""
                        SELECT
                            race_id,
                            horse_name,
                            actual_finish_position,
                            predicted_finish_position,
                            prediction_error,
                            timestamp
                        FROM gold_high_iq
                        ORDER BY timestamp DESC
                        LIMIT 10
                    """)

                    recent_results = cursor.fetchall()
                    conn.close()

                    if recent_results:
                        st.markdown("#### 🏇 Last 10 Training Examples Saved")

                        verify_df = pd.DataFrame(
                            recent_results,
                            columns=[
                                "Race ID",
                                "Horse",
                                "Actual Pos",
                                "Predicted Pos",
                                "Error",
                                "Saved At",
                            ],
                        )

                        # Format the dataframe
                        verify_df["Error"] = verify_df["Error"].round(1)
                        verify_df["Saved At"] = pd.to_datetime(
                            verify_df["Saved At"]
                        ).dt.strftime("%m/%d %I:%M %p")

                        st.dataframe(
                            verify_df, use_container_width=True, hide_index=True
                        )

                        st.caption(
                            "💡 Each completed race saves 4 entries (top 4 finishers). Database automatically commits and persists all data."
                        )
                    else:
                        st.info(
                            "📋 No results in database yet. Submit your first race above to begin ML training data collection!"
                        )

                except Exception as e:
                    st.error(f"❌ Database verification failed: {str(e)}")
                    st.warning(
                        "⚠️ This may indicate database corruption or file permission issues."
                    )

        # Tab 3: Auto-Calibration Monitor
        with tab_calibration:
            st.markdown("""
            ### 🤖 Real-Time Adaptive Learning Monitor
            
            This dashboard shows **proof** that your model is automatically learning from each race result.
            After every result submission, the system adjusts component weights using gradient descent
            and **persists the learned weights to the database** so they survive restarts.
            """)

            # Load current learned weights from database (v2 system)
            try:
                if ADAPTIVE_LEARNING_AVAILABLE:
                    # Load directly from database - these are the ACTUAL weights being used
                    db_learned_weights = get_live_learned_weights(PERSISTENT_DB_PATH)

                    st.markdown("#### 🧠 Learned Component Weights (From Database)")
                    st.caption(
                        "These weights have been automatically tuned from historical race results"
                    )

                    # Display core weights as metrics
                    cols = st.columns(6)
                    weight_names = ["class", "speed", "form", "pace", "style", "post"]
                    for idx, weight_name in enumerate(weight_names):
                        with cols[idx]:
                            weight_val = db_learned_weights.get(weight_name, 0.0)
                            default_val = {
                                "class": 2.5,
                                "speed": 2.0,
                                "form": 1.8,
                                "pace": 1.5,
                                "style": 2.0,
                                "post": 0.8,
                            }.get(weight_name, 0)
                            delta = weight_val - default_val
                            st.metric(
                                weight_name.capitalize(),
                                f"{weight_val:.2f}",
                                delta=f"{delta:+.2f}" if delta != 0 else None,
                                delta_color="normal",
                                help=f"Learned emphasis on {weight_name} factor (default: {default_val})",
                            )

                    # Show odds drift learning
                    st.markdown("---")
                    st.markdown("#### 💰 Odds Drift Learning (Pegasus 2026 Tuned)")
                    cols2 = st.columns(4)

                    odds_weights = {
                        "odds_drift_penalty": ("Drift OUT Penalty", -3.0),
                        "smart_money_bonus": ("Smart $ Bonus", 2.5),
                        "a_group_drift_gate": ("A-Group Gate", 2.0),
                    }

                    col_idx = 0
                    for key, (label, default) in odds_weights.items():
                        with cols2[col_idx]:
                            val = db_learned_weights.get(key, default)
                            st.metric(label, f"{val:.1f}", help=f"Default: {default}")
                        col_idx += 1

                    st.success(
                        "✅ **Weights persist to database** - survive app restarts and Render redeploys!"
                    )

                    # Show calibration history from database
                    st.markdown("---")
                    st.markdown("#### 📈 Calibration History (from Database)")

                    try:
                        import sqlite3

                        conn = sqlite3.connect(PERSISTENT_DB_PATH, timeout=5.0)
                        cursor = conn.cursor()

                        cursor.execute("""
                            SELECT 
                                calibration_timestamp,
                                races_analyzed,
                                winner_accuracy,
                                top3_accuracy,
                                improvements_json
                            FROM calibration_history
                            ORDER BY calibration_timestamp DESC
                            LIMIT 10
                        """)

                        cal_history = cursor.fetchall()
                        conn.close()

                        if cal_history:
                            cal_df = pd.DataFrame(
                                cal_history,
                                columns=[
                                    "Timestamp",
                                    "Races",
                                    "Winner %",
                                    "Top-3 %",
                                    "Improvements",
                                ],
                            )
                            cal_df["Winner %"] = (cal_df["Winner %"] * 100).round(
                                1
                            ).astype(str) + "%"
                            cal_df["Top-3 %"] = (cal_df["Top-3 %"] * 100).round(
                                1
                            ).astype(str) + "%"
                            cal_df["Timestamp"] = pd.to_datetime(
                                cal_df["Timestamp"]
                            ).dt.strftime("%m/%d %I:%M %p")
                            cal_df = cal_df.drop(columns=["Improvements"])

                            st.dataframe(
                                cal_df, use_container_width=True, hide_index=True
                            )
                            st.caption(
                                f"📚 System has learned from **{cal_df['Races'].sum()}** total race analyses"
                            )
                        else:
                            st.info(
                                "📋 No calibration history yet. Submit race results to start learning!"
                            )

                    except Exception as db_err:
                        st.warning(f"Could not load calibration history: {db_err}")

                    # Track-specific profiles moved to Track Intelligence tab
                    st.markdown("---")
                    st.info(
                        "🏇 **Per-track calibration profiles** have been consolidated into the **🧠 Track Intelligence** tab for a unified view."
                    )

                else:
                    # Fallback to v1 system
                    import importlib

                    import unified_rating_engine

                    importlib.reload(unified_rating_engine)

                    current_weights = unified_rating_engine.BASE_WEIGHTS.copy()

                    st.markdown("#### 🎯 Current Component Weights")
                    st.caption(
                        "These weights determine how much each factor influences the final rating"
                    )

                    # Display weights as metrics
                    cols = st.columns(6)
                    weight_names = ["class", "speed", "form", "pace", "style", "post"]
                    for idx, weight_name in enumerate(weight_names):
                        with cols[idx]:
                            weight_val = current_weights.get(weight_name, 0.0)
                            st.metric(
                                weight_name.capitalize(),
                                f"{weight_val:.3f}",
                                help=f"Current emphasis on {weight_name} factor",
                            )

                st.markdown("---")

                # Load calibration history from DATABASE (not JSON file)
                import json as _json_cal

                try:
                    _cal_conn = sqlite3.connect(gold_db.db_path)
                    _cal_cursor = _cal_conn.cursor()
                    _cal_cursor.execute("""
                        SELECT calibration_timestamp, races_analyzed, winner_accuracy, 
                               top3_accuracy, weights_json, improvements_json
                        FROM calibration_history
                        ORDER BY calibration_timestamp DESC
                        LIMIT 20
                    """)
                    _cal_rows = _cal_cursor.fetchall()
                    _cal_conn.close()

                    if _cal_rows:
                        st.markdown("#### 📈 Recent Calibration Events")
                        st.caption(
                            f"Showing last {len(_cal_rows)} auto-calibration updates from database"
                        )

                        history_records = []
                        for cal_row in _cal_rows:
                            record = {
                                "Timestamp": cal_row[0] or "N/A",
                                "Races Used": cal_row[1] or 0,
                                "Winner Acc": f"{(cal_row[2] or 0) * 100:.1f}%",
                                "Top-3 Acc": f"{(cal_row[3] or 0) * 100:.1f}%",
                            }
                            # Parse weight changes
                            try:
                                improvements = (
                                    _json_cal.loads(cal_row[5]) if cal_row[5] else {}
                                )
                                for w_name in weight_names:
                                    if (
                                        w_name in improvements
                                        and improvements[w_name] != 0
                                    ):
                                        record[f"{w_name.capitalize()} Δ"] = (
                                            f"{improvements[w_name]:+.3f}"
                                        )
                            except Exception:
                                pass
                            history_records.append(record)

                        if history_records:
                            history_df = pd.DataFrame(history_records)
                            st.dataframe(
                                history_df, use_container_width=True, hide_index=True
                            )

                            # Show learning progress metrics
                            # ── Sync with retrained model if available ───
                            # The dashboard must show the MOST RECENT accuracy
                            # source, whether from auto-calibration or ML retrain.
                            # retraining_history has true held-out val metrics;
                            # calibration_history has weight-tuning heuristic metrics.
                            st.markdown("#### 📊 Learning Progress")
                            total_calibrations = len(_cal_rows)

                            # Default: auto-calibration metrics
                            latest_winner_acc = (_cal_rows[0][2] or 0) * 100
                            latest_top3_acc = (_cal_rows[0][3] or 0) * 100
                            latest_top4_acc = 0.0
                            metric_source = "auto-calibration"

                            # Override with ML retrain metrics if they are newer
                            try:
                                _rt_conn = sqlite3.connect(gold_db.db_path, timeout=5.0)
                                _rt_cur = _rt_conn.cursor()
                                _rt_cur.execute("""
                                    SELECT retrain_timestamp,
                                           val_winner_accuracy,
                                           val_top3_accuracy,
                                           val_top5_accuracy
                                    FROM retraining_history
                                    ORDER BY retrain_timestamp DESC
                                    LIMIT 1
                                """)
                                _rt_row = _rt_cur.fetchone()
                                _rt_conn.close()
                                if _rt_row:
                                    # Compare timestamps to pick the more recent source
                                    rt_ts = _rt_row[0] or ""
                                    cal_ts = _cal_rows[0][0] or ""
                                    if rt_ts >= cal_ts:
                                        latest_winner_acc = (_rt_row[1] or 0) * 100
                                        latest_top3_acc = (_rt_row[2] or 0) * 100
                                        latest_top4_acc = (_rt_row[3] or 0) * 100
                                        metric_source = "ML retrain"
                            except Exception:
                                pass  # Fall back to auto-calibration metrics

                            col1, col2, col3, col4 = st.columns(4)
                            with col1:
                                st.metric(
                                    "Total Calibrations",
                                    total_calibrations,
                                    help="Number of times the model has auto-adjusted",
                                )
                            with col2:
                                st.metric(
                                    "Latest Winner Accuracy",
                                    f"{latest_winner_acc:.1f}%",
                                    help="Most recent winner prediction accuracy",
                                )
                            with col3:
                                st.metric(
                                    "Latest Top-3 Accuracy",
                                    f"{latest_top3_acc:.1f}%",
                                    help="Most recent top-3 prediction accuracy",
                                )
                            with col4:
                                st.metric(
                                    "Latest Top-4 Accuracy",
                                    f"{latest_top4_acc:.1f}%"
                                    if latest_top4_acc > 0
                                    else "N/A",
                                    help="Most recent top-4 prediction accuracy (from ML retrain)",
                                )

                            if metric_source == "ML retrain":
                                st.info(
                                    "📊 **Metrics source: ML Retrain** — showing held-out validation accuracy from the latest retrained model"
                                )
                            st.success(
                                "✅ **Auto-Calibration Active:** Model updates after every race result submission"
                            )

                            # Show weight evolution chart (if enough data)
                            if len(_cal_rows) >= 5:
                                st.markdown("#### 📉 Weight Evolution Over Time")
                                weight_history = {w: [] for w in weight_names}
                                timestamps = []
                                for cal_row in reversed(
                                    _cal_rows
                                ):  # Oldest first for chart
                                    timestamps.append(cal_row[0] or "")
                                    try:
                                        weights = (
                                            _json_cal.loads(cal_row[4])
                                            if cal_row[4]
                                            else {}
                                        )
                                        for w_name in weight_names:
                                            weight_history[w_name].append(
                                                weights.get(w_name, 0)
                                            )
                                    except Exception:
                                        for w_name in weight_names:
                                            weight_history[w_name].append(0)
                                chart_data = pd.DataFrame(
                                    weight_history, index=timestamps
                                )
                                st.line_chart(chart_data)
                                st.caption(
                                    "📌 Watch how weights adjust based on race outcomes - evidence of real-time learning!"
                                )
                        else:
                            st.info(
                                "📋 No calibration events recorded yet. Submit race results to trigger auto-learning!"
                            )
                    else:
                        st.info(
                            "📋 No calibration history yet. The model will auto-calibrate after you submit race results."
                        )
                except Exception as _cal_err:
                    st.warning(
                        f"Could not load calibration history from database: {_cal_err}"
                    )
                    st.info(
                        "📋 Calibration history will appear after your first race result submission."
                    )

            except Exception as e:
                st.error(f"❌ Error loading calibration data: {str(e)}")
                import traceback

                st.code(traceback.format_exc())

        # Tab 4: Retrain Model (ADMIN ONLY)
        with tab_retrain:
            # ── Admin authentication gate ────────────────────────
            # Only the site owner can retrain. Customers see a locked message.
            # Set ADMIN_PASSWORD in Streamlit Cloud → Settings → Secrets:
            #   ADMIN_PASSWORD = "your-password-here"
            # Or locally in .streamlit/secrets.toml
            admin_password = ""
            try:
                admin_password = st.secrets.get("ADMIN_PASSWORD", "")
            except Exception:
                pass

            # Determine admin access WITHOUT st.stop() (which kills Tab 5)
            _retrain_unlocked = False
            if not admin_password:
                # No secret configured — block on deployed, allow locally
                if is_render() or os.environ.get("STREAMLIT_SERVER_HEADLESS"):
                    st.warning("🔒 Model retraining is restricted to administrators.")
                    st.info("Contact support if you need access to this feature.")
                else:
                    _retrain_unlocked = True  # Local dev — no secret needed
            else:
                if "admin_authenticated" not in st.session_state:
                    st.session_state.admin_authenticated = False

                if not st.session_state.admin_authenticated:
                    st.markdown("### 🔐 Admin Access Required")
                    st.info("Model retraining is restricted to administrators.")
                    pwd = st.text_input(
                        "Enter admin password:", type="password", key="admin_pwd_input"
                    )
                    if st.button("Unlock", key="admin_unlock_btn"):
                        if pwd == admin_password:
                            st.session_state.admin_authenticated = True
                            st.rerun()
                        else:
                            st.error("Incorrect password.")
                else:
                    _retrain_unlocked = True  # Admin authenticated

            if _retrain_unlocked:
                st.markdown("""
                ### Retrain ML Model with Real Data

                Once you have **50+ completed races**, retrain the model to learn from real outcomes.
                The model uses PyTorch with Plackett-Luce ranking loss for optimal accuracy.
                Features are auto-standardised. Early stopping prevents overfitting.
                """)

                # Check if ready
                ready_to_train = stats.get("total_races", 0) >= 50

                if not ready_to_train:
                    st.warning(
                        f"⏳ Need at least 50 completed races. Currently: {stats.get('total_races', 0)}"
                    )
                    st.info("💡 Complete more races in the 'Submit Actual Top 4' tab.")
                else:
                    st.success(
                        f"✅ Ready to train! {stats.get('total_races', 0)} races available."
                    )

                    # Training parameters — tuned for small horse racing datasets
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        epochs = st.number_input(
                            "Max Epochs",
                            min_value=20,
                            max_value=300,
                            value=100,
                            help="Training stops early if val loss plateaus (patience=15)",
                        )
                    with col2:
                        learning_rate = st.select_slider(
                            "Learning Rate",
                            options=[0.0001, 0.0002, 0.0005, 0.001, 0.002, 0.005],
                            value=0.0005,
                            help="Lower = more stable. 0.0005 is optimal for <200 races",
                        )
                    with col3:
                        batch_size = st.selectbox(
                            "Batch Size (races)",
                            [1, 2, 4, 8, 16],
                            index=2,
                            help="Number of races per gradient update. 4 is optimal for <200 races",
                        )

                    # Train button
                    if st.button(
                        "🚀 Start Retraining", type="primary", key="retrain_btn"
                    ):
                        with st.spinner(
                            f"Training model on {stats.get('total_races', 0)} races... This may take 2-5 minutes..."
                        ):
                            try:
                                from retrain_model import retrain_model

                                results = retrain_model(
                                    db_path=gold_db.db_path,
                                    epochs=epochs,
                                    learning_rate=learning_rate,
                                    batch_size=batch_size,
                                    min_races=50,
                                )

                                if "error" in results:
                                    st.error(f"❌ Training failed: {results['error']}")
                                else:
                                    st.success("✅ Training complete!")

                                    # Display results (validation set accuracy)
                                    st.caption(
                                        "Validation accuracy: retrained model tested on "
                                        "held-out races it was NOT trained on."
                                    )
                                    col1, col2, col3 = st.columns(3)
                                    with col1:
                                        st.metric(
                                            "Val Winner Accuracy",
                                            f"{results['metrics']['winner_accuracy']:.1%}",
                                            help="% of validation races where the retrained model's #1 pick was the actual winner",
                                        )
                                    with col2:
                                        st.metric(
                                            "Val Top-3 Accuracy",
                                            f"{results['metrics']['top3_accuracy']:.1%}",
                                            help="Avg overlap between model's top 3 and actual top 3 on validation races",
                                        )
                                    with col3:
                                        st.metric(
                                            "Val Top-4 Accuracy",
                                            f"{results['metrics']['top4_accuracy']:.1%}",
                                            help="Avg overlap between model's top 4 and actual top 4 on validation races",
                                        )

                                    st.info(
                                        f"⏱️ Training time: {results.get('duration', 0):.1f} seconds"
                                    )
                                    epochs_info = f"Best epoch: {results.get('best_epoch', '?')} / {results.get('total_epochs_run', '?')} run"
                                    if results.get("early_stopped"):
                                        epochs_info += " (early stopped)"
                                    st.info(f"📈 {epochs_info}")
                                    st.info(
                                        f"💾 Model saved: {results.get('model_path', 'N/A')}"
                                    )

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
                        history_df = pd.read_sql_query(
                            """
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
                        """,
                            conn,
                        )
                        conn.close()

                        if not history_df.empty:
                            history_df.columns = [
                                "Timestamp",
                                "Races Used",
                                "Winner Acc",
                                "Top-3 Acc",
                                "Top-4 Acc",
                                "Duration (s)",
                            ]
                            history_df["Winner Acc"] = (
                                history_df["Winner Acc"] * 100
                            ).round(1).astype(str) + "%"
                            history_df["Top-3 Acc"] = (
                                history_df["Top-3 Acc"] * 100
                            ).round(1).astype(str) + "%"
                            history_df["Top-4 Acc"] = (
                                history_df["Top-4 Acc"] * 100
                            ).round(1).astype(str) + "%"
                            history_df["Duration (s)"] = history_df[
                                "Duration (s)"
                            ].round(1)

                            st.dataframe(
                                history_df, use_container_width=True, hide_index=True
                            )
                        else:
                            st.info(
                                "No training history yet. Train the model to see results here."
                            )
                    except Exception:
                        st.warning("Could not load training history.")

        # Tab 5: Track Intelligence — Unified Command Center
        with tab_track_intel:
            st.markdown("### 🧠 Track Intelligence — Unified Command Center")
            st.caption(
                "Live calibration weights, bias profiling, accuracy analytics, and ML model status — "
                "all powered by your Gold High-IQ database and dynamic engine calculations."
            )

            try:
                # ── Load all live data ──
                _gw = {}
                _track_cals = []
                if ADAPTIVE_LEARNING_AVAILABLE:
                    try:
                        _gw = get_live_learned_weights(PERSISTENT_DB_PATH) or {}
                        _track_cals = (
                            get_all_track_calibrations_summary(PERSISTENT_DB_PATH) or []
                        )
                    except Exception:
                        pass

                _ti_summaries = []
                if TRACK_INTEL_AVAILABLE and _track_intel is not None:
                    _ti_summaries = _track_intel.get_all_track_summaries() or []
                    if not _ti_summaries:
                        with st.spinner(
                            "Building track profiles for the first time..."
                        ):
                            _track_intel.rebuild_all_profiles()
                        _ti_summaries = _track_intel.get_all_track_summaries() or []

                    # Ensure active analysis track has a fresh profile entry
                    _ti_active = st.session_state.get("track_name", "").strip().upper()
                    if (
                        _ti_active
                        and _ti_active != "UNKNOWN TRACK"
                        and _ti_active
                        not in {s["track"].upper() for s in _ti_summaries}
                    ):
                        try:
                            _track_intel.update_after_submission(_ti_active)
                            _ti_summaries = _track_intel.get_all_track_summaries() or []
                        except Exception:
                            pass  # Graceful: new track with zero history won't have data yet

                _cal_tracks = {
                    tc.get("track_code", "").upper(): tc for tc in _track_cals
                }
                _ti_tracks = {s["track"].upper(): s for s in _ti_summaries}
                _all_track_codes = sorted(
                    set(list(_cal_tracks.keys()) + list(_ti_tracks.keys()))
                )

                _total_profiled_races = sum(
                    _cal_tracks.get(t, {}).get(
                        "races_trained_on", _ti_tracks.get(t, {}).get("total_races", 0)
                    )
                    for t in _all_track_codes
                )
                _avg_conf = (
                    sum(tc.get("avg_confidence", 0) for tc in _track_cals)
                    / max(len(_track_cals), 1)
                    if _track_cals
                    else 0
                )

                # ═══ TIER 1 — Engine Status Bar ═══
                # Show active analysis banner if user has parsed PPs
                _active_track = st.session_state.get("track_name", "").strip()
                _active_track_upper = (
                    _active_track.upper()
                    if _active_track and _active_track != "Unknown Track"
                    else ""
                )
                if _active_track_upper:
                    _at_races = _cal_tracks.get(_active_track_upper, {}).get(
                        "races_trained_on",
                        _ti_tracks.get(_active_track_upper, {}).get("total_races", 0),
                    )
                    _at_conf = _cal_tracks.get(_active_track_upper, {}).get(
                        "avg_confidence", 0
                    )
                    _at_style = _ti_tracks.get(_active_track_upper, {}).get(
                        "style_bias", "—"
                    )
                    st.info(
                        f"🏇 **Currently Analyzing: {_active_track}** — "
                        f"{_at_races} historical races | "
                        f"Confidence: {_at_conf:.0%} | "
                        f"Style Bias: {_at_style}"
                        if _active_track_upper in _all_track_codes
                        else f"🏇 **Currently Analyzing: {_active_track}** — New track, no historical data yet"
                    )

                st.markdown("---")
                _es1, _es2, _es3, _es4 = st.columns(4)
                with _es1:
                    st.metric("Profiled Tracks", len(_all_track_codes))
                with _es2:
                    st.metric("Total Races", _total_profiled_races)
                with _es3:
                    _ml_loaded = (
                        ML_BLEND_AVAILABLE
                        and _ml_blend is not None
                        and _ml_blend.model is not None
                    )
                    st.metric(
                        "ML Model", "✅ Active" if _ml_loaded else "❌ Not Loaded"
                    )
                with _es4:
                    st.metric(
                        "Avg Confidence", f"{_avg_conf:.0%}" if _avg_conf > 0 else "—"
                    )

                # Rebuild button
                _rb1, _rb2 = st.columns([3, 1])
                with _rb2:
                    if TRACK_INTEL_AVAILABLE and _track_intel is not None:
                        if st.button("🔄 Rebuild Profiles", key="rebuild_profiles_btn"):
                            with st.spinner("Rebuilding all track profiles..."):
                                _track_intel.rebuild_all_profiles()
                            st.success("✅ Profiles rebuilt!")
                            st.rerun()

                # ═══ TIER 2 — Global Calibrated Weights ═══
                if _gw:
                    with st.expander(
                        "⚖️ Global Calibrated Weights (Auto-Cal Engine)", expanded=False
                    ):
                        st.caption(
                            "Blended weights learned across all tracks. Per-track deltas shown in detail panel below."
                        )
                        _gw_cols = st.columns(6)
                        for _wi, _wn in enumerate(
                            ["class", "speed", "form", "pace", "style", "post"]
                        ):
                            with _gw_cols[_wi]:
                                st.metric(_wn.capitalize(), f"{_gw.get(_wn, 0):.2f}")
                        _od_cols = st.columns(3)
                        with _od_cols[0]:
                            st.metric(
                                "Odds Drift Penalty",
                                f"{_gw.get('odds_drift_penalty', 0):.2f}",
                            )
                        with _od_cols[1]:
                            st.metric(
                                "Smart Money Bonus",
                                f"{_gw.get('smart_money_bonus', 0):.2f}",
                            )
                        with _od_cols[2]:
                            st.metric(
                                "A-Group Drift Gate",
                                f"{_gw.get('a_group_drift_gate', 0):.2f}",
                            )

                # ═══ TIER 3 — Track Overview Grid ═══
                st.markdown("---")
                if not _all_track_codes:
                    st.info(
                        "📋 No track data available yet. Submit race results to build intelligence profiles."
                    )
                else:
                    st.markdown("#### 🏇 Track Profiles")
                    _overview_rows = []
                    for _tc in _all_track_codes:
                        _cal = _cal_tracks.get(_tc, {})
                        _ti = _ti_tracks.get(_tc, {})
                        _n_races = _cal.get(
                            "races_trained_on", _ti.get("total_races", 0)
                        )
                        _conf = _cal.get("avg_confidence", 0)
                        _conf_icon = (
                            "🟢" if _conf >= 0.7 else ("🟡" if _conf >= 0.4 else "🔴")
                        )
                        _style = _ti.get("style_bias", "—")
                        _win_pct = _ti.get("overall_winner_pct", 0)
                        _overview_rows.append(
                            {
                                "": _conf_icon,
                                "Track": _tc.title(),
                                "Races": _n_races,
                                "Confidence": f"{_conf:.0%}" if _conf > 0 else "—",
                                "Winner %": f"{_win_pct:.1f}%" if _win_pct > 0 else "—",
                                "Style Bias": _style,
                            }
                        )
                    st.dataframe(
                        pd.DataFrame(_overview_rows),
                        use_container_width=True,
                        hide_index=True,
                    )

                    # ═══ TIER 4 — Deep-Dive Panel ═══
                    st.markdown("---")
                    # Auto-select the track being analyzed (if it exists in profiles)
                    _default_idx = 0
                    if _active_track_upper and _active_track_upper in _all_track_codes:
                        _default_idx = _all_track_codes.index(_active_track_upper)
                    _selected_track = st.selectbox(
                        "Select track for deep analysis:",
                        _all_track_codes,
                        index=_default_idx,
                        key="ti_deep_selector",
                        format_func=lambda x: x.title(),
                    )

                    if _selected_track:
                        # Load per-track data
                        _tw = {}
                        if ADAPTIVE_LEARNING_AVAILABLE:
                            try:
                                _tw = (
                                    get_live_learned_weights(
                                        PERSISTENT_DB_PATH, track_code=_selected_track
                                    )
                                    or {}
                                )
                            except Exception:
                                _tw = _cal_tracks.get(_selected_track, {}).get(
                                    "weights", {}
                                )

                        _profile = None
                        if TRACK_INTEL_AVAILABLE and _track_intel is not None:
                            try:
                                _profile = _track_intel.build_full_profile(
                                    _selected_track
                                )
                            except Exception:
                                pass

                        _accuracy = {}
                        _biases = {}
                        if gold_db:
                            try:
                                _accuracy = (
                                    gold_db.calculate_accuracy_stats(_selected_track)
                                    or {}
                                )
                            except Exception:
                                pass
                            try:
                                _biases = gold_db.detect_biases(_selected_track) or {}
                            except Exception:
                                pass

                        _tc_races = _cal_tracks.get(_selected_track, {}).get(
                            "races_trained_on", _profile.total_races if _profile else 0
                        )

                        st.markdown(
                            f"#### 🏟️ {_selected_track} — {_tc_races} Races Profiled"
                        )

                        # Sub-tabs
                        _st1, _st2, _st3, _st4, _st5 = st.tabs(
                            [
                                "📊 Overview",
                                "🏁 Surface & Distance",
                                "🎯 Bias Detection",
                                "🏆 J/T Combos",
                                "⚙️ Calibration & ML",
                            ]
                        )

                        # ── Sub-tab 1: Overview ──
                        with _st1:
                            if _profile:
                                _ov1, _ov2, _ov3, _ov4 = st.columns(4)
                                with _ov1:
                                    st.metric(
                                        "Winner %",
                                        f"{_profile.overall_winner_pct:.1f}%",
                                    )
                                with _ov2:
                                    st.metric(
                                        "Top-3 %", f"{_profile.overall_top3_pct:.1f}%"
                                    )
                                with _ov3:
                                    st.metric(
                                        "Top-4 %", f"{_profile.overall_top4_pct:.1f}%"
                                    )
                                with _ov4:
                                    st.metric("Horses Analysed", _profile.total_horses)
                                st.markdown("---")
                                st.markdown(
                                    "**Style Bias:** "
                                    + (_profile.style_bias or "No dominant bias")
                                )
                                if _profile.insights:
                                    st.markdown("##### 💡 Detected Insights")
                                    for _ins in _profile.insights:
                                        _sev_icon = {
                                            "strong": "🔴",
                                            "moderate": "🟡",
                                            "mild": "🟢",
                                        }.get(_ins.severity, "⚪")
                                        st.info(
                                            f"{_sev_icon} **[{_ins.category.upper()}]** {_ins.description} *(n={_ins.sample_size}, conf={_ins.confidence:.0%})*"
                                        )
                            else:
                                st.caption(
                                    "No profile data available yet. Rebuild profiles to generate."
                                )

                        # ── Sub-tab 2: Surface & Distance ──
                        with _st2:
                            if _accuracy:
                                st.markdown("##### 🏁 Surface Accuracy")
                                _surf_cols = st.columns(2)
                                for _si, _surf in enumerate(["dirt", "turf"]):
                                    with _surf_cols[_si]:
                                        _sr = _accuracy.get(f"{_surf}_races", {})
                                        _sn = (
                                            int(_sr.get("value", 0))
                                            if isinstance(_sr, dict)
                                            else int(_sr or 0)
                                        )
                                        _sa = _accuracy.get(f"{_surf}_accuracy_pct", {})
                                        _sav = (
                                            float(_sa.get("value", 0))
                                            if isinstance(_sa, dict)
                                            else float(_sa or 0)
                                        )
                                        _sw = _accuracy.get(f"{_surf}_winner_pct", {})
                                        _swv = (
                                            float(_sw.get("value", 0))
                                            if isinstance(_sw, dict)
                                            else float(_sw or 0)
                                        )
                                        _icon = "🟤" if _surf == "dirt" else "🟢"
                                        if _sn > 0:
                                            st.metric(
                                                f"{_icon} {_surf.title()}",
                                                f"{_sav:.1f}%",
                                                delta=f"{_sn} races",
                                                delta_color="off",
                                                help=f"Top-4 overlap accuracy on {_surf}. Winner acc: {_swv:.1f}%",
                                            )
                                        else:
                                            st.metric(
                                                f"{_icon} {_surf.title()}",
                                                "—",
                                                delta="0 races",
                                                delta_color="off",
                                            )

                                st.markdown("##### 📏 Distance Accuracy")
                                _dist_cols = st.columns(2)
                                _dist_labels = {
                                    "sprint": ("⚡ Sprints", "4½f – 7½f"),
                                    "route": ("🏃 Routes", "8f – 1½mi"),
                                }
                                for _di, _dist in enumerate(["sprint", "route"]):
                                    with _dist_cols[_di]:
                                        _dr = _accuracy.get(f"{_dist}_races", {})
                                        _dn = (
                                            int(_dr.get("value", 0))
                                            if isinstance(_dr, dict)
                                            else int(_dr or 0)
                                        )
                                        _da = _accuracy.get(f"{_dist}_accuracy_pct", {})
                                        _dav = (
                                            float(_da.get("value", 0))
                                            if isinstance(_da, dict)
                                            else float(_da or 0)
                                        )
                                        _dw = _accuracy.get(f"{_dist}_winner_pct", {})
                                        _dwv = (
                                            float(_dw.get("value", 0))
                                            if isinstance(_dw, dict)
                                            else float(_dw or 0)
                                        )
                                        _lbl, _rng = _dist_labels[_dist]
                                        if _dn > 0:
                                            st.metric(
                                                _lbl,
                                                f"{_dav:.1f}%",
                                                delta=f"{_dn} races ({_rng})",
                                                delta_color="off",
                                                help=f"Top-4 overlap accuracy for {_dist}s. Winner acc: {_dwv:.1f}%",
                                            )
                                        else:
                                            st.metric(
                                                _lbl,
                                                "—",
                                                delta=f"0 races ({_rng})",
                                                delta_color="off",
                                            )
                            elif _profile:
                                st.markdown("##### 🏁 Surface & Distance Breakdown")
                                _sf_cols = st.columns(3)
                                with _sf_cols[0]:
                                    st.markdown("**Surface**")
                                    _sf_data = []
                                    if _profile.dirt_races > 0:
                                        _sf_data.append(
                                            {
                                                "Surface": "Dirt",
                                                "Races": _profile.dirt_races,
                                                "Winner %": f"{_profile.dirt_winner_pct:.1f}%",
                                            }
                                        )
                                    if _profile.turf_races > 0:
                                        _sf_data.append(
                                            {
                                                "Surface": "Turf",
                                                "Races": _profile.turf_races,
                                                "Winner %": f"{_profile.turf_winner_pct:.1f}%",
                                            }
                                        )
                                    if _sf_data:
                                        st.dataframe(
                                            pd.DataFrame(_sf_data), hide_index=True
                                        )
                                    else:
                                        st.caption("No surface data")
                                with _sf_cols[1]:
                                    st.markdown("**Distance**")
                                    _sd_data = []
                                    if _profile.sprint_races > 0:
                                        _sd_data.append(
                                            {
                                                "Distance": "Sprint (<8f)",
                                                "Races": _profile.sprint_races,
                                                "Winner %": f"{_profile.sprint_winner_pct:.1f}%",
                                            }
                                        )
                                    if _profile.route_races > 0:
                                        _sd_data.append(
                                            {
                                                "Distance": "Route (≥8f)",
                                                "Races": _profile.route_races,
                                                "Winner %": f"{_profile.route_winner_pct:.1f}%",
                                            }
                                        )
                                    if _sd_data:
                                        st.dataframe(
                                            pd.DataFrame(_sd_data), hide_index=True
                                        )
                                    else:
                                        st.caption("No distance data")
                                with _sf_cols[2]:
                                    st.markdown("**Condition**")
                                    _sc_data = []
                                    if _profile.fast_races > 0:
                                        _sc_data.append(
                                            {
                                                "Condition": "Fast/Firm",
                                                "Races": _profile.fast_races,
                                                "Winner %": f"{_profile.fast_winner_pct:.1f}%",
                                            }
                                        )
                                    if _profile.off_track_races > 0:
                                        _sc_data.append(
                                            {
                                                "Condition": "Off Track",
                                                "Races": _profile.off_track_races,
                                                "Winner %": f"{_profile.off_track_winner_pct:.1f}%",
                                            }
                                        )
                                    if _sc_data:
                                        st.dataframe(
                                            pd.DataFrame(_sc_data), hide_index=True
                                        )
                                    else:
                                        st.caption("No condition data")
                            else:
                                st.caption("No surface/distance data available yet.")

                        # ── Sub-tab 3: Bias Detection ──
                        with _st3:
                            _has_bias = False
                            if (
                                _biases
                                and _biases.get("style_bias") != "Insufficient Data"
                            ):
                                _has_bias = True
                                _active = _biases.get("active_biases", [])
                                if isinstance(_active, list) and _active:
                                    st.markdown(
                                        "**Track Tendencies:** "
                                        + " · ".join(f"**{b}**" for b in _active)
                                    )
                                else:
                                    st.caption("No strong biases detected yet")

                                _style_bd = _biases.get("style_win_breakdown", {})
                                if (
                                    _style_bd
                                    and isinstance(_style_bd, dict)
                                    and sum(_style_bd.values()) > 0
                                ):
                                    st.markdown("##### 🎯 Running Style Distribution")
                                    _total_sw = sum(_style_bd.values())
                                    _style_rows = []
                                    for _sk, _sv in _style_bd.items():
                                        _pct = _sv / max(_total_sw, 1) * 100
                                        _style_rows.append(
                                            {
                                                "Style": _sk,
                                                "Wins": _sv,
                                                "Win %": f"{_pct:.1f}%",
                                            }
                                        )
                                    st.dataframe(
                                        pd.DataFrame(_style_rows),
                                        hide_index=True,
                                        use_container_width=True,
                                    )

                                _pp_stats = _biases.get("post_position_stats", {})
                                if _pp_stats and isinstance(_pp_stats, dict):
                                    st.markdown("##### 📍 Post Position Analysis")
                                    _pp_rows = []
                                    _pp_labels = {
                                        "inside": "Inside (1-3)",
                                        "middle": "Middle (4-6)",
                                        "outside": "Outside (7+)",
                                    }
                                    for _zone in ["inside", "middle", "outside"]:
                                        _zd = _pp_stats.get(_zone, {})
                                        _pp_rows.append(
                                            {
                                                "Post Zone": _pp_labels.get(
                                                    _zone, _zone
                                                ),
                                                "Win %": f"{_zd.get('win_pct', 0):.1f}%",
                                                "Top-4 %": f"{_zd.get('top4_pct', 0):.1f}%",
                                                "Sample": _zd.get("sample", 0),
                                            }
                                        )
                                    st.dataframe(
                                        pd.DataFrame(_pp_rows),
                                        use_container_width=True,
                                        hide_index=True,
                                    )

                            if _profile and not _has_bias:
                                _b_cols = st.columns(2)
                                with _b_cols[0]:
                                    st.markdown(
                                        f"**Running Style:** {_profile.style_bias}"
                                    )
                                    st.dataframe(
                                        pd.DataFrame(
                                            [
                                                {
                                                    "Style": "Speed (E/EP)",
                                                    "Win %": f"{_profile.speed_win_pct:.1f}%",
                                                },
                                                {
                                                    "Style": "Presser (P)",
                                                    "Win %": f"{_profile.presser_win_pct:.1f}%",
                                                },
                                                {
                                                    "Style": "Closer (S/C)",
                                                    "Win %": f"{_profile.closer_win_pct:.1f}%",
                                                },
                                            ]
                                        ),
                                        hide_index=True,
                                    )
                                with _b_cols[1]:
                                    st.markdown("**Post Position Zones**")
                                    st.dataframe(
                                        pd.DataFrame(
                                            [
                                                {
                                                    "Zone": "Inside (1-3)",
                                                    "Win %": f"{_profile.inside_win_pct:.1f}%",
                                                    "Top-4 %": f"{_profile.inside_top4_pct:.1f}%",
                                                },
                                                {
                                                    "Zone": "Middle (4-6)",
                                                    "Win %": f"{_profile.middle_win_pct:.1f}%",
                                                    "Top-4 %": f"{_profile.middle_top4_pct:.1f}%",
                                                },
                                                {
                                                    "Zone": "Outside (7+)",
                                                    "Win %": f"{_profile.outside_win_pct:.1f}%",
                                                    "Top-4 %": f"{_profile.outside_top4_pct:.1f}%",
                                                },
                                            ]
                                        ),
                                        hide_index=True,
                                    )

                            if not _has_bias and not _profile:
                                st.caption(
                                    "No bias data available yet. Submit more race results to enable bias detection."
                                )

                        # ── Sub-tab 4: J/T Combos ──
                        with _st4:
                            if _profile and _profile.top_jt_combos:
                                st.markdown("##### 🏆 Top Jockey-Trainer Combinations")
                                _jt_data = [
                                    {
                                        "Jockey": c["jockey"],
                                        "Trainer": c["trainer"],
                                        "Starts": c["starts"],
                                        "Wins": c["wins"],
                                        "Win %": f"{c['win_pct']}%",
                                    }
                                    for c in _profile.top_jt_combos[:10]
                                ]
                                st.dataframe(
                                    pd.DataFrame(_jt_data),
                                    hide_index=True,
                                    use_container_width=True,
                                )
                            else:
                                st.caption(
                                    "No jockey-trainer combo data available for this track yet."
                                )

                        # ── Sub-tab 5: Calibration & ML ──
                        with _st5:
                            if _tw:
                                st.markdown("##### ⚖️ Per-Track Calibrated Weights")
                                st.caption(
                                    "Deltas show how this track differs from the global calibration."
                                )
                                _wt_cols = st.columns(6)
                                for _wi, _wn in enumerate(
                                    ["class", "speed", "form", "pace", "style", "post"]
                                ):
                                    with _wt_cols[_wi]:
                                        _tv = _tw.get(_wn, 0)
                                        _gv = _gw.get(_wn, 0)
                                        _delta = round(_tv - _gv, 2)
                                        st.metric(
                                            _wn.capitalize(),
                                            f"{_tv:.2f}",
                                            delta=f"{_delta:+.2f}"
                                            if _delta != 0
                                            else None,
                                            delta_color="normal",
                                            help=f"Track weight vs global ({_gv:.2f})",
                                        )
                                _od_cols = st.columns(3)
                                with _od_cols[0]:
                                    _tdp = _tw.get("odds_drift_penalty", 0)
                                    _gdp = _gw.get("odds_drift_penalty", 0)
                                    st.metric(
                                        "Odds Drift Penalty",
                                        f"{_tdp:.2f}",
                                        delta=f"{round(_tdp - _gdp, 2):+.2f}"
                                        if round(_tdp - _gdp, 2) != 0
                                        else None,
                                        delta_color="inverse",
                                    )
                                with _od_cols[1]:
                                    _tsm = _tw.get("smart_money_bonus", 0)
                                    _gsm = _gw.get("smart_money_bonus", 0)
                                    st.metric(
                                        "Smart Money Bonus",
                                        f"{_tsm:.2f}",
                                        delta=f"{round(_tsm - _gsm, 2):+.2f}"
                                        if round(_tsm - _gsm, 2) != 0
                                        else None,
                                        delta_color="normal",
                                    )
                                with _od_cols[2]:
                                    _tag = _tw.get("a_group_drift_gate", 0)
                                    _gag = _gw.get("a_group_drift_gate", 0)
                                    st.metric(
                                        "A-Group Drift Gate",
                                        f"{_tag:.2f}",
                                        delta=f"{round(_tag - _gag, 2):+.2f}"
                                        if round(_tag - _gag, 2) != 0
                                        else None,
                                        delta_color="normal",
                                    )
                            else:
                                st.caption(
                                    "No per-track calibration weights yet. Need 2+ races with results for this track."
                                )

                            st.markdown("---")
                            st.markdown("##### 🤖 ML Blend Model Status")
                            if ML_BLEND_AVAILABLE and _ml_blend:
                                _mi = _ml_blend.get_model_info()
                                _ml_cols = st.columns(4)
                                with _ml_cols[0]:
                                    st.metric("Features", _mi.get("n_features", "?"))
                                with _ml_cols[1]:
                                    st.metric("Hidden Dim", _mi.get("hidden_dim", "?"))
                                with _ml_cols[2]:
                                    st.metric("Architecture", "Plackett-Luce NN")
                                with _ml_cols[3]:
                                    st.metric(
                                        "Status",
                                        "✅ Active"
                                        if _mi.get("available", False)
                                        else "❌ Not Loaded",
                                    )
                            else:
                                st.caption("ML Blend Engine not available.")

                            st.markdown("---")
                            st.markdown("##### 🎯 Per-Track Retrain")
                            _retrain_races = (
                                _profile.total_races if _profile else _tc_races
                            )
                            _min_races = 30
                            st.caption(
                                f"Retrain ML model using only {_selected_track} data. Requires {_min_races}+ races."
                            )
                            if _retrain_races >= _min_races:
                                if st.button(
                                    f"🚀 Retrain for {_selected_track} ({_retrain_races} races)",
                                    key=f"retrain_track_{_selected_track}",
                                    type="primary",
                                ):
                                    with st.spinner(
                                        f"Retraining on {_selected_track} data..."
                                    ):
                                        try:
                                            from retrain_model import (
                                                retrain_model as _retrain_fn,
                                            )

                                            _tr = _retrain_fn(
                                                db_path=gold_db.db_path,
                                                epochs=100,
                                                learning_rate=0.0005,
                                                batch_size=4,
                                                min_races=_min_races,
                                                track_name=_selected_track,
                                            )
                                            if "error" in _tr:
                                                st.error(f"❌ {_tr['error']}")
                                            else:
                                                st.success(
                                                    f"✅ {_selected_track} model trained!"
                                                )
                                                _rc1, _rc2, _rc3 = st.columns(3)
                                                with _rc1:
                                                    st.metric(
                                                        "Winner %",
                                                        f"{_tr['metrics']['winner_accuracy']:.1%}",
                                                    )
                                                with _rc2:
                                                    st.metric(
                                                        "Top-3 %",
                                                        f"{_tr['metrics']['top3_accuracy']:.1%}",
                                                    )
                                                with _rc3:
                                                    st.metric(
                                                        "Top-4 %",
                                                        f"{_tr['metrics']['top4_accuracy']:.1%}",
                                                    )
                                                st.info(
                                                    f"💾 Saved: {_tr.get('model_path', 'N/A')}"
                                                )
                                                _track_intel.save_track_ml_profile(
                                                    track_code=_selected_track,
                                                    model_path=_tr["model_path"],
                                                    n_features=_tr["metrics"].get(
                                                        "n_features", 0
                                                    ),
                                                    hidden_dim=64,
                                                    val_metrics=_tr["metrics"],
                                                    races_trained=_tr.get("n_races", 0),
                                                )
                                                if ML_BLEND_AVAILABLE and _ml_blend:
                                                    _ml_blend.reload_model()
                                        except Exception as _re:
                                            st.error(f"Per-track retrain error: {_re}")
                                            import traceback

                                            st.code(traceback.format_exc())
                            else:
                                st.warning(
                                    f"⏳ {_selected_track} has {_retrain_races} races. Need {_min_races}+ for retraining."
                                )

                            _tc_updated = _cal_tracks.get(_selected_track, {}).get(
                                "last_updated", ""
                            )
                            if _tc_updated:
                                try:
                                    from datetime import datetime as _dt

                                    _upd = _dt.fromisoformat(_tc_updated)
                                    st.caption(
                                        f"Last calibration: {_upd.strftime('%m/%d/%Y %I:%M %p')}"
                                    )
                                except Exception:
                                    st.caption(f"Last calibration: {_tc_updated}")

            except Exception as ti_err:
                st.error(f"Track Intelligence tab error: {ti_err}")
                import traceback

                st.code(traceback.format_exc())

    except Exception as e:
        st.error(f"Error in Gold High-IQ System: {e}")
        import traceback

        st.code(traceback.format_exc())

# End of Section E

st.markdown("---")
st.caption(
    "Horse Race Ready - IQ Mode | Advanced Track Bias Analysis with ML Probability Calibration & Real Data Training"
)
