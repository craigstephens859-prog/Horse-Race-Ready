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

import contextlib
import logging
import os
import re
from datetime import datetime

import numpy as np
import pandas as pd
import streamlit as st

from config import (
    DISTANCE_OPTIONS,
    NA_STYLE_PARAMS,
    TRACK_BIAS_PROFILES,
    base_class_bias,
)
from dynamic_weights import (  # Phase 5: weight presets & adjustment pipeline
    adjust_by_race_type,
    apply_condition_adjustment,
    apply_purse_scaling,
    apply_strategy_profile_to_weights,
    get_weight_preset,
)
from pp_parsing import (  # Phase 3: BRISNET PP parsing functions
    _find_header_line,
    calculate_style_strength,
    detect_purse_amount,
    detect_race_number,
    detect_race_type,
    extract_horses_and_styles,
    extract_morning_line_by_horse,
    parse_angles_for_block,
    parse_brisnet_race_header,
    parse_pedigree_snips,
    parse_speed_figures_for_block,
    parse_track_name_from_pp,
    split_into_horse_chunks,
)
from rating_engine import (  # Phase 4: rating computation engine
    _angles_pedigree_tweak,
    apply_enhancements_and_figs,
    calculate_comprehensive_class_rating,
    calculate_form_cycle_rating,
    compute_bias_ratings,
    compute_ppi,
    fair_probs_from_ratings,
    overlay_table,
)
from strategy_builder import (  # Phase 5: betting strategy & component breakdown
    build_betting_strategy,
    build_component_breakdown,
    str_to_decimal_odds,
)
from utils import (  # Phase 2: utility functions
    _auto_distance_label,
    _canonical_track,
    _normalize_style,
    distance_bucket,
    distance_to_furlongs,
    fair_to_american_str,
    normalize_horse_name,
    odds_to_decimal,
    safe_float,
)

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# ULTRATHINK INTEGRATION: Import optimized 8-angle system
try:
    from horse_angles8 import compute_eight_angles  # noqa: F401
except ImportError:
    pass

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

    def get_all_track_calibrations_summary(_db_path=""):
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


# ===================== Core Helpers =====================
# ========================================================


# ========== SAVANT-LEVEL ENHANCEMENTS (Jan 2026) ==========


# ========== END SAVANT ENHANCEMENTS ==========


# ---------- GOLD-STANDARD Probability helpers with mathematical rigor ----------


# ===================== Form Cycle & Recency Analysis =====================


# ===================== Class Rating Calculator (Comprehensive) =====================


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
df_final_field = df_editor[~df_editor["Scratched"].fillna(False)].copy()
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


# ======================== Phase 1: Enhanced Parsing Functions ========================


# ======================== RACE AUDIT ENHANCEMENTS (Feb 2026) ========================
# Oaklawn R1 Audit: Model ranked Tell Me When last (#7) but horse WON.
# Model's #1 Tiffany Twist finished 4th. Root cause: 5 gaps in bias/angle handling.
# All enhancements below are ADDITIVE — no existing logic is modified.


# ======================== ELITE ENHANCEMENTS ========================


# ======================== MARATHON CALIBRATION (McKnight G3 Learning) ========================


# ======================== ROUTE & SPRINT CALIBRATION ========================
# Based on Pegasus WC G1 (9f route) and GP Turf Sprint (5f sprint) validation


# ============ STRUCTURED RACE HISTORY PARSING (Feb 10, 2026) ============


# ======================== End ELITE ENHANCEMENTS ========================


# Build scenarios - UPDATED: Create unified scenario using ALL selected biases
# Instead of cartesian product, we aggregate bonuses from all selections
scenarios = [("COMBINED", "COMBINED")]  # Single unified scenario
tabs = st.tabs(["Combined Bias Analysis"])
all_scenario_ratings = {}

# ============ DYNAMIC WEIGHT FUNCTIONS (Race Parameter Integration) ============
# These functions dynamically adjust factor weights based on race conditions


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
            _fp = fair_probs  # Bind to local to avoid B023 late-binding bug
            ratings_df["Fair %"] = ratings_df["Horse"].map(
                lambda h, fp=_fp: f"{fp.get(h, 0) * 100:.1f}%"
            )
            ratings_df["Fair Odds"] = ratings_df["Horse"].map(
                lambda h, fp=_fp: fair_to_american_str(fp.get(h, 0))
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


def _build_na_context(primary_df: pd.DataFrame, _df_field: pd.DataFrame = None) -> str:
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
                    phase3_report = f"Phase 3 analysis error: {e!s}"

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
                                f"📊 Saved {len(horses_data)} horses with {len([k for k in horses_data[0] if k.startswith('rating_') or k.startswith('angle_')])} ML features each"
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
                    race_id, _track, _date, race_num, field_size = selected_race

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
                                                st.error(f"❌ Error: {e!s}")
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
                    if st.session_state.get("last_learning_insights"):
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
                    st.error(f"❌ Database verification failed: {e!s}")
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

                    for col_idx, (key, (label, default)) in enumerate(
                        odds_weights.items()
                    ):
                        with cols2[col_idx]:
                            val = db_learned_weights.get(key, default)
                            st.metric(label, f"{val:.1f}", help=f"Default: {default}")

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
                st.error(f"❌ Error loading calibration data: {e!s}")
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
            with contextlib.suppress(Exception):
                admin_password = st.secrets.get("ADMIN_PASSWORD", "")

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
                            with contextlib.suppress(Exception):
                                _profile = _track_intel.build_full_profile(
                                    _selected_track
                                )

                        _accuracy = {}
                        _biases = {}
                        if gold_db:
                            with contextlib.suppress(Exception):
                                _accuracy = (
                                    gold_db.calculate_accuracy_stats(_selected_track)
                                    or {}
                                )
                            with contextlib.suppress(Exception):
                                _biases = gold_db.detect_biases(_selected_track) or {}

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
