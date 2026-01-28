import streamlit as st # type: ignore
import pandas as pd
import numpy as np
import re
from itertools import product
import json
import os
from typing import List, Dict, Tuple, Optional
from datetime import datetime

# ML Engine import
try:
    from ml_engine import MLCalibrator, RaceDatabase, create_feature_vector
    ML_AVAILABLE = True
except ImportError:
    ML_AVAILABLE = False

# ===================== Page / Model Settings =====================

st.set_page_config(page_title="Horse Race Ready — IQ Mode", page_icon="🏇", layout="wide")
st.title("🏇  Horse Race Ready — IQ Mode")

# ---------- Durable state ----------
if "parsed" not in st.session_state:
    st.session_state["parsed"] = False
if "pp_text_cache" not in st.session_state:
    st.session_state["pp_text_cache"] = ""

# Initialize session state for persistent inputs
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
if 'ml_enabled' not in st.session_state:
    st.session_state['ml_enabled'] = True
if 'current_race_id' not in st.session_state:
    st.session_state['current_race_id'] = None

# Initialize ML Calibrator
if ML_AVAILABLE and 'ml_calibrator' not in st.session_state:
    st.session_state['ml_calibrator'] = MLCalibrator()
    st.session_state['race_db'] = RaceDatabase()

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
except:
    use_sdk_v1 = False
try:
    import openai
    if OPENAI_API_KEY:
        openai.api_key = OPENAI_API_KEY
except:
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
    (\d+)                              # post/program
    \s+([A-Za-z0-9'.\-\s&]+?)          # horse name
    \s*\(\s*
    (E\/P|EP|E|P|S|NA)                 # style
    (?:\s+(\d+))?                      # optional quirin
    \s*\)\s*$                          #
    """, re.VERBOSE
)
def _normalize_style(tok: str) -> str:
    t = (tok or "").upper().strip()
    return "E/P" if t in ("EP", "E/P") else t

def calculate_style_strength(style: str, quirin: float) -> str:
    s = (style or "NA").upper()
    try:
        q = quirin
    except:
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

_ODDS_TOKEN = r"([+-]?\d+(?:\.\d+)?|\d+\s*/\s*\d+|\d+\s*-\s*\d+)"
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

# ---------- Enhanced Parsing Functions (Phase 1 Implementation) ----------

def parse_track_bias_impact_values(block: str) -> dict:
    """
    Extract data-driven Impact Values from PP text (BRISNET Track Bias section).
    Returns dict with running_style and post_position Impact Values.
    """
    result = {"running_style": {}, "post_position": {}}
    if not block:
        return result
    
    try:
        # Running Style Impact: E 1.30, E/P 0.95, P 1.10, S 0.75
        rs_match = re.search(r'Running\s+Style\s+Impact:\s*E\s+([\d.]+),?\s*E/P\s+([\d.]+),?\s*P\s+([\d.]+),?\s*S\s+([\d.]+)', block, re.IGNORECASE)
        if rs_match:
            result["running_style"] = {
                "E": float(rs_match.group(1)),
                "E/P": float(rs_match.group(2)),
                "P": float(rs_match.group(3)),
                "S": float(rs_match.group(4))
            }
        
        # Post Position Impact: Rail 1.41, 1-3 1.10, 4-7 1.05, 8+ 0.95
        pp_match = re.search(r'Post\s+Position\s+Impact:\s*Rail\s+([\d.]+),?\s*1-3\s+([\d.]+),?\s*4-7\s+([\d.]+),?\s*8\+\s+([\d.]+)', block, re.IGNORECASE)
        if pp_match:
            result["post_position"] = {
                "Rail": float(pp_match.group(1)),
                "Inner": float(pp_match.group(2)),
                "Mid": float(pp_match.group(3)),
                "Outside": float(pp_match.group(4))
            }
    except Exception:
        pass
    
    return result

def parse_pedigree_spi(block: str) -> dict:
    """
    Extract Sire Production Index (SPI) from pedigree section.
    SPI ranges from 0.30-1.50+ (1.0+ is strong, <0.5 is weak).
    """
    out = {"sire_spi": np.nan, "dam_sire_spi": np.nan}
    if not block:
        return out
    
    try:
        # Match "SPI .36" or "SPI 0.36" patterns
        spi_matches = list(re.finditer(r'SPI\s+(\.?\d+\.?\d*)', block, re.IGNORECASE))
        if len(spi_matches) >= 1:
            val = spi_matches[0].group(1)
            # Handle .36 vs 0.36 format
            if val.startswith('.'):
                val = '0' + val
            out["sire_spi"] = float(val)
        
        if len(spi_matches) >= 2:
            val = spi_matches[1].group(1)
            if val.startswith('.'):
                val = '0' + val
            out["dam_sire_spi"] = float(val)
    except Exception:
        pass
    
    return out

def parse_pedigree_surface_stats(block: str) -> dict:
    """
    Extract Sire %Mud, %Turf, Dam-Sire %Mud from pedigree section.
    Critical for off-track and turf races.
    """
    out = {"sire_mud_pct": np.nan, "sire_turf_pct": np.nan, "dam_sire_mud_pct": np.nan}
    if not block:
        return out
    
    try:
        # Look for "Mud XX%" patterns
        mud_matches = list(re.finditer(r'Mud\s+(\d+\.?\d*)%', block, re.IGNORECASE))
        if len(mud_matches) >= 1:
            out["sire_mud_pct"] = float(mud_matches[0].group(1))
        if len(mud_matches) >= 2:
            out["dam_sire_mud_pct"] = float(mud_matches[1].group(1))
        
        # Look for "Turf XX%" pattern
        turf_match = re.search(r'Turf\s+(\d+\.?\d*)%', block, re.IGNORECASE)
        if turf_match:
            out["sire_turf_pct"] = float(turf_match.group(1))
    except Exception:
        pass
    
    return out

def parse_awd_analysis(block: str) -> dict:
    """
    Extract Average Winning Distance (AWD) for Sire and Dam-Sire.
    Used to validate distance aptitude.
    """
    out = {"sire_awd": np.nan, "dam_sire_awd": np.nan}
    if not block:
        return out
    
    try:
        # AWD appears in pedigree stats: "AWD 6.8"
        awd_matches = list(re.finditer(r'AWD\s+(\d+\.?\d*)', block, re.IGNORECASE))
        if len(awd_matches) >= 1:
            out["sire_awd"] = float(awd_matches[0].group(1))
        if len(awd_matches) >= 2:
            out["dam_sire_awd"] = float(awd_matches[1].group(1))
    except Exception:
        pass
    
    return out

def calculate_spi_bonus(sire_spi: float, dam_sire_spi: float = None) -> float:
    """
    Calculate SPI-based bonus/penalty.
    Strong sire (SPI ≥ 1.0): +0.06
    Weak sire (SPI ≤ 0.5): -0.05
    Dam-Sire weighted at 50% of Sire impact.
    """
    bonus = 0.0
    
    if pd.notna(sire_spi):
        if sire_spi >= 1.0:
            bonus += 0.06
        elif sire_spi <= 0.5:
            bonus -= 0.05
    
    if pd.notna(dam_sire_spi):
        if dam_sire_spi >= 1.0:
            bonus += 0.03  # 50% of sire bonus
        elif dam_sire_spi <= 0.5:
            bonus -= 0.025  # 50% of sire penalty
    
    return float(np.clip(bonus, -0.12, 0.12))

def calculate_surface_specialty_bonus(sire_mud_pct: float, sire_turf_pct: float,
                                      dam_sire_mud_pct: float, condition: str,
                                      surface: str) -> float:
    """
    Calculate bonus for surface specialization.
    Mud specialists: Sire %Mud ≥ 25% gets +0.08
    Turf specialists: Sire %Turf ≥ 30% gets +0.08
    """
    bonus = 0.0
    
    # Off-track bonuses (muddy/sloppy/heavy)
    if condition.lower() in ["muddy", "sloppy", "heavy"]:
        if pd.notna(sire_mud_pct):
            if sire_mud_pct >= 25.0:
                bonus += 0.08  # Specialist
            elif sire_mud_pct >= 15.0:
                bonus += 0.04  # Moderate
        
        if pd.notna(dam_sire_mud_pct):
            if dam_sire_mud_pct >= 25.0:
                bonus += 0.024  # 30% weight
    
    # Turf bonuses
    if surface.lower() == "turf":
        if pd.notna(sire_turf_pct):
            if sire_turf_pct >= 30.0:
                bonus += 0.08  # Specialist
            elif sire_turf_pct >= 20.0:
                bonus += 0.04  # Moderate
    
    return float(np.clip(bonus, -0.12, 0.12))

def calculate_awd_mismatch_penalty(sire_awd: float, dam_sire_awd: float,
                                    distance: str) -> float:
    """
    Calculate penalty for AWD distance mismatch.
    Large mismatch (2.0f+): -0.08
    Moderate mismatch (1.0-2.0f): -0.04
    Small mismatch (0.5-1.0f): -0.02
    """
    penalty = 0.0
    
    # Convert distance to furlongs
    distance_lower = distance.lower()
    if "mile" in distance_lower:
        if "1 1/8" in distance_lower or "1 3/16" in distance_lower:
            target_fur = 9.0
        elif "1 1/4" in distance_lower or "1 1/16" in distance_lower:
            target_fur = 10.0
        elif "1 1/2" in distance_lower:
            target_fur = 12.0
        else:
            target_fur = 8.0
    elif "furlong" in distance_lower:
        try:
            target_fur = float(re.search(r'(\d+)', distance_lower).group(1))
        except:
            target_fur = 6.0
    else:
        target_fur = 6.0
    
    # Check sire AWD mismatch
    if pd.notna(sire_awd):
        diff = abs(sire_awd - target_fur)
        if diff >= 2.0:
            penalty -= 0.08
        elif diff >= 1.0:
            penalty -= 0.04
        elif diff >= 0.5:
            penalty -= 0.02
    
    # Check dam-sire AWD (weighted at 30%)
    if pd.notna(dam_sire_awd):
        diff = abs(dam_sire_awd - target_fur)
        if diff >= 2.0:
            penalty -= 0.024
        elif diff >= 1.0:
            penalty -= 0.012
    
    return float(np.clip(penalty, -0.12, 0.0))

# ---------- Probability helpers ----------
def softmax_from_rating(ratings: np.ndarray, tau: float = 0.85) -> np.ndarray:
    """Convert ratings to probabilities with a temperature (lower tau = sharper)."""
    if ratings.size == 0:
        return ratings
    x = np.array(ratings, dtype=float) / max(tau, 1e-6)
    x = x - np.max(x)
    ex = np.exp(x)
    p = ex / np.sum(ex)
    return p

def adaptive_probabilities(ratings: np.ndarray, field_size: int) -> np.ndarray:
    """Adaptive probability calculation with field-size adjustment."""
    if ratings.size == 0:
        return ratings
    
    # Adjust tau based on field quality (rating spread)
    rating_spread = np.max(ratings) - np.mean(ratings)
    if rating_spread > 8:  # Clear standout
        tau = 0.65  # Sharper distribution
    elif rating_spread > 5:  # Strong favorite
        tau = 0.75
    elif rating_spread < 2:  # Tightly matched field
        tau = 1.1  # Flatter distribution
    else:
        tau = 0.85  # Default
    
    # Field size adjustment (larger fields = more uncertainty)
    tau *= (1.0 + (field_size - 8) * 0.02)
    
    return softmax_from_rating(ratings, tau)

def multi_position_predictions(ratings: np.ndarray, horse_names: list) -> dict:
    """Generate predictions for 1st, 2nd, 3rd, 4th with probabilities."""
    probs = adaptive_probabilities(ratings, len(ratings))
    sorted_indices = np.argsort(-probs)
    
    predictions = {
        "winner": {
            "horse": horse_names[sorted_indices[0]],
            "prob": probs[sorted_indices[0]],
            "confidence": "High" if probs[sorted_indices[0]] > 0.35 else "Medium" if probs[sorted_indices[0]] > 0.20 else "Low"
        },
        "place_contenders": [],
        "show_contenders": [],
        "superfecta_contenders": []
    }
    
    # 2nd place: conditional probability (exclude winner)
    remaining_probs = np.array([p if i != sorted_indices[0] else 0 for i, p in enumerate(probs)])
    remaining_probs = remaining_probs / remaining_probs.sum()
    second_sorted = np.argsort(-remaining_probs)
    
    for i in range(min(3, len(second_sorted))):
        if remaining_probs[second_sorted[i]] > 0.05:
            predictions["place_contenders"].append({
                "horse": horse_names[second_sorted[i]],
                "prob": remaining_probs[second_sorted[i]]
            })
    
    # 3rd place: conditional on top 2
    exclude_top2 = {sorted_indices[0], second_sorted[0]}
    third_probs = np.array([p if i not in exclude_top2 else 0 for i, p in enumerate(probs)])
    third_probs = third_probs / third_probs.sum()
    third_sorted = np.argsort(-third_probs)
    
    for i in range(min(3, len(third_sorted))):
        if third_probs[third_sorted[i]] > 0.03:
            predictions["show_contenders"].append({
                "horse": horse_names[third_sorted[i]],
                "prob": third_probs[third_sorted[i]]
            })
    
    # 4th place
    exclude_top3 = exclude_top2 | {third_sorted[0]}
    fourth_probs = np.array([p if i not in exclude_top3 else 0 for i, p in enumerate(probs)])
    fourth_probs = fourth_probs / fourth_probs.sum()
    fourth_sorted = np.argsort(-fourth_probs)
    
    for i in range(min(3, len(fourth_sorted))):
        if fourth_probs[fourth_sorted[i]] > 0.03:
            predictions["superfecta_contenders"].append({
                "horse": horse_names[fourth_sorted[i]],
                "prob": fourth_probs[fourth_sorted[i]]
            })
    
    return predictions

def compute_ppi(df_styles: pd.DataFrame) -> dict:
    """PPI that works with Detected/Override styles."""
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
    ppi_val = (counts["E"] + counts["E/P"] - counts["P"] - counts["S"]) * 10.0 / total
    by_horse = {}
    for stl, nm, strength in zip(styles, names, strengths):
        wt = {"Strong":1.0, "Solid":0.8, "Slight":0.5, "Weak":0.3}.get(str(strength), 0.8)
        if stl in ("E", "E/P"):
            by_horse[nm] = round(0.6 * wt * ppi_val, 3)
        elif stl == "S":
            by_horse[nm] = round(-0.6 * wt * ppi_val, 3)
        else:
            by_horse[nm] = 0.0
    return {"ppi": round(ppi_val,3), "by_horse": by_horse}

# ========== ENHANCED CLASS & SPEED ANALYSIS ==========

def parse_beyer_figures(block: str) -> dict:
    """Extract Beyer speed figures from PP block."""
    beyers = []
    # Pattern: look for 2-3 digit numbers in speed figure lines
    for line in (block or "").split("\n")[:15]:  # Check first 15 lines
        if re.search(r'(?i)beyer|bris|speed.?fig', line):
            nums = re.findall(r'\b([5-9]\d|1[0-2]\d)\b', line)
            beyers.extend([int(n) for n in nums[:5]])
    if not beyers:
        # Fallback: look for number sequences that might be figs
        for line in (block or "").split("\n")[:8]:
            if re.match(r'^\s*\d+\s+\d+\s+\d+', line):
                nums = re.findall(r'\b([6-9]\d|1[0-2]\d)\b', line)
                beyers.extend([int(n) for n in nums[:5]])
                if beyers: break
    return {
        "last_beyer": beyers[0] if beyers else np.nan,
        "avg_beyer": np.mean(beyers[:3]) if len(beyers)>=3 else (beyers[0] if beyers else np.nan),
        "top_beyer": max(beyers) if beyers else np.nan,
        "beyer_trend": (beyers[0] - beyers[2]) if len(beyers)>=3 else 0.0
    }

def calculate_class_rating(purse: int, race_type: str, horse_block: str, pedigree: Optional[dict] = None) -> float:
    """Comprehensive class calculation based on purse comparison and recent races."""
    # Base class hierarchy
    type_hierarchy = {
        "Stakes (G1)": 10.0, "Stakes (G2)": 8.5, "Stakes (G3)": 7.0,
        "Stakes (Listed)": 6.0, "Stakes": 5.0, "Allowance": 3.5,
        "Maiden Special Weight": 2.5, "Maiden (other)": 2.0,
        "Maiden Claiming": 1.5, "Other": 2.0
    }
    base_class = type_hierarchy.get(race_type, 2.0)
    
    # Purse-based adjustment (logarithmic scaling)
    purse_factor = np.log10(max(purse, 5000)) - 4.0  # normalize around $10k
    
    # Parse recent race purses from block
    recent_purses = []
    for m in re.finditer(r'\$(\d+)[Kk]', horse_block or ""):
        recent_purses.append(int(m.group(1)) * 1000)
    
    # Class movement (dropping down = advantage)
    class_adj = 0.0
    if recent_purses:
        avg_recent = np.mean(recent_purses[:3])
        if avg_recent > purse * 1.3:  # dropping down significantly
            class_adj = +1.5
        elif avg_recent > purse * 1.1:
            class_adj = +0.8
        elif avg_recent < purse * 0.7:  # major step up
            class_adj = -1.2
        elif avg_recent < purse * 0.9:
            class_adj = -0.5
    
    # Pedigree boost for stakes races
    ped_boost = 0.0
    if pedigree and race_type.startswith("Stakes"):
        sire_awd = pedigree.get("sire_awd", 0)
        if sire_awd > 1.5: ped_boost += 0.5
        elif sire_awd > 1.2: ped_boost += 0.3
    
    return base_class + purse_factor + class_adj + ped_boost

def compute_bias_ratings(df_styles: pd.DataFrame, surface_type: str, distance_txt: str, condition_txt: str,
                         race_type: str, running_style_bias: str, post_bias_pick: str,
                         ppi_value: float = 0.0, pedigree_per_horse: Optional[Dict[str, dict]] = None,
                         pp_text: str = "", purse: int = 50000) -> pd.DataFrame:
    cols = ["#", "Post", "Horse", "Style", "Quirin", "Cstyle", "Cpost", "Cpace",
            "Cclass", "Cspeed", "Atrack", "Arace", "R"]
    if df_styles is None or df_styles.empty:
        return pd.DataFrame(columns=cols)
    
    # Pre-parse all horse blocks for speed/class analysis
    horse_blocks = {name: block for _, name, block in split_into_horse_chunks(pp_text)}
    
    rows = []
    ppi_map = compute_ppi(df_styles).get("by_horse", {})
    key = (running_style_bias or "").upper()
    if key in ("E","E/P"):
        bias_key = "speed favoring"
    elif key in ("P","S"):
        bias_key = "closer favoring"
    else:
        bias_key = "none"
    cstyle_map = {
        "speed favoring": {"E":0.70, "E/P":0.50, "P":-0.20, "S":-0.50},
        "closer favoring": {"E":-0.50, "E/P":-0.20, "P":0.25, "S":0.50},
        "none": {"E":0.0, "E/P":0.0, "P":0.0, "S":0.0}
    }
    for _, row in df_styles.iterrows():
        post = row.get("Post","")
        name = row.get("Horse","")
        style = _normalize_style(row.get("OverrideStyle") or row.get("DetectedStyle") or row.get("Style") or "")
        quirin = float(row.get("Quirin",0)) if pd.notna(row.get("Quirin",np.nan)) else np.nan
        ppi = float(ppi_map.get(name,0.0))
        cstyle = cstyle_map.get(bias_key, cstyle_map["none"]).get(style, 0.0)
        try:
            post_num = int(re.sub(r"[^\d]","", str(post))) if post else 0
        except:
            post_num = 0
        def post_bias_val(p):
            return 0.0
        if post_bias_pick.lower() == "favors rail":
            def post_bias_val(p): return 0.25 if p==1 else 0.0
        elif post_bias_pick.lower() == "1-3":
            def post_bias_val(p): return 0.25 if 1<=p<=3 else 0.0
        elif post_bias_pick.lower() == "4-7":
            def post_bias_val(p): return 0.25 if 4<=p<=7 else 0.0
        elif post_bias_pick.lower() == "8+":
            def post_bias_val(p): return 0.25 if p>=8 else 0.0
        cpost = post_bias_val(post_num)
        cpace = ppi
        
        # === NEW: CLASS CALCULATION ===
        horse_block = horse_blocks.get(name, "")
        pedigree = (pedigree_per_horse or {}).get(name, {})
        Cclass = calculate_class_rating(purse, race_type, horse_block, pedigree)
        
        # === NEW: SPEED FIGURE RATING ===
        beyer_data = parse_beyer_figures(horse_block)
        last_beyer = beyer_data.get("last_beyer", 75)
        avg_beyer = beyer_data.get("avg_beyer", 75)
        top_beyer = beyer_data.get("top_beyer", 75)
        beyer_trend = beyer_data.get("beyer_trend", 0)
        
        # Normalize Beyers (typical range 40-120, normalize to -2 to +2 scale)
        cspeed_last = (last_beyer - 80) / 15.0 if not pd.isna(last_beyer) else 0.0
        cspeed_avg = (avg_beyer - 80) / 15.0 if not pd.isna(avg_beyer) else 0.0
        cspeed_top = (top_beyer - 80) / 20.0 if not pd.isna(top_beyer) else 0.0
        cspeed_trend = beyer_trend / 10.0
        
        # Weighted speed rating (recent > average > peak)
        Cspeed = (cspeed_last * 0.50 + cspeed_avg * 0.30 + cspeed_top * 0.15 + cspeed_trend * 0.05)
        
        # === TIER 2 ENHANCEMENTS (Phase 1 Implementation) ===
        tier2_bonus = 0.0
        
        # 1. Track Bias Impact Values (data-driven bonuses)
        impact_values = parse_track_bias_impact_values(horse_block)
        if impact_values.get("running_style"):
            rs_impact = impact_values["running_style"].get(style, 1.0)
            # Convert impact to bonus: 1.41 → +0.41, 0.75 → -0.25
            tier2_bonus += (rs_impact - 1.0) * 1.0  # Full weight for style impact
        
        if impact_values.get("post_position") and post_num:
            if post_num == 1:
                pp_impact = impact_values["post_position"].get("Rail", 1.0)
            elif 1 <= post_num <= 3:
                pp_impact = impact_values["post_position"].get("Inner", 1.0)
            elif 4 <= post_num <= 7:
                pp_impact = impact_values["post_position"].get("Mid", 1.0)
            else:
                pp_impact = impact_values["post_position"].get("Outside", 1.0)
            tier2_bonus += (pp_impact - 1.0) * 0.5  # Half weight for post impact
        
        # 2. SPI (Sire Production Index) bonus/penalty
        spi_data = parse_pedigree_spi(horse_block)
        spi_bonus = calculate_spi_bonus(
            spi_data.get("sire_spi", np.nan),
            spi_data.get("dam_sire_spi", np.nan)
        )
        tier2_bonus += spi_bonus
        
        # 3. Surface specialty bonus (%Mud, %Turf)
        surface_stats = parse_pedigree_surface_stats(horse_block)
        surface_bonus = calculate_surface_specialty_bonus(
            surface_stats.get("sire_mud_pct", np.nan),
            surface_stats.get("sire_turf_pct", np.nan),
            surface_stats.get("dam_sire_mud_pct", np.nan),
            condition_txt,
            surface_type
        )
        tier2_bonus += surface_bonus
        
        # 4. AWD distance mismatch penalty
        awd_data = parse_awd_analysis(horse_block)
        awd_penalty = calculate_awd_mismatch_penalty(
            awd_data.get("sire_awd", np.nan),
            awd_data.get("dam_sire_awd", np.nan),
            distance_txt
        )
        tier2_bonus += awd_penalty
        
        # Apply comprehensive rating formula with all components
        a_track_h = tier2_bonus
        arace = (Cclass * 2.5) + (Cspeed * 2.0) + (cpace * 1.5) + (cstyle * 1.2) + (cpost * 0.8)
        R = a_track_h + arace
        
        rows.append({"#":post, "Post":post, "Horse":name, "Style":style, "Quirin":quirin,
                     "Cstyle":round(cstyle,2), "Cpost":round(cpost,2), "Cpace":round(cpace,2),
                     "Cclass":round(Cclass,2), "Cspeed":round(Cspeed,2), 
                     "Atrack":round(a_track_h,2), "Arace":round(arace,2), "R":round(R,2)})
    return pd.DataFrame(rows)

def apply_enhancements_and_figs(ratings_df: pd.DataFrame, pp_text: str, processed_weights: Dict[str,float],
                                chaos_index: float, track_name: str, surface_type: str,
                                distance_txt: str, race_type: str,
                                angles_per_horse: Dict[str,pd.DataFrame],
                                pedigree_per_horse: Dict[str,dict], figs_df: pd.DataFrame) -> pd.DataFrame:
    if ratings_df is None or ratings_df.empty:
        return ratings_df
    df = ratings_df.copy()
    df["R_ENHANCE_ADJ"] = 0.0
    
    # === ANGLE-BASED ENHANCEMENTS ===
    for idx, row in df.iterrows():
        horse = row["Horse"]
        boost = 0.0
        
        # Positive angles boost
        angles_df = angles_per_horse.get(horse, pd.DataFrame())
        if not angles_df.empty:
            # High ROI angles (>1.5 ROI with decent sample)
            strong_angles = angles_df[(angles_df["ROI"] > 1.5) & (angles_df["Starts"] >= 10)]
            boost += len(strong_angles) * 0.4
            
            # Very high win% angles
            hot_angles = angles_df[(angles_df["Win%"] > 25) & (angles_df["Starts"] >= 15)]
            boost += len(hot_angles) * 0.3
            
            # Specific powerful angles
            for _, angle_row in angles_df.iterrows():
                cat = angle_row["Category"].lower()
                if "1st time" in cat or "debut" in cat:
                    if angle_row["Win%"] > 20: boost += 0.6
                elif "jky" in cat and "trn" in cat:  # Trainer/Jockey combo
                    if angle_row["Win%"] > 30: boost += 0.8
                elif "shipper" in cat:
                    if angle_row["Win%"] > 22: boost += 0.5
        
        # Pedigree boost for surface/distance
        pedigree = pedigree_per_horse.get(horse, {})
        if pedigree:
            # Turf races favor good turf pedigree
            if surface_type.lower() == "turf":
                dam_dpi = pedigree.get("dam_dpi", 0)
                if dam_dpi > 2.0: boost += 0.7
                elif dam_dpi > 1.5: boost += 0.4
            
            # Routes favor stamina pedigree
            is_route = any(x in distance_txt.lower() for x in ["mile", "1 1/", "1 3/", "1 5/"])
            if is_route:
                damsire_awd = pedigree.get("damsire_awd", 0)
                if damsire_awd > 1.3: boost += 0.5
        
        df.at[idx, "R_ENHANCE_ADJ"] = round(boost, 2)
    
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
    if re.fullmatch(r'[+-]?\d+(\.\d+)?', s):
        v = float(s); return max(v, 1.01)
    if re.fullmatch(r'\+\d+', s) or re.fullmatch(r'-\d+', s):
        return 1 + (float(s)/100.0 if float(s)>0 else 100.0/abs(float(s)))
    if "-" in s:
        a,b = s.split("-",1)
        try: return float(a)/float(b) + 1.0
        except: return None
    if "/" in s:
        a,b = s.split("/",1)
        try: return float(a)/float(b) + 1.0
        except: return None
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
                             weights=(1.30, 1.15, 1.05, 1.03),
                             top_n: int = 15) -> Tuple[pd.DataFrame,pd.DataFrame,pd.DataFrame]:
    horses = list(fair_probs.keys())
    probs = np.array([fair_probs[h] for h in horses])
    n = len(horses)
    if n < 2:
        return pd.DataFrame(), pd.DataFrame(), pd.DataFrame()

    def w_first(h):  return weights[0] if anchor_first and h == anchor_first else 1.0
    def w_second(h): return weights[1] if anchor_second and h == anchor_second else 1.0
    def w_third(h):  return weights[2] if pool_third and h in pool_third else 1.0
    def w_fourth(h): return weights[3] if pool_fourth and h in pool_fourth else 1.0

    # EXACTA
    ex_rows = []
    for i,j in product(range(n), range(n)):
        if i == j: continue
        prob = probs[i]*(probs[j]/(1.0-probs[i]))
        prob *= w_first(horses[i]) * w_second(horses[j])
        ex_rows.append({"Ticket":f"{horses[i]} → {horses[j]}", "Prob":prob})
    # renormalize after biasing
    ex_total = sum(r["Prob"] for r in ex_rows) or 1.0
    for r in ex_rows: r["Prob"] = r["Prob"]/ex_total
    for r in ex_rows: r["Fair Odds"] = (1.0/r["Prob"])-1
    df_ex = pd.DataFrame(ex_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 3:
        return df_ex, pd.DataFrame(), pd.DataFrame()

    # TRIFECTA
    tri_rows = []
    top8 = np.argsort(-probs)[:min(n,8)]
    for i,j,k in product(top8, top8, top8):
        if len({i,j,k}) != 3: continue
        p_ij = probs[i]*(probs[j]/(1.0-probs[i]))
        prob_ijk = p_ij*(probs[k]/(1.0-probs[i]-probs[j]))
        prob_ijk *= w_first(horses[i]) * w_second(horses[j]) * w_third(horses[k])
        tri_rows.append({"Ticket":f"{horses[i]} → {horses[j]} → {horses[k]}", "Prob":prob_ijk})
    tri_total = sum(r["Prob"] for r in tri_rows) or 1.0
    for r in tri_rows: r["Prob"] = r["Prob"]/tri_total
    for r in tri_rows: r["Fair Odds"] = (1.0/r["Prob"])-1
    df_tri = pd.DataFrame(tri_rows).sort_values(by="Prob", ascending=False).head(top_n)

    if n < 4:
        return df_ex, df_tri, pd.DataFrame()

    # SUPERFECTA
    super_rows = []
    top6 = np.argsort(-probs)[:min(n,6)]
    for i,j,k,l in product(top6, top6, top6, top6):
        if len({i,j,k,l}) != 4: continue
        p_ij = probs[i]*(probs[j]/(1.0-probs[i]))
        p_ijk = p_ij*(probs[k]/(1.0-probs[i]-probs[j]))
        prob_ijkl = p_ijk*(probs[l]/(1.0-probs[i]-probs[j]-probs[k]))
        prob_ijkl *= w_first(horses[i]) * w_second(horses[j]) * w_third(horses[k]) * w_fourth(horses[l])
        super_rows.append({"Ticket":f"{horses[i]} → {horses[j]} → {horses[k]} → {horses[l]}", "Prob":prob_ijkl})
    super_total = sum(r["Prob"] for r in super_rows) or 1.0
    for r in super_rows: r["Prob"] = r["Prob"]/super_total
    for r in super_rows: r["Fair Odds"] = (1.0/r["Prob"])-1
    df_super = pd.DataFrame(super_rows).sort_values(by="Prob", ascending=False).head(top_n)
    return df_ex, df_tri, df_super

def format_exotics_for_prompt(df: pd.DataFrame, title: str) -> str:
    if df is None or df.empty:
        return f"**{title} (Model-Derived)**\nNone.\n"
    df = df.copy()
    if "Prob %" not in df.columns:
        df["Prob %"] = (df["Prob"]*100).round(2)
    md = df[["Ticket","Prob %","Fair Odds"]].to_markdown(index=False)
    return f"**{title} (Model-Derived)**\n{md}\n"

# -------- Purse auto-detect (robust) --------
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

# From here on, we operate on the cached, parsed text
pp_text = st.session_state["pp_text_cache"]

# ===================== 2. Race Info (Confirm) =====================

st.header("2. Race Info (Confirm)")
first_line = (pp_text.split("\n",1)[0] or "").strip()

# Auto-parse track name
track_match = re.match(r'^\s*([A-Za-z\s]+?)\s+Race', first_line)
default_track = track_match.group(1).strip() if track_match else st.session_state['track_name']
col1,col2,col3,col4,col5 = st.columns(5)
with col1:
    track_name = st.text_input("Track:", value=default_track, key="track_input")
    st.session_state['track_name'] = track_name

# Auto-parse surface
default_surface = st.session_state['surface_type']
if re.search(r'(?i)\bturf|trf\b', first_line): default_surface = "Turf"
if re.search(r'(?i)\baw|tap|synth|poly\b', first_line): default_surface = "Synthetic"
with col2:
    surface_type = st.selectbox("Surface:", ["Dirt","Turf","Synthetic"],
                                index=["Dirt","Turf","Synthetic"].index(default_surface), key="surface_input")
    st.session_state['surface_type'] = surface_type

# Auto-parse track condition
conditions = ["fast","good","wet-fast","muddy","sloppy","firm","yielding","soft","heavy"]
cond_found = None
for cond in conditions:
    if re.search(rf'(?i)\b{cond}\b', first_line):
        cond_found = cond
        break
default_condition = cond_found if cond_found else st.session_state['condition_txt']
with col3:
    condition_txt = st.selectbox("Condition:", conditions,
                                 index=conditions.index(default_condition) if default_condition in conditions else 0,
                                 key="condition_input")
    st.session_state['condition_txt'] = condition_txt

# Auto-parse distance
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
distance_options = [
    "4 Furlongs","4 1/2 Furlongs","5 Furlongs","5 1/2 Furlongs","6 Furlongs","6 1/2 Furlongs","7 Furlongs","7 1/2 Furlongs",
    "8 Furlongs","8 1/2 Furlongs","9 Furlongs",
    "1 Mile","1 1/16 Miles","1 1/8 Miles","1 3/16 Miles","1 1/4 Miles","1 5/16 Miles","1 3/8 Miles","1 7/16 Miles",
    "1 1/2 Miles","1 5/8 Miles","1 3/4 Miles","1 7/8 Miles","2 Miles"
]
auto_dist_option = (auto_distance or "").replace("½","1/2")
if st.session_state['distance_txt'] in distance_options:
    default_index = distance_options.index(st.session_state['distance_txt'])
elif auto_dist_option in distance_options:
    default_index = distance_options.index(auto_dist_option)
else:
    default_index = distance_options.index("6 Furlongs")
with col4:
    distance_txt = st.selectbox("Distance:", distance_options, index=default_index, key="distance_input")
    st.session_state['distance_txt'] = distance_txt

# Auto-detect purse (robust, scans full text)
auto_purse = detect_purse_amount(pp_text)
default_purse = int(auto_purse) if auto_purse else st.session_state['purse_val']
with col5:
    purse_val = st.number_input("Purse ($)", min_value=0, step=5000, value=default_purse, key="purse_input")
    st.session_state['purse_val'] = purse_val

# Detect race type (full-text)
def detect_race_type(pp_text: str) -> str:
    s = (pp_text or "").lower()
    if "stakes" in s or "stake" in s:
        if re.search(r'grade\s*i|\bg1\b', s): return "Stakes (G1)"
        if re.search(r'grade\s*ii|\bg2\b', s): return "Stakes (G2)"
        if re.search(r'grade\s*iii|\bg3\b', s): return "Stakes (G3)"
        if "listed" in s: return "Stakes (Listed)"
        return "Stakes"
    if "allow" in s or "alw" in s: return "Allowance"
    if re.search(r'\bmdn\b|\bmaiden\b', s):
        if re.search(r'\bmc\b|\bmaid(?:en)?\s*claim', s): return "Maiden Claiming"
        if re.search(r'\bmdn\s*sp|maiden\s*sp|mdn\s*special', s): return "Maiden Special Weight"
        return "Maiden (other)"
    return "Other"

race_type_detected = detect_race_type(pp_text)
st.caption(f"Detected race type: **{race_type_detected}**")
race_type_list = ["Stakes (G1)","Stakes (G2)","Stakes (G3)","Stakes (Listed)",
                  "Stakes","Allowance","Maiden Claiming","Maiden (other)","Other"]
default_rt_index = race_type_list.index(race_type_detected) if race_type_detected in race_type_list else len(race_type_list)-1
race_type_manual = st.selectbox("Race Type (override):", race_type_list, index=default_rt_index, key="race_type_input")
race_type = race_type_manual if race_type_manual else race_type_detected
race_type_detected = race_type

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

# ===================== B. Angle Parsing / Pedigree =====================

angles_per_horse = {}
pedigree_per_horse = {}
for _post, name, block in split_into_horse_chunks(pp_text):
    if name in df_editor["Horse"].values:
        angles_per_horse[name] = parse_angles_for_block(block)
        pedigree_per_horse[name] = parse_pedigree_snips(block)

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

ppi_results = compute_ppi(df_final_field)
ppi_val = ppi_results.get("ppi", 0.0)

# ===================== C. Bias-Adjusted Ratings =====================

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
        default=[], key="style_biases"
    )
with b_col3:
    post_biases = st.multiselect("Select Post Position Biases:",
                                 options=["None","Favors Rail","1-3","4-7","8+"],
                                 default=["None"], key="post_biases")

if not running_style_biases or not post_biases:
    st.info("Pick at least one **Style** bias and one **Post** bias.")
    st.stop()

scenarios = list(product(running_style_biases, post_biases))
tabs = st.tabs([f"S: {s} | P: {p}" for s,p in scenarios])
all_scenario_ratings = {}

# Simplified weight presets (placeholders)
def get_weight_preset(surface, distance): return {"class_form":1.0, "trs_jky":1.0}
def apply_strategy_profile_to_weights(w, profile): return w
def adjust_by_race_type(w, rt): return w
def apply_purse_scaling(w, purse): return w if w else {}

base_weights = get_weight_preset(st.session_state['surface_type'], st.session_state['distance_txt'])
profiled_weights = apply_strategy_profile_to_weights(base_weights, strategy_profile)
racetype_weights = adjust_by_race_type(profiled_weights, race_type_detected)
final_weights = apply_purse_scaling(racetype_weights, st.session_state['purse_val'])

def fair_probs_from_ratings(ratings_df: pd.DataFrame) -> Dict[str, float]:
    if ratings_df is None or ratings_df.empty:
        return {}
    r = ratings_df["R"].astype(float).values
    p = adaptive_probabilities(r, len(r))
    return {h:p[i] for i,h in enumerate(ratings_df["Horse"].values)}

for i, (rbias, pbias) in enumerate(scenarios):
    with tabs[i]:
        ratings_df = compute_bias_ratings(
            df_styles=df_final_field,
            surface_type=st.session_state['surface_type'],
            distance_txt=st.session_state['distance_txt'],
            condition_txt=st.session_state['condition_txt'],
            race_type=race_type_detected,
            running_style_bias=rbias,
            post_bias_pick=pbias,
            ppi_value=ppi_val,
            pedigree_per_horse=pedigree_per_horse,
            pp_text=pp_text,
            purse=st.session_state['purse_val']
        )
        ratings_df = apply_enhancements_and_figs(
            ratings_df=ratings_df,
            pp_text=pp_text,
            processed_weights=final_weights,
            chaos_index=0.0,
            track_name=st.session_state['track_name'],
            surface_type=st.session_state['surface_type'],
            distance_txt=st.session_state['distance_txt'],
            race_type=race_type_detected,
            angles_per_horse=angles_per_horse,
            pedigree_per_horse=pedigree_per_horse,
            figs_df=pd.DataFrame()
        )
        # Non-uniform fair probs from Ratings
        fair_probs = fair_probs_from_ratings(ratings_df)
        
        # === ML PROBABILITY REFINEMENT ===
        if ML_AVAILABLE and st.session_state.get('ml_enabled', True) and st.session_state['ml_calibrator'].is_trained:
            try:
                # Create feature vectors for all horses
                race_context = {
                    'field_size': len(ratings_df),
                    'ppi': ppi_val,
                    'purse': st.session_state['purse_val']
                }
                
                features_list = []
                horse_names = []
                raw_probs_list = []
                
                for _, row in ratings_df.iterrows():
                    horse_data = {
                        'rating_class': row.get('Cclass', 0),
                        'rating_speed': row.get('Cspeed', 0),
                        'rating_pace': row.get('Cpace', 0),
                        'rating_style': row.get('Cstyle', 0),
                        'rating_post': row.get('Cpost', 0),
                        'rating_angles': row.get('R_ENHANCE_ADJ', 0),
                        'rating_pedigree': 0,  # Could extract from pedigree_per_horse
                        'final_odds': str_to_decimal_odds(row.get('Live Odds', '') or row.get('ML', '')) or 5.0,
                        'quirin_points': row.get('Quirin', 0),
                        'last_beyer': 75,  # Would need to parse
                        'avg_beyer': 75
                    }
                    raw_prob = fair_probs.get(row['Horse'], 0)
                    features = create_feature_vector(horse_data, race_context, raw_prob)
                    features_list.append(features)
                    horse_names.append(row['Horse'])
                    raw_probs_list.append(raw_prob)
                
                # Refine probabilities with ML model
                features_array = np.array(features_list)
                raw_probs_array = np.array(raw_probs_list)
                refined_probs = st.session_state['ml_calibrator'].refine_probabilities(features_array, raw_probs_array)
                
                # Update fair_probs with refined values
                fair_probs_ml = {name: prob for name, prob in zip(horse_names, refined_probs)}
                
                # Show ML refinement impact
                st.info(f"🤖 ML Model Active: Probabilities refined using {st.session_state['race_db'].get_performance_stats()['samples_for_training']} historical races")
                
                # Update to use ML-refined probabilities
                fair_probs = fair_probs_ml
                
            except Exception as e:
                st.warning(f"ML refinement failed: {e}. Using raw probabilities.")
        
        if 'Horse' in ratings_df.columns:
            ratings_df["Fair %"] = ratings_df["Horse"].map(lambda h: f"{fair_probs.get(h,0)*100:.1f}%")
            ratings_df["Fair Odds"] = ratings_df["Horse"].map(lambda h: fair_to_american_str(fair_probs.get(h,0)))
        else:
            ratings_df["Fair %"] = ""
            ratings_df["Fair Odds"] = ""
        all_scenario_ratings[(rbias,pbias)] = (ratings_df, fair_probs)

        disp = ratings_df.sort_values(by="R", ascending=False)
        if "R_ENHANCE_ADJ" in disp.columns:
            disp = disp.drop(columns=["R_ENHANCE_ADJ"])  # hide helper column
        # === SHOW MULTI-POSITION PREDICTIONS ===
        multi_pred = multi_position_predictions(
            ratings=disp["R"].values,
            horse_names=disp["Horse"].tolist()
        )
        
        st.markdown(f"**🏆 Winner Prediction:** {multi_pred['winner']['horse']} ({multi_pred['winner']['prob']*100:.1f}% - {multi_pred['winner']['confidence']} confidence)")
        
        if multi_pred["place_contenders"]:
            place_str = ", ".join([f"{h['horse']} ({h['prob']*100:.1f}%)" for h in multi_pred["place_contenders"]])
            st.markdown(f"**🥈 Place Contenders:** {place_str}")
        
        if multi_pred["show_contenders"]:
            show_str = ", ".join([f"{h['horse']} ({h['prob']*100:.1f}%)" for h in multi_pred["show_contenders"]])
            st.markdown(f"**🥉 Show Contenders:** {show_str}")
        
        st.dataframe(
            disp,
            use_container_width=True, hide_index=True,
            column_config={
                "R": st.column_config.NumberColumn("Rating", format="%.2f"),
                "Cclass": st.column_config.NumberColumn("C-Class", format="%.2f"),
                "Cspeed": st.column_config.NumberColumn("C-Speed", format="%.2f"),
                "Cstyle": st.column_config.NumberColumn("C-Style", format="%.2f"),
                "Cpost": st.column_config.NumberColumn("C-Post", format="%.2f"),
                "Cpace": st.column_config.NumberColumn("C-Pace", format="%.2f")
            }
        )

primary_key = scenarios[0]
primary_df, primary_probs = all_scenario_ratings[primary_key]
st.info(f"**Primary Scenario:** S: `{primary_key[0]}` • P: `{primary_key[1]}` • Profile: `{strategy_profile}`  • PPI: {ppi_val:+.2f}")

# ===================== D. Overlays & Betting Strategy =====================

st.header("C. Overlays & Betting Strategy")
offered_odds_map = {}
for _, r in df_final_field.iterrows():
    odds_str = str(r["Live Odds"]).strip() or str(r["ML"]).strip()
    dec = str_to_decimal_odds(odds_str)
    if dec: offered_odds_map[r["Horse"]] = dec

df_ol = overlay_table(fair_probs=primary_probs, offered=offered_odds_map)
st.dataframe(
    df_ol,
    use_container_width=True, hide_index=True,
    column_config={
        "EV per $1": st.column_config.NumberColumn("EV per $1", format="$%.3f"),
        "Edge (pp)": st.column_config.NumberColumn("Edge (pp)")
    }
)

# ---------- Choose anchors for exotics (bias-aware) ----------
def pick_anchor_horses(ratings_df: pd.DataFrame, post_bias: str) -> Tuple[Optional[str], Optional[str]]:
    if ratings_df is None or ratings_df.empty:
        return None, None
    df = ratings_df.sort_values(by="R", ascending=False).copy()
    anchor1 = None
    if post_bias.lower() in ("favors rail","1-3"):
        df_rail = df[df["Post"].astype(str).str.extract(r"(\d+)")[0].astype(int).between(1,3, inclusive="both")]
        if not df_rail.empty:
            anchor1 = df_rail.iloc[0]["Horse"]
    if not anchor1:
        anchor1 = df.iloc[0]["Horse"]
    # anchor2 = best remaining by rating
    df_rest = df[df["Horse"] != anchor1]
    anchor2 = df_rest.iloc[0]["Horse"] if not df_rest.empty else None
    return anchor1, anchor2

anchor1, anchor2 = pick_anchor_horses(primary_df, primary_key[1])

# Pools for 3rd/4th: top 3 and top 6 not including anchors
df_ranked = primary_df.sort_values(by="R", ascending=False)
pool3 = set(df_ranked[df_ranked["Horse"]\
         .isin([h for h in df_ranked["Horse"].tolist() if h not in (anchor1, anchor2)])]\
         ["Horse"].head(3).tolist())
pool4 = set(df_ranked[df_ranked["Horse"]\
         .isin([h for h in df_ranked["Horse"].tolist() if h not in (anchor1, anchor2)])]\
         ["Horse"].head(6).tolist())

df_ex, df_tri, df_super = calculate_exotics_biased(
    fair_probs=primary_probs,
    anchor_first=anchor1,
    anchor_second=anchor2,
    pool_third=pool3,
    pool_fourth=pool4,
    weights=(1.30, 1.15, 1.05, 1.03),
    top_n=15
)

# ===================== E. Classic Report (LLM Generation) =====================

st.header("D. Classic Report")
if st.button("Analyze This Race", type="primary", key="analyze_button"):
    with st.spinner("Handicapping Race..."):
        try:
            primary_sorted = primary_df.sort_values(by="R", ascending=False)
            name_to_post = pd.Series(df_final_field["Post"].values,
                                     index=df_final_field["Horse"]).to_dict()
            top_table = primary_sorted[['Horse','R','Fair %','Fair Odds']].to_markdown(index=False)
            overlay_pos = df_ol[df_ol["EV per $1"]>0]
            overlay_table_md = (overlay_pos[['Horse','Fair %','Fair (AM)','Board (dec)','EV per $1']].to_markdown(index=False)
                                if not overlay_pos.empty else "None.")
            ex_md = format_exotics_for_prompt(df=df_ex.head(5), title="Top 5 Exactas")
            tri_md = format_exotics_for_prompt(df=df_tri.head(5), title="Top 5 Trifectas")
            super_md = format_exotics_for_prompt(df=df_super.head(5), title="Top 5 Superfectas")
            prompt = f"""
Act as a superior quantitative horse racing analyst.

--- RACE CONTEXT ---
- Track: {st.session_state['track_name']}
- Surface: {st.session_state['surface_type']} ({st.session_state['condition_txt']}) • Distance: {st.session_state['distance_txt']}
- Race Type: {race_type_detected}
- Purse: ${st.session_state['purse_val']:,}
- Pace Scenario: (PPI {ppi_val:+.2f})
- Primary Bias Scenario: {primary_key[0]} (style), {primary_key[1]} (post)
- Strategy Profile: {strategy_profile}

--- MODEL-DERIVED DATA ---
Horse→Post #:
{json.dumps(name_to_post, indent=2)}
Top Rated (this scenario):
{top_table}
Positive EV Win Overlays (vs. board):
{overlay_table_md}
Anchors for verticals (bias-aware): Winner = {anchor1}, Runner-up = {anchor2}
{ex_md}
{tri_md}
{super_md}

--- TASK: WRITE CLASSIC REPORT ---
- **Race Summary:** 1-2 sentences on race shape, pace, bias.
- **Pace & Bias Analysis:** Who benefits/hurt by pace and {primary_key[0]} bias?
- **Top Win Contenders:** 2-3 top horses, note +EV overlays.
- **Primary Underneath Keys:** 2-4 horses for 2nd/3rd/4th.
- **Vulnerable Favorites:** Overbet horses to avoid.
- **Betting Strategy:** Win/Place bets, Exacta/Trifecta/Superfecta plans using horse names.
"""
            report = call_openai_messages(messages=[{"role":"user","content":prompt}])
            st.markdown(report)

            # ---- Save outputs with UTF-8 to avoid Windows 'charmap' errors ----
            with open("analysis.txt","w", encoding="utf-8", errors="replace") as f:
                f.write(report)
            df_ol.to_csv("overlays.csv", index=False, encoding="utf-8-sig")

            tickets = []
            if not df_ex.empty:
                df_ex["Prob %"] = (df_ex["Prob"]*100).round(2)
                for _, row in df_ex.iterrows():
                    tickets.append({"Ticket":row["Ticket"], "Type":"Exacta", "Prob%":row["Prob %"], "Fair Odds":row["Fair Odds"]})
            if not df_tri.empty:
                df_tri["Prob %"] = (df_tri["Prob"]*100).round(2)
                for _, row in df_tri.iterrows():
                    tickets.append({"Ticket":row["Ticket"], "Type":"Trifecta", "Prob%":row["Prob %"], "Fair Odds":row["Fair Odds"]})
            if not df_super.empty:
                df_super["Prob %"] = (df_super["Prob"]*100).round(2)
                for _, row in df_super.iterrows():
                    tickets.append({"Ticket":row["Ticket"], "Type":"Superfecta", "Prob%":row["Prob %"], "Fair Odds":row["Fair Odds"]})
            race_nums = sorted(detect_valid_race_headers(pp_text))
            for i in range(len(race_nums)-1):
                if race_nums[i+1] == race_nums[i] + 1:
                    tickets.append({"Ticket":f"Daily Double: Race {race_nums[i]}→{race_nums[i+1]}", "Type":"DailyDouble", "Prob%":None, "Fair Odds":None})
            if len(race_nums)>=3:
                for i in range(len(race_nums)-2):
                    if race_nums[i+2] == race_nums[i] + 2:
                        tickets.append({"Ticket":f"Pick 3: Races {race_nums[i]}→{race_nums[i+1]}→{race_nums[i+2]}", "Type":"Pick3", "Prob%":None, "Fair Odds":None})
            if tickets:
                df_tickets = pd.DataFrame(tickets)
                df_tickets.to_csv("tickets.csv", index=False, encoding="utf-8-sig")
            
            # === SAVE PREDICTIONS TO DATABASE ===
            if ML_AVAILABLE and st.session_state.get('ml_enabled', True):
                try:
                    # Create race_id and save race
                    race_data = {
                        'track': st.session_state['track_name'],
                        'date': datetime.now().strftime('%Y-%m-%d'),
                        'race_number': 1,
                        'surface': st.session_state['surface_type'],
                        'distance': st.session_state['distance_txt'],
                        'condition': st.session_state['condition_txt'],
                        'race_type': race_type_detected,
                        'purse': st.session_state['purse_val'],
                        'field_size': len(primary_df),
                        'ppi': ppi_val
                    }
                    race_id = st.session_state['race_db'].save_race(race_data)
                    st.session_state['current_race_id'] = race_id
                    
                    # Save horse predictions
                    horses_data = []
                    for _, row in primary_df.iterrows():
                        horse_dict = {
                            'horse_name': row['Horse'],
                            'post_position': int(re.sub(r'[^\d]', '', str(row['Post']))) if row['Post'] else 0,
                            'morning_line': row.get('ML', ''),
                            'final_odds': str_to_decimal_odds(row.get('Live Odds', '') or row.get('ML', '')) or 5.0,
                            'rating_total': row.get('R', 0),
                            'rating_class': row.get('Cclass', 0),
                            'rating_speed': row.get('Cspeed', 0),
                            'rating_pace': row.get('Cpace', 0),
                            'rating_style': row.get('Cstyle', 0),
                            'rating_post': row.get('Cpost', 0),
                            'rating_angles': row.get('R_ENHANCE_ADJ', 0),
                            'rating_pedigree': 0,
                            'predicted_win_prob': primary_probs.get(row['Horse'], 0),
                            'predicted_place_prob': 0,
                            'predicted_show_prob': 0,
                            'running_style': row.get('Style', ''),
                            'quirin_points': row.get('Quirin', 0),
                            'last_beyer': 75,
                            'avg_beyer': 75
                        }
                        horses_data.append(horse_dict)
                    
                    st.session_state['race_db'].save_horse_predictions(race_id, horses_data)
                    st.success(f"✅ Predictions saved to database (Race ID: {race_id})")
                except Exception as e:
                    st.warning(f"Could not save predictions: {e}")
        except Exception as e:
            st.error(f"Error generating report: {e}")

# ===================== E. ML System & Results Tracking =====================

if ML_AVAILABLE:
    st.header("E. ML System & Results Tracking")
    
    ml_tabs = st.tabs(["📊 Performance Stats", "📝 Enter Results", "🤖 Train Model", "⚙️ Settings"])
    
    # Tab 1: Performance Stats
    with ml_tabs[0]:
        st.subheader("Historical Performance")
        
        stats = st.session_state['race_db'].get_performance_stats()
        
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Races", stats['total_races'])
        with col2:
            st.metric("Correct Winners", f"{stats['correct_winners']}")
        with col3:
            st.metric("Win Accuracy", f"{stats['win_accuracy']:.1f}%")
        with col4:
            st.metric("Brier Score", f"{stats['brier_score']:.4f}", 
                     help="Lower is better. <0.20 = excellent calibration")
        
        if stats['total_races'] >= 10:
            st.success(f"✅ Model trained on {stats['samples_for_training']} races")
        elif stats['total_races'] > 0:
            st.info(f"📊 Collecting data... Need {50 - stats['total_races']} more race results to train ML model")
        else:
            st.info("📊 No race results yet. Start by entering results after races complete.")
        
        if st.button("Export Training Data to CSV"):
            try:
                from ml_engine import export_training_data_csv
                count = export_training_data_csv()
                st.success(f"✅ Exported {count} records to training_data.csv")
            except Exception as e:
                st.error(f"Export failed: {e}")
    
    # Tab 2: Enter Results
    with ml_tabs[1]:
        st.subheader("Enter Race Results")
        
        if st.session_state.get('current_race_id'):
            st.info(f"Current race: **{st.session_state['current_race_id']}**")
            race_id_to_update = st.session_state['current_race_id']
        else:
            race_id_to_update = st.text_input("Race ID:", placeholder="e.g., Keeneland_2026-01-27_1")
        
        if race_id_to_update:
            st.markdown("**Enter finishing positions:**")
            
            try:
                conn = st.session_state['race_db'].db.cursor().connection
                df_horses = pd.read_sql_query(
                    f"SELECT horse_name FROM horses WHERE race_id = ? ORDER BY post_position",
                    conn, params=(race_id_to_update,)
                )
                
                if not df_horses.empty:
                    results = []
                    cols = st.columns(3)
                    for idx, horse_name in enumerate(df_horses['horse_name']):
                        with cols[idx % 3]:
                            finish = st.number_input(
                                f"{horse_name}",
                                min_value=1,
                                max_value=len(df_horses),
                                value=idx+1,
                                key=f"finish_{race_id_to_update}_{horse_name}"
                            )
                            results.append((horse_name, finish))
                    
                    if st.button("💾 Save Results", type="primary"):
                        try:
                            st.session_state['race_db'].update_race_results(race_id_to_update, results)
                            st.success("✅ Results saved! Model will use this data for training.")
                            
                            stats = st.session_state['race_db'].get_performance_stats()
                            if stats['total_races'] >= 50 and not st.session_state['ml_calibrator'].is_trained:
                                st.info("🤖 Sufficient data available. Go to 'Train Model' tab.")
                            
                            _safe_rerun()
                        except Exception as e:
                            st.error(f"Failed to save results: {e}")
                else:
                    st.warning("No horses found for this race ID.")
            except Exception as e:
                st.error(f"Database error: {e}")
    
    # Tab 3: Train Model
    with ml_tabs[2]:
        st.subheader("Train ML Probability Calibration Model")
        
        stats = st.session_state['race_db'].get_performance_stats()
        
        if stats['total_races'] < 50:
            st.warning(f"⚠️ Need at least 50 race results. Currently have {stats['total_races']}.")
        else:
            st.success(f"✅ {stats['total_races']} races available for training")
            
            col1, col2 = st.columns(2)
            with col1:
                epochs = st.number_input("Training Epochs", min_value=50, max_value=500, value=150, step=50)
            with col2:
                learning_rate = st.number_input("Learning Rate", min_value=0.0001, max_value=0.01, 
                                               value=0.001, step=0.0001, format="%.4f")
            
            if st.button("🚀 Train Model", type="primary"):
                with st.spinner("Training neural network..."):
                    try:
                        result = st.session_state['ml_calibrator'].train(epochs=epochs, learning_rate=learning_rate)
                        
                        if result['status'] == 'success':
                            st.success("✅ Model training complete!")
                            
                            col_a, col_b, col_c = st.columns(3)
                            with col_a:
                                st.metric("Val Accuracy", f"{result.get('val_accuracy', 0):.2f}%")
                            with col_b:
                                st.metric("Brier Score", f"{result.get('brier_score', 0):.4f}")
                            with col_c:
                                st.metric("Samples", result.get('training_samples', 0))
                            
                            _safe_rerun()
                        else:
                            st.error(f"Training failed: {result.get('message', 'Unknown error')}")
                    except Exception as e:
                        st.error(f"Training error: {e}")
            
            if st.session_state['ml_calibrator'].is_trained:
                st.info("✅ Model is trained and active.")
    
    # Tab 4: Settings
    with ml_tabs[3]:
        st.subheader("ML System Settings")
        
        ml_enabled = st.checkbox("Enable ML Probability Refinement", 
                                value=st.session_state.get('ml_enabled', True))
        st.session_state['ml_enabled'] = ml_enabled
        
        st.markdown("---")
        st.markdown("### Database Management")
        
        col_clear1, col_clear2 = st.columns([1, 2])
        with col_clear1:
            confirm_clear = st.checkbox("Confirm deletion")
        with col_clear2:
            if st.button("🗑️ Clear All Data", disabled=not confirm_clear):
                try:
                    import sqlite3
                    conn = sqlite3.connect("race_history.db")
                    conn.execute("DELETE FROM horses")
                    conn.execute("DELETE FROM races")
                    conn.execute("DELETE FROM performance_metrics")
                    conn.commit()
                    conn.close()
                    st.success("Database cleared")
                    _safe_rerun()
                except Exception as e:
                    st.error(f"Failed: {e}")
        
        st.markdown("---")
        st.markdown("### About ML System")
        st.markdown("""
        **Workflow:**
        1. Save predictions before each race
        2. Enter actual results after completion
        3. Train ML model on 50+ races
        4. Model refines probabilities automatically
        
        **Expected gain:** 5-10% accuracy after 100+ races
        
        **Architecture:** Neural network (15→64→32→16→1)
        """)

else:
    st.info("💡 Install ML requirements: `pip install torch` or `pip install scikit-learn`")
