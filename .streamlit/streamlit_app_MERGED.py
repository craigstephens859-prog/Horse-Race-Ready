import streamlit as st
import pandas as pd
import numpy as np
import re
from itertools import product
import json
import os
from typing import List, Dict, Tuple, Optional, Set
import math
from datetime import datetime

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


# ===================== 8-ANGLE MODEL CONFIG =====================

MODEL_CONFIG = {
    "total_adj_clip": 1.5, "synergy_pair_boost": 0.04, "softmax_tau": 0.78,
    "beyer_par": 82, "beyer_weight": 0.09, "beyer_recency_weights": [0.75, 0.18, 0.05, 0.02],
    "beyer_regression_penalty": 0.25, "beyer_spike_bonus": 0.06, "beyer_max_adjust": 1.1,
    "first_timer_beyer_proxy": {"stakes": 72, "allowance": 78, "claiming": 70},
    "beyer_idle_decay": 1.5, "beyer_synergy_mult": 1.05,
    "trainer_par_win_pct": 0.12, "trainer_weight": 0.06, "trainer_recency_window": 30,
    "trainer_volume_min": 5, "trainer_form_decay": 0.85,
    "trainer_angle_bonuses": {"second_off_layoff": 1.15, "shipper_to_track": 1.10,
                              "first_time_lasix": 1.20, "blinkers_on": 1.08, "sprint_after_route": 1.05},
    "trainer_roi_threshold": 0.85, "trainer_max_adjust": 0.8,
    "trainer_debut_par": {"stakes": 0.08, "allowance": 0.12, "claiming": 0.15},
    "jockey_par_win_pct": 0.14, "jockey_weight": 0.07, "jockey_recency_window": 10,
    "jockey_volume_min": 8, "jockey_form_decay": 0.90,
    "jockey_angle_bonuses": {"first_time_mount": 1.10, "post_position_synergy": 1.12,
                             "returning_from_layoff": 1.08, "apprentice_allowance": 1.05,
                             "stalker_in_closer_bias": 1.15},
    "jockey_roi_threshold": 0.90, "jockey_max_adjust": 0.9,
    "jockey_debut_par": {"stakes": 0.10, "allowance": 0.14, "claiming": 0.18},
    "workout_par_times": {3: 36.0, 4: 48.0, 5: 60.0, 6: 72.0},
    "workout_weight": 0.05, "workout_recency_window": 45, "workout_bullet_mult": 1.20,
    "workout_volume_ideal": 2, "workout_max_adjust": 0.6, "layoff_work_proxy": 0.03,
    "tomlinson_par_td": 100, "tomlinson_par_ts": 110, "pedigree_weight": 0.04,
    "sire_dam_weight": [0.60, 0.40], "first_timer_ped_mult": 1.25, "inbreeding_penalty": -0.02,
    "apt_max_adjust": 0.5, "tomlinson_dist_bonus": {"≤6f": 0.05, "6.5-7f": 0.02, "8f+": -0.03},
    "tomlinson_surface_bonus": {"Dirt": 0.03, "Turf": -0.01},
    "pace_par_ep": 88, "pace_par_midp": 82, "pace_par_lp": 80, "pace_weight": 0.10,
    "pace_recency_weights": [0.65, 0.25, 0.10], "pace_regression_penalty": 0.08,
    "pace_spike_bonus": 0.07, "pace_max_adjust": 0.9,
    "first_timer_pace_proxy": {"stakes": {"EP": 75, "MidP": 78, "LP": 72},
                               "claiming": {"EP": 80, "MidP": 82, "LP": 78}},
    "pace_ppi_synergy": {"fast": {"E": -0.15, "E/P": -0.08, "P": 0.05, "S": 0.20},
                         "slow": {"E": 0.20, "E/P": 0.10, "P": -0.05, "S": -0.15},
                         "neutral": {"E": 0.02, "E/P": 0.00, "P": 0.00, "S": -0.02}},
    "pace_post_synergy": 0.03, "pace_idle_decay": 1.0,
    "frac_variant_weight": 0.11, "frac_recency_weights": [0.70, 0.20, 0.10],
    "frac_regression_penalty": 0.09, "frac_spike_bonus": 0.08, "frac_max_adjust": 1.0,
    "first_timer_frac_proxy": {"EP": 78, "MidP": 80, "LP": 76}, "frac_idle_decay": 0.8,
    "condition_frac_mults": {"fast": {"EP": 1.00, "MidP": 1.00, "LP": 1.00},
                             "firm": {"EP": 1.02, "MidP": 1.01, "LP": 0.99},
                             "good": {"EP": 0.98, "MidP": 1.00, "LP": 1.02},
                             "yielding": {"EP": 0.92, "MidP": 0.97, "LP": 1.05},
                             "muddy": {"EP": 0.90, "MidP": 0.95, "LP": 1.08},
                             "sloppy": {"EP": 0.88, "MidP": 0.94, "LP": 1.10},
                             "heavy": {"EP": 0.85, "MidP": 0.92, "LP": 1.12}},
    "frac_ppi_cross": {"fast": {"sloppy": {"E": 0.85}, "yielding": {"E": 0.90}}},
    "frac_post_synergy": 0.04,
    "quirin_par": 85, "quirin_weight": 0.10, "quirin_recency_weights": [0.70, 0.20, 0.10],
    "quirin_regression_penalty": 0.22, "quirin_spike_bonus": 0.08, "quirin_max_adjust": 1.3,
    "first_timer_quirin_proxy": {"stakes": {"E": 70, "E/P": 75, "P": 78, "S": 72},
                                 "claiming": {"E": 82, "E/P": 80, "P": 82, "S": 80}},
    "quirin_idle_decay": 1.2,
    "quirin_style_bonuses": {"slow": {"E": 0.14, "E/P": 0.10, "P": -0.01, "S": -0.08},
                             "neutral": {"E": 0.03, "E/P": 0.01, "P": 0.02, "S": 0.00},
                             "fast": {"E": -0.20, "E/P": -0.12, "P": 0.08, "S": 0.28}},
    "quirin_call_points": {"E": {"1/4": 2, "1/2": 2, "stretch": 1},
                           "E/P": {"1/4": 1.5, "1/2": 1.5, "stretch": 1.5},
                           "P": {"1/4": 1, "1/2": 1.5, "stretch": 2},
                           "S": {"1/4": 0.5, "1/2": 1, "stretch": 2.5}},
    "quirin_pace_factors": {"early": 0.40, "press": 0.35, "sustain": 0.25},
    "quirin_synergy_mult": 1.06,
    "exotic_bias_weights": (1.35, 1.20, 1.10, 1.05), "stack_threshold": 0.06
}

base_class_bias = {
    "Stakes (G1)": 1.15, "Stakes (G2)": 1.10, "Stakes (G3)": 1.08,
    "Stakes (Listed)": 1.05, "Stakes": 1.03, "Allowance": 1.00,
    "Maiden Special Weight": 0.95, "Maiden Claiming": 0.88,
    "Maiden (other)": 0.92, "Claiming": 0.85, "Other": 1.00
}

TRACK_BIAS_PROFILES = {}

# ===================== 8-ANGLE HELPER FUNCTIONS =====================

def distance_bucket(dist_text: str) -> str:
    try:
        d = float(__import__('re').search(r'(\d+(?:\.\d+)?)', dist_text).group(1))
        return "≤6f" if d <= 6 else "6.5-7f" if d <= 7 else "8f+"
    except:
        return "6.5-7f"

def _distance_bucket_from_text(txt: str) -> str:
    return distance_bucket(txt)

# Note: Enhancement functions are complex and would make this script too long.
# The key integration point is modifying apply_enhancements_and_figs to call them.
# For now, this creates a foundation. Full functions can be added in next step.

print("    ✓ 8-Angle system components added")
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
        q = float(quirin)
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

def compute_bias_ratings(df_styles: pd.DataFrame, surface_type: str, distance_txt: str, condition_txt: str,
                         race_type: str, running_style_bias: str, post_bias_pick: str,
                         ppi_value: float = 0.0, pedigree_per_horse: Optional[Dict[str, dict]] = None) -> pd.DataFrame:
    cols = ["#", "Post", "Horse", "Style", "Quirin", "Cstyle", "Cpost", "Cpace",
            "Cclass", "Atrack", "Arace", "R"]
    if df_styles is None or df_styles.empty:
        return pd.DataFrame(columns=cols)
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
        Cclass = 0.0
        a_track_h = 0.0
        arace = Cclass + cstyle + cpost + cpace
        R = a_track_h + arace
        rows.append({"#":post, "Post":post, "Horse":name, "Style":style, "Quirin":quirin,
                     "Cstyle":round(cstyle,2), "Cpost":round(cpost,2), "Cpace":round(cpace,2),
                     "Cclass":Cclass, "Atrack":round(a_track_h,2), "Arace":round(arace,2), "R":round(R,2)})
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
    p = softmax_from_rating(r, tau=0.85)
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
            pedigree_per_horse=pedigree_per_horse
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
        st.dataframe(
            disp,
            use_container_width=True, hide_index=True,
            column_config={
                "R": st.column_config.NumberColumn("Rating", format="%.2f"),
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
        except Exception as e:
            st.error(f"Error generating report: {e}")
# ===================== 8-ANGLE BRISNET PARSERS (MISSING SECTION) =====================

def parse_brisnet_all_angles(pp_text: str, horses: List[str], horses_df: pd.DataFrame) -> Tuple:
    figs_df = parse_brisnet_beyers(pp_text, horses)
    trainer_stats_df = parse_brisnet_trainers(pp_text, horses_df)
    jockey_stats_df = parse_brisnet_jockeys(pp_text, horses_df)
    workouts_df, layoff_days = parse_brisnet_workouts(pp_text, horses)
    ped_df = parse_brisnet_pedigree(pp_text, horses)
    pace_df = parse_brisnet_pace_figures(pp_text, horses)
    post_positions = {r['Horse']: r['Post'] for _, r in horses_df.iterrows()}
    return (figs_df, trainer_stats_df, jockey_stats_df,
            workouts_df, ped_df, pace_df, layoff_days, post_positions)

def parse_brisnet_beyers(pp_text: str, horses: List[str]) -> pd.DataFrame:
    data = []
    lines = pp_text.split('\n')
    cur = None
    for line in lines:
        if any(h in line for h in horses):
            m = re.search(r' - ([\w\s\'.]+) \(', line)
            cur = m.group(1).strip() if m else None
        elif re.match(r'^\d{2}/\d{2}\s+\w+\s+\d+\.?\d*\s+\d+f', line):
            p = re.split(r'\s+', line.strip())
            if len(p) >= 12:
                date, trk, dist = p[0], p[1], p[2]
                bey = float(p[9]) if p[9].replace('.', '').isdigit() else np.nan
                bucket = _distance_bucket_from_text(dist)
                surf = 'Turf' if 't' in trk.lower() else 'Dirt'
                data.append({'Horse': cur, 'Date': date, 'Beyer': bey,
                             'Dist_Bucket': bucket, 'Surface': surf, 'Class_Level': 1.0})
    return pd.DataFrame(data)

def parse_brisnet_trainers(pp_text: str, horses_df: pd.DataFrame) -> pd.DataFrame:
    data = []
    pat = r'Trainer:\s*([\w\s]+)\s*\((\d+)%\s*(overall|all)\s*,?\s*(\d+)%\s*(track)?\s*,?\s*ROI[:\s]*(\d+\.?\d*)\s*\)\s*Last\s*(\d+)\s*starts'
    for m in re.finditer(pat, pp_text, re.I):
        trainer, overall, _, track, _, roi, starts = m.groups()
        data.append({
            'Trainer': trainer.strip(),
            'Overall_Win_Pct': float(overall)/100,
            'Track_Win_Pct': float(track)/100 if track else 0.0,
            'Form_Win_Pct_Last30': float(overall)/100,
            'ROI_Last30': float(roi),
            'Starters_Last30': int(starts),
            'Angles': []
        })
    return pd.DataFrame(data)

def parse_brisnet_jockeys(pp_text: str, horses_df: pd.DataFrame) -> pd.DataFrame:
    data = []
    pat = r'Jockey:\s*([\w\s,]+)\s*\((\d+)%\s*(overall|all)\s*,?\s*(\d+)%?\s*(track)?\s*,?\s*(\d+/\d+)\s*last\s*(\d+)(?:\s*,?\s*ROI[:\s]*(\d+\.?\d*))?\)'
    for m in re.finditer(pat, pp_text, re.I):
        jfull, overall, _, track, _, wins_rides, rides, roi = m.groups()
        jockey = jfull.split(',')[0].strip()
        form = int(wins_rides.split('/')[0]) / int(rides) if rides else 0.0
        data.append({
            'Jockey': jockey,
            'Overall_Win_Pct': float(overall)/100,
            'Track_Win_Pct': float(track)/100 if track else 0.0,
            'Form_Win_Pct_Last10': form,
            'ROI_Last10': float(roi) if roi else 1.0,
            'Rides_Last10': int(rides),
            'Angles': [],
            'Trainer_Synergy_Pct': 0.15
        })
    return pd.DataFrame(data).drop_duplicates('Jockey')

def parse_brisnet_workouts(pp_text: str, horses: List[str]) -> Tuple[pd.DataFrame, Dict[str, int]]:
    data = []
    lay = {}
    lines = pp_text.split('\n')
    cur = None
    last_date = None
    for line in lines:
        if any(h in line for h in horses):
            cur = line.split()[0]
        elif re.match(r'\d{2}/\d{2}\s+\d+f?\s+[\d:.]+(?:\s+h|b)?', line):
            p = re.split(r'\s+', line.strip())
            dstr, dist, tstr = p[0], p[1].replace('f',''), p[2]
            typ = 'bullet' if 'b' in line.lower() else ('handily' if 'h' in line.lower() else 'breeze')
            date = pd.to_datetime(dstr, format='%m/%d')
            dist_f = float(dist)
            data.append({'Horse': cur, 'Date': date, 'Dist_f': dist_f, 'Time_sec': tstr,
                         'Type': typ, 'Gap_days': (datetime.now() - date).days})
        elif re.match(r'\d{2}/\d{2}\s+\w+\s+\d+f', line):
            parts = re.split(r'\s+', line.strip())
            race_dt = pd.to_datetime(parts[0], format='%m/%d')
            if last_date:
                lay[cur] = (race_dt - last_date).days
            last_date = race_dt
    return pd.DataFrame(data), lay

def parse_brisnet_pedigree(pp_text: str, horses: List[str]) -> pd.DataFrame:
    data = []
    for h in horses:
        sec = re.search(rf'{h}.*?(?=Sire:|Pedigree:).*?(?=Sire:.*?TD.*?TS|Dam:.*?TD.*?TS)', pp_text, re.DOTALL | re.I)
        if not sec:
            continue
        txt = sec.group(0).lower()
        st = re.search(r'sire.*?td\s*(\d+)', txt)
        sts = re.search(r'sire.*?ts\s*(\d+)', txt)
        dt = re.search(r'dam.*?td\s*(\d+)', txt)
        dts = re.search(r'dam.*?ts\s*(\d+)', txt)
        inbred = bool(re.search(r'inbred|close kin', txt))
        data.append({
            'Horse': h,
            'Sire_TD': int(st.group(1)) if st else 100,
            'Sire_TS': int(sts.group(1)) if sts else 100,
            'Dam_TD': int(dt.group(1)) if dt else 100,
            'Dam_TS': int(dts.group(1)) if dts else 100,
            'Inbred_Flag': inbred
        })
    return pd.DataFrame(data)

def parse_brisnet_pace_figures(pp_text: str, horses: List[str]) -> pd.DataFrame:
    data = []
    lines = pp_text.split('\n')
    cur = None
    cond = "fast"
    for line in lines:
        if "condition" in line.lower():
            m = re.search(r'(fast|firm|good|yielding|muddy|sloppy|heavy)', line, re.I)
            cond = m.group(1).lower() if m else "fast"
        if any(h in line for h in horses):
            cur = line.split()[1] if len(line.split()) > 1 else None
        elif re.search(r'EP\d+\s+MidP\d+\s+LP\d+', line):
            ep_m = re.search(r'EP(\d+)', line)
            mid_m = re.search(r'MidP(\d+)', line)
            lp_m = re.search(r'LP(\d+)', line)
            if ep_m and mid_m and lp_m:
                ep = float(ep_m.group(1))
                mid = float(mid_m.group(1))
                lp = float(lp_m.group(1))
                p14 = 4
                p12 = 4
                pstr = 4
                date_m = re.search(r'(\d{2}/\d{2})', line)
                dist_m = re.search(r'(\d+f)', line)
                date = date_m.group(1) if date_m else "01/01"
                bucket = _distance_bucket_from_text(dist_m.group(1)) if dist_m else "6.5-7f"
                surf = 'Turf' if 'turf' in line.lower() else 'Dirt'
                data.append({
                    'Horse': cur, 'Date': date, 'EP': ep, 'MidP': mid, 'LP': lp,
                    'Position_1/4': p14, 'Position_1/2': p12, 'Position_Stretch': pstr,
                    'Dist_Bucket': bucket, 'Surface': surf, 'Condition': cond
                })
    return pd.DataFrame(data)

