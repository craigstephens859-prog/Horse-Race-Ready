# app.py
# Horse Race Ready — IQ Mode
# Single-page UI (no sidebar). User-confirmed race info, multi-bias scenarios,
# deterministic overlays, narrative report + position-based tickets (Tri/Super/SH5).

import os
import re
from itertools import product
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
import streamlit as st

# ===================== Page / Model Settings =====================
st.set_page_config(page_title="Horse Race Ready — IQ Mode", page_icon="🏇", layout="wide")
st.title("🏇 Horse Race Ready — IQ Mode")

# ---------------- OpenAI (for narrative report only) ----------------
MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5")
TEMP = float(st.secrets.get("OPENAI_TEMPERATURE", "0.5"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
use_sdk_v1 = True
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY) if OPENAI_API_KEY else None
except Exception:
    client = None
    use_sdk_v1 = False
    try:
        import openai
        openai.api_key = OPENAI_API_KEY
    except Exception:
        openai = None

def model_supports_temperature(model_name: str) -> bool:
    m = (model_name or "").lower()
    return not (m.startswith("gpt-5") or m.startswith("o4") or m.startswith("o3"))

def call_openai_messages(messages: List[Dict]) -> str:
    if not OPENAI_API_KEY or (client is None and openai is None):
        return "(Narrative generation disabled — add OPENAI_API_KEY to .streamlit/secrets.toml)"
    if use_sdk_v1 and client is not None:
        try:
            kwargs = {"model": MODEL, "messages": messages}
            if model_supports_temperature(MODEL):
                kwargs["temperature"] = TEMP
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
                resp = client.chat_completions.create(model=MODEL, messages=messages)
                return resp.choices[0].message.content
            raise
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
            raise

# ===================== Core Helpers =====================
def detect_valid_race_headers(pp_text: str):
    toks = ("purse", "furlong", "mile", "clm", "allow", "stake", "pars", "post time")
    headers = []
    for m in re.finditer(r"(?mi)^\s*Race\s+(\d+)\b", pp_text or ""):
        start = m.start()
        win = (pp_text[m.end():m.end()+250] or "").lower()
        if any(t in win for t in toks):
            headers.append((start, int(m.group(1))))
    return headers

# Auto track detection (heuristics)
def detect_track_name(text: str) -> Optional[str]:
    if not text:
        return None
    lines = (text or "").splitlines()
    head = "\n".join(lines[:30])
    pats = [
        r"(?mi)^\s*([A-Za-z][A-Za-z .&'/-]+?)\s*[-–—]\s*Race\s*\d+\b",
        r"(?mi)^\s*Race\s*\d+\s*(?:at|@)\s*([A-Za-z][A-Za-z .&'/-]+)\b",
        r"(?mi)^\s*TRACK:\s*([A-Za-z][A-Za-z .&'/-]+)\b",
    ]
    for pat in pats:
        m = re.search(pat, head)
        if m:
            return m.group(1).strip().title()
    for s in lines[:10]:
        t = s.strip()
        if len(t) >= 6 and t.upper() == t and any(
            kw in t for kw in [
                " PARK", " DOWNS", " RACECOURSE", " RACE COURSE", " RACETRACK",
                " RACEWAY", " FAIRGROUNDS", " MEADOWS", " TURF CLUB", " RACE PARK"
            ]
        ):
            return t.title()
    return None

# Clean distance phrase finder (for default of "Confirm Track Distance")
def detect_distance_phrase(text: str) -> Optional[str]:
    s = (text or "")
    # Common phrases near header
    m = re.search(r"(?i)\b(\d+(?:\s+\d/\d)?(?:\.\d+)?)\s*(furlongs?|f)\b", s)
    if m:
        val = m.group(1).replace(" ", "")
        try:
            if "/" in val:
                a,b = val.split("/",1)
                v = float(a) / float(b)
                val = str(v)
        except Exception:
            pass
        return f"{m.group(1).strip()} furlongs".replace("Furlongs","furlongs")
    m = re.search(r"(?i)\b(\d+(?:\s+\d/\d)?(?:\.\d+)?)\s*(miles?|m|mile)\b", s)
    if m:
        return f"{m.group(1).strip()} miles"
    return None

def detect_surface_word(text: str) -> Optional[str]:
    s = (text or "").lower()
    if re.search(r"\bturf\b|\btrf\b", s): return "Turf"
    if re.search(r"\b(synth|tapeta|aw|poly)\b", s): return "Synthetic"
    if re.search(r"\bdirt\b", s): return "Dirt"
    return None

def detect_condition_word(text: str) -> Optional[str]:
    s = (text or "").lower()
    for c in ["fast","good","wet-fast","muddy","sloppy","firm","yielding","soft","heavy"]:
        if c in s: return c
    return None

def detect_purse(text: str) -> Optional[int]:
    m = re.search(r"(?i)\bPurse\s*\$?\s*([\d,]+)", text or "")
    if m:
        try:
            return int(m.group(1).replace(",",""))
        except:
            return None
    return None

# ---------- Horse header regex (Style + optional Quirin points) ----------
HORSE_HDR_RE = re.compile(
    r"""(?mi)^\s*
        (\d+)                              # program/post
        \s+([A-Za-z0-9'.\-\s&]+?)          # horse name
        \s*\(\s*                           # open paren
        (E\/P|EP|E|P|S|NA)                 # style
        (?:\s+(\d+))?                      # optional quirin points 0-8
        \s*\)\s*$                          # close paren
    """,
    re.VERBOSE
)

def _normalize_style(tok: str) -> str:
    t = (tok or "").upper().strip()
    return "E/P" if t in ("EP","E/P") else t

def split_into_horse_chunks(pp_text: str) -> List[Tuple[str, str, str]]:
    chunks = []
    matches = list(HORSE_HDR_RE.finditer(pp_text or ""))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(pp_text)
        post = m.group(1).strip()
        name = m.group(2).strip()
        chunk = pp_text[start:end]
        chunks.append((post, name, chunk))
    return chunks

def extract_horses_and_styles(pp_text: str) -> pd.DataFrame:
    rows = []
    for m in HORSE_HDR_RE.finditer(pp_text or ""):
        post = m.group(1).strip()
        name = m.group(2).strip()
        style = _normalize_style(m.group(3))
        qpts = m.group(4)
        quirin = int(qpts) if qpts is not None else np.nan
        rows.append({
            "#": post, "Horse": name, "DetectedStyle": style,
            "Quirin": quirin, "OverrideStyle": "", "StyleStrength": "Solid"
        })
    # de-dup
    seen, uniq = set(), []
    for r in rows:
        k = (r["#"], r["Horse"].lower())
        if k not in seen:
            seen.add(k); uniq.append(r)
    df = pd.DataFrame(uniq)
    if not df.empty and "Quirin" in df.columns:
        df["Quirin"] = df["Quirin"].clip(lower=0, upper=8)
    return df

# Morning Line extractor (tolerant)
ML_TOKEN_RE = re.compile(
    r'(?m)^\s*(?:\$)?([0-9]+(?:\.\d+)?|[0-9]+\s*\/\s*[0-9]+|[0-9]+\s*-\s*[0-9]+|\+\d+|-\d+)\b'
)

def extract_morning_line_by_horse(pp_text: str) -> Dict[str, str]:
    ml = {}
    for _post, name, block in split_into_horse_chunks(pp_text):
        m = ML_TOKEN_RE.search(block or "")
        if m:
            ml[name] = m.group(1).replace(" ", "")
    return ml

# ---- Angles / pedigree parsing ----
ANGLE_LINE_RE = re.compile(
    r'(?mi)^\s*(\d{4}\s+)?(1st\s*time\s*str|Debut\s*MdnSpWt|Maiden\s*Sp\s*Wt|2nd\s*career\s*race|Turf\s*to\s*Dirt|Dirt\s*to\s*Turf|Shipper|Blinkers\s*off|46-90daysAway|JKYw\/\s*Sprints|JKYw\/\s*Trn\s*L60|JKYw\/\s*[EPS]|JKYw\/\s*NA\s*types)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$'
)

def parse_angles_for_block(block: str) -> pd.DataFrame:
    rows=[]
    for m in ANGLE_LINE_RE.finditer(block or ""):
        _yr, cat, starts, win, itm, roi = m.groups()
        rows.append({
            "Category": re.sub(r'\s+',' ',cat.strip()),
            "Starts": int(starts), "Win%": float(win),
            "ITM%": float(itm), "ROI": float(roi)
        })
    return pd.DataFrame(rows)

def parse_pedigree_snips(block: str) -> Dict[str, float]:
    out = {"sire_awd":np.nan,"sire_1st":np.nan,"damsire_awd":np.nan,"damsire_1st":np.nan,"dam_dpi":np.nan}
    s = re.search(r'(?mi)^\s*Sire\s*Stats:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%Mud.*?\s*(\d+)%\s*1st.*?(\d+(?:\.\d+)?)\s*spi', block or "")
    if s:
        out["sire_awd"] = float(s.group(1)); out["sire_1st"] = float(s.group(3))
    ds = re.search(r'(?mi)^\s*Dam\'sSire:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%Mud.*?(\d+)%\s*1st.*?(\d+(?:\.\d+)?)\s*spi', block or "")
    if ds:
        out["damsire_awd"] = float(ds.group(1)); out["damsire_1st"] = float(ds.group(3))
    d = re.search(r"(?mi)Dam'sStats:.*?(\d+(?:\.\d+)?)\s*dpi", block or "")
    if d:
        out["dam_dpi"] = float(d.group(1))
    return out

# Workouts
WORKOUT_LINE_RE = re.compile(
    r"""(?mi)^\s*
        (?:\d{2}[A-Za-z]{3}\d{2,4})
        .*?
        ([Bb]\b)?
        \s+
        (\d+)
        \/
        (\d+)
        \s*$
    """,
    re.VERBOSE
)

def parse_workouts_for_block(block: str) -> float:
    matches = list(WORKOUT_LINE_RE.finditer(block or ""))
    if not matches:
        return 0.0
    scores = []
    for m in matches:
        is_bullet = bool(m.group(1))
        try:
            rank = float(m.group(2)); total = float(m.group(3))
            if total == 0:
                continue
            score = 1.0 if is_bullet else (1.0 - (rank/total))
            scores.append(score)
        except Exception:
            continue
    if not scores:
        return 0.0
    weights = np.exp(-np.arange(len(scores)) * 0.25)
    try:
        weighted_avg = np.average(scores, weights=weights)
    except ZeroDivisionError:
        return 0.0
    return float(np.clip(weighted_avg * 0.8, 0.0, 1.2))

# ---- Odds helpers ----
def dec_to_prob(dec_odds: Optional[float]) -> float:
    return 1.0/dec_odds if dec_odds and dec_odds > 0 else 0.0

def am_to_dec(american: float) -> float:
    return 1 + (american/100.0 if american > 0 else 100.0/abs(american))

def fair_to_american(p: float) -> float:
    if p <= 0: return float("inf")
    if p >= 1: return 0.0
    dec = 1.0/p
    return round((dec-1)*100,0) if dec >= 2 else round(-100/(dec-1),0)

def str_to_decimal_odds(s: str) -> Optional[float]:
    s=(s or "").strip()
    if not s: return None
    if re.fullmatch(r'[+-]?\d+(\.\d+)?', s):
        v=float(s); return max(v,1.01)
    if re.fullmatch(r'\+\d+', s) or re.fullmatch(r'-\d+', s):
        return am_to_dec(float(s))
    if "-" in s:
        a,b=s.split("-",1)
        try: return float(a)/float(b)+1.0
        except: return None
    if "/" in s:
        a,b=s.split("/",1)
        try: return float(a)/float(b)+1.0
        except: return None
    return None

def overlay_table(fair_probs: Dict[str,float], offered: Dict[str,float]) -> pd.DataFrame:
    rows=[]
    for h, p in fair_probs.items():
        off_dec = offered.get(h)
        if off_dec is None: 
            continue
        off_prob = dec_to_prob(off_dec)
        ev = (off_dec-1)*p - (1-p)
        rows.append({
            "Horse": h,
            "Fair %": round(p*100,2),
            "Fair (AM)": fair_to_american(p),
            "Board (dec)": round(off_dec,3),
            "Board %": round(off_prob*100,2),
            "Edge (pp)": round((p - off_prob)*100,2),
            "EV per $1": round(ev,3),
            "Overlay?": "YES" if off_prob < p else "No"
        })
    return pd.DataFrame(rows).sort_values(by=["Overlay?","EV per $1"], ascending=[False, False])

# ---------- Weight presets ----------
DEFAULT_WEIGHTS = {
    "global": {"pace_shape":1.0,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":0.7,"workout_snap":0.8,"projection":1.0,"ft_firsters":1.0},
    ("Dirt","<=6f"):  {"pace_shape":1.25,"bias_fit":1.1,"class_form":1.0,"trs_jky":1.0,"pedigree":0.6,"workout_snap":0.7,"projection":1.05,"ft_firsters":1.05},
    ("Dirt","6.5-7f"):{"pace_shape":1.1,"bias_fit":1.1,"class_form":1.0,"trs_jky":1.0,"pedigree":0.7,"workout_snap":0.8,"projection":1.05,"ft_firsters":1.05},
    ("Dirt","8f+"):  {"pace_shape":1.0,"bias_fit":1.1,"class_form":1.15,"trs_jky":1.0,"pedigree":1.0,"workout_snap":0.8,"projection":1.0,"ft_firsters":0.95},
    ("Turf","<=6f"):  {"pace_shape":0.95,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.05,"pedigree":1.1,"workout_snap":0.75,"projection":1.0,"ft_firsters":1.25},
    ("Turf","6.5-7f"):{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.1,"trs_jky":1.05,"pedigree":1.05,"workout_snap":0.85,"projection":1.0,"ft_firsters":1.1},
    ("Turf","8f+"):  {"pace_shape":1.0,"bias_fit":1.1,"class_form":1.2,"trs_jky":1.1,"pedigree":1.1,"workout_snap":0.85,"projection":1.0,"ft_firsters":0.9},
    ("Synthetic","<=6f"):{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":0.9,"workout_snap":0.8,"projection":1.0,"ft_firsters":1.1},
    ("Synthetic","6.5-7f"):{"pace_shape":1.0,"bias_fit":1.05,"class_form":1.0,"trs_jky":1.0,"pedigree":0.95,"workout_snap":0.8,"projection":1.0},
    ("Synthetic","8f+"):{"pace_shape":0.95,"bias_fit":1.0,"class_form":1.1,"trs_jky":1.0,"pedigree":1.0,"workout_snap":0.85,"projection":1.0,"ft_firsters":0.95},
}

def distance_bucket(distance_text: str) -> str:
    s=(distance_text or "").lower()
    if "6 1/2" in s or "6.5" in s: return "6.5-7f"
    if "furlong" in s:
        m=re.search(r'(\d+(?:\.\d+)?)\s*furlong', s)
        if m:
            v=float(m.group(1))
            if v<=6.0: return "<=6f"
            if v<=7.0: return "6.5-7f"
            return "8f+"
        return "<=6f"
    if "mile" in s: return "8f+"
    return "<=6f"

def get_weight_preset(surface: str, distance_txt: str) -> Dict[str,float]:
    return DEFAULT_WEIGHTS.get((surface, distance_bucket(distance_txt)), DEFAULT_WEIGHTS["global"])

# ---- Race type detection ----
def detect_race_type(pp_text: str) -> str:
    s = (pp_text or "").lower()
    if re.search(r'\bmdn\b|\bmaiden\b', s):
        if re.search(r'\bmc\b|\bmaid(en)?\s*claim', s):
            return "Maiden Claiming"
        if re.search(r'\bmdn\s*sp|maiden\s*sp|mdn\s*special', s):
            return "Maiden Special Weight"
        return "Maiden (other)"
    if "allow" in s: return "Allowance"
    if "stakes" in s or "stake" in s:
        if re.search(r'grade\s*i|\bg1\b', s): return "Stakes (G1)"
        if re.search(r'grade\s*ii|\bg2\b', s): return "Stakes (G2)"
        if re.search(r'grade\s*iii|\bg3\b', s): return "Stakes (G3)"
        if "listed" in s: return "Stakes (Listed)"
        return "Stakes"
    return "Other"

def adjust_by_race_type(weights: Dict[str,float], race_type: str) -> Dict[str,float]:
    w = weights.copy()
    bump = lambda k, m: w.__setitem__(k, round(w.get(k,1.0)*m,3))
    if race_type.startswith("Maiden Special"):
        bump("ft_firsters",1.15); bump("pedigree",1.08); bump("workout_snap",1.08); bump("projection",1.05)
    elif race_type=="Maiden Claiming":
        bump("class_form",1.08); bump("trs_jky",1.05); bump("projection",0.98)
    elif race_type=="Allowance":
        bump("class_form",1.10); bump("trs_jky",1.06)
    elif race_type.startswith("Stakes"):
        bump("class_form",1.15); bump("trs_jky",1.10); bump("pedigree",1.05)
    return w

# ---------- PPI ----------
def _style_of(row) -> str:
    s = (row.get("OverrideStyle") or row.get("DetectedStyle") or "NA").strip().upper()
    return "E/P" if s in ("E/P","EP") else s

def _strength_wt(row) -> float:
    s = (row.get("StyleStrength") or "Solid").strip().lower()
    return {"strong": 1.0, "solid": 0.8, "slight": 0.5, "weak": 0.3, "bias": 0.9}.get(s, 0.8)

def compute_ppi(df_styles: pd.DataFrame) -> dict:
    if df_styles is None or df_styles.empty:
        return {"ppi": 0.0, "by_horse": {}}
    e_like, closers = 0.0, 0.0
    by_horse = {}
    for _, r in df_styles.iterrows():
        stl = _style_of(r); wt  = _strength_wt(r)
        if stl in ("E","E/P"): e_like += wt
        elif stl == "S":       closers += wt
    raw_ppi = e_like - closers
    denom = max(e_like + closers, 1e-6)
    ppi_norm = max(min(raw_ppi/denom, 1.0), -1.0)
    def tailwind(style: str, ppi: float) -> float:
        if style == "E":   return -0.6*ppi
        if style == "E/P": return -0.35*ppi
        if style == "P":   return  0.25*ppi
        if style == "S":   return  0.55*ppi
        return 0.0
    for _, r in df_styles.iterrows():
        h = str(r["Horse"]); stl = _style_of(r)
        by_horse[h] = round(tailwind(stl, ppi_norm), 3)
    return {"ppi": round(ppi_norm,3), "by_horse": by_horse}

# ---------- Bias helpers ----------
def style_from_row(row) -> str:
    s = (row.get("OverrideStyle") or row.get("DetectedStyle") or "NA").strip().upper()
    return "E/P" if s in ("EP", "E/P") else s

def quirin_from_row(row) -> float:
    q = row.get("Quirin", np.nan)
    try:
        q = float(q)
    except Exception:
        return np.nan
    if pd.isna(q): return np.nan
    return max(0.0, min(8.0, q))

def track_adjustments(surface_type: str, distance_txt: str, condition_txt: str) -> Tuple[float,float,float]:
    srf = (surface_type or "Dirt").strip().lower()
    dis = distance_bucket(distance_txt)
    con = (condition_txt or "").strip().lower()
    Csurf = {"dirt": 0.0, "turf": 0.0, "synthetic": 0.0}.get(srf, 0.0)
    Cdist = {"<=6f": 0.0, "6.5-7f": 0.0, "8f+": 0.0}.get(dis, 0.0)
    Ccond = {
        "fast": 0.0, "good": 0.0, "wet-fast": 0.0, "muddy": 0.0, "sloppy": 0.0,
        "firm": 0.0, "yielding": 0.0, "soft": 0.0, "heavy": 0.0,
    }.get(con, 0.0)
    return (Csurf, Cdist, Ccond)

def wet_nudge(ped: dict, cond: str) -> float:
    if (cond or "").lower() not in {"muddy", "sloppy", "wet-fast"}:
        return 1.0
    s1 = ped.get("sire_1st", np.nan); d1 = ped.get("damsire_1st", np.nan)
    boost = 1.0
    for v in (s1, d1):
        if v == v and v >= 16: boost *= 1.03
        if v == v and v <= 8:  boost *= 0.97
    return round(boost, 3)

def style_match_score(running_style_bias: str, style: str, quirin: float) -> float:
    bias = (running_style_bias or "").strip().lower()
    st   = (style or "NA").upper()
    table = {
        "favors early (e)":      {"E": +3.0, "E/P": +2.0, "P": -0.5, "S": -1.0},
        "favors e/p (early/presser)": {"E": +1.5, "E/P": +2.5, "P": +0.5, "S": -0.5},
        "favors p (presser/stalker)": {"E": +0.5, "E/P": +1.5, "P": +2.0, "S": +0.5},
        "favors s (closer)":     {"E": -1.5, "E/P": -0.5, "P": +1.0, "S": +2.5},
        "fair/neutral":          {"E": 0.0,  "E/P": 0.0,  "P": 0.0,  "S": 0.0},
        "":                      {"E": 0.0,  "E/P": 0.0,  "P": 0.0,  "S": 0.0},
    }
    norm = bias.replace("style:", "").replace("–", "-")
    base = table.get(norm, table["fair/neutral"]).get(st, 0.0)
    if not pd.isna(quirin):
        if "favors early" in norm or "favors e/p" in norm:
            base += 0.15 * (quirin - 4)
        elif "favors s" in norm:
            base += -0.10 * (quirin - 4)
    return float(round(base, 2))

def post_bias_score(post_bias_pick: str, post_str: str) -> float:
    pick = (post_bias_pick or "").strip().lower()
    try:
        post = int(re.sub(r"[^\d]", "", str(post_str)))
    except Exception:
        post = None
    if "no significant" in pick or not post:
        return 0.0
    if "rail" in pick:
        return +1.0 if post == 1 else 0.0
    if "inner" in pick and 1 <= post <= 3:
        return +0.8
    if "mid" in pick and 4 <= post <= 7:
        return +0.6
    if "outside" in pick and 8 <= post <= 12:
        return +0.6
    return 0.0

def field_size_class(n: int) -> str:
    if n <= 5:  return "Small"
    if n <= 7:  return "Normal"
    if n <= 12: return "Large"
    return "Extra Large"

def advanced_tailwind(style: str, ppi: float, quirin: float,
                      distance_txt: str, surface: str,
                      running_style_bias: str, field_size: int) -> float:
    stl = (style or "NA").upper()
    base = {"E": -0.60, "E/P": -0.35, "P": 0.25, "S": 0.55}.get(stl, 0.0)
    bucket = distance_bucket(distance_txt)
    srf    = (surface or "Dirt").lower()

    dist_scale = 1.0
    if bucket == "<=6f":
        if stl == "E":   dist_scale = 0.80
        if stl == "E/P": dist_scale = 0.90
        if stl == "P":   dist_scale = 0.95
        if stl == "S":   dist_scale = 0.85
    elif bucket == "8f+":
        if stl == "E":   dist_scale = 1.15
        if stl == "E/P": dist_scale = 1.10
        if stl == "P":   dist_scale = 1.10
        if stl == "S":   dist_scale = 1.15

    surf_scale = 1.0
    if srf == "turf":
        if stl in ("P", "S"): surf_scale = 1.10
        if stl in ("E", "E/P"): surf_scale = 1.05
    elif srf == "synthetic":
        surf_scale = 0.95

    fclass = field_size_class(field_size)
    fs_scale = {"Small": 0.90, "Normal": 1.00, "Large": 1.05, "Extra Large": 1.10}[fclass]
    if field_size < 8: fs_scale = 1.00

    q_scale = 1.0
    if not pd.isna(quirin):
        if stl in ("E", "E/P"): q_scale += 0.02 * (quirin - 4)
        elif stl in ("P", "S"): q_scale += 0.015 * max(0.0, 4 - quirin)

    coef = base * dist_scale * surf_scale * fs_scale * q_scale
    cpace = coef * float(ppi or 0.0)

    rsb = (running_style_bias or "").lower()
    if ("favors early" in rsb or "favors e/p" in rsb) and stl in ("E","E/P") and ppi > 0:
        cpace *= 0.85
    if ("favors s" in rsb or "closer" in rsb) and stl == "S" and ppi < 0:
        cpace *= 0.90

    cpace = float(np.tanh(cpace / 2.0) * 2.5)  # soft clamp
    return float(round(cpace, 2))

def compute_bias_ratings(df_styles: pd.DataFrame, surface_type: str, distance_txt: str, condition_txt: str,
                         race_type: str, running_style_bias: str, post_bias_pick: str,
                         ppi_value: float = 0.0,
                         pedigree_per_horse: dict | None = None) -> pd.DataFrame:
    cols = ["#", "Horse", "Style", "Quirin", "Cstyle", "Cpost", "Cpace", "Cclass", "Atrack", "Arace", "R"]
    if df_styles is None or df_styles.empty:
        return pd.DataFrame(columns=cols)

    Csurf, Cdist, Ccond = track_adjustments(surface_type, distance_txt, condition_txt)
    Atrack = Csurf + Cdist + Ccond
    Cclass = 0.0

    rows = []
    for _, r in df_styles.iterrows():
        post  = str(r["#"])
        horse = str(r["Horse"])
        stl   = style_from_row(r)
        q     = quirin_from_row(r)

        cstyle = style_match_score(running_style_bias, stl, q)
        cpost  = post_bias_score(post_bias_pick, post)

        strength_mult = _strength_wt(r)

        cpace_base  = advanced_tailwind(
            style=stl, ppi=float(ppi_value or 0.0),
            quirin=float(q) if not pd.isna(q) else np.nan,
            distance_txt=distance_txt, surface=surface_type,
            running_style_bias=running_style_bias, field_size=len(df_styles)
        )
        cpace = cpace_base * strength_mult

        a_track_h = Atrack
        if pedigree_per_horse and horse in pedigree_per_horse:
            a_track_h = round(a_track_h * wet_nudge(pedigree_per_horse.get(horse, {}), condition_txt), 3)

        arace = Cclass + cstyle + cpost + cpace
        rbase = 0.0
        R     = rbase + a_track_h + arace

        rows.append({
            "#": post, "Horse": horse, "Style": stl,
            "Quirin": ("" if pd.isna(q) else int(q)),
            "Cstyle": round(cstyle, 2), "Cpost": round(cpost, 2),
            "Cpace":  round(cpace, 2), "Cclass": Cclass,
            "Atrack": round(a_track_h, 2), "Arace": round(arace, 2),
            "R": round(R, 1),
        })

    return pd.DataFrame(rows)

# ===== BRIS figures =====
FIG_PATTERNS = {
    "prime_power": r"Prime\s*Power\s*[:\-]?\s*(\d+)",
    "spd_last":    r"Last\s*Speed\s*[:\-]?\s*(\d+)",
    "e1":          r"\bE1\s*[:\-]?\s*(\d+)",
    "e2":          r"\bE2\s*[:\-]?\s*(\d+)",
    "late":        r"\bLate\s*[:\-]?\s*(\d+)",
}

def _zscale(vals: List[float]) -> List[float]:
    a = np.array(vals, dtype=float); mu = np.nanmean(a); sd = np.nanstd(a)
    sd = sd if (sd and np.isfinite(sd)) else 1.0
    return list((a - mu) / sd)

def parse_bris_figs(pp_text: str, horses: List[str]) -> pd.DataFrame:
    rows=[]; name_set=set(horses or [])
    for post, name, block in split_into_horse_chunks(pp_text):
        if name not in name_set: continue
        rec={"Horse":name}; s=block or ""
        for k,pat in FIG_PATTERNS.items():
            m=re.search(pat, s, flags=re.I)
            rec[k]=float(m.group(1)) if m else np.nan
        rows.append(rec)
    return pd.DataFrame(rows)

def add_fig_signal(ratings_df: pd.DataFrame, figs_df: pd.DataFrame, base_weights: Dict[str, float]) -> pd.DataFrame:
    if ratings_df is None or ratings_df.empty:
        return ratings_df
    figs = figs_df.copy()
    if figs.empty:
        return ratings_df
    for col in ["prime_power","spd_last","e1","e2","late"]:
        if col in figs:
            figs[col+"_z"] = _zscale(figs[col].tolist())
    figs["ABILITY_SIG"] = (0.60 * figs.get("prime_power_z", 0)) + (0.40 * figs.get("spd_last_z", 0))
    figs["PACE_FIGURE_SIG"] = (0.25 * figs.get("e1_z", 0)) + (0.35 * figs.get("e2_z", 0)) + (0.40 * figs.get("late_z", 0))
    out = ratings_df.merge(figs[["Horse","ABILITY_SIG","PACE_FIGURE_SIG"]], on="Horse", how="left")
    w_ability = float(base_weights.get("class_form", 1.0)) * 2.2
    w_pacefig = float(base_weights.get("projection", 1.0)) * 1.8
    out["R"] = out["R"] + (out["ABILITY_SIG"].fillna(0.0) * w_ability) + (out["PACE_FIGURE_SIG"].fillna(0.0) * w_pacefig)
    return out

# ===== Angle/pedigree/intent =====
def is_first_time_starter(df_angles: pd.DataFrame) -> bool:
    if df_angles is None or df_angles.empty:
        return False
    cats = " ".join(df_angles["Category"].astype(str).tolist()).lower()
    return ("1st time" in cats) or ("debut" in cats)

def trainer_jockey_signal(df_angles: pd.DataFrame) -> float:
    if df_angles is None or df_angles.empty:
        return 0.0
    sig = 0.0
    for _, r in df_angles.iterrows():
        cat = str(r.get("Category",""))
        if "JKYw/" not in cat and "L60" not in cat:
            continue
        win = float(r.get("Win%", 12.0))
        roi = float(r.get("ROI", 0.0))
        starts = int(r.get("Starts", 0))
        if starts < 15:
            continue
        multiplier = 1.5 if ("L60" in cat and "Trn" in cat) else 1.0
        sig += multiplier * (0.03*(win - 12.0) + 0.10*roi)
    return float(np.clip(sig, -1.8, 1.8))

def pedigree_fit(surface: str, distance_txt: str, ped: dict) -> float:
    if not ped:
        return 0.0
    buck = distance_bucket(distance_txt)
    s_awd = ped.get("sire_awd", np.nan)
    d_awd = ped.get("damsire_awd", np.nan)
    awds = [v for v in [s_awd, d_awd] if v==v]
    if not awds:
        return 0.0
    mean_awd = float(np.mean(awds))
    score = 0.0
    if buck == "<=6f":
        score += 0.20*(7.0 - mean_awd)
    elif buck == "8f+":
        score += 0.20*(mean_awd - 7.5)
    if (surface or "").lower() == "turf" and buck != "<=6f":
        score += 0.10
    return float(np.clip(score, -0.6, 0.6))

def fts_bonus(surface: str, distance_txt: str, df_angles: pd.DataFrame, ped: dict, race_type: str) -> float:
    if not is_first_time_starter(df_angles):
        return 0.0
    buck = distance_bucket(distance_txt)
    srf = (surface or "").lower()
    base = 0.35
    if srf == "turf" and buck == "<=6f":
        base += 0.45
    if race_type.lower().startswith("maiden special"):
        base += 0.15
    s1 = ped.get("sire_1st", np.nan) if ped else np.nan
    if s1==s1:
        if s1 >= 15: base += 0.20
        if s1 <= 8:  base -= 0.15
    return float(np.clip(base, -0.2, 1.2))

def situational_adjustments(track_name: str, surface: str, distance_txt: str,
                            style: str, post_str: str,
                            race_type: str, df_angles: pd.DataFrame) -> float:
    delta = 0.0
    if df_angles is not None and not df_angles.empty:
        cats = " ".join(df_angles["Category"].astype(str).tolist()).lower()
        if "turf to dirt" in cats and "claim" in race_type.lower():
            delta += 0.40 if style in ("E","E/P") else 0.20
    return float(np.clip(delta, -1.0, 1.2))

def apply_enhancements(ratings_df: pd.DataFrame,
                       pp_text: str,
                       base_weights: Dict[str,float],
                       chaos_index: float,
                       track_name: str,
                       surface_type: str, distance_txt: str,
                       race_type: str,
                       angles_per_horse: Dict[str,pd.DataFrame],
                       pedigree_per_horse: Dict[str,dict]) -> pd.DataFrame:
    if ratings_df is None or ratings_df.empty:
        return ratings_df
    df = ratings_df.copy()

    # Meta-weighting by chaos
    chaos_mod = float(chaos_index)
    adj_weights = base_weights.copy()
    adj_weights["class_form"] = adj_weights.get("class_form",1.0) * (1.0 - 0.3*chaos_mod)
    adj_weights["projection"] = adj_weights.get("projection",1.0) * (1.0 - 0.2*chaos_mod)
    adj_weights["ft_firsters"] = adj_weights.get("ft_firsters",1.0) * (1.0 + 0.4*chaos_mod)
    adj_weights["pedigree"] = adj_weights.get("pedigree",1.0) * (1.0 + 0.3*chaos_mod)

    ped_map = pedigree_per_horse or {}
    ang_map = angles_per_horse or {}

    # Pre-calc workouts
    workout_map = {}
    for _post, name, block in split_into_horse_chunks(pp_text):
        if name in df["Horse"].values:
            workout_map[name] = parse_workouts_for_block(block)

    w_pace = float(adj_weights.get("pace_shape",1.0))
    w_bias = float(adj_weights.get("bias_fit",1.0))

    df["R_wt_adj"] = 0.0
    df["PedFit"] = 0.0
    df["FTS"] = 0.0
    df["TRNJKY"] = 0.0
    df["Workouts"] = 0.0
    df["Situational"] = 0.0

    for i, r in df.iterrows():
        h = str(r["Horse"])
        retro = (w_pace-1.0)*float(r.get("Cpace",0.0)) + (w_bias-1.0)*(float(r.get("Cstyle",0.0))+float(r.get("Cpost",0.0)))

        ped = ped_map.get(h, {})
        ang = ang_map.get(h, pd.DataFrame())

        work_sig = workout_map.get(h, 0.0) * float(adj_weights.get("workout_snap", 1.0))
        ped_sig = pedigree_fit(surface_type, distance_txt, ped) * float(adj_weights.get("pedigree",1.0))
        fts_sig = fts_bonus(surface_type, distance_txt, ang, ped, race_type) * float(adj_weights.get("ft_firsters",1.0))
        tj_sig  = trainer_jockey_signal(ang) * float(adj_weights.get("trs_jky",1.0))
        sit     = situational_adjustments(track_name, surface_type, distance_txt, str(r.get("Style","")), str(r.get("#","")), race_type, ang)

        total_adj = retro + ped_sig + fts_sig + tj_sig + sit + work_sig

        df.at[i, "R_wt_adj"] = round(total_adj, 3)
        df.at[i, "PedFit"] = round(ped_sig, 3)
        df.at[i, "FTS"] = round(fts_sig, 3)
        df.at[i, "TRNJKY"] = round(tj_sig, 3)
        df.at[i, "Workouts"] = round(work_sig, 3)
        df.at[i, "Situational"] = round(sit, 3)

    df["R"] = (df["R"].astype(float) + df["R_wt_adj"].astype(float)).round(2)
    return df

# ===== R → fair line (softmax) =====
def compute_chaos_index(kept_horses: List[str],
                        df_styles: pd.DataFrame,
                        ppi_value: float,
                        pp_text: str,
                        angles_per_horse: Dict[str,pd.DataFrame],
                        figs_df: pd.DataFrame) -> float:
    n = len(kept_horses or [])
    ci = 0.0
    if n >= 12: ci += 0.40
    elif n >= 10: ci += 0.25

    ft_count = 0
    for h in kept_horses:
        ang = angles_per_horse.get(h, pd.DataFrame())
        if is_first_time_starter(ang):
            ft_count += 1
    ci += min(0.30, 0.05*ft_count)

    if ppi_value >= 0.55:
        ci += 0.20
    elif ppi_value <= -0.55:
        ci -= 0.10

    figs = figs_df
    if not figs.empty:
        kept_figs = figs[figs['Horse'].isin(kept_horses)].copy()
        if not kept_figs.empty:
            for col in ["prime_power","spd_last","e1","e2","late"]:
                if col in kept_figs:
                    kept_figs[col+"_z"] = _zscale(kept_figs[col].tolist())
            if "prime_power_z" in kept_figs:
                sd = float(np.nanstd(kept_figs["prime_power_z"].to_numpy()))
            else:
                sd = float(np.nanstd(kept_figs.get("spd_last_z", pd.Series(np.zeros(len(kept_figs))))))  # noqa: PD011
            if sd == sd and np.isfinite(sd):
                if sd <= 0.60: ci += 0.20
                elif sd >= 1.20: ci -= 0.05

    return float(np.clip(ci, 0.0, 1.0))

def ratings_to_probs(ratings_df: pd.DataFrame, distance_txt: str, chaos_index: float = 0.0) -> dict:
    if ratings_df is None or ratings_df.empty: return {}
    r = ratings_df["R"].astype(float).to_numpy()
    r = r - np.nanmean(r)
    bucket = distance_bucket(distance_txt)
    base_tau = 5.5 if bucket=="<=6f" else (6.5 if bucket=="6.5-7f" else 7.5)
    tau = base_tau * (1.0 + 0.60*float(chaos_index))
    z = np.exp(r / max(tau, 1e-6))
    p = z / z.sum()
    return {h: float(p[i]) for i, h in enumerate(ratings_df["Horse"])}

# ---------------- Ticket helpers (deterministic; no LLM) ----------------
def _fmt_names(rows: pd.DataFrame) -> str:
    if rows is None or rows.empty:
        return "(none)"
    return ", ".join([f"{str(r['#']).strip()}–{str(r['Horse']).strip()}" for _, r in rows.iterrows()])

def _tier_sizes_for_profile(profile: str) -> tuple[int, int, int]:
    p = (profile or "").lower()
    if "aggress" in p:
        return (3, 2, 2)
    if "value" in p:
        return (2, 3, 3)
    return (2, 2, 2)

def _split_tiers(df_sorted: pd.DataFrame, profile: str) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    a,b,c = _tier_sizes_for_profile(profile)
    A = df_sorted.head(a)
    B = df_sorted.iloc[a:a+b]
    C = df_sorted.iloc[a+b:a+b+c]
    return A, B, C

def build_pos_tickets(df_sorted: pd.DataFrame, profile: str) -> dict:
    A, B, C = _split_tiers(df_sorted, profile)

    tri = [
        {"label": "Ticket 1 (core)", "pos": {"1st": A, "2nd": pd.concat([A,B], ignore_index=True), "3rd": pd.concat([A,B,C], ignore_index=True)}},
        {"label": "Ticket 2 (saver)", "pos": {"1st": B, "2nd": A, "3rd": pd.concat([A,B,C], ignore_index=True)}},
        {"label": "Ticket 3 (price pop)", "pos": {"1st": A, "2nd": B, "3rd": pd.concat([A,B,C], ignore_index=True)}},
    ]
    supert = [
        {"label": "Ticket 1 (core)", "pos": {"1st": A, "2nd": pd.concat([A,B], ignore_index=True), "3rd": pd.concat([A,B,C], ignore_index=True), "4th": pd.concat([A,B,C], ignore_index=True)}},
        {"label": "Ticket 2 (saver)", "pos": {"1st": B, "2nd": A, "3rd": pd.concat([A,B,C], ignore_index=True), "4th": pd.concat([A,B,C], ignore_index=True)}},
        {"label": "Ticket 3 (price pop)", "pos": {"1st": A, "2nd": B, "3rd": pd.concat([A,B], ignore_index=True), "4th": pd.concat([A,B,C], ignore_index=True)}},
    ]
    sh5 = [
        {"label": "Ticket 1 (core)", "pos": {"1st": A, "2nd": pd.concat([A,B], ignore_index=True), "3rd": pd.concat([A,B], ignore_index=True), "4th": pd.concat([A,B,C], ignore_index=True), "5th": pd.concat([A,B,C], ignore_index=True)}},
        {"label": "Ticket 2 (saver)", "pos": {"1st": B, "2nd": A, "3rd": pd.concat([A,B], ignore_index=True), "4th": pd.concat([A,B,C], ignore_index=True), "5th": pd.concat([A,B,C], ignore_index=True)}},
    ]
    return {"TRI": tri, "SUPER": supert, "SH5": sh5}

def tickets_markdown(tix: dict, base: float = 0.10) -> str:
    lines = []
    lines.append("### Trifecta (scale base to bankroll)")
    for t in tix["TRI"]:
        lines.append(f"- **{t['label']}**")
        for k in ["1st","2nd","3rd"]:
            lines.append(f"  - {k}: {_fmt_names(t['pos'][k])}")
    lines.append("\n### Superfecta")
    for t in tix["SUPER"]:
        lines.append(f"- **{t['label']}**")
        for k in ["1st","2nd","3rd","4th"]:
            lines.append(f"  - {k}: {_fmt_names(t['pos'][k])}")
    lines.append("\n### Super High-5")
    for t in tix["SH5"]:
        lines.append(f"- **{t['label']}**")
        for k in ["1st","2nd","3rd","4th","5th"]:
            lines.append(f"  - {k}: {_fmt_names(t['pos'][k])}")
    lines.append(f"\n_Example base:_ ${base:.2f}. Use Ticket 1 as primary; others as light savers.")
    return "\n".join(lines)

# ===================== UI — Main Page =====================

# Strategy Profile
st.header("Strategy Profile")
profile = st.radio("Profile:", ["Confident", "Aggressive", "Value Hunter"], index=2, horizontal=True,
                   help="Sets overlay thresholds and narrative emphasis.")
iq_mode = st.toggle("IQ Mode (advanced heuristics)", value=True,
                    help="Enable/disable advanced heuristics (workouts, FTS/pedigree, meta-weighting).")

# 1) Race Data
st.header("1. Paste BRIS PPs")
pp_text_input = st.text_area("Paste BRISNET 'All-Ways' PPs here:", height=240,
                             placeholder="Paste entire race card or a single race...")

pp_text = ""
headers = detect_valid_race_headers(pp_text_input)
if pp_text_input.strip():
    if headers:
        race_options = [f"Race {r} (pos {s})" for s, r in headers]
        sel = st.selectbox("Select Race:", options=race_options, index=0)
        if sel:
            start_pos = int(re.search(r'pos (\d+)\)', sel).group(1))
            end_pos = len(pp_text_input)
            for s, _r in headers:
                if s > start_pos:
                    end_pos = s
                    break
            pp_text = pp_text_input[start_pos:end_pos]
    else:
        st.caption("No explicit 'Race #' headers found — treating the pasted text as a single race.")
        pp_text = pp_text_input

if not pp_text.strip():
    st.info("Please paste BRIS PPs above to begin.")
    st.stop()

# 2) Race Info (user-confirmed; autobias OFF — we only auto-suggest)
st.header("2. Race Info (confirm or edit)")
track_guess = detect_track_name(pp_text) or detect_track_name(pp_text_input) or ""
distance_guess = detect_distance_phrase(pp_text) or ""
surface_guess = detect_surface_word(pp_text) or "Dirt"
condition_guess = detect_condition_word(pp_text) or "fast"
race_type_guess = detect_race_type(pp_text)
purse_guess = detect_purse(pp_text)

c1, c2, c3 = st.columns([1.1, 1.1, 0.9])
with c1:
    track_display = st.text_input("Track", value=track_guess)
    surface_type = st.selectbox("Surface Type", ["Dirt","Turf","Synthetic"], index=["Dirt","Turf","Synthetic"].index(surface_guess))
with c2:
    distance_txt_user = st.text_input("Confirm Track Distance (e.g., '7 furlongs')", value=distance_guess)
    condition_txt = st.selectbox("Surface Condition", ["fast","good","wet-fast","muddy","sloppy","firm","yielding","soft","heavy"],
                                 index=["fast","good","wet-fast","muddy","sloppy","firm","yielding","soft","heavy"].index(condition_guess))
with c3:
    race_type_user = st.selectbox("Race Type", ["Allowance","Maiden Special Weight","Maiden Claiming","Stakes","Stakes (Listed)","Stakes (G3)","Stakes (G2)","Stakes (G1)","Other"],
                                  index=max(0, ["Allowance","Maiden Special Weight","Maiden Claiming","Stakes","Stakes (Listed)","Stakes (G3)","Stakes (G2)","Stakes (G1)","Other"].index(race_type_guess) if race_type_guess in ["Allowance","Maiden Special Weight","Maiden Claiming","Stakes","Stakes (Listed)","Stakes (G3)","Stakes (G2)","Stakes (G1)","Other"] else 0))
    purse_amount = st.number_input("Purse ($)", min_value=0, step=1000, value=int(purse_guess or 0))

st.caption("Autobias is **off** — you choose style/post biases in the next step. Type of race & purse are auto-suggested but editable.")

# 3) Multi-Bias scenarios (user picks — autobias OFF)
st.header("3. Multi-Bias Scenarios")
col_bias1, col_bias2, col_bias3 = st.columns([1.2, 1.2, 1.0])
with col_bias1:
    running_style_biases = st.multiselect(
        "Running Style Bias:",
        ["fair/neutral", "favors early (e)", "favors e/p (early/presser)", "favors p (presser/stalker)", "favors s (closer)"],
        default=["fair/neutral"]
    )
with col_bias2:
    post_biases = st.multiselect(
        "Post Bias Type:",
        ["no significant post bias", "favors rail (1)", "favors inner (1-3)", "favors mid (4-7)", "favors outside (8+)"],
        default=["no significant post bias"]
    )
with col_bias3:
    primary_scenario = st.radio("Primary Scenario:", ["Most Likely", "Most Optimistic (for longshots)"], index=0)

# 4) Field & Pace Setup
st.header("4. Field & Pace Setup")
df_styles = extract_horses_and_styles(pp_text)
ml_map = extract_morning_line_by_horse(pp_text)
if df_styles.empty:
    st.error("No horses found. Check your PP text paste.")
    st.stop()

df_styles["ML"] = df_styles["Horse"].map(ml_map).fillna("10/1")
df_styles["LiveOdds"] = ""                      # user enters live odds here
df_styles["Scratched"] = False

col_cfg = {
    "#": st.column_config.TextColumn("Post", width="small"),
    "Horse": st.column_config.TextColumn("Horse", width="medium"),
    "ML": st.column_config.TextColumn("M/L", width="small", help="Formats: 5/2, +250, 3.6"),
    "LiveOdds": st.column_config.TextColumn("Live Odds", width="small", help="Optional: 7/2, +350, 4.5"),
    "DetectedStyle": st.column_config.TextColumn("BRIS Style", width="small", help="E / E/P / P / S"),
    "Quirin": st.column_config.NumberColumn("Quirin", width="small", help="0–8"),
    "OverrideStyle": st.column_config.SelectboxColumn("Override Style", width="small", options=["", "E", "E/P", "P", "S"]),
    "StyleStrength": st.column_config.SelectboxColumn("Style Strength", width="medium", options=["Strong", "Solid", "Slight", "Weak"]),
    "Scratched": st.column_config.CheckboxColumn("Scratched?", width="small")
}
df_editor = st.data_editor(df_styles, column_config=col_cfg, use_container_width=True, hide_index=True)

# Pre-compute for all horses (before scratches)
angles_per_horse = {}
pedigree_per_horse = {}
for _post, name, block in split_into_horse_chunks(pp_text):
    if name in df_editor["Horse"].values:
        angles_per_horse[name] = parse_angles_for_block(block)
        pedigree_per_horse[name] = parse_pedigree_snips(block)

# Apply Scratches & Compute Pace
df_styles_scratched = df_editor[df_editor["Scratched"] == False].copy()
kept_horses = list(df_styles_scratched["Horse"])
if df_styles_scratched.empty:
    st.warning("All horses are scratched.")
    st.stop()

ppi_results = compute_ppi(df_styles_scratched)
ppi_val = ppi_results.get("ppi", 0.0)
pace_desc = "Neutral Pace"
if ppi_val > 0.45: pace_desc = f"Hot Pace ({ppi_val:+.2f})"
elif ppi_val < -0.45: pace_desc = f"Slow Pace ({ppi_val:+.2f})"
elif ppi_val > 0.15: pace_desc = f"Warm Pace ({ppi_val:+.2f})"
elif ppi_val < -0.15: pace_desc = f"Moderate Pace ({ppi_val:+.2f})"
st.subheader(f"Pace Profile: {pace_desc}")

# 5) Scenario Analysis & Ratings
st.header("5. Scenario Analysis & Ratings")
weights_base = get_weight_preset(surface_type, distance_txt_user)
weights_final = adjust_by_race_type(weights_base, race_type_user)

all_figs_df = parse_bris_figs(pp_text, list(df_styles["Horse"]))
figs_df_scratched = all_figs_df[all_figs_df['Horse'].isin(kept_horses)].copy()

chaos_index = compute_chaos_index(kept_horses, df_styles_scratched, ppi_val, pp_text, angles_per_horse, figs_df=all_figs_df)
st.caption(f"Chaos Index: {chaos_index:.2f} (0=formful, 1=chaotic)")

scenarios = list(product(running_style_biases or ["fair/neutral"], post_biases or ["no significant post bias"]))
tab_names = [f"S: {s} | P: {p}" for s, p in scenarios]

all_scenario_ratings: Dict[Tuple[str,str], Tuple[pd.DataFrame, Dict[str,float]]] = {}
with st.spinner("Computing scenarios…"):
    for (rbias, pbias) in scenarios:
        ratings_df = compute_bias_ratings(
            df_styles_scratched, surface_type, distance_txt_user, condition_txt,
            race_type_user, rbias, pbias, ppi_val, pedigree_per_horse
        )
        if iq_mode:
            ratings_df = apply_enhancements(
                ratings_df, pp_text, base_weights=weights_final, chaos_index=chaos_index,
                track_name=track_display or "", surface_type=surface_type, distance_txt=distance_txt_user,
                race_type=race_type_user, angles_per_horse=angles_per_horse, pedigree_per_horse=pedigree_per_horse
            )
        ratings_df = add_fig_signal(ratings_df, figs_df=figs_df_scratched, base_weights=weights_final)
        fair_probs = ratings_to_probs(ratings_df, distance_txt_user, chaos_index)
        ratings_df["Fair %"] = ratings_df["Horse"].map(lambda h: f"{fair_probs.get(h,0)*100:.1f}%")
        ratings_df["Fair Odds"] = ratings_df["Horse"].map(lambda h: fair_to_american(fair_probs.get(h,0)))
        all_scenario_ratings[(rbias, pbias)] = (ratings_df, fair_probs)

# Render tabs
tabs = st.tabs(tab_names)
for i, (rbias, pbias) in enumerate(scenarios):
    with tabs[i]:
        df_show, _p = all_scenario_ratings[(rbias, pbias)]
        st.dataframe(
            df_show.sort_values(by="R", ascending=False),
            use_container_width=True,
            hide_index=True,
            column_config={
                "R": st.column_config.NumberColumn("Rating", format="%.2f"),
                "Fair %": st.column_config.TextColumn("Fair %"),
                "Fair Odds": st.column_config.NumberColumn("Fair Odds")
            }
        )

# 6) Determine Primary Scenario & Overlays
if not all_scenario_ratings:
    st.error("No scenarios were computed.")
    st.stop()

primary_key = scenarios[0]
if primary_scenario == "Most Optimistic (for longshots)" and len(all_scenario_ratings) > 1:
    best_key, best_prob = primary_key, 0.0
    longshot_horses = set(df_editor[df_editor["ML"].str.contains(r'(10/1|12/1|15/1|20/1|30/1|50/1)', regex=True, na=False)]["Horse"])
    if not longshot_horses:
        longshot_horses = set(df_editor[df_editor["ML"].str.contains(r'(6/1|8/1)', regex=True, na=False)]["Horse"])
    for key, (_df, probs) in all_scenario_ratings.items():
        for h, p in probs.items():
            if h in longshot_horses and p > best_prob:
                best_prob = p; best_key = key
    primary_key = best_key

primary_df, primary_probs = all_scenario_ratings[primary_key]
st.info(f"Primary Scenario for overlays: S={primary_key[0]} | P={primary_key[1]}")

# Overlays vs Morning Line and Live Odds
offered_odds_map_ml = {}
offered_odds_map_live = {}
for _, r in df_editor.iterrows():
    dec_ml = str_to_decimal_odds(r["ML"])
    if dec_ml:
        offered_odds_map_ml[r["Horse"]] = dec_ml
    dec_live = str_to_decimal_odds(r.get("LiveOdds",""))
    if dec_live:
        offered_odds_map_live[r["Horse"]] = dec_live

df_ol_ml = overlay_table(primary_probs, offered_odds_map_ml)
df_ol_live = overlay_table(primary_probs, offered_odds_map_live) if offered_odds_map_live else pd.DataFrame()

st.header("6. Overlays & Betting Strategy")
col_ol1, col_ol2 = st.columns(2)
with col_ol1:
    st.subheader("Overlay vs. Morning Line")
    st.dataframe(
        df_ol_ml[df_ol_ml["Horse"].isin(kept_horses)],
        use_container_width=True, hide_index=True,
        column_config={"EV per $1": st.column_config.NumberColumn("EV per $1", format="$%.3f"),
                       "Edge (pp)": st.column_config.NumberColumn("Edge (pp)")}
    )
with col_ol2:
    st.subheader("Overlay vs. Live Odds")
    if df_ol_live.empty:
        st.caption("Enter **Live Odds** in Step 4 to enable this table.")
    else:
        st.dataframe(
            df_ol_live[df_ol_live["Horse"].isin(kept_horses)],
            use_container_width=True, hide_index=True,
            column_config={"EV per $1": st.column_config.NumberColumn("EV per $1", format="$%.3f"),
                           "Edge (pp)": st.column_config.NumberColumn("Edge (pp)")}
        )

# 7) Actionable Insights + Tickets
st.header("7. Actionable Insights")
if st.button("Analyze this race", type="primary"):
    try:
        primary_df_sorted = primary_df.sort_values(by="R", ascending=False)

        overlays_pos_ml = df_ol_ml[(df_ol_ml["EV per $1"] > 0) & (df_ol_ml["Horse"].isin(kept_horses))]
        overlays_pos_live = df_ol_live[(df_ol_live["EV per $1"] > 0) & (df_ol_live["Horse"].isin(kept_horses))] if not df_ol_live.empty else pd.DataFrame()

        top_rated_txt = primary_df_sorted[['#','Horse','R','Fair %','Fair Odds']].to_string(index=False)
        top_ol_ml_txt = overlays_pos_ml[['Horse','Fair %','Fair (AM)','Board (dec)','EV per $1']].to_string(index=False) if not overlays_pos_ml.empty else "None."
        top_ol_live_txt = overlays_pos_live[['Horse','Fair %','Fair (AM)','Board (dec)','EV per $1']].to_string(index=False) if not overlays_pos_live.empty else "None."

        # Deterministic tickets
        tix = build_pos_tickets(primary_df_sorted, profile)
        tix_md = tickets_markdown(tix, base=0.10)

        # Narrative prompt (tickets are printed by the app)
        prompt = f"""
Act as a superior BRISNET PP horse racing analyst. Provide a concise, professional report.
Do **not** print ticket grids; the app prints the tickets.

Race:
- Track: {track_display}
- Race: {distance_txt_user} | Surface: {surface_type} | Condition: {condition_txt}
- Type: {race_type_user} | Purse: {'$'+format(purse_amount,',') if purse_amount else 'N/A'}
- Pace: {pace_desc} (PPI {ppi_val:.2f}); Chaos Index {chaos_index:.2f}
- Primary Bias: {primary_key[0]} (style), {primary_key[1]} (post)
- Strategy Profile: {profile}

Top Rated (model):
{top_rated_txt}

Positive EV vs Morning Line:
{top_ol_ml_txt}

Positive EV vs Live:
{top_ol_live_txt}

Write:
1) Race synopsis (pace & bias).
2) 2–3 key win contenders with quick reasons.
3) Value (overlays) and likely underlays with thresholds.
4) Bankroll guidance (how to lean toward core opinions).
"""
        with st.spinner("Handicapping..."):
            messages = [
                {"role": "system", "content": "You are a world-class horse racing analyst providing expert, concise betting strategies."},
                {"role": "user", "content": prompt}
            ]
            report = call_openai_messages(messages)

        st.markdown(report)
        st.markdown("---")
        st.markdown("## Ticket Maker (position-based)")
        st.markdown(tix_md)

        st.session_state['last_report'] = report + "\n\n---\n\n" + tix_md
    except Exception as e:
        st.error(f"Error generating report: {e}")

if 'last_report' in st.session_state:
    st.download_button("Download Report", st.session_state['last_report'],
                       file_name="HorseRaceReady_Analysis.txt", mime="text/plain")

st.caption("Fair line is model-derived (Ratings → softmax). User confirms Surface/Condition/Distance/Type. Autobias is off — you choose biases. Tickets are deterministic and aligned with the model rankings.")









