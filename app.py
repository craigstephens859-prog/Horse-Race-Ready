# app.py
# Horse Race Ready — IQ Mode 🔥 (Beginner-friendly + Pro controls)
# • Robust horse parsing (incl. NA firsters) • Race-type detection (MSW/MC/Alw/Stakes/Graded/Listed)
# • Per-horse AUTO angle/pedigree read (1st time str / Debut MSW / 2nd career / Turf->Dirt / Shipper / Blinkers etc.)
# • NEW: Pace Pressure Index (PPI) + per-horse Style Tailwind (trip bias) — auto used in analysis
# • Sire/Dam (AWD, 1st%, SPI/DPI) nudges • Surface + condition impact
# • Step 4 = ONE table (Horse • ML Odds • Live Odds) + auto overlays (Live > ML fallback)
# • Strategy profiles: Confident / Aggressive / Value Hunter
# • No confidence slider; ranges appear in analysis text only
# • Downloads: analysis (.txt), overlays (CSV), ticket costs (CSV)

import os, re, json
from typing import List, Dict, Tuple
import pandas as pd
import numpy as np
import streamlit as st

# ===================== Page / Model Settings =====================
st.set_page_config(page_title="Horse Race Ready — IQ Mode", page_icon="🏇", layout="wide")
st.title("🏇 Horse Race Ready — IQ Mode")

with st.popover("Glossary"):
    st.markdown(
        """
**Running styles**: E (early), E/P (early/presser), P (mid-pack), S (closer), NA (unknown/firster).  
**Bias**: Track/post pattern favoring Inside (1–3), Mid (4–7), Outside (8+).  
**Overlay**: Board price > our fair price → positive value.  
**Underlay**: Board too short vs. fair → avoid/underneath only.  
**EV per $1**: Expected return per dollar at current odds (positive is good).  
**PPI**: Pace Pressure Index (−1 soft pace … +1 hot pace). Hot pace helps S/P; soft pace helps E/E-P.
"""
    )

# ---------------- Secrets / OpenAI ----------------
MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5")
TEMP  = float(st.secrets.get("OPENAI_TEMPERATURE", "0.5"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Add your key to `.streamlit/secrets.toml`:\n\nOPENAI_API_KEY = \"sk-...\"")
    st.stop()

use_sdk_v1 = True
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    import openai
    openai.api_key = OPENAI_API_KEY
    use_sdk_v1 = False

def model_supports_temperature(model_name: str) -> bool:
    m = (model_name or "").lower()
    return not (m.startswith("gpt-5") or m.startswith("o4") or m.startswith("o3"))

def call_openai_messages(messages: List[Dict]) -> str:
    if use_sdk_v1:
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

# ===================== Legal/Links (optional) =====================
REPO_USER = st.secrets.get("GH_USER", "craigstephens859-prog")
REPO_NAME = st.secrets.get("GH_REPO", "horse-race-ready")
_BASE_GH = f"https://github.com/{REPO_USER}/{REPO_NAME}/blob/main"
TERMS_URL   = st.secrets.get("TERMS_URL",   f"{_BASE_GH}/TERMS.md")
PRIVACY_URL = st.secrets.get("PRIVACY_URL", f"{_BASE_GH}/PRIVACY.md")
st.markdown(
    f"""
**Disclaimer:** Informational handicapping analysis only — **not** financial or wagering advice.  
Use at your own risk. By using this app, you agree to the
[Terms]({TERMS_URL}) and [Privacy Policy]({PRIVACY_URL}).
""",
    unsafe_allow_html=True,
)

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

# ---- Horse / style / ML odds parsing ----
HORSE_HDR_RE = re.compile(
    r"(?mi)^\s*(\d+)\s+([A-Za-z0-9\'\.\-\s&]+?)\s+\(\s*(E\/P|E|P|S|NA)\s*[\d]*\s*\)\s*$"
)

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
    rows=[]
    for m in HORSE_HDR_RE.finditer(pp_text or ""):
        rows.append({
            "#": m.group(1).strip(),
            "Horse": m.group(2).strip(),
            "DetectedStyle": m.group(3).replace("E/P","E/P"),
            "OverrideStyle": "",
            "StyleStrength": "Solid",
        })
    seen=set(); uniq=[]
    for r in rows:
        k=(r["#"], r["Horse"].lower())
        if k not in seen: seen.add(k); uniq.append(r)
    return pd.DataFrame(uniq)

# Morning Line extractor
ML_TOKEN_RE = re.compile(r'(?m)^\s*([0-9]+\/[0-9]+|[0-9]+-[0-9]+|\+\d+|-\d+|\d+(?:\.\d+)?)\s+')

def extract_morning_line_by_horse(pp_text: str) -> Dict[str, str]:
    ml = {}
    for _post, name, block in split_into_horse_chunks(pp_text):
        m = ML_TOKEN_RE.search(block or "")
        if m:
            ml[name] = m.group(1)
    return ml

# ---- Angle / pedigree parsing (per horse) ----
ANGLE_LINE_RE = re.compile(
    r'(?mi)^\s*(\d{4}\s+)?(1st\s*time\s*str|Debut\s*MdnSpWt|Maiden\s*Sp\s*Wt|2nd\s*career\s*race|Turf\s*to\s*Dirt|Dirt\s*to\s*Turf|Shipper|Blinkers\s*off|46-90daysAway|JKYw\/\s*Sprints|JKYw\/\s*Trn\s*L60|JKYw\/\s*[EPS]|JKYw\/\s*NA\s*types)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$'
)

def parse_angles_for_block(block: str) -> pd.DataFrame:
    rows=[]
    for m in ANGLE_LINE_RE.finditer(block or ""):
        _yr, cat, starts, win, itm, roi = m.groups()
        rows.append({
            "Category": re.sub(r'\s+',' ',cat.strip()),
            "Starts": int(starts),
            "Win%": float(win),
            "ITM%": float(itm),
            "ROI": float(roi)
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
    d = re.search(r"Dam'sStats:.*?(\d+(?:\.\d+)?)\s*dpi", block or "", flags=re.IGNORECASE)
    if d:
        out["dam_dpi"] = float(d.group(1))
    return out

# ---- Odds helpers ----
def dec_to_prob(dec_odds: float) -> float:
    return 1.0/dec_odds if dec_odds and dec_odds > 0 else 0.0

def am_to_dec(american: float) -> float:
    return 1 + (american/100.0 if american > 0 else 100.0/abs(american))

def fair_to_american(p: float) -> float:
    if p <= 0: return float("inf")
    if p >= 1: return 0.0
    dec = 1.0/p
    return round((dec-1)*100,0) if dec >= 2 else round(-100/(dec-1),0)

def str_to_decimal_odds(s: str) -> float|None:
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
        off_dec  = offered.get(h)
        if off_dec is None: continue
        off_prob = dec_to_prob(off_dec)
        ev = (off_dec-1)*p - (1-p)
        rows.append({
            "Horse": h,
            "Fair %": round(p*100,2),
            "Fair (AM)": fair_to_american(p),
            "Board (dec)": round(off_dec,3),
            "Board %": round(off_prob*100,2),
            "Edge (Board%-Fair%)": round((off_prob-p)*100,2),
            "EV per $1": round(ev,3),
            "Overlay?": "YES" if off_prob < p else "No"
        })
    return pd.DataFrame(rows).sort_values(by=["Overlay?","EV per $1"], ascending=[False, False])

# ---- Ticket cost helpers ----

def superfecta_cost(a:int,b:int,c:int,d:int, base:float)->float: return base * a*b*c*d

def super_high5_cost(a:int,b:int,c:int,d:int,e:int, base:float)->float: return base * a*b*c*d*e

# ---------- Weight presets ----------
DEFAULT_WEIGHTS = {
    "global": {"pace_shape":1.0,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":0.7,"workout_snap":0.8,"projection":1.0,"ft_firsters":1.0},
    ("Dirt","≤6f"):  {"pace_shape":1.2,"bias_fit":1.1,"class_form":1.0,"trs_jky":1.0,"pedigree":0.6,"workout_snap":0.7,"projection":1.05,"ft_firsters":1.1},
    ("Dirt","6.5–7f"):{"pace_shape":1.1,"bias_fit":1.1,"class_form":1.0,"trs_jky":1.0,"pedigree":0.7,"workout_snap":0.8,"projection":1.05,"ft_firsters":1.1},
    ("Dirt","8f+"):  {"pace_shape":1.0,"bias_fit":1.1,"class_form":1.2,"trs_jky":1.0,"pedigree":0.8,"workout_snap":0.8,"projection":1.0,"ft_firsters":0.95},
    ("Turf","≤6f"):  {"pace_shape":0.95,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":1.0,"workout_snap":0.7,"projection":1.0,"ft_firsters":1.15},
    ("Turf","6.5–7f"):{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.1,"trs_jky":1.0,"pedigree":1.0,"workout_snap":0.8,"projection":1.0,"ft_firsters":1.1},
    ("Turf","8f+"):  {"pace_shape":1.0,"bias_fit":1.1,"class_form":1.2,"trs_jky":1.1,"pedigree":1.1,"workout_snap":0.8,"projection":1.0,"ft_firsters":0.9},
    ("Synthetic","≤6f"):{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":0.8,"workout_snap":0.7,"projection":1.0,"ft_firsters":1.1},
    ("Synthetic","6.5–7f"):{"pace_shape":1.0,"bias_fit":1.05,"class_form":1.0,"trs_jky":1.0,"pedigree":0.9,"workout_snap":0.7,"projection":1.0,"ft_firsters":1.0},
    ("Synthetic","8f+"):{"pace_shape":0.95,"bias_fit":1.0,"class_form":1.1,"trs_jky":1.0,"pedigree":1.0,"workout_snap":0.8,"projection":1.0,"ft_firsters":0.9},
}

def distance_bucket(distance_text: str) -> str:
    s=(distance_text or "").lower()
    if "6½" in s or "6 1/2" in s or "6.5" in s: return "6.5–7f"
    if "furlong" in s:
        m=re.search(r'(\d+(?:\.\d+)?)\s*furlong', s)
        if m:
            v=float(m.group(1))
            if v<=6.0: return "≤6f"
            if v<=7.0: return "6.5–7f"
            return "8f+"
        return "≤6f"
    if "mile" in s: return "8f+"
    return "≤6f"


def get_weight_preset(surface: str, distance_txt: str) -> Dict[str,float]:
    return DEFAULT_WEIGHTS.get((surface, distance_bucket(distance_txt)), DEFAULT_WEIGHTS["global"])


def compute_post_bias_label(field_size: int) -> str:
    return f"Inside:1-3 | Mid:4-7 | Outside:{'8+' if field_size >= 8 else '8+'}"

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
        bump("ft_firsters",1.12); bump("pedigree",1.08); bump("workout_snap",1.08); bump("projection",1.05)
    elif race_type=="Maiden Claiming":
        bump("class_form",1.08); bump("trs_jky",1.05); bump("projection",0.98)
    elif race_type=="Allowance":
        bump("class_form",1.10); bump("trs_jky",1.06)
    elif race_type.startswith("Stakes"):
        bump("class_form",1.15); bump("trs_jky",1.10); bump("pedigree",1.05)
    return w


def nudge_from_pedigree(weights: Dict[str,float], ped: Dict[str,float], is_firster: bool)->Dict[str,float]:
    w = weights.copy()
    if is_firster:
        for key in ("sire_1st","damsire_1st"):
            v = ped.get(key)
            if isinstance(v,(int,float)) and v==v:
                if v >= 18: w["ft_firsters"] = round(w.get("ft_firsters",1.0)*1.06,3)
                elif v <= 8: w["ft_firsters"] = round(w.get("ft_firsters",1.0)*0.96,3)
    for key in ("sire_awd","damsire_awd"):
        v = ped.get(key)
        if isinstance(v,(int,float)) and v==v:
            if 6.0<=v<=7.5: w["pedigree"]=round(w.get("pedigree",1.0)*1.03,3)
    return w


def incorporate_angles_into_weights(weights: Dict[str,float], angle_df: pd.DataFrame, is_firster: bool) -> Dict[str,float]:
    w = weights.copy()
    if angle_df is None or angle_df.empty: return w
    def has(cat): 
        return any(angle_df["Category"].str.contains(cat, case=False, regex=True)) if "Category" in angle_df else False
    bump = lambda k, m: w.__setitem__(k, round(w.get(k,1.0)*m,3))
    if has("Debut") or has("1st time str"):
        bump("ft_firsters",1.10); bump("workout_snap",1.05)
    if has("2nd career"):
        bump("projection",1.05)
    if has("Turf to Dirt"):
        bump("projection",1.04); bump("pace_shape",1.03)
    if has("Blinkers off"):
        bump("projection",1.03)
    if has("Shipper"):
        bump("class_form",1.03)
    return w

# ---------- Pace Pressure Index & Style Tailwind ----------

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
        stl = _style_of(r)
        wt  = _strength_wt(r)
        if stl in ("E","E/P"): e_like += wt
        elif stl == "S":       closers += wt
    raw_ppi = e_like - closers       # >0 hot early; <0 soft
    ppi_norm = max(min(raw_ppi/4.0, 1.0), -1.0)  # cap realistic range
    def tailwind(style: str, ppi: float) -> float:
        if style == "E":   return -0.6*ppi
        if style == "E/P": return -0.3*ppi
        if style == "P":   return  0.3*ppi
        if style == "S":   return  0.6*ppi
        return 0.0
    for _, r in df_styles.iterrows():
        h = str(r["Horse"]); stl = _style_of(r)
        by_horse[h] = round(tailwind(stl, ppi_norm), 3)
    return {"ppi": round(ppi_norm,3), "by_horse": by_horse}

# ===================== Sidebar =====================
with st.sidebar:
    st.header("Strategy Profile")
    profile = st.radio(
        "Profile:",
        ["Confident", "Aggressive", "Value Hunter"],
        index=2,
        help="Sets risk, overlay aggression, and ticket complexity."
    )
    iq_mode = st.toggle("IQ Mode (advanced heuristics, FTS logic, presets)", value=True)

# ===================== Main Flow =====================

# Step 1 — PPs
st.markdown("## Step 1 — Paste one race (PP text)")
st.caption("Tip: copy only one race from your PDF. If we detect more than one, we’ll ask you to repaste.")
pp_text = st.text_area("BRIS PPs:", height=260, placeholder="Paste the full text block of a single race…")

race_headers = detect_valid_race_headers(pp_text) if pp_text.strip() else []
if pp_text.strip():
    if len(race_headers) == 0:
        st.info("No explicit 'Race #' header detected — OK if it’s a single race block.")
    elif len(race_headers) > 1:
        st.error(f"Detected **{len(race_headers)} races**. Paste **only one** race block.")
        st.stop()

# Surface / distance / race type
colA, colB, colC = st.columns([1.1,1,1])
with colA:
    guess_surface = "Dirt"
    if re.search(r'\bturf\b', (pp_text or ""), flags=re.I): guess_surface = "Turf"
    if re.search(r'\b(aw|tapeta|synthetic)\b', (pp_text or ""), flags=re.I): guess_surface = "Synthetic"
    surface_type = st.selectbox("Surface:", ["Dirt","Turf","Synthetic"], index=["Dirt","Turf","Synthetic"].index(guess_surface))
with colB:
    default_dist = ""
    m = re.search(r'(\d+\s*½?|\d+\.\d+)\s*Furlongs|\b1\s*Mile', pp_text or "", flags=re.IGNORECASE)
    if m: default_dist = m.group(0)
    distance_txt = st.text_input("Distance label (for preset bucket):", value=default_dist or "6 Furlongs")
with colC:
    race_type = detect_race_type(pp_text)
    st.text_input("Race type (auto-detected):", value=race_type, disabled=True)

# Step 2 — Styles, Bias, Scratches
st.markdown("## Step 2 — Confirm scratches & running styles")
df_styles_full = extract_horses_and_styles(pp_text) if pp_text.strip() else pd.DataFrame()
detected_horses = list(df_styles_full["Horse"]) if not df_styles_full.empty else []
if pp_text.strip() and not detected_horses:
    st.warning("We couldn’t find the numbered entry lines (like `1 Horse Name (E 4)` or `(NA 0)`). Paste the part with the horse list.")
field_size = len(detected_horses)

col1, col2 = st.columns([1.2, 1])
with col1:
    bias_options = [
        "favors speed", "favors stalkers", "favors closers",
        "inside bias (posts 1-3)", "mid bias (posts 4-7)",
        f"outside bias ({'8+' if field_size>=8 else '8+'})",
        "tiring speed", "fair/neutral"
    ]
    biases = st.multiselect("Track bias today:", options=bias_options, default=["fair/neutral"])
with col2:
    st.write("**Post Bias Key**")
    st.info(compute_post_bias_label(field_size or 8))

st.markdown("**Scratches**")
scratched_by_pick = st.multiselect("Detected horses to scratch:", options=detected_horses, default=[])
scratches_manual = st.text_input("Or type numbers/names (comma or new line):", placeholder="e.g., 2, 7, Holiday Fantasy")
scratch_tokens = set()
if scratches_manual.strip():
    for tok in re.split(r"[,\n]+", scratches_manual):
        tok=tok.strip()
        if tok: scratch_tokens.add(tok.lower())
scratch_set = set([h.strip().lower() for h in scratched_by_pick]) | scratch_tokens

df_styles = df_styles_full.copy()
if not df_styles.empty and scratch_set:
    mask = ~df_styles["Horse"].str.lower().isin(scratch_set) & ~df_styles["#"].astype(str).str.lower().isin(scratch_set)
    df_styles = df_styles[mask].reset_index(drop=True)

style_options = ["", "E", "E/P", "P", "S", "NA"]
strength_options = ["Strong", "Solid", "Slight", "Weak", "Bias"]
edited = st.data_editor(
    df_styles,
    column_config={
        "#": st.column_config.Column(disabled=True, width="small"),
        "Horse": st.column_config.Column(disabled=True),
        "DetectedStyle": st.column_config.Column(disabled=True, help="Parsed from lines like (P 2) or (NA 0)"),
        "OverrideStyle": st.column_config.SelectboxColumn("OverrideStyle", options=style_options, help="Leave blank to keep detected."),
        "StyleStrength": st.column_config.SelectboxColumn("StyleStrength", options=strength_options, help="Run-style conviction."),
    },
    hide_index=True, use_container_width=True, num_rows="fixed",
)
st.caption("E=Early; E/P=Early/Presser; P=Mid-pack; S=Closer; NA=Unknown. ‘Style Strength’ = consistency (Strong/Solid/Slight/Weak/Bias).")

def _format_running_styles(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "None detected"
    lines=[]
    for _, r in df.iterrows():
        style = (r.get("OverrideStyle") or r.get("DetectedStyle") or "NA").strip()
        strength = (r.get("StyleStrength") or "Solid").strip()
        lines.append(f"{r['Horse']} = {style} ({strength})")
    return "\n".join(lines) if lines else "None detected"

running_styles_text = _format_running_styles(edited)
kept_horses = list(edited["Horse"]) if isinstance(edited, pd.DataFrame) and not edited.empty else []
scratches_list = sorted(scratch_set)

# --- NEW: Pace Pressure Index & Tailwind per horse ---
ppi_pack = compute_ppi(edited)
ppi_value = ppi_pack["ppi"]
style_tailwind = ppi_pack["by_horse"]

# --- AUTO angle/pedigree per horse (no Step 3 UI; silent feed into prompt) ---
angles_per_horse = {}
pedigree_per_horse = {}
if pp_text.strip():
    for post, name, block in split_into_horse_chunks(pp_text):
        if name.lower() in scratch_set or post.lower() in scratch_set:
            continue
        angles_per_horse[name] = parse_angles_for_block(block)
        pedigree_per_horse[name] = parse_pedigree_snips(block)

# Step 4 — ONE odds table (ML auto + Live input) + auto overlays
st.markdown("## Step 3 — Odds & Overlays (single table)")
st.caption("Morning Line auto-fills from the pasted PPs. Enter Live Odds to update overlays. Formats: 7/2, 5-1, +250, 3.8.")

c1, c2 = st.columns(2)
with c1:
    reset_board = st.button("Reset odds table (clear)", use_container_width=True)
if reset_board and "board_df" in st.session_state:
    del st.session_state["board_df"]

ml_map = extract_morning_line_by_horse(pp_text) if pp_text.strip() else {}
horses_for_board = [h for h in kept_horses]

if "board_df" not in st.session_state or reset_board:
    st.session_state.board_df = pd.DataFrame({
        "Horse": pd.Series(horses_for_board, dtype="string"),
        "Morning Line": pd.Series([ml_map.get(h,"") for h in horses_for_board], dtype="string"),
        "Live Odds": pd.Series(["" for _ in horses_for_board], dtype="string"),
        "Use (dec)": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
        "Use %": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
    })

board_df_in = st.session_state.board_df.copy()
board_df_in["Horse"] = board_df_in["Horse"].astype("string")
board_df_in["Morning Line"] = board_df_in["Morning Line"].astype("string")
board_df_in["Live Odds"] = board_df_in["Live Odds"].astype("string")

edit_df = st.data_editor(
    board_df_in,
    column_config={
        "Horse": st.column_config.Column(disabled=True),
        "Morning Line": st.column_config.TextColumn(help="Auto from PPs; edit if needed"),
        "Live Odds": st.column_config.TextColumn(help="Type odds; leave blank if none yet"),
        "Use (dec)": st.column_config.NumberColumn(format="%.3f", disabled=True),
        "Use %": st.column_config.NumberColumn(format="%.2f", disabled=True),
    },
    hide_index=True, use_container_width=True
)

use_dec_map = {}
use_pct = []
use_dec_col=[]
for _, r in edit_df.iterrows():
    h = str(r["Horse"])
    live = str(r.get("Live Odds","") or "").strip()
    ml   = str(r.get("Morning Line","") or "").strip()
    pick = live if live else ml
    dec = str_to_decimal_odds(pick) if pick else None
    use_dec_map[h] = dec
    use_dec_col.append(dec if dec else np.nan)
    use_pct.append(dec_to_prob(dec)*100 if dec else np.nan)

edit_df["Use (dec)"] = use_dec_col
edit_df["Use %"] = use_pct
st.session_state.board_df = edit_df
st.dataframe(edit_df, use_container_width=True, hide_index=True)

# Step 5 — Analyze
st.markdown("## Step 4 — Run analysis")

profile_rules = {
    "Confident": {"overlay_push_pp": 3.0},
    "Aggressive": {"overlay_push_pp": 1.5},
    "Value Hunter": {"overlay_push_pp": 2.0},
}
overlay_push_pp = profile_rules[profile]["overlay_push_pp"]

base_weights = get_weight_preset(surface_type, distance_txt) if iq_mode else DEFAULT_WEIGHTS["global"]
base_weights = adjust_by_race_type(base_weights, race_type)


def build_user_prompt(pp_text: str):
    style_glossary = (
        "Running Style Glossary:\n"
        "• E (Early/Front-Runner); • E/P (Early/Presser); • P (Stalker mid-pack); • S (Closer); • NA (Unknown/Firster).\n"
        "Use strength tags (Strong/Solid/Slight/Weak/Bias) to calibrate confidence.\n"
    )
    scratches_txt = ", ".join(scratches_list) if scratches_list else "none"

    # AUTO angles/pedigree highlights (brief — the math already nudges)
    angle_summ_lines=[]
    for h, dfh in angles_per_horse.items():
        if dfh is None or dfh.empty: 
            continue
        # FIX: use dict-style access for columns with % in their names
        sample = " | ".join(
            f"{r['Category']}:{int(r['Starts'])}/{int(r['Win%'])}%/{int(r['ITM%'])}%/{float(r['ROI']):+.2f}"
            for _, r in dfh.head(2).iterrows()
        )
        angle_summ_lines.append(f"{h}: {sample}")
    angle_block = "AUTO angle highlights:\n" + ("\n".join("• "+x for x in angle_summ_lines) if angle_summ_lines else "• (none)")

    ped_lines=[]
    for h, pdg in pedigree_per_horse.items():
        if not pdg: continue
        s1 = pdg.get("sire_awd"); s2 = pdg.get("sire_1st")
        d1 = pdg.get("damsire_awd"); d2 = pdg.get("damsire_1st")
        piece = []
        if s1==s1: piece.append(f"Sire AWD {s1}")
        if s2==s2: piece.append(f"Sire 1st% {int(s2)}%")
        if d1==d1: piece.append(f"DamSire AWD {d1}")
        if d2==d2: piece.append(f"DamSire 1st% {int(d2)}%")
        if piece: ped_lines.append(f"{h}: " + ", ".join(piece))
    ped_block = "Pedigree notes:\n" + ("\n".join("• "+x for x in ped_lines) if ped_lines else "• (none)")

    # Pace Pressure context injected to the model
    pace_shape_line = f"Pace Pressure Index (PPI): {ppi_value:+.2f}  (neg = soft/loose lead; pos = hot/duel risk)"
    tailwind_lines = []
    for h in kept_horses:
        if h in style_tailwind:
            tw = style_tailwind[h]
            if abs(tw) >= 0.05:
                tailwind_lines.append(f"{h}: {tw:+.2f}")
    tailwind_block = "Style Tailwind (trip bias):\n" + ("• " + "\n• ".join(tailwind_lines) if tailwind_lines else "• (neutral)")

    eff_w = base_weights.copy()
    preset_line = "Effective Weights (global): " + json.dumps(eff_w)

    return f"""
You are an elite, value-driven handicapper. Use pace figures, class ratings, bias patterns (post+run style),
form cycles, trainer & jockey intent, pedigree tendencies, workout signals, and first-time starter (FTS) heuristics.

Strategy Profile: {profile}
IQ Mode: {"ON" if iq_mode else "OFF"}
{preset_line}

Track Bias Today: {', '.join(biases) if biases else 'fair/neutral'}
Surface: {surface_type} • Condition: (user-selected) • Race Type: {race_type}
{pace_shape_line}
{tailwind_block}

Scratches (exclude entirely): {scratches_txt}

{style_glossary}
Horse Running Styles (respect; adjust only if projection is obvious):
{running_styles_text or 'None detected'}

{angle_block}

{ped_block}

Past Performances (BRIS — single race):
{pp_text}

Return the result in this exact structure (concise bullets):
Race Summary – Track, race #, surface, distance, class, purse.
Pace Shape Analysis – Early/fast/honest/slow? Collapse risk? Who benefits?
Bias Fit – Inside/Mid/Outside posts; speed vs closers; cite strength tags if relevant.
Top Contenders – Rank 1–4 with one-line reasons (style • bias • form cycle • trainer/jockey • pedigree/FTS).
Fair Odds Line – Assign win % (sum ≤ 100%), and fair odds (AM).
Overlays / Underlays – Based on fair vs likely tote (use our board table if provided).
Ticket Builder –
• Win – horses with minimum acceptable odds.
• Exacta – keys/boxes (A/B structure).
• Trifecta – efficient structure (A with B/C; saver).
• Superfecta – A/B/C/D tiers (efficient coverage).
• Super High Five – compact A/B/C/D/E sketch if field size ≥ 7.
Pass/Press Guidance – When to pass; when to press if overlays ≥ {overlay_push_pp:.1f} percentage points.
Confidence Rating – 1–5 stars with a brief rationale (no sliders).
Rules: No scratched horses; adjust for bias/condition; respect listed Running Styles; apply FTS/MSW logic.
"""


go = st.button("Analyze this race", type="primary", use_container_width=True)

analysis_text = ""
fair_probs = {}

if go:
    if not pp_text.strip():
        st.warning("Please paste BRIS PPs for a single race.")
        st.stop()
    if len(race_headers) > 1:
        st.error("Multiple races detected — please paste only one race.")
        st.stop()

    user_prompt = build_user_prompt(pp_text=pp_text.strip())

    with st.spinner("Handicapping…"):
        try:
            messages = [
                {"role": "system",
                 "content": "You are a professional value-based handicapper. Focus on value, pace, bias, class, form cycles, trainer/jockey intent, pedigree tendencies, first-time starter logic, and efficient exotic structures."},
                {"role": "user", "content": user_prompt},
            ]
            analysis_text = call_openai_messages(messages)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

    st.success("Analysis complete.")
    st.markdown(analysis_text)
    st.download_button("Download analysis (.txt)", data=analysis_text,
                       file_name="horse_racing_analysis.txt",
                       mime="text/plain", use_container_width=True)

    # Parse Fair Odds Line (Horse — %)
    try:
        for line in analysis_text.splitlines():
            m = re.match(r'^\s*([A-Za-z0-9\'\"\.\-\s&]+?)\s+[–-]\s+(\d+(?:\.\d+)?)\s*%', line)
            if m:
                name = m.group(1).strip()
                p = float(m.group(2))/100.0
                fair_probs[name] = p
    except Exception:
        fair_probs = {}

# ===== Overlays (live if available, else ML) =====
st.markdown("### Overlays/Underlays")
board_dec = {}
for _, r in st.session_state.board_df.iterrows():
    h = str(r["Horse"])
    dec = use_dec_map.get(h)
    if dec: board_dec[h] = dec

if fair_probs:
    if board_dec:
        df_overlay = overlay_table(fair_probs, board_dec)
        st.dataframe(df_overlay, use_container_width=True, hide_index=True)
        st.download_button(
            "Download overlay table (CSV)",
            data=df_overlay.to_csv(index=False),
            file_name="overlays.csv",
            mime="text/csv",
            use_container_width=True
        )
    else:
        st.caption("We’ll compute overlays once Live Odds or Morning Line are present in the table above.")
else:
    st.caption("Run analysis first to generate a fair odds line; then overlays will calculate automatically.")

# ===================== Build tickets (optional) =====================
st.markdown("## Build tickets (optional)")
colW, colE = st.columns(2)
with colW:
    base_win = st.number_input("Base Win stake ($):", min_value=0.5, value=2.0, step=0.5)
    base_ex  = st.number_input("Exacta base ($):",   min_value=0.1, value=1.0, step=0.1)
    base_tri = st.number_input("Trifecta base ($):", min_value=0.1, value=0.5, step=0.1)
with colE:
    base_super = st.number_input("Superfecta base ($):", min_value=0.1, value=0.1, step=0.1)
    base_sh5   = st.number_input("Super High Five base ($):", min_value=0.1, value=0.1, step=0.1)

contenders = [h for h in kept_horses]
A = st.multiselect("Tier A — Top win candidates", contenders, default=contenders[:1] if contenders else [])
B = st.multiselect("Tier B — Win threats / strong underneath", [h for h in contenders if h not in A], default=[])
C = st.multiselect("Tier C — Underneath value (price)", [h for h in contenders if h not in A and h not in B], default=[])
D = st.multiselect("Tier D — Deep bombs (bottom slots)", [h for h in contenders if h not in A and h not in B and h not in C], default=[])

ex_cost  = base_ex  * (len(A)*len(B) + len(B)*len(A) + len(A)*max(len(A)-1,0)) if A and (B or len(A)>=2) else 0.0
tri_cost = base_tri * (len(A) * max(len(B),1) * max(len(C)+len(B),1)) if A else 0.0
sup_cost = superfecta_cost(len(A) or 0, max(len(B),1) if A else 0, max(len(C),1) if A else 0, max(len(D),1) if A else 0, base_super)
sh5_cost = super_high5_cost(len(A) or 0, max(len(B),1) if A else 0, max(len(C),1) if A else 0, max(len(D),1) if A else 0, max(len(contenders)- (len(A)+len(B)+len(C)+len(D)),1) if A else 0, base_sh5)

cost_df = pd.DataFrame([
    ["Win (per horse)", base_win],
    ["Exacta (est.)", round(ex_cost,2)],
    ["Trifecta (est.)", round(tri_cost,2)],
    ["Superfecta (est.)", round(sup_cost,2)],
    ["Super High Five (est.)", round(sh5_cost,2)],
], columns=["Bet", "Cost ($)"])
st.dataframe(cost_df, use_container_width=True, hide_index=True)
st.download_button(
    "Download ticket cost (CSV)",
    data=cost_df.to_csv(index=False),
    file_name="ticket_costs.csv",
    mime="text/csv",
    use_container_width=True
)

# ===================== Ledger (optional) =====================
st.markdown("## Track your bets (optional)")
if "ledger" not in st.session_state:
    st.session_state.ledger = []

with st.form("ledger_form"):
    lcol1, lcol2, lcol3 = st.columns([2,1,1])
    with lcol1: desc = st.text_input("Bet description", placeholder="Win: Ashkenazi @ 7/2")
    with lcol2: stake = st.number_input("Stake ($)", min_value=0.1, value=2.0, step=0.1)
    with lcol3: ret = st.number_input("Return ($)", min_value=0.0, value=0.0, step=0.1, help="Leave 0 if unsettled; update later.")
    add = st.form_submit_button("Add/Update")
    if add and desc:
        st.session_state.ledger.append({"desc": desc, "stake": float(stake), "return": float(ret)})

if st.session_state.ledger:
    led_df = pd.DataFrame(st.session_state.ledger)
    led_df["P/L"] = led_df["return"] - led_df["stake"]
    total_stake = led_df["stake"].sum()
    total_return = led_df["return"].sum()
    pl = total_return - total_stake
    roi = (total_return/total_stake - 1.0)*100 if total_stake>0 else 0.0
    st.dataframe(led_df, use_container_width=True, hide_index=True)
    c1,c2,c3,c4 = st.columns(4)
    c1.metric("Session Stake ($)", f"{total_stake:.2f}")
    c2.metric("Session Return ($)", f"{total_return:.2f}")
    c3.metric("Session P/L ($)", f"{pl:.2f}")
    c4.metric("Session ROI (%)", f"{roi:.2f}")

st.caption("Step 3 uses Live Odds when provided, otherwise Morning Line. Overlays/EV update automatically after analysis.")
