# app.py
# Horse Race Ready — IQ Mode 🔥 (Beginner-friendly + Pro controls)
# • Strict horse parsing incl. run style + ES points from (E/E-P/P/S N)
# • BRIS Ultimate PP signals: Prime Power, E1/E2/LP, RR/CR, ACL, Speed Last (+ z-scores)
# • Pedigree (AWD, Mud%, SPI/DPI), trainer intent, trip/ground, rider delta, Pace Pressure 2.0
# • Step 1: Surface / Distance / Race Type + Analyze button directly below PP box (no model branding)
# • Step 2: Styles/Bias/Scratches
# • Step 3: One odds table (ML auto + Live) → Overlays/EV + Reset buttons
# • Strategy Profiles: High IQ (default), Value, Aggressive
#   - Aggressive: adds a secondary "Creative Add-On" ticket set in analysis feedback
#   - Value: adds a "Value Plays Add-On" with price-sensitive alternatives
# • Downloadable analysis text (lists horses by number & name; no A/B/C tiers)

import os, re, json, math
from typing import List, Dict, Tuple
import numpy as np
import pandas as pd
import streamlit as st

# ===================== Page / Settings =====================
st.set_page_config(page_title="Horse Race Ready — IQ Mode", page_icon="🏇", layout="wide")
st.title("🏇 Horse Race Ready — IQ Mode")

# --- Glossary (compact) ---
with st.popover("Glossary"):
    st.markdown("""
**Running styles**: E (front), E/P (press), P (stalk), S (close).  
**ES points**: 0–8 early-speed points from PPs, e.g., **(E 7)**.  
**Overlay**: Our fair % > board implied % → value.  
**EV/$1**: Expected return per $1 at current odds (positive is good).
""")

# ---------------- API key (no branding in UI) ----------------
MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5")
TEMP  = float(st.secrets.get("OPENAI_TEMPERATURE", "0.4"))
OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")

use_sdk_v1 = True
client = None
if OPENAI_API_KEY:
    try:
        from openai import OpenAI
        client = OpenAI(api_key=OPENAI_API_KEY)
    except Exception:
        import openai
        openai.api_key = OPENAI_API_KEY
        use_sdk_v1 = False

def _supports_temp(name: str) -> bool:
    m = (name or "").lower()
    return not (m.startswith("gpt-5") or m.startswith("o4") or m.startswith("o3"))

def call_model(messages: List[Dict]) -> str:
    if not OPENAI_API_KEY:
        return "Add your API key in `.streamlit/secrets.toml` to enable analysis."
    if use_sdk_v1:
        try:
            kwargs = {"model": MODEL, "messages": messages}
            if _supports_temp(MODEL): kwargs["temperature"] = TEMP
            resp = client.chat.completions.create(**kwargs)
            return resp.choices[0].message.content
        except Exception as e:
            if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
                resp = client.chat.completions.create(model=MODEL, messages=messages)
                return resp.choices[0].message.content
            raise
    else:
        import openai
        try:
            kwargs = {"model": MODEL, "messages": messages}
            if _supports_temp(MODEL): kwargs["temperature"] = TEMP
            resp = openai.ChatCompletion.create(**kwargs)
            return resp["choices"][0]["message"]["content"]
        except Exception as e:
            if "temperature" in str(e).lower() and "unsupported" in str(e).lower():
                resp = openai.ChatCompletion.create(model=MODEL, messages=messages)
                return resp["choices"][0]["message"]["content"]
            raise

# ===================== Helpers =====================
def detect_valid_race_headers(pp_text: str):
    toks = ("purse", "furlong", "mile", "clm", "allow", "stake", "pars", "post time")
    headers = []
    for m in re.finditer(r"(?mi)^\s*Race\s+(\d+)\b", pp_text or ""):
        start = m.start()
        win = (pp_text[m.end():m.end()+250] or "").lower()
        if any(t in win for t in toks):
            headers.append((start, int(m.group(1))))
    return headers

# Strict horse header: "7 Camila Catalina (E 7)" or "(E/P 5)" or "(P 2)" or "(S 0)"
HORSE_HDR_RE = re.compile(
    r"(?mi)^\s*(\d+)\s+([-A-Za-z0-9'.&\s]+?)\s+\(\s*(E\/P|E|P|S)\s*(\d)\s*\)\s*$"
)

def split_into_horse_chunks(pp_text: str) -> List[Tuple[str, str, str]]:
    chunks=[]
    s = pp_text or ""
    matches=list(HORSE_HDR_RE.finditer(s))
    for i,m in enumerate(matches):
        start=m.end()
        end=matches[i+1].start() if i+1<len(matches) else len(s)
        post=m.group(1).strip(); name=m.group(2).strip()
        chunk=s[start:end]
        chunks.append((post,name,chunk))
    return chunks

def extract_horses_and_styles(pp_text: str) -> pd.DataFrame:
    rows=[]
    for m in HORSE_HDR_RE.finditer(pp_text or ""):
        rows.append({
            "#": m.group(1).strip(),
            "Horse": m.group(2).strip(),
            "DetectedStyle": m.group(3).replace("E/P","E/P"),
            "ES": int(m.group(4)),
            "OverrideStyle": "",
            "StyleStrength": "Solid",
        })
    # de-dupe
    seen=set(); uniq=[]
    for r in rows:
        k=(r["#"], r["Horse"].lower())
        if k not in seen:
            seen.add(k); uniq.append(r)
    return pd.DataFrame(uniq)

# Morning line odds inside each horse block (first odds-like token on a line)
ML_TOKEN_RE = re.compile(r'(?m)^\s*([0-9]+\/[0-9]+|[0-9]+-[0-9]+|\+\d+|-\d+|\d+(?:\.\d+)?)\s+')
def extract_morning_line_by_horse(pp_text: str) -> Dict[str, str]:
    ml={}
    for _, name, block in split_into_horse_chunks(pp_text):
        m = ML_TOKEN_RE.search(block or "")
        if m: ml[name]=m.group(1)
    return ml

# --- Trip/trouble → length credit (cap 3)
TRIP_DICT = {
    r'\bbrk out|break out|brk slow|bumped|brsh|check|steady|steadied': 1.0,
    r'\bbox|boxed|shut off|pinch': 1.0,
    r'\b3-5w|4w|5w|6w|wide|5w both turns|4w both turns': 1.5,
    r'\bin tight|traffic|split foes': 0.5,
    r'\blost whip|stumbled': 1.0,
}
def trip_credit(block:str)->float:
    b=(block or "").lower()
    credit=0.0
    for pat,val in TRIP_DICT.items():
        if re.search(pat, b): credit+=val
    return min(3.0, credit)

# Trainer intent: count bullet/1-of-x works
def trainer_intent(block:str)->float:
    cnt=0
    for _ in re.finditer(r'(?mi):\s*\d{2}(?:¨|«|ª|©)?\s+[BHg]?\s*1\/\d+', block or ""):
        cnt+=1
    works=len(re.findall(r'(?mi)\b\d+\s*(?:Jan|Feb|Mar|Apr|May|Jun|Jly|Jul|Aug|Sep|Oct|Nov|Dec)', block or ""))
    return min(1.0, 0.15*cnt + 0.02*works)

# Rider delta via "JKYw/ Sprints <N> <Win%>"
def rider_delta(block:str)->float:
    m=re.search(r'(?mi)JKYw\/\s*Sprints\s+\d+\s+(\d+)%', block or "")
    if not m: return 0.0
    win=int(m.group(1))
    if win>=22: return 0.08
    if win>=18: return 0.04
    if win<=8:  return -0.04
    return 0.0

# Pace Pressure 2.0 (field-wide)
def pace_pressure(styles: List[str]) -> Dict[str,float]:
    nE = sum(1 for s in styles if s=="E")
    nEP= sum(1 for s in styles if s=="E/P")
    nP = sum(1 for s in styles if s=="P")
    nS = sum(1 for s in styles if s=="S")
    field = nE+nEP+nP+nS
    entropy = 0.0
    for n in [nE,nEP,nP,nS]:
        if field>0 and n>0:
            p=n/field
            entropy += -p*math.log(p+1e-9)
    ppi = nE + 0.5*nEP
    return {"ppi": ppi, "entropy": entropy}

# Odds conversions
def am_to_dec(a: float)->float: return 1 + (a/100.0 if a>0 else 100.0/abs(a))
def str_to_dec(s: str)->float|None:
    s=(s or "").strip()
    if not s: return None
    if re.fullmatch(r'[+-]?\d+(\.\d+)?', s):
        v=float(s); return max(v,1.01)
    if re.fullmatch(r'\+\d+|-?\d+', s): return am_to_dec(float(s))
    if "-" in s:
        a,b=s.split("-",1)
        try: return float(a)/float(b)+1
        except: return None
    if "/" in s:
        a,b=s.split("/",1)
        try: return float(a)/float(b)+1
        except: return None
    return None

def dec_to_prob(d: float)->float: return 1.0/d if d and d>0 else 0.0
def fair_to_am(p: float)->float:
    if p<=0: return 9999
    if p>=1: return -100000
    d=1.0/p
    return round((d-1)*100,0) if d>=2 else round(-100/(d-1),0)

# Surface/distance/race-type helpers
def guess_surface(pp: str)->str:
    s=(pp or "").lower()
    if re.search(r'\bturf\b', s): return "Turf"
    if re.search(r'\b(tapeta|synthetic|poly)\b', s): return "Synthetic"
    return "Dirt"

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

# Race type choices (explicit list)
RACE_TYPE_CHOICES = [
    "Maiden Special Weight",
    "Maiden Claiming",
    "Claiming",
    "Optional Claiming",
    "Allowance",
    "Allowance Optional Claiming (AOC)",
    "Starter Allowance",
    "Listed Stakes",
    "Grade 3 (GIII)",
    "Grade 2 (GII)",
    "Grade 1 (GI)",
]
def autodetect_race_type(pp_text: str) -> str:
    s=(pp_text or "").lower()
    if re.search(r'\b(maiden special|mdn\s*sp|msw)\b', s):    return "Maiden Special Weight"
    if re.search(r'\b(maiden\s*claim|mdn\s*clm|mcl)\b', s):   return "Maiden Claiming"
    if re.search(r'\ballowance optional|aoc\b', s):           return "Allowance Optional Claiming (AOC)"
    if re.search(r'\bstarter\s*allow', s):                    return "Starter Allowance"
    if re.search(r'\ballow', s) and not re.search(r'\b(optional|starter)\b', s):
        return "Allowance"
    if re.search(r'\boptional\s*claim', s):                   return "Optional Claiming"
    if re.search(r'\bclaim(ing)?\b', s):                      return "Claiming"
    if re.search(r'\blisted\s*stakes?\b', s):                 return "Listed Stakes"
    if re.search(r'\b(gr(ade)?\s*3|giii)\b', s):              return "Grade 3 (GIII)"
    if re.search(r'\b(gr(ade)?\s*2|gii)\b', s):               return "Grade 2 (GII)"
    if re.search(r'\b(gr(ade)?\s*1|gi)\b', s):                return "Grade 1 (GI)"
    return "Allowance"

# ---------- BRIS figure parsing ----------
BRIS_PP_RE = {
    "prime_power": re.compile(r'(?mi)\bPrime\s*Power\b[:\-]?\s*(\d{2,3})'),
    "e1e2lp":      re.compile(r'(?mi)\bE1\s*(\d{2,3}).*?\bE2\s*(\d{2,3}).*?\bLP\s*(\d{2,3})'),
    "rrcr":        re.compile(r'(?mi)\bRR[:\s]*(\d{2,3}).*?\bCR[:\s]*(\d{2,3})'),
    "acl":         re.compile(r'(?mi)\b(?:Average|Avg)\s*Class\s*(?:Last\s*3|L3)\b[:\-]?\s*(\d{2,3})'),
    "speed_last":  re.compile(r'(?mi)\bSpeed\s*Last\s*Race\b[:\-]?\s*(\d{2,3})'),
}

def parse_bris_figures_from_block(block: str) -> Dict[str, float | None]:
    s = block or ""
    out = {"prime_power": None, "e1": None, "e2": None, "lp": None,
           "rr": None, "cr": None, "acl": None, "speed_last": None}
    m = BRIS_PP_RE["prime_power"].search(s)
    if m: out["prime_power"] = float(m.group(1))
    m = BRIS_PP_RE["e1e2lp"].search(s)
    if m: out["e1"], out["e2"], out["lp"] = map(lambda x: float(x), m.groups())
    m = BRIS_PP_RE["rrcr"].search(s)
    if m: out["rr"], out["cr"] = map(lambda x: float(x), m.groups())
    m = BRIS_PP_RE["acl"].search(s)
    if m: out["acl"] = float(m.group(1))
    m = BRIS_PP_RE["speed_last"].search(s)
    if m: out["speed_last"] = float(m.group(1))
    return out

def parse_bris_figures_for_field(pp_text: str) -> Dict[str, Dict[str, float | None]]:
    figs = {}
    for _, name, block in split_into_horse_chunks(pp_text):
        figs[name] = parse_bris_figures_from_block(block)
    return figs

# ========= Field-relative scaling =========
def _zscale(values: List[float]) -> Dict[int, float]:
    arr = np.array(values, dtype=float)
    mu = np.nanmean(arr); sd = np.nanstd(arr)
    if not np.isfinite(sd) or sd == 0:
        return {i: 0.0 for i in range(len(values))}
    return {i: float((arr[i]-mu)/sd) if np.isfinite(arr[i]) else 0.0 for i in range(len(values))}

def field_zscores(per_horse: Dict[str, Dict[str, float|None]], key: str) -> Dict[str, float]:
    names = list(per_horse.keys())
    vals  = [per_horse[n].get(key) if per_horse[n].get(key) is not None else np.nan for n in names]
    zmap_idx = _zscale(vals)
    return {names[i]: zmap_idx[i] for i in range(len(names))}

# ---------- Pedigree parsing ----------
PED_SIRE = re.compile(r"Sire\s*Stats:\s*AWD\s*(?P<sire_awd>\d+(?:\.\d+)?)\s*(?P<sire_mud>\d+)%Mud.*?\s*(?P<sire_first>\d+)%1st\s*(?P<sire_spi>\d+(?:\.\d+)?)spi", re.IGNORECASE)
PED_DAMSIRE = re.compile(r"Dam'?sSire:\s*AWD\s*(?P<damsire_awd>\d+(?:\.\d+)?)\s*(?P<damsire_mud>\d+)%Mud.*?\s*(?P<damsire_first>\d+)%1st\s*(?P<damsire_spi>\d+(?:\.\d+)?)spi", re.IGNORECASE)
PED_DAM = re.compile(r"Dam'?sStats:.*?(?P<dam_2yo>\d+)%2yo.*?(?P<dam_dpi>\d+(?:\.\d+)?)dpi", re.IGNORECASE)

def parse_pedigree_stats(block: str) -> Dict[str, float]:
    s = block or ""
    out={}
    ms = PED_SIRE.search(s); md = PED_DAMSIRE.search(s); dm = PED_DAM.search(s)
    if ms:
        out.update({"sire_awd": float(ms.group("sire_awd")),
                    "sire_mud": float(ms.group("sire_mud")),
                    "sire_first": float(ms.group("sire_first")),
                    "sire_spi": float(ms.group("sire_spi"))})
    if md:
        out.update({"damsire_awd": float(md.group("damsire_awd")),
                    "damsire_mud": float(md.group("damsire_mud")),
                    "damsire_first": float(md.group("damsire_first")),
                    "damsire_spi": float(md.group("damsire_spi"))})
    if dm:
        out.update({"dam_2yo": float(dm.group("dam_2yo")),
                    "dam_dpi": float(dm.group("dam_dpi"))})
    return out

# ===================== Sidebar =====================
with st.sidebar:
    st.header("Strategy Profile")
    profile = st.radio(
        "Profile:",
        ["High IQ", "Value", "Aggressive"],
        index=0,
        help="High IQ (default): best overall win/coverage mix • Value: hunt overlays • Aggressive: creative, higher-variance exotics"
    )
    # IQ Mode always ON
    iq_mode = st.toggle("IQ Mode (advanced heuristics)", value=True, disabled=True)
    beginner_mode = st.toggle("Beginner Mode (guided)", value=False)

# ===================== Step 1 — PPs + Race setup =====================
st.markdown("## Step 1 — Paste one race (PP text)")
pp_text = st.text_area("BRIS PPs:", height=260, placeholder="Paste the full text block of a single race…")

colA, colB, colC = st.columns([1.1,1,1])
with colA:
    surface_type = st.selectbox("Surface:", ["Dirt","Turf","Synthetic"], index=["Dirt","Turf","Synthetic"].index(guess_surface(pp_text)))
with colB:
    default_dist = ""
    m = re.search(r'(\d+\s*½?|\d+\.\d+)\s*Furlongs|\b1\s*Mile', pp_text or "", flags=re.IGNORECASE)
    if m: default_dist = m.group(0)
    distance_txt = st.text_input("Distance label:", value=default_dist or "6 Furlongs")
with colC:
    auto_rt = autodetect_race_type(pp_text)
    try:
        rt_index = RACE_TYPE_CHOICES.index(auto_rt)
    except ValueError:
        rt_index = RACE_TYPE_CHOICES.index("Allowance")
    race_type = st.selectbox("Race type:", RACE_TYPE_CHOICES, index=rt_index)

# Analyze button directly under Step 1
go = st.button("Analyze this race", type="primary", use_container_width=True)

# ===================== Step 2 — Styles / Bias / Scratches =====================
st.markdown("## Step 2 — Confirm scratches & running styles")
df_styles_full = extract_horses_and_styles(pp_text) if pp_text.strip() else pd.DataFrame()
detected_horses = list(df_styles_full["Horse"]) if not df_styles_full.empty else []
field_size = len(detected_horses)

bias_options = [
    "fair/neutral",
    "inside bias (posts 1-3)",
    "mid bias (posts 4-7)",
    f"outside bias ({'8+' if field_size>=8 else '8+'})",
    "favors early speed",
    "favors pressers (E/P)",
    "favors mid-pack (P)",
    "favors closers (S)",
]
biases = st.multiselect("Track bias today:", options=bias_options, default=["fair/neutral"])

st.markdown("**Scratches**")
scratched_by_pick = st.multiselect("Detected horses to scratch:", options=detected_horses, default=[])
scratches_manual = st.text_input("Or type numbers/names (comma or new line):", placeholder="e.g., 2, 7, Holiday Fantasy")
scratch_tokens=set()
if scratches_manual.strip():
    for tok in re.split(r"[,\n]+", scratches_manual):
        tok=tok.strip()
        if tok: scratch_tokens.add(tok.lower())
scratch_set = set([h.lower() for h in scratched_by_pick]) | scratch_tokens

df_styles = df_styles_full.copy()
if not df_styles.empty and scratch_set:
    mask = ~df_styles["Horse"].str.lower().isin(scratch_set) & ~df_styles["#"].astype(str).str.lower().isin(scratch_set)
    df_styles = df_styles[mask].reset_index(drop=True)

style_options=["","E","E/P","P","S"]
strength_options=["Strong","Solid","Slight","Weak","Bias"]
edited = st.data_editor(
    df_styles,
    column_config={
        "#": st.column_config.Column(disabled=True, width="small"),
        "Horse": st.column_config.Column(disabled=True),
        "DetectedStyle": st.column_config.Column(disabled=True),
        "ES": st.column_config.Column(disabled=True, help="Early Speed points (0–8) parsed from PPs"),
        "OverrideStyle": st.column_config.SelectboxColumn("OverrideStyle", options=style_options),
        "StyleStrength": st.column_config.SelectboxColumn("StyleStrength", options=strength_options),
    },
    hide_index=True, use_container_width=True, num_rows="fixed",
)
kept_horses=list(edited["Horse"]) if isinstance(edited,pd.DataFrame) and not edited.empty else []
styles = [(r.get("OverrideStyle") or r.get("DetectedStyle") or "NA") for _,r in edited.iterrows()] if not edited.empty else []
pace_env = pace_pressure([s for s in styles if s in ("E","E/P","P","S")])

# Surface condition input
if surface_type=="Dirt":
    surface_condition = st.selectbox("Condition:", ["fast","muddy","sloppy","wet-fast","good","off"], index=0)
elif surface_type=="Turf":
    surface_condition = st.selectbox("Condition:", ["firm","good","yielding","soft","off"], index=0)
else:
    surface_condition = st.selectbox("Condition:", ["fast","standard","wet"], index=0)

# ===================== Step 3 — Odds & Overlays =====================
st.markdown("## Step 3 — Odds & Overlays")
st.caption("Morning Line auto-fills from the pasted PPs. Enter Live Odds to update overlays. Formats: 7/2, 5-1, +250, or 3.8.")

# Reset buttons
r1, r2 = st.columns([1,1])
with r1:
    reset_board_btn = st.button("Reset odds table (clear)")
with r2:
    reset_all_btn = st.button("Reset all (rebuild odds & clear caches)")

if reset_all_btn:
    for k in ["board_df"]:
        if k in st.session_state: del st.session_state[k]

ml_map = extract_morning_line_by_horse(pp_text) if pp_text.strip() else {}
horses_for_board = [h for h in kept_horses]

if "board_df" not in st.session_state or reset_board_btn:
    st.session_state.board_df = pd.DataFrame({
        "Horse": pd.Series(horses_for_board, dtype="string"),
        "Morning Line": pd.Series([ml_map.get(h,"") for h in horses_for_board], dtype="string"),
        "Live Odds": pd.Series(["" for _ in horses_for_board], dtype="string"),
        "Use (dec)": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
        "Use %": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
    })
else:
    df=st.session_state.board_df
    if set(df["Horse"])!=set(horses_for_board):
        st.session_state.board_df = pd.DataFrame({
            "Horse": pd.Series(horses_for_board, dtype="string"),
            "Morning Line": pd.Series([ml_map.get(h,"") for h in horses_for_board], dtype="string"),
            "Live Odds": pd.Series(["" for _ in horses_for_board], dtype="string"),
            "Use (dec)": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
            "Use %": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
        })

board_df = st.data_editor(
    st.session_state.board_df,
    column_config={
        "Horse": st.column_config.Column(disabled=True),
        "Morning Line": st.column_config.TextColumn(help="Auto from PPs; edit if needed"),
        "Live Odds": st.column_config.TextColumn(help="Type odds; leave blank if none yet"),
        "Use (dec)": st.column_config.NumberColumn(format="%.3f", disabled=True),
        "Use %": st.column_config.NumberColumn(format="%.2f", disabled=True),
    },
    hide_index=True, use_container_width=True
)

# Compute Use(dec) and %
use_dec_map={}
for i,r in board_df.iterrows():
    h=str(r["Horse"])
    pick = (r.get("Live Odds") or "").strip() or (r.get("Morning Line") or "").strip()
    dec = str_to_dec(pick) if pick else None
    use_dec_map[h]=dec
    board_df.at[i,"Use (dec)"]=dec if dec else np.nan
    board_df.at[i,"Use %"]=dec_to_prob(dec)*100 if dec else np.nan
st.session_state.board_df=board_df

# ===================== Analysis (math + prompt) =====================
# Weight presets (subtle; tuned by surface/distance)
DEFAULT_WEIGHTS = {
    "global":{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":0.7,"workout_snap":0.8,"projection":1.0,"ft_firsters":1.0},
    ("Dirt","≤6f")   :{"pace_shape":1.2,"bias_fit":1.1,"class_form":1.0,"trs_jky":1.0,"pedigree":0.6,"workout_snap":0.7,"projection":1.05,"ft_firsters":1.1},
    ("Dirt","6.5–7f"):{"pace_shape":1.1,"bias_fit":1.05,"class_form":1.0,"trs_jky":1.0,"pedigree":0.7,"workout_snap":0.8,"projection":1.05,"ft_firsters":1.1},
    ("Dirt","8f+")   :{"pace_shape":1.0,"bias_fit":1.1,"class_form":1.15,"trs_jky":1.0,"pedigree":0.8,"workout_snap":0.8,"projection":1.0,"ft_firsters":0.95},
    ("Turf","≤6f")   :{"pace_shape":0.95,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":1.0,"workout_snap":0.7,"projection":1.0,"ft_firsters":1.15},
    ("Turf","6.5–7f"):{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.1,"trs_jky":1.0,"pedigree":1.0,"workout_snap":0.8,"projection":1.0,"ft_firsters":1.1},
    ("Turf","8f+")   :{"pace_shape":1.0,"bias_fit":1.1,"class_form":1.2,"trs_jky":1.1,"pedigree":1.1,"workout_snap":0.8,"projection":1.0,"ft_firsters":0.9},
    ("Synthetic","≤6f")   :{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":0.8,"workout_snap":0.7,"projection":1.0,"ft_firsters":1.1},
    ("Synthetic","6.5–7f"):{"pace_shape":1.0,"bias_fit":1.05,"class_form":1.0,"trs_jky":1.0,"pedigree":0.9,"workout_snap":0.7,"projection":1.0,"ft_firsters":1.0},
    ("Synthetic","8f+")   :{"pace_shape":0.98,"bias_fit":1.0,"class_form":1.1,"trs_jky":1.0,"pedigree":1.0,"workout_snap":0.8,"projection":1.0,"ft_firsters":0.9},
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

# Bias nudges
bias_mult={"pace":1.0,"bias":1.0}
if "favors early speed" in biases: bias_mult["pace"]*=1.05
if "favors pressers (E/P)" in biases: bias_mult["pace"]*=1.03
if "favors mid-pack (P)" in biases: bias_mult["pace"]*=1.02
if "favors closers (S)" in biases: bias_mult["pace"]*=0.98
if "inside bias (posts 1-3)" in biases: bias_mult["bias"]*=1.03
if any(x.startswith("outside bias") for x in biases): bias_mult["bias"]*=1.02

# Build blocks and parse per-horse signals
blocks = {name:block for _,name,block in split_into_horse_chunks(pp_text)}
scores={}; explain={}

# BRIS figures (field) + z-scores
bris_figs = parse_bris_figures_for_field(pp_text)
z_prime  = field_zscores(bris_figs, "prime_power")
z_acl    = field_zscores(bris_figs, "acl")
z_rr     = field_zscores(bris_figs, "rr")
z_cr     = field_zscores(bris_figs, "cr")
z_e1     = field_zscores(bris_figs, "e1")
z_e2     = field_zscores(bris_figs, "e2")
z_lp     = field_zscores(bris_figs, "lp")
z_speed  = field_zscores(bris_figs, "speed_last")

# ES points (0–8) from the header
ES_RE = re.compile(r'\((?:E\/P|E|P|S)\s*(\d)\)')
def es_points_from_block(block: str) -> int | None:
    m = ES_RE.search(block or "")
    return int(m.group(1)) if m else None

es_points_map: Dict[str, int | None] = {}
for h in kept_horses:
    es_points_map[h] = es_points_from_block(blocks.get(h,""))

# Distance presets
DB = distance_bucket(distance_txt)
BASE = get_weight_preset(surface_type, distance_txt)

# Strategy profile multipliers (leaning logic)
profile_tilts = {
    "High IQ":   dict(class_gain=0.08, pace_gain=0.02, late_gain=0.02, cap_low=0.93, cap_high=1.14),
    "Value":     dict(class_gain=0.05, pace_gain=0.01, late_gain=0.01, cap_low=0.95, cap_high=1.10),
    "Aggressive":dict(class_gain=0.06, pace_gain=0.03, late_gain=0.03, cap_low=0.92, cap_high=1.16),
}
pt = profile_tilts.get(profile, profile_tilts["High IQ"])

# Per-horse feature extraction + scoring
styles_map = {}
for _,row in edited.iterrows():
    h=row["Horse"]
    style=(row.get("OverrideStyle") or row.get("DetectedStyle") or "NA").strip()
    styles_map[h]=style
pace_env = pace_pressure([styles_map.get(h,"NA") for h in kept_horses if styles_map.get(h,"NA") in ("E","E/P","P","S")])

def zget(m,h): 
    try: return float(m.get(h,0.0))
    except: return 0.0

def es_nudge(h: str)->float:
    p = es_points_map.get(h, None)
    if p is None: return 0.0
    return (p - 3.5) * 0.012  # ~[-0.04, +0.06]

for _,row in edited.iterrows():
    h=row["Horse"]
    style=(row.get("OverrideStyle") or row.get("DetectedStyle") or "NA").strip()
    block=blocks.get(h,"")

    # Trainer / Trip / Rider
    t_intent = trainer_intent(block)
    t_credit = trip_credit(block)/3.0
    r_delta  = rider_delta(block)

    # Pedigree (wet + AWD fit) per-horse block
    ped = parse_pedigree_stats(block)
    wet_fit = 0.0
    if surface_type=="Dirt" and surface_condition in ("muddy","sloppy","wet-fast","good"):
        mud = max(ped.get("sire_mud",0.0), ped.get("damsire_mud",0.0))
        if mud>=18: wet_fit=+0.06
        elif mud<=8 and mud>0: wet_fit=-0.03

    awd_vals=[v for v in [ped.get("sire_awd"), ped.get("damsire_awd")] if isinstance(v,(int,float))]
    fit=0.0
    if awd_vals:
        awd=float(np.mean(awd_vals))
        if DB=="≤6f" and 6.0<=awd<=7.2: fit+=0.05
        if DB=="8f+" and awd>=7.2:      fit+=0.05
    else:
        awd=None

    # Pace environment impact
    pace_adj=0.0
    if style in ("P","S") and pace_env["ppi"]>=2.5: pace_adj+=0.05  # hot pace → closers
    if style=="E" and pace_env["ppi"]<=1.0: pace_adj+=0.05          # soft pace → lone E

    # Base multiplicative score (presets + bias)
    score = 1.0
    score *= BASE["pace_shape"] * bias_mult["pace"] * (1.0 + pace_adj)
    score *= BASE.get("bias_fit",1.0) * bias_mult["bias"]
    score *= BASE.get("class_form",1.0)
    score *= BASE.get("trs_jky",1.0)
    score *= BASE.get("pedigree",1.0)  * (1.0 + fit)
    score *= BASE.get("workout_snap",1.0)
    score *= BASE.get("projection",1.0) * (1.0 + t_intent + t_credit + r_delta + wet_fit)

    # BRIS-driven multipliers (class / pace / projection layers), with profile tilts
    cf_tilt = pt["class_gain"] * ( zget(z_prime, h) + 0.5*zget(z_acl, h) + 0.4*zget(z_cr, h) + 0.3*zget(z_rr, h) )
    class_form_mult = math.exp(cf_tilt)

    pace_tilt = pt["pace_gain"] * ( 0.6*zget(z_e2, h) + 0.4*zget(z_e1, h) )
    late_tilt = pt["late_gain"] * ( 0.6*zget(z_lp, h) + 0.4*zget(z_speed, h) )
    es_tilt   = es_nudge(h)

    bris_mult = math.exp(pace_tilt + late_tilt + es_tilt) * class_form_mult
    bris_mult = max(pt["cap_low"], min(pt["cap_high"], bris_mult))
    score *= bris_mult

    scores[h]=max(0.05, score)
    explain[h]=dict(
        style=style, es=es_points_map.get(h),
        t_intent=round(t_intent,3), trip=round(t_credit,3), r_delta=round(r_delta,3),
        wet=round(wet_fit,3), awd=awd if awd==awd else None, pace_adj=round(pace_adj,3),
        Prime=bris_figs.get(h,{}).get("prime_power"),
        E1E2LP="{}/{}/{}".format(bris_figs.get(h,{}).get("e1"), bris_figs.get(h,{}).get("e2"), bris_figs.get(h,{}).get("lp")),
        RRCR="{}/{}".format(bris_figs.get(h,{}).get("rr"), bris_figs.get(h,{}).get("cr")),
        ACL=bris_figs.get(h,{}).get("acl"),
        SLR=bris_figs.get(h,{}).get("speed_last"),
        BRIS_tilt=round((math.log(bris_mult) if bris_mult>0 else 0.0),3)
    )

# Convert scores → fair probabilities via softmax-like shaping
if scores:
    vals=np.array(list(scores.values()))
    temp = 1.05 if profile=="High IQ" else (1.15 if profile=="Value" else 1.10)
    probs=np.exp(vals/np.mean(vals)/temp)
    probs=probs/np.sum(probs)
    fair_probs={h:float(p) for h,p in zip(scores.keys(), probs)}
else:
    fair_probs={}

# ======= Display fair line =======
st.markdown("### Model fair line")
if fair_probs:
    fair_df=pd.DataFrame([
        {"Horse":h, "Fair %": round(p*100,2), "Fair (AM)": fair_to_am(p)} for h,p in sorted(fair_probs.items(), key=lambda x:-x[1])
    ])
    st.dataframe(fair_df, use_container_width=True, hide_index=True)
else:
    st.info("Paste PPs and confirm Step 2 to compute the fair line (then press Analyze).")

# ======= Overlays / EV =======
st.markdown("### Overlays / EV (win pool)")
board_dec={}
for _, r in st.session_state.board_df.iterrows():
    h=str(r["Horse"])
    dec = str_to_dec((r.get("Live Odds") or "").strip() or (r.get("Morning Line") or "").strip())
    if dec: board_dec[h]=dec

def overlay_table(fair_probs: Dict[str,float], offered: Dict[str,float]) -> pd.DataFrame:
    rows=[]
    for h,p in fair_probs.items():
        off_dec=offered.get(h)
        if off_dec is None: continue
        off_prob=dec_to_prob(off_dec)
        ev=(off_dec-1)*p-(1-p)
        rows.append({
            "Horse":h,"Fair %":round(p*100,2),"Fair (AM)":fair_to_am(p),
            "Board (dec)":round(off_dec,3),"Board %":round(off_prob*100,2),
            "Edge (pp)": round((p-off_prob)*100,2),"EV per $1": round(ev,3),
            "Overlay?":"YES" if p>off_prob else "No"
        })
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["Overlay?","EV per $1"], ascending=[False,False])

if fair_probs:
    if board_dec:
        df_overlay = overlay_table(fair_probs, board_dec)
        st.dataframe(df_overlay, use_container_width=True, hide_index=True)
        st.download_button("Download overlay table (CSV)",
                           data=df_overlay.to_csv(index=False),
                           file_name="overlays.csv",
                           mime="text/csv",
                           use_container_width=True)
    else:
        st.caption("Enter Live Odds or confirm Morning Line to compute overlays.")
else:
    st.caption("Run analysis first to generate a fair odds line.")

# ======= Explain signals =======
if explain:
    st.markdown("### Why the model likes them (signals)")
    exp_rows=[]
    for h,ex in explain.items():
        exp_rows.append({
            "Horse":h,"Style":ex["style"],"ES":ex["es"],"Trainer intent":ex["t_intent"],"Trip":ex["trip"],
            "Rider Δ":ex["r_delta"],"Wet fit":ex["wet"],"AWD":ex["awd"],"Pace adj":ex["pace_adj"],
            "Prime":ex["Prime"],"E1/E2/LP":ex["E1E2LP"],"RR/CR":ex["RRCR"],"ACL":ex["ACL"],"SLR":ex["SLR"],
            "BRIS tilt":ex["BRIS_tilt"]
        })
    st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)

# ======= Generate natural-language analysis (adds profile-specific add-ons) =======
def build_user_prompt():
    style_glossary = (
        "Running Style Glossary:\n"
        "• E (Early/Front-Runner); • E/P (Early/Presser); • P (Mid-pack); • S (Closer). "
        "ES points (0–8) indicate early speed intensity.\n"
    )
    scratches = [s for s in sorted(scratch_set)]
    scratches_txt = ", ".join(scratches) if scratches else "none"

    # Figure snippets block
    fig_lines=[]
    for h in kept_horses:
        f = bris_figs.get(h, {})
        if any(f.get(k) for k in ("prime_power","e1","e2","lp","rr","cr","acl","speed_last")):
            fig_lines.append(
                f"{h}: Prime {f.get('prime_power')}, E1/E2/LP {f.get('e1')}/{f.get('e2')}/{f.get('lp')}, "
                f"RR/CR {f.get('rr')}/{f.get('cr')}, ACL {f.get('acl')}, SLR {f.get('speed_last')}"
            )
    figs_block = "BRIS fig snippets:\n" + ("\n".join("• "+x for x in fig_lines) if fig_lines else "• (none)")

    # Bias text
    bias_txt = ", ".join(biases) if biases else "fair/neutral"

    # Running styles text
    rs_lines=[]
    for _,r in edited.iterrows():
        rs_lines.append(f"{r['#']} {r['Horse']} = {(r.get('OverrideStyle') or r.get('DetectedStyle') or 'NA')} ({r.get('StyleStrength') or 'Solid'}) ES:{r.get('ES')}")
    rs_text = "\n".join(rs_lines) if rs_lines else "None detected"

    # Fair line
    fair_lines=[]
    for h,p in sorted(fair_probs.items(), key=lambda x:-x[1]):
        fair_lines.append(f"- {h} — {round(p*100,2)}% — {fair_to_am(p)}")
    fair_block = "\n".join(fair_lines) if fair_lines else "(none)"

    # --- Profile-specific add-ons for ticket strategies ---
    profile_addon = ""
    if profile == "Aggressive":
        profile_addon = (
            "Aggressive Creativity Add-On – In addition to primary tickets, also include:\n"
            "• Trifecta bombs: Use a price horse (the two lowest Fair% non-scratched runners) in **3rd** with your top 2 in 1st/2nd.\n"
            "• Superfecta spreads: Top key on top; spread **five** in 3rd and **ALL remaining** in 4th (budget-aware).\n"
            "• Super High-5: Anchor 1st with top key; rotate two chalks in 2nd; **sprinkle two prices** in 3rd/4th; close with widest in 5th.\n"
            "• Lone-E or Lone-S exploit: If pace model flags soft/hot, promote the lone type **one slot up** in each exotic tier.\n"
        )
    elif profile == "Value":
        profile_addon = (
            "Value Plays Add-On – Emphasize overlays and price leverage:\n"
            "• Win bets only at/above fair thresholds + 10% cushion; avoid underlays completely.\n"
            "• Exacta: Key overlay **over** logicals; small saver **under** logical favorite.\n"
            "• Tri/Super: Include overlays **in every tier**; exclude short-priced underlays from top.\n"
        )
    else:  # High IQ
        profile_addon = (
            "High-IQ Press Plan – Keep efficient and press when edge is real:\n"
            "• Increase stake when overlay edge ≥ 3–5 pp and EV>$0.\n"
            "• Keep exotics compact; use price horses for **3rd/4th only** unless pace/bias screams otherwise.\n"
        )

    return f"""
You are an elite, value-driven handicapper. Use pace shape (E1/E2/LP + ES points & field pressure), class (Prime Power, RR/CR, ACL),
bias (post/run-style), form cycles, trainer/jockey intent, trip/ground-loss, wet pedigree (Mud%), and AWD sprint/route fit. 
Incorporate SPI/DPI notes implicitly.

Profile: {profile}
IQ Mode: ON

Track Bias Today: {bias_txt}
Surface: {surface_type} ({surface_condition}) • Distance: {distance_txt} • Race Type: {race_type}
Scratches: {scratches_txt}

Running Styles (ES points in parentheses):
{rs_text}

{figs_block}

Model Fair Odds Line (guidance):
{fair_block}

Return the result in this exact structure (concise bullets, list horses by **number and name**, no A/B/C labels):
Race Summary – Track, race #, surface, distance, class, purse.
Pace Shape – Early/honest/fast? Collapse risk? Who benefits (by # and name).
Bias Fit – Inside/Mid/Outside & run-style impact; mention ES points when relevant.
Top Contenders (value-weighted) – Rank 1–4 with one-line reasons (style • bias • form/figs • trainer/jockey • pedigree/FTS).
Fair Odds Line – Win % (sum ≤ 100%) + fair odds (AM) listed by # and name.
Overlays / Underlays – Based on fair vs board; specify playable thresholds.
Ticket Builder – **Win**, **Exacta**, **Trifecta**, **Superfecta**, **Super High Five** (compact, efficient).
{profile_addon}
Pass/Press Guidance – When to pass; when to press based on overlay size.
Confidence – 1–5 stars.
Rules: Exclude scratched; respect listed styles; adjust for bias/condition; emphasize value.
"""

if go:
    if not OPENAI_API_KEY:
        st.error("Add your API key to `.streamlit/secrets.toml` to enable the analysis button.")
    elif not pp_text.strip():
        st.warning("Please paste BRIS PPs for a single race.")
    else:
        user_prompt = build_user_prompt()
        with st.spinner("Handicapping…"):
            try:
                messages=[
                    {"role":"system","content":"You are a professional value-based handicapper. Focus on value, pace, bias, class, BRIS figs (Prime, E1/E2/LP, RR/CR, ACL), form cycles, trainer/jockey intent, pedigree tendencies (Mud%/AWD/SPI/DPI), and efficient exotic structures. List horses by number and name. Include Aggressive/Value add-ons when requested."},
                    {"role":"user","content":user_prompt},
                ]
                analysis_text = call_model(messages)
            except Exception as e:
                st.error(f"Error: {e}")
                analysis_text = ""

        if analysis_text:
            st.success("Analysis complete.")
            st.markdown(analysis_text)
            st.download_button("Download analysis (.txt)", data=analysis_text, file_name="horse_racing_analysis.txt", mime="text/plain", use_container_width=True)






