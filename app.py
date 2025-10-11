# app.py
# Horse Race Ready — IQ Mode (concise build)
# - Robust horse/ML extraction (incl. NA/FTS)
# - Auto race-factor weights + auto angle scan
# - Odds table type-safe; Reset button
# - Overlays auto: Tote if present else Morning Line
# - Confidence band fixed (no sliders)
# - Tickets + ROI tracker
# --------------------------------------------------

import os, re, json
from typing import List, Dict
import pandas as pd
import numpy as np
import streamlit as st

# ============== Page / Model ==============
st.set_page_config(page_title="Horse Race Ready — IQ Mode", page_icon="🏇", layout="wide")
st.title("🏇 Horse Race Ready — IQ Mode")

with st.popover("Glossary"):
    st.markdown(
        "E=Early • E/P=Early/Presser • P=Stalker • S=Closer  \n"
        "Bias: Inside(1–3) / Mid(4–7) / Outside(8+)  \n"
        "Overlay: board implied % < fair %  \n"
        "EV/$1=(dec-1)*p - (1-p)"
    )

MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5")
TEMP  = float(st.secrets.get("OPENAI_TEMPERATURE", "0.5"))
CONF_BAND_POINTS = float(st.secrets.get("CONF_BAND_POINTS", "3.5"))  # fixed band

OPENAI_API_KEY = st.secrets.get("OPENAI_API_KEY") or os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    st.error("Add OPENAI_API_KEY to .streamlit/secrets.toml")
    st.stop()

use_sdk_v1 = True
try:
    from openai import OpenAI
    client = OpenAI(api_key=OPENAI_API_KEY)
except Exception:
    import openai
    openai.api_key = OPENAI_API_KEY
    use_sdk_v1 = False

def model_supports_temperature(m: str) -> bool:
    m = (m or "").lower()
    return not (m.startswith("gpt-5") or m.startswith("o4") or m.startswith("o3"))

def call_openai_messages(messages: List[Dict]) -> str:
    if use_sdk_v1:
        try:
            kw = {"model": MODEL, "messages": messages}
            if model_supports_temperature(MODEL): kw["temperature"] = TEMP
            r = client.chat.completions.create(**kw)
            return r.choices[0].message.content
        except Exception as e:
            if "temperature" in str(e).lower():
                r = client.chat.completions.create(model=MODEL, messages=messages)
                return r.choices[0].message.content
            raise
    else:
        try:
            kw = {"model": MODEL, "messages": messages}
            if model_supports_temperature(MODEL): kw["temperature"] = TEMP
            r = openai.ChatCompletion.create(**kw)
            return r["choices"][0]["message"]["content"]
        except Exception as e:
            if "temperature" in str(e).lower():
                r = openai.ChatCompletion.create(model=MODEL, messages=messages)
                return r["choices"][0]["message"]["content"]
            raise

# ============== Helpers ==============
def detect_valid_race_headers(txt: str):
    toks=("purse","furlong","mile","clm","allow","stake","pars","post time")
    out=[]
    for m in re.finditer(r"(?mi)^\s*Race\s+(\d+)\b", txt or ""):
        win=(txt[m.end():m.end()+250] or "").lower()
        if any(t in win for t in toks): out.append((m.start(), int(m.group(1))))
    return out

# Stronger horse header parser: handles "7 Camila Catalina (P 0)", "4 Justice Addition (NA 0)", etc.
HORSE_LINE = re.compile(
    r"(?m)^\s*(\d{1,2})\s+([A-Za-z0-9’'\.\-\s/&]+?)\s+\(\s*(E\/P|E|P|S|NA)\s*\d*\s*\)"
)

def extract_horses_and_styles(txt: str) -> pd.DataFrame:
    rows=[]
    for m in HORSE_LINE.finditer(txt or ""):
        post=m.group(1).strip()
        name=re.sub(r'\s+', ' ', m.group(2).strip())
        style=m.group(3).replace("E/P","E/P")
        rows.append({"#":post,"Horse":name,"DetectedStyle":"NA" if style=="NA" else style,
                     "OverrideStyle":"","StyleStrength":"Solid"})
    # de-dupe
    seen=set(); uniq=[]
    for r in rows:
        k=(r["#"], r["Horse"].lower())
        if k not in seen:
            seen.add(k); uniq.append(r)
    return pd.DataFrame(uniq)

# Morning line parser: looks for odds token near each horse header line
def _token_to_dec(tok: str) -> float|None:
    tok=tok.strip()
    if re.fullmatch(r'[+-]?\d+(\.\d+)?', tok):
        v=float(tok); return max(v,1.01)
    if re.fullmatch(r'\+\d+', tok) or re.fullmatch(r'-\d+', tok):
        a=float(tok); return 1 + (a/100.0 if a>0 else 100.0/abs(a))
    if "/" in tok or "-" in tok:
        a,b=re.split(r'[/\-]', tok, 1)
        try: return float(a)/float(b)+1.0
        except: return None
    return None

def extract_morning_line(txt: str) -> Dict[str,float]:
    ml={}
    # scan per horse block: capture odds immediately after owner/color lines OR next numeric token like "9/2", "5-1", "+250"
    for m in HORSE_LINE.finditer(txt or ""):
        name=re.sub(r'\s+', ' ', m.group(2).strip())
        start=m.end()
        window=(txt[start:start+220] or "")
        # take first odds-like token
        om = re.search(r'(?:(?:\b\d{1,2}\/\d{1,2}\b)|(?:\b\d{1,2}-\d{1,2}\b)|(?:[+-]\d{2,4})|(?:\b\d+\.\d+\b))', window)
        if om:
            dec=_token_to_dec(om.group(0))
            if dec: ml[name]=dec
    return ml

def parse_stat_table_lines(lines: List[str]) -> pd.DataFrame:
    rows=[]
    for raw in lines:
        s=re.sub(r'\s+',' ', (raw or "").strip())
        if not s: continue
        m=re.match(r'(.+?)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)$', s)
        if m:
            rows.append({"Category":m.group(1),"Starts":int(m.group(2)),"Win%":float(m.group(3)),
                         "ITM%":float(m.group(4)),"ROI":float(m.group(5))})
    return pd.DataFrame(rows)

def _format_running_styles(df: pd.DataFrame) -> str:
    if df is None or df.empty: return "None detected"
    out=[]
    for _,r in df.iterrows():
        sty=(r.get("OverrideStyle") or r.get("DetectedStyle") or "P").strip()
        out.append(f"{r['Horse']} = {sty} ({r.get('StyleStrength','Solid')})")
    return "\n".join(out) if out else "None detected"

# Weight presets (unchanged)
DEFAULT_WEIGHTS = {
    "global":{"pace_shape":1.0,"bias_fit":1.1,"class_form":1.0,"trs_jky":0.9,"pedigree":0.6,"workout_snap":0.7,"projection":1.0,"ft_firsters":1.0},
    ("Dirt","≤6f") :{"pace_shape":1.2,"bias_fit":1.2,"class_form":1.0,"trs_jky":0.9,"pedigree":0.5,"workout_snap":0.6,"projection":1.1,"ft_firsters":1.1},
    ("Dirt","6.5–7f") :{"pace_shape":1.1,"bias_fit":1.1,"class_form":1.0,"trs_jky":1.0,"pedigree":0.6,"workout_snap":0.7,"projection":1.1,"ft_firsters":1.1},
    ("Dirt","8f+") :{"pace_shape":1.0,"bias_fit":1.1,"class_form":1.2,"trs_jky":1.0,"pedigree":0.7,"workout_snap":0.7,"projection":1.0,"ft_firsters":0.9},
    ("Turf","≤6f") :{"pace_shape":0.9,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":0.9,"workout_snap":0.6,"projection":1.0,"ft_firsters":1.2},
    ("Turf","6.5–7f") :{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.1,"trs_jky":1.0,"pedigree":0.9,"workout_snap":0.7,"projection":1.0,"ft_firsters":1.1},
    ("Turf","8f+") :{"pace_shape":1.0,"bias_fit":1.1,"class_form":1.2,"trs_jky":1.1,"pedigree":1.0,"workout_snap":0.7,"projection":1.0,"ft_firsters":0.8},
    ("Synthetic","≤6f") :{"pace_shape":1.0,"bias_fit":1.0,"class_form":1.0,"trs_jky":1.0,"pedigree":0.7,"workout_snap":0.6,"projection":1.0,"ft_firsters":1.1},
    ("Synthetic","6.5–7f") :{"pace_shape":1.0,"bias_fit":1.1,"class_form":1.0,"trs_jky":1.0,"pedigree":0.8,"workout_snap":0.6,"projection":1.0,"ft_firsters":1.0},
    ("Synthetic","8f+") :{"pace_shape":0.9,"bias_fit":1.0,"class_form":1.1,"trs_jky":1.0,"pedigree":0.9,"workout_snap":0.7,"projection":1.0,"ft_firsters":0.8},
}

def distance_bucket(s: str) -> str:
    s=(s or "").lower()
    if any(k in s for k in ["6½","6 1/2","6.5"]): return "6.5–7f"
    if "furlong" in s:
        m=re.search(r'(\d+(?:\.\d+)?)\s*furlong', s); 
        if m:
            v=float(m.group(1))
            return "≤6f" if v<=6 else ("6.5–7f" if v<=7 else "8f+")
        return "≤6f"
    if "mile" in s: return "8f+"
    return "≤6f"

def get_weight_preset(surface: str, distance_txt: str) -> Dict[str,float]:
    return DEFAULT_WEIGHTS.get((surface, distance_bucket(distance_txt)), DEFAULT_WEIGHTS["global"])

def compute_post_bias_label(n:int)->str: return f"Inside:1-3 | Mid:4-7 | Outside:{'8+' if n>=8 else '8+'}"

# Odds utils
def dec_to_prob(d: float) -> float: return 1.0/d if d and d>0 else 0.0
def fair_to_american(p: float) -> float:
    if p<=0: return float("inf")
    if p>=1: return 0.0
    dec=1.0/p
    return round((dec-1)*100,0) if dec>=2 else round(-100/(dec-1),0)

def str_to_decimal_odds(s: str) -> float|None:
    return _token_to_dec((s or "").strip())

def overlay_table(fair: Dict[str,float], offered: Dict[str,float]) -> pd.DataFrame:
    rows=[]
    for h,p in fair.items():
        if h not in offered: continue
        d=offered[h]; imp=dec_to_prob(d); ev=(d-1)*p - (1-p)
        rows.append({"Horse":h,"Fair %":round(p*100,2),"Fair (AM)":fair_to_american(p),
                     "Board (dec)":round(d,3),"Board %":round(imp*100,2),
                     "Edge (Board%-Fair%)":round((imp-p)*100,2),"EV per $1":round(ev,3),
                     "Overlay?":"YES" if imp<p else "No"})
    if not rows: return pd.DataFrame()
    return pd.DataFrame(rows).sort_values(by=["Overlay?","EV per $1"], ascending=[False,False])

def superfecta_cost(a:int,b:int,c:int,d:int, base:float)->float: return base*a*b*c*d
def super_high5_cost(a:int,b:int,c:int,d:int,e:int, base:float)->float: return base*a*b*c*d*e

# Race factors
FACTOR_OPTIONS=[
    "Speed-favoring bias today","Closer-friendly / tiring speed","Outside posts advantaged",
    "Rail dead / inside dull","Vulnerable favorite","Chaos pace (many E7/E8)",
    "Pace meltdown risk","Many first-time starters (FTS)","Small field (≤6)","Big field (≥10)"
]
def adjust_weights_by_factors(w: Dict[str,float], fs: List[str]) -> Dict[str,float]:
    w=w.copy(); bump=lambda k,m: w.__setitem__(k, round(w.get(k,1.0)*m,3))
    for f in fs:
        if f=="Speed-favoring bias today": bump("bias_fit",1.12); bump("pace_shape",1.06)
        if f=="Closer-friendly / tiring speed": bump("bias_fit",1.10); bump("pace_shape",0.96)
        if f in ("Outside posts advantaged","Rail dead / inside dull"): bump("bias_fit",1.06)
        if f=="Vulnerable favorite": bump("projection",1.08)
        if f=="Chaos pace (many E7/E8)": bump("pace_shape",1.10); bump("projection",1.06)
        if f=="Pace meltdown risk": bump("pace_shape",1.06); bump("projection",1.06)
        if f=="Many first-time starters (FTS)": bump("ft_firsters",1.12); bump("projection",1.05)
        if f=="Small field (≤6)": bump("pace_shape",0.96); bump("class_form",1.04)
        if f=="Big field (≥10)": bump("pace_shape",1.05); bump("projection",1.05)
    return w

# Auto angle scan (includes NA/FTS / Maiden Sp Wt keywords)
_AUTO_ANGLE = re.compile(
    r"(?mi)^\s*(202\d|1st\s*time\s*str|Debut\s*MdnSpWt|Maiden\s*Sp\s*Wt)\s+"
    r"(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$"
)
def scan_auto_angles(txt: str) -> pd.DataFrame:
    rows=[]
    for m in _AUTO_ANGLE.finditer(txt or ""):
        cat,st,wn,itm,roi=m.groups()
        rows.append({"Category":cat.strip(),"Starts":int(st),"Win%":float(wn),
                     "ITM%":float(itm),"ROI":float(roi)})
    if not rows: return pd.DataFrame()
    df=pd.DataFrame(rows)
    g=(df.groupby("Category", as_index=False)
         .agg(Starts=("Starts","sum"), **{"Win%":("Win%","mean"),"ITM%":("ITM%","mean"),"ROI":("ROI","mean")}))
    g["Source"]="auto"
    return g

def incorporate_angles_into_weights(w: Dict[str,float], auto: pd.DataFrame)->Dict[str,float]:
    w=w.copy()
    if auto is None or auto.empty: return w
    msw=auto[auto["Category"].str.contains("Mdn|Maiden", case=False, regex=True)]
    if not msw.empty and ((msw["Win%"].mean()>=8.0) or (msw["ROI"].mean()>-0.6)):
        w["ft_firsters"]=round(w.get("ft_firsters",1.0)*1.08,3)
        w["projection"]=round(w.get("projection",1.0)*1.04,3)
    return w

# ============== Sidebar (no sliders) ==============
with st.sidebar:
    st.header("Strategy Profile")
    profile = st.radio("Profile:", ["Value Hunter","Balanced","Exotic Builder","Steady Chalk"], index=0)
    iq_mode = st.toggle("IQ Mode (advanced heuristics, FTS logic, presets)", value=True)
    beginner_mode = st.toggle("Beginner Mode (guided)", value=False)
    st.caption(f"Confidence band is automatic (±{CONF_BAND_POINTS:.1f} pts). "
               "Overlays computed from live tote if entered, else Morning Line.")

conf_band = CONF_BAND_POINTS  # fixed

# ============== Main Flow ==============
st.markdown("## Step 1 — Paste one race (PP text)")
pp_text = st.text_area("BRIS PPs:", height=260, placeholder="Paste the full text block of a single race…")

race_headers = detect_valid_race_headers(pp_text) if pp_text.strip() else []
if pp_text.strip():
    if len(race_headers)==0: st.info("No explicit 'Race #' header detected — OK if it’s a single race block.")
    elif len(race_headers)>1:
        st.error(f"Detected {len(race_headers)} races — paste only one.")
        st.stop()

# Surface / distance / notes
cA,cB,cC = st.columns([1.1,1,1])
with cA:
    surface_type = st.selectbox("Surface:", ["Dirt","Turf","Synthetic"], index=0)
with cB:
    default_dist=""
    m=re.search(r'((\d+\s*½?|\d+\.\d+)\s*Furlongs|\b1\s*Mile)', pp_text or "", flags=re.IGNORECASE)
    if m: default_dist=m.group(0)
    distance_txt = st.text_input("Distance label:", value=default_dist or "6 Furlongs")
with cC:
    # race factors (auto suggest FTS if MSW)
    factor_choices = st.multiselect(
        "Race factors (check any):",
        FACTOR_OPTIONS,
        default=(["Many first-time starters (FTS)"] if any(x in (pp_text or "") for x in ["Mdn","MSW","Maiden"]) else [])
    )
    ml_context_free = st.text_input("Add your own note (optional):", placeholder="e.g., vulnerable fave; outside sharp; off-turf watch…")

# Step 2 — Styles / Bias / Scratches
st.markdown("## Step 2 — Confirm scratches & running styles")
df_styles_full = extract_horses_and_styles(pp_text) if pp_text.strip() else pd.DataFrame()
detected_horses = list(df_styles_full["Horse"]) if not df_styles_full.empty else []
if pp_text.strip() and not detected_horses:
    st.warning("Couldn’t find numbered entry lines; paste the section with horse list.")
field_size = len(detected_horses)

c1,c2 = st.columns([1.2,1])
with c1:
    bias_options=["favors speed","favors stalkers","favors closers",
                  "inside bias (posts 1-3)","mid bias (posts 4-7)",
                  f"outside bias ({'8+' if field_size>=8 else '8+'})",
                  "tiring speed","fair/neutral"]
    biases = st.multiselect("Track bias today:", options=bias_options, default=["fair/neutral"])
with c2:
    st.write("**Post Bias Key**")
    st.info(compute_post_bias_label(field_size or 8))

st.markdown("**Scratches**")
scratched_by_pick = st.multiselect("Detected horses to scratch:", options=detected_horses, default=[])
scratches_manual = st.text_input("Or type numbers/names (comma or new line):",
                                 placeholder="e.g., 2, 7, Holiday Fantasy")
scratch_set=set([h.strip().lower() for h in scratched_by_pick])
if scratches_manual.strip():
    for tok in re.split(r"[,\n]+", scratches_manual):
        t=tok.strip()
        if t: scratch_set.add(t.lower())

df_styles=df_styles_full.copy()
if not df_styles.empty and scratch_set:
    mask = ~df_styles["Horse"].str.lower().isin(scratch_set) & ~df_styles["#"].astype(str).str.lower().isin(scratch_set)
    df_styles = df_styles[mask].reset_index(drop=True)

edited = st.data_editor(
    df_styles,
    column_config={
        "#": st.column_config.Column(disabled=True, width="small"),
        "Horse": st.column_config.Column(disabled=True),
        "DetectedStyle": st.column_config.Column(disabled=True, help="Parsed from horse header"),
        "OverrideStyle": st.column_config.SelectboxColumn("OverrideStyle", options=["","E","E/P","P","S","NA"]),
        "StyleStrength": st.column_config.SelectboxColumn("StyleStrength", options=["Strong","Solid","Slight","Weak","Bias"]),
    },
    hide_index=True, use_container_width=True, num_rows="fixed",
)
st.caption("E=Early; E/P=Early/Presser; P=Mid-pack; S=Closer; NA=Unknown.")

running_styles_text=_format_running_styles(edited)
scratches_list=sorted(scratch_set)

# Surface condition
if surface_type=="Dirt":
    surface_condition=st.selectbox("Condition:", ["fast","muddy","sloppy","wet-fast","good","off"], index=0)
elif surface_type=="Turf":
    surface_condition=st.selectbox("Condition:", ["firm","good","yielding","soft","off"], index=0)
else:
    surface_condition=st.selectbox("Condition:", ["fast","standard","wet"], index=0)

# Step 3 — Angle stats (USER + AUTO)
st.markdown("## Step 3 — (Optional) add angle stats")
st.caption("Paste rows: `Category  Starts  Win%  ITM%  ROI`")
angles_raw = st.text_area("Angle rows:", height=120, placeholder="1st time str   146   3%   25%   -1.22")
user_angles_df = parse_stat_table_lines(angles_raw.split("\n")) if angles_raw.strip() else pd.DataFrame()
if not user_angles_df.empty: user_angles_df["Source"]="user"
auto_angles_df = scan_auto_angles(pp_text) if pp_text.strip() else pd.DataFrame()
angles_df = pd.concat([auto_angles_df, user_angles_df], ignore_index=True) if (not auto_angles_df.empty or not user_angles_df.empty) else pd.DataFrame()
if not angles_df.empty: st.dataframe(angles_df, use_container_width=True)

# Step 4 — Live Tote / Board
st.markdown("## Step 4 — (Optional) enter current odds (from the tote)")
st.caption("Type odds like 7/2, 5-1, +250, or 4.6; we convert automatically.")
cA,cB=st.columns([1,1])
with cA:
    reset_board = st.button("Reset odds table (clear)")
if reset_board:
    if "board_df" in st.session_state: del st.session_state["board_df"]

board_mode = st.radio("How do you want to enter odds?", ["Quick paste (free text)","Table editor (per horse)"], index=1, horizontal=True)
board_dec: Dict[str,float] = {}

if board_mode=="Quick paste (free text)":
    tote_raw = st.text_area("Board odds (one per line: Horse = odds):", height=120,
                            placeholder="Ashkenazi = 7/2\nCinnamon Sugar = 9/2\nBlazing Freedom = +300")
    def parse_board(block: str)->Dict[str,float]:
        out={}
        for line in (block or "").split("\n"):
            if "=" not in line: continue
            name,od=line.split("=",1)
            d=str_to_decimal_odds(od.strip())
            if d: out[name.strip()]=d
        return out
    board_dec = parse_board(tote_raw)
else:
    horses_for_board=[h for h in detected_horses if h.lower() not in scratch_set]
    if "board_df" not in st.session_state or reset_board:
        st.session_state.board_df=pd.DataFrame({
            "Horse": pd.Series(horses_for_board, dtype="string"),
            "Odds (enter any format)": pd.Series(["" for _ in horses_for_board], dtype="string"),
            "Decimal": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
            "Board %": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
        })
    df_in=st.session_state.board_df.copy()
    df_in["Odds (enter any format)"]=df_in["Odds (enter any format)"].astype("string")
    df_in["Horse"]=df_in["Horse"].astype("string")
    edit_df=st.data_editor(
        df_in,
        column_config={
            "Horse": st.column_config.Column(disabled=True),
            "Odds (enter any format)": st.column_config.TextColumn(help="Type odds; leave blank if no read"),
            "Decimal": st.column_config.NumberColumn(format="%.3f", disabled=True),
            "Board %": st.column_config.NumberColumn(format="%.2f", disabled=True),
        },
        hide_index=True, use_container_width=True
    )
    decs,probs=[],[]
    board_dec={}
    for _,r in edit_df.iterrows():
        d=str_to_decimal_odds(r.get("Odds (enter any format)",""))
        p=dec_to_prob(d) if d else np.nan
        decs.append(d if d else np.nan)
        probs.append(p*100 if d else np.nan)
        if d: board_dec[str(r["Horse"])]=d
    edit_df["Decimal"]=decs; edit_df["Board %"]=probs
    st.session_state.board_df=edit_df
    st.dataframe(edit_df, use_container_width=True, hide_index=True)

# Step 5 — Analyze
st.markdown("## Step 5 — Run analysis")
base_weights = get_weight_preset(surface_type, distance_txt) if iq_mode else DEFAULT_WEIGHTS["global"]
factored = adjust_weights_by_factors(base_weights, factor_choices)
factored = incorporate_angles_into_weights(factored, auto_angles_df)
weight_preset = factored
first_time_starter_hint = any(x in (pp_text or "") for x in ["Mdn","MSW","Maiden"])

def build_user_prompt(pp_text: str, biases, surface_type: str, surface_condition: str,
                      scratches_list, running_styles_text: str, ml_context: str,
                      profile: str, angles_df: pd.DataFrame, iq_mode: bool,
                      weight_preset: Dict[str, float], factor_choices: List[str], auto_angles_df: pd.DataFrame):
    style_glossary=("Running Style Glossary:\n"
                    "• E (Early) • E/P (Early/Presser) • P (Stalker) • S (Closer)\n")
    scratches_txt=", ".join(scratches_list) if scratches_list else "none"
    angles_block=""
    if angles_df is not None and not angles_df.empty:
        angles_block="Angle/Stat Table (Category • Starts • Win% • ITM% • ROI):\n" + \
            "\n".join([f"• {r.Category} • {int(r.Starts)} • {r['Win%']:.1f}% • {r['ITM%']:.1f}% • {r.ROI:+.2f}"
                       for _,r in angles_df.iterrows()])
    auto_note="AUTO angles fed into weights.\n" if (auto_angles_df is not None and not auto_angles_df.empty) else ""
    factors_txt=", ".join(factor_choices) if factor_choices else "none"
    return f"""
You are an elite, value-driven handicapper. Emphasize pace/bias interactions, class & form cycles,
trainer/jockey intent, pedigree tendencies, workout signals, and FTS heuristics.

Strategy Profile: {profile}
IQ Mode: {"ON" if iq_mode else "OFF"}
Effective Weights: {json.dumps(weight_preset)}

Track Bias Today: {', '.join(biases) if biases else 'fair/neutral'}
Surface: {surface_type} ({surface_condition})
Factors: {factors_txt}
Scratches: {scratches_txt}

{style_glossary}
Horse Running Styles (respect unless projection is obvious):
{running_styles_text}

{auto_note}User note: {ml_context or 'N/A'}

{angles_block}

Past Performances (BRIS — single race):
{pp_text}

Return structure:
Race Summary — Track/race/surface/distance/class.
Pace Shape — Early/fast/honest/slow; collapse risk; beneficiaries.
Bias Fit — Inside/Mid/Outside; speed vs closers; cite strength tags.
Top Contenders — Rank 1–4 with one-line reasons.
Fair Odds Line — win % (sum ≤100) + fair AM odds.
Overlays/Underlays — relative to typical board.
Ticket Builder — Win / Exacta / Tri / Super / SH5.
Pass/Press Guidance — concise rule of thumb.
Confidence — 1–5 stars.
"""

go = st.button("Analyze this race", type="primary", use_container_width=True)

if go:
    if not pp_text.strip():
        st.warning("Paste BRIS PPs for a single race."); st.stop()
    if len(race_headers)>1:
        st.error("Multiple races detected — paste only one."); st.stop()

    prompt = build_user_prompt(
        pp_text=pp_text.strip(), biases=biases, surface_type=surface_type,
        surface_condition=surface_condition, scratches_list=scratches_list,
        running_styles_text=running_styles_text, ml_context=ml_context_free.strip(),
        profile=profile, angles_df=angles_df, iq_mode=iq_mode,
        weight_preset=weight_preset, factor_choices=factor_choices, auto_angles_df=auto_angles_df
    )

    with st.spinner("Handicapping…"):
        try:
            msgs=[{"role":"system","content":"You are a professional value-based handicapper."},
                  {"role":"user","content":prompt}]
            analysis_text = call_openai_messages(msgs)
        except Exception as e:
            st.error(f"OpenAI error: {e}"); st.stop()

    st.success("Analysis complete.")
    st.caption(f"Confidence band applied automatically: ±{conf_band:.1f} percentage points.")
    st.markdown(analysis_text)
    st.download_button("Download analysis (.txt)", data=analysis_text,
                       file_name="horse_racing_analysis.txt", mime="text/plain",
                       use_container_width=True)

    # Parse simple fair line (Horse — %)
    fair_probs={}
    try:
        for line in analysis_text.splitlines():
            m=re.match(r'^\s*([A-Za-z0-9’\'\.\-\s/&]+?)\s+[–-]\s+(\d+(?:\.\d+)?)\s*%', line)
            if m:
                fair_probs[m.group(1).strip()] = float(m.group(2))/100.0
    except Exception:
        pass

    # Confidence bands table
    if fair_probs:
        band=conf_band/100.0
        rows=[]
        for h,p in fair_probs.items():
            lo=max(0,p-band); hi=min(1,p+band)
            rows.append({"Horse":h,"Fair%":round(p*100,2),"Low%":round(lo*100,2),"High%":round(hi*100,2),
                         "Fair AM": fair_to_american(p)})
        st.markdown("#### Confidence band (informational)")
        st.dataframe(pd.DataFrame(rows), use_container_width=True)

    # Overlays auto (tote if present, else morning line)
    st.markdown("#### Overlay / underlay scan")
    if fair_probs:
        offered = board_dec if board_dec else extract_morning_line(pp_text)
        if board_dec:
            st.caption("Live overlay/underlay (vs. current tote)")
        else:
            st.caption("Predicted overlay/underlay (vs. Morning Line)")
        if offered:
            df_overlay=overlay_table(fair_probs, offered)
            if not df_overlay.empty:
                st.dataframe(df_overlay, use_container_width=True)
                st.download_button("Download overlay table (CSV)",
                                   data=df_overlay.to_csv(index=False),
                                   file_name="overlays.csv", mime="text/csv",
                                   use_container_width=True)
        else:
            st.caption("No live odds or Morning Line parsed.")
    else:
        st.caption("Run analysis to create a fair odds line.")

    # ===== Tickets =====
    st.markdown("## Build tickets (optional)")
    colW,colE = st.columns(2)
    with colW:
        base_win = st.number_input("Base Win stake ($):", min_value=0.5, value=2.0, step=0.5)
        base_ex  = st.number_input("Exacta base ($):",   min_value=0.1, value=1.0, step=0.1)
        base_tri = st.number_input("Trifecta base ($):", min_value=0.1, value=0.5, step=0.1)
    with colE:
        base_super = st.number_input("Superfecta base ($):", min_value=0.1, value=0.1, step=0.1)
        base_sh5   = st.number_input("Super High Five base ($):", min_value=0.1, value=0.1, step=0.1)

    contenders=[h for h in detected_horses if h.lower() not in scratch_set]
    A = st.multiselect("Tier A — Top win candidates", contenders, default=contenders[:1] if contenders else [])
    B = st.multiselect("Tier B — Win threats / strong underneath", [h for h in contenders if h not in A], default=[])
    C = st.multiselect("Tier C — Underneath value (price)", [h for h in contenders if h not in A and h not in B], default=[])
    D = st.multiselect("Tier D — Deep bombs (bottom slots)", [h for h in contenders if h not in A and h not in B and h not in C], default=[])

    ex_cost  = base_ex  * (len(A)*len(B) + len(B)*len(A) + len(A)*max(len(A)-1,0)) if A and (B or len(A)>=2) else 0.0
    tri_cost = base_tri * (len(A)*max(len(B),1)*max(len(C)+len(B),1)) if A else 0.0
    sup_cost = superfecta_cost(len(A) or 0, max(len(B),1) if A else 0, max(len(C),1) if A else 0, max(len(D),1) if A else 0, base_super)
    sh5_cost = super_high5_cost(len(A) or 0, max(len(B),1) if A else 0, max(len(C),1) if A else 0, max(len(D),1) if A else 0, max(len(D),1) if A else 0, base_sh5)

    cost_df=pd.DataFrame([
        ["Win (per horse)", base_win],
        ["Exacta (est.)", round(ex_cost,2)],
        ["Trifecta (est.)", round(tri_cost,2)],
        ["Superfecta (est.)", round(sup_cost,2)],
        ["Super High Five (est.)", round(sh5_cost,2)],
    ], columns=["Bet","Cost ($)"])
    st.dataframe(cost_df, use_container_width=True, hide_index=True)
    st.download_button("Download ticket cost (CSV)", data=cost_df.to_csv(index=False),
                       file_name="ticket_costs.csv", mime="text/csv", use_container_width=True)

# ===== Ledger =====
st.markdown("## Track your bets (optional)")
if "ledger" not in st.session_state: st.session_state.ledger=[]
with st.form("ledger_form"):
    l1,l2,l3=st.columns([2,1,1])
    with l1: desc=st.text_input("Bet description", placeholder="Win: Ashkenazi @ 7/2")
    with l2: stake=st.number_input("Stake ($)", min_value=0.1, value=2.0, step=0.1)
    with l3: ret=st.number_input("Return ($)", min_value=0.0, value=0.0, step=0.1, help="0 if unsettled; update later.")
    add=st.form_submit_button("Add/Update")
    if add and desc: st.session_state.ledger.append({"desc":desc,"stake":float(stake),"return":float(ret)})

if st.session_state.ledger:
    led=pd.DataFrame(st.session_state.ledger)
    led["P/L"]=led["return"]-led["stake"]
    stake_sum=led["stake"].sum(); ret_sum=led["return"].sum(); pl=ret_sum-stake_sum
    roi=(ret_sum/stake_sum - 1.0)*100 if stake_sum>0 else 0.0
    st.dataframe(led, use_container_width=True, hide_index=True)
    c1,c2,c3,c4=st.columns(4)
    c1.metric("Session Stake ($)", f"{stake_sum:.2f}")
    c2.metric("Session Return ($)", f"{ret_sum:.2f}")
    c3.metric("Session P/L ($)", f"{pl:.2f}")
    c4.metric("Session ROI (%)", f"{roi:.2f}")

st.caption("Overlays/EV compute automatically. Beginner Mode keeps things simple; add tote odds anytime.")









