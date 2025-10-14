# app.py
# Horse Race Ready — IQ Mode (single-file)
# - Robust PP parsing (incl. NA firsters)
# - Automatic per-horse angles: Trainer Intent, Pace Pressure 2.0, Trip/Ground-loss, Wet/Sprint/Route fit, Rider Delta
# - Single odds table (Morning Line auto + Live Odds input) → overlays/EV
# - Strategy profiles: Confident / Aggressive / Value Hunter
# - No Step 3 UI; angles auto-feed the math
# - Removed ticket builder & ledger sections

import os, re, json, math
from typing import Dict, List, Tuple
import pandas as pd
import numpy as np
import streamlit as st

# --------------------- Page ---------------------
st.set_page_config(page_title="Horse Race Ready — IQ Mode", page_icon="🏇", layout="wide")
st.title("🏇 Horse Race Ready — IQ Mode")

st.caption("**Note**: Step 3 was removed. Angle parsing now happens automatically under the hood and feeds the equation directly.")

# --------------------- Helpers ---------------------
def detect_valid_race_headers(pp_text: str):
    toks = ("purse", "furlong", "mile", "pars", "post time")
    headers = []
    for m in re.finditer(r"(?mi)^\s*Race\s+(\d+)\b", pp_text or ""):
        start = m.start()
        win = (pp_text[m.end():m.end()+250] or "").lower()
        if any(t in win for t in toks):
            headers.append((start, int(m.group(1))))
    return headers

# Horse header like: "7 Camila Catalina (P 0)" or "(NA 0)"
HORSE_HDR_RE = re.compile(r'(?mi)^\s*(\d+)\s+([A-Za-z0-9\'\.\-\s&]+?)\s+\(\s*(E\/P|E|P|S|NA)\s*[\d]*\s*\)\s*$')

def split_into_horse_blocks(pp_text: str) -> List[Tuple[str,str,str]]:
    out=[]
    matches=list(HORSE_HDR_RE.finditer(pp_text or ""))
    for i,m in enumerate(matches):
        start=m.end()
        end=matches[i+1].start() if i+1<len(matches) else len(pp_text or "")
        out.append((m.group(1).strip(), m.group(2).strip(), (pp_text or "")[start:end]))
    return out

def extract_horses_styles(pp_text: str) -> pd.DataFrame:
    rows=[]
    for m in HORSE_HDR_RE.finditer(pp_text or ""):
        rows.append({"#":m.group(1).strip(),"Horse":m.group(2).strip(),"DetectedStyle":m.group(3).replace("E/P","E/P"),
                     "OverrideStyle":"","StyleStrength":"Solid"})
    # de-dupe
    seen=set(); uniq=[]
    for r in rows:
        k=(r["#"], r["Horse"].lower())
        if k not in seen: seen.add(k); uniq.append(r)
    return pd.DataFrame(uniq)

# ML odds (first odds token on a line inside each horse block)
ML_TOKEN_RE = re.compile(r'(?m)^\s*([0-9]+\/[0-9]+|[0-9]+-[0-9]+|\+\d+|-\d+|\d+(?:\.\d+)?)\s+')

def extract_ml_map(pp_text: str) -> Dict[str,str]:
    out={}
    for post,name,block in split_into_horse_blocks(pp_text):
        m=ML_TOKEN_RE.search(block or "")
        if m: out[name]=m.group(1)
    return out

# Simple sire/damsire mud stats (for wet fit)
SIRE_RE  = re.compile(r'(?mi)^\s*Sire\s*Stats:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%Mud', re.IGNORECASE)
DAMSIRE_RE = re.compile(r'(?mi)^\s*Dam\'sSire:\s*AWD\s*(\d+(?:\.\d+)?)\s+(\d+)%Mud', re.IGNORECASE)

def parse_pedigree(block:str):
    out={"sire_awd":np.nan,"sire_mud":np.nan,"damsire_awd":np.nan,"damsire_mud":np.nan}
    s=SIRE_RE.search(block or "")
    if s: out["sire_awd"]=float(s.group(1)); out["sire_mud"]=float(s.group(2))
    ds=DAMSIRE_RE.search(block or "")
    if ds: out["damsire_awd"]=float(ds.group(1)); out["damsire_mud"]=float(ds.group(2))
    return out

# Trip/trouble dictionary → length credit
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
    return min(3.0, credit)  # cap

# Trainer intent: bullets or top-ranked works signal a “go”
def trainer_intent(block:str)->float:
    # Count “B1/”, “Bg1/”, “Hg1/” or rank 1/xx
    cnt=0
    for m in re.finditer(r'(?mi):\s*\d{2}(?:¨|«|ª|©)?\s+[BHg]?\s*1\/\d+', block or ""):
        cnt+=1
    # minor bump also if many works listed
    works=len(re.findall(r'(?mi)\b\d+Sep|\b\d+Oct|\b\d+Aug|\b\d+Jly', block or ""))  # rough
    return min(1.0, 0.15*cnt + 0.02*works)

# Rider delta: read a jockey “Sprints” line win% if present
def rider_delta(block:str)->float:
    m=re.search(r'(?mi)JKYw\/\s*Sprints\s+\d+\s+(\d+)%', block or "")
    if not m: return 0.0
    win=int(m.group(1))
    # Above ~18% gets a small positive; below 8% a small negative
    if win>=22: return 0.08
    if win>=18: return 0.04
    if win<=8:  return -0.04
    return 0.0

# Pace Pressure 2.0 on field
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
    ppi = nE + 0.5*nEP  # crude “need-the-lead” pressure proxy
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

# EV (with takeout) for $1 in win pool. Approx: keep 15% default if track unknown.
def ev_per_dollar(dec_board: float, fair_p: float, takeout: float=0.15)->float:
    if not dec_board or dec_board<=1 or fair_p<=0: return -1.0
    payout=(dec_board-1)*(1-takeout)
    return payout*fair_p - (1-fair_p)

# --------------------- Sidebar ---------------------
with st.sidebar:
    st.header("Strategy Profile")
    profile = st.radio("Profile:", ["Confident","Aggressive","Value Hunter"], index=2)
    st.caption("Affects edge thresholds and how hard we press overlays.")

# --------------------- Step 1: PPs ---------------------
st.markdown("## Step 1 — Paste one race (PP text)")
pp_text = st.text_area("BRIS PPs:", height=260, placeholder="Paste the full text block of a single race…")

# Surface / distance / race-type quick guesses (fixing earlier parse accuracy)
def guess_surface(pp: str)->str:
    s=(pp or "").lower()
    if re.search(r'\bturf\b', s): return "Turf"
    if re.search(r'\b(tapeta|synthetic|poly)\b', s): return "Synthetic"
    return "Dirt"

def guess_distance_label(pp:str)->str:
    s=(pp or "")
    m = re.search(r'(\d+\s*(?:½|1/2)?)(?:\s*furlongs|\s*furlong|f\s)', s, flags=re.I)
    if m: 
        val=m.group(1).replace("½"," 1/2")
        return f"{val} furlongs".replace("  "," ")
    if re.search(r'\b1\s*mile\b', s, flags=re.I):
        return "1 mile"
    return "6 furlongs"

def detect_race_type(pp:str)->str:
    s=(pp or "").lower()
    if re.search(r'\bmdn|maiden\b', s):
        if re.search(r'claim', s): return "Maiden Claiming"
        if re.search(r'(mdn|maiden)\s*(sp|special)', s): return "Maiden Special Weight"
        return "Maiden (other)"
    if "allow" in s: return "Allowance"
    if "stakes" in s or "stake" in s:
        if re.search(r'grade\s*i|\bg1\b', s): return "Stakes (G1)"
        if re.search(r'grade\s*ii|\bg2\b', s): return "Stakes (G2)"
        if re.search(r'grade\s*iii|\bg3\b', s): return "Stakes (G3)"
        if "listed" in s: return "Stakes (Listed)"
        return "Stakes"
    return "Other"

colA,colB,colC = st.columns([1.1,1,1])
with colA:
    surface_type = st.selectbox("Surface:", ["Dirt","Turf","Synthetic"], index=["Dirt","Turf","Synthetic"].index(guess_surface(pp_text)))
with colB:
    distance_txt = st.text_input("Distance label (for preset bucket):", value=guess_distance_label(pp_text))
with colC:
    st.text_input("Race type (auto-detected):", value=detect_race_type(pp_text), disabled=True)

# --------------------- Step 2: Styles / Bias / Scratches ---------------------
st.markdown("## Step 2 — Confirm scratches & running styles")
df_styles_full = extract_horses_styles(pp_text) if pp_text.strip() else pd.DataFrame()
detected_horses = list(df_styles_full["Horse"]) if not df_styles_full.empty else []
field_size = len(detected_horses)

bias_options = [
    "fair/neutral",
    "inside bias (posts 1-3)",
    "mid bias (posts 4-7)",
    f"outside bias ({'8+' if field_size>=8 else '8+'})",
    # explicit run-style bias options (kept)
    "favors early speed",
    "favors pressers (E/P)",
    "favors mid-pack (P)",
    "favors closers (S)",  # <-- kept by request; if you prefer remove later
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

style_options=["","E","E/P","P","S","NA"]
strength_options=["Strong","Solid","Slight","Weak","Bias"]
edited = st.data_editor(
    df_styles,
    column_config={
        "#": st.column_config.Column(disabled=True, width="small"),
        "Horse": st.column_config.Column(disabled=True),
        "DetectedStyle": st.column_config.Column(disabled=True),
        "OverrideStyle": st.column_config.SelectboxColumn("OverrideStyle", options=style_options),
        "StyleStrength": st.column_config.SelectboxColumn("StyleStrength", options=strength_options),
    },
    hide_index=True, use_container_width=True, num_rows="fixed",
)
kept_horses=list(edited["Horse"]) if isinstance(edited,pd.DataFrame) and not edited.empty else []

def final_style(row)->str:
    a=(row.get("OverrideStyle") or "").strip()
    b=(row.get("DetectedStyle") or "").strip()
    return a if a else (b if b else "NA")

styles = [final_style(r) for _,r in edited.iterrows()] if not edited.empty else []
pace_env = pace_pressure(styles)

# Surface condition (affects Wet Fit)
if surface_type=="Dirt":
    surface_condition = st.selectbox("Condition:", ["fast","muddy","sloppy","wet-fast","good","off"], index=0)
elif surface_type=="Turf":
    surface_condition = st.selectbox("Condition:", ["firm","good","yielding","soft","off"], index=0)
else:
    surface_condition = st.selectbox("Condition:", ["fast","standard","wet"], index=0)

# --------------------- Step 3 removed (now auto) ---------------------

# --------------------- Step 4 — Single odds table ---------------------
st.markdown("## Step 4 — Odds & Overlays (single table)")
st.caption("Morning Line auto-fills from the pasted PPs. Enter Live Odds to update overlays. Formats: 7/2, 5-1, +250, or 3.8.")

ml_map = extract_ml_map(pp_text) if pp_text.strip() else {}
horses_for_board = [h for h in kept_horses]

if "board_df" not in st.session_state:
    st.session_state.board_df = pd.DataFrame({
        "Horse": pd.Series(horses_for_board, dtype="string"),
        "Morning Line": pd.Series([ml_map.get(h,"") for h in horses_for_board], dtype="string"),
        "Live Odds": pd.Series(["" for _ in horses_for_board], dtype="string"),
        "Use (dec)": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
        "Use %": pd.Series([np.nan for _ in horses_for_board], dtype="float"),
    })
else:
    # refresh horse list if changed
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

# Compute Use(dec) = Live if present else ML
use_dec_map={}
for i,r in board_df.iterrows():
    h=str(r["Horse"])
    pick = (r.get("Live Odds") or "").strip() or (r.get("Morning Line") or "").strip()
    dec = str_to_dec(pick) if pick else None
    use_dec_map[h]=dec
    board_df.at[i,"Use (dec)"]=dec if dec else np.nan
    board_df.at[i,"Use %"]=dec_to_prob(dec)*100 if dec else np.nan
st.session_state.board_df=board_df

# --------------------- Model: compute fair line (local) ---------------------
st.markdown("## Step 5 — Analyze (local model)")

# Distance bucket for preset weighting
def dist_bucket(txt:str)->str:
    s=(txt or "").lower()
    if "6 1/2" in s or "6½" in s or "6.5" in s or "7" in s: return "6.5–7f"
    if "furlong" in s:
        m=re.search(r'(\d+(?:\.\d+)?)', s)
        if m and float(m.group(1))<=6: return "≤6f"
        if m and float(m.group(1))>7: return "8f+"
        return "6.5–7f"
    if "mile" in s: return "8f+"
    return "≤6f"

# Base weights (subtle)
PRESETS={
    ("Dirt","≤6f"):   dict(pace=1.10,bias=1.00,proj=1.00,ped=0.95,wet=1.00),
    ("Dirt","6.5–7f"):dict(pace=1.05,bias=1.00,proj=1.00,ped=1.00,wet=1.00),
    ("Dirt","8f+"):   dict(pace=1.00,bias=1.05,proj=1.05,ped=1.05,wet=1.00),
    ("Turf","≤6f"):   dict(pace=0.95,bias=1.00,proj=1.00,ped=1.10,wet=1.00),
    ("Turf","6.5–7f"):dict(pace=1.00,bias=1.00,proj=1.05,ped=1.10,wet=1.00),
    ("Turf","8f+"):   dict(pace=0.98,bias=1.05,proj=1.05,ped=1.10,wet=1.00),
    ("Synthetic","≤6f"):dict(pace=1.00,bias=1.00,proj=1.00,ped=1.05,wet=1.00),
    ("Synthetic","6.5–7f"):dict(pace=1.00,bias=1.00,proj=1.00,ped=1.05,wet=1.00),
    ("Synthetic","8f+"):dict(pace=0.98,bias=1.05,proj=1.05,ped=1.05,wet=1.00),
}

DB = dist_bucket(distance_txt)
BASE=PRESETS.get((surface_type, DB), dict(pace=1.00,bias=1.00,proj=1.00,ped=1.00,wet=1.00))

# Bias nudge from dropdown (small)
bias_mult={"pace":1.0,"bias":1.0}
if "favors early speed" in biases: bias_mult["pace"]*=1.05
if "favors pressers (E/P)" in biases: bias_mult["pace"]*=1.03
if "favors mid-pack (P)" in biases: bias_mult["pace"]*=1.02
if "favors closers (S)" in biases: bias_mult["pace"]*=0.98
if "inside bias (posts 1-3)" in biases: bias_mult["bias"]*=1.03
if f"outside bias (8+)" in biases: bias_mult["bias"]*=1.02

# Per-horse feature extraction + scoring
blocks = {name:block for _,name,block in split_into_horse_blocks(pp_text)}
scores={}
explain={}
for _,row in edited.iterrows():
    h=row["Horse"]
    style=final_style(row)
    block=blocks.get(h,"")
    ped = parse_pedigree(block)
    # Feature knobs
    t_intent = trainer_intent(block)                   # 0 → 1
    t_credit = trip_credit(block) / 3.0                # 0 → 1
    r_delta  = rider_delta(block)                      # -0.04 → +0.08
    # Wet fit
    wet_fit = 0.0
    if surface_type=="Dirt" and surface_condition in ("muddy","sloppy","wet-fast","good"):
        mud = np.nanmean([ped.get("sire_mud"), ped.get("damsire_mud")])
        if mud==mud:
            if mud>=16: wet_fit=+0.06
            elif mud<=8: wet_fit=-0.03
    # Sprint/Route fit via AWD
    fit=0.0
    awd=np.nanmean([ped.get("sire_awd"), ped.get("damsire_awd")])
    if awd==awd:
        if DB=="≤6f" and 6.0<=awd<=7.2: fit+=0.05
        if DB=="8f+" and awd>=7.2:      fit+=0.05
    # Pace environment impact (closer boost if heavy pressure)
    pace_adj=0.0
    if style in ("P","S") and pace_env["ppi"]>=2.5: pace_adj+=0.05
    if style=="E" and pace_env["ppi"]<=1.0: pace_adj+=0.05

    # Combine with presets
    score = 1.0
    score *= BASE["pace"] * bias_mult["pace"] * (1.0 + pace_adj)
    score *= BASE["bias"] * bias_mult["bias"]
    score *= BASE["proj"] * (1.0 + t_intent + t_credit + r_delta)
    score *= BASE["ped"]  * (1.0 + fit)
    score *= BASE["wet"]  * (1.0 + wet_fit)

    scores[h]=max(0.05, score)  # floor
    explain[h]=dict(style=style, t_intent=round(t_intent,3), trip=round(t_credit,3), r_delta=round(r_delta,3),
                    wet=round(wet_fit,3), awd=awd if awd==awd else None, pace_adj=round(pace_adj,3))

# Convert scores → fair probabilities via softmax
if scores:
    vals=np.array(list(scores.values()))
    probs=np.exp(vals/np.mean(vals))  # temperature 1.0 approx
    probs=probs/np.sum(probs)
    fair_probs={h:float(p) for h,p in zip(scores.keys(), probs)}
else:
    fair_probs={}

st.markdown("### Model fair line")
if fair_probs:
    fair_df=pd.DataFrame([
        {"Horse":h, "Fair %": round(p*100,2), "Fair (AM)": fair_to_am(p)} for h,p in sorted(fair_probs.items(), key=lambda x:-x[1])
    ])
    st.dataframe(fair_df, use_container_width=True, hide_index=True)
else:
    st.info("Paste PPs and confirm styles to compute the fair line.")

# Overlays / EV
st.markdown("### Overlays / EV (win pool)")
if fair_probs and board_df is not None and not board_df.empty:
    rows=[]
    for h,p in fair_probs.items():
        dec=use_dec_map.get(h)
        if not dec: continue
        brd=dec_to_prob(dec)
        ev = ev_per_dollar(dec, p, takeout=0.15)
        rows.append({"Horse":h,"Fair %":round(p*100,2),"Board (dec)":round(dec,3),
                     "Board %":round(brd*100,2),"Edge (pp)": round((p-brd)*100,2),
                     "EV per $1": round(ev,3), "Overlay?": "YES" if p>brd else "No"})
    if rows:
        df=pd.DataFrame(rows).sort_values(by=["Overlay?","EV per $1"], ascending=[False,False])
        st.dataframe(df, use_container_width=True, hide_index=True)
    else:
        st.caption("Enter Live Odds or ensure Morning Line parsed for each horse.")
else:
    st.caption("Run the model (just paste PPs and confirm Step 2) to see overlays.")

# Explanations (why the model liked a horse)
if explain:
    st.markdown("### Why the model likes them (key signals)")
    exp_rows=[]
    for h,ex in explain.items():
        exp_rows.append({"Horse":h,"Style":ex["style"],"Trainer intent":ex["t_intent"],"Trip credit":ex["trip"],
                         "Rider Δ":ex["r_delta"],"Wet fit":ex["wet"],"AWD":ex["awd"],"Pace adj":ex["pace_adj"]})
    st.dataframe(pd.DataFrame(exp_rows), use_container_width=True, hide_index=True)

st.caption("Profiles: **Confident** needs bigger edges, **Aggressive** chases overlays harder, **Value Hunter** balances both. Angles feed the math automatically.")





