# app.py
import os
import re
import requests
import pandas as pd
import streamlit as st

# ===================== Page / App Header =====================
st.set_page_config(page_title="Horse Race Ready", page_icon="🏇", layout="centered")
st.title("🏇 Horse Race Ready 🏇")

# ---------- Legal / Links (override via secrets if you want) ----------
SUPPORT_EMAIL = st.secrets.get("SUPPORT_EMAIL", "support@handicappinghorseraces.org")
TERMS_URL   = (st.secrets.get("TERMS_URL", "") or "").strip()
PRIVACY_URL = (st.secrets.get("PRIVACY_URL", "") or "").strip()

TERMS_MD = """
# Terms of Use

This service provides informational handicapping analysis **only** and does not constitute financial, legal, or wagering advice. Use at your own risk.

**Eligibility & Compliance.** You are solely responsible for complying with all laws, rules, and regulations of your jurisdiction.

**No Guarantees.** Outcomes are uncertain. Past performance does not guarantee future results.

**Limitation of Liability.** The app and its providers are not liable for any losses or damages arising from use of the service.

**Governing Law.** These terms are governed by the laws of the United States and, to the extent applicable, the state/province/region from which you access the site, without regard to conflicts of law principles.

**Changes.** We may update these terms at any time by posting an updated version.
"""

PRIVACY_MD = f"""
# Privacy Policy

**What we collect.** If provided, we may collect contact info (e.g., email) and operational logs necessary to run the app.

**How we use it.** To operate, secure, improve, and support the service (e.g., troubleshooting, support responses).

**Sharing.** We do not sell your personal data. We may share with service providers (e.g., hosting, analytics) who process data on our behalf under appropriate safeguards.

**Security.** We use reasonable technical and organizational measures but cannot guarantee absolute security.

**Your Choices.** You may request deletion of your contact info by emailing the address below.

**Governing Law.** This policy is governed by U.S. law and, to the extent applicable, the state/province/region from which you access the site.

**Contact.** {SUPPORT_EMAIL}
"""

def render_disclaimer():
    # Safe defaults for toggles
    st.session_state.setdefault("show_terms", False)
    st.session_state.setdefault("show_privacy", False)

    left, right = st.columns([3, 2])

    with left:
        # Simple disclaimer + mailto (no external TOS/Privacy links here)
        st.markdown(
            f"""
**Disclaimer:** This tool provides informational handicapping analysis only and is **not** financial or wagering advice.  
Questions? <a href="mailto:{SUPPORT_EMAIL}?subject=Horse%20Race%20Ready%20Support">{SUPPORT_EMAIL}</a>.
""",
            unsafe_allow_html=True,
        )

    with right:
        # If external URLs exist, link out. Otherwise, show inline modals.
        if TERMS_URL and PRIVACY_URL:
            st.link_button("Terms", TERMS_URL, use_container_width=True)
            st.link_button("Privacy", PRIVACY_URL, use_container_width=True)
        else:
            t, p = st.columns(2)
            with t:
                if st.button("View Terms", use_container_width=True):
                    st.session_state.show_terms = True
            with p:
                if st.button("View Privacy", use_container_width=True):
                    st.session_state.show_privacy = True

    # Inline viewers (when toggled)
    if st.session_state.get("show_terms"):
        with st.expander("Terms of Use", expanded=True):
            st.markdown(TERMS_MD)
            if st.button("Close Terms"):
                st.session_state.show_terms = False

    if st.session_state.get("show_privacy"):
        with st.expander("Privacy Policy", expanded=True):
            st.markdown(PRIVACY_MD)
            if st.button("Close Privacy"):
                st.session_state.show_privacy = False

# Call it near the top (right after st.title)
render_disclaimer()


# ===================== Model Settings =====================
MODEL = st.secrets.get("OPENAI_MODEL", "gpt-5")
TEMP = float(st.secrets.get("OPENAI_TEMPERATURE", "0.5"))

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

def call_openai_messages(messages):
    """Wrapper that omits temperature when the model doesn't allow it."""
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

# ===================== Supabase Ping =====================
def supabase_ping() -> tuple[bool, str]:
    """
    Requires a table: public.zz_ping (id bigserial pk, created_at timestamptz default now()).
    RLS disabled (or a permissive insert policy).
    """
    url = st.secrets.get("SUPABASE_URL")
    key = st.secrets.get("SUPABASE_ANON_KEY")
    if not url or not key:
        return False, "Missing SUPABASE_URL or SUPABASE_ANON_KEY in secrets."

    try:
        headers = {
            "apikey": key,
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "Prefer": "return=minimal",
        }
        # Bulk-insert form; [{}] inserts a default row
        r = requests.post(f"{url}/rest/v1/zz_ping", headers=headers, json=[{}])
        if r.status_code in (201, 204):
            return True, "Supabase ping OK ✅"
        return False, f"{r.status_code}: {r.text}"
    except Exception as e:
        return False, str(e)

SHOW_PING = str(st.secrets.get("SHOW_PING", "false")).lower() == "true"

if SHOW_PING:
    if st.button("🔌 Test Supabase connection", use_container_width=True):
        ok, msg = supabase_ping()
        (st.success if ok else st.error)(f"Ping result: {msg}")
# ===================== Parsing Helpers =====================
def detect_valid_race_headers(pp_text: str):
    toks = ("purse", "furlong", "mile", "clm", "allow", "stake", "pars", "post time")
    headers = []
    for m in re.finditer(r"(?mi)^\s*Race\s+(\d+)\b", pp_text or ""):
        start = m.start()
        win = (pp_text[m.end():m.end()+250] or "").lower()
        if any(t in win for t in toks):
            headers.append((start, int(m.group(1))))
    return headers

def extract_horses_and_styles(pp_text: str) -> pd.DataFrame:
    pat = re.compile(r'^\s*(\d+)\s+([A-Za-z0-9\'\.\-\s&]+?)\s+\(\s*(E\/P|E|P|S)\s*[\d]*\s*\)', re.MULTILINE)
    rows = []
    for m in pat.finditer(pp_text or ""):
        rows.append({
            "#": m.group(1).strip(),
            "Horse": m.group(2).strip(),
            "DetectedStyle": m.group(3).replace("E/P", "E/P"),
            "OverrideStyle": ""
        })
    seen, uniq = set(), []
    for r in rows:
        k = (r["#"], r["Horse"].lower())
        if k not in seen:
            seen.add(k)
            uniq.append(r)
    return pd.DataFrame(uniq)

def build_user_prompt(pp_text: str, biases, surface_type: str, surface_condition: str,
                      scratches_list, running_styles_text: str, ml_context: str):
    style_glossary = (
        "Running Style Glossary:\n"
        "• E (Early/Front-Runner): wants the lead; tries to wire the field.\n"
        "• E/P (Early/Presser): can make the lead, often sits just off it.\n"
        "• P (Presser/Stalker): mid-pack; moves mid-race.\n"
        "• S (Sustain/Closer): far back early; late kick; needs pace.\n"
    )
    scratches_txt = ", ".join(scratches_list) if scratches_list else "none"
    return f"""
You are a professional value-based horse racing handicapper using pace figures, class ratings, bias patterns, form cycle analysis, trainer & jockey intent, and pedigree tendencies.

Analyze ONLY the single race in the text below.

Track Bias Today: {', '.join(biases) if biases else 'neutral'}
Surface: {surface_type} ({surface_condition})
Scratches (exclude entirely): {scratches_txt}

{style_glossary}
Horse Running Styles (use these; do not reassign unless projection is obvious):
{running_styles_text or 'None detected'}

Likely Morning Line / Tote Context: {ml_context or 'N/A'}

Past Performances (BRIS — single race):
{pp_text}

Return the result in this exact structure (no long paragraphs):

Race Summary – Track, race #, surface, distance, class, purse.
Pace Shape Analysis – Early, honest, or slow pace? Who controls it? Who benefits if it collapses?
Top Contenders – Rank 1st through 3rd with a one-line reason (style • bias fit • form cycle • trainer/jockey stats • pedigree note if relevant).
Fair Odds Line – Assign win % to each contender, convert to fair odds.
Overlays / Underlays – Based on fair odds vs. likely tote.
Betting Strategy –
• Win Bets – horses with minimum acceptable odds.
• Exacta – key/box with value caveats.
• Trifecta – structure if value.
Confidence Rating – 1–5 stars (5 = strong bet, 1 = pass).

Rules:
• Do NOT use or rank any horse listed in Scratches.
• Adjust for the stated track bias and surface/condition.
• Respect the listed Running Styles when projecting pace and trip.
• Keep each section concise and scan-friendly.
"""

# ===================== UI =====================
st.markdown(
    "Paste the **BRIS PPs for a single race only** (from any track). "
    "If you paste a full card with multiple ‘Race #’ headers, the app will stop and ask you to retry."
)

pp_text = st.text_area("BRIS PPs (one race):", height=260, placeholder="Paste the full text block of a single race…")

race_headers = detect_valid_race_headers(pp_text) if pp_text.strip() else []
if pp_text.strip():
    if len(race_headers) == 0:
        st.info("No explicit 'Race #' header detected — that's OK if your paste is a single-race block.")
    elif len(race_headers) > 1:
        st.error(
            f"Detected **{len(race_headers)} races**. "
            "Please paste **only one race** (copy just that race block from your BRIS file)."
        )
        st.stop()

bias_options = [
    "favors speed", "favors stalkers", "favors closers",
    "inside bias", "outside bias", "tiring speed", "fair/neutral"
]
biases = st.multiselect("Track bias today (choose one or more):", options=bias_options, default=["favors speed"])

surface_type = st.selectbox("Surface type:", ["Dirt", "Turf", "Synthetic"], index=0)
if surface_type == "Dirt":
    surface_condition = st.selectbox("Dirt condition:", ["fast", "muddy", "sloppy", "wet-fast", "good", "off"], index=0)
elif surface_type == "Turf":
    surface_condition = st.selectbox("Turf condition:", ["firm", "good", "yielding", "soft", "off"], index=0)
else:
    surface_condition = st.selectbox("Synthetic condition:", ["fast", "standard", "wet"], index=0)

ml_context = st.text_input(
    "Likely Morning Line / Tote Context (optional):",
    placeholder="e.g., heavy chalk; spread race; vulnerable favorite…"
)

st.markdown("### Scratches")
df_styles_full = extract_horses_and_styles(pp_text) if pp_text.strip() else pd.DataFrame()
detected_horses = list(df_styles_full["Horse"]) if not df_styles_full.empty else []
scratched_by_pick = st.multiselect("Select horses to scratch (detected from the paste):", options=detected_horses, default=[])
scratches_manual = st.text_input("Or type numbers/names (comma or new line):", placeholder="e.g., 2, 7, Holiday Fantasy")

scratch_set = set([h.strip().lower() for h in scratched_by_pick])
if scratches_manual.strip():
    for tok in re.split(r"[,\n]+", scratches_manual):
        tok = tok.strip()
        if tok:
            scratch_set.add(tok.lower())
scratches_list = sorted(scratch_set)

st.subheader("Running Styles")
df_styles = df_styles_full.copy()
if not df_styles.empty and scratch_set:
    mask = ~df_styles["Horse"].str.lower().isin(scratch_set) & ~df_styles["#"].astype(str).str.lower().isin(scratch_set)
    df_styles = df_styles[mask].reset_index(drop=True)

style_options = ["", "E", "E/P", "P", "S"]
edited = st.data_editor(
    df_styles,
    column_config={
        "#": st.column_config.Column(disabled=True, width="small"),
        "Horse": st.column_config.Column(disabled=True),
        "DetectedStyle": st.column_config.Column(disabled=True, help="Parsed from BRIS header e.g., (P 2)"),
        "OverrideStyle": st.column_config.SelectboxColumn("OverrideStyle", options=style_options,
                                                          help="Choose to override; leave blank to keep detected."),
    },
    hide_index=True,
    use_container_width=True,
    num_rows="fixed",
)

final_style_lines = []
if isinstance(edited, pd.DataFrame) and not edited.empty:
    for _, r in edited.iterrows():
        style = (r.get("OverrideStyle") or r.get("DetectedStyle") or "P").strip()
        final_style_lines.append(f"{r['Horse']} = {style}")
running_styles_text = "\n".join(final_style_lines) if final_style_lines else "None detected"

# ===================== Analyze =====================
go = st.button("Analyze", type="primary", use_container_width=True)

if go:
    if not pp_text.strip():
        st.warning("Please paste BRIS PPs for a single race.")
        st.stop()
    if len(race_headers) > 1:
        st.error("Multiple races detected — please paste only one race.")
        st.stop()

    user_prompt = build_user_prompt(
        pp_text=pp_text.strip(),
        biases=biases,
        surface_type=surface_type,
        surface_condition=surface_condition,
        scratches_list=scratches_list,
        running_styles_text=running_styles_text,
        ml_context=ml_context.strip(),
    )

    with st.spinner("Handicapping…"):
        try:
            messages = [
                {"role": "system", "content": (
                    "You are a professional value-based handicapper. "
                    "Focus on value, pace, bias, class, form cycles, trainer/jockey intent and pedigree tendencies."
                )},
                {"role": "user", "content": user_prompt},
            ]
            text = call_openai_messages(messages)
        except Exception as e:
            st.error(f"OpenAI error: {e}")
            st.stop()

    st.success("Analysis complete.")
    st.markdown(text)
    st.download_button(
        "Download analysis (.txt)",
        data=text,
        file_name="horse_racing_analysis.txt",
        mime="text/plain",
        use_container_width=True,
    )

st.caption("Tip: This app expects **one race** per paste. You can use it for any track and any single race.")
