
import streamlit as st
import pandas as pd
import numpy as np
import json
import itertools
import io
import math
import re
from fractions import Fraction

st.set_page_config(page_title="Racing Toolkit — ABC vs Caveman + Fair Odds + Prompt Builder", layout="wide")

# ----------------------------
# Helpers
# ----------------------------
def product(ns):
    out = 1
    for n in ns:
        out *= max(1, n)
    return out

def flatten(lst):
    return [x for sub in lst for x in sub]

def clean_horses(seq):
    # accepts list of strings or numbers; returns list of trimmed strings
    out = []
    for x in seq:
        if x is None: continue
        s = str(x).strip()
        if not s: continue
        out.append(s)
    return out

def make_ticket_string(wager, legs, base):
    # legs: list like ["5,6","7,1","2","4,8","3"]
    return f"{wager} " + "/".join(legs) + f" @ {base:.2f}"

def decimal_from_fractional_string(s):
    # Return decimal odds (e.g., 3.50) from strings like '5-2', '7/2', 'EV', 'even', '+150', '-120', '3.5'
    if s is None or (isinstance(s, float) and np.isnan(s)):
        return np.nan
    s = str(s).strip().lower()
    if s == "":
        return np.nan
    if s in {"ev", "even", "evens"}:
        return 2.0
    # American +150 or -120
    if re.fullmatch(r"[+-]?\d+(\.\d+)?", s):
        # could be US or decimal; assume decimal if contains '.' or value >= 2 and <= 20
        if "." in s:
            try: 
                val = float(s)
                return val if val >= 1.01 else np.nan
            except:
                return np.nan
        else:
            val = int(s)
            # if between -2000 and 2000 treat as US
            if val >= 100 or val <= -100:
                if val > 0:
                    return 1 + val/100.0
                else:
                    return 1 + 100.0/abs(val)
            # else assume decimal (e.g., '3')
            return float(val)
    # Fractional like 5-2 or 7/2
    m = re.fullmatch(r"\s*(\d+)\s*[-/]\s*(\d+)\s*", s)
    if m:
        a = float(m.group(1)); b = float(m.group(2))
        if b == 0: return np.nan
        return 1.0 + (a/b)
    return np.nan

def us_from_decimal(d):
    if d is None or (isinstance(d, float) and np.isnan(d)) or d <= 1:
        return np.nan
    if d >= 2.0:
        return round((d - 1.0) * 100)
    else:
        return round(-100.0 / (d - 1.0))

def fraction_string_from_decimal(d, max_den=20):
    # Return a 'A-B' style fractional string approximating decimal (including stake) -> (d-1) to 1
    if d is None or (isinstance(d, float) and np.isnan(d)) or d <= 1:
        return ""
    r = d - 1.0
    frac = Fraction(r).limit_denominator(max_den)
    return f"{frac.numerator}-{frac.denominator}"

def decimal_from_prob(p):
    if p <= 0: 
        return np.nan
    return 1.0 / p

def implied_prob_from_decimal(d):
    if d is None or (isinstance(d, float) and np.isnan(d)) or d <= 1:
        return np.nan
    return 1.0 / d

def kelly_fraction(p, price_decimal):
    # Kelly for fixed-odds win market: f* = (bp - q) / b, where b = d-1
    if price_decimal is None or (isinstance(price_decimal, float) and np.isnan(price_decimal)) or price_decimal <= 1:
        return np.nan
    b = price_decimal - 1.0
    q = 1.0 - p
    f = (b * p - q) / b
    return max(0.0, f)

def calc_combos(legs):
    return product([len(x) for x in legs])

def uniq(seq):
    seen = set()
    out = []
    for x in seq:
        key = json.dumps(x, sort_keys=True)
        if key not in seen:
            seen.add(key)
            out.append(x)
    return out

def build_all_A(legs_A):
    if any(len(a)==0 for a in legs_A):
        return []
    return [legs_A]

def build_exactly_one_B(legs_A, legs_B):
    tickets = []
    n = len(legs_A)
    for i in range(n):
        if len(legs_B[i]) == 0: 
            continue
        legs = []
        ok = True
        for j in range(n):
            if j == i:
                legs.append(legs_B[j])
            else:
                if len(legs_A[j]) == 0:
                    ok = False; break
                legs.append(legs_A[j])
        if ok:
            tickets.append(legs)
    return tickets

def build_exactly_two_B(legs_A, legs_B):
    tickets = []
    n = len(legs_A)
    idxs = [i for i in range(n) if len(legs_B[i])>0]
    for i in range(len(idxs)):
        for j in range(i+1, len(idxs)):
            ii, jj = idxs[i], idxs[j]
            legs = []
            ok = True
            for k in range(n):
                if k==ii or k==jj:
                    legs.append(legs_B[k])
                else:
                    if len(legs_A[k]) == 0:
                        ok = False; break
                    legs.append(legs_A[k])
            if ok:
                tickets.append(legs)
    return tickets

def build_exactly_one_C_no_B(legs_A, legs_B, legs_C):
    tickets = []
    n = len(legs_A)
    for i in range(n):
        if len(legs_C[i]) == 0: 
            continue
        # rest must be A; and require no Bs anywhere in this ticket
        if any(len(legs_A[j])==0 for j in range(n) if j!=i):
            continue
        legs = []
        for j in range(n):
            if j==i:
                legs.append(legs_C[j])
            else:
                legs.append(legs_A[j])
        tickets.append(legs)
    return tickets

def leg_str(leg):
    return ",".join(leg)

def legs_to_strings(ticket):
    return [leg_str(leg) for leg in ticket]

def union_space_size(legs_A, legs_B, legs_C):
    return product([len(legs_A[i]) + len(legs_B[i]) + len(legs_C[i]) for i in range(len(legs_A))])

def caveman_ticket(legs_union):
    return [legs_union]

def to_df_tickets(built, bases, label_prefix, wager="P5", start_race=None):
    rows = []
    for label, tickets, base in built:
        for t in tickets:
            leg_strs = legs_to_strings(t)
            combos = calc_combos(t)
            cost = combos * base
            rows.append({
                "TicketLabel": f"{label_prefix}: {label}",
                "WagerType": wager,
                "Base": base,
                "Leg1": leg_strs[0] if len(leg_strs)>0 else "",
                "Leg2": leg_strs[1] if len(leg_strs)>1 else "",
                "Leg3": leg_strs[2] if len(leg_strs)>2 else "",
                "Leg4": leg_strs[3] if len(leg_strs)>3 else "",
                "Leg5": leg_strs[4] if len(leg_strs)>4 else "",
                "NumCombos": combos,
                "Cost": cost,
                "TicketString": make_ticket_string(wager, leg_strs, base),
            })
    return pd.DataFrame(rows)

# ----------------------------
# Sidebar
# ----------------------------
st.sidebar.title("Racing Toolkit")
st.sidebar.caption("Caveman vs ABC • Fair Odds • Prompt Builder • CSV Export")

with st.sidebar.expander("Sample JSON (copy/paste)", expanded=False):
    st.code('[{"race":"Race 3","A":["5","6"],"B":["4"],"C":["1"]},'
            ' {"race":"Race 4","A":["7","1"],"B":["3","4"],"C":[]},'
            ' {"race":"Race 5","A":["2"],"B":["1","6"],"C":["5"]},'
            ' {"race":"Race 6","A":["4","8"],"B":[],"C":["2"]},'
            ' {"race":"Race 7","A":["3"],"B":["7"],"C":["1","5"]}]', language="json")

# ----------------------------
# Tabs
# ----------------------------
tab1, tab2, tab3, tab4 = 
# === Paywall: Stripe Checkout + JWT cookie ===
import os, time
import jwt
import stripe
from urllib.parse import urlencode, urlparse, urlunparse, parse_qs
from streamlit_cookies_manager import EncryptedCookieManager

APP_SECRET = st.secrets.get("APP_SECRET", None)
STRIPE_SECRET_KEY = st.secrets.get("STRIPE_SECRET_KEY", None)
PRICE_ID_MONTHLY = st.secrets.get("STRIPE_PRICE_ID_MONTHLY", None)  # e.g., price_123
PRICE_ID_DAYPASS = st.secrets.get("STRIPE_PRICE_ID_DAYPASS", None)  # e.g., price_456
PUBLIC_URL = st.secrets.get("PUBLIC_URL", None)  # e.g., https://your-app.streamlit.app

# Prepare cookie manager (if APP_SECRET exists)
cookies = None
if APP_SECRET:
    cookies = EncryptedCookieManager(prefix="paywall_", password=APP_SECRET)
    if not cookies.ready():
        st.stop()

def _base_url():
    # Build base URL without query params
    try:
        ctx = st.context
    except Exception:
        ctx = None
    # Use PUBLIC_URL if provided
    if PUBLIC_URL:
        return PUBLIC_URL.rstrip("/")
    # Fallback to current request if available via query params/hack
    # Best-effort: remove query and fragment
    try:
        # st.experimental_get_query_params ensures we have a URL context
        params = st.experimental_get_query_params()
        # Hackish, but we'll build relative success url
        return ""
    except Exception:
        return ""

def _success_url():
    # We rely on relative path; Streamlit will keep route
    q = {"sid": "{CHECKOUT_SESSION_ID}"}
    qp = urlencode(q)
    return f"?{qp}"

def _cancel_url():
    return "?"

def _create_checkout(mode="subscription"):
    if not STRIPE_SECRET_KEY:
        st.error("Stripe not configured. Add STRIPE keys in st.secrets then reload.")
        st.stop()
    stripe.api_key = STRIPE_SECRET_KEY
    kwargs = dict(
        ui_mode="hosted",
        success_url=_success_url(),
        cancel_url=_cancel_url(),
        automatic_tax={"enabled": False},
        allow_promotion_codes=True,
    )
    if mode == "subscription":
        kwargs.update(mode="subscription", line_items=[{"price": PRICE_ID_MONTHLY, "quantity": 1}])
    else:
        kwargs.update(mode="payment", line_items=[{"price": PRICE_ID_DAYPASS, "quantity": 1}])
    session = stripe.checkout.Session.create(**kwargs)
    return session.url

def _issue_token(email, days):
    payload = {"email": email, "exp": int(time.time()) + days*24*3600}
    return jwt.encode(payload, APP_SECRET, algorithm="HS256")

def _validate_token(tok):
    try:
        data = jwt.decode(tok, APP_SECRET, algorithms=["HS256"])
        return True, data.get("email", "")
    except Exception:
        return False, ""

def paywall_gate():
    # If secrets missing, show setup UI but do not block owner (localhost)
    if APP_SECRET is None:
        st.warning("Paywall disabled: set APP_SECRET in st.secrets to enable authentication.")
        return True

    # 1) Existing cookie?
    if cookies and cookies.get("auth_token"):
        ok, email = _validate_token(cookies.get("auth_token"))
        if ok:
            st.session_state["user_email"] = email
            return True

    # 2) Returning from Stripe?
    qp = st.experimental_get_query_params()
    sid_list = qp.get("sid", [])
    if sid_list and STRIPE_SECRET_KEY:
        sid = sid_list[0]
        try:
            stripe.api_key = STRIPE_SECRET_KEY
            sess = stripe.checkout.Session.retrieve(sid, expand=["subscription", "invoice", "customer"])
            mode = sess.get("mode")
            email = (sess.get("customer_details") or {}).get("email") or "paid@user"
            # Decide entitlement window
            if mode == "subscription":
                sub = sess.get("subscription") or {}
                status = (sub.get("status") if isinstance(sub, dict) else None) or sess.get("status")
                if status in ("active", "trialing", "complete"):
                    days = 30
                else:
                    days = 0
            else:
                if sess.get("payment_status") == "paid":
                    days = 1
                else:
                    days = 0
            if days > 0:
                token = _issue_token(email, days)
                cookies["auth_token"] = token
                cookies.save()
                # Clean query params
                st.experimental_set_query_params()
                st.success("Payment verified. Welcome!")
                st.session_state["user_email"] = email
                return True
            else:
                st.error("Payment not verified yet. If you just paid, wait a moment and refresh.")
        except Exception as e:
            st.error(f"Stripe verification error: {e}")

    # 3) Show paywall UI
    st.markdown("### 🔒 Access required")
    st.write("Purchase a pass to unlock the app.")
    col1, col2 = st.columns(2)
    with col1:
        if st.button("💳 Subscribe (Monthly)"):
            if not PRICE_ID_MONTHLY:
                st.error("Missing STRIPE_PRICE_ID_MONTHLY in secrets.")
            else:
                url = _create_checkout("subscription")
                st.link_button("Continue to Checkout", url, use_container_width=True)
    with col2:
        if st.button("🕒 Day Pass (24h)"):
            if not PRICE_ID_DAYPASS:
                st.error("Missing STRIPE_PRICE_ID_DAYPASS in secrets.")
            else:
                url = _create_checkout("payment")
                st.link_button("Continue to Checkout", url, use_container_width=True)
    st.info("After payment you'll return here automatically. We'll verify your session and grant access.")
    st.stop()

# Gate entry
paywall_gate()
# Optional: logout button
def _logout():
    if cookies:
        cookies["auth_token"] = ""
        cookies.save()
        st.success("Logged out.")
        st.experimental_rerun()

with st.sidebar:
    if APP_SECRET:
        if st.button("Log out"):
            _logout()
st.tabs(["🎫 Tickets: Caveman vs ABC", "📈 Fair Odds & Value", "🧠 Prompt Builder", "📤 Export / ADW CSV"])

# ----------------------------
# TAB 1 — TICKETS
# ----------------------------
with tab1:
    st.header("Caveman vs. ABC Ticket Builder")
    st.write("Paste A/B/C legs JSON, choose which ABC tiers to include, and compare to a Caveman build.")

    legs_json = st.text_area("Paste A/B/C legs JSON (list of races with A/B/C arrays)", height=140)
    colA, colB = st.columns([1,1])
    with colA:
        include_allA = st.checkbox("Include All-A ticket", True)
        include_oneB = st.checkbox("Include Exactly one-B tickets", True)
        include_twoB = st.checkbox("Include Exactly two-B tickets (optional)", False)
        include_oneC = st.checkbox("Include Exactly one-C (no B) tickets (optional)", False)
    with colB:
        base_allA = st.number_input("Base for All-A ($)", min_value=0.10, step=0.10, value=1.00, format="%.2f")
        base_oneB = st.number_input("Base for one-B tickets ($)", min_value=0.10, step=0.10, value=0.50, format="%.2f")
        base_twoB = st.number_input("Base for two-B tickets ($)", min_value=0.10, step=0.10, value=0.25, format="%.2f")
        base_oneC = st.number_input("Base for one-C tickets ($)", min_value=0.10, step=0.10, value=0.25, format="%.2f")

    st.subheader("Caveman Options")
    caveman_pool = st.selectbox("Caveman includes:", ["A only", "A + B", "A + B + C"])
    caveman_base = st.number_input("Caveman base ($)", min_value=0.10, step=0.10, value=0.50, format="%.2f")

    if st.button("Build Tickets", type="primary"):
        try:
            data = json.loads(legs_json)
            races = [d.get("race", f"Leg {i+1}") for i, d in enumerate(data)]
            legs_A = [clean_horses(d.get("A", [])) for d in data]
            legs_B = [clean_horses(d.get("B", [])) for d in data]
            legs_C = [clean_horses(d.get("C", [])) for d in data]

            # ABC tickets
            abc_built = []
            if include_allA:
                abc_built.append(("All-A", build_all_A(legs_A), base_allA))
            if include_oneB:
                abc_built.append(("Exactly 1-B", build_exactly_one_B(legs_A, legs_B), base_oneB))
            if include_twoB:
                abc_built.append(("Exactly 2-B", build_exactly_two_B(legs_A, legs_B), base_twoB))
            if include_oneC:
                abc_built.append(("Exactly 1-C (no B)", build_exactly_one_C_no_B(legs_A, legs_B, legs_C), base_oneC))

            abc_df = to_df_tickets(abc_built, 
                                   [base_allA, base_oneB, base_twoB, base_oneC],
                                   "ABC Bundle", wager="P5")

            if len(abc_df) == 0:
                st.warning("No valid ABC tickets were generated — check that each required A/B/C leg has selections.")
            else:
                # Coverage metrics
                # space: all combos using union(A,B,C) per race
                space = union_space_size(legs_A, legs_B, legs_C)
                total_combos = int(abc_df["NumCombos"].sum())
                total_cost = float(abc_df["Cost"].sum())
                coverage = 0.0 if space == 0 else total_combos / space
                eff = np.nan if total_combos == 0 else (total_cost / total_combos) * 1000.0  # $ per 1000 combos covered

                st.subheader("ABC Tickets")
                st.dataframe(abc_df, use_container_width=True, hide_index=True)
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("ABC Total Cost", f"${total_cost:,.2f}")
                c2.metric("ABC Total Combos", f"{total_combos:,}")
                c3.metric("Grid Size (A∪B∪C)", f"{space:,}")
                c4.metric("$ / 1,000 covered combos", f"${eff:,.2f}")

            # Caveman
            if caveman_pool == "A only":
                legs_union = legs_A
            elif caveman_pool == "A + B":
                legs_union = [sorted(set(legs_A[i] + legs_B[i])) for i in range(len(legs_A))]
            else:
                legs_union = [sorted(set(legs_A[i] + legs_B[i] + legs_C[i])) for i in range(len(legs_A))]

            caveman_tix = caveman_ticket(legs_union)
            caveman_df = to_df_tickets([("Caveman", caveman_tix, caveman_base)], [caveman_base], "Caveman", "P5")
            cav_combos = int(caveman_df["NumCombos"].sum())
            cav_cost = float(caveman_df["Cost"].sum())

            st.subheader("Caveman Ticket")
            st.dataframe(caveman_df, use_container_width=True, hide_index=True)
            d1, d2 = st.columns(2)
            d1.metric("Caveman Cost", f"${cav_cost:,.2f}")
            d2.metric("Caveman Combos", f"{cav_combos:,}")

            # Keep in session for Export tab
            st.session_state["abc_df"] = abc_df if len(abc_df) else pd.DataFrame()
            st.session_state["caveman_df"] = caveman_df

            st.success("Tickets built — see Export tab for ADW CSV.")
        except Exception as e:
            st.error(f"Could not parse JSON / build tickets: {e}")

# ----------------------------
# TAB 2 — FAIR ODDS & VALUE
# ----------------------------
with tab2:
    st.header("Fair Odds & Value — Win Market")
    st.write("Enter runners & your win probabilities. Optionally add Morning Line/Tote odds to flag overlays/underlays and compute Kelly sizing.")

    sample_names = "5 Street Humor\n4 Catholic Guilt\n5 Words of Wisdom\n1 Holiday Fantasy\n2 Deflater\n3 McGeorge"
    names_text = st.text_area("Runners (one per line; you can include numbers and names)", value=sample_names, height=120)
    names = [n.strip() for n in names_text.splitlines() if n.strip()]

    default_p = round(1.0 / max(len(names), 1) * 100.0, 1)
    win_pcts = st.text_input("Win % for each (comma-separated, same order) — leave blank for equal split",
                             value="")
    if win_pcts.strip():
        try:
            vals = [float(x.strip()) for x in win_pcts.split(",")]
            # normalize if they don't sum 100
            s = sum(vals)
            if s <= 0:
                probs = [1.0/len(names)] * len(names)
            else:
                probs = [v/s for v in vals]
        except:
            st.warning("Could not parse win %s — falling back to equal probabilities.")
            probs = [1.0/len(names)] * len(names)
    else:
        probs = [1.0/len(names)] * len(names)

    col_ml, col_tote, col_bank = st.columns([2,2,1])
    with col_ml:
        ml_text = st.text_input("Morning Line (fractional like 5-2 or 3/1; comma-separated; optional)", value="")
    with col_tote:
        tote_text = st.text_input("Live Tote / Estimate (accepts +150,-120, 5-2, or decimal; comma-separated; optional)", value="")
    with col_bank:
        bankroll = st.number_input("Bankroll ($)", min_value=0.0, value=1000.0, step=50.0)

    def parse_odds_list(odds_text):
        if not odds_text.strip():
            return [np.nan]*len(names)
        parts = [x.strip() for x in odds_text.split(",")]
        out = []
        for i in range(len(names)):
            val = parts[i] if i < len(parts) else ""
            out.append(decimal_from_fractional_string(val))
        return out

    ml_dec = parse_odds_list(ml_text)
    tote_dec = parse_odds_list(tote_text)

    rows = []
    for i, name in enumerate(names):
        p = probs[i] if i < len(probs) else 0.0
        fair_dec = decimal_from_prob(p)
        fair_us = us_from_decimal(fair_dec)
        fair_frac = fraction_string_from_decimal(fair_dec)
        ml_d = ml_dec[i] if i < len(ml_dec) else np.nan
        tote_d = tote_dec[i] if i < len(tote_dec) else np.nan
        ml_imp = implied_prob_from_decimal(ml_d)
        tote_imp = implied_prob_from_decimal(tote_d)

        edge_ml = np.nan if np.isnan(ml_imp) else (p - ml_imp)
        edge_tote = np.nan if np.isnan(tote_imp) else (p - tote_imp)

        kelly_full = np.nan if np.isnan(tote_d) else kelly_fraction(p, tote_d)
        bet_full = np.nan if np.isnan(kelly_full) else bankroll * kelly_full
        bet_half = np.nan if np.isnan(kelly_full) else bankroll * (kelly_full/2.0)

        rows.append({
            "Horse": name,
            "Win%": round(p*100.0, 2),
            "Fair Decimal": None if np.isnan(fair_dec) else round(fair_dec, 3),
            "Fair US": None if np.isnan(fair_us) else int(fair_us),
            "Fair Fractional": fair_frac,
            "ML (dec)": None if np.isnan(ml_d) else round(ml_d, 3),
            "ML Impl%": None if np.isnan(ml_imp) else round(ml_imp*100.0, 2),
            "Edge vs ML (pp)": None if np.isnan(edge_ml) else round(edge_ml*100.0, 2),
            "Tote (dec)": None if np.isnan(tote_d) else round(tote_d, 3),
            "Tote Impl%": None if np.isnan(tote_imp) else round(tote_imp*100.0, 2),
            "Edge vs Tote (pp)": None if np.isnan(edge_tote) else round(edge_tote*100.0, 2),
            "Kelly f*": None if np.isnan(kelly_full) else round(kelly_full, 4),
            "Bet (Kelly)": None if np.isnan(bet_full) else round(bet_full, 2),
            "Bet (Half-Kelly)": None if np.isnan(bet_half) else round(bet_half, 2),
            "Min Acceptable (US)": None if np.isnan(fair_us) else int(fair_us),
        })
    odds_df = pd.DataFrame(rows)
    st.dataframe(odds_df, use_container_width=True, hide_index=True)

    overlays = odds_df[odds_df["Edge vs Tote (pp)"].notna() & (odds_df["Edge vs Tote (pp)"]>0)]
    underlays = odds_df[odds_df["Edge vs Tote (pp)"].notna() & (odds_df["Edge vs Tote (pp)"]<=0)]
    c1, c2 = st.columns(2)
    c1.subheader("Overlays (vs Tote)")
    c1.dataframe(overlays[["Horse","Win%","Tote (dec)","Fair Decimal","Edge vs Tote (pp)","Bet (Half-Kelly)"]], 
                 use_container_width=True, hide_index=True)
    c2.subheader("Underlays (vs Tote)")
    c2.dataframe(underlays[["Horse","Win%","Tote (dec)","Fair Decimal","Edge vs Tote (pp)"]], 
                 use_container_width=True, hide_index=True)

    st.download_button(
        "Download value table (CSV)",
        data=odds_df.to_csv(index=False).encode(),
        file_name="fair_odds_value.csv",
        mime="text/csv"
    )

# ----------------------------
# TAB 3 — PROMPT BUILDER
# ----------------------------
with tab3:
    st.header("Prompt Builder — Bias • Pace • Fair Odds • Bets")
    st.write("Compose a structured handicapping prompt you can paste into your AI assistant.")

    bias = st.text_input("Track Bias Today", value="favors speed")
    ml_context = st.text_input("Likely Morning Line / Tote Context (optional)", value="")
    intro = "You are a professional value-based horse racing handicapper using pace figures, class ratings, bias patterns, form cycle analysis, and trainer intent."
    rules = (
        "Rules for Output:\n\n"
        "- Keep each race’s output short, structured, and easy to scan — no paragraphs.\n"
        "- Highlight value over raw probability — passing a race is acceptable.\n"
        "- Do not assume favorites are playable without value confirmation.\n"
        "- Adjust pace and contender ranking for the stated track bias.\n"
        "- Output results race-by-race for full cards, clearly labeled."
    )
    past_perfs = st.text_area("Paste BRIS Past Performances text here", height=200, placeholder="[paste full BRIS PPs here — for a full card, paste all races in order]")
    add_fair = st.checkbox("Append Fair-Odds table request per race", True)
    add_bets = st.checkbox("Append Win/Exacta/Trifecta strategy request", True)

    if st.button("Build Prompt", type="primary"):
        sections = [
            intro,
            f"Track Bias Today: {bias}",
        ]
        if ml_context.strip():
            sections.append(f"Likely Morning Line / Tote Context: {ml_context}")
        sections.append("Past Performances:")
        sections.append(past_perfs.strip() if past_perfs.strip() else "[PASTE PPs ABOVE]")
        sections.append("\nFor EACH race, return:\n\n"
                        "Race Summary – Track, race #, surface, distance, class, purse.\n\n"
                        "Pace Shape Analysis – Early, honest, or slow pace? Who controls it? Who benefits if it collapses?\n\n"
                        "Top Contenders – Rank 1st through 3rd with a one-line reason for each (include style, bias fit, form cycle, trainer stats).\n\n"
                        "Fair Odds Line – Assign win % to each contender, convert to fair odds.\n\n"
                        "Overlays / Underlays – Based on fair odds vs. likely tote.\n\n"
                        "Betting Strategy –\n"
                        "Win Bets – List horses with minimum acceptable odds.\n"
                        "Exacta Strategy – Key/box recommendations with value caveats.\n"
                        "Trifecta Strategy – If value is present, suggest structure.\n\n"
                        "Confidence Rating – 1–5 stars (5 = strong bet, 1 = pass race).")
        sections.append(rules)
        if add_fair:
            sections.append("\nAdditionally, provide a compact table per race with columns: Horse | Style | Recent Pace figs | Class rating | Form cycle note | Trainer/Jockey note | Fair % | Fair US odds.")
        if add_bets:
            sections.append("\nFinally, conclude with a 'Bet Slip' summary: Win targets with minimum prices, Exacta keys/spreads, and Trifecta scaffolds if (and only if) value is confirmed.")
        final_prompt = "\n\n".join(sections)
        st.text_area("Your composed prompt", value=final_prompt, height=450)

# ----------------------------
# TAB 4 — EXPORT
# ----------------------------
with tab4:
    st.header("Export — ADW CSV & shareable text")

    abc_df = st.session_state.get("abc_df", pd.DataFrame())
    caveman_df = st.session_state.get("caveman_df", pd.DataFrame())

    if caveman_df is None or len(caveman_df)==0:
        st.info("Build tickets in the Tickets tab first.")
    else:
        st.subheader("Current Tickets")
        if len(abc_df):
            show_df = pd.concat([abc_df, caveman_df], ignore_index=True)
        else:
            show_df = caveman_df.copy()
        st.dataframe(show_df, use_container_width=True, hide_index=True)

        # ADW CSV format
        st.markdown("**ADW CSV (generic)** — columns: WagerType, Base, TicketLabel, Leg1..Leg5, NumCombos, Cost, TicketString")
        csv_bytes = show_df.to_csv(index=False).encode()
        st.download_button("Download ADW CSV", data=csv_bytes, file_name="tickets_adw.csv", mime="text/csv")

        # Text summary
        txt_lines = []
        for _, r in show_df.iterrows():
            legs = [r.get(f"Leg{i}", "") for i in range(1,6)]
            legs_clean = [x for x in legs if isinstance(x, str) and x!=""]
            txt_lines.append(f"- {r['TicketLabel']}: {r['TicketString']} — ${r['Cost']:.2f}")
        txt = "\n".join(txt_lines)
        st.download_button("Download ticket summary (.txt)", data=txt.encode(), file_name="tickets.txt", mime="text/plain")

st.caption("Pro tip: Save this app to your private repo or Streamlit Cloud and gate access with a secret or payment wall.")
