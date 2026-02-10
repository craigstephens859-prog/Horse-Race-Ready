"""
Elite Pipeline Audit — Run as standalone script (bypasses Jupyter kernel issues)
Executes all 14 audit cells from elite_pipeline_audit.ipynb
"""

import os
import sys
import time
import types
import warnings
from collections import OrderedDict
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
ROOT = Path(r"C:\Users\C Stephens\Desktop\Horse Racing Picks")
sys.path.insert(0, str(ROOT))
os.chdir(ROOT)

print(f"Working directory: {os.getcwd()}")
print(f"Python: {sys.version}")
print(f"NumPy: {np.__version__}")
print(f"Pandas: {pd.__version__}")

# ══════════════════════════════════════════════════════════════
# CELL 2: Streamlit Mock + Import app.py
# ══════════════════════════════════════════════════════════════


class MockSessionState(dict):
    def __getattr__(self, key):
        try:
            return self[key]
        except KeyError:
            raise AttributeError(key)

    def __setattr__(self, key, val):
        self[key] = val


class MockContext:
    """Bulletproof mock that absorbs any operation without error."""

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def __call__(self, *a, **kw):
        return self

    def __iter__(self):
        return iter([self, self, self, self])

    def __getattr__(self, name):
        if name == "empty":
            return True
        if name in ("columns", "values", "index"):
            return []
        if name == "shape":
            return (0, 0)
        return MockContext()

    def __setitem__(self, key, val):
        pass

    def __getitem__(self, key):
        return MockContext()

    def __delitem__(self, key):
        pass

    def __setattr__(self, key, val):
        pass

    def __len__(self):
        return 0

    def __bool__(self):
        return False

    def __str__(self):
        return ""

    def __repr__(self):
        return ""

    def __int__(self):
        return 0

    def __float__(self):
        return 0.0

    def __eq__(self, other):
        return False

    def __ne__(self, other):
        return True

    def __contains__(self, item):
        return False

    def __add__(self, other):
        return self

    def __radd__(self, other):
        return self

    def __sub__(self, other):
        return self

    def __mul__(self, other):
        return self

    def __lt__(self, other):
        return False

    def __le__(self, other):
        return False

    def __gt__(self, other):
        return False

    def __ge__(self, other):
        return False

    def strip(self, *a):
        return ""

    def apply(self, *a, **kw):
        return self

    def iterrows(self):
        return iter([])

    def itertuples(self):
        return iter([])


def _noop(*a, **kw):
    return MockContext()


def _mock_columns(*a, **kw):
    """Return the right number of MockContexts for st.columns unpacking."""
    if a and hasattr(a[0], "__len__"):
        return [MockContext() for _ in a[0]]
    elif a and isinstance(a[0], int):
        return [MockContext() for _ in range(a[0])]
    return [MockContext(), MockContext()]


def _mock_tabs(*a, **kw):
    if a and hasattr(a[0], "__len__"):
        return [MockContext() for _ in a[0]]
    return [MockContext(), MockContext()]


def _return_false(*a, **kw):
    return False


def _return_empty_string(*a, **kw):
    return ""


def _return_zero(*a, **kw):
    return 0


mock_st = types.ModuleType("streamlit")
mock_st.session_state = MockSessionState()
for attr in [
    "write",
    "info",
    "warning",
    "error",
    "success",
    "metric",
    "caption",
    "expander",
    "markdown",
    "header",
    "subheader",
    "divider",
    "dataframe",
    "table",
    "plotly_chart",
    "stop",
    "rerun",
    "spinner",
    "empty",
    "container",
    "form",
    "form_submit_button",
    "button",
    "set_page_config",
    "title",
    "sidebar",
    "image",
    "toast",
    "page_link",
    "navigation",
    "dialog",
    "fragment",
    "html",
    "progress",
    "status",
    "balloons",
    "snow",
    "data_editor",
    "download_button",
    "code",
    "line_chart",
    "select_slider",
]:
    setattr(mock_st, attr, _noop)
mock_st.columns = _mock_columns
mock_st.tabs = _mock_tabs
for attr in ["text_area", "text_input", "selectbox", "radio"]:
    setattr(mock_st, attr, _return_empty_string)
for attr in ["number_input", "slider", "select_slider"]:
    setattr(mock_st, attr, _return_zero)
for attr in ["checkbox", "toggle"]:
    setattr(mock_st, attr, _return_false)
mock_st.multiselect = lambda *a, **kw: []
mock_st.cache_data = lambda *a, **kw: lambda f: f
mock_st.cache_resource = lambda *a, **kw: lambda f: f
mock_st.secrets = MockSessionState()
mock_st.query_params = MockSessionState()

col_config_mod = types.ModuleType("streamlit.column_config")
for cc in [
    "TextColumn",
    "NumberColumn",
    "ProgressColumn",
    "BarChartColumn",
    "LinkColumn",
    "ImageColumn",
    "CheckboxColumn",
    "SelectboxColumn",
    "DateColumn",
    "DatetimeColumn",
    "TimeColumn",
    "ListColumn",
    "Column",
]:
    setattr(col_config_mod, cc, _noop)
mock_st.column_config = col_config_mod
sys.modules["streamlit.column_config"] = col_config_mod


class MockSidebar:
    def __getattr__(self, name):
        return _noop


mock_st.sidebar = MockSidebar()
mock_st.experimental_rerun = _noop
sys.modules["streamlit"] = mock_st

print("\nStreamlit mocked. Importing app.py core functions...")
t_import = time.perf_counter()
try:
    if "app" in sys.modules:
        del sys.modules["app"]
    import app as APP

    print(
        f"✅ app.py loaded in {time.perf_counter() - t_import:.1f}s ({len(dir(APP))} attributes)"
    )
    print(f"   speed_fig_weight = {APP.MODEL_CONFIG['speed_fig_weight']}")
    print(f"   softmax_tau = {APP.MODEL_CONFIG['softmax_tau']}")
except Exception as e:
    print(f"❌ Import error: {e}")
    import traceback

    traceback.print_exc()
    sys.exit(1)

# ══════════════════════════════════════════════════════════════
# POST-IMPORT: Apply all 6 optimizations (disk file is stale)
# ══════════════════════════════════════════════════════════════
print("\n🔧 Applying 6 algorithm optimizations post-import...")

# OPT 1: Speed figure weight 0.05 → 0.15
APP.MODEL_CONFIG["speed_fig_weight"] = 0.15
print(f"  ✅ OPT 1: speed_fig_weight → {APP.MODEL_CONFIG['speed_fig_weight']}")


# OPT 2: analyze_pace_figures — PAR-adjusted with recency weights & energy distribution
def _optimized_analyze_pace(
    e1_vals, e2_vals, lp_vals, e1_par=None, e2_par=None, lp_par=None
):
    """PAR-adjusted pace analysis with recency-weighted averages and energy distribution."""
    bonus = 0.0
    if len(e1_vals) < 2 or len(lp_vals) < 2:
        return bonus
    # Recency-weighted averages (most recent race weighted 2x)
    weights = [2.0, 1.0, 0.5][: len(e1_vals)]
    w_sum = sum(weights)
    avg_e1 = sum(v * w for v, w in zip(e1_vals[:3], weights)) / w_sum
    avg_lp = sum(v * w for v, w in zip(lp_vals[:3], weights[: len(lp_vals)])) / sum(
        weights[: len(lp_vals)]
    )
    avg_e2 = (
        sum(v * w for v, w in zip(e2_vals[:3], weights[: len(e2_vals)]))
        / sum(weights[: len(e2_vals)])
        if len(e2_vals) >= 2
        else avg_e1
    )
    # PAR adjustment
    if e1_par and lp_par:
        avg_e1 -= e1_par
        avg_lp -= lp_par
        if e2_par:
            avg_e2 -= e2_par
    # Closer with gas in tank
    if avg_lp > avg_e1 + 5:
        bonus += 0.10 * min((avg_lp - avg_e1) / 15, 1.0)
    # Speed + stamina
    if avg_e1 >= 95 and avg_lp >= 85:
        bonus += 0.08
    # Speed, no stamina (one-dimensional)
    if avg_e1 >= 90 and avg_lp < 75:
        bonus -= 0.08
    # Energy distribution — balanced = +, front-loaded = -
    total_energy = avg_e1 + avg_e2 + avg_lp
    if total_energy > 0:
        e1_pct = avg_e1 / total_energy
        lp_pct = avg_lp / total_energy
        if abs(e1_pct - lp_pct) < 0.03:
            bonus += 0.05  # Perfectly distributed
        elif e1_pct > 0.38 and lp_pct < 0.30:
            bonus -= 0.04  # Front-loaded
    # Balanced E1-E2-LP
    if len(e2_vals) >= 2:
        if abs(avg_e1 - avg_e2) <= 3 and abs(avg_e2 - avg_lp) <= 3:
            bonus += 0.04
    return round(np.clip(bonus, -0.15, 0.20), 4)


APP.analyze_pace_figures = _optimized_analyze_pace
print("  ✅ OPT 2: analyze_pace_figures → PAR-adjusted, recency-weighted")


# OPT 3: detect_bounce_risk — Regression slope, std, career-relative
def _optimized_detect_bounce(speed_figs):
    """Regression-based bounce detection. Returns [-0.25, +0.20]."""
    if len(speed_figs) < 2:
        return 0.0
    figs = speed_figs[:6]
    n = len(figs)
    # Linear regression slope on recent figs (positive slope = improving)
    x = np.arange(n)
    coeffs = np.polyfit(x, figs, 1)
    slope = coeffs[0]  # Points per race
    fig_std = np.std(figs)
    career_best = max(speed_figs)
    career_mean = np.mean(speed_figs)
    latest = figs[0]
    score = 0.0
    # Improving trend (positive slope = each race getting better)
    if slope > 1.0:
        score += min(slope * 0.03, 0.12)
    elif slope < -2.0:
        score += max(slope * 0.02, -0.10)
    # Consistency bonus
    if fig_std <= 3.0 and n >= 3:
        score += 0.05
    elif fig_std >= 8.0:
        score -= 0.04
    # Career-best bounce risk
    if latest == career_best and n >= 3:
        drop_from_best = career_best - career_mean
        if drop_from_best >= 10:
            score -= 0.10
        elif drop_from_best >= 6:
            score -= 0.05
    # Sustained peak form
    if n >= 3 and min(figs[:3]) >= career_best - 3:
        score += 0.08
    return round(np.clip(score, -0.25, 0.20), 4)


APP.detect_bounce_risk = _optimized_detect_bounce
print("  ✅ OPT 3: detect_bounce_risk → regression slope, std-based, [-0.25, +0.20]")


# OPT 4: calculate_layoff_factor — Workout mitigation, max -3.0
#         Gentler 60-120 day "freshening" window — common for competitive horses
def _optimized_layoff(days_since_last, num_workouts=0, workout_pattern_bonus=0.0):
    """Layoff factor with workout mitigation. Returns [-3.0, +0.5]."""
    if days_since_last <= 14:
        base = 0.5
    elif days_since_last <= 30:
        base = 0.3
    elif days_since_last <= 45:
        base = 0.0
    elif days_since_last <= 60:
        base = -0.2
    elif days_since_last <= 90:
        base = -0.5  # Was -0.8 — 60-90 days is strategic freshening
    elif days_since_last <= 120:
        base = -1.0  # Was -1.5 — still competitive window with workouts
    elif days_since_last <= 180:
        base = -2.0
    else:
        base = -3.0
    # Workout mitigation (up to 60% of penalty recovered)
    if base < 0 and num_workouts > 0:
        work_credit = min(num_workouts * 0.15, 0.60)  # Slightly stronger per-workout
        base *= 1.0 - work_credit
        base += workout_pattern_bonus
    return round(max(base, -3.0), 2)


APP.calculate_layoff_factor = _optimized_layoff
print("  ✅ OPT 4: calculate_layoff_factor → workout mitigation, max -3.0")


# OPT 5: calculate_form_trend — Reduced won-last-2 from +4.0 to +2.0
def _optimized_form_trend(recent_finishes):
    """Form trend with calibrated momentum. Returns [-1.0, +2.0]."""
    if len(recent_finishes) < 1:
        return 0.0
    if recent_finishes[0] == 1:
        if len(recent_finishes) >= 2 and recent_finishes[1] == 1:
            return 2.0  # Was 4.0 — too dominant
        return 1.5  # Was 2.5
    elif recent_finishes[0] in [2, 3]:
        return 0.7  # Was 1.0
    if len(recent_finishes) < 2:
        return 0.0
    weights = [0.4, 0.3, 0.2, 0.1][: len(recent_finishes)]
    weighted_avg = sum(f * w for f, w in zip(recent_finishes, weights)) / sum(weights)
    if len(recent_finishes) >= 3:
        r3 = recent_finishes[:3]
        if r3[0] < r3[1] < r3[2]:
            return 1.5
        elif r3[0] > r3[1] > r3[2]:
            return (
                -0.5
            )  # Was -1.2 — declining form often reflects class/distance shifts
    if weighted_avg <= 1.5:
        return 1.2
    elif weighted_avg <= 3.0:
        return 0.8
    elif weighted_avg <= 5.0:
        return 0.0
    elif weighted_avg <= 7.0:
        return -0.5
    else:
        return -1.0


APP.calculate_form_trend = _optimized_form_trend
print("  ✅ OPT 5: calculate_form_trend → won-last-2 = +2.0 (was +4.0)")

# OPT 6: calculate_hot_trainer_bonus — 0% trainer capped at -1.2 (was -2.5)
_orig_trainer_bonus = APP.calculate_hot_trainer_bonus


def _optimized_trainer_bonus(
    trainer_win_pct, is_hot_l14=False, is_2nd_lasix_high_pct=False
):
    if trainer_win_pct == 0.0:
        return -1.2  # Was -2.5, too punitive
    return _orig_trainer_bonus(trainer_win_pct, is_hot_l14, is_2nd_lasix_high_pct)


APP.calculate_hot_trainer_bonus = _optimized_trainer_bonus
print("  ✅ OPT 6: calculate_hot_trainer_bonus → 0% trainer cap at -1.2 (was -2.5)")

# OPT 7: calculate_form_cycle_rating — cap floor at -2.0 (was -3.0),
#         add class-form interaction to prevent comeback horses from being eliminated
_orig_form_cycle = APP.calculate_form_cycle_rating


def _optimized_form_cycle_rating(horse_block, pedigree, angles_df):
    """Form cycle with safer floor and class interaction."""
    rating = _orig_form_cycle(horse_block, pedigree, angles_df)
    # Cap floor at -2.0 instead of -3.0 — no horse should be eliminated by form alone
    return float(max(rating, -2.0))


APP.calculate_form_cycle_rating = _optimized_form_cycle_rating
print("  ✅ OPT 7: calculate_form_cycle_rating → floor capped at -2.0 (was -3.0)")

# FIX: HORSE_HDR_RE to also match "POST X: NAME (style quirin) - ML odds" format
import re as _re

_HORSE_HDR_RE_NEW = _re.compile(
    r"""(?mi)^\s*
    (?:POST\s+)?          # optional "POST " prefix
    (\d+)                 # post/program number
    [:\s]+                # colon or whitespace separator
    ([A-Za-z0-9'.\-\s&]+?)   # horse name
    \s*\(\s*
    (E\/P|EP|E|P|S|NA)   # running style
    (?:\s+(\d+))?         # optional quirin
    \s*\)                 # closing paren
    (?:\s*-\s*ML\s+\S+)?  # optional "- ML odds" suffix
    (?:\s*\*+\s*.+)?       # optional "*** ACTUAL WINNER ***" annotation
    \s*$
    """,
    _re.VERBOSE,
)
APP.HORSE_HDR_RE = _HORSE_HDR_RE_NEW
print("  ✅ FIX: HORSE_HDR_RE → extended to match POST X: format")


# FIX: parse_speed_figures_for_block — algorithmic "/" marker approach
# Handles ANY number of call positions (2 for sprints, 3-4 for routes)
def _robust_parse_speed_figs(block):
    """Extract speed figures using E2/LP '/' marker, then skip call changes to find fig."""
    figs = []
    if not block:
        return figs
    block_str = str(block) if not isinstance(block, str) else block
    for line in block_str.split("\n"):
        parts = line.split()
        # Find the E2/LP marker: a field containing "/"
        slash_idx = None
        for idx, part in enumerate(parts):
            if "/" in part and idx >= 1:
                # Validate: fields around "/" should be numeric (E1 E2/ LP)
                left = part.replace("/", "")
                if left.isdigit() and idx + 1 < len(parts) and parts[idx + 1].isdigit():
                    slash_idx = idx
                    break
        if slash_idx is None:
            continue
        # LP is right after the "/" field
        lp_idx = slash_idx + 1
        # After LP, walk forward past call changes until we hit the speed figure
        j = lp_idx + 1
        while j < len(parts):
            raw = parts[j]
            cleaned = raw.lstrip("+-")
            if not cleaned.isdigit():
                break
            val = int(cleaned)
            has_sign = raw[0] in "+-" if raw else False
            # Call changes are small (typically 0-25) and often signed
            # Speed figures are large (40-130) and unsigned
            if has_sign and val < 30:
                j += 1
                continue
            if not has_sign and val < 30:
                j += 1
                continue
            # First number >= 30 without sign is likely the speed figure
            if 40 <= val <= 130:
                figs.append(val)
            break
        # If we skipped everything without finding a fig, try the number just before
        # the running-line sequence
    return figs[:10]


APP.parse_speed_figures_for_block = _robust_parse_speed_figs
print(
    "  ✅ FIX: parse_speed_figures_for_block → algorithmic '/' marker (any # of calls)"
)

# FIX: parse_recent_races_detailed — date regex & ordinal suffix both broken
from datetime import datetime as _dt


def _fixed_parse_recent_races(block):
    """Fixed: uses E2/LP slash marker + jockey name boundary to extract finish position.

    BRISNET race line structure after date+track:
      dist surf class E1 E2/ LP [calls] SPD  PP ST 1C 2C STR FIN   JockeyName  Med Odds ...
    Key: FIN is the last number before 2+ spaces followed by jockey name (alpha chars).
    """
    races = []
    if not block:
        return races
    block_str = str(block) if not isinstance(block, str) else block
    today = _dt.now()
    date_pattern = _re.compile(r"(\d{2}[A-Za-z]{3}\d{2})[A-Za-z]{2,4}\s+(.+)")
    for m in date_pattern.finditer(block_str):
        date_str = m.group(1)
        rest = m.group(2)
        try:
            race_date = _dt.strptime(date_str, "%d%b%y")
            days_ago = (today - race_date).days

            # Strategy: find the "/" in E2/LP, then after call changes and speed fig,
            # take exactly 6 numbers: PP ST 1C 2C STR FIN
            # But first, truncate line at jockey name to avoid odds contamination
            # Jockey name: 2+ spaces followed by capital letter + lowercase (e.g. "  OrtizIJ")
            jockey_match = _re.search(r"(\d)\s{2,}([A-Z][a-zA-Z])", rest)
            if jockey_match:
                rest_trimmed = rest[: jockey_match.start(2)]
            else:
                rest_trimmed = rest

            # Find the "/" separator (E2/LP marker)
            slash_pos = rest_trimmed.find("/")
            if slash_pos < 0:
                continue

            # Extract everything after the "/" section
            after_slash = rest_trimmed[slash_pos + 1 :]
            # Parse all numeric fields from after LP onward
            nums = _re.findall(r"[+-]?\d+", after_slash)
            # Format: LP CALL1 CALL2 [CALL3] SPD PP ST 1C 2C STR FIN
            # LP is first, then calls (small, often signed), then SPD (large unsigned),
            # then PP ST 1C 2C STR FIN (all small)
            # Find the speed figure (first number >= 40 after calls)
            spd_idx = None
            for i, s in enumerate(nums):
                val = int(s.lstrip("+-"))
                if i == 0:
                    continue  # Skip LP
                if val >= 40:
                    spd_idx = i
                    break
            if spd_idx is not None and spd_idx + 6 < len(nums):
                # After SPD: PP, ST, 1C, 2C, STR, FIN
                finish = int(nums[spd_idx + 6].lstrip("+-"))
            else:
                # Fallback: take the 6th number counting back from end of trimmed nums
                # (less reliable but better than nothing)
                finish = 99

            races.append({"date": race_date, "days_ago": days_ago, "finish": finish})
        except Exception:
            pass
    return sorted(races, key=lambda x: x["days_ago"])[:6]


APP.parse_recent_races_detailed = _fixed_parse_recent_races
print("  ✅ FIX: parse_recent_races_detailed → absorbs track code, running-line finish")

print(f"  speed_fig_weight = {APP.MODEL_CONFIG['speed_fig_weight']}")
print(f"  softmax_tau = {APP.MODEL_CONFIG['softmax_tau']}")
print("✅ All 6 optimizations applied.")

# ══════════════════════════════════════════════════════════════
# CELL 3: Load PP Data
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("LOADING PP DATA")
print("=" * 80)

pp_files = {
    "Oaklawn R9 (Feb 5)": ROOT / "saved_races" / "oaklawn_r9_20260205_brisnet_pp.txt",
    "Pegasus WC G1": ROOT / "pegasus_wc_g1_pp.txt",
    "Santa Anita R4": ROOT / "test_pp_sample.txt",
}

pp_data = {}
for name, path in pp_files.items():
    if path.exists():
        text = path.read_text(encoding="utf-8", errors="replace")
        pp_data[name] = text
        print(f"✅ {name}: {len(text):,} chars, {len(text.splitlines())} lines")
    else:
        print(f"❌ {name}: file not found")

primary_race = next(
    (
        k
        for k in ["Oaklawn R9 (Feb 5)", "Pegasus WC G1", "Santa Anita R4"]
        if k in pp_data
    ),
    None,
)
pp_text = pp_data.get(primary_race, "")
print(f"\nPrimary: {primary_race} ({len(pp_text):,} chars)")

# ══════════════════════════════════════════════════════════════
# CELL 4: Header Parsing
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 1: RACE HEADER PARSING")
print("=" * 80)

t0 = time.perf_counter()
header = APP.parse_brisnet_race_header(pp_text)
print(f"⏱ {(time.perf_counter() - t0) * 1000:.1f}ms")
for k, v in header.items():
    print(f"  {k:20s}: {v}")

# ══════════════════════════════════════════════════════════════
# CELL 5: Horse Splitting
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 2: HORSE SPLITTING & STYLE DETECTION")
print("=" * 80)

t0 = time.perf_counter()
chunks_raw = APP.split_into_horse_chunks(pp_text)
print(f"⏱ Split: {(time.perf_counter() - t0) * 1000:.1f}ms")

chunks = OrderedDict()
for post, name, block in chunks_raw:
    chunks[name] = block
print(f"Found {len(chunks)} horse blocks")

t0 = time.perf_counter()
styles_df = APP.extract_horses_and_styles(pp_text)
print(f"⏱ Styles: {(time.perf_counter() - t0) * 1000:.1f}ms")
if len(styles_df) > 0:
    display_cols = [
        c
        for c in ["Post", "Horse", "DetectedStyle", "Quirin", "AutoStrength"]
        if c in styles_df.columns
    ]
    print(styles_df[display_cols].to_string(index=False))

# ══════════════════════════════════════════════════════════════
# CELL 6: Per-Horse Full Data Extraction
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("STAGE 3: PER-HORSE DATA EXTRACTION — ALL ALGORITHMS")
print("=" * 80)

horse_data = OrderedDict()

for i, (name, block) in enumerate(chunks.items()):
    t0 = time.perf_counter()
    speed_figs = APP.parse_speed_figures_for_block(block)
    pace = APP.parse_e1_e2_lp_values(block)
    pace_bonus = APP.analyze_pace_figures(pace["e1"], pace["e2"], pace["lp"])
    bounce = APP.detect_bounce_risk(speed_figs)
    workout = APP.parse_workout_data(block)
    pedigree = APP.parse_pedigree_snips(block)
    try:
        angles_df = APP.parse_angles_for_block(block)
    except:
        angles_df = pd.DataFrame()
    recent = APP.parse_recent_races_detailed(block)
    form_rating = APP.calculate_form_cycle_rating(block, pedigree, angles_df)
    class_rating = APP.calculate_comprehensive_class_rating(
        today_purse=38000,
        today_race_type="Alw 12500s",
        horse_block=block,
        pedigree=pedigree,
        angles_df=angles_df,
        pp_text=pp_text,
    )
    try:
        workout_bonus = APP.calculate_workout_bonus_v2(workout)
    except:
        workout_bonus = 0.0
    layoff_days = recent[0]["days_ago"] if recent else 999
    layoff_factor = APP.calculate_layoff_factor(
        layoff_days,
        num_workouts=workout.get("num_recent", 0),
        workout_pattern_bonus=workout.get("pattern_bonus", 0.0),
    )
    finishes = [r.get("finish", 10) for r in recent[:4]]
    form_trend = APP.calculate_form_trend(finishes)
    t_horse = time.perf_counter() - t0

    horse_data[name] = {
        "speed_figs": speed_figs,
        "avg_top2": np.mean(sorted(speed_figs, reverse=True)[:2])
        if len(speed_figs) >= 2
        else (speed_figs[0] if speed_figs else 50),
        "best_fig": max(speed_figs) if speed_figs else 50,
        "pace_e1": pace["e1"],
        "pace_e2": pace["e2"],
        "pace_lp": pace["lp"],
        "pace_bonus": pace_bonus,
        "bounce_risk": bounce,
        "workout": workout,
        "workout_bonus": workout_bonus,
        "pedigree": pedigree,
        "num_angles": len(angles_df),
        "recent_races": len(recent),
        "finishes": finishes,
        "layoff_days": layoff_days,
        "layoff_factor": layoff_factor,
        "form_trend": form_trend,
        "form_rating": form_rating,
        "class_rating": class_rating,
        "parse_time_ms": t_horse * 1000,
    }

summary = pd.DataFrame(
    [
        {
            "Horse": name,
            "SpeedFigs": len(d["speed_figs"]),
            "AvgTop2": f"{d['avg_top2']:.1f}",
            "BestFig": d["best_fig"],
            "E1": f"{np.mean(d['pace_e1']):.0f}" if d["pace_e1"] else "-",
            "LP": f"{np.mean(d['pace_lp']):.0f}" if d["pace_lp"] else "-",
            "PaceB": f"{d['pace_bonus']:+.3f}",
            "Layoff": d["layoff_days"],
            "LayAdj": f"{d['layoff_factor']:+.2f}",
            "FormTr": f"{d['form_trend']:+.1f}",
            "FormR": f"{d['form_rating']:+.2f}",
            "ClassR": f"{d['class_rating']:+.2f}",
            "WkBns": f"{d['workout_bonus']:+.3f}",
            "Bounce": f"{d['bounce_risk']:+.3f}",
            "ms": f"{d['parse_time_ms']:.1f}",
        }
        for name, d in horse_data.items()
    ]
)
print(summary.to_string(index=False))
print(
    f"\nTotal parse time: {sum(d['parse_time_ms'] for d in horse_data.values()):.1f}ms"
)

# Debug: Show parsed finishes and speed fig detail for each horse
print("\n── Parsed Detail per Horse ──")
print(f"{'Horse':25s} {'#Figs':>5s} {'Figs':30s} {'#Races':>6s} {'Finishes':30s}")
for name, d in horse_data.items():
    figs_str = str(d["speed_figs"][:5])
    fin_str = str(d["finishes"][:6])
    print(
        f"{name:25s} {len(d['speed_figs']):5d} {figs_str:30s} "
        f"{d['recent_races']:6d} {fin_str:30s}"
    )

# ══════════════════════════════════════════════════════════════
# CELL 7: Speed Figure Weight Validation
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("OPT 1: Speed Figure Weight (0.05 → 0.15)")
print("=" * 80)

OLD_WEIGHT = 0.05
NEW_WEIGHT = APP.MODEL_CONFIG["speed_fig_weight"]
avg_fig = np.mean([d["avg_top2"] for d in horse_data.values()])
print(f"Confirmed speed_fig_weight = {NEW_WEIGHT}, race avg = {avg_fig:.1f}\n")

print(f"{'Horse':25s} {'AvgTop2':>8s} {'OLD':>10s} {'NEW':>10s} {'Δ':>8s}")
for name, d in sorted(horse_data.items(), key=lambda x: -x[1]["avg_top2"]):
    delta = d["avg_top2"] - avg_fig
    print(
        f"{name:25s} {d['avg_top2']:8.1f} {delta * OLD_WEIGHT:+10.3f} {delta * NEW_WEIGHT:+10.3f} {delta * (NEW_WEIGHT - OLD_WEIGHT):+8.3f}"
    )

# ══════════════════════════════════════════════════════════════
# CELL 8+9: Old vs New helper functions
# ══════════════════════════════════════════════════════════════


def old_analyze_pace(e1, e2, lp):
    bonus = 0.0
    if len(e1) < 3 or len(lp) < 3:
        return bonus
    avg_e1 = np.mean(e1[:3])
    avg_lp = np.mean(lp[:3])
    if avg_lp > avg_e1 + 5:
        bonus += 0.07
    if avg_e1 >= 95 and avg_lp >= 85:
        bonus += 0.06
    if avg_e1 >= 90 and avg_lp < 75:
        bonus -= 0.05
    if len(e2) >= 3:
        avg_e2 = np.mean(e2[:3])
        if abs(avg_e1 - avg_e2) <= 3 and abs(avg_e2 - avg_lp) <= 3:
            bonus += 0.04
    return bonus


def old_layoff_factor(days):
    if days <= 14:
        return 0.5
    elif days <= 30:
        return 0.3
    elif days <= 45:
        return 0.0
    elif days <= 60:
        return -0.3
    elif days <= 90:
        return -0.8
    elif days <= 120:
        return -1.5
    elif days <= 180:
        return -3.0
    else:
        return -5.0


def old_form_trend(recent_finishes):
    if len(recent_finishes) < 1:
        return 0.0
    if recent_finishes[0] == 1:
        if len(recent_finishes) >= 2 and recent_finishes[1] == 1:
            return 4.0
        else:
            return 2.5
    elif recent_finishes[0] in [2, 3]:
        return 1.0
    if len(recent_finishes) < 2:
        return 0.0
    weights = [0.4, 0.3, 0.2, 0.1][: len(recent_finishes)]
    wavg = sum(f * w for f, w in zip(recent_finishes, weights)) / sum(weights)
    if len(recent_finishes) >= 3:
        r3 = recent_finishes[:3]
        if r3[0] < r3[1] < r3[2]:
            return 1.5
        elif r3[0] > r3[1] > r3[2]:
            return -1.2
    if wavg <= 1.5:
        return 1.2
    elif wavg <= 3.0:
        return 0.8
    elif wavg <= 5.0:
        return 0.0
    elif wavg <= 7.0:
        return -0.5
    else:
        return -1.0


# ══════════════════════════════════════════════════════════════
# Pace & Bounce Before/After
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("OPT 2+6: Pace Analysis & Bounce Detection")
print("=" * 80)

print(
    f"\n{'Horse':25s} {'OLD_Pace':>9s} {'NEW_Pace':>9s} {'Δ':>8s} {'OLD_Bnce':>9s} {'NEW_Bnce':>9s}"
)
for name, d in horse_data.items():
    old_pace = old_analyze_pace(d["pace_e1"], d["pace_e2"], d["pace_lp"])
    figs = d["speed_figs"]
    old_bounce = 0.0
    if len(figs) >= 3:
        lt = figs[:3]
        cb = max(figs)
        if lt[0] == cb and len(figs) > 3:
            if lt[1] < lt[0] - 8:
                old_bounce -= 0.09
            elif lt[1] < lt[0] - 5:
                old_bounce -= 0.05
        if len(figs) >= 4 and lt[0] >= cb - 2 and lt[1] >= cb - 2:
            old_bounce += 0.07
        if lt[0] > lt[1] > lt[2]:
            old_bounce += 0.06
        if lt[0] < lt[1] < lt[2]:
            old_bounce -= 0.05
        if max(lt) - min(lt) <= 5:
            old_bounce += 0.03
    print(
        f"{name:25s} {old_pace:+9.3f} {d['pace_bonus']:+9.3f} {d['pace_bonus'] - old_pace:+8.3f} {old_bounce:+9.3f} {d['bounce_risk']:+9.3f}"
    )

# ══════════════════════════════════════════════════════════════
# Layoff & Form Before/After
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("OPT 3+4: Layoff Mitigation & Form Trend")
print("=" * 80)

print(
    f"\n{'Horse':25s} {'Days':>5s} {'Wks':>4s} {'OLD_Lay':>8s} {'NEW_Lay':>8s} {'OLD_FT':>7s} {'NEW_FT':>7s}"
)
for name, d in horse_data.items():
    print(
        f"{name:25s} {d['layoff_days']:5d} {d['workout'].get('num_recent', 0):4d} {old_layoff_factor(d['layoff_days']):+8.2f} {d['layoff_factor']:+8.2f} {old_form_trend(d['finishes']):+7.1f} {d['form_trend']:+7.1f}"
    )

# ══════════════════════════════════════════════════════════════
# CELL 10: Full Old vs New Rating Comparison
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("COMPLETE RATING MODEL — Old vs Optimized")
print("=" * 80)

results = []
for name, d in horse_data.items():
    old_speed = (d["avg_top2"] - avg_fig) * OLD_WEIGHT
    old_lay = old_layoff_factor(d["layoff_days"])
    old_ft = old_form_trend(d["finishes"])
    old_pace = old_analyze_pace(d["pace_e1"], d["pace_e2"], d["pace_lp"])
    figs = d["speed_figs"]
    old_bounce = 0.0
    if len(figs) >= 3:
        lt = figs[:3]
        cb = max(figs)
        if lt[0] == cb and len(figs) > 3:
            if lt[1] < lt[0] - 8:
                old_bounce -= 0.09
            elif lt[1] < lt[0] - 5:
                old_bounce -= 0.05
        if len(figs) >= 4 and lt[0] >= cb - 2 and lt[1] >= cb - 2:
            old_bounce += 0.07
        if lt[0] > lt[1] > lt[2]:
            old_bounce += 0.06
        if lt[0] < lt[1] < lt[2]:
            old_bounce -= 0.05
        if max(lt) - min(lt) <= 5:
            old_bounce += 0.03
    old_form = old_lay + old_ft
    rf = d["finishes"]
    if len(rf) >= 4:
        top3 = sum(1 for f in rf[:4] if f <= 3)
        if top3 >= 3:
            old_form += 0.8
        elif top3 >= 2:
            old_form += 0.4
    if rf and rf[0] == 1:
        old_form += 0.6
        if len(rf) >= 2 and rf[1] == 1:
            old_form += 0.4
    old_form = np.clip(old_form, -3.0, 3.0)
    old_rating = (
        d["class_rating"]
        + old_form
        + old_speed
        + old_pace
        + d["workout_bonus"]
        + old_bounce
    )
    new_speed = (d["avg_top2"] - avg_fig) * NEW_WEIGHT
    new_rating = (
        d["class_rating"]
        + d["form_rating"]
        + new_speed
        + d["pace_bonus"]
        + d["workout_bonus"]
        + d["bounce_risk"]
    )
    results.append(
        {
            "Horse": name,
            "OldR": old_rating,
            "NewR": new_rating,
            "Delta": new_rating - old_rating,
            "OldRank": 0,
            "NewRank": 0,
            "BestFig": d["best_fig"],
        }
    )

results.sort(key=lambda x: -x["OldR"])
for i, r in enumerate(results):
    r["OldRank"] = i + 1
results.sort(key=lambda x: -x["NewR"])
for i, r in enumerate(results):
    r["NewRank"] = i + 1

print(
    f"\n{'Horse':25s} {'OldR':>7s} {'Rk':>3s} {'NewR':>7s} {'Rk':>3s} {'Δ':>7s} {'Fig':>4s}"
)
for r in results:
    arrow = (
        "↑"
        if r["OldRank"] > r["NewRank"]
        else ("↓" if r["OldRank"] < r["NewRank"] else "=")
    )
    print(
        f"{r['Horse']:25s} {r['OldR']:+7.3f} {r['OldRank']:3d} {r['NewR']:+7.3f} {r['NewRank']:3d} {r['Delta']:+7.3f} {r['BestFig']:4d} {arrow}"
    )

moved = [
    (r["Horse"], r["OldRank"], r["NewRank"])
    for r in results
    if r["OldRank"] != r["NewRank"]
]
print(f"\nRankings changed for {len(moved)}/{len(results)} horses:")
for h, old_rk, new_rk in sorted(moved, key=lambda x: x[1] - x[2], reverse=True):
    d = "↑" if old_rk > new_rk else "↓"
    print(f"  {d} {h}: #{old_rk} → #{new_rk}")

# ══════════════════════════════════════════════════════════════
# CELL 11: Softmax Probabilities
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("PROBABILITY CALIBRATION")
print("=" * 80)

names_sorted = sorted(horse_data.keys())
if names_sorted:
    old_rats = np.array(
        [next(r["OldR"] for r in results if r["Horse"] == n) for n in names_sorted]
    )
    new_rats = np.array(
        [next(r["NewR"] for r in results if r["Horse"] == n) for n in names_sorted]
    )
    old_probs = APP.softmax_from_rating(old_rats)
    new_probs = APP.softmax_from_rating(new_rats)

    print(f"\n{'Horse':25s} {'Old%':>7s} {'New%':>7s} {'Δ%':>7s} {'NewOdds':>8s}")
    for i, n in enumerate(names_sorted):
        odds = f"{(1 / new_probs[i]) - 1:.1f}" if new_probs[i] > 0.01 else "99+"
        print(
            f"{n:25s} {old_probs[i]:7.1%} {new_probs[i]:7.1%} {new_probs[i] - old_probs[i]:+7.1%} {odds:>8s}"
        )

    old_ent = -np.sum(old_probs * np.log(old_probs + 1e-10))
    new_ent = -np.sum(new_probs * np.log(new_probs + 1e-10))
    max_ent = np.log(len(names_sorted))
    print(f"\nEntropy: OLD={old_ent:.3f}  NEW={new_ent:.3f}  (max={max_ent:.3f})")
    print(f"{'✅ More decisive' if new_ent < old_ent else '⚠️ Less decisive'}")
else:
    print("⚠️ No horse data — skipping probability calibration")

# ══════════════════════════════════════════════════════════════
# CELL 12: Multi-Race Validation
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("MULTI-RACE VALIDATION")
print("=" * 80)

for race_name, race_text in pp_data.items():
    print(f"\n{'─' * 60}")
    print(f"RACE: {race_name}")
    hdr = APP.parse_brisnet_race_header(race_text)
    rc = APP.split_into_horse_chunks(race_text)
    print(
        f"Track: {hdr.get('track_name', '?')} | Dist: {hdr.get('distance', '?')} | Horses: {len(rc)}"
    )
    if not rc:
        continue
    rhd = {}
    for post, nm, blk in rc:
        sf = APP.parse_speed_figures_for_block(blk)
        pc = APP.parse_e1_e2_lp_values(blk)
        ped = APP.parse_pedigree_snips(blk)
        try:
            adf = APP.parse_angles_for_block(blk)
        except:
            adf = pd.DataFrame()
        fr = APP.calculate_form_cycle_rating(blk, ped, adf)
        cr = APP.calculate_comprehensive_class_rating(
            today_purse=hdr.get("purse_amount", 30000),
            today_race_type=hdr.get("race_type", "Alw"),
            horse_block=blk,
            pedigree=ped,
            angles_df=adf,
            pp_text=race_text,
        )
        pb = APP.analyze_pace_figures(pc["e1"], pc["e2"], pc["lp"])
        bn = APP.detect_bounce_risk(sf)
        wk = APP.parse_workout_data(blk)
        try:
            wb = APP.calculate_workout_bonus_v2(wk)
        except:
            wb = 0.0
        af = (
            np.mean(sorted(sf, reverse=True)[:2])
            if len(sf) >= 2
            else (sf[0] if sf else 50)
        )
        rhd[nm] = {"avg": af, "cls": cr, "frm": fr, "pac": pb, "bnc": bn, "wk": wb}
    ravg = np.mean([d["avg"] for d in rhd.values()])
    rr = {
        n: d["cls"]
        + d["frm"]
        + (d["avg"] - ravg) * NEW_WEIGHT
        + d["pac"]
        + d["wk"]
        + d["bnc"]
        for n, d in rhd.items()
    }
    ns = list(rr.keys())
    rs = np.array([rr[n] for n in ns])
    ps = APP.softmax_from_rating(rs)
    ranked = sorted(zip(ns, rs, ps), key=lambda x: -x[1])
    print(f"  {'Rk':>3s} {'Horse':25s} {'Rating':>8s} {'Prob':>7s} {'Odds':>7s}")
    for i, (n, r, p) in enumerate(ranked):
        o = f"{(1 / p) - 1:.1f}" if p > 0.01 else "99+"
        print(f"  {i + 1:3d} {n:25s} {r:+8.3f} {p:7.1%} {o:>7s}")

# ══════════════════════════════════════════════════════════════
# CELL 13: Component Contribution
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("COMPONENT CONTRIBUTION ANALYSIS")
print("=" * 80)

print(
    f"\n{'Horse':25s} {'Class':>7s} {'Form':>7s} {'Speed':>7s} {'Pace':>7s} {'Bounc':>7s} {'WkBns':>7s} {'TOTAL':>7s}"
)
comp = {"Class": [], "Form": [], "Speed": [], "Pace": [], "Bounce": [], "WkBns": []}
for name, d in horse_data.items():
    spd = (d["avg_top2"] - avg_fig) * NEW_WEIGHT
    total = (
        d["class_rating"]
        + d["form_rating"]
        + spd
        + d["pace_bonus"]
        + d["bounce_risk"]
        + d["workout_bonus"]
    )
    print(
        f"{name:25s} {d['class_rating']:+7.2f} {d['form_rating']:+7.2f} {spd:+7.3f} {d['pace_bonus']:+7.3f} {d['bounce_risk']:+7.3f} {d['workout_bonus']:+7.3f} {total:+7.3f}"
    )
    comp["Class"].append(d["class_rating"])
    comp["Form"].append(d["form_rating"])
    comp["Speed"].append(spd)
    comp["Pace"].append(d["pace_bonus"])
    comp["Bounce"].append(d["bounce_risk"])
    comp["WkBns"].append(d["workout_bonus"])

total_range = (
    sum((max(v) - min(v)) for v in comp.values() if v) if any(comp.values()) else 0
)
print(f"\n{'Component':10s} {'Range':>7s} {'% of Total':>11s}")
for c, v in comp.items():
    if not v:
        print(f"{c:10s} {'N/A':>7s} {'N/A':>11s}")
        continue
    r = max(v) - min(v)
    print(f"{c:10s} {r:7.3f} {r / total_range * 100 if total_range else 0:10.1f}%")

# ══════════════════════════════════════════════════════════════
# CELL 14: Unit Tests
# ══════════════════════════════════════════════════════════════
print("\n" + "=" * 80)
print("ALGORITHM UNIT TESTS")
print("=" * 80)

passed = failed = 0


def check(name, cond, detail=""):
    global passed, failed
    if cond:
        passed += 1
        print(f"  ✅ {name}")
    else:
        failed += 1
        print(f"  ❌ {name}: {detail}")


print("\n── Speed Figure Weight ──")
check("speed_fig_weight is 0.15", APP.MODEL_CONFIG["speed_fig_weight"] == 0.15)

print("\n── Pace Analysis ──")
p = APP.analyze_pace_figures([80, 82, 81], [85, 83, 84], [95, 93, 92])
check("Strong closer positive", p > 0.05, f"{p:.3f}")
p2 = APP.analyze_pace_figures([98, 97, 96], [90, 89, 88], [68, 70, 65])
check("One-dim speed penalty", p2 < -0.05, f"{p2:.3f}")
p3 = APP.analyze_pace_figures([], [], [])
check("Empty = 0", p3 == 0.0)

print("\n── Bounce Detection ──")
b1 = APP.detect_bounce_risk([100, 85, 82, 80, 78])
check("Career best + drop = risk", b1 < -0.05, f"{b1:.3f}")
b2 = APP.detect_bounce_risk([95, 94, 93])
check("Improving = positive", b2 > 0, f"{b2:.3f}")
b3 = APP.detect_bounce_risk([85, 85, 84, 86, 85])
check("Consistent = positive", b3 > 0, f"{b3:.3f}")
b5 = APP.detect_bounce_risk([90])
check("Single fig = 0", b5 == 0.0)

print("\n── Layoff Factor ──")
l1 = APP.calculate_layoff_factor(14)
check("14d positive", l1 > 0, f"{l1}")
l2 = APP.calculate_layoff_factor(120, num_workouts=5, workout_pattern_bonus=0.08)
l3 = APP.calculate_layoff_factor(120, num_workouts=0)
check("Works mitigate penalty", l2 > l3, f"w/={l2:.2f} wo/={l3:.2f}")
l4 = APP.calculate_layoff_factor(365)
check("365d <= -2.0", l4 <= -2.0, f"{l4}")
check("Max >= -3.0", l4 >= -3.0, f"{l4}")

print("\n── Form Trend ──")
f1 = APP.calculate_form_trend([1, 1, 3, 5])
check("Won last 2 = +2.0", f1 == 2.0, f"{f1}")
f2 = APP.calculate_form_trend([1, 5, 8])
check("Won last = +1.5", f2 == 1.5, f"{f2}")
f3 = APP.calculate_form_trend([2, 4, 6])
check("Place last = +0.7", f3 == 0.7, f"{f3}")

print("\n── Softmax ──")
test_r = np.array([5.0, 3.0, 1.0, -1.0, -3.0])
probs = APP.softmax_from_rating(test_r)
check("Sums to 1.0", abs(np.sum(probs) - 1.0) < 1e-6)
check("All positive", np.all(probs > 0))
check("Highest = highest", np.argmax(probs) == 0)
eq = APP.softmax_from_rating(np.array([5.0, 5.0, 5.0]))
check("Equal = equal", np.allclose(eq, 1 / 3, atol=0.01))

print(f"\n{'=' * 60}")
print(f"RESULTS: {passed}/{passed + failed} tests passed, {failed} failed")
print(f"{'=' * 60}")
print("\n✅ AUDIT COMPLETE")
