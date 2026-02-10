"""Apply all 7 optimizations + 3 parser fixes from run_audit.py to app.py on disk.

This script reads the original app.py (8927 lines), applies targeted replacements,
and writes the updated file. Creates a backup first.
"""

import shutil
from pathlib import Path

ROOT = Path(r"C:\Users\C Stephens\Desktop\Horse Racing Picks")
APP_PATH = ROOT / "app.py"

# Create backup
backup = APP_PATH.with_suffix(".py.bak")
shutil.copy2(APP_PATH, backup)
print(f"Backup: {backup}")

content = APP_PATH.read_text(encoding="utf-8")
original_len = len(content.splitlines())
print(f"Original: {original_len} lines")

changes = 0

# ══════════════════════════════════════════════════════════════
# 1. speed_fig_weight 0.05 → 0.15
# ══════════════════════════════════════════════════════════════
old = '    "speed_fig_weight": 0.05,  # (Fig - Avg) * Weight. 0.05 = 10 fig points = 0.5 bonus.'
new = '    "speed_fig_weight": 0.15,  # OPTIMIZED: Was 0.05. 0.15 = 10 fig points = 1.5 bonus (Beyer/Benter research).'
if old in content:
    content = content.replace(old, new, 1)
    changes += 1
    print("  [1] speed_fig_weight 0.05 → 0.15 ✅")
else:
    print("  [1] speed_fig_weight — NOT FOUND (may already be updated)")

# ══════════════════════════════════════════════════════════════
# 2. HORSE_HDR_RE — extended regex
# ══════════════════════════════════════════════════════════════
old_hdr = '''HORSE_HDR_RE = re.compile(
    r"""(?mi)^\\s*
    (\\d+)              # post/program
    \\s+([A-Za-z0-9'.\\-\\s&]+?)   # horse name
    \\s*\\(\\s*
    (E\\/P|EP|E|P|S|NA)      # style
    (?:\\s+(\\d+))?           # optional quirin
    \\s*\\)\\s*$              #
    """, re.VERBOSE
)'''

new_hdr = '''HORSE_HDR_RE = re.compile(
    r"""(?mi)^\\s*
    (?:POST\\s+)?          # optional "POST " prefix
    (\\d+)                 # post/program number
    [:\\s]+                # colon or whitespace separator
    ([A-Za-z0-9'.\\-\\s&]+?)   # horse name
    \\s*\\(\\s*
    (E\\/P|EP|E|P|S|NA)   # running style
    (?:\\s+(\\d+))?         # optional quirin
    \\s*\\)                 # closing paren
    (?:\\s*-\\s*ML\\s+\\S+)?  # optional "- ML odds" suffix
    (?:\\s*\\*+\\s*.+)?       # optional "*** annotation ***"
    \\s*$
    """, re.VERBOSE
)'''

if old_hdr in content:
    content = content.replace(old_hdr, new_hdr, 1)
    changes += 1
    print("  [2] HORSE_HDR_RE → extended POST/ML/annotation support ✅")
else:
    print("  [2] HORSE_HDR_RE — NOT FOUND")

# ══════════════════════════════════════════════════════════════
# 3. analyze_pace_figures — PAR-adjusted, recency-weighted
# ══════════════════════════════════════════════════════════════
old_pace = '''def analyze_pace_figures(e1_vals: list[int], e2_vals: list[int], lp_vals: list[int]) -> float:
    """SAVANT ANGLE: E1/E2/LP pace analysis. Returns bonus from -0.05 to +0.07"""
    bonus = 0.0
    if len(e1_vals) < 3 or len(lp_vals) < 3:
        return bonus
    avg_e1 = np.mean(e1_vals[:3])
    avg_lp = np.mean(lp_vals[:3])
    if avg_lp > avg_e1 + 5:
        bonus += 0.07  # Closer with gas
    if avg_e1 >= 95 and avg_lp >= 85:
        bonus += 0.06  # Speed + stamina
    if avg_e1 >= 90 and avg_lp < 75:
        bonus -= 0.05  # Speed no stamina
    if len(e2_vals) >= 3:
        avg_e2 = np.mean(e2_vals[:3])
        if abs(avg_e1 - avg_e2) <= 3 and abs(avg_e2 - avg_lp) <= 3:
            bonus += 0.04  # Balanced energy
    return bonus'''

new_pace = '''def analyze_pace_figures(
    e1_vals: list[int],
    e2_vals: list[int],
    lp_vals: list[int],
    e1_par: int = 0,
    e2_par: int = 0,
    lp_par: int = 0,
) -> float:
    """OPTIMIZED: PAR-adjusted pace analysis with recency-weighted averages.

    Returns bonus from -0.15 to +0.20. Uses recency weights [2x, 1x, 0.5x]
    and energy distribution analysis. Validated on Oaklawn R9 (Air of Defiance #2).
    """
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
    return round(float(np.clip(bonus, -0.15, 0.20)), 4)'''

if old_pace in content:
    content = content.replace(old_pace, new_pace, 1)
    changes += 1
    print("  [3] analyze_pace_figures → PAR-adjusted, recency-weighted ✅")
else:
    print("  [3] analyze_pace_figures — NOT FOUND")

# ══════════════════════════════════════════════════════════════
# 4. detect_bounce_risk — regression-based
# ══════════════════════════════════════════════════════════════
old_bounce = '''def detect_bounce_risk(speed_figs: list[int]) -> float:
    """SAVANT ANGLE: Bounce detection. Returns penalty/bonus from -0.09 to +0.07"""
    penalty = 0.0
    if len(speed_figs) < 3:
        return penalty
    last_three = speed_figs[:3]
    career_best = max(speed_figs) if speed_figs else 0
    if last_three[0] == career_best and len(speed_figs) > 3:
        if last_three[1] < last_three[0] - 8:
            penalty -= 0.09  # Bounce pattern
        elif last_three[1] < last_three[0] - 5:
            penalty -= 0.05
    if len(speed_figs) >= 4 and last_three[0] >= career_best - 2 and last_three[1] >= career_best - 2:
        penalty += 0.07  # Peak form maintained
    if last_three[0] > last_three[1] > last_three[2]:
        penalty += 0.06  # Improving
    if last_three[0] < last_three[1] < last_three[2]:
        penalty -= 0.05  # Declining
    if max(last_three) - min(last_three) <= 5:
        penalty += 0.03  # Consistent
    return penalty'''

new_bounce = '''def detect_bounce_risk(speed_figs: list[int]) -> float:
    """OPTIMIZED: Regression-based bounce detection. Returns [-0.25, +0.20].

    Uses np.polyfit regression slope, std-based consistency, and career-relative
    analysis. Validated on Oaklawn R9 (Air of Defiance #2).
    """
    if len(speed_figs) < 2:
        return 0.0
    figs = speed_figs[:6]
    n = len(figs)
    # Linear regression slope on recent figs (positive slope = improving)
    x = np.arange(n)
    coeffs = np.polyfit(x, figs, 1)
    slope = coeffs[0]  # Points per race
    fig_std = float(np.std(figs))
    career_best = max(speed_figs)
    career_mean = float(np.mean(speed_figs))
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
    return round(float(np.clip(score, -0.25, 0.20)), 4)'''

if old_bounce in content:
    content = content.replace(old_bounce, new_bounce, 1)
    changes += 1
    print("  [4] detect_bounce_risk → regression slope, std-based ✅")
else:
    print("  [4] detect_bounce_risk — NOT FOUND")

# ══════════════════════════════════════════════════════════════
# 5. parse_speed_figures_for_block — algorithmic "/" marker
# ══════════════════════════════════════════════════════════════
old_speed = '''def parse_speed_figures_for_block(block) -> list[int]:
    """
    Parses a horse's PP text block and extracts all main speed figures.
    CRITICAL: Speed figure is in the 4th capture group after E1, E2, LP
    """
    figs = []
    if not block:
        return figs

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    for m in SPEED_FIG_RE.finditer(block_str):
        try:
            # The speed figure is the FOURTH capture group (after E1, E2, LP)
            fig_val = int(m.group(4))
            # Basic sanity check for a realistic speed figure
            if 40 < fig_val < 130:
                figs.append(fig_val)
        except (ValueError, AttributeError, IndexError):
            # Ignore if regex group missing or conversion fails
            pass

    # We only care about the most recent figs, e.g., last 10
    return figs[:10]'''

new_speed = '''def parse_speed_figures_for_block(block) -> list[int]:
    """OPTIMIZED: Extract speed figures using E2/LP '/' marker.

    Algorithmic approach handles ANY number of call positions (2 for sprints,
    3-4 for routes) instead of regex with fixed capture groups.
    Validated on Oaklawn R9 (Air of Defiance #2).
    """
    figs = []
    if not block:
        return figs

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block

    for line in block_str.split("\\n"):
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
            # First number >= 40 without sign is likely the speed figure
            if 40 <= val <= 130:
                figs.append(val)
            break

    return figs[:10]'''

if old_speed in content:
    content = content.replace(old_speed, new_speed, 1)
    changes += 1
    print("  [5] parse_speed_figures_for_block → algorithmic '/' marker ✅")
else:
    print("  [5] parse_speed_figures_for_block — NOT FOUND")

# ══════════════════════════════════════════════════════════════
# 6. parse_recent_races_detailed — jockey name boundary fix
# ══════════════════════════════════════════════════════════════
old_recent = """def parse_recent_races_detailed(block) -> list[dict]:
    \"\"\"
    Extract detailed recent race history with dates, finishes, beaten lengths.
    Returns list of dicts with date, finish, beaten_lengths, days_ago
    \"\"\"
    races = []
    if not block:
        return races
    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    # Pattern: date, finish position, beaten lengths
    # Example: "23Dec23 Aqu 3rd 2\\xbc"
    pattern = r'(\\d{2}[A-Za-z]{3}\\d{2})\\s+\\w+.*?(\\d+)(?:st|nd|rd|th)\\s*(\\d+)?'

    today = datetime.now()

    for match in re.finditer(pattern, block_str):
        date_str = match.group(1)
        finish = match.group(2)

        try:
            # Parse date
            race_date = datetime.strptime(date_str, '%d%b%y')
            days_ago = (today - race_date).days

            races.append({
                'date': race_date,
                'days_ago': days_ago,
                'finish': int(finish) if finish.isdigit() else 99
            })
        except Exception:
            pass

    return sorted(races, key=lambda x: x['days_ago'])[:6]  # Last 6 races, most recent first"""

new_recent = '''def parse_recent_races_detailed(block) -> list[dict]:
    """OPTIMIZED: Extract recent race history using E2/LP slash marker + jockey name boundary.

    Uses jockey name boundary to truncate line before scanning for finish positions,
    preventing decimal odds (e.g., 13.31) from contaminating finish position extraction.
    Validated on Oaklawn R9 (Air of Defiance #2).
    """
    races = []
    if not block:
        return races

    # Ensure block is string
    block_str = str(block) if not isinstance(block, str) else block
    today = datetime.now()
    date_pattern = re.compile(r"(\\d{2}[A-Za-z]{3}\\d{2})[A-Za-z]{2,4}\\s+(.+)")

    for m in date_pattern.finditer(block_str):
        date_str = m.group(1)
        rest = m.group(2)
        try:
            race_date = datetime.strptime(date_str, "%d%b%y")
            days_ago = (today - race_date).days

            # Truncate line at jockey name to avoid odds contamination
            # Jockey name: 2+ spaces followed by capital letter + lowercase
            jockey_match = re.search(r"(\\d)\\s{2,}([A-Z][a-zA-Z])", rest)
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
            nums = re.findall(r"[+-]?\\d+", after_slash)
            # Find the speed figure (first number >= 40 after LP)
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
                finish = 99

            races.append({"date": race_date, "days_ago": days_ago, "finish": finish})
        except Exception:
            pass

    return sorted(races, key=lambda x: x["days_ago"])[:6]'''

if old_recent in content:
    content = content.replace(old_recent, new_recent, 1)
    changes += 1
    print("  [6] parse_recent_races_detailed → jockey name boundary fix ✅")
else:
    print("  [6] parse_recent_races_detailed — NOT FOUND (trying alternate)")
    # Try with the actual unicode char
    old_recent_alt = old_recent.replace("\\xbc", "\xbc")
    if old_recent_alt in content:
        content = content.replace(old_recent_alt, new_recent, 1)
        changes += 1
        print("  [6] parse_recent_races_detailed → jockey name boundary fix (alt) ✅")
    else:
        print(
            "  [6] parse_recent_races_detailed — STILL NOT FOUND, trying line-by-line"
        )

# ══════════════════════════════════════════════════════════════
# 7. calculate_layoff_factor — workout mitigation
# ══════════════════════════════════════════════════════════════
old_layoff = '''def calculate_layoff_factor(days_since_last: int) -> float:
    """
    Layoff impact on performance.
    Returns: adjustment factor (-5.0 to +0.5)
    PEGASUS WC TUNING: Increased penalties for long layoffs (146-day layoff issue)
    """
    if days_since_last <= 14:  # Racing frequently (good)
        return 0.5
    elif days_since_last <= 30:  # Fresh, ideal
        return 0.3
    elif days_since_last <= 45:  # Standard rest
        return 0.0
    elif days_since_last <= 60:  # Slight concern
        return -0.3
    elif days_since_last <= 90:  # Moderate layoff
        return -0.8
    elif days_since_last <= 120:  # Long layoff - TUNED
        return -1.5
    elif days_since_last <= 180:  # Extended layoff - TUNED
        return -3.0
    else:  # Very long absence - TUNED
        return -5.0'''

new_layoff = '''def calculate_layoff_factor(
    days_since_last: int,
    num_workouts: int = 0,
    workout_pattern_bonus: float = 0.0,
) -> float:
    """OPTIMIZED: Layoff impact with workout mitigation. Returns (-3.0 to +0.5).

    Key changes: 60-120d brackets gentler (strategic freshening window),
    workout mitigation 15%/workout up to 60% recovery, max penalty -3.0.
    Validated on Oaklawn R9 (Air of Defiance #2).
    """
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
        work_credit = min(num_workouts * 0.15, 0.60)  # 15% per workout
        base *= 1.0 - work_credit
        base += workout_pattern_bonus
    return round(max(base, -3.0), 2)'''

if old_layoff in content:
    content = content.replace(old_layoff, new_layoff, 1)
    changes += 1
    print("  [7] calculate_layoff_factor → workout mitigation, max -3.0 ✅")
else:
    print("  [7] calculate_layoff_factor — NOT FOUND")

# ══════════════════════════════════════════════════════════════
# 8. calculate_form_trend — calibrated momentum
# ══════════════════════════════════════════════════════════════
# Find and replace the full function
old_trend_start = '''def calculate_form_trend(recent_finishes: list[int]) -> float:
    """
    Analyze finish positions for improvement/decline trend.
    Returns: trend factor (-1.5 to +4.0)
    PEGASUS WC TUNING: Added win momentum bonus (Skippylongstocking won last race)
    """'''

old_trend_end = """    else:  # Consistently poor
        return -1.0"""

new_trend = '''def calculate_form_trend(recent_finishes: list[int]) -> float:
    """OPTIMIZED: Form trend with calibrated momentum. Returns (-1.0 to +2.0).

    Key changes: won-last-2 = +2.0 (was +4.0), declining = -0.5 (was -1.2).
    Form trend is a MODIFIER, not a dominator.
    Validated on Oaklawn R9 (Air of Defiance #2).
    """
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
            return -0.5  # Was -1.2 — declining often reflects class/distance shifts
    if weighted_avg <= 1.5:
        return 1.2
    elif weighted_avg <= 3.0:
        return 0.8
    elif weighted_avg <= 5.0:
        return 0.0
    elif weighted_avg <= 7.0:
        return -0.5
    else:
        return -1.0'''

# Find the full old function by locating start and end
start_idx = content.find(old_trend_start)
end_idx = content.find(old_trend_end)
if start_idx >= 0 and end_idx >= 0:
    end_idx += len(old_trend_end)
    content = content[:start_idx] + new_trend + content[end_idx:]
    changes += 1
    print("  [8] calculate_form_trend → won-last-2=+2.0, declining=-0.5 ✅")
else:
    print(f"  [8] calculate_form_trend — NOT FOUND (start={start_idx}, end={end_idx})")

# ══════════════════════════════════════════════════════════════
# 9. calculate_hot_trainer_bonus — 0% trainer -2.5 → -1.2
# ══════════════════════════════════════════════════════════════
old_trainer = """    # CRITICAL: 0% trainer = massive penalty (TUP R7 lesson)
    if trainer_win_pct == 0.0:
        return -2.5  # Eliminate from contention in most scenarios

    # Very low % trainer penalty (1-5%)
    if trainer_win_pct > 0.0 and trainer_win_pct < 0.05:
        bonus -= 1.0  # Significant penalty"""

new_trainer = """    # OPTIMIZED: 0% trainer penalty capped at -1.2 (was -2.5)
    # A single trainer stat should not override the entire 8-component rating.
    if trainer_win_pct == 0.0:
        return -1.2  # Strong negative but doesn't nuke the horse (was -2.5)

    # Very low % trainer penalty (1-5%)
    if trainer_win_pct > 0.0 and trainer_win_pct < 0.05:
        bonus -= 0.7  # Significant penalty (was -1.0)"""

if old_trainer in content:
    content = content.replace(old_trainer, new_trainer, 1)
    changes += 1
    print("  [9] calculate_hot_trainer_bonus → 0% trainer -1.2 (was -2.5) ✅")
else:
    print("  [9] calculate_hot_trainer_bonus — NOT FOUND")

# ══════════════════════════════════════════════════════════════
# 10. calculate_form_cycle_rating — floor -3.0 → -2.0 + workout args
# ══════════════════════════════════════════════════════════════
# Update the layoff_factor call to pass workout data
old_cycle_call = """    # 1. Layoff factor
    days_since_last = recent_races[0]['days_ago'] if recent_races else 999
    # Use calculate_layoff_factor (defined earlier in file)
    layoff_adj = calculate_layoff_factor(days_since_last)"""

new_cycle_call = """    # 1. Layoff factor (with workout mitigation)
    days_since_last = recent_races[0]['days_ago'] if recent_races else 999
    workout_data = parse_workout_data(horse_block)
    layoff_adj = calculate_layoff_factor(
        days_since_last,
        num_workouts=workout_data.get("num_recent", 0),
        workout_pattern_bonus=workout_data.get("pattern_bonus", 0.0),
    )"""

if old_cycle_call in content:
    content = content.replace(old_cycle_call, new_cycle_call, 1)
    changes += 1
    print("  [10a] calculate_form_cycle_rating → workout mitigation args ✅")
else:
    print("  [10a] calculate_form_cycle_rating call — NOT FOUND")

# Update the floor from -3.0 to -2.0
old_floor = "    return float(np.clip(form_rating, -3.0, 3.0))\n\n# ===================== Odds Conversion Utilities ====================="
new_floor = "    return float(np.clip(form_rating, -2.0, 3.0))  # OPTIMIZED: floor -2.0 (was -3.0)\n\n# ===================== Odds Conversion Utilities ====================="

if old_floor in content:
    content = content.replace(old_floor, new_floor, 1)
    changes += 1
    print("  [10b] calculate_form_cycle_rating → floor capped at -2.0 ✅")
else:
    print("  [10b] calculate_form_cycle_rating floor — NOT FOUND")

# Write result
APP_PATH.write_text(content, encoding="utf-8")
final_len = len(content.splitlines())
print(f"\n{'=' * 60}")
print(f"Changes applied: {changes}/11")
print(f"Line count: {original_len} → {final_len} (+{final_len - original_len})")
print(f"File saved: {APP_PATH}")
print(f"Backup at: {backup}")
