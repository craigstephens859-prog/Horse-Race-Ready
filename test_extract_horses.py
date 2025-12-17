#!/usr/bin/env python3
"""Diagnose the actual PP text parsing issue"""
import re
import pandas as pd

HORSE_HDR_RE = re.compile(
    r"""(?mi)^\s*
    (\d+)              # post/program
    \s+([A-Za-z0-9'.\-\s&]+?)   # horse name
    \s*\(\s*
    (E\/P|EP|E|P|S|NA)      # style
    (?:\s+(\d+))?           # optional quirin
    \s*\)\s*$              #
    """, re.VERBOSE
)

def _normalize_style(tok: str) -> str:
    t = (tok or "").upper().strip()
    return "E/P" if t in ("EP", "E/P") else t

def calculate_style_strength(style: str, quirin: float) -> str:
    s = (style or "NA").upper()
    try:
        q = float(quirin)
    except Exception:
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

def extract_horses_and_styles(pp_text: str) -> pd.DataFrame:
    rows = []
    for m in HORSE_HDR_RE.finditer(pp_text or ""):
        post = m.group(1).strip()
        name = m.group(2).strip()
        style = _normalize_style(m.group(3))
        qpts = m.group(4)
        quirin = int(qpts) if qpts else float('nan')
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

# Your actual PP text from the user message (simplified version)
pp_text_sample = """
Race 2 Ultimate PP's Mountaineer ™'Mdn 16.5k 5½ Furlongs 3&up, F & M Wednesday, August 20, 2025 Race 2 E1 E2 / Late SPD
86 /78 64 58

1 Way of Appeal (S 3)
Own: Someone

2 Spuns Kitten (S 3)
Own: Someone

3 Emily Katherine (E 5)
Own: Someone

4 Zipadeedooda (E 4)
Own: Becky Gibbs
9/2 Yellow, Yellow Cap
TAPARA BRANDON (121 17-15-21 14%)

5 Lastshotatlightnin (S 3)
Own: J Michael Baird
10/1 Black, Black Cap

6 Zees Clozure (P 4)
Own: Becky Gibbs
4/1 Lime, Lime Cap
"""

print("Testing extract_horses_and_styles:")
print("=" * 70)

df = extract_horses_and_styles(pp_text_sample)
print(f"\nExtracted {len(df)} horses from PP text:\n")
print(df[["Post", "Horse", "DetectedStyle", "Quirin", "AutoStrength"]].to_string(index=False))

print("\n" + "=" * 70)
print("DIAGNOSIS:")
if len(df) < 6:
    print(f"❌ PROBLEM: Only {len(df)} horses extracted, expected 6!")
    print("\nMissing horses:")
    for i in range(1, 7):
        found = any(df["Post"].astype(str) == str(i))
        status = "✓" if found else "✗"
        print(f"  {status} Post #{i}")
else:
    print(f"✓ All 6 horses extracted successfully!")
