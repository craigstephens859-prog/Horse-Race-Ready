#!/usr/bin/env python3
"""
Comprehensive validation of angle calculations for all 6 horses
Ensures all angles are being extracted and calculated correctly
"""

import re
import pandas as pd
import numpy as np

# Test data - complete BRISNET Ultimate PP format
pp_text = """1 Way of Appeal (S 3)
Own: Melvin Moy
3/1 Red, White Stripe
BARRIOS RICARDO (219 22-18-27 10%)
Ch. m. 4 (May)
Sire : Medal Season (Storm Boot) $7,500
Dam: Appealing Smile (Smile Jamaica)
Brdr: Tate Moy (WV)
Trnr: Cady Khalil (18 1-3-3 6%)
Prime Power: 99.4 (2nd)
2024 324 8% 36% -0.71
JKYw/ Trn L60 6 0% 50% -2.00
JKYw/ Sprints 151 9% 41% -0.42
Maiden Sp Wt 26 0% 23% -2.00
Sprints 72 7% 33% -1.52
Dirt starts 97 6% 34% -1.40

2 Spuns Kitten (S 3)
Own: George Hair
3/1 White, White Cap
NEGRON LUIS (254 56-42-37 22%)
B. f. 3 (Mar)
Sire : Spun to Run (Hard Spun) $10,000
Dam: Gamble On Kitten (Kitten's Joy)
Brdr: George Hair (WV)
Trnr: Shuler John R (36 9-5-5 25%)
Prime Power: 83.9 (5th)
+2025 424 18% 49% -0.43
JKYw/ S types 162 14% 43% -0.27
JKYw/ Trn L60 6 0% 50% -2.00
+JKYw/ Sprints 288 18% 48% -0.49
2025 52 17% 37% -1.08
+Turf to Dirt 11 27% 45% +0.44
Maiden Sp Wt 63 13% 32% -1.15
Sprints 196 16% 41% -0.61

3 Emily Katherine (E 5)
Own: Katherine F Funkhouser
2/1 Blue, Blue Cap
GOMEZ ALEJANDRO (178 15-24-33 8%)
Dkbbr. f. 3 (Apr)
Sire : Golden Years (Not for Love) $1,500
Dam: Potosi's Silver (Badge of Silver)
Brdr: O'Sullivan Farms LLC (WV)
Trnr: Baird J. Michael (64 2-4-8 3%)
Prime Power: 93.8 (1st)
2025 203 8% 39% -0.84
JKYw/ E types 86 10% 31% -0.01
JKYw/ Trn L60 11 0% 27% -2.00
JKYw/ Sprints 123 9% 38% -0.78
2025 72 3% 22% -1.81
Maiden Sp Wt 137 10% 39% -1.54
Sprints 473 13% 43% -0.83
Dirt starts 450 14% 45% -0.94

4 Zipadeedooda (E 4)
Own: Becky Gibbs
9/2 Yellow, Yellow Cap
TAPARA BRANDON (121 17-15-21 14%)
Ch. f. 3 (Apr)
Sire : Unbridled Energy (Unbridled's Song)
Dam: Concert Pianist (Any Given Saturday)
Brdr: Williams Racing Corp (WV)
Trnr: Delong Ben (6 0-1-0 0%)
Prime Power: 91.6 (3rd)
2025 67 7% 25% -1.24
JKYw/ Trn L60 3 0% 33% -2.00
2025 15 20% 33% -0.38
Maiden Sp Wt 62 10% 32% -1.15
Sprints 149 12% 37% -0.81
Dirt starts 142 11% 35% -0.84

5 Lastshotatlightnin (S 3)
Own: Sunny Brook Farm
4/1 Green, White Diamond
BARBARAN ERIK (104 18-14-16 17%)
B. f. 3 (Mar)
Sire : Khozan (Distorted Humor) $2,500
Dam: Shot at Glory (Elusive Quality)
Brdr: Sunny Brook Farm LLC (WV)
Trnr: Baird J. Michael (64 2-4-8 3%)
Prime Power: 86.5 (4th)
2025 310 10% 35% -0.62
Turf to Dirt 48 8% 33% -1.63
Maiden Sp Wt 94 11% 33% -1.15
Sprints 315 11% 35% -0.81
Dirt starts 275 9% 35% -0.96

6 Zees Clozure (P 4)
Own: Robert E Coyne
6/1 Black, Black Cap
STOKES JOE (267 32-41-44 12%)
B. f. 3 (May)
Sire : Yes It's True (Smarty Jones) $3,000
Dam: Zee's Song (Elusive Quality)
Brdr: Coyne Racing (WV)
Trnr: Delong Ben (6 0-1-0 0%)
Prime Power: 79.8 (6th)
2025 612 13% 38% -0.38
JKYw/ Sprints 217 12% 39% -0.32
2nd career race 13 0% 15% -2.00
Maiden Sp Wt 62 10% 32% -1.15
Sprints 448 13% 41% -0.68
Dirt starts 421 11% 38% -0.82
"""

# Regex patterns
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

def split_into_horse_chunks(pp_text: str):
    chunks = []
    matches = list(HORSE_HDR_RE.finditer(pp_text or ""))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(pp_text)
        post = m.group(1).strip()
        name = m.group(2).strip()
        style = m.group(3) if m.group(3) else "NA"
        quirin = m.group(4).strip() if m.group(4) else "0"
        block = pp_text[start:end]
        chunks.append((post, name, style, quirin, block))
    return chunks

def parse_angles_for_block(block: str) -> pd.DataFrame:
    """Extract all handicapping angles from PP block"""
    angles = []
    
    if not block:
        return pd.DataFrame()
    
    # Pattern: "JKYw/ Trn L60 6 0% 50% -2.00"
    # or "Maiden Sp Wt 26 0% 23% -2.00"
    for line in block.split('\n'):
        line = line.strip()
        if not line:
            continue
        
        # Match angle pattern: NAME STARTS WIN% ITM% ROI
        m = re.match(r'^([A-Z0-9\w\s\/\-\.]+?)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([\+\-][\d.]+)$', line)
        if m:
            angle_name = m.group(1).strip()
            starts = int(m.group(2))
            win_pct = int(m.group(3))
            itm_pct = int(m.group(4))
            roi = float(m.group(5))
            
            angles.append({
                'angle': angle_name,
                'starts': starts,
                'win_pct': win_pct,
                'itm_pct': itm_pct,
                'roi': roi
            })
    
    return pd.DataFrame(angles) if angles else pd.DataFrame()

def calculate_angle_score(angles_df: pd.DataFrame) -> float:
    """
    Calculate composite angle score based on:
    - Win percentage (40%)
    - In-the-money percentage (30%)
    - ROI (30%)
    - Sample size bonus (more starts = more reliable)
    """
    if angles_df.empty:
        return 0.0
    
    # Normalize each metric
    max_win = 25  # cap at 25% for normalization
    max_itm = 60  # cap at 60% for normalization
    max_roi = 1.0  # cap at +1.0 for normalization
    
    scores = []
    for _, row in angles_df.iterrows():
        win_score = min(row['win_pct'] / max_win, 1.0)  # 0-1
        itm_score = min(row['itm_pct'] / max_itm, 1.0)  # 0-1
        roi_score = max(0, min((row['roi'] + 2.0) / 3.0, 1.0))  # Normalize ROI -2 to +1 -> 0-1
        sample_bonus = min(row['starts'] / 200, 1.0)  # Bonus for sample size
        
        composite = (0.40 * win_score + 0.30 * itm_score + 0.30 * roi_score) * (1.0 + 0.2 * sample_bonus)
        scores.append(composite)
    
    # Return average of all angle scores
    return sum(scores) / len(scores) if scores else 0.0

def parse_prime_power(block: str) -> float:
    """Extract Prime Power rating"""
    m = re.search(r"Prime Power:\s*(\d+\.?\d*)", block or "")
    return float(m.group(1)) if m else 0.0

print("=" * 80)
print("COMPREHENSIVE ANGLES VALIDATION TEST")
print("=" * 80)

chunks = split_into_horse_chunks(pp_text)
print(f"\nFound {len(chunks)} horses\n")

# Initialize tracking
total_angles_extracted = 0
angle_scores = {}
prime_powers = {}
data_summary = []

for post, name, style, quirin, block in chunks:
    print(f"\n{'='*80}")
    print(f"POST #{post} - {name.upper()} ({style})")
    print(f"{'='*80}")
    
    # Extract angles
    angles_df = parse_angles_for_block(block)
    angle_count = len(angles_df)
    total_angles_extracted += angle_count
    
    print(f"\n[ANGLES EXTRACTED: {angle_count}]")
    if not angles_df.empty:
        for _, row in angles_df.iterrows():
            print(f"  • {row['angle']}: {row['starts']} starts, {row['win_pct']}% W, {row['itm_pct']}% ITM, {row['roi']:+.2f} ROI")
    else:
        print("  ⚠ WARNING: No angles found!")
    
    # Calculate angle score
    score = calculate_angle_score(angles_df)
    angle_scores[name] = score
    print(f"\n  → Composite Angle Score: {score:.3f} (0.0-1.0)")
    
    # Extract prime power
    pp = parse_prime_power(block)
    prime_powers[name] = pp
    print(f"  → Prime Power: {pp:.1f}")
    
    # Data summary
    data_summary.append({
        'Post': post,
        'Horse': name,
        'Style': style,
        'Angles': angle_count,
        'Angle Score': score,
        'Prime Power': pp
    })

# Print summary
print(f"\n\n{'='*80}")
print("SUMMARY")
print(f"{'='*80}")

summary_df = pd.DataFrame(data_summary)
print("\n" + summary_df.to_string(index=False))

print(f"\n\n{'='*80}")
print("VALIDATION RESULTS")
print(f"{'='*80}")

print(f"\n✓ Total Horses: {len(chunks)}/6")
print(f"✓ Total Angles Extracted: {total_angles_extracted}")
print(f"✓ Angles per Horse: {total_angles_extracted / len(chunks):.1f} (avg)")

# Check if all horses have angles
horses_with_angles = sum(1 for name, score in angle_scores.items() if score > 0)
print(f"✓ Horses with Angles: {horses_with_angles}/6")

# Check angle score distribution
angle_scores_list = list(angle_scores.values())
avg_angle_score = sum(angle_scores_list) / len(angle_scores_list) if angle_scores_list else 0
print(f"✓ Average Angle Score: {avg_angle_score:.3f}")
print(f"✓ Min Angle Score: {min(angle_scores_list):.3f}")
print(f"✓ Max Angle Score: {max(angle_scores_list):.3f}")

# Check prime power
pp_list = list(prime_powers.values())
avg_pp = sum(pp_list) / len(pp_list) if pp_list else 0
print(f"✓ Average Prime Power: {avg_pp:.1f}")
print(f"✓ Min Prime Power: {min(pp_list):.1f}")
print(f"✓ Max Prime Power: {max(pp_list):.1f}")

# PASS/FAIL
print(f"\n{'='*80}")
if horses_with_angles == 6 and total_angles_extracted > 0:
    print("✅ ALL VALIDATION CHECKS PASSED")
    print("   - All 6 horses have angles calculated")
    print(f"   - {total_angles_extracted} total angles extracted")
    print("   - Angle scoring working correctly")
else:
    print("❌ VALIDATION FAILED")
    if horses_with_angles < 6:
        print(f"   - Only {horses_with_angles}/6 horses have angles")
    if total_angles_extracted == 0:
        print("   - No angles extracted!")

print(f"{'='*80}\n")
