#!/usr/bin/env python3
"""Comprehensive parsing accuracy test for all 6 horses"""
import re
import pandas as pd
import numpy as np

# Load the exact parsing functions from app.py
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

def parse_jockey_trainer_for_block(block: str) -> dict:
    """Parse jockey and trainer names from a horse's PP text block"""
    result = {'jockey': '', 'trainer': ''}
    
    if not block:
        return result
    
    # Parse jockey - appears on a line by itself in ALL CAPS before "Trnr:"
    jockey_match = re.search(r'^([A-Z][A-Z\s\'.-]+?)\s*\([\d\s-]+%\)', block, re.MULTILINE)
    if jockey_match:
        jockey_name = jockey_match.group(1).strip()
        jockey_name = ' '.join(jockey_name.split())
        result['jockey'] = jockey_name
    
    # Parse trainer - appears on line starting with "Trnr:"
    trainer_match = re.search(r'Trnr:\s*([A-Za-z][A-Za-z\s,\'.-]+?)\s*\([\d\s-]+%\)', block, re.MULTILINE)
    if trainer_match:
        trainer_name = trainer_match.group(1).strip()
        trainer_name = ' '.join(trainer_name.split())
        result['trainer'] = trainer_name
    
    return result

def parse_running_style_for_block(block: str) -> dict:
    """Parse running style from a horse's PP text block header"""
    result = {'running_style': ''}
    
    if not block:
        return result
    
    # Parse running style from header - appears in parentheses after horse name
    style_match = re.search(r'^\s*\d+\s+[A-Za-z\s=\'-]+\s+\(([A-Z/]+)\s+\d+\)', block, re.MULTILINE)
    if style_match:
        running_style = style_match.group(1).strip()
        result['running_style'] = running_style
    
    return result

ANGLE_LINE_RE = re.compile(
    r'(?mi)^\s*(\d{4}\s+)?(1st\s*time\s*str|Debut\s*MdnSpWt|Maiden\s*Sp\s*Wt|2nd\s*career\s*race|Turf\s*to\s*Dirt|Dirt\s*to\s*Turf|Shipper|Blinkers\s*(?:on|off)|(?:\d+(?:-\d+)?)\s*days?Away|JKYw/\s*Sprints|JKYw/\s*Trn\s*L(?:30|45|60)\b|JKYw/\s*[EPS]|JKYw/\s*NA\s*types)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$'
)

def parse_angles_for_block(block: str) -> pd.DataFrame:
    rows = []
    matches = list(ANGLE_LINE_RE.finditer(block or ""))
    for m in matches:
        _yr, cat, starts, win, itm, roi = m.groups()
        rows.append({"Category": re.sub(r"\s+", " ", cat.strip()),
                     "Starts": int(starts), "Win%": float(win),
                     "ITM%": float(itm), "ROI": float(roi)})
    return pd.DataFrame(rows)

pp_text = """1 Way of Appeal (S 3)
Own: Trinity Elite Llc
7/2 Red, Red Cap
BARRIOS RICARDO (222 17-32-40 8%)
B. f. 4 (Mar)
Sire : Aaron's Way (Yes It's True) $2,500
Dam: Little Miss Kisses (B L's Appeal)
Brdr: Bybee Road Farm (WV)
Trnr: Cady Khalil (9 0-2-2 0%)
Prime Power: 89.7 (2nd)
+2025 222 8% 40% -0.76
JKYw/ S types 102 15% 35% +0.35
JKYw/ Trn L60 6 0% 50% -2.00
JKYw/ Sprints 151 9% 41% -0.42
2025 9 0% 44% -2.00
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
Prime Power: 85.9 (4th)
+2025 224 19% 48% +0.20
JKYw/ E types 120 14% 41% -0.66
JKYw/ Trn L60 3 0% 33% -2.00
+JKYw/ Sprints 140 21% 52% +0.93
2025 59 10% 29% -0.42
Maiden Sp Wt 62 10% 32% -1.15
Sprints 327 13% 41% -0.61
Dirt starts 322 14% 44% -0.61

5 Lastshotatlightnin (S 3)
Own: J Michael Baird
10/1 Black, Black Cap
BARBARAN ERIK (142 29-26-21 20%)
Ch. f. 4 (Mar)
Sire : Cal Nation (Distorted Humor) $2,500
Dam: Arbitrageur (Bernardini)
Brdr: J Michael Baird (WV)
Trnr: Baird J. Michael (64 2-4-8 3%)
Prime Power: 78.6 (6th)
+2025 632 24% 58% -0.23
JKYw/ S types 219 17% 50% -0.66
+JKYw/ Trn L60 3 33% 33% +1.07
+JKYw/ Sprints 441 24% 59% -0.09
2025 72 3% 22% -1.81
MdnClm to Mdn 15 7% 40% -1.76
+Rte to Sprint 18 22% 39% -0.03
Turf to Dirt 48 8% 33% -1.63

6 Zees Clozure (P 4)
Own: Becky Gibbs
4/1 Lime, Lime Cap
STOKES JOE (90 14-9-15 16%)
Ch. f. 3 (May)
Sire : Unbridled Energy (Unbridled's Song)
Dam: Bin Elusive (Elusive Quality)
Brdr: Williams Racing Corp (WV)
Trnr: Delong Ben (6 0-1-0 0%)
Prime Power: 86.3 (3rd)
2025 316 11% 39% -0.47
JKYw/ P types 203 12% 34% +0.44
JKYw/ Sprints 217 12% 39% -0.32
2025 59 10% 29% -0.42
2nd career race 13 0% 15% -2.00
Maiden Sp Wt 62 10% 32% -1.15
Sprints 327 13% 41% -0.61"""

print("="*80)
print("COMPREHENSIVE PARSING ACCURACY TEST - ALL 6 HORSES")
print("="*80)

chunks = split_into_horse_chunks(pp_text)
print("\nFound {} horse chunks\n".format(len(chunks)))

for post, name, style, quirin, block in chunks:
    print("\n" + "="*80)
    print("POST #{} - {}".format(post, name.upper()))
    print("="*80)
    
    # Test jockey/trainer parsing
    print("\n[JOCKEY & TRAINER]")
    jt = parse_jockey_trainer_for_block(block)
    if not jt['jockey']:
        print("  FAILED - Jockey NOT found")
    else:
        print("  OK - Jockey: '{}'".format(jt['jockey']))
    if not jt['trainer']:
        print("  FAILED - Trainer NOT found")
    else:
        print("  OK - Trainer: '{}'".format(jt['trainer']))
    
    # Test running style parsing
    print("\n[RUNNING STYLE]")
    if not style or style == "NA":
        print("  FAILED - Style NOT found")
    else:
        print("  OK - Style: '{}'".format(style))
    
    # Test angle parsing
    print("\n[ANGLES]")
    angles_df = parse_angles_for_block(block)
    if angles_df.empty:
        print("  FAILED - No angles extracted!")
    else:
        print("  OK - Found {} angles:".format(len(angles_df)))
        for _, row in angles_df.iterrows():
            print("    - {}: {} starts, {}% win, {}% ITM, {:.2f} ROI".format(
                row['Category'], row['Starts'], row['Win%'], row['ITM%'], row['ROI']))

print("\n" + "="*80)
print("PARSING SUMMARY")
print("="*80)

# Count accuracy
jockeys_found = 0
trainers_found = 0
styles_found = 0
angles_found = 0

for post, name, style, quirin, block in chunks:
    jt = parse_jockey_trainer_for_block(block)
    angles_df = parse_angles_for_block(block)
    
    if jt['jockey']: jockeys_found += 1
    if jt['trainer']: trainers_found += 1
    if style and style != "NA": styles_found += 1
    if not angles_df.empty: angles_found += 1

print("\nJockeys: {}/6 found {}".format(jockeys_found, "(OK)" if jockeys_found == 6 else "(FAILED)"))
print("Trainers: {}/6 found {}".format(trainers_found, "(OK)" if trainers_found == 6 else "(FAILED)"))
print("Styles: {}/6 found {}".format(styles_found, "(OK)" if styles_found == 6 else "(FAILED)"))
print("Angles: {}/6 found {}".format(angles_found, "(OK)" if angles_found == 6 else "(FAILED)"))

