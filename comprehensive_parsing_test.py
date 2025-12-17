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
        block = pp_text[start:end]
        chunks.append((post, name, block))
    return chunks

def parse_jockey_trainer_for_block(block: str, debug: bool = False) -> dict:
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
        if debug:
            print(f"  ✓ Jockey found: '{jockey_name}'")
    
    # Parse trainer - appears on line starting with "Trnr:"
    trainer_match = re.search(r'Trnr:\s*([A-Za-z][A-Za-z\s,\'.-]+?)\s*\([\d\s-]+%\)', block, re.MULTILINE)
    if trainer_match:
        trainer_name = trainer_match.group(1).strip()
        trainer_name = ' '.join(trainer_name.split())
        result['trainer'] = trainer_name
        if debug:
            print(f"  ✓ Trainer found: '{trainer_name}'")
    
    return result

def parse_running_style_for_block(block: str, debug: bool = False) -> dict:
    """Parse running style from a horse's PP text block header"""
    result = {'running_style': ''}
    
    if not block:
        return result
    
    # Parse running style from header - appears in parentheses after horse name
    style_match = re.search(r'^\s*\d+\s+[A-Za-z\s=\'-]+\s+\(([A-Z/]+)\s+\d+\)', block, re.MULTILINE)
    if style_match:
        running_style = style_match.group(1).strip()
        result['running_style'] = running_style
        if debug:
            print(f"  ✓ Running Style found: '{running_style}'")
    
    return result

ANGLE_LINE_RE = re.compile(
    r'(?mi)^\s*(\d{4}\s+)?(1st\s*time\s*str|Debut\s*MdnSpWt|Maiden\s*Sp\s*Wt|2nd\s*career\s*race|Turf\s*to\s*Dirt|Dirt\s*to\s*Turf|Shipper|Blinkers\s*(?:on|off)|(?:\d+(?:-\d+)?)\s*days?Away|JKYw/\s*Sprints|JKYw/\s*Trn\s*L(?:30|45|60)\b|JKYw/\s*[EPS]|JKYw/\s*NA\s*types)\s+(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$'
)

def parse_angles_for_block(block: str, debug: bool = False) -> pd.DataFrame:
    rows = []
    matches = list(ANGLE_LINE_RE.finditer(block or ""))
    for m in matches:
        _yr, cat, starts, win, itm, roi = m.groups()
        rows.append({"Category": re.sub(r"\s+", " ", cat.strip()),
                     "Starts": int(starts), "Win%": float(win),
                     "ITM%": float(itm), "ROI": float(roi)})
    if debug and matches:
        print(f"  ✓ Found {len(matches)} angle lines")
    elif debug:
        print(f"  ✗ No angle lines found")
    return pd.DataFrame(rows)

# PP text loaded directly
pp_text = r"""Ultimate PP's Mountaineer ™'Mdn 16.5k 5½ Furlongs 3&up, F & M Wednesday, August 20, 2025 Race 2
2 2nd Half Daily Double $2 Exacta $1 Box $1 Trifecta $.50 Box $1 Superfecta
$.20 / Box $2 Pick 3 (races 2-4) $1 Wheel E1 E2/LATE SPEED
PARS: 86 78/ 64 58
5½ Furlongs. ™'Mdn 16.5k Purse $16,500 FOR ACCREDITED WEST
VIRGINIA-BRED MAIDENS, FILLIES AND MARES THREE YEARS OLD AND UPWARD.
Three Year Olds, 116 lbs., Older, 123 lbs. (Preference To Horses That Have Not Started
For Less Than $15,000)
Post Time: ( 7:25)/ 6:25/ 5:25/ 4:25
1 Way of Appeal (S 3)
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
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS
21Jly25Mnrª 5f ft :23 :48 1:01ª ¦ ¨§¨ ™'Mdn 17k ¨§« 86/ 63 58 3 1 1 1² 1© 2 BarriosR¨©ª Lb 20.40

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
Sprints 327 13% 41% -0.61
"""

print("=" * 80)
print("COMPREHENSIVE PARSING ACCURACY TEST - ALL 6 HORSES")
print("=" * 80)

chunks = split_into_horse_chunks(pp_text)
print(f"\n✓ Found {len(chunks)} horse chunks\n")

for post, name, block in chunks:
    print(f"\n{'='*80}")
    print(f"POST #{post} - {name.upper()}")
    print(f"{'='*80}")
    
    # Test jockey/trainer parsing
    print("\n[JOCKEY & TRAINER]")
    jt = parse_jockey_trainer_for_block(block, debug=True)
    if not jt['jockey']:
        print(f"  ✗ Jockey NOT found")
    if not jt['trainer']:
        print(f"  ✗ Trainer NOT found")
    print(f"  Result: Jockey='{jt['jockey']}' | Trainer='{jt['trainer']}'")
    
    # Test running style parsing
    print("\n[RUNNING STYLE]")
    rs = parse_running_style_for_block(block, debug=True)
    if not rs['running_style']:
        print(f"  ✗ Style NOT found")
    print(f"  Result: '{rs['running_style']}'")
    
    # Test angle parsing
    print("\n[ANGLES]")
    angles_df = parse_angles_for_block(block, debug=True)
    if angles_df.empty:
        print(f"  ✗ No angles extracted!")
    else:
        print(f"  Angles found:")
        for _, row in angles_df.iterrows():
            print(f"    • {row['Category']}: {row['Starts']} starts, {row['Win%']}% win, {row['ITM%']}% ITM, {row['ROI']:.2f} ROI")

print(f"\n{'='*80}")
print("PARSING SUMMARY")
print(f"{'='*80}")

# Count accuracy
jockeys_found = 0
trainers_found = 0
styles_found = 0
angles_found = 0

for post, name, block in chunks:
    jt = parse_jockey_trainer_for_block(block)
    rs = parse_running_style_for_block(block)
    angles_df = parse_angles_for_block(block)
    
    if jt['jockey']: jockeys_found += 1
    if jt['trainer']: trainers_found += 1
    if rs['running_style']: styles_found += 1
    if not angles_df.empty: angles_found += 1

print(f"\nJockeys: {jockeys_found}/6 found ({'✓ OK' if jockeys_found == 6 else '✗ FAILED'})")
print(f"Trainers: {trainers_found}/6 found ({'✓ OK' if trainers_found == 6 else '✗ FAILED'})")
print(f"Styles: {styles_found}/6 found ({'✓ OK' if styles_found == 6 else '✗ FAILED'})")
print(f"Angles: {angles_found}/6 found ({'✓ OK' if angles_found == 6 else '✗ FAILED'})")
