"""Diagnostic: Run parser on CT R6 PP data and show what it extracts for each horse."""
import sys
sys.path.insert(0, r"c:\Users\C Stephens\Desktop\Horse Racing Picks")

from elite_parser_v2_gold import GoldStandardBRISNETParser

# Read CT R6 PP text
with open(r"c:\Users\C Stephens\Desktop\Horse Racing Picks\_ct_r6_pp.txt", "r", encoding="utf-8") as f:
    pp_text = f.read()

parser = GoldStandardBRISNETParser()

# Step 1: Test chunking
print("=" * 80)
print("STEP 1: CHUNKING")
print("=" * 80)
chunks = parser._split_into_chunks(pp_text, debug=True)
print(f"\nTotal chunks found: {len(chunks)}")
for i, (post, name, style, quirin, block) in enumerate(chunks):
    print(f"\n--- Chunk {i+1}: Post={post}, Name='{name}', Style={style}, Quirin={quirin}")
    print(f"    Block length: {len(block)} chars")
    # Show first 200 and last 200 chars of block
    print(f"    Block start: {repr(block[:200])}")
    print(f"    Block end:   {repr(block[-200:])}")

# Step 2: Parse each horse and show extracted data
print("\n" + "=" * 80)
print("STEP 2: PARSED DATA PER HORSE")
print("=" * 80)

results = parser.parse_full_pp(pp_text, debug=False)
for name, horse in results.items():
    print(f"\n{'─' * 60}")
    print(f"HORSE: {name} (Post {horse.post}, Style {horse.pace_style})")
    print(f"{'─' * 60}")
    print(f"  ML Odds: {horse.ml_odds} (decimal: {horse.ml_odds_decimal}), conf={horse.odds_confidence:.2f}")
    print(f"  Jockey: {horse.jockey} ({horse.jockey_win_pct}%), conf={horse.jockey_confidence:.2f}")
    print(f"  Trainer: {horse.trainer} ({horse.trainer_win_pct}%), conf={horse.trainer_confidence:.2f}")
    print(f"  Speed Figures: {horse.speed_figures}")
    print(f"    avg_top2={horse.avg_top2}, peak={horse.peak_fig}, last={horse.last_fig}, conf={horse.speed_confidence:.2f}")
    print(f"  Form: days_since={horse.days_since_last}, last_date={horse.last_race_date}")
    print(f"    recent_finishes={horse.recent_finishes}, conf={horse.form_confidence:.2f}")
    print(f"  Class: avg_purse={horse.avg_purse}, conf={horse.class_confidence:.2f}")
    print(f"    purses={horse.recent_purses}, types={horse.race_types}")
    print(f"  Pedigree: sire={horse.sire}, dam={horse.dam}, sire_spi={horse.sire_spi}, damsire_spi={horse.damsire_spi}")
    print(f"    sire_mud_pct={horse.sire_mud_pct}, conf={horse.pedigree_confidence:.2f}")
    print(f"  Prime Power: {horse.prime_power} (rank: {horse.prime_power_rank})")
    print(f"  Angles: {len(horse.angles)} angles, conf={horse.angle_confidence:.2f}")
    print(f"  Errors: {horse.errors}")
    
    # Highlight zeros
    has_zeros = (horse.avg_top2 == 0 and horse.peak_fig == 0 and 
                 len(horse.recent_finishes) == 0 and horse.avg_purse == 0)
    if has_zeros:
        print(f"  *** ALL ZEROS - PARSER FAILED FOR THIS HORSE ***")

print("\n" + "=" * 80)  
print("STEP 3: SPEED FIGURE REGEX DEEP DIVE")
print("=" * 80)

import re

# Test speed patterns against individual horse blocks
for post, name, style, quirin, block in chunks:
    if name in ("Look Ahead", "Dream Street Rose", "Cut Him Loose", "Lovely Odds"):
        print(f"\n--- {name} (Post {post}) ---")
        
        # Find race lines (lines with dates)
        for line in block.split('\n'):
            date_match = re.search(r'(\d{2}[A-Za-z]{3}\d{2})', line)
            if date_match:
                print(f"  RACE LINE: {repr(line[:120])}")
                
                # Test primary pattern
                primary = re.compile(
                    r"(?mi)(\d{2}[A-Za-z]{3}\d{2})\s+\w+\s+(?:Clm|Md Sp Wt|Mdn|Alw|OC|Stk|G[123]|Hcp)\s+.*?\s+(\d{2,3})(?:\s+|$)"
                )
                pm = primary.findall(line)
                print(f"    Primary pattern matches: {pm}")
                
                # Test fallback pattern
                fallback = re.compile(r"(?mi)(\d{2}[A-Za-z]{3}\d{2}).*?(\d{2,3})")
                fm = fallback.findall(line)
                print(f"    Fallback pattern matches: {fm}")

print("\n" + "=" * 80)
print("STEP 4: FORM CYCLE DEEP DIVE")  
print("=" * 80)

for post, name, style, quirin, block in chunks:
    if name in ("Look Ahead", "Dream Street Rose", "Cut Him Loose", "Lovely Odds"):
        print(f"\n--- {name} (Post {post}) ---")
        lines = block.split('\n')
        for line in lines:
            date_match = re.search(r'(\d{2}[A-Za-z]{3}\d{2})', line)
            if date_match:
                # Pattern 2 finish extraction
                finish_match = re.search(
                    r'\s(\d{1,2})[ƒ®«ª³©¨°¬²‚±]+\s+\w+\s+[\d.]+\s*$', line
                )
                # Pattern 3
                finish_match3 = re.search(
                    r'\s(\d{1,2})(?:st|nd|rd|th|[ƒ®«ª³©¨°¬²‚±])\s+\w+\s+', line
                )
                print(f"  Date: {date_match.group(1)}")
                print(f"    P2 finish: {finish_match.group(1) if finish_match else 'NO MATCH'}")
                print(f"    P3 finish: {finish_match3.group(1) if finish_match3 else 'NO MATCH'}")
