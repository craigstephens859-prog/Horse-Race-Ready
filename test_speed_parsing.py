"""Test speed figure parsing with actual BRISNET PP text"""
import re
import numpy as np

# Copy regex patterns from app.py
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

SPEED_FIG_RE = re.compile(
    # CRITICAL FIX: Match the actual BRISNET format with E1 E2/ LP +calls SPEED_FIG
    # Example: "88 81/ 77 +4 +1 70"  where 70 is the speed figure
    r"(?mi)"
    r"\s+(\d{2,3})"                    # E1 pace figure (88)
    r"\s+(\d{2,3})\s*/\s*(\d{2,3})"   # E2/LP figures (81/ 77)
    r"\s+[+-]\d+"                      # Call position (+4)
    r"\s+[+-]\d+"                      # Call position (+1)
    r"\s+(\d{2,3})"                    # SPEED FIGURE (70) - THIS IS WHAT WE WANT!
    r"(?:\s|$)"                        # End with space or end of line
)

# Sample PP text (just one horse)
pp_text = """3 Rascally Rabbit (E/P 5)
Own: Kevin Eikleberry Reunion Racing Stables
4/1 Black, White "r" On Red Ball, Black Band On Red Sleeves
ALVARADO FRANK T (101 12-21-11 12%)
B. f. 3 ARZOCT 2024 $6k
Sire : Lotsa Mischief (Into Mischief)
Dam: Hard Buns (Rock Hard Ten)
Brdr: Fleming Thoroughbred Farm LLC & WilliamMatthews Family Tr (AZ)
Trnr: Eikleberry Kevin (50 11-7-4 22%)
Prime Power: 120.6 (2nd) Life: 4 3 - 0 - 1 $69,985 77
2025 4 3 - 0 - 1 $69,985 77
2024 0 0 - 0 - 0 $0
TUP 2 1 - 0 - 1 $28,345 74
Fst (98) 4 3 - 0 - 1 $69,985 77
Off (97) 0 0 - 0 - 0 $0
Dis (100) 1 1 - 0 - 0 $25,345 74
Trf (96) 0 0 - 0 - 0 $0
AW 0 0 - 0 - 0 $0
Blnkr On
L 122
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment
29Dec25Tup 6½ ft :22¨ :45ª1:11 1:18¨ ¨¨© ™AzJuvFilly 30k¨¨ª 88 81/ 77 +4 +1 70 2 7 7ª‚ 5¬ 5 3ª AlvaradoFT¨©¨ *1.40 WskHg¨RnYrMth©RsclyRbt¨ 3-2w;closed for 3rd 7
09Dec25Tup 6f ft :22© :45ª :58¨ 1:11 ¨§ ™'AZBrdrFutB 50k¨¨¨ 92 88/ 77 +2 -1 74 3 4 1 1 1© 1¬ƒ AlvaradoFT¨©§ *0.40 RsclyRbt¬ƒWtrnFl¨ƒ
SrtAltrntv©‚ Shook free;ridden out 8
22Sep25Prm° 5½ ft :22 :46¨ :58« 1:05© ¨¨§ OC100k/n1x ¨¨© 93 83/ 73 +5 +1 64 2 2 1 1¨ 1ª 1³ GonzalezE¨¨° B *0.05
RsclyRbt³PrctclJ«CrkdsCnB¨ Pace;widened;held late 6
01Aug25Prm® 5f ft :22¨ :45« :58 ¨§¯ ™Mdn 35k ¨¨ª 90/ 78 77 1 3 3¨ 3 1¬ 1¨§ GonzalezE¨¨° B *1.00
RsclyRbt¨§
SmmrsWthSnya©Vrn² Responded willingly 6"""

print("=" * 80)
print("TESTING HORSE HEADER REGEX")
print("=" * 80)

matches = list(HORSE_HDR_RE.finditer(pp_text))
print(f"\nFound {len(matches)} horse header matches\n")

for i, m in enumerate(matches):
    print(f"Match {i+1}:")
    print(f"  Post: {m.group(1)}")
    print(f"  Name: {m.group(2)}")
    print(f"  Style: {m.group(3)}")
    print(f"  Quirin: {m.group(4)}")
    print()

print("=" * 80)
print("TESTING SPEED FIGURE REGEX")
print("=" * 80)

# Find speed figures
speed_matches = list(SPEED_FIG_RE.finditer(pp_text))
print(f"\nFound {speed_matches} speed figure matches\n")

for i, m in enumerate(speed_matches):
    print(f"Speed Figure {i+1}:")
    print(f"  E1: {m.group(1)}")
    print(f"  E2/LP: {m.group(2)}")
    print(f"  Speed Fig: {m.group(3)}")
    print(f"  Full match: {m.group(0)[:100]}")
    print()

# Now test the actual parsing function
def parse_speed_figures_for_block(block):
    """Parses a horse's PP text block and extracts all main speed figures."""
    figs = []
    if not block:
        return figs
    
    block_str = str(block) if not isinstance(block, str) else block
    
    for m in SPEED_FIG_RE.finditer(block_str):
        try:
            # The speed figure is the FOURTH capture group (after E1, E2, LP)
            fig_val = int(m.group(4))
            # Basic sanity check for a realistic speed figure
            if 40 < fig_val < 130:
                figs.append(fig_val)
                print(f"  ✓ Extracted figure: {fig_val}")
        except (ValueError, AttributeError, IndexError) as e:
            print(f"  ✗ Failed to extract: {e}")
            pass
    
    return figs[:10]

print("=" * 80)
print("TESTING parse_speed_figures_for_block()")
print("=" * 80)
figures = parse_speed_figures_for_block(pp_text)
print(f"\nExtracted figures: {figures}")
print(f"Best figure: {max(figures) if figures else 'NONE'}")
print(f"Avg top 2: {np.mean(sorted(figures, reverse=True)[:2]) if len(figures) >= 2 else 'NONE'}")
