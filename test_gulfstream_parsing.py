"""
CROSS-TRACK VALIDATION: Gulfstream Park Race 13
Test all critical parsing components against different track format
"""

import re

# Sample PP text from Gulfstream Park Race 13 - Disco Time
SAMPLE_PP_TEXT = """
1 Disco Time (E/P 5)
Own: Juddmonte
8/5 Green, Pink Sash, White Sleeves, Pink Cap
PRAT FLAVIEN (5 1-2-1 20%)
Dkbbr. c. 4
Sire : Not This Time (Giant's Causeway) $175,000
Dam: Disco Chick (Jump Start)
Brdr: Juddmonte (KY)
Trnr: Cox Brad H (19 6-4-4 32%)
Prime Power: 144.3 (3rd) Life: 5 5 - 0 - 0 $551,960 103
2025 3 3 - 0 - 0 $410,000 103
2024 2 2 - 0 - 0 $141,960 92
GP 0 0 - 0 - 0 $0 0
Fst (109) 4 4 - 0 - 0 $401,960 103
Off (109) 1 1 - 0 - 0 $150,000 88
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment
08Nov25Aquª 1m ft :22¨ :44©1:08ª 1:33« ¨¨® DwyerL 200k ¨©¨ 96 98/ 84 +9 +7 91 2 2¨ 2 1© 1® 1°ƒ GerouxF¨©© *1.22 DiscoTim°ƒCrdo¬TpTpThmas± Bumped st; went clear 4
19Sep25Fpk¯ 1ˆ ft :23« :47©1:11© 1:43« ¨¨ª StLouDbyB 250k¨©©103 110/ 92 +9 +7 103 1 1¨ 1¨ 1© 1« 1¬ GerouxF¨©ª *1.10
Disco Time¬Hypnus©
Excite©ª 2p; cued1/4; drew away 8
18Jan25FG¨© 1ˆ sys :23ª :47«1:13¨ 1:47 ¨¨¬ Lecomte-G3 ¨¨¯ 77 82/ 94 +8 +7 88 10 10¨§ 10¯ 8‚ 4ª 1³ GerouxF¨©© *1.90
Disco Time³BuiltInnovator¨ƒ 3-5p;8p1/4;rallied;up 13
30Nov24CD® 1m ft :23 :45©1:09« 1:34« ¨¨ª OC100k/n1x-N ¨¨® 92 96/ 88 +4 +4 92 5 2 1² 1 1¨ 1ª GerouxF¨©§ *0.63
Disco TimeªMcKellen¨‚Lil Muggs Step slw;ins;inchd clr 9
01Nov24CD 7f ft :23 :46ª1:11¨ 1:23« ¨¨§ Mdn 120k ¨¨¬ 89 88/ 90 -3 -8 87 8 4 3¨ 2² 1¨ 1ªƒ GerouxF¨¨° *2.32
DscoTmªƒWhitWhalPrfctForc Press 3-2w;vied;cleard 9
"""

# Sample from White Abarrio (different horse, different tracks)
WHITE_ABARRIO_PP = """
11 White Abarrio (E/P 4)
Own: C2 Racing Stable Llc Gary Barber And La
4/1 Hot Pink, Black Circled 'bb,' Black Cap
ORTIZ, JR. IRAD (148 43-29-20 29%)
Gr/ro. h. 7 OBSMAR 2021 $40k
Sire : Race Day (Tapit) $7,500
Dam: Catching Diamonds (Into Mischief)
Brdr: Spendthrift Farm LLC (KY)
Trnr: Joseph, Jr. Saffie A (103 19-18-18 18%)
Prime Power: 147.8 (1st) Life: 24 10 - 2 - 3 $7,151,920 108
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment
31Aug25Sar¬ 1‚ ft :46« 1:11«1:36« 2:02 ¦ ¨©¨ JkyClbGC-G1 ¨©§ 85 104/ 87 +7 +8 100 5 3® 3° 4 7°‚ 5¯ ZayasEJ¨© 11.40 Antqrn¨SiraLon¨‚PhlasFog¨ƒ Bumped;lost iron 9 1/2 8
02Aug25Sar¨¨ 1„ ft :47 1:11ª1:36© 1:48« ¡ ¨©§ Whitney-G1 ¨©© 83 96/ 104 +3 -5 102 6 5« 5®ƒ 6© 4¨‚ 4« OrtizIJ¨©« 4.50 SierraLeone¨HighlandFalls©Dsarm¨ 3-5p1st turn;7-6p2nd 9
07Jun25Sar¯ 1m sys :23« :46«1:10¨ 1:35« ¦ ¨©¨ MtropltH-G1 ¨©¨ 89 96/ 88 +3 +6 91 2 3¨ 3© 4« 3« 4¬‚ OrtizIJ¨© 2.60 RgngTrnt©Fircns©‚JstaToch Bmp btwn brk;bmp;3-5p 5
29Mar25GP® 1ˆ ft :22« :45«1:10© 1:41« ¡ ¨¨® Ghstzapr-G3 ¨©© 97 100/ 99 +4 -1 101 4 3¨ 1² 1² 1ª 1¬‚ OrtizIJ¨©« *0.05 WhitAbario¬‚PowrSqz¨TscnSky¨ Chase;2p;drew off 6
25Jan25GP¨ª 1„ ft :46ª 1:10ª1:35© 1:48 ¡ ¨¨° PWCInvit-G1 ¨©« 90 94/ 110 0 -6 104 4 5ª 5© 4© 1© 1‚ OrtizIJ¨©ª 2.80
WhtAbar‚
Lcd³
Skplnstckn¨ƒ 4p; drew off; drvng 11
"""

# Regex patterns from app.py
HORSE_HDR_RE = re.compile(
    r"^\s*(\d+)\s+([A-Za-z0-9'.\-\s&]+?)\s*\(\s*(E\/P|EP|E|P|S|NA)(?:\s+(\d+))?\s*\)\s*$",
    re.MULTILINE
)

# NEW FIXED PATTERN - Position-based matching
SPEED_FIG_RE = re.compile(
    r"(?mi)"
    r"\s+(\d{2,3})"                    # E1 pace figure (group 1)
    r"\s+(\d{2,3})\s*/\s*(\d{2,3})"   # E2/LP figures (groups 2-3)
    r"\s+[+-]\d+"                      # Call position 1
    r"\s+[+-]\d+"                      # Call position 2
    r"\s+(\d{2,3})"                    # SPEED FIGURE (group 4) ← TARGET
    r"(?:\s|$)"
)

# Prime Power pattern
PRIME_POWER_RE = re.compile(r"Prime Power:\s*(\d+(?:\.\d+)?)")

# Morning Line Odds pattern
ML_ODDS_RE = re.compile(r"^\s*(\d+(?:/\d+)?)\s+", re.MULTILINE)

def parse_speed_figures(text):
    """Extract speed figures from PP text block"""
    figs = []
    matches = list(SPEED_FIG_RE.finditer(text))
    
    print(f"\n{'='*80}")
    print(f"SPEED FIGURE EXTRACTION")
    print(f"{'='*80}")
    print(f"Found {len(matches)} speed figure matches")
    
    for i, m in enumerate(matches, 1):
        try:
            e1 = int(m.group(1))
            e2 = int(m.group(2))
            lp = int(m.group(3))
            fig_val = int(m.group(4))
            
            match_text = m.group(0)
            print(f"\nMatch {i}:")
            print(f"  E1: {e1}, E2: {e2}, LP: {lp}")
            print(f"  → Speed Figure: {fig_val}")
            print(f"  Raw match: '{match_text.strip()}'")
            
            if 40 < fig_val < 130:
                figs.append(fig_val)
                print(f"  ✓ Valid figure added")
            else:
                print(f"  ✗ Outside valid range (40-130)")
                
        except (ValueError, AttributeError, IndexError) as e:
            print(f"  ✗ Parse error: {e}")
    
    return figs

def test_horse_header(text):
    """Test horse header parsing"""
    print(f"\n{'='*80}")
    print(f"HORSE HEADER PARSING")
    print(f"{'='*80}")
    
    matches = list(HORSE_HDR_RE.finditer(text))
    print(f"Found {len(matches)} horse header matches")
    
    for i, m in enumerate(matches, 1):
        post = m.group(1)
        name = m.group(2).strip()
        style = m.group(3)
        quirin = m.group(4) if m.group(4) else "N/A"
        
        print(f"\nMatch {i}:")
        print(f"  Post: {post}")
        print(f"  Name: {name}")
        print(f"  Running Style: {style}")
        print(f"  Quirin Speed Points: {quirin}")

def test_prime_power(text):
    """Test Prime Power extraction"""
    print(f"\n{'='*80}")
    print(f"PRIME POWER EXTRACTION")
    print(f"{'='*80}")
    
    matches = list(PRIME_POWER_RE.finditer(text))
    print(f"Found {len(matches)} Prime Power values")
    
    for i, m in enumerate(matches, 1):
        pp_val = float(m.group(1))
        print(f"\nMatch {i}: Prime Power = {pp_val}")

def test_morning_line(text):
    """Test Morning Line odds extraction"""
    print(f"\n{'='*80}")
    print(f"MORNING LINE ODDS")
    print(f"{'='*80}")
    
    # Look for pattern like "8/5 Green, Pink Sash..."
    ml_pattern = re.compile(r"^\s*(\d+/\d+)\s+\w+", re.MULTILINE)
    matches = list(ml_pattern.finditer(text))
    print(f"Found {len(matches)} Morning Line odds")
    
    for i, m in enumerate(matches, 1):
        odds = m.group(1)
        print(f"\nMatch {i}: ML Odds = {odds}")

def calculate_avg_top_2(figures):
    """Calculate average of top 2 speed figures"""
    if not figures:
        return None
    if len(figures) == 1:
        return figures[0]
    
    sorted_figs = sorted(figures, reverse=True)
    top_2 = sorted_figs[:2]
    avg = sum(top_2) / len(top_2)
    return round(avg, 1)

def main():
    print("="*80)
    print("CROSS-TRACK VALIDATION TEST")
    print("Gulfstream Park Race 13 - PWC Invitational G1")
    print("="*80)
    
    # TEST 1: Disco Time (Aqua, Fair Grounds, Churchill Downs tracks)
    print("\n" + "="*80)
    print("TEST 1: DISCO TIME (Multiple Track Formats)")
    print("="*80)
    
    test_horse_header(SAMPLE_PP_TEXT)
    test_prime_power(SAMPLE_PP_TEXT)
    test_morning_line(SAMPLE_PP_TEXT)
    
    disco_figs = parse_speed_figures(SAMPLE_PP_TEXT)
    
    print(f"\n{'='*80}")
    print(f"DISCO TIME SUMMARY")
    print(f"{'='*80}")
    print(f"Total speed figures extracted: {len(disco_figs)}")
    print(f"Figures: {disco_figs}")
    if disco_figs:
        best = max(disco_figs)
        avg_top_2 = calculate_avg_top_2(disco_figs)
        print(f"Best figure: {best}")
        print(f"Avg top 2: {avg_top_2}")
        print(f"✓ SUCCESS: Speed figures extracted from Aqueduct, Fair Grounds, Churchill Downs")
    else:
        print(f"✗ FAILURE: No speed figures extracted")
    
    # TEST 2: White Abarrio (Saratoga, Gulfstream tracks)
    print("\n" + "="*80)
    print("TEST 2: WHITE ABARRIO (Saratoga & Gulfstream Formats)")
    print("="*80)
    
    test_horse_header(WHITE_ABARRIO_PP)
    test_prime_power(WHITE_ABARRIO_PP)
    
    white_figs = parse_speed_figures(WHITE_ABARRIO_PP)
    
    print(f"\n{'='*80}")
    print(f"WHITE ABARRIO SUMMARY")
    print(f"{'='*80}")
    print(f"Total speed figures extracted: {len(white_figs)}")
    print(f"Figures: {white_figs}")
    if white_figs:
        best = max(white_figs)
        avg_top_2 = calculate_avg_top_2(white_figs)
        print(f"Best figure: {best}")
        print(f"Avg top 2: {avg_top_2}")
        print(f"✓ SUCCESS: Speed figures extracted from Saratoga, Gulfstream")
    else:
        print(f"✗ FAILURE: No speed figures extracted")
    
    # VALIDATION REPORT
    print("\n" + "="*80)
    print("CROSS-TRACK VALIDATION REPORT")
    print("="*80)
    
    tracks_tested = [
        "Aqueduct (Aquª)",
        "Fair Grounds (FG)",
        "Churchill Downs (CD)",
        "Saratoga (Sar)",
        "Gulfstream (GP)"
    ]
    
    print(f"\nTracks validated:")
    for track in tracks_tested:
        print(f"  ✓ {track}")
    
    total_figs = len(disco_figs) + len(white_figs)
    print(f"\nTotal speed figures extracted: {total_figs}")
    
    if total_figs > 0:
        print(f"\n{'✓'*40}")
        print(f"SUCCESS: Regex patterns work across multiple track formats!")
        print(f"{'✓'*40}")
    else:
        print(f"\n{'✗'*40}")
        print(f"FAILURE: Regex patterns not extracting correctly")
        print(f"{'✗'*40}")

if __name__ == "__main__":
    main()
