"""
Test script for QuickPlay comments parsing from BRISNET PP sample
"""
import re

def parse_quickplay_comments_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses QuickPlay handicapping comments from a horse's PP text block.
    
    BRISNET Format:
    - Positive: "ñ 21% trainer: NonGraded Stk"
    - Negative: "× Has not raced for more than 3 months"
    
    Returns dict with keys: 'positive_comments' (list), 'negative_comments' (list)
    """
    result = {
        'positive_comments': [],
        'negative_comments': []
    }
    
    if not block:
        return result
    
    # Parse positive comments - lines starting with ñ
    positive_matches = re.findall(r'^ñ\s*(.+)$', block, re.MULTILINE)
    result['positive_comments'] = [comment.strip() for comment in positive_matches]
    
    # Parse negative comments - lines starting with ×
    negative_matches = re.findall(r'^×\s*(.+)$', block, re.MULTILINE)
    result['negative_comments'] = [comment.strip() for comment in negative_matches]
    
    if debug:
        if result['positive_comments']:
            print(f"  Positive comments ({len(result['positive_comments'])}):")
            for comment in result['positive_comments']:
                print(f"    ñ {comment}")
        if result['negative_comments']:
            print(f"  Negative comments ({len(result['negative_comments'])}):")
            for comment in result['negative_comments']:
                print(f"    × {comment}")
    
    return result


# Test data from the sample PP
sample_blocks = {
    "Omnipontet": """Sire Stats: AWD 7.3 13%Mud 571MudSts 13%Turf 12%1stT 1.08spi
Dam'sSire: AWD 6.3 15%Mud 1622MudSts 14%Turf 12%1stT 1.14spi
Dam'sStats: Unraced 1trfW 2str 2w 2sw 2.20dpi
ñ 21% trainer: NonGraded Stk 
ñ Sharp 4F workout (Oct-25) 
× Has not raced for more than 3 months 
× Speed Figs rank poorly vs others
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment""",
    
    "Nay V Belle": """Sire Stats: AWD 6.6 18%Mud 1544MudSts 13%Turf 12%1stT 1.18spi
Dam'sSire: AWD 7.7 16%Mud 1532MudSts 13%Turf 12%1stT 2.44spi
Dam'sStats: Unraced 2trfW 2str 2w 1sw 3.43dpi
SoldAt: OBSAPR 2023 $200.0k (3/17) SireAvg: $79.4k StudFee: $8,500
ñ Highest last race speed rating 
ñ Won last race (DMR 08/29 5f Turf fm fOC100000b)
ñ Hot Jockey in last 7 days (11 3-4-0) 
× Moves up in class from last start 
× Has not raced for more than 2 months
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment""",
    
    "Queen Maxima": """Sire Stats: AWD 6.0 16%Mud 199MudSts 11%Turf 10%1stT 1.04spi
Dam'sSire: AWD 6.0 14%Mud 192MudSts 11%Turf 10%1stT 0.81spi
Dam'sStats: Winner 1trfW 1str 1w 1sw 7.00dpi
SoldAt: OBSOPN 2023 $40.0k (13/30) SireAvg: $51.1k StudFee: $5,000
ñ Beaten only 1.90 lengths in last start 
ñ Best Turf Speed is fastest among today's starters
ñ Hot Jockey in last 7 days (17 5-4-2) 
ñ Sharp 4F workout (Oct-26) 
× Has not raced for more than 2 months
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment""",
    
    "Sunglow": """Sire Stats: AWD 7.0 13%Mud 101MudSts 12%Turf 12%1stT 1.70spi
Dam'sSire: AWD 10.5 4%Mud 94MudSts 15%Turf 13%1stT 5.65spi
Dam'sStats: Stksplaced 5trfW 8str 6w 0sw 0.97dpi
SoldAt: GOFOR 2023 $274.9k (25/81) SireAvg: $216.0k StudFee: $125,000
ñ Won last race (SA 09/28 6f Turf fm fOC50000n1x) 
ñ Early speed running style helps chances
ñ Sharp 5F workout (Oct-24) 
× Moves up in class from last start
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment""",

    "Jungle Peace": """Sire Stats: AWD 6.1 29%Mud 52MudSts 10%Turf 8%1stT 0.67spi
Dam'sSire: AWD 8.9 4%Mud 23MudSts 11%Turf 6%1stT 0.58spi
Dam'sStats: Unplaced 1trfW 3str 1w 1sw 2.52dpi
SoldAt: GOFOR2 2023 $14.7k (30/52) SireAvg: $25.4k StudFee: $8,000
ñ Drops in class today 
ñ 21% trainer: NonGraded Stk 
ñ Early speed running style helps chances
ñ May improve at the shorter distance 
× Has not raced in 55 days
DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST 1C 2C STR FIN JOCKEY ODDS Top Finishers Comment"""
}

# Expected results
expected = {
    "Omnipontet": {
        "positive": ["21% trainer: NonGraded Stk", "Sharp 4F workout (Oct-25)"],
        "negative": ["Has not raced for more than 3 months", "Speed Figs rank poorly vs others"]
    },
    "Nay V Belle": {
        "positive": ["Highest last race speed rating", "Won last race (DMR 08/29 5f Turf fm fOC100000b)", "Hot Jockey in last 7 days (11 3-4-0)"],
        "negative": ["Moves up in class from last start", "Has not raced for more than 2 months"]
    },
    "Queen Maxima": {
        "positive": ["Beaten only 1.90 lengths in last start", "Best Turf Speed is fastest among today's starters", "Hot Jockey in last 7 days (17 5-4-2)", "Sharp 4F workout (Oct-26)"],
        "negative": ["Has not raced for more than 2 months"]
    },
    "Sunglow": {
        "positive": ["Won last race (SA 09/28 6f Turf fm fOC50000n1x)", "Early speed running style helps chances", "Sharp 5F workout (Oct-24)"],
        "negative": ["Moves up in class from last start"]
    },
    "Jungle Peace": {
        "positive": ["Drops in class today", "21% trainer: NonGraded Stk", "Early speed running style helps chances", "May improve at the shorter distance"],
        "negative": ["Has not raced in 55 days"]
    }
}

print("="*60)
print("Testing QuickPlay Comments Parsing")
print("="*60)

all_passed = True
for horse, block in sample_blocks.items():
    print(f"\n🐎 Testing: {horse}")
    print("-"*60)
    
    result = parse_quickplay_comments_for_block(block, debug=True)
    expected_result = expected[horse]
    
    positive_match = result['positive_comments'] == expected_result['positive']
    negative_match = result['negative_comments'] == expected_result['negative']
    
    print(f"  Positive: {'✅' if positive_match else '❌'}")
    if not positive_match:
        print(f"    Expected: {expected_result['positive']}")
        print(f"    Got: {result['positive_comments']}")
    
    print(f"  Negative: {'✅' if negative_match else '❌'}")
    if not negative_match:
        print(f"    Expected: {expected_result['negative']}")
        print(f"    Got: {result['negative_comments']}")
    
    if not (positive_match and negative_match):
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("✅ ALL TESTS PASSED!")
    print(f"Successfully parsed {len(sample_blocks)} horses")
else:
    print("❌ SOME TESTS FAILED")
print("="*60)
