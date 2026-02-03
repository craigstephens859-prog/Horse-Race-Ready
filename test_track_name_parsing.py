"""
Test track name parsing from BRISNET PP text
Validates that all major tracks are correctly identified
"""

import re

# Copy TRACK_ALIASES from app.py
TRACK_ALIASES = {
    "Del Mar": ["del mar", "dmr"],
    "Keeneland": ["keeneland", "kee"],
    "Churchill Downs": ["churchill downs", "cd", "churchill"],
    "Kentucky Downs": ["kentucky downs", "kd"],
    "Saratoga": ["saratoga", "sar"],
    "Santa Anita": ["santa anita", "sa", "santa anita park"],
    "Mountaineer": ["mountaineer", "mnr"],
    "Charles Town": ["charlestown", "charles town", "ct"],
    "Gulfstream": ["gulfstream", "gulfstream park", "gp"],
    "Tampa Bay Downs": ["tampa", "tampa bay downs", "tam"],
    "Turf Paradise": ["turf paradise", "tup", "turf"],
    "Belmont Park": ["belmont", "belmont park", "bel", "aqueduct at belmont", "belmont at aqueduct", "big a"],
    "Horseshoe Indianapolis": ["horseshoe indianapolis", "indiana grand", "ind", "indy"],
    "Penn National": ["penn national", "pen"],
    "Presque Isle Downs": ["presque isle", "presque isle downs", "pid"],
    "Woodbine": ["woodbine", "wo"],
    "Evangeline Downs": ["evangeline", "evangeline downs", "evd"],
    "Fairmount Park": ["fairmount park", "fanduel fairmount", "cah", "collinsville"],
    "Finger Lakes": ["finger lakes", "fl"]
}

_CANON_BY_TOKEN = {}
for canon, toks in TRACK_ALIASES.items():
    for t in toks:
        _CANON_BY_TOKEN[t] = canon

def parse_track_name_from_pp(pp_text: str) -> str:
    """
    Parse track name from BRISNET PP text.
    Checks both:
    1. Header/title text for full track names
    2. Race history lines for track abbreviations (e.g., 29Dec25Tup, 08Nov25Aquª)
    """
    text = (pp_text or "")[:2000].lower()  # Increased to capture race history
    
    # First, check for track abbreviations in race history date lines
    # Pattern: DDMmmYYTrk (e.g., 29Dec25Tup, 08Nov25Aquª)
    date_line_pattern = r'\d{2}[A-Za-z]{3}\d{2}([A-Za-z]{2,4})'
    for match in re.finditer(date_line_pattern, text):
        track_code = match.group(1).lower()
        if track_code in _CANON_BY_TOKEN:
            return _CANON_BY_TOKEN[track_code]
    
    # Second, check for full track names in header
    for token, canon in _CANON_BY_TOKEN.items():
        if re.search(rf'\b{re.escape(token)}\b', text):
            return canon
    
    # Third, check for multi-word track names
    for canon, toks in TRACK_ALIASES.items():
        for t in toks:
            t_words = [w for w in t.split() if len(w) > 2]
            if t_words and all(re.search(rf'\b{re.escape(w)}\b', text) for w in t_words):
                return canon
    
    return ""

# Test cases with sample PP text headers
TEST_CASES = [
    {
        "name": "Turf Paradise (full name)",
        "header": "Ultimate PP's w/ QuickPlay Comments Turf Paradise Claiming $4,000 6½f Dirt 3&up",
        "expected": "Turf Paradise"
    },
    {
        "name": "Turf Paradise (abbreviation Tup)",
        "header": "29Dec25Tup 6½ ft :22¨ :45ª1:11 1:18¨ ¨¨© ™AzJuvFilly 30k",
        "expected": "Turf Paradise"
    },
    {
        "name": "Gulfstream Park (full name)",
        "header": "Ultimate PP's w/ QuickPlay Comments Gulfstream Park PWCInvit-G1 1„ Mile 4&up",
        "expected": "Gulfstream"
    },
    {
        "name": "Gulfstream Park (abbreviation GP)",
        "header": "29Mar25GP® 1ˆ ft :22« :45«1:10© 1:41« ¡ ¨¨® Ghstzapr-G3",
        "expected": "Gulfstream"
    },
    {
        "name": "Keeneland (full name)",
        "header": "Ultimate PP's w/ QuickPlay Comments Keeneland Dirt 1 1/16m",
        "expected": "Keeneland"
    },
    {
        "name": "Keeneland (abbreviation Kee)",
        "header": "24Apr25Kee¬ à 1½ fm :49« 1:14«2:04¨ 2:28«",
        "expected": "Keeneland"
    },
    {
        "name": "Saratoga (full name)",
        "header": "Ultimate PP's w/ QuickPlay Comments Saratoga Stakes Race",
        "expected": "Saratoga"
    },
    {
        "name": "Saratoga (abbreviation Sar)",
        "header": "02Aug25Sar¨¨ 1„ ft :47 1:11ª1:36© 1:48« ¡ ¨©§ Whitney-G1",
        "expected": "Saratoga"
    },
    {
        "name": "Churchill Downs (full name)",
        "header": "Ultimate PP's w/ QuickPlay Comments Churchill Downs Kentucky Derby",
        "expected": "Churchill Downs"
    },
    {
        "name": "Churchill Downs (abbreviation CD)",
        "header": "30Nov24CD® 1m ft :23 :45©1:09« 1:34« ¨¨ª OC100k/n1x-N",
        "expected": "Churchill Downs"
    }
]

def run_tests():
    print("="*80)
    print("TRACK NAME PARSING VALIDATION")
    print("="*80)
    
    passed = 0
    failed = 0
    
    for test in TEST_CASES:
        result = parse_track_name_from_pp(test["header"])
        status = "✓ PASS" if result == test["expected"] else "✗ FAIL"
        
        if result == test["expected"]:
            passed += 1
        else:
            failed += 1
        
        print(f"\n{status} - {test['name']}")
        print(f"  Expected: {test['expected']}")
        print(f"  Got:      {result or '(empty string)'}")
        if result != test["expected"]:
            print(f"  Header:   {test['header'][:80]}...")
    
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"Total tests: {len(TEST_CASES)}")
    print(f"Passed: {passed}")
    print(f"Failed: {failed}")
    
    if failed == 0:
        print("\n✓✓✓ ALL TESTS PASSED ✓✓✓")
    else:
        print(f"\n✗✗✗ {failed} TESTS FAILED ✗✗✗")
    
    return failed == 0

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
