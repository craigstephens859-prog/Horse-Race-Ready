"""
Test comprehensive race header parsing from BRISNET PP text
Format: Ultimate PP's w/ QuickPlay Comments | Track Name | Race Type Purse | Distance | Age/Sex | Date | Race #
"""

import re
from typing import Dict, Any

def parse_brisnet_header(pp_text: str) -> Dict[str, Any]:
    """
    Parse complete BRISNET race header.
    
    Expected format from screenshot:
    Ultimate PP's w/ QuickPlay Comments | Turf Paradise | ©Hcp 50000 | 6 Furlongs | 3yo Fillies | Monday, February 02, 2026 | Race 8
    
    Returns dict with:
    - track_name: str
    - race_number: int
    - race_type: str (e.g., "©Hcp", "Clm", "Alw", "Stakes")
    - purse_amount: int
    - distance: str (e.g., "6 Furlongs", "1 Mile")
    - age_restriction: str (e.g., "3yo", "4&up")
    - sex_restriction: str (e.g., "Fillies", "Colts", "F&M")
    - race_date: str
    - day_of_week: str
    """
    if not pp_text:
        return {}
    
    result = {
        'track_name': '',
        'race_number': 0,
        'race_type': '',
        'purse_amount': 0,
        'distance': '',
        'age_restriction': '',
        'sex_restriction': '',
        'race_date': '',
        'day_of_week': '',
        'raw_header': ''
    }
    
    # Get first line (header)
    lines = pp_text.strip().split('\n')
    if not lines:
        return result
    
    header_line = lines[0].strip()
    result['raw_header'] = header_line
    
    # Split by pipe delimiter
    parts = [p.strip() for p in header_line.split('|')]
    
    if len(parts) < 2:
        # Try alternate format without pipes
        # Look for pattern: Track Name  RaceType Purse  Distance  Age/Sex  Date  Race N
        return _parse_no_pipe_format(pp_text, result)
    
    # Parse each section
    for i, part in enumerate(parts):
        part_lower = part.lower()
        
        # Skip "Ultimate PP's w/ QuickPlay Comments"
        if 'ultimate' in part_lower or 'quickplay' in part_lower:
            continue
        
        # Track name (typically 2nd part, or first non-Ultimate part)
        if not result['track_name'] and i > 0:
            # Check if this looks like a track name (not a race type, not a distance)
            if not re.search(r'\d+\s*(?:furlong|mile|yard)', part_lower) and \
               not re.search(r'(clm|alw|stk|hcp|mdn|msw|aoc)', part_lower) and \
               not re.search(r'race\s+\d+', part_lower) and \
               not re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)', part_lower):
                result['track_name'] = part
                continue
        
        # Race type + purse (e.g., "©Hcp 50000", "Clm 4000")
        race_type_match = re.search(r'([©¨§]?[A-Za-z]+)\s+(\d+)', part)
        if race_type_match and not result['race_type']:
            result['race_type'] = race_type_match.group(1)
            result['purse_amount'] = int(race_type_match.group(2))
            continue
        
        # Distance (e.g., "6 Furlongs", "1 Mile", "1„ Mile")
        if re.search(r'\d+\s*(?:furlong|mile|yard|f\b)', part_lower):
            result['distance'] = part
            continue
        
        # Age/Sex restrictions (e.g., "3yo Fillies", "4&up", "F&M")
        if re.search(r'(?:\d+yo|f&m|fillies|mares|colts|geldings|4&up)', part_lower):
            # Extract age and sex separately
            age_match = re.search(r'(\d+yo|4&up|3&up)', part_lower)
            if age_match:
                result['age_restriction'] = age_match.group(1)
            
            sex_match = re.search(r'(fillies?|mares?|colts?|geldings?|f&m)', part_lower, re.IGNORECASE)
            if sex_match:
                result['sex_restriction'] = sex_match.group(1).title()
            continue
        
        # Date (e.g., "Monday, February 02, 2026")
        date_match = re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[,\s]+(.+\d{4})', part_lower)
        if date_match:
            result['day_of_week'] = date_match.group(1).title()
            result['race_date'] = date_match.group(2).strip()
            continue
        
        # Race number (e.g., "Race 8")
        race_num_match = re.search(r'race\s+(\d+)', part_lower)
        if race_num_match:
            result['race_number'] = int(race_num_match.group(1))
            continue
    
    return result

def _parse_no_pipe_format(pp_text: str, result: Dict) -> Dict:
    """Parse header without pipe delimiters (space-separated)"""
    header = pp_text[:200]
    
    # Track name
    track_match = re.search(r'(?:Ultimate PP.+?)?([A-Z][a-z]+(?:\s+[A-Z][a-z]+)*)', header)
    if track_match:
        result['track_name'] = track_match.group(1)
    
    # Race number
    race_match = re.search(r'Race\s+(\d+)', header, re.IGNORECASE)
    if race_match:
        result['race_number'] = int(race_match.group(1))
    
    # Distance
    dist_match = re.search(r'(\d+(?:\s*½)?\s*(?:Furlongs?|Miles?|f|m))', header, re.IGNORECASE)
    if dist_match:
        result['distance'] = dist_match.group(1)
    
    # Purse
    purse_match = re.search(r'(\d+)(?:\s+|$)', header)
    if purse_match:
        result['purse_amount'] = int(purse_match.group(1))
    
    return result

# Test with the exact header from screenshot
TEST_HEADER = "Ultimate PP's w/ QuickPlay Comments Turf Paradise ©Hcp 50000 6 Furlongs 3yo Fillies Monday, February 02, 2026 Race 8"

# Test with pipe-delimited version
TEST_HEADER_PIPES = "Ultimate PP's w/ QuickPlay Comments | Turf Paradise | ©Hcp 50000 | 6 Furlongs | 3yo Fillies | Monday, February 02, 2026 | Race 8"

def run_tests():
    print("="*80)
    print("BRISNET HEADER PARSING TEST")
    print("="*80)
    
    # Test 1: Pipe-delimited (most common)
    print("\nTEST 1: Pipe-delimited header")
    print("-" * 80)
    print(f"Input: {TEST_HEADER_PIPES}")
    print()
    
    result1 = parse_brisnet_header(TEST_HEADER_PIPES)
    
    print("Extracted Fields:")
    print(f"  Track Name:        {result1['track_name']}")
    print(f"  Race Number:       {result1['race_number']}")
    print(f"  Race Type:         {result1['race_type']}")
    print(f"  Purse Amount:      ${result1['purse_amount']:,}")
    print(f"  Distance:          {result1['distance']}")
    print(f"  Age Restriction:   {result1['age_restriction']}")
    print(f"  Sex Restriction:   {result1['sex_restriction']}")
    print(f"  Day of Week:       {result1['day_of_week']}")
    print(f"  Race Date:         {result1['race_date']}")
    
    # Validate
    print("\nValidation:")
    errors = []
    
    if result1['track_name'] != "Turf Paradise":
        errors.append(f"❌ Track name: expected 'Turf Paradise', got '{result1['track_name']}'")
    else:
        print("  ✓ Track name correct: Turf Paradise")
    
    if result1['race_number'] != 8:
        errors.append(f"❌ Race number: expected 8, got {result1['race_number']}")
    else:
        print("  ✓ Race number correct: 8")
    
    if result1['race_type'] != "©Hcp":
        errors.append(f"❌ Race type: expected '©Hcp', got '{result1['race_type']}'")
    else:
        print("  ✓ Race type correct: ©Hcp (Handicap)")
    
    if result1['purse_amount'] != 50000:
        errors.append(f"❌ Purse: expected 50000, got {result1['purse_amount']}")
    else:
        print("  ✓ Purse correct: $50,000")
    
    if result1['distance'] != "6 Furlongs":
        errors.append(f"❌ Distance: expected '6 Furlongs', got '{result1['distance']}'")
    else:
        print("  ✓ Distance correct: 6 Furlongs")
    
    if result1['age_restriction'] != "3yo":
        errors.append(f"❌ Age: expected '3yo', got '{result1['age_restriction']}'")
    else:
        print("  ✓ Age restriction correct: 3yo")
    
    if result1['sex_restriction'] != "Fillies":
        errors.append(f"❌ Sex: expected 'Fillies', got '{result1['sex_restriction']}'")
    else:
        print("  ✓ Sex restriction correct: Fillies")
    
    if result1['day_of_week'] != "Monday":
        errors.append(f"❌ Day: expected 'Monday', got '{result1['day_of_week']}'")
    else:
        print("  ✓ Day of week correct: Monday")
    
    if "february 02, 2026" not in result1['race_date'].lower():
        errors.append(f"❌ Date: expected 'February 02, 2026', got '{result1['race_date']}'")
    else:
        print("  ✓ Race date correct: February 02, 2026")
    
    print("\n" + "="*80)
    if not errors:
        print("✅ ALL FIELDS EXTRACTED CORRECTLY!")
        print("="*80)
        return True
    else:
        print("❌ ERRORS FOUND:")
        for err in errors:
            print(f"  {err}")
        print("="*80)
        return False

if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)
