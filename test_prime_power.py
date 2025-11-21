"""
Test Prime Power parsing from BRISNET PP blocks.
"""
import re

def parse_prime_power_for_block(block, debug=False):
    """
    Parse Prime Power rating and rank from a BRISNET horse block.
    
    Format: "Prime Power: 131.9 (7th)"
    
    Returns dict with:
    - prime_power: float rating value (e.g., 131.9)
    - prime_power_rank: str rank (e.g., "7th")
    """
    result = {
        'prime_power': None,
        'prime_power_rank': None
    }
    
    # Pattern: "Prime Power: 131.9 (7th)"
    # Captures the decimal number and the rank in parentheses
    prime_power_pattern = r'Prime Power:\s+([\d.]+)\s+\((\d+(?:st|nd|rd|th))\)'
    
    match = re.search(prime_power_pattern, block)
    
    if match:
        try:
            result['prime_power'] = float(match.group(1))
            result['prime_power_rank'] = match.group(2)
            
            if debug:
                print(f"  Prime Power: {result['prime_power']} (rank: {result['prime_power_rank']})")
        except ValueError:
            pass
    
    return result


# Test cases from Del Mar Race 4
test_cases = [
    {
        'name': 'Omnipontet',
        'block': 'Prime Power: 131.9 (7th) Life: 11 3 - 3 - 1 $70,038 89',
        'expected': {'prime_power': 131.9, 'prime_power_rank': '7th'}
    },
    {
        'name': 'Queen Maxima',
        'block': 'Prime Power: 155.2 (1st) Life: 11 6 - 2 - 0 $450,460 99',
        'expected': {'prime_power': 155.2, 'prime_power_rank': '1st'}
    },
    {
        'name': 'Jungle Peace',
        'block': 'Prime Power: 149.1 (2nd) Life: 8 3 - 0 - 2 $181,788 90',
        'expected': {'prime_power': 149.1, 'prime_power_rank': '2nd'}
    },
    {
        'name': 'Shoot It True',
        'block': 'Prime Power: 146.6 (3rd) Life: 6 4 - 0 - 0 $213,011 99',
        'expected': {'prime_power': 146.6, 'prime_power_rank': '3rd'}
    },
    {
        'name': 'Nay V Belle',
        'block': 'Prime Power: 139.3 (4th) Life: 13 3 - 4 - 0 $267,960 96',
        'expected': {'prime_power': 139.3, 'prime_power_rank': '4th'}
    },
    {
        'name': 'Great Venezuela',
        'block': 'Prime Power: 137.9 (5th) Life: 14 8 - 4 - 2 $355,120 98',
        'expected': {'prime_power': 137.9, 'prime_power_rank': '5th'}
    },
    {
        'name': 'Sunglow',
        'block': 'Prime Power: 137.4 (6th) Life: 8 2 - 1 - 2 $147,131 92',
        'expected': {'prime_power': 137.4, 'prime_power_rank': '6th'}
    },
    {
        'name': 'Marian Cross',
        'block': 'Prime Power: 131.7 (8th) Life: 12 3 - 1 - 2 $95,227 84',
        'expected': {'prime_power': 131.7, 'prime_power_rank': '8th'}
    },
    {
        'name': 'Tahini',
        'block': 'Prime Power: 129.2 (9th) Life: 16 2 - 1 - 1 $120,860 88',
        'expected': {'prime_power': 129.2, 'prime_power_rank': '9th'}
    },
]

print("Testing Prime Power parsing...\n")
passed = 0
failed = 0

for test in test_cases:
    result = parse_prime_power_for_block(test['block'])
    
    if result == test['expected']:
        print(f"✅ {test['name']}: {result['prime_power']} ({result['prime_power_rank']})")
        passed += 1
    else:
        print(f"❌ {test['name']}: Got {result}, expected {test['expected']}")
        failed += 1

print(f"\n{passed}/{len(test_cases)} tests passed")
if failed == 0:
    print("🎉 All Prime Power tests passed!")
