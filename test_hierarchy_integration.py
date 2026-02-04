#!/usr/bin/env python3
"""Test industry-standard hierarchy integration"""

from race_class_parser import parse_and_calculate_class

# Test cases with various race types (must include header info for parser)
test_cases = [
    (
        "Ultimate PP's w/ QuickPlay Comments | Churchill Downs | Grade1 1000000 | 1 1/4 Mile | 3yo | Race 11",
        'Lv7 base + Grade Boost +3 = Lv10', 'G1'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Gulfstream | MSW 50000 | 1 Mile | 3yo | Race 5",
        'Lv1 Maiden Special Weight', 'MSW'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Santa Anita | Clm 10000 | 6f | 3yo+ | Race 2",
        'Lv2 Claiming', 'Clm'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Oaklawn | Str 40000 | 1 1/16 Mile | 3yo+ | Race 7",
        'Lv3 Starter Allowance', 'Str'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Keeneland | Alw 60000 | 1 Mile | 3yo | Race 8",
        'Lv4 Allowance', 'Alw'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Belmont | Aoc 80000 | 1 1/8 Mile | 3yo+ | Race 9",
        'Lv5 Allowance Optional Claiming', 'Aoc'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Saratoga | Hcp 100000 | 7f | 3yo+ | Race 10",
        'Lv6 Handicap', 'Hcp'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Del Mar | Stk 150000 | 1 Mile | 3yo | Race 6",
        'Lv7 Stakes', 'Stk'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Santa Anita | Grade3 200000 | 1 Mile | 3yo+ | Race 8",
        'Lv7 base + Grade Boost +1 = Lv8', 'G3'
    ),
    (
        "Ultimate PP's w/ QuickPlay Comments | Churchill Downs | Grade2 500000 | 1 1/16 Mile | 3yo | Race 10",
        'Lv7 base + Grade Boost +2 = Lv9', 'G2'
    ),
]

print('=' * 80)
print('INDUSTRY-STANDARD HIERARCHY INTEGRATION TEST')
print('=' * 80)
print()

all_passed = True
for pp_text, expected, race_abbr in test_cases:
    result = parse_and_calculate_class(pp_text)
    
    # Get summary data
    summary = result.get('summary', {})
    level = summary.get('hierarchy_level', 0)
    class_type = summary.get('class_type', 'Unknown')
    weight = summary.get('class_weight', 0)
    track = summary.get('track', 'Unknown')
    
    print(f'{race_abbr:5} ({track:20}) → Level {level:.1f} ({class_type})')
    print(f'  Expected: {expected}')
    print(f'  Weight: {weight:.2f}')
    
    # Validate expected levels
    if 'G1' in race_abbr and level != 10.0:
        print('  ❌ FAIL - Expected level 10 for G1')
        all_passed = False
    elif 'G2' in race_abbr and level != 9.0:
        print('  ❌ FAIL - Expected level 9 for G2')
        all_passed = False
    elif 'G3' in race_abbr and level != 8.0:
        print('  ❌ FAIL - Expected level 8 for G3')
        all_passed = False
    elif 'MSW' in race_abbr and level != 1.0:
        print('  ❌ FAIL - Expected level 1 for MSW')
        all_passed = False
    elif 'Clm' in race_abbr and level != 2.0:
        print('  ❌ FAIL - Expected level 2 for Clm')
        all_passed = False
    elif 'Str' in race_abbr and level != 3.0:
        print('  ❌ FAIL - Expected level 3 for Str')
        all_passed = False
    elif 'Alw' in race_abbr and level != 4.0:
        print('  ❌ FAIL - Expected level 4 for Alw')
        all_passed = False
    elif 'Aoc' in race_abbr and level != 5.0:
        print('  ❌ FAIL - Expected level 5 for Aoc')
        all_passed = False
    elif 'Hcp' in race_abbr and level != 6.0:
        print('  ❌ FAIL - Expected level 6 for Hcp')
        all_passed = False
    elif 'Stk' in race_abbr and level != 7.0:
        print('  ❌ FAIL - Expected level 7 for Stk')
        all_passed = False
    else:
        print('  ✓ PASS')
    print()

print('=' * 80)
if all_passed:
    print('✅ ALL TESTS PASSED - Industry-standard hierarchy working correctly!')
else:
    print('❌ SOME TESTS FAILED - Check results above')
print('=' * 80)
