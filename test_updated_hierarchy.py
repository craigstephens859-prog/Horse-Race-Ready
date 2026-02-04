"""
Test updated race type hierarchy with industry-standard levels
"""
from race_class_parser import CLASS_MAP, LEVEL_MAP, parse_race_conditions, get_hierarchy_level

print('='*70)
print('UPDATED RACE TYPE HIERARCHY - INDUSTRY STANDARD (1-7 Scale)')
print('='*70)

# Test key abbreviations from user's list
test_cases = [
    ('MSW 50000', 'Maiden Special Weight', 1),
    ('MCL 15000', 'Maiden Claiming', 1),
    ('MOC 30000', 'Maiden Optional Claiming', 1),
    ('CLM 10000', 'Claiming', 2),
    ('CLH 12000', 'Claiming Handicap', 2),
    ('CST 25000', 'Claiming Stakes', 2),
    ('STR 40000', 'Starter Allowance', 3),
    ('SHP 35000', 'Starter Handicap', 3),
    ('SOC 50000', 'Starter Optional Claiming', 5),
    ('ALW 60000', 'Allowance', 4),
    ('AOC 80000', 'Allowance Optional Claiming', 5),
    ('OCL 75000', 'Optional Claiming', 5),
    ('OCH 90000', 'Optional Claiming Handicap', 6),
    ('HCP 100000', 'Handicap', 6),
    ('STK 150000', 'Stakes', 7),
    ('G3 200000', 'Grade 3 Stakes', 7),
    ('G2 500000', 'Grade 2 Stakes', 7),
    ('G1 1000000', 'Grade 1 Stakes', 7),
    ('SST 120000', 'Starter Stakes', 7),
    ('DBY 750000', 'Derby', 7),
    ('FTR 300000', 'Futurity', 7),
    ('TRL 50000', 'Trial', 4),
    ('INV 250000', 'Invitational', 7),
    ('MAT', 'Match Race', 0),
    ('TR', 'Training Race', 0),
]

print('\n✓ Testing Industry-Standard Abbreviations:\n')
all_passed = True
for conditions, expected_class, expected_level in test_cases:
    result = parse_race_conditions(conditions)
    hierarchy = get_hierarchy_level(result['class_type'])
    
    class_match = result['class_type'] == expected_class
    level_match = hierarchy['base_level'] == expected_level
    
    status = '✓' if (class_match and level_match) else '✗'
    
    if not (class_match and level_match):
        all_passed = False
    
    # Show grade boost for graded stakes
    boost_info = ''
    if hierarchy['grade_boost'] > 0:
        boost_info = f' +{hierarchy["grade_boost"]} = Final {hierarchy["final_level"]}'
    
    print(f'{status} {conditions:20s} → Lv{expected_level} {expected_class:40s}{boost_info}')

print('\n' + '='*70)
print('LEVEL DISTRIBUTION')
print('='*70)

level_counts = {}
for class_type, level in LEVEL_MAP.items():
    level_counts[level] = level_counts.get(level, 0) + 1

for level in sorted(level_counts.keys(), reverse=True):
    if level == 7:
        print(f'Level 7 (Elite Stakes): {level_counts[level]:2d} types - Graded, Listed, High-Purse')
    elif level == 6:
        print(f'Level 6 (Handicap):     {level_counts[level]:2d} types - Weight-assigned races')
    elif level == 5:
        print(f'Level 5 (Optional):     {level_counts[level]:2d} types - Optional claiming/High allowance')
    elif level == 4:
        print(f'Level 4 (Allowance):    {level_counts[level]:2d} types - Conditional allowance')
    elif level == 3:
        print(f'Level 3 (Starter):      {level_counts[level]:2d} types - Starter allowance')
    elif level == 2:
        print(f'Level 2 (Claiming):     {level_counts[level]:2d} types - Claiming races')
    elif level == 1:
        print(f'Level 1 (Maiden):       {level_counts[level]:2d} types - First-time winners')
    elif level == 0:
        print(f'Level 0 (Special):      {level_counts[level]:2d} types - Match/Training/Unknown')

print('\n' + '='*70)
print('STATISTICS')
print('='*70)
print(f'Total Abbreviations: {len(CLASS_MAP)}')
print(f'Total Class Types: {len(LEVEL_MAP)}')
print(f'Base Level Range: 0-7')
print(f'Final Level Range: 0-10 (with grade boosts: G1 +3, G2 +2, G3 +1)')
print('='*70)

if all_passed:
    print('\n✅ ALL TESTS PASSED - Industry-standard hierarchy implemented!')
else:
    print('\n⚠️  SOME TESTS FAILED')
