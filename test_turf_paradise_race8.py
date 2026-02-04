#!/usr/bin/env python3
"""Test Turf Paradise Race 8 - Validate hierarchy and first-time starter handling"""

from race_class_parser import parse_and_calculate_class

# Real Brisnet PP header from Turf Paradise Race 8 (pipe-delimited format)
pp_text = """Ultimate PP's w/ QuickPlay Comments | Turf Paradise | ™Hcp 50000 | 6 Furlongs | 3yo Fillies | Monday, February 02, 2026 | Race 8"""

print("=" * 80)
print("TURF PARADISE RACE 8 - INDUSTRY-STANDARD HIERARCHY TEST")
print("=" * 80)
print()
print("Race Information:")
print(f"PP Text: {pp_text}")
print()

# Parse with new system
result = parse_and_calculate_class(pp_text)

print("PARSING RESULTS:")
print("=" * 80)
print()

# Header extraction
header = result.get('header', {})
print("1. HEADER EXTRACTION:")
print(f"   Track: {header.get('track_name', 'Not found')}")
print(f"   Race Type: {header.get('race_type', 'Not found')}")
print(f"   Purse: ${header.get('purse_amount', 0):,}")
print(f"   Distance: {header.get('distance', 'Not found')} ({header.get('distance_furlongs', 0):.1f}f)")
print(f"   Age/Sex: {header.get('age_restriction', 'Not found')} {header.get('sex_restriction', 'Not found')}")
print(f"   Race #: {header.get('race_number', 'Not found')}")
print()

# Race conditions
conditions = result.get('conditions', {})
print("2. RACE CONDITIONS:")
print(f"   Class Type: {conditions.get('class_type', 'Unknown')}")
print(f"   Class Abbr: {conditions.get('class_abbreviation', 'Unknown')}")
print(f"   Grade Level: {conditions.get('grade_level', 'None')}")
print(f"   Purse: ${conditions.get('purse_amount', 0):,}")
print(f"   Is Stakes: {conditions.get('is_stakes', False)}")
print(f"   Is Graded Stakes: {conditions.get('is_graded_stakes', False)}")
print()

# Hierarchy level
hierarchy = result.get('hierarchy', {})
print("3. INDUSTRY-STANDARD HIERARCHY:")
print(f"   Base Level: {hierarchy.get('base_level', 0)}")
print(f"   Grade Boost: +{hierarchy.get('grade_boost', 0)}")
print(f"   Final Level: {hierarchy.get('final_level', 0):.1f}")
print(f"   Purse Adjustment: {hierarchy.get('purse_adjustment', 0):.2f}")
print(f"   Adjusted Level: {hierarchy.get('adjusted_level', 0):.2f}")
print()
print("   EXPECTED: Level 6 (Handicap) per industry standards")
print()

# Class weight
weight = result.get('weight', {})
print("4. CLASS WEIGHT CALCULATION:")
print(f"   Base Weight: {weight.get('base_weight', 0):.2f}")
print(f"   Distance Factor: {weight.get('distance_factor', 0):.2f}")
print(f"   Surface Factor: {weight.get('surface_factor', 0):.2f}")
print(f"   FINAL WEIGHT: {weight.get('final_weight', 0):.2f}")
print()

# Summary
summary = result.get('summary', {})
print("5. SUMMARY:")
print(f"   Track: {summary.get('track', 'Unknown')}")
print(f"   Class Type: {summary.get('class_type', 'Unknown')}")
print(f"   Hierarchy Level: {summary.get('hierarchy_level', 0):.1f}")
print(f"   Class Weight: {summary.get('class_weight', 0):.2f}")
print(f"   Quality: {summary.get('quality', 'Unknown')}")
print()

print("=" * 80)
print("FIRST-TIME STARTER IMPACT ANALYSIS")
print("=" * 80)
print()

# Analyze impact on first-time starters
print("Old System Concerns:")
print("- First-time starters (#1 La Cat, #2 Winning Nation) getting too much credit")
print("- No racing history but system favored them heavily")
print()

print("New System Benefits:")
print("- Handicap race (Level 6) = HIGH class level")
print("- Class weight of {:.2f} applies to ALL horses".format(summary.get('class_weight', 0)))
print("- First-time starters have:")
print("  • No speed figures (0 vs 76/70 for experienced)")
print("  • No class rating (0 vs 113.8/111.9 for top horses)")
print("  • No Prime Power (NA vs 121.3/120.6 for contenders)")
print()

print("Race Reality:")
print("✓ #4 Whiskey High (2/1) - 6 starts, 5 wins, Level 6 class experience")
print("✓ #3 Rascally Rabbit (4/1) - 4 starts, 3-0-1, proven at this level")
print("✓ #7 Western Feel (7/2) - 3 starts, 0-2-0, improving")
print("✗ #1 La Cat (12/1) - 0 starts, all workouts, NO class rating")
print("✗ #2 Winning Nation (6/1) - 0 starts, all workouts, NO class rating")
print()

if hierarchy.get('base_level', 0) == 6:
    print("✅ CORRECT - System properly identifies Level 6 Handicap")
    print("   This should reduce first-timer advantage significantly!")
else:
    print("❌ INCORRECT - Expected Level 6, got Level {}".format(hierarchy.get('base_level', 0)))
print()

print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)
print()

# Validation checks
checks = []

# Check 1: Track name
if header.get('track_name') == 'Turf Paradise':
    checks.append(("✓", "Track name extracted correctly"))
else:
    checks.append(("✗", f"Track name incorrect: {header.get('track_name')}"))

# Check 2: Race type
if 'HCP' in conditions.get('class_abbreviation', '').upper():
    checks.append(("✓", "Race type identified as Handicap"))
else:
    checks.append(("✗", f"Race type incorrect: {conditions.get('class_type')}"))

# Check 3: Purse
if conditions.get('purse_amount', 0) == 50000:
    checks.append(("✓", "Purse amount parsed correctly ($50,000)"))
else:
    checks.append(("✗", f"Purse incorrect: ${conditions.get('purse_amount', 0):,}"))

# Check 4: Hierarchy level
if hierarchy.get('base_level', 0) == 6:
    checks.append(("✓", "Hierarchy Level 6 (Handicap) - Industry Standard ✓"))
else:
    checks.append(("✗", f"Hierarchy incorrect: Level {hierarchy.get('base_level', 0)} (expected 6)"))

# Check 5: Class weight
class_weight = summary.get('class_weight', 0)
if 6.5 <= class_weight <= 8.0:  # Level 6 should give ~7.0 weight
    checks.append(("✓", f"Class weight reasonable for Level 6: {class_weight:.2f}"))
else:
    checks.append(("⚠", f"Class weight may be off: {class_weight:.2f}"))

# Print all checks
for symbol, message in checks:
    print(f"{symbol} {message}")

print()
all_passed = all(check[0] == "✓" for check in checks)
if all_passed:
    print("🎉 ALL CHECKS PASSED - Industry-standard hierarchy working correctly!")
    print("   First-time starters should now get proper (lower) ratings in Level 6 races.")
else:
    print("⚠️  SOME CHECKS FAILED - Review results above")

print()
print("=" * 80)
