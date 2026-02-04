"""
Test race_class_parser with actual Pegasus World Cup G1 PP text
"""

from race_class_parser import parse_and_calculate_class

# Actual Pegasus World Cup G1 header from user's PP
pegasus_pp = """Ultimate PP's w/ QuickPlay Comments Gulfstream Park PWCInvit-G1 1„ Mile 4&up Saturday, January 24, 2026 Race 13
# Speed Last Race # Prime Power # Class Rating # Best Speed at Dist
2 British Isles 105
10 Mika 102
11 White Abarrio 100
11 White Abarrio 147.8
3 Full Serrano 145.2
1 Disco Time 144.3
11 White Abarrio 120.9
1 Disco Time 120.4
3 Full Serrano 120.0
5 Skippylongstocking 107
11 White Abarrio 107
2 British Isles 105
13 $1 Exacta / $.50 Trifecta / $.10 Superfecta / $1 Super Hi 5 E1 E2/LATE SPEED
PARS: 95 105/ 94 105
1„ Mile. PWCInvit-G1 Pegasus World Cup Invitational S. Grade I. Purse
$3,000,000 FOUR YEAR OLDS AND UPWARD
"""

print("=" * 80)
print("TESTING PEGASUS WORLD CUP G1 PARSER")
print("=" * 80)

result = parse_and_calculate_class(pegasus_pp)

print("\n" + "=" * 80)
print("RACE CLASS ANALYSIS")
print("=" * 80)

print(f"\n📋 RACE DETAILS:")
print(f"   Track: {result['header']['track_name']}")
print(f"   Race: #{result['header']['race_number']}")
print(f"   Date: {result['header']['day_of_week']}, {result['header']['race_date']}")
print(f"   Distance: {result['header']['distance']} ({result['header']['distance_furlongs']:.1f}F)")
print(f"   Age/Sex: {result['header']['age_restriction']}")

print(f"\n🏆 CLASS IDENTIFICATION:")
print(f"   Race Type: {result['summary']['class_type']}")
print(f"   Quality Rating: {result['summary']['quality'].upper()}")
print(f"   Base Level: {result['hierarchy']['base_level']}")
print(f"   Grade Boost: +{result['hierarchy']['grade_boost']}")
print(f"   Final Level: {result['hierarchy']['final_level']}")

print(f"\n⚖️  CLASS WEIGHT:")
print(f"   Hierarchy Score: {result['weight']['breakdown']['hierarchy_score']:.2f}")
print(f"   Purse Score: +{result['weight']['breakdown']['purse_score']:.3f}")
print(f"   Stakes Bonus: +{result['weight']['breakdown']['stakes_bonus']:.2f}")
print(f"   FINAL CLASS WEIGHT: {result['weight']['class_weight']:.2f}")

print(f"\n💰 PURSE:")
print(f"   Amount: ${result['conditions']['purse_amount']:,}")

print(f"\n📊 RACE INFO:")
print(f"   Is Stakes: {result['conditions']['is_stakes']}")
print(f"   Is Graded: {result['conditions']['is_graded_stakes']}")
print(f"   Grade Level: {result['conditions']['grade_level']}")
print(f"   Purse: ${result['conditions']['purse_amount']:,}")

print(f"\n🎯 EXPECTED BEHAVIOR:")
print(f"   - This is a GRADE 1 STAKES race (highest level)")
print(f"   - Class weight is {result['weight']['class_weight']:.2f} (vs old hardcoded 3.0)")
print(f"   - This {result['weight']['class_weight']/3.0:.1f}x multiplier should dramatically change predictions")
print(f"   - Elite class should favor proven graded stakes winners")
print(f"   - Horses like White Abarrio (defending champ) should get major boost")
print(f"   - First-timers and low-class horses should be heavily suppressed")

print("\n" + "=" * 80)
print("✅ PARSER TEST COMPLETE")
print("=" * 80)
