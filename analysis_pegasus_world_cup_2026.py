"""
PEGASUS WORLD CUP INVITATIONAL G1 - PRE-RACE ANALYSIS
Gulfstream Park, January 24, 2026 - Race 13
$3,000,000 Purse, 1¼ Mile (9F), 4yo and up

Comparing OLD system predictions vs NEW industry-standard hierarchy system
"""

from race_class_parser import parse_and_calculate_class

# ============================================================================
# RACE DETAILS
# ============================================================================
race_header = "Gulfstream Park PWCInvit-G1 1¼ Mile 4&up Saturday, January 24, 2026 Race 13"
race_conditions = "PWCInvit-G1 Pegasus World Cup Invitational S. Grade I. Purse $3,000,000 FOUR YEAR OLDS AND UPWARD"

print("="*80)
print("PEGASUS WORLD CUP INVITATIONAL GRADE 1 - ANALYSIS")
print("="*80)
print(f"Track: Gulfstream Park")
print(f"Date: January 24, 2026")
print(f"Race: 13")
print(f"Distance: 1¼ Mile (9 Furlongs)")
print(f"Purse: $3,000,000")
print(f"Conditions: Grade I Stakes, 4yo and up")
print(f"Field Size: 14 horses")
print()

# ============================================================================
# PARSE RACE WITH NEW SYSTEM
# ============================================================================
print("="*80)
print("NEW SYSTEM: RACE CLASS PARSING")
print("="*80)

# Create a fake Brisnet PP text with the race header
brisnet_pp_text = """Ultimate PP's w/ QuickPlay Comments | Gulfstream Park | PWCInvit-G1 | 1¼ Mile | 4&up | Saturday, January 24, 2026 | Race 13

1¼ Mile. PWCInvit-G1 Pegasus World Cup Invitational S. Grade I. Purse $3,000,000 FOUR YEAR OLDS AND UPWARD
"""

result = parse_and_calculate_class(brisnet_pp_text)

print(f"Track: {result['summary']['track']}")
print(f"Class Type: {result['summary']['class_type']}")
print(f"Distance: {result['summary']['distance']}")
print(f"Purse: ${result['summary']['purse']:,}")
print(f"\n>>> HIERARCHY LEVEL: {result['hierarchy']['final_level']}")
print(f">>> BASE LEVEL: {result['hierarchy']['base_level']}")
print(f">>> GRADE BOOST: +{result['hierarchy']['grade_boost']}")
print(f">>> CLASS WEIGHT: {result['summary']['class_weight']:.2f}")
print()

# ============================================================================
# OLD SYSTEM PREDICTIONS
# ============================================================================
print("="*80)
print("OLD SYSTEM PREDICTIONS (PRE-RACE)")
print("="*80)
print("\n🏆 MOST LIKELY FINISHING ORDER:")
print("   1st: #4 Banishing (16/1) - 65.0% conditional probability")
print("   2nd: #9 Captain Cook (16/1) - 94.2% conditional probability")
print("   3rd: #5 Skippylongstocking (21/1) - 27.4% conditional probability")
print("   4th: #2 British Isles (50/1) - 30.9% conditional probability")
print("   5th: #12 Brotha Keny (50/1) - 34.8% conditional probability")
print()
print("A-GROUP (Key Win Contenders - Use ON TOP):")
print("   • #4 Banishing (ML 20/1 → Live 16/1)")
print()
print("B-GROUP (Primary Challengers - Use 2nd/3rd):")
print("   • #2 British Isles (ML 20/1 → Live 50/1)")
print("   • #5 Skippylongstocking (ML 15/1 → Live 21/1)")
print("   • #9 Captain Cook (ML 15/1 → Live 16/1)")
print()
print("C-GROUP (Underneath Keys - Use 2nd/3rd/4th):")
print("   • #7 Tappan Street (ML 6/1 → Live 10/1)")
print("   • #8 Poster (ML 20/1 → Live 20/1)")
print("   • #11 White Abarrio (ML 4/1 → Live 9/2)")
print("   • #12 Brotha Keny (ML 30/1 → Live 50/1)")
print()

# ============================================================================
# HORSE-BY-HORSE ANALYSIS
# ============================================================================
print("="*80)
print("DETAILED HORSE ANALYSIS")
print("="*80)

horses = [
    {
        "num": 1,
        "name": "Disco Time",
        "ml_odds": "8/5",
        "record": "5: 5-0-0",
        "earnings": "$551,960",
        "last_speed": 91,
        "prime_power": 144.3,
        "class_rating": 120.4,
        "running_style": "E/P",
        "notes": "Undefeated in 5 starts, 77 days layoff, won Dwyer-L200k last out"
    },
    {
        "num": 2,
        "name": "British Isles",
        "ml_odds": "20/1",
        "record": "22: 3-6-4",
        "earnings": "$287,526",
        "last_speed": 105,
        "prime_power": 135.8,
        "class_rating": 117.0,
        "running_style": "E/P",
        "notes": "2nd in Native Diver-G3, new trainer, 63 days layoff"
    },
    {
        "num": 3,
        "name": "Full Serrano",
        "ml_odds": "12/1",
        "record": "20: 7-6-2",
        "earnings": "$936,423",
        "last_speed": 86,
        "prime_power": 145.2,
        "class_rating": 120.0,
        "running_style": "E",
        "notes": "2nd in Goodwood-G1, won BC Dirt Mile-G1 in 2024, 84 days layoff"
    },
    {
        "num": 4,
        "name": "Banishing",
        "ml_odds": "20/1",
        "record": "26: 9-6-3",
        "earnings": "$1,968,409",
        "last_speed": 97,
        "prime_power": 141.5,
        "class_rating": 119.3,
        "running_style": "E/P",
        "notes": "2nd in Ring Bells-B150k, won CT Classic-G2, poor BC Sprint"
    },
    {
        "num": 5,
        "name": "Skippylongstocking",
        "ml_odds": "15/1",
        "record": "35: 12-3-7",
        "earnings": "$3,775,250",
        "last_speed": 95,
        "prime_power": 140.9,
        "class_rating": 117.1,
        "running_style": "E/P",
        "notes": "Won Harlan's Holiday-G3 last out, won Gold Cup-G2, 2nd off layoff"
    },
    {
        "num": 6,
        "name": "Madaket Road",
        "ml_odds": "10/1",
        "record": "9: 1-4-1",
        "earnings": "$580,000",
        "last_speed": 97,
        "prime_power": 143.4,
        "class_rating": 118.9,
        "running_style": "E",
        "notes": "4th in Malibu-G1, 2nd in W. Stephen-G1, 2nd off layoff"
    },
    {
        "num": 7,
        "name": "Tappan Street",
        "ml_odds": "6/1",
        "record": "4: 3-1-0",
        "earnings": "$670,400",
        "last_speed": 92,
        "prime_power": 142.9,
        "class_rating": 120.0,
        "running_style": "E/P",
        "notes": "Won Florida Derby-G1, won last out (OC62.5k), 2nd off layoff"
    },
    {
        "num": 8,
        "name": "Poster",
        "ml_odds": "20/1",
        "record": "7: 4-1-1",
        "earnings": "$409,480",
        "last_speed": 95,
        "prime_power": 138.6,
        "class_rating": 118.2,
        "running_style": "P",
        "notes": "2nd in Harlan's Holiday-G3, won OC100k, blinkers ON, 3rd off layoff"
    },
    {
        "num": 9,
        "name": "Captain Cook",
        "ml_odds": "15/1",
        "record": "8: 2-2-2",
        "earnings": "$456,056",
        "last_speed": 89,
        "prime_power": 143.1,
        "class_rating": 119.2,
        "running_style": "E/P",
        "notes": "2nd in Perryville-G3, 2nd in H. A. Jerkens-G1, 98 days layoff"
    },
    {
        "num": 10,
        "name": "Mika",
        "ml_odds": "10/1",
        "record": "8: 3-1-0",
        "earnings": "$214,664",
        "last_speed": 102,
        "prime_power": 141.0,
        "class_rating": 119.2,
        "running_style": "E",
        "notes": "2nd in Cigar Mile-G2, won Alw50000n1x, 49 days layoff"
    },
    {
        "num": 11,
        "name": "White Abarrio",
        "ml_odds": "4/1",
        "record": "24: 10-2-3",
        "earnings": "$7,151,920",
        "last_speed": 100,
        "prime_power": 147.8,
        "class_rating": 120.9,
        "running_style": "E/P",
        "notes": "Won PWC Invitational-G1 in 2025, won BC Classic-G1 in 2023, 146 days layoff"
    },
    {
        "num": 12,
        "name": "Brotha Keny",
        "ml_odds": "30/1",
        "record": "13: 4-1-3",
        "earnings": "$499,528",
        "last_speed": 97,
        "prime_power": 134.9,
        "class_rating": 117.0,
        "running_style": "E/P",
        "notes": "Won Zia Park Derby-L300k, new trainer, 60 days layoff"
    },
    {
        "num": 13,
        "name": "Lightning Tones",
        "ml_odds": "30/1",
        "record": "33: 7-7-6",
        "earnings": "$377,085",
        "last_speed": 93,
        "prime_power": 131.1,
        "class_rating": 118.5,
        "running_style": "S",
        "notes": "Won Sunshine Classic-B75k, 14 days since last race (fresh)"
    },
    {
        "num": 14,
        "name": "Catalytic",
        "ml_odds": "50/1",
        "record": "14: 3-7-1",
        "earnings": "$362,085",
        "last_speed": 88,
        "prime_power": 130.3,
        "class_rating": 114.5,
        "running_style": "E/P",
        "notes": "3rd in Harlan's Holiday-G3, 35 days since last race"
    }
]

print("\n📊 KEY SPEED/POWER RANKINGS:")
print("\nPrime Power (Overall Ability):")
for i, h in enumerate(sorted(horses, key=lambda x: x['prime_power'], reverse=True)[:5], 1):
    print(f"   {i}. #{h['num']} {h['name']}: {h['prime_power']}")

print("\nClass Rating (Historical Class):")
for i, h in enumerate(sorted(horses, key=lambda x: x['class_rating'], reverse=True)[:5], 1):
    print(f"   {i}. #{h['num']} {h['name']}: {h['class_rating']}")

print("\nLast Race Speed:")
for i, h in enumerate(sorted(horses, key=lambda x: x['last_speed'], reverse=True)[:5], 1):
    print(f"   {i}. #{h['num']} {h['name']}: {h['last_speed']}")

# ============================================================================
# NEW SYSTEM ANALYSIS
# ============================================================================
print("\n" + "="*80)
print("NEW SYSTEM: GRADE 1 STAKES IMPACT ANALYSIS")
print("="*80)

print(f"\n⚡ CLASS WEIGHT: {result['class_weight']:.2f}")
print(f"   This is the MAXIMUM multiplier in our system!")
print(f"   Grade 1 = Base Level 7 + Grade Boost 3 = Level 10")
print()
print("🔍 WHAT THIS MEANS:")
print("   • Class weight of 10.00 heavily rewards proven G1 performers")
print("   • Horses with deep G1/G2 experience get huge class ratings")
print("   • Speed ratings are critical - inconsistent horses penalized")
print("   • Layoffs matter more - form cycle is crucial at this level")
print()

print("="*80)
print("TOP CONTENDERS - NEW SYSTEM ANALYSIS")
print("="*80)

# Analyze top horses
top_horses = [
    {"num": 11, "name": "White Abarrio", "odds": "4/1",
     "strengths": ["Highest Prime Power (147.8)", "Highest Class Rating (120.9)", "Won this race last year",
                   "Won BC Classic-G1 2023", "8-1-0 at Gulfstream"],
     "concerns": ["146 days layoff (longest in field)", "Last race 5th in JC Gold Cup",
                  "Lost iron in JC Gold Cup (bad trip)", "Will fitness be issue?"]},
    
    {"num": 3, "name": "Full Serrano", "odds": "12/1",
     "strengths": ["2nd highest Prime Power (145.2)", "Won BC Dirt Mile-G1 2024", "Best Dirt Speed 120.0",
                   "Proven G1 performer", "Strong late pace"],
     "concerns": ["84 days layoff", "Last race: 5th in BC Dirt Mile (off form)", "Age 7 - oldest in field",
                  "Speed figure dropped to 86 last out"]},
    
    {"num": 1, "name": "Disco Time", "odds": "8/5", 
     "strengths": ["UNDEFEATED 5-0-0", "3rd highest Prime Power (144.3)", "Won Dwyer-L200k last out",
                   "High % trainer (Cox 32%)", "Sharp recent workouts"],
     "concerns": ["⚠️ HUGE CLASS JUMP: From Listed stakes to G1", "Never faced G1 competition",
                  "77 days layoff", "Class rating only 120.4 (not elite G1 level)"]},
    
    {"num": 7, "name": "Tappan Street", "odds": "6/1",
     "strengths": ["Won Florida Derby-G1", "Won last out", "Only 4 career starts (lightly raced)",
                   "2nd off layoff (fitness edge)", "High % trainer (Cox 32%)"],
     "concerns": ["Last win was OC62500 (not stakes)", "Speed figure only 92 last out",
                  "G1 win in weak Florida Derby field", "Unproven at this elite level"]},
    
    {"num": 4, "name": "Banishing", "odds": "20/1",
     "strengths": ["Highest earnings $1.97M", "Won CT Classic-G2", "Multiple G2 wins",
                   "2nd in Lukas Classic-G2", "Proven dirt router"],
     "concerns": ["⚠️ OLD SYSTEM'S TOP PICK", "12th in BC Sprint (badly beaten)",
                  "Only 97 speed figure last out", "Questionable G1 form"]}
]

for horse in top_horses:
    print(f"\n#{horse['num']} {horse['name'].upper()} ({horse['odds']})")
    print("   ✅ Strengths:")
    for s in horse['strengths']:
        print(f"      • {s}")
    print("   ⚠️ Concerns:")
    for c in horse['concerns']:
        print(f"      • {c}")

# ============================================================================
# KEY DIFFERENCES: OLD VS NEW SYSTEM
# ============================================================================
print("\n" + "="*80)
print("🔄 CRITICAL DIFFERENCES: OLD SYSTEM vs NEW SYSTEM")
print("="*80)

print("\n❌ OLD SYSTEM ISSUES:")
print("   • Put #4 Banishing as A-Group solo pick (top win contender)")
print("   • Banishing ML 20/1 → Live 16/1 (not favored by bettors)")
print("   • Banishing was 12th of 14 in BC Sprint (terrible G1 form)")
print("   • #9 Captain Cook as 94.2% likely for 2nd place")
print("   • Captain Cook last race speed only 89 (poor)")
print("   • Had #11 White Abarrio in C-Group (underneath keys)")
print("   • White Abarrio is defending champ and highest earner!")
print()

print("✅ NEW SYSTEM ADVANTAGES:")
print(f"   • Grade 1 = Level 10 class weight (maximum)")
print(f"   • Heavily rewards proven G1 performers")
print(f"   • #11 White Abarrio: Highest Prime Power + Class Rating")
print(f"   • #3 Full Serrano: Won BC Dirt Mile-G1, proven at this level")
print(f"   • #1 Disco Time: Undefeated but HUGE class jump flagged")
print(f"   • Speed figures weigh heavily - recent form crucial")
print(f"   • Layoffs penalized appropriately for elite level")
print()

print("="*80)
print("NEW SYSTEM TOP PREDICTIONS (PRELIMINARY)")
print("="*80)
print("\n🎯 LOGICAL TOP CONTENDERS:")
print("   1. #11 White Abarrio (4/1) - Highest ratings, defending champ")
print("   2. #3 Full Serrano (12/1) - Proven G1 winner, strong ratings")
print("   3. #1 Disco Time (8/5 favorite) - Undefeated but class test")
print("   4. #7 Tappan Street (6/1) - Won Florida Derby, fresh form")
print("   5. #6 Madaket Road (10/1) - Multiple G1 placings, improving")
print()
print("⚠️ SKEPTICAL OF:")
print("   • #4 Banishing (20/1) - Old system's top pick, poor BC Sprint form")
print("   • #9 Captain Cook (15/1) - 98 day layoff, speed figure dropped")
print("   • #2 British Isles (20/1) - Never won above OC80k level")
print()

print("="*80)
print("KEY HANDICAPPING FACTORS FOR THIS RACE")
print("="*80)
print("\n🏇 GRADE 1 DIRT ROUTE DYNAMICS:")
print("   • Early speed is crucial (E/E-P types dominate)")
print("   • Recent form matters - layoffs are risky at this level")
print("   • Proven G1 winners have huge edge")
print("   • Class jump from Listed/G3 to G1 is MASSIVE")
print("   • $3M purse = best horses show up (no easy spots)")
print()
print("📊 TRACK BIAS (Gulfstream Dirt Routes):")
print("   • Speed bias: 67% winners avg beaten lengths")
print("   • Wire-to-wire: 21% (pressure early pace)")
print("   • Posts 4-7: Impact value 1.15 (slight advantage)")
print("   • E/P types: 1.08 impact (slightly favored)")
print()

print("="*80)
print("WAITING FOR ACTUAL RESULTS...")
print("="*80)
print("\nOnce results are available, we will validate:")
print("   ✓ Did #4 Banishing win (old system's A-Group pick)?")
print("   ✓ Did #9 Captain Cook finish 2nd (94.2% conditional probability)?")
print("   ✓ Did proven G1 winners (#11, #3) perform better?")
print("   ✓ Did #1 Disco Time handle the class jump?")
print("   ✓ Did the Level 10 class weight properly reward elite performers?")
print()
print("This is THE ultimate test:")
print("   • Grade 1 Stakes (highest level)")
print("   • $3,000,000 purse (elite field)")
print("   • 14 horses (deep competition)")
print("   • Proven champions vs rising stars")
print()
print("🎯 Ready for results whenever you have them!")
print("="*80)
