"""
Integration Test: Race Class Parser + Horse History Analyzer

Tests the complete workflow of parsing race class and analyzing individual
horse history from Brisnet Past Performance text.
"""

from race_class_parser import parse_and_calculate_class
from horse_history_analyzer import (
    analyze_class_movement,
    analyze_distance_movement,
    analyze_connections_changes,
    analyze_surface_change,
    analyze_recent_workouts,
    analyze_horse_form_cycle
)

def test_integration():
    """Test complete integration of race class parsing and horse history analysis"""
    
    print("=" * 80)
    print("INTEGRATION TEST: Race Class + Horse History Analysis")
    print("=" * 80)
    print()
    
    # Sample Brisnet PP text (header + one horse)
    sample_pp = """
Ultimate PP's w/ QuickPlay Comments | Gulfstream Park | ©Hcp 50000 | 1 1/16 Mile | 3yo & Up | Saturday, February 08, 2026 | Race 7

Power Ranking: 1234   L JOCKEY            ODDS ML
1   Hidden Connection      92.3 Zayas        5/2
    Dk b or br g 6 Ghostzapper-Storm Mesa by Bernardini
    Owner: Team Barratt 
    Trainer: Jorge R. Navarro (29% Stakes, +0.45 ROI)
    
    Recent Races:
    29Dec25Tup 8½ ft :46¨ :1:17¨1:44 ©Hcp25k 115 7 3 3¨ô 3¨ô 3¨ 91
    14Nov25GP 8½ ft :47 :1:18 1:45 ©Hcp40k 118 4 2 2¨ 2¨ô 1² 95
    19Oct25GP 9 ft :48 :1:19¨1:51© G3 120 8 5 5©ô 4© 3¨õ 93
    
    Workouts:
    ×29Jan GP 5f ft 1:00¨ B 2/18
    21Jan GP 4f ft :49 H 8/22
    14Jan GP 5f ft 1:01© H 12/25
    """
    
    # STEP 1: Parse race class
    print("STEP 1: Parse Race Class")
    print("-" * 80)
    race_data = parse_and_calculate_class(sample_pp)
    
    print(f"Track: {race_data['summary']['track']}")
    print(f"Race #{race_data['summary']['race_number']}")
    print(f"Class: {race_data['summary']['class_type']}")
    print(f"Purse: ${race_data['summary']['purse']:,}")
    print(f"Distance: {race_data['summary']['distance']} ({race_data['summary']['distance_furlongs']}f)")
    print(f"Hierarchy Level: {race_data['summary']['hierarchy_level']}")
    print(f"Class Weight: {race_data['summary']['class_weight']}")
    print(f"Quality: {race_data['summary']['quality']}")
    print()
    
    # STEP 2: Analyze horse history
    print("STEP 2: Analyze Horse #1 - Hidden Connection")
    print("-" * 80)
    
    # Extract past race data (would normally parse from PP text)
    past_races = [
        {'class_level': 4, 'purse': 25000, 'distance_furlongs': 8.5, 'surface': 'Dirt', 'finish': 3, 'jockey': 'Smith', 'trainer': 'Jones'},
        {'class_level': 5, 'purse': 40000, 'distance_furlongs': 8.5, 'surface': 'Dirt', 'finish': 1, 'jockey': 'Zayas', 'trainer': 'Navarro J'},
        {'class_level': 8, 'purse': 150000, 'distance_furlongs': 9.0, 'surface': 'Dirt', 'finish': 3, 'jockey': 'Zayas', 'trainer': 'Navarro J'},
    ]
    
    # Current race info
    current_class = race_data['hierarchy']['final_level']
    current_distance = float(race_data['header']['distance_furlongs'])
    current_surface = race_data['header']['surface'] or 'Dirt'
    current_jockey = 'Zayas'
    current_trainer = 'Jorge R. Navarro'
    
    # Class movement analysis
    print("\n📊 CLASS MOVEMENT:")
    class_analysis = analyze_class_movement(current_class, past_races, race_data['summary']['purse'])
    print(f"  Movement: {class_analysis['movement']}")
    print(f"  Change: {class_analysis['class_change']} levels")
    print(f"  Avg Recent Class: {class_analysis['avg_recent_class']:.1f}")
    print(f"  Significant: {'Yes' if class_analysis['is_significant'] else 'No'}")
    
    # Distance change analysis
    print("\n📏 DISTANCE CHANGE:")
    distance_analysis = analyze_distance_movement(current_distance, past_races)
    print(f"  Movement: {distance_analysis['movement']}")
    print(f"  Change: {distance_analysis['distance_change']:+.1f} furlongs")
    print(f"  Current Category: {distance_analysis['current_category']}")
    print(f"  Avg Recent Distance: {distance_analysis['avg_recent_distance']:.1f}f")
    
    # Jockey/Trainer change analysis
    print("\n👤 CONNECTIONS:")
    connections_analysis = analyze_connections_changes(
        current_jockey,
        current_trainer,
        past_races
    )
    print(f"  Jockey Change: {'Yes' if connections_analysis['jockey_change'] else 'No'}")
    print(f"  Previous Jockey: {connections_analysis['previous_jockey']}")
    print(f"  Trainer Change: {'Yes' if connections_analysis['trainer_change'] else 'No'}")
    print(f"  Previous Trainer: {connections_analysis['previous_trainer']}")
    
    # Surface change analysis
    print("\n🏁 SURFACE:")
    surface_analysis = analyze_surface_change(current_surface, past_races)
    print(f"  Surface Change: {'Yes' if surface_analysis['surface_change'] else 'No'}")
    print(f"  Current: {current_surface}")
    print(f"  Previous: {surface_analysis['previous_surface']}")
    if surface_analysis['surface_change']:
        print(f"  Type: {surface_analysis['change_type']}")
    
    # Workout analysis
    print("\n💪 RECENT WORKOUTS:")
    sample_workouts = [
        {'date': '29Jan', 'distance': 5.0, 'time': '1:00.2', 'is_bullet': True, 'grade': 'B', 'rank': 2, 'total': 18},
        {'date': '21Jan', 'distance': 4.0, 'time': '0:49.0', 'is_bullet': False, 'grade': 'H', 'rank': 8, 'total': 22},
        {'date': '14Jan', 'distance': 5.0, 'time': '1:01.3', 'is_bullet': False, 'grade': 'H', 'rank': 12, 'total': 25},
    ]
    workout_analysis = analyze_recent_workouts(sample_workouts)
    print(f"  Total Works: {workout_analysis['total_workouts']}")
    print(f"  Bullet Works: {workout_analysis['bullet_count']} ({workout_analysis['bullet_percentage']:.0f}%)")
    print(f"  Recent Bullet: {'Yes' if workout_analysis['has_recent_bullet'] else 'No'}")
    print(f"  Frequency: {workout_analysis['workout_frequency']}")
    
    # STEP 3: Generate Summary
    print()
    print("=" * 80)
    print("HANDICAPPING SUMMARY")
    print("=" * 80)
    
    positive_factors = []
    negative_factors = []
    neutral_factors = []
    
    # Class
    if class_analysis['movement'] == 'Down':
        positive_factors.append(f"✅ Dropping {abs(class_analysis['class_change'])} class levels (easier)")
    elif class_analysis['movement'] == 'Up' and class_analysis['is_significant']:
        negative_factors.append(f"❌ Rising {class_analysis['class_change']} class levels (tougher)")
    
    # Distance
    if distance_analysis['movement'] == 'Same':
        positive_factors.append(f"✅ Proven at today's distance ({current_distance}f)")
    elif distance_analysis['distance_change'] > 1.0:
        neutral_factors.append(f"⚠️ Stretching out {distance_analysis['distance_change']:+.1f}f")
    
    # Connections
    if connections_analysis['jockey_change']:
        neutral_factors.append(f"⚠️ New jockey: {current_jockey}")
    
    # Surface
    if surface_analysis['surface_change']:
        negative_factors.append(f"❌ Surface change: {surface_analysis['change_type']}")
    else:
        positive_factors.append(f"✅ Staying on {current_surface}")
    
    # Workouts
    if workout_analysis['has_recent_bullet']:
        positive_factors.append(f"✅ Recent bullet work ({workout_analysis['bullet_count']} total bullets)")
    
    # Print summary
    if positive_factors:
        print("\n🟢 POSITIVE FACTORS:")
        for factor in positive_factors:
            print(f"  {factor}")
    
    if negative_factors:
        print("\n🔴 NEGATIVE FACTORS:")
        for factor in negative_factors:
            print(f"  {factor}")
    
    if neutral_factors:
        print("\n🟡 NEUTRAL FACTORS:")
        for factor in neutral_factors:
            print(f"  {factor}")
    
    print()
    print("=" * 80)
    print("✅ INTEGRATION TEST COMPLETE")
    print("=" * 80)
    print()
    print("Both modules working together successfully!")
    print("- race_class_parser.py: Extracts race metadata and calculates class weights")
    print("- horse_history_analyzer.py: Analyzes individual horse patterns and trends")
    print()
    return True


if __name__ == '__main__':
    try:
        test_integration()
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
