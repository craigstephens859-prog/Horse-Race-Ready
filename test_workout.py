"""
Test script for recent workout parsing from BRISNET PP sample
"""
import re

def parse_recent_workout_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses the most recent workout from a horse's PP text block.
    
    BRISNET Format (workout lines appear at bottom of horse block):
    - "25Oct SA 4f ft :47« H 12/62"
    - "18Oct SA 5f ft 1:02© Hg 37/48"
    
    Format: Date Track Distance Surface Time Grade Rank/Total
    
    Returns dict with keys: 'workout_date', 'workout_track', 'workout_distance', 
                           'workout_time', 'workout_rank', 'workout_total'
    """
    result = {
        'workout_date': '',
        'workout_track': '',
        'workout_distance': '',
        'workout_time': '',
        'workout_rank': '',
        'workout_total': ''
    }
    
    if not block:
        return result
    
    # Workout lines typically start with date pattern like "25Oct" or "18Oct"
    # Format: DDMmmYY Track Distance Surface Time Grade Rank/Total
    # Example: "25Oct SA 4f ft :47« H 12/62"
    # Example with bullet: "×26Oct SA 4f ft :47¨ H 1/11" (× char 215 indicates best workout)
    # Look for lines with this pattern after the race data ends
    # Time can include special chars: « (char 171), © (169), ª (170), ¬ (172), ® (174), ¯ (175), ° (176), ¨ (168)
    workout_pattern = r'×?(\d{1,2}[A-Z][a-z]{2})\s+([A-Z][A-Za-z]{1,3})\s+(\d+f?)\s+(?:ft|gd|sy|sl|fm|hy|my|tr\.t|˜)\s+([\d:\.«©ª¬®¯°¨]+)\s+[A-Z]?g?\s+(\d+)/(\d+)'
    
    matches = re.findall(workout_pattern, block, re.MULTILINE)
    
    if matches:
        # Take the first (most recent) workout
        first_workout = matches[0]
        result['workout_date'] = first_workout[0]
        result['workout_track'] = first_workout[1]
        result['workout_distance'] = first_workout[2]
        result['workout_time'] = first_workout[3]
        result['workout_rank'] = first_workout[4]
        result['workout_total'] = first_workout[5]
        
        if debug:
            print(f"  Recent workout: {result['workout_date']} {result['workout_track']} {result['workout_distance']} {result['workout_time']} (#{result['workout_rank']}/{result['workout_total']})")
    
    return result


# Test data from the sample PP (workout sections)
sample_blocks = {
    "Omnipontet": """20Jly25Dmr° à 1m fm :23 :47 1:11« 1:34« ¦ ¨¨ ™'OsunitasL 100k¨¨¬ 87 87/ 90 +9 +3 87 4 8 8¬ 8«ƒ 9«ƒ 10¬ƒ FresuA¨©¬ b 23.20
25Oct SA 4f ft :47« H 12/62 
18Oct SA 5f ft 1:02© Hg 37/48 
12Oct SA 5f ft 1:02 H 45/68""",
    
    "Nay V Belle": """29Aug25Dmr® à 5f fm :22« :45« :57¨ ¦ ¨¨¬ ™OC100k/b-N ¨¨® 89/ 98 96 9 3 6« 6©ƒ 3 1³ DesormeauxKJ¨©§ L 7.00
24Oct SA 4f ft :50« H 37/39 
17Oct SA 5f ft 1:00© H 13/43 
28Sep SA 5f ft 1:03¨ H 103/110""",
    
    "Queen Maxima": """30Aug25Dmr° à 5f fm :21© :44 :55« ¦ ¨¨® GrnFlshH-G3 ¨¨® 95/ 91 95 10 6 9¬ 9 9¬‚ 6© HernandezJJ¨¨¯ 3.00
×26Oct SA 4f ft :47¨ H 1/11 
19Oct SA 4f ft :47¨ H 3/25 
12Oct SA 5f ft :59© H 5/19""",
    
    "Sunglow": """28Sep25SA« à 6f fm :22ª :45© :56© 1:07« ¦ ¨¨« ™OC50k/n1x-N¨¨ 88 89/ 98 -7 -11 92 4 3 1 1¨ 1© 1ª‚ HernandezJJ¨©§ L 7.70
24Oct SA 5f ft :59 H 2/12 
17Oct SA 5f ft :59« H 1/3 
10Oct SA 4f ft :48ª H 4/9""",
    
    "Jungle Peace": """06Sep25KD° à 6½ gd :21« :44©1:08© 1:14© ¨¨ ™MusicCty-G2¨¨® 97 105/ 75 +11 +8 88 6 4 8«ƒ 8©ƒ 7¬ƒ 5¬ RosarioJ¨©§ 31.81
25Oct SA 5f ft 1:00ª H 3/15 
19Oct SA 5f ft 1:00« H 9/15 
12Oct SA 5f ft :59« H 9/19"""
}

# Expected results
expected = {
    "Omnipontet": {
        "date": "25Oct",
        "track": "SA",
        "distance": "4f",
        "time": ":47«",
        "rank": "12",
        "total": "62"
    },
    "Nay V Belle": {
        "date": "24Oct",
        "track": "SA",
        "distance": "4f",
        "time": ":50«",
        "rank": "37",
        "total": "39"
    },
    "Queen Maxima": {
        "date": "26Oct",
        "track": "SA",
        "distance": "4f",
        "time": ":47¨",
        "rank": "1",
        "total": "11"
    },
    "Sunglow": {
        "date": "24Oct",
        "track": "SA",
        "distance": "5f",
        "time": ":59",
        "rank": "2",
        "total": "12"
    },
    "Jungle Peace": {
        "date": "25Oct",
        "track": "SA",
        "distance": "5f",
        "time": "1:00ª",
        "rank": "3",
        "total": "15"
    }
}

print("="*60)
print("Testing Recent Workout Parsing")
print("="*60)

all_passed = True
for horse, block in sample_blocks.items():
    print(f"\n🐎 Testing: {horse}")
    print("-"*60)
    
    result = parse_recent_workout_for_block(block, debug=True)
    expected_result = expected[horse]
    
    date_match = result['workout_date'] == expected_result['date']
    track_match = result['workout_track'] == expected_result['track']
    distance_match = result['workout_distance'] == expected_result['distance']
    time_match = result['workout_time'] == expected_result['time']
    rank_match = result['workout_rank'] == expected_result['rank']
    total_match = result['workout_total'] == expected_result['total']
    
    all_match = date_match and track_match and distance_match and time_match and rank_match and total_match
    
    print(f"  Date: {'✅' if date_match else '❌'} Expected: '{expected_result['date']}', Got: '{result['workout_date']}'")
    print(f"  Track: {'✅' if track_match else '❌'} Expected: '{expected_result['track']}', Got: '{result['workout_track']}'")
    print(f"  Distance: {'✅' if distance_match else '❌'} Expected: '{expected_result['distance']}', Got: '{result['workout_distance']}'")
    print(f"  Time: {'✅' if time_match else '❌'} Expected: '{expected_result['time']}', Got: '{result['workout_time']}'")
    print(f"  Rank: {'✅' if rank_match else '❌'} Expected: '{expected_result['rank']}', Got: '{result['workout_rank']}'")
    print(f"  Total: {'✅' if total_match else '❌'} Expected: '{expected_result['total']}', Got: '{result['workout_total']}'")
    
    if not all_match:
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("✅ ALL TESTS PASSED!")
    print(f"Successfully parsed {len(sample_blocks)} horses")
else:
    print("❌ SOME TESTS FAILED")
print("="*60)
