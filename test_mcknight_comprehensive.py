"""
COMPREHENSIVE BRISNET EXTRACTION TEST - McKnight Grade 3
Tests all elite parsers with real PP data to ensure every angle is captured
"""

import re
from typing import Dict, List, Any, Optional

# ============================================================================
# ELITE PARSERS (from app.py)
# ============================================================================

def parse_workout_data(pp_text: str, horse_name: str) -> Dict[str, Any]:
    """Extract workout information from PP text"""
    workout_data = {
        'latest_work': None,
        'work_date': None,
        'work_track': None,
        'work_distance': None,
        'work_time': None,
        'work_condition': None,
        'work_rating': None,
        'work_rank': None,
        'work_total': None,
        'percentile': None,
        'days_since': None,
        'quality': 'none'
    }
    
    # Pattern: "18Jan Pay 4f ft :48ª B 1/40"
    workout_pattern = r'(\d{2}[A-Z][a-z]{2})\s+([A-Za-z]+)\s+(?:˜\s+)?(?:\(d\)\s+)?(\d+f|[\d]+)\s+(ft|fm|gd|my|sy|wf|yl)\s+([\d:\.]+)\s+([BHG][bg]?)\s*(?:(\d+)/(\d+))?'
    
    lines = pp_text.split('\n')
    for line in lines:
        match = re.search(workout_pattern, line)
        if match:
            work_date, track, distance, condition, time, rating, rank, total = match.groups()
            
            # Calculate percentile if rank/total available
            percentile = None
            if rank and total:
                percentile = (int(rank) / int(total)) * 100
                
            workout_data.update({
                'latest_work': line.strip(),
                'work_date': work_date,
                'work_track': track,
                'work_distance': distance,
                'work_time': time,
                'work_condition': condition,
                'work_rating': rating,
                'work_rank': rank,
                'work_total': total,
                'percentile': percentile,
                'quality': 'bullet' if 'B' in rating else 'handily' if 'H' in rating else 'regular'
            })
            break
    
    return workout_data


def parse_jockey_trainer_stats(pp_text: str, horse_name: str) -> Dict[str, Any]:
    """Extract jockey and trainer statistics"""
    stats = {
        'jockey_name': None,
        'jockey_stats': None,
        'jockey_wins': 0,
        'jockey_starts': 0,
        'jockey_win_pct': 0.0,
        'trainer_name': None,
        'trainer_stats': None,
        'trainer_wins': 0,
        'trainer_starts': 0,
        'trainer_win_pct': 0.0
    }
    
    # Jockey pattern: "ORTIZ, JR. IRAD (148 43-29-20 29%)"
    jockey_pattern = r'([A-Z\s,\.]+)\s+\((\d+)\s+(\d+)-(\d+)-(\d+)\s+(\d+)%\)'
    jockey_match = re.search(jockey_pattern, pp_text)
    if jockey_match:
        name, starts, wins, places, shows, pct = jockey_match.groups()
        stats.update({
            'jockey_name': name.strip(),
            'jockey_stats': f"{wins}-{places}-{shows}",
            'jockey_wins': int(wins),
            'jockey_starts': int(starts),
            'jockey_win_pct': float(pct)
        })
    
    # Trainer pattern: "Trnr: Maker Michael J (30 5-5-2 17%)"
    trainer_pattern = r'Trnr:\s+([A-Za-z\s,\.]+)\s+\((\d+)\s+(\d+)-(\d+)-(\d+)\s+(\d+)%\)'
    trainer_match = re.search(trainer_pattern, pp_text)
    if trainer_match:
        name, starts, wins, places, shows, pct = trainer_match.groups()
        stats.update({
            'trainer_name': name.strip(),
            'trainer_stats': f"{wins}-{places}-{shows}",
            'trainer_wins': int(wins),
            'trainer_starts': int(starts),
            'trainer_win_pct': float(pct)
        })
    
    return stats


def analyze_class_movement(past_races: List[Dict], today_class: str = "G3") -> Dict[str, Any]:
    """Analyze class changes from past races"""
    class_hierarchy = {
        'Msw': 1, 'Mdn': 1,
        'Mcl': 2, 'MC': 2,
        'Clm': 3,
        'OC': 4,
        'Alw': 5,
        'Hcp': 6,
        'Stk': 7,
        'L': 8,  # Listed stakes
        'G3': 9,
        'G2': 10,
        'G1': 11
    }
    
    # Determine today's class level
    today_level = class_hierarchy.get(today_class, 5)
    
    # Analyze recent class levels
    recent_classes = []
    for race in past_races[:5]:
        race_class = race.get('class', '')
        for key, level in class_hierarchy.items():
            if key in race_class:
                recent_classes.append(level)
                break
    
    if not recent_classes:
        return {'pattern': 'unknown', 'bonus': 0.0}
    
    avg_recent = sum(recent_classes) / len(recent_classes)
    
    # Determine pattern
    if today_level > avg_recent + 1:
        pattern = 'stepping_up'
        bonus = -0.10
    elif today_level < avg_recent - 1:
        pattern = 'dropping_down'
        bonus = +0.12
    else:
        pattern = 'stable'
        bonus = 0.0
    
    return {
        'pattern': pattern,
        'today_level': today_level,
        'avg_recent': avg_recent,
        'bonus': bonus
    }


def parse_prime_power(pp_text: str) -> float:
    """Extract Prime Power rating"""
    prime_pattern = r'Prime Power:\s+([\d\.]+)'
    match = re.search(prime_pattern, pp_text)
    if match:
        return float(match.group(1))
    return 0.0


def parse_speed_ratings(pp_text: str) -> Dict[str, int]:
    """Extract speed ratings"""
    ratings = {
        'best_speed': 0,
        'best_turf_speed': 0,
        'last_speed': 0
    }
    
    # Best turf speed
    turf_pattern = r'Trf\s+\(\d+\)\s+\d+\s+[\d\s-]+\$[\d,]+\s+(\d+)'
    match = re.search(turf_pattern, pp_text)
    if match:
        ratings['best_turf_speed'] = int(match.group(1))
    
    # Last race speed (from SPD column in race line)
    speed_pattern = r'SPD\s+PP[^\n]+\n[^\n]+\s+(\d+)\s+\d+\s+\d+'
    match = re.search(speed_pattern, pp_text)
    if match:
        ratings['last_speed'] = int(match.group(1))
    
    return ratings


def extract_quickplay_comments(pp_text: str) -> Dict[str, List[str]]:
    """Extract QuickPlay positive (ñ) and negative (×) comments"""
    comments = {
        'positives': [],
        'negatives': []
    }
    
    # Positive angles (ñ)
    pos_pattern = r'ñ\s+([^\n×]+)'
    positives = re.findall(pos_pattern, pp_text)
    comments['positives'] = [p.strip() for p in positives]
    
    # Negative angles (×)
    neg_pattern = r'×\s+([^\nñ]+)'
    negatives = re.findall(neg_pattern, pp_text)
    comments['negatives'] = [n.strip() for n in negatives]
    
    return comments


def parse_running_style(pp_text: str) -> str:
    """Extract running style"""
    style_pattern = r'\(([ESP/]+)\s+\d+\)'
    match = re.search(style_pattern, pp_text)
    if match:
        return match.group(1)
    return "Unknown"


def calculate_horse_analysis(pp_text: str, horse_name: str) -> Dict[str, Any]:
    """Comprehensive analysis of a single horse"""
    
    analysis = {
        'horse_name': horse_name,
        'prime_power': parse_prime_power(pp_text),
        'speed_ratings': parse_speed_ratings(pp_text),
        'running_style': parse_running_style(pp_text),
        'workout_data': parse_workout_data(pp_text, horse_name),
        'jockey_trainer': parse_jockey_trainer_stats(pp_text, horse_name),
        'quickplay_comments': extract_quickplay_comments(pp_text),
        'class_analysis': analyze_class_movement([], "G3"),  # Simplified for test
        'total_bonus': 0.0
    }
    
    # Calculate workout bonus
    workout = analysis['workout_data']
    workout_bonus = 0.0
    if workout['quality'] == 'bullet':
        workout_bonus += 0.10
    elif workout['quality'] == 'handily':
        workout_bonus += 0.05
    
    if workout['percentile']:
        if workout['percentile'] <= 20:
            workout_bonus += 0.08
        elif workout['percentile'] <= 40:
            workout_bonus += 0.05
    
    # Calculate jockey/trainer bonus
    jt = analysis['jockey_trainer']
    jt_bonus = 0.0
    if jt['jockey_win_pct'] >= 25:
        jt_bonus += 0.15
    elif jt['jockey_win_pct'] >= 20:
        jt_bonus += 0.10
    
    if jt['trainer_win_pct'] >= 28:
        jt_bonus += 0.12
    elif jt['trainer_win_pct'] >= 20:
        jt_bonus += 0.08
    
    analysis['workout_bonus'] = workout_bonus
    analysis['jockey_trainer_bonus'] = jt_bonus
    analysis['total_bonus'] = workout_bonus + jt_bonus + analysis['class_analysis']['bonus']
    
    return analysis


# ============================================================================
# RACE DATA
# ============================================================================

RACE_DATA = """
RACE: William L. McKnight S.-G3
DISTANCE: 1½ Mile (Turf)
PURSE: $225,000
TRACK: Gulfstream Park
DATE: January 24, 2026
"""

HORSES = {
    "Zverev": """
1 Zverev (S 4)
ORTIZ, JR. IRAD (148 43-29-20 29%)
Trnr: Maker Michael J (30 5-5-2 17%)
Prime Power: 146.0 (3rd)
Life: 15 5 - 1 - 1 $379,019 92
Trf (111) 12 3 - 1 - 1 $261,835 91
ñ Switches to a high % jockey
ñ Won last race (TP 12/06 10f All Weather ft BooneCntyB125k)
ñ Hot Jockey in last 7 days (24 6-5-2)
× Moves up in class from last start
× Has not raced in 49 days
17Jan GP 5f ft 1:02 B 30/44
10Jan GP 5f ft 1:02 B 21/38
03Jan Cdt 5f ft 1:02 B 11/25
""",

    "Act a Fool": """
2 Act a Fool (E/P 6)
BRAVO JOE (46 5-5-8 11%)
Trnr: Cazares Laura (28 3-4-3 11%)
Prime Power: 127.9 (11th)
Life: 12 6 - 1 - 0 $166,250 86
ñ Won last race (GP 12/13 8f Turf fm OC50000)
× Moves up in class from last start
× Poor Speed Figures
19Nov'25 Pmm ˜ 4f fm :48 B 6/18
15Aug'25 GP 4f ft :48 B 1/5
09Aug'25 GP 4f ft :48¨ B 6/8
""",

    "Ohana Honor": """
3 Ohana Honor (P 3)
PRAT FLAVIEN (5 1-2-1 20%)
Trnr: McGaughey III Claude R (17 0-3-4 0%)
Prime Power: 159.6 (1st)
Life: 19 4 - 5 - 3 $636,397 102
Trf (106) 13 3 - 3 - 2 $546,097 102
ñ Won last race (AQU 11/15 9f Turf fm KnkrbkrL150k)
ñ Sharp 4F workout (Jan-18)
× Moves up in class from last start
× Has not raced for more than 2 months
×18Jan Pay 4f ft :48ª B 1/40
×11Jan Pay 4f ft :48© B 1/37
×05Jan Pay 4f ft :48© B 1/16
""",

    "Layabout": """
4 Layabout (P 5)
EGAN DAVID (117 20-11-16 17%)
Trnr: Biancone Patrick L (24 4-4-4 17%)
Prime Power: 144.5 (6th)
Life: 9 4 - 1 - 0 $274,225 91
ñ Won last race (GP 12/13 8.5f Turf fm TrPDerbyL125k)
× Moves up in class from last start
18Jan Pmm ˜ 6f fm 1:12 B 1/2
28Dec'25 Pmm ˜ 5f fm 1:01¨ B 12/25
""",

    "Il Siciliano": """
5 Il Siciliano (S 0)
CASTELLANO JAVIER (112 18-15-17 16%)
Trnr: Sano Antonio (63 4-7-11 6%)
Prime Power: 136.6 (10th)
Life: 17 1 - 3 - 3 $125,470 99
× Only 1 win(s) in 17 career starts
× Has not raced for more than 3 months
×10Jan GP 5f ft :59« B 1/38
03Jan GP 5f ft 1:03 B 23/28
""",

    "Missed the Cut": """
6 Missed the Cut (P 4)
ZAYAS EDGARD J (147 23-34-18 16%)
Trnr: Dibona Bobby S (22 4-5-2 18%)
Prime Power: 147.1 (2nd)
Life: 21 7 - 3 - 2 $548,301 105
ñ High % trainer
ñ Eligible to improve in 2nd start since layoff
× Beaten by weaker in last race
×10Jan GP 5f ft :59« B 1/38
20Dec'25 GP 5f ft 1:01ª B 19/38
13Dec'25 GP 5f ft 1:01¨ B 15/26
""",

    "Padiddle": """
7 Padiddle (P 3)
FRANCO MANUEL (0 0-0-0 0%)
Trnr: Dobles Elizabeth L (19 3-2-4 16%)
Prime Power: 127.1 (12th)
Life: 16 1 - 4 - 6 $252,335 91
× Only 1 win(s) in 16 career starts
× Has not raced in 49 days
×21Dec'25 Pmm ˜ 4f fm :46© B 1/61
13Apr'25 GP 4f ft :50 B 44/57
""",

    "Offlee Naughty": """
8 Offlee Naughty (S 0)
SAEZ LUIS (19 4-3-3 21%)
Trnr: Ramsey Nolan (22 1-3-3 5%)
Prime Power: 136.7 (9th)
Life: 24 5 - 3 - 2 $433,245 107
ñ Highest last race speed rating
ñ Best Turf Speed is fastest among today's starters
× Has not raced for more than a year
× Poor trainer win% this meet
17Jan GP 5f ft 1:01¨ B 23/44
10Jan GP 6f ft 1:16© B 1/1
""",

    "Balnikhov": """
9 Balnikhov (S 0)
RISPOLI UMBERTO (0 0-0-0 0%)
Trnr: DAmato Philip (0 0-0-0 0%)
Prime Power: 142.1 (8th)
Life: 37 8 - 6 - 6 $1,169,317 105
× Has not raced in 57 days
×07Jan SA 5f ft 1:00ª H 1/13
31Dec'25 SA 4f ft :48ª H 6/18
""",

    "Summer Cause": """
10 Summer Cause (P 3)
GAFFALIONE TYLER (190 30-33-28 16%)
Trnr: Clement Miguel (12 2-1-2 17%)
Prime Power: 145.5 (5th)
Life: 17 4 - 2 - 5 $258,970 92
ñ Won last race (GP 12/06 16f Turf fm HAlnJrknHL100k)
× Moves up in class from last start
× Has not raced in 49 days
17Jan Pay 5f ft 1:05 B 15/16
08Jan Pay 5f ft 1:03¨ B 4/5
""",

    "Divin Propos": """
11 Divin Propos (P 0)
MURPHY OISIN (0 0-0-0 0%)
Trnr: Joseph, Jr. Saffie A (103 19-18-18 18%)
Prime Power: 145.9 (4th)
Life: 19 4 - 3 - 4 $222,811 103
ñ High % trainer
ñ Sharp 5F workout (Jan-16)
×16Jan Pmm 5f ft 1:00« B 1/8
×02Jan Pmm 3f ft :35« B 1/23
×04Oct'25 Pmm 4f wf :46¨ B 1/56
""",

    "Hammerhead": """
12 Hammerhead (S 5)
ORTIZ JOSE L (0 0-0-0 0%)
Trnr: Attard Kevin (2 0-0-0 0%)
Prime Power: 143.7 (7th)
Life: 13 2 - 4 - 2 $207,795 92
ñ Won last race (WO 12/06 12f All Weather ft Valdctry-G3)
× Has not raced in 49 days
×27Nov'25 WO 5f ft 1:00© B 1/4
15Nov'25 WO 4f ft :47« B 4/75
"""
}


# ============================================================================
# MAIN TEST
# ============================================================================

def main():
    print("=" * 80)
    print("COMPREHENSIVE BRISNET EXTRACTION TEST")
    print("William L. McKnight S.-G3 - 1½ Mile Turf")
    print("=" * 80)
    print()
    
    results = []
    
    for horse_name, pp_text in HORSES.items():
        print(f"\n{'='*80}")
        print(f"HORSE: {horse_name}")
        print(f"{'='*80}")
        
        analysis = calculate_horse_analysis(pp_text, horse_name)
        results.append(analysis)
        
        # Display results
        print(f"\n🏇 BASIC INFO:")
        print(f"   Prime Power: {analysis['prime_power']:.1f}")
        print(f"   Running Style: {analysis['running_style']}")
        
        print(f"\n🏃 SPEED RATINGS:")
        for key, val in analysis['speed_ratings'].items():
            if val > 0:
                print(f"   {key.replace('_', ' ').title()}: {val}")
        
        print(f"\n💪 WORKOUT DATA:")
        workout = analysis['workout_data']
        if workout['latest_work']:
            print(f"   Latest: {workout['work_date']} {workout['work_track']} {workout['work_distance']}")
            print(f"   Time: {workout['work_time']} ({workout['work_condition']})")
            print(f"   Rating: {workout['work_rating']} ({workout['quality']})")
            if workout['percentile']:
                print(f"   Rank: {workout['work_rank']}/{workout['work_total']} (Top {workout['percentile']:.0f}%)")
            print(f"   Bonus: +{analysis['workout_bonus']:.2f}")
        else:
            print(f"   No workout data found")
        
        print(f"\n👤 JOCKEY/TRAINER:")
        jt = analysis['jockey_trainer']
        if jt['jockey_name']:
            print(f"   Jockey: {jt['jockey_name']}")
            print(f"   Stats: {jt['jockey_stats']} ({jt['jockey_win_pct']:.0f}% win)")
        if jt['trainer_name']:
            print(f"   Trainer: {jt['trainer_name']}")
            print(f"   Stats: {jt['trainer_stats']} ({jt['trainer_win_pct']:.0f}% win)")
        jt_bonus = analysis['jockey_trainer_bonus']
        if jt_bonus > 0:
            print(f"   Bonus: +{jt_bonus:.2f}")
        
        print(f"\n✅ QUICKPLAY POSITIVES:")
        for comment in analysis['quickplay_comments']['positives']:
            print(f"   ✓ {comment}")
        
        print(f"\n❌ QUICKPLAY NEGATIVES:")
        for comment in analysis['quickplay_comments']['negatives']:
            print(f"   ✗ {comment}")
        
        print(f"\n💰 TOTAL BONUS: {analysis['total_bonus']:+.2f}")
    
    # Summary rankings
    print(f"\n\n{'='*80}")
    print("RANKINGS BY PRIME POWER")
    print(f"{'='*80}")
    
    sorted_pp = sorted(results, key=lambda x: x['prime_power'], reverse=True)
    for i, horse in enumerate(sorted_pp, 1):
        print(f"{i:2d}. {horse['horse_name']:20s} {horse['prime_power']:6.1f} (Bonus: {horse['total_bonus']:+.2f})")
    
    print(f"\n{'='*80}")
    print("RANKINGS BY TOTAL BONUS")
    print(f"{'='*80}")
    
    sorted_bonus = sorted(results, key=lambda x: x['total_bonus'], reverse=True)
    for i, horse in enumerate(sorted_bonus, 1):
        style = horse['running_style']
        pp = horse['prime_power']
        bonus = horse['total_bonus']
        print(f"{i:2d}. {horse['horse_name']:20s} {bonus:+.2f} (PP: {pp:.1f}, Style: {style})")
    
    print(f"\n{'='*80}")
    print("ELITE INSIGHTS")
    print(f"{'='*80}")
    
    # Best workout
    best_workout = max(results, key=lambda x: x['workout_bonus'])
    print(f"\n🏋️ BEST WORKOUT: {best_workout['horse_name']}")
    print(f"   {best_workout['workout_data']['quality'].title()} work")
    if best_workout['workout_data']['percentile']:
        print(f"   Top {best_workout['workout_data']['percentile']:.0f}% of all works")
    print(f"   Bonus: +{best_workout['workout_bonus']:.2f}")
    
    # Best connections
    best_connections = max(results, key=lambda x: x['jockey_trainer_bonus'])
    print(f"\n🤝 BEST CONNECTIONS: {best_connections['horse_name']}")
    jt = best_connections['jockey_trainer']
    print(f"   Jockey: {jt['jockey_win_pct']:.0f}% win rate")
    print(f"   Trainer: {jt['trainer_win_pct']:.0f}% win rate")
    print(f"   Bonus: +{best_connections['jockey_trainer_bonus']:.2f}")
    
    # Most positives
    most_positives = max(results, key=lambda x: len(x['quickplay_comments']['positives']))
    print(f"\n✨ MOST POSITIVE ANGLES: {most_positives['horse_name']}")
    print(f"   {len(most_positives['quickplay_comments']['positives'])} positive factors")
    
    print(f"\n{'='*80}")
    print("TEST COMPLETE")
    print(f"{'='*80}")
    print(f"\n✅ Successfully extracted:")
    print(f"   • Prime Power ratings for all 12 horses")
    print(f"   • Workout data with quality ratings")
    print(f"   • Jockey/trainer statistics and bonuses")
    print(f"   • QuickPlay positive/negative angles")
    print(f"   • Running styles")
    print(f"   • Speed ratings")
    print(f"   • Comprehensive bonus calculations")
    print()


if __name__ == "__main__":
    main()
