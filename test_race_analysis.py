"""
Comprehensive analysis of Mountaineer Race 8 (Nov 17, 2025)
Testing all handicapping angles for accuracy
"""

import re
import numpy as np
from typing import Dict, List, Tuple

# Sample PP data for each horse
HORSES = {
    "Ice Floe": {
        "pp_text": """1 Ice Floe (E/P 3)
Prime Power: 101.5 (4th) Life: 45 9 - 6 - 9 $104,789 80
Trnr: Cluley Denis (120 23-17-14 19%)
Sire Stats: AWD 7.5 13%Mud 1450MudSts 1.31spi
Dam'sSire: AWD 6.1 18%Mud 547MudSts 1.09spi
ñ High % trainer ñ Rail post is winning at 21% clip ñ Drops in class today × Has not raced in 57 days × Poor Speed Figures in last 2 starts
21Sep25Mnr® 5½ ft :22© :46« :59« 1:06« ¦ ¨§¯ ™C4000 ¨§® 87 72/ 74 +5 +1 56 4 6 6ªƒ 8 8¨§ 6¨§ GonzalezK¨©ª Lb 12.60
09Sep25Mnr« 6f ft :22ª :46© :59ª 1:13¨ ¦ ¨§¯ ™C4000n2y ¨§ 90 85/ 62 +2 0 59 2 2 2¨ 3© 4ª 5¯ƒ BarriosR¨©ª Lb 2.10
11Aug25Mnr 5f ft :22« :46ª 1:00ª ¦ ¨§¯ ™C4000n2x-c ¨§¯ 83/ 77 68 7 6 5« 4¬ 5® 4ª BrachoA¨©§ Lb 2.10
28Oct Mnr 4f ft :47ª B
07Apr Mnr (d) 4f my :49« B""",
        "expected": {
            "prime": 101.5,
            "trainer_win": 19.0,
            "dam_sire_mud": 18,
            "sire_mud": 13,
            "class_drop": True,
            "lp_count": 3,
            "bullet_count": 2
        }
    },
    
    "Del Rey Dolly": {
        "pp_text": """2 Del Rey Dolly (S 0)
Prime Power: 101.4 (5th) Life: 47 13 - 5 - 6 $210,250 86
Trnr: Ewell, Sr. Devan (49 1-6-10 2%)
Sire Stats: AWD 7.0 16%Mud 575MudSts 0.84spi
Dam'sSire: AWD 7.2 20%Mud 1914MudSts 2.22spi
ñ Drops in class today ñ Highest speed figure at today's distance
05Oct25Mnr¬ 6f ft :23 :46© :58« 1:12¨ ¦ ¨§° ™C4000 ¨§ 78 68/ 79 -4 -5 59 1 5 6« 6¯ƒ 6¯‚ 5¨§ SimpsonJ¨© Lb 8.60
09Sep25Mnr« 6f ft :22ª :46© :59ª 1:13¨ ¦ ¨§¯ ™C4000n2y ¨¨¨ 81 81/ 84 +2 0 74 3 3 5¬ 4« 3ª 1¨ OliverosC¨©ª Lb 5.80
01Aug25CT© 7f sys :23ª :47«1:13¨ 1:26« ¦ ¨§° ™C5000b ¨§® 76 72/ 61 -3 -5 54 5 3 5¬ 7 6¨ª 5¨« DiazSJ¨©© Lbf 20.10
25Apr CT 4f ft :49© B""",
        "expected": {
            "prime": 101.4,
            "trainer_win": 2.0,
            "dam_sire_mud": 20,
            "class_drop": True,
            "lp_count": 3,
            "jockey_switch": True  # Multiple jockeys: SimpsonJ, OliverosC, DiazSJ
        }
    },
    
    "Shewearsmyring": {
        "pp_text": """3 Shewearsmyring (S 0)
Prime Power: 98.7 (9th) Life: 32 7 - 3 - 5 $125,948 79
Trnr: Poole Jami C (201 26-30-21 13%)
Sire Stats: AWD 6.9 11%Mud 1066MudSts 0.79spi
Dam'sSire: AWD 6.6 18%Mud 1790MudSts 1.62spi
ñ Drops in class today × Showed declining form and speed in last start
06Oct25Mnr¨ 1m ft :24« :48©1:14 1:41« ¦ ¨§® ™C5000n2y ¨§¬ 63 56/ 73 +1 +1 59 4 3ª 5¯ 6® 6¨ª 6¨§ GomezA¨©ª L 26.60
31Aug25Mnr 6f ft :23¨ :47¨1:00¨ 1:14 ¦ ¨§® ™C4000n2y ¨§¯ 81 83/ 81 -1 -1 73 5 3 4« 4© 4¨ƒ 1³ GomezA¨©§ L 25.80
29Jly25Mnrª à 7f fm :22« :45©1:10© 1:22ª ¦ ¨§¯ ™C4000n1L ¨§« 67 51/ 80 +12 +9 54 1 4 6¨¯ 8©ª 8©§ 8¨¬ AvilesY¨©§ L 56.70
01Apr Mnr 5f ft 1:03© Bg""",
        "expected": {
            "prime": 98.7,
            "trainer_win": 13.0,
            "class_drop": True,
            "lp_count": 3,
            "bullet_count": 1  # "Bg" = B
        }
    },
    
    "Love of Grace": {
        "pp_text": """4 Love of Grace (E/P 4)
Prime Power: 101.9 (3rd) Life: 25 6 - 2 - 3 $87,299 78
Trnr: Collins Timothy M (10 2-0-2 20%)
Sire Stats: AWD 6.0 16%Mud 892MudSts 0.77spi
Dam'sSire: AWD 7.4 17%Mud 1054MudSts 1.38spi
× Showed declining form and speed in last start
08Oct25CTª 7f ft :23ª :48¨1:14© 1:28« ¦ ¨§ ™'Clm5000/4.5b¨§ª 81 77/ 44 -3 -2 46 2 4 3© 5ª 6 7¨¬ HoW¨©§ L 11.00
03Sep25Mnr 5½ ft :23 :47©1:00© 1:07© ¦ ¨§® ™C4000n2x ¨¨¨ 87 82/ 77 0 -2 67 4 1 2² 2² 1² 1ª OliverosC¨©ª L 3.30
15Aug25CT© 4½ ft :22 :46 :52© ¦ ¨§¯ ™'Clm5000b ¨§¬ 64/ 80 50 6 5 8¨ª - 8¨© 7¨¨ PeltrocheF¨©© L 10.10
05Aug CT 4f ft :51ª B""",
        "expected": {
            "prime": 101.9,
            "trainer_win": 20.0,
            "dam_sire_mud": 17,
            "lp_count": 3,
            "bullet_count": 1
        }
    },
    
    "Banned From Midway": {
        "pp_text": """8 Banned From Midway (E 6)
Prime Power: 106.1 (1st) Life: 46 10 - 6 - 4 $114,997 77
Trnr: Fletcher Wes (0 0-0-0 0%)
Sire Stats: AWD 6.7 14%Mud 195MudSts 0.58spi
Dam'sSire: AWD 7.1 14%Mud 2496MudSts 1.20spi
ñ Switches to a high % jockey ñ Highest last race speed rating (tie) ñ Drops in class today
ñ Ran 2nd vs tougher in last race × Has not raced for more than 2 months × Poor record at this track
02Sep25Btp« 6f ft :23¨ :47 :59ª 1:12ª ¦ ¨§® ™C5000n2y ¨§¯ 83 80/ 78 -5 -5 68 1 4 3¨ 4ª 2¨ 2¬‚ CorreaYL¨©ª Lbf 2.60
30Jly25Btp© 6f ft :22© :46ª :59« 1:13ª ¦ ¨§° ™C5000b-c ¨§° 92 91/ 68 +7 +5 69 4 1 2© 2¨ 1 3ª‚ FigueroaS¨©¬ Lb *0.70
11Jly25Btp 6f ft :22ª :46ª :59© 1:12« ¦ ¨§° ™C5000n1y ¨¨¨ 91 91/ 76 +1 +1 76 3 2 2 2² 1© 1ª PilaresCP¨©¬ Lb *1.80
30Sep Btp 4f ft :49© B""",
        "expected": {
            "prime": 106.1,
            "trainer_win": 0.0,  # First-time trainer
            "class_drop": True,
            "jockey_switch": True,  # Switches to Negron (high %)
            "lp_count": 3,
            "layoff_days": 76,  # Sep 2 to Nov 17
            "bullet_count": 1
        }
    },
    
    "Sweet Talia": {
        "pp_text": """7 Sweet Talia (E/P 4)
Prime Power: 104.4 (2nd) Life: 57 6 -12 - 8 $180,439 80
Trnr: Scallan Robert S (75 10-17-12 13%)
Sire Stats: AWD 6.8 16%Mud 352MudSts 0.77spi
Dam'sSire: AWD 7.6 16%Mud 2023MudSts 3.40spi
ñ Highest last race speed rating (tie) ñ Ran 2nd vs similar in last race
02Nov25Mnrª 6f ft :22« :46« :59© 1:12© ¦ ¨§¯ ™C4000n2y ¨¨§ 89 84/ 74 -1 -7 68 3 1 2² 2² 2ª 2 WeatherlyB¨©ª Lb 14.60
15Sep25Mnr® 1m ft :24© :48©1:14¨ 1:42ª ¦ ¨§¯ ™A18000n1x ¨§¬ 85 75/ 55 +8 +7 59 2 4© 4©‚ 4¬‚ 4¯ 4¨¬ VilchezM¨¨¯* Lb 6.50
02Sep25Mnr 6f ft :22 :46 :59© 1:13© ¦ ¨§¯ ™C5000n2y ¨§° 85 82/ 77 +7 +5 69 1 5 3 4¬ 4¬ 3© WeatherlyB¨©ª Lb 6.40
27Oct Mnr 3f ft :36¨ B""",
        "expected": {
            "prime": 104.4,
            "trainer_win": 13.0,
            "dam_sire_mud": 16,
            "lp_count": 3,
            "bullet_count": 1,
            "ran_2nd": True
        }
    }
}

def parse_prime_power(text: str) -> float:
    """Extract Prime Power rating"""
    m = re.search(r"Prime Power:\s*(\d+\.?\d*)", text)
    return float(m.group(1)) if m else np.nan

def parse_trainer_win_pct(text: str) -> float:
    """Extract trainer win percentage"""
    m = re.search(r"Trnr:.*?\([^\)]+\s+(\d+)%\)", text)
    return float(m.group(1)) if m else 0.0

def parse_dam_sire_mud(text: str) -> tuple:
    """Extract Dam's Sire AWD and Mud %"""
    m = re.search(r"Dam'sSire:\s*AWD\s*([\d.]+)\s*(\d+)%Mud", text)
    if m:
        return (float(m.group(2)), float(m.group(1)))  # (mud%, awd)
    return (0, 0)

def parse_sire_mud(text: str) -> float:
    """Extract Sire mud %"""
    m = re.search(r"Sire Stats:\s*AWD\s*[\d.]+\s*(\d+)%Mud", text)
    return float(m.group(1)) if m else 0.0

def parse_class_drop(text: str) -> bool:
    """Detect class drop from QuickPlay comments"""
    return bool(re.search(r"ñ.*drops?\s+in\s+class", text, re.I))

def parse_lp_values(text: str) -> List[int]:
    """Extract Late Pace values from E1 E2/ LP pattern"""
    lp_values = []
    for m in re.finditer(r"(?m)^\d{2}[A-Za-z]{3}\d{2}.*?\s+(\d{2,3})\s+(\d{2,3})/\s*(\d{2,3})", text):
        lp_values.append(int(m.group(3)))
    return lp_values

def parse_bullets(text: str) -> int:
    """Count bullet workouts (marked with B or Bg)"""
    return len(re.findall(r"(?m)^\d{2}[A-Za-z]{3}.*?\sB(?:g|\b)", text))

def parse_jockey_switch(text: str) -> bool:
    """Detect jockey changes in recent races"""
    jockeys = re.findall(r"(?m)^\d{2}[A-Za-z]{3}\d{2}.*?([A-Z][a-z]+[A-Z]?\d*[ª©§¨]*)\s+(?:Lb?f?|L)\s+", text)
    if len(jockeys) >= 2:
        return jockeys[0] != jockeys[1]  # Compare most recent two
    return False

def parse_e1_e2(text: str) -> List[Tuple[int, int, int]]:
    """Extract E1, E2, and LP from race lines"""
    e1_e2_lp = []
    for m in re.finditer(r"(?m)^\d{2}[A-Za-z]{3}\d{2}.*?\s+(\d{2,3})\s+(\d{2,3})/\s*(\d{2,3})", text):
        e1_e2_lp.append((int(m.group(1)), int(m.group(2)), int(m.group(3))))
    return e1_e2_lp

def parse_fractional_positions(text: str) -> List[Tuple[int, int]]:
    """Extract 1C and 2C fractional call positions"""
    fracs = []
    # Pattern: after SPD and PP columns, look for position pairs like "6ªƒ 8"
    for m in re.finditer(r"(?m)^\d{2}[A-Za-z]{3}\d{2}.*?\s+\d+\s+\d+\s+\d+\s+(\d+)[ªƒ²³¨«¬©°±´‚]*\s+(\d+)[ªƒ²³¨«¬©°±´‚]*", text):
        try:
            fracs.append((int(m.group(1)), int(m.group(2))))
        except:
            pass
    return fracs

def analyze_horse(name: str, data: dict) -> dict:
    """Comprehensive analysis of one horse"""
    text = data["pp_text"]
    expected = data["expected"]
    
    results = {
        "name": name,
        "tests_passed": 0,
        "tests_failed": 0,
        "details": []
    }
    
    # Test Prime Power
    prime = parse_prime_power(text)
    if abs(prime - expected["prime"]) < 0.01:
        results["tests_passed"] += 1
        results["details"].append(f"✓ Prime Power: {prime} (expected {expected['prime']})")
    else:
        results["tests_failed"] += 1
        results["details"].append(f"✗ Prime Power: {prime} (expected {expected['prime']})")
    
    # Test Trainer Win %
    trainer_win = parse_trainer_win_pct(text)
    if abs(trainer_win - expected["trainer_win"]) < 0.01:
        results["tests_passed"] += 1
        results["details"].append(f"✓ Trainer Win %: {trainer_win}% (expected {expected['trainer_win']}%)")
    else:
        results["tests_failed"] += 1
        results["details"].append(f"✗ Trainer Win %: {trainer_win}% (expected {expected['trainer_win']}%)")
    
    # Test Dam's Sire Mud
    if "dam_sire_mud" in expected:
        dam_sire = parse_dam_sire_mud(text)
        if dam_sire[0] == expected["dam_sire_mud"]:
            results["tests_passed"] += 1
            results["details"].append(f"✓ Dam's Sire Mud: {dam_sire[0]}% (expected {expected['dam_sire_mud']}%)")
        else:
            results["tests_failed"] += 1
            results["details"].append(f"✗ Dam's Sire Mud: {dam_sire[0]}% (expected {expected['dam_sire_mud']}%)")
    
    # Test Class Drop detection
    if "class_drop" in expected:
        class_drop = parse_class_drop(text)
        if class_drop == expected["class_drop"]:
            results["tests_passed"] += 1
            results["details"].append(f"✓ Class Drop: {class_drop} (expected {expected['class_drop']})")
        else:
            results["tests_failed"] += 1
            results["details"].append(f"✗ Class Drop: {class_drop} (expected {expected['class_drop']})")
    
    # Test LP value extraction
    if "lp_count" in expected:
        lp_values = parse_lp_values(text)
        if len(lp_values) >= expected["lp_count"]:
            results["tests_passed"] += 1
            results["details"].append(f"✓ LP Values: {len(lp_values)} extracted (expected >={expected['lp_count']}): {lp_values[:5]}")
        else:
            results["tests_failed"] += 1
            results["details"].append(f"✗ LP Values: {len(lp_values)} extracted (expected >={expected['lp_count']}): {lp_values}")
    
    # Test Bullet workouts
    if "bullet_count" in expected:
        bullets = parse_bullets(text)
        if bullets >= expected["bullet_count"]:
            results["tests_passed"] += 1
            results["details"].append(f"✓ Bullet Works: {bullets} (expected >={expected['bullet_count']})")
        else:
            results["tests_failed"] += 1
            results["details"].append(f"✗ Bullet Works: {bullets} (expected >={expected['bullet_count']})")
    
    # Test Jockey Switch
    if "jockey_switch" in expected:
        jockey_switch = parse_jockey_switch(text)
        if jockey_switch == expected["jockey_switch"]:
            results["tests_passed"] += 1
            results["details"].append(f"✓ Jockey Switch: {jockey_switch} (expected {expected['jockey_switch']})")
        else:
            results["tests_failed"] += 1
            results["details"].append(f"✗ Jockey Switch: {jockey_switch} (expected {expected['jockey_switch']})")
    
    # Test E1/E2/LP extraction
    e1_e2_lp = parse_e1_e2(text)
    results["details"].append(f"  E1/E2/LP values: {e1_e2_lp[:3]}")
    
    # Test Fractional Positions
    fracs = parse_fractional_positions(text)
    results["details"].append(f"  Fractional Positions (1C, 2C): {fracs[:3]}")
    
    return results

def main():
    """Run comprehensive analysis on all horses"""
    print("="*80)
    print("COMPREHENSIVE RACE ANALYSIS - MOUNTAINEER RACE 8 (Nov 17, 2025)")
    print("Testing All Handicapping Angles for Accuracy")
    print("="*80)
    print()
    
    total_passed = 0
    total_failed = 0
    
    for horse_name in ["Ice Floe", "Del Rey Dolly", "Shewearsmyring", "Love of Grace", "Banned From Midway", "Sweet Talia"]:
        if horse_name in HORSES:
            print(f"\n{'='*80}")
            print(f"HORSE: {horse_name}")
            print('='*80)
            
            results = analyze_horse(horse_name, HORSES[horse_name])
            
            for detail in results["details"]:
                print(detail)
            
            total_passed += results["tests_passed"]
            total_failed += results["tests_failed"]
            
            print(f"\nSummary: {results['tests_passed']} passed, {results['tests_failed']} failed")
    
    print("\n" + "="*80)
    print("OVERALL RESULTS")
    print("="*80)
    print(f"Total Tests Passed: {total_passed}")
    print(f"Total Tests Failed: {total_failed}")
    print(f"Success Rate: {total_passed/(total_passed+total_failed)*100:.1f}%")
    print("="*80)
    
    # Additional Analysis: Race-level patterns
    print("\n" + "="*80)
    print("RACE-LEVEL ANALYSIS")
    print("="*80)
    
    prime_powers = []
    for horse_name, data in HORSES.items():
        prime = parse_prime_power(data["pp_text"])
        prime_powers.append((horse_name, prime))
    
    prime_powers.sort(key=lambda x: x[1], reverse=True)
    print("\nPrime Power Rankings:")
    for i, (name, prime) in enumerate(prime_powers, 1):
        print(f"  {i}. {name}: {prime}")
    
    # Class drop analysis
    print("\nClass Drop Horses:")
    for horse_name, data in HORSES.items():
        if parse_class_drop(data["pp_text"]):
            print(f"  - {horse_name}")
    
    # Trainer Intent signals
    print("\nHigh-Win % Trainers (>15%):")
    for horse_name, data in HORSES.items():
        trainer_win = parse_trainer_win_pct(data["pp_text"])
        if trainer_win > 15:
            print(f"  - {horse_name}: {trainer_win}%")
    
    print("\n" + "="*80)

if __name__ == "__main__":
    main()
