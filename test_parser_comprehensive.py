#!/usr/bin/env python3
"""
🧪 COMPREHENSIVE PARSER TEST SUITE
Tests 50+ edge cases and scenarios for elite_parser_v2_gold.py

COVERAGE:
1. Format variations (multiline, compressed, typos)
2. Missing data scenarios
3. Edge case odds formats
4. Foreign horses (= prefix)
5. Coupled entries (1A, 1B)
6. Scratched horses
7. Debut horses (no past performances)
8. Irregular spacing/formatting
9. Special characters in names
10. Extreme data values

TARGET: 95%+ parsing success rate across all scenarios
"""

import sys
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple
from elite_parser_v2_gold import GoldStandardBRISNETParser, HorseData
import traceback

# ===================== TEST SCENARIOS =====================

TEST_CASES = {

    # ===== CATEGORY 1: PERFECT FORMAT =====
    "01_perfect_format": """
1 Way of Appeal (S 3)
7/2
BARRIOS RICARDO (254 58-42-39 23%)
Trnr: Cady Khalil (150 18-24-31 12%)
Prime Power: 101.5 (4th)
23Sep23 Mtn Md Sp Wt 16500 98 4th
15Aug23 Mtn Md Sp Wt 16500 92 6th
Sire Stats: AWD 115.2 18% FTS 22% 108.5 spi
""",

    # ===== CATEGORY 2: MISSING DATA =====
    "02_no_odds": """
1 Mystery Horse (E/P 5)
SMITH J (180 40-35-28 22%)
Trnr: Jones T (200 45-38-35 22%)
Prime Power: 95.0 (6th)
""",

    "03_no_jockey": """
2 Riderless (P 4)
5-2
Trnr: Anonymous (100 12-10-8 12%)
10Oct23 Mtn Alw 25000 88 5th
""",

    "04_no_trainer": """
3 No Trainer (E 6)
3-1
DAVIS M (200 50-40-35 25%)
10Oct23 Mtn Alw 25000 90 3rd
""",

    "05_no_speed_figs": """
4 First Timer (E/P 7)
8-1
LOPEZ R (150 35-28-22 23%)
Trnr: Smith B (180 40-35-30 22%)
""",

    # ===== CATEGORY 3: ODDS VARIATIONS =====
    "06_fractional_odds": """
5 Fractional (S 3)
5/2
GARCIA L (200 45-38-32 22%)
Trnr: Brown J (220 50-42-38 23%)
15Sep23 Mtn Clm 18000 92 2nd
""",

    "07_range_odds": """
6 Range Format (E 5)
3-1
MILLER K (180 40-35-28 22%)
Trnr: Davis T (200 45-38-35 22%)
20Sep23 Mtn Alw 22000 95 1st
""",

    "08_decimal_odds": """
7 Decimal Format (P 4)
4.5
RODRIGUEZ A (190 42-36-30 22%)
Trnr: Wilson M (210 48-40-36 23%)
18Sep23 Mtn Md Sp Wt 16500 88 4th
""",

    "09_even_money": """
8 Even Money (E/P 8)
1-1
SANTOS J (250 60-50-45 24%)
Trnr: Martin R (240 58-48-42 24%)
25Sep23 Mtn Alw 28000 102 1st
""",

    "10_longshot": """
9 Longshot (S 2)
30-1
NGUYEN T (120 20-18-15 17%)
Trnr: Lee K (130 22-19-16 17%)
12Sep23 Mtn Clm 12000 78 8th
""",

    # ===== CATEGORY 4: STYLE VARIATIONS =====
    "11_early_speed_strong": """
10 Speed Demon (E 9)
2-1
JOHNSON B (220 52-45-40 24%)
Trnr: Thompson J (230 55-47-42 24%)
Prime Power: 108.5 (1st)
""",

    "12_presser": """
11 Presser Type (E/P 6)
5-1
MARTINEZ C (200 45-38-32 22%)
Trnr: Anderson P (210 48-40-36 23%)
""",

    "13_closer": """
12 Closer Style (S 1)
15-1
THOMAS D (160 32-28-24 20%)
Trnr: White L (170 35-30-26 21%)
""",

    "14_sustained": """
13 Sustained Pace (P 5)
4-1
WILLIAMS E (210 48-40-36 23%)
Trnr: Harris M (220 52-45-40 24%)
""",

    "15_no_style": """
14 Unknown Style (NA 0)
10-1
CLARK R (140 28-24-20 20%)
Trnr: Allen T (150 30-26-22 20%)
""",

    # ===== CATEGORY 5: FOREIGN HORSES =====
    "16_foreign_prefix": """
=1 European Star (E/P 6)
7-2
ROSARIO J (230 55-47-42 24%)
Trnr: McLaughlin K (240 58-48-42 24%)
15Sep23 Wod Alw 30000 95 2nd
""",

    # ===== CATEGORY 6: COUPLED ENTRIES =====
    "17_coupled_entry_a": """
1A First Half (E 6)
5-2
PRAT F (240 58-48-42 24%)
Trnr: Baffert B (280 70-60-55 25%)
""",

    "18_coupled_entry_b": """
1B Second Half (P 4)
5-2
SMITH M (230 55-47-42 24%)
Trnr: Baffert B (280 70-60-55 25%)
""",

    # ===== CATEGORY 7: SCRATCHED HORSES =====
    "19_scratched_scr": """
15 Scratched Horse (E/P 5)
SCR
JONES T (200 45-38-32 22%)
Trnr: Johnson M (210 48-40-36 23%)
""",

    "20_withdrawn": """
16 Withdrawn (E 6)
WDN
DAVIS K (190 42-36-30 22%)
Trnr: Miller R (200 45-38-35 22%)
""",

    # ===== CATEGORY 8: TYPOS & FORMATTING =====
    "21_extra_spaces": """
2    Lucky   Strike    (  E/P   7  )
5-2
SMITH  J  (180  40-35-28  22%)
Trnr:  Jones  T  (200  45-38-35  22%)
""",

    "22_missing_parens": """
3 No Parens E/P 6
3-1
GARCIA M (170 35-30-26 21%)
Trnr: Lopez R (180 40-35-30 22%)
""",

    "23_lowercase": """
4 lowercase name (e/p 5)
4-1
lopez r (150 30-26-22 20%)
trnr: smith b (160 32-28-24 20%)
""",

    "24_special_chars": """
5 O'Brien's Choice (E 7)
6-1
O'NEILL D (210 48-40-36 23%)
Trnr: O'Donnell P (220 52-45-40 24%)
""",

    # ===== CATEGORY 9: COMPREHENSIVE DATA =====
    "25_full_data": """
6 Complete Package (E/P 8)
Own: Smith Racing Stable
3-1 Blue, White Star
CASTELLANO J (280 70-60-55 25%)
Ch. g. 4 (Feb)
Sire : Curlin (Smart Strike) $100,000
Dam: Miss America (Mr Prospector)
Brdr: Claiborne Farm (KY)
Trnr: Pletcher T (350 90-80-75 26%)
Prime Power: 110.5 (1st)
15Oct23 Bel Alw 65000 105 1st
28Sep23 Bel Alw 65000 102 2nd
10Sep23 Sar Stk 100000 108 1st
25Aug23 Sar G3 150000 106 3rd
Sire Stats: AWD 120.5 25% FTS 30% 115.8 spi
Dam's Sire: AWD 118.2 22% FTS 28% 112.5 spi
Dam: DPI 2.1 22%
15Dec23 5f :59.0 b
08Dec23 4f :48.2 H
12 work(s)
2023 1st time str 85 22% 48% +2.5
2023 Blinkers on 42 18% 45% +1.8
2023 JKYw/ E 120 24% 52% +2.2
""",

    # ===== CATEGORY 10: MINIMAL DATA (DEBUT) =====
    "26_debut_horse": """
7 First Timer (E/P 6)
12-1
ROSARIO J (230 55-47-42 24%)
Trnr: McLaughlin K (240 58-48-42 24%)
B. c. 2 (Apr)
Sire : Into Mischief (Harlan's Holiday)
Dam: Lucky Lady (Lucky Pulpit)
Sire Stats: AWD 112.5 28% FTS 32% 118.5 spi
10Dec23 4f :48.5 Bg
""",

    # ===== CATEGORY 11: MULTIPLE HORSES (RACE SCENARIO) =====
    "27_three_horse_race": """
1 Speed King (E 8)
2-1
SMITH J (200 45-38-32 22%)
Trnr: Jones T (210 48-40-36 23%)
Prime Power: 105.0 (2nd)
20Sep23 Mtn Alw 25000 100 2nd

2 Steady Eddie (P 5)
3-1
GARCIA M (190 42-36-30 22%)
Trnr: Lopez R (200 45-38-35 22%)
Prime Power: 102.5 (3rd)
18Sep23 Mtn Alw 25000 98 3rd

3 Late Charge (S 3)
5-1
MARTINEZ C (180 40-35-28 22%)
Trnr: Davis K (190 42-36-30 22%)
Prime Power: 98.0 (5th)
15Sep23 Mtn Clm 18000 95 4th
""",

    # ===== CATEGORY 12: EXTREME VALUES =====
    "28_very_high_beyer": """
8 Champion Class (E/P 9)
1-2
ORTIZ I (270 68-58-52 25%)
Trnr: Brown C (320 82-72-68 26%)
25Oct23 Bel G1 500000 118 1st
Prime Power: 125.5 (1st)
""",

    "29_very_low_beyer": """
9 Slow Starter (S 1)
50-1
UNNAMED RIDER (50 8-6-5 16%)
Trnr: Unknown (60 10-8-6 17%)
10Sep23 Mtn Clm 8000 45 12th
""",

    "30_zero_quirin": """
10 No Early (S 0)
20-1
DAVIS M (150 30-26-22 20%)
Trnr: Wilson R (160 32-28-24 20%)
""",

    # ===== ADDITIONAL 20 EDGE CASES =====

    "31_multiline_name": """
11 Very Long Horse
Name Here (E 5)
8-1
LOPEZ A (170 35-30-26 21%)
Trnr: Smith J (180 40-35-30 22%)
""",

    "32_no_prime_power": """
12 No Prime (P 4)
6-1
JOHNSON K (200 45-38-32 22%)
Trnr: Brown M (210 48-40-36 23%)
15Sep23 Mtn Alw 22000 92 5th
""",

    "33_multiple_angles": """
13 Angle Stacker (E/P 7)
4-1
GARCIA J (210 48-40-36 23%)
Trnr: Martinez R (220 52-45-40 24%)
2023 1st time str 45 18% 42% +1.2
2023 Blinkers on 38 16% 40% +0.8
2023 JKYw/ E 52 20% 45% +1.5
2023 30daysAway 28 15% 38% +0.5
""",

    "34_no_angles": """
14 Angle Free (S 2)
15-1
THOMAS P (140 28-24-20 20%)
Trnr: Clark L (150 30-26-22 20%)
""",

    "35_many_races": """
15 Busy Horse (E 6)
5-1
WILLIAMS M (200 45-38-32 22%)
Trnr: Harris T (210 48-40-36 23%)
20Oct23 Mtn Alw 25000 95 2nd
10Oct23 Mtn Alw 25000 93 3rd
28Sep23 Mtn Clm 18000 90 1st
15Sep23 Mtn Clm 18000 88 2nd
05Sep23 Mtn Clm 16000 86 4th
25Aug23 Mtn Clm 16000 84 5th
15Aug23 Mtn Md Sp Wt 15000 82 6th
""",

    "36_recent_race": """
16 Just Ran (E/P 5)
7-2
RODRIGUEZ M (190 42-36-30 22%)
Trnr: Anderson K (200 45-38-35 22%)
26Jan26 Mtn Alw 24000 98 3rd
""",

    "37_long_layof": """
17 Long Break (P 3)
12-1
NGUYEN K (130 22-19-16 17%)
Trnr: Lee M (140 28-24-20 20%)
15Mar23 Mtn Clm 14000 85 8th
""",

    "38_consistent_finishes": """
18 Mr Reliable (E/P 6)
5-2
SANTOS M (220 52-45-40 24%)
Trnr: Martin K (230 55-47-42 24%)
18Oct23 Mtn Alw 26000 100 2nd
05Oct23 Mtn Alw 26000 98 2nd
22Sep23 Mtn Alw 26000 99 1st
10Sep23 Mtn Alw 26000 97 2nd
""",

    "39_erratic_finishes": """
19 Up And Down (S 2)
10-1
CLARK M (160 32-28-24 20%)
Trnr: Allen R (170 35-30-26 21%)
15Oct23 Mtn Clm 18000 92 1st
02Oct23 Mtn Clm 18000 78 10th
18Sep23 Mtn Clm 18000 88 3rd
05Sep23 Mtn Clm 18000 72 12th
""",

    "40_high_class": """
20 Graded Stakes (E 7)
2-1
ORTIZ J (260 65-55-50 25%)
Trnr: Pletcher T (350 90-80-75 26%)
25Oct23 Bel G2 300000 110 2nd
10Oct23 Sar G3 200000 108 1st
Sire Stats: AWD 122.5 26% FTS 31% 118.9 spi
""",

    "41_low_class": """
21 Bottom Claimer (S 1)
30-1
JONES K (100 18-15-12 18%)
Trnr: Smith R (110 20-17-14 18%)
12Oct23 Mtn Clm 5000 68 9th
""",

    "42_turf_horse": """
22 Turf Specialist (P 4)
6-1
PRAT F (240 58-48-42 24%)
Trnr: Brown C (320 82-72-68 26%)
18Oct23 Bel Alw T 65000 95 1st
""",

    "43_shipper": """
23 Cross Country (E/P 6)
8-1
ROSARIO J (230 55-47-42 24%)
Trnr: McLaughlin K (240 58-48-42 24%)
20Oct23 SA Alw 75000 102 2nd
2023 Shipper 32 18% 45% +1.5
""",

    "44_blinkers_on": """
24 New Equipment (E 5)
7-1
CASTELLANO J (280 70-60-55 25%)
Trnr: Pletcher T (350 90-80-75 26%)
15Oct23 Bel Alw 65000 90 5th
2023 Blinkers on 48 20% 46% +1.8
""",

    "45_jockey_switch": """
25 New Rider (E/P 7)
4-1
ORTIZ I (270 68-58-52 25%)
Trnr: Brown C (320 82-72-68 26%)
10Oct23 Bel Alw 65000 95 3rd
""",

    "46_trainer_switch": """
26 New Barn (P 5)
9-1
DAVIS J (190 42-36-30 22%)
Trnr: Pletcher T (350 90-80-75 26%)
25Sep23 Mtn Alw 24000 88 6th
""",

    "47_first_lasix": """
27 First Lasix (E 6)
5-1
SMITH M (200 45-38-32 22%)
Trnr: Jones T (210 48-40-36 23%)
12Oct23 Mtn Alw 26000 92 4th
""",

    "48_surface_switch": """
28 Turf to Dirt (E/P 5)
10-1
GARCIA L (180 40-35-28 22%)
Trnr: Lopez M (190 42-36-30 22%)
20Oct23 Mtn Alw T 28000 85 7th
2023 Turf to Dirt 25 16% 40% +0.5
""",

    "49_distance_change": """
29 Stretch Out (S 3)
12-1
MARTINEZ J (170 35-30-26 21%)
Trnr: Davis R (180 40-35-30 22%)
15Oct23 Mtn Alw 6f 22000 88 5th
""",

    "50_maiden_special": """
30 Maiden Debut (E/P 6)
15-1
WILLIAMS K (210 48-40-36 23%)
Trnr: Harris M (220 52-45-40 24%)
B. f. 2 (May)
Sire : Quality Road (Elusive Quality)
Dam: Perfect Princess (Pulpit)
Sire Stats: AWD 116.5 24% FTS 29% 114.2 spi
15Dec23 4f :48.8 Bg
2024 Debut MdnSpWt 180 12% 28% -0.2
"""
}

# ===================== TEST HARNESS =====================

def run_comprehensive_tests(verbose: bool = False) -> Dict:
    """
    Run all test cases and generate detailed report.

    Returns:
        {
            'total_tests': int,
            'passed': int,
            'failed': int,
            'success_rate': float,
            'details': [list of test results]
        }
    """
    parser = GoldStandardBRISNETParser()
    results = []

    print("\n" + "="*80)
    print("🧪 COMPREHENSIVE PARSER TEST SUITE")
    print("="*80)
    print(f"Running {len(TEST_CASES)} test scenarios...\n")

    passed = 0
    failed = 0

    for test_name, pp_text in TEST_CASES.items():
        try:
            # Parse
            horses = parser.parse_full_pp(pp_text, debug=False)

            # Basic validation
            if horses:
                # Success criteria
                avg_confidence = np.mean([h.parsing_confidence for h in horses.values()])

                status = "PASS" if avg_confidence >= 0.5 else "WARN"
                if avg_confidence >= 0.5:
                    passed += 1
                else:
                    failed += 1

                result = {
                    'test': test_name,
                    'status': status,
                    'horses_found': len(horses),
                    'avg_confidence': avg_confidence,
                    'details': list(horses.keys())
                }

                if verbose or status == "WARN":
                    print(f"{'✓' if status == 'PASS' else '⚠'} {test_name}: {len(horses)} horses, {avg_confidence:.1%} confidence")
                    if status == "WARN":
                        for name, horse in horses.items():
                            if horse.warnings:
                                print(f"    Warnings for {name}: {', '.join(horse.warnings[:2])}")
            else:
                failed += 1
                result = {
                    'test': test_name,
                    'status': "FAIL",
                    'horses_found': 0,
                    'avg_confidence': 0.0,
                    'details': "No horses parsed"
                }
                print(f"✗ {test_name}: FAILED - No horses parsed")

            results.append(result)

        except Exception as e:
            failed += 1
            result = {
                'test': test_name,
                'status': "ERROR",
                'horses_found': 0,
                'avg_confidence': 0.0,
                'details': str(e)
            }
            results.append(result)
            print(f"✗ {test_name}: ERROR - {str(e)}")
            if verbose:
                traceback.print_exc()

    # Summary
    total = len(TEST_CASES)
    success_rate = (passed / total) * 100

    print(f"\n{'='*80}")
    print("TEST SUMMARY")
    print(f"{'='*80}")
    print(f"Total Tests: {total}")
    print(f"Passed: {passed} ({(passed/total)*100:.1f}%)")
    print(f"Failed: {failed} ({(failed/total)*100:.1f}%)")
    print(f"Success Rate: {success_rate:.1%}")

    # Category breakdown
    print(f"\n{'='*80}")
    print("CATEGORY BREAKDOWN")
    print(f"{'='*80}")

    categories = {
        '01-05': 'Perfect & Missing Data',
        '06-10': 'Odds Variations',
        '11-15': 'Style Variations',
        '16-18': 'Foreign & Coupled',
        '19-20': 'Scratched',
        '21-24': 'Typos & Formatting',
        '25-26': 'Full & Debut',
        '27': 'Multi-horse Race',
        '28-30': 'Extreme Values',
        '31-50': 'Additional Edge Cases'
    }

    for range_str, category_name in categories.items():
        if '-' in range_str:
            start, end = map(int, range_str.split('-'))
            category_results = [r for r in results if int(r['test'].split('_')[0]) in range(start, end+1)]
        else:
            num = int(range_str)
            category_results = [r for r in results if int(r['test'].split('_')[0]) == num]

        if category_results:
            category_passed = sum(1 for r in category_results if r['status'] == 'PASS')
            category_total = len(category_results)
            print(f"{category_name}: {category_passed}/{category_total} ({(category_passed/category_total)*100:.0f}%)")

    # Grade
    print(f"\n{'='*80}")
    print("OVERALL GRADE")
    print(f"{'='*80}")
    if success_rate >= 95:
        grade = "A+ (ELITE)"
    elif success_rate >= 90:
        grade = "A (EXCELLENT)"
    elif success_rate >= 85:
        grade = "B+ (VERY GOOD)"
    elif success_rate >= 80:
        grade = "B (GOOD)"
    elif success_rate >= 75:
        grade = "C+ (ACCEPTABLE)"
    else:
        grade = "C (NEEDS IMPROVEMENT)"

    print(f"Grade: {grade}")
    print("Target: A (90%+) for production deployment")

    if success_rate < 90:
        print("\n⚠️ RECOMMENDATION: Review failed cases and enhance parser before production")
    else:
        print("\n✅ PARSER IS PRODUCTION-READY")

    return {
        'total_tests': total,
        'passed': passed,
        'failed': failed,
        'success_rate': success_rate,
        'details': results
    }


def test_specific_scenario(scenario_name: str):
    """Test a specific scenario with full debug output"""
    if scenario_name not in TEST_CASES:
        print(f"❌ Scenario '{scenario_name}' not found")
        print(f"Available scenarios: {', '.join(TEST_CASES.keys())}")
        return

    print(f"\n{'='*80}")
    print(f"TESTING: {scenario_name}")
    print(f"{'='*80}")

    pp_text = TEST_CASES[scenario_name]
    print("\nINPUT PP TEXT:")
    print("-" * 60)
    print(pp_text)
    print("-" * 60)

    parser = GoldStandardBRISNETParser()
    horses = parser.parse_full_pp(pp_text, debug=True)

    if horses:
        print(f"\n{'='*80}")
        print("PARSED OUTPUT:")
        print(f"{'='*80}")

        for name, horse in horses.items():
            print(f"\n🐎 {name}")
            print(f"   Post: {horse.post}")
            print(f"   Style: {horse.pace_style} (Q={horse.quirin_points:.0f}, {horse.style_strength})")
            print(f"   Odds: {horse.ml_odds} → {horse.ml_odds_decimal} (conf: {horse.odds_confidence:.1%})")
            print(f"   Jockey: {horse.jockey} ({horse.jockey_win_pct:.1%}, conf: {horse.jockey_confidence:.1%})")
            print(f"   Trainer: {horse.trainer} ({horse.trainer_win_pct:.1%}, conf: {horse.trainer_confidence:.1%})")
            print(f"   Speed: {horse.speed_figures} (avg: {horse.avg_top2:.1f}, conf: {horse.speed_confidence:.1%})")
            print(f"   Form: {horse.days_since_last} days (conf: {horse.form_confidence:.1%})")
            print(f"   Overall Confidence: {horse.parsing_confidence:.1%}")

            if horse.warnings:
                print(f"   ⚠️ Warnings: {', '.join(horse.warnings)}")
            if horse.errors:
                print(f"   ❌ Errors: {', '.join(horse.errors)}")
    else:
        print("\n❌ NO HORSES PARSED")


def export_test_results_to_csv(results: Dict, output_file: str = "parser_test_results.csv"):
    """Export test results to CSV for analysis"""
    df = pd.DataFrame(results['details'])
    df.to_csv(output_file, index=False)
    print(f"\n📊 Results exported to: {output_file}")


# ===================== MAIN EXECUTION =====================

if __name__ == "__main__":
    import argparse

    parser_args = argparse.ArgumentParser(description="Comprehensive BRISNET Parser Test Suite")
    parser_args.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    parser_args.add_argument("--test", "-t", type=str, help="Test specific scenario (e.g., '01_perfect_format')")
    parser_args.add_argument("--export", "-e", action="store_true", help="Export results to CSV")

    args = parser_args.parse_args()

    if args.test:
        # Test specific scenario
        test_specific_scenario(args.test)
    else:
        # Run all tests
        results = run_comprehensive_tests(verbose=args.verbose)

        if args.export:
            export_test_results_to_csv(results)

        # Exit with appropriate code
        if results['success_rate'] >= 90:
            sys.exit(0)  # Success
        else:
            sys.exit(1)  # Failure
