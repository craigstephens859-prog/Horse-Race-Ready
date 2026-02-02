"""
SA R8 Results Analysis - Validate New Weight Scheme (Class 3.0×, Speed 1.8×)
============================================================================

Actual Results: 13, 12, 8, 5, 9

System Prediction (with NEW weights):
1. #5 Clubhouse Bride - 8.18 rating → Finished 4th ❌
2. #7 Ryan's Girl - 4.78 rating → Did not finish in top 5
3. #1 Clarina - 3.35 rating → Did not finish in top 5
4. #8 Stay in Line - 2.49 rating → Finished 3rd ✓ (close!)

Winner: #13 Rizzleberry Rose - NOT in system's top picks ❌
2nd: #12 Miss Practical - NOT in system's top picks ❌

Key Observation from BRISNET Data:
- #12 Miss Practical: Prime Power 127.5 (1st) → Finished 2nd ✓
- #8 Stay in Line: Prime Power 125.4 (2nd) → Finished 3rd ✓
- #13 Rizzleberry Rose: Prime Power 125.3 (3rd) → WON ✓

Top 3 finishers = Top 3 Prime Power horses!

This suggests:
1. Prime Power (comprehensive metric) predicted correctly
2. System's component-based approach with new weights missed the mark
3. May need to analyze WHY #5 was overrated and #13/#12 were underrated
"""

import pandas as pd
import numpy as np

# SA R8 Horse Data from BRISNET
horses = {
    'num': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13],
    'name': ['Clarina', 'Timekeeper\'s Charm', 'Lavender Love', 'Fibonaccis Ride', 
             'Clubhouse Bride', 'Clubhouse Cutie', 'Ryan\'s Girl', 'Stay in Line',
             'Maniae', 'Petite Treat', 'Sexy Blue', 'Miss Practical', 'Rizzleberry Rose'],
    'speed_last': [91, 78, 79, 77, 77, 81, 80, 73, 65, 76, 72, 82, 77],
    'speed_best_dist': [91, 89, 82, 84, 88, 81, 88, 93, 83, 82, 87, 89, 83],
    'prime_power': [117.7, 124.4, 123.5, 120.4, 122.3, 121.0, 118.1, 125.4, 114.6, 116.6, 119.0, 127.5, 125.3],
    'class_rating': [113.3, 114.4, 114.6, 112.5, 113.6, 112.5, 112.3, 113.9, 111.3, 112.5, 112.8, 114.8, 113.5],
    'e1_pace': [99, 95, 78, 93, 88, 94, 88, 87, 87, 88, 84, 85, 92],
    'e2_late': [87, 73, 85, 71, 79, 80, 73, 75, 70, 79, 77, 88, 74],
    'actual_finish': [99, 99, 99, 99, 4, 99, 99, 3, 5, 99, 99, 2, 1]  # 99 = DNF top 5
}

df = pd.DataFrame(horses)

# Actual finishing order
ACTUAL = [13, 12, 8, 5, 9]
print("=" * 80)
print("SA R8 VALIDATION - New Weights Performance Check")
print("=" * 80)
print(f"Actual Finish: {ACTUAL}")
print()

# Test weight schemes
schemes = {
    'Current (NEW)': {'class': 3.0, 'speed': 1.8, 'form': 1.8, 'pace': 1.5},
    'Old (PRE-FIX)': {'class': 2.5, 'speed': 2.0, 'form': 1.8, 'pace': 1.5},
    'Speed Emphasis': {'class': 2.5, 'speed': 2.5, 'form': 1.8, 'pace': 1.5},
    'Class Emphasis': {'class': 3.5, 'speed': 1.5, 'form': 1.8, 'pace': 1.5},
    'Balanced': {'class': 2.0, 'speed': 2.0, 'form': 2.0, 'pace': 1.5},
    'Prime Power Proxy': {'class': 2.0, 'speed': 1.5, 'form': 2.5, 'pace': 2.0}
}

def calculate_rating(row, weights):
    """Simplified rating calculation based on available metrics"""
    # Normalize components to 0-1 scale
    class_norm = (row['class_rating'] - 111) / 4  # Range ~111-115
    speed_norm = (row['speed_best_dist'] - 81) / 12  # Range ~81-93
    form_norm = (row['speed_last'] - 65) / 26  # Range ~65-91
    pace_norm = (row['e1_pace'] - 78) / 21  # Range ~78-99
    
    rating = (
        class_norm * weights['class'] +
        speed_norm * weights['speed'] +
        form_norm * weights['form'] +
        pace_norm * weights['pace']
    )
    return rating

for scheme_name, weights in schemes.items():
    print(f"\n{scheme_name} Weights: Class={weights['class']}× Speed={weights['speed']}× Form={weights['form']}× Pace={weights['pace']}×")
    print("-" * 80)
    
    # Calculate ratings
    df['rating'] = df.apply(lambda row: calculate_rating(row, weights), axis=1)
    
    # Sort by rating (descending)
    df_sorted = df.sort_values('rating', ascending=False).reset_index(drop=True)
    
    # Show top 5 predictions
    exact_matches = 0
    for i in range(min(5, len(df_sorted))):
        horse_num = df_sorted.iloc[i]['num']
        horse_name = df_sorted.iloc[i]['name']
        rating = df_sorted.iloc[i]['rating']
        actual_pos = df_sorted.iloc[i]['actual_finish']
        
        if actual_pos < 99:
            match = "✓" if (i + 1) == actual_pos else "✗"
            if (i + 1) == actual_pos:
                exact_matches += 1
            print(f"{match} Pred:{i+1} Act:{actual_pos} | #{horse_num:2d}_{horse_name:20s} Rating: {rating:5.2f}")
        else:
            print(f"✗ Pred:{i+1} Act:-- | #{horse_num:2d}_{horse_name:20s} Rating: {rating:5.2f}")
    
    print(f"✓ Exact position matches: {exact_matches}/5")
    
    # Check if winner was in top 3
    top3_nums = df_sorted.head(3)['num'].tolist()
    winner_in_top3 = 13 in top3_nums
    print(f"{'✓' if winner_in_top3 else '✗'} Winner (#13) in top 3 predictions: {winner_in_top3}")
    
    # Check if top 2 actual finishers were in top 5 predictions
    top5_nums = df_sorted.head(5)['num'].tolist()
    top2_captured = sum([1 for h in [13, 12] if h in top5_nums])
    print(f"Top 2 actual finishers in top 5 predictions: {top2_captured}/2")

print("\n" + "=" * 80)
print("PRIME POWER COMPARISON")
print("=" * 80)
df_pp = df.sort_values('prime_power', ascending=False).head(5)
print("\nTop 5 by Prime Power (BRISNET's comprehensive metric):")
for i, row in df_pp.iterrows():
    horse_num = int(row['num'])
    horse_name = row['name']
    pp = row['prime_power']
    actual_pos = row['actual_finish']
    if actual_pos < 99:
        print(f"✓ #{horse_num:2d} {horse_name:20s} PP:{pp:6.1f} → Finished {actual_pos}{['st','nd','rd','th','th'][actual_pos-1]}")
    else:
        print(f"✗ #{horse_num:2d} {horse_name:20s} PP:{pp:6.1f} → Did not finish top 5")

print("\n" + "=" * 80)
print("ANALYSIS SUMMARY")
print("=" * 80)
print("Key Finding: Prime Power top 3 horses finished 1-2-3!")
print("- #12 Miss Practical (PP 127.5 - 1st) → Finished 2nd")
print("- #8 Stay in Line (PP 125.4 - 2nd) → Finished 3rd")
print("- #13 Rizzleberry Rose (PP 125.3 - 3rd) → WON")
print()
print("System Performance with NEW weights (Class 3.0×, Speed 1.8×):")
print("- Predicted #5 Clubhouse Bride to win → Finished 4th ❌")
print("- Missed winner #13 Rizzleberry Rose completely")
print("- Missed #12 Miss Practical (2nd place) completely")
print("- Correctly identified #8 Stay in Line value (ranked 4th, finished 3rd)")
print()
print("Recommendation: Analyze component differences between system picks vs actual finishers")
