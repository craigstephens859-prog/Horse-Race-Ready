"""
Test Prime Power Adjustment on SA R8 Data
==========================================
"""

import pandas as pd
import numpy as np

horses_r8 = {
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
    'actual_finish': [99, 99, 99, 99, 4, 99, 99, 3, 5, 99, 99, 2, 1]
}

df = pd.DataFrame(horses_r8)

# Current weights (Class 3.0×, Speed 1.8×, Form 1.8×, Pace 1.5×)
def calculate_rating_original(row):
    """Original additive model WITHOUT Prime Power adjustment"""
    class_norm = (row['class_rating'] - 111) / 4
    speed_norm = (row['speed_best_dist'] - 81) / 12
    form_norm = (row['speed_last'] - 65) / 26
    pace_norm = (row['e1_pace'] - 78) / 21
    return class_norm * 3.0 + speed_norm * 1.8 + form_norm * 1.8 + pace_norm * 1.5

def calculate_rating_with_pp(row):
    """NEW model WITH Prime Power adjustment"""
    # Base additive components
    class_norm = (row['class_rating'] - 111) / 4
    speed_norm = (row['speed_best_dist'] - 81) / 12
    form_norm = (row['speed_last'] - 65) / 26
    pace_norm = (row['e1_pace'] - 78) / 21
    base_rating = class_norm * 3.0 + speed_norm * 1.8 + form_norm * 1.8 + pace_norm * 1.5
    
    # Prime Power adjustment: (PP / avg_PP)^1.2
    # Increased exponent to make PP more influential
    avg_pp = 120.0
    pp_adjustment = (row['prime_power'] / avg_pp) ** 1.2
    
    return base_rating * pp_adjustment

print("=" * 90)
print("PRIME POWER ADJUSTMENT TEST - SA R8 Results")
print("=" * 90)
print("Actual Finish: 13, 12, 8, 5, 9")
print()

# Test ORIGINAL model
print("ORIGINAL MODEL (No Prime Power Adjustment):")
print("-" * 90)
df['rating_orig'] = df.apply(calculate_rating_original, axis=1)
df_sorted_orig = df.sort_values('rating_orig', ascending=False).reset_index(drop=True)

for i in range(5):
    horse = df_sorted_orig.iloc[i]
    actual = horse['actual_finish']
    pp = horse['prime_power']
    rating = horse['rating_orig']
    symbol = "✓" if (i+1) == actual else "✗"
    if actual <= 5:
        print(f"{symbol} Pred:{i+1} Act:{int(actual)} | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")
    else:
        print(f"{symbol} Pred:{i+1} Act:-- | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")

# Test NEW model with Prime Power
print("\n")
print("NEW MODEL (With Prime Power Adjustment Factor):")
print("-" * 90)
df['rating_pp'] = df.apply(calculate_rating_with_pp, axis=1)
df_sorted_pp = df.sort_values('rating_pp', ascending=False).reset_index(drop=True)

exact_matches = 0
for i in range(5):
    horse = df_sorted_pp.iloc[i]
    actual = horse['actual_finish']
    pp = horse['prime_power']
    rating = horse['rating_pp']
    
    if actual <= 5:
        symbol = "✓" if (i+1) == actual else "✗"
        if (i+1) == actual:
            exact_matches += 1
        print(f"{symbol} Pred:{i+1} Act:{int(actual)} | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")
    else:
        symbol = "✗"
        print(f"{symbol} Pred:{i+1} Act:-- | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")

print()
print(f"Exact position matches: {exact_matches}/5")

# Check winner placement
winner_in_top3 = 13 in df_sorted_pp.head(3)['num'].values
print(f"Winner (#13) in top 3: {winner_in_top3}")

# Show rating changes
print("\n" + "=" * 90)
print("RATING CHANGES WITH PRIME POWER ADJUSTMENT")
print("=" * 90)
print(f"{'Horse':<25} {'PP':<8} {'Orig Rank':<12} {'New Rank':<12} {'Change':<10}")
print("-" * 90)

df_comparison = df.copy()
df_comparison['orig_rank'] = df_sorted_orig['num'].reset_index(drop=True).index + 1
for idx, row in df.iterrows():
    horse_num = row['num']
    horse_name = row['name']
    pp = row['prime_power']
    
    orig_rank = df_sorted_orig[df_sorted_orig['num'] == horse_num].index[0] + 1
    new_rank = df_sorted_pp[df_sorted_pp['num'] == horse_num].index[0] + 1
    change = orig_rank - new_rank
    
    if change != 0:
        arrow = "↑" if change > 0 else "↓"
        print(f"#{horse_num:2d} {horse_name:<20} {pp:6.1f}  {orig_rank:2d}            {new_rank:2d}            {arrow}{abs(change)}")

print("\n✓ Winner #13 Rizzleberry Rose had high Prime Power (125.3 - 3rd highest)")
print("✓ System pick #1 Clarina had low Prime Power (117.7 - 11th highest)")
print("✓ Adjustment reorders horses based on balanced strength vs single-component dominance")
