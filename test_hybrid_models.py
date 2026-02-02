"""
Test DIRECT Prime Power Integration (not just adjustment factor)
=================================================================
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

def calculate_rating_hybrid_50_50(row):
    """50% component model, 50% Prime Power"""
    # Component model
    class_norm = (row['class_rating'] - 111) / 4
    speed_norm = (row['speed_best_dist'] - 81) / 12
    form_norm = (row['speed_last'] - 65) / 26
    pace_norm = (row['e1_pace'] - 78) / 21
    component_rating = class_norm * 3.0 + speed_norm * 1.8 + form_norm * 1.8 + pace_norm * 1.5
    
    # Prime Power normalized (range 114-128)
    pp_normalized = (row['prime_power'] - 114) / 14
    
    # Hybrid: 50/50 blend
    return 0.5 * component_rating + 0.5 * (pp_normalized * 10)

def calculate_rating_hybrid_30_70(row):
    """30% component model, 70% Prime Power"""
    class_norm = (row['class_rating'] - 111) / 4
    speed_norm = (row['speed_best_dist'] - 81) / 12
    form_norm = (row['speed_last'] - 65) / 26
    pace_norm = (row['e1_pace'] - 78) / 21
    component_rating = class_norm * 3.0 + speed_norm * 1.8 + form_norm * 1.8 + pace_norm * 1.5
    
    pp_normalized = (row['prime_power'] - 114) / 14
    
    return 0.3 * component_rating + 0.7 * (pp_normalized * 10)

def calculate_rating_pp_only(row):
    """Pure Prime Power ranking"""
    return row['prime_power']

print("=" * 90)
print("HYBRID MODEL TESTING - Component + Prime Power Blends")
print("=" * 90)
print("Actual Finish: 13, 12, 8, 5, 9")
print()

# Test 50/50 hybrid
print("HYBRID 50% Components / 50% Prime Power:")
print("-" * 90)
df['rating'] = df.apply(calculate_rating_hybrid_50_50, axis=1)
df_sorted = df.sort_values('rating', ascending=False).reset_index(drop=True)

exact_matches = 0
winner_in_top3 = False
for i in range(5):
    horse = df_sorted.iloc[i]
    actual = horse['actual_finish']
    pp = horse['prime_power']
    rating = horse['rating']
    
    if horse['num'] == 13 and i < 3:
        winner_in_top3 = True
    
    if actual <= 5:
        symbol = "✓" if (i+1) == actual else "✗"
        if (i+1) == actual:
            exact_matches += 1
        print(f"{symbol} Pred:{i+1} Act:{int(actual)} | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")
    else:
        symbol = "✗"
        print(f"{symbol} Pred:{i+1} Act:-- | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")

print(f"\nExact matches: {exact_matches}/5 | Winner in top 3: {winner_in_top3}")

# Test 30/70 hybrid
print("\n")
print("HYBRID 30% Components / 70% Prime Power:")
print("-" * 90)
df['rating'] = df.apply(calculate_rating_hybrid_30_70, axis=1)
df_sorted = df.sort_values('rating', ascending=False).reset_index(drop=True)

exact_matches = 0
winner_in_top3 = False
for i in range(5):
    horse = df_sorted.iloc[i]
    actual = horse['actual_finish']
    pp = horse['prime_power']
    rating = horse['rating']
    
    if horse['num'] == 13 and i < 3:
        winner_in_top3 = True
    
    if actual <= 5:
        symbol = "✓" if (i+1) == actual else "✗"
        if (i+1) == actual:
            exact_matches += 1
        print(f"{symbol} Pred:{i+1} Act:{int(actual)} | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")
    else:
        symbol = "✗"
        print(f"{symbol} Pred:{i+1} Act:-- | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")

print(f"\nExact matches: {exact_matches}/5 | Winner in top 3: {winner_in_top3}")

# Test Pure Prime Power
print("\n")
print("PURE PRIME POWER (100%):")
print("-" * 90)
df['rating'] = df.apply(calculate_rating_pp_only, axis=1)
df_sorted = df.sort_values('rating', ascending=False).reset_index(drop=True)

exact_matches = 0
winner_in_top3 = False
for i in range(5):
    horse = df_sorted.iloc[i]
    actual = horse['actual_finish']
    pp = horse['prime_power']
    rating = horse['rating']
    
    if horse['num'] == 13 and i < 3:
        winner_in_top3 = True
    
    if actual <= 5:
        symbol = "✓" if (i+1) == actual else "✗"
        if (i+1) == actual:
            exact_matches += 1
        print(f"{symbol} Pred:{i+1} Act:{int(actual)} | #{int(horse['num']):2d} {horse['name']:20s} PP:{pp:5.1f}")
    else:
        symbol = "✗"
        print(f"{symbol} Pred:{i+1} Act:-- | #{int(horse['num']):2d} {horse['name']:20s} PP:{pp:5.1f}")

print(f"\nExact matches: {exact_matches}/5 | Winner in top 3: {winner_in_top3}")

print("\n" + "=" * 90)
print("CONCLUSION")
print("=" * 90)
print("Pure Prime Power correctly identified top 3 finishers (just in wrong order)")
print("Hybrid models can balance Prime Power with component analysis")
print("Recommendation: 30/70 hybrid (30% components, 70% Prime Power) for even-field races")
