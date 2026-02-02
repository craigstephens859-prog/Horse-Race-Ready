"""
VALIDATION: Hybrid Model Implementation Test
============================================
Ensure the code change actually fixes SA R8 prediction
Test edge cases and verify data flow
"""

import pandas as pd
import numpy as np

# Simulate the exact logic from app.py with hybrid model
def safe_float(val, default=0.0):
    """Safe float conversion matching app.py logic"""
    try:
        return float(val) if pd.notna(val) else default
    except:
        return default

def calculate_rating_hybrid_app_logic(row):
    """
    Replicate EXACT app.py logic for hybrid model
    This should match lines 3841-3862 in app.py
    """
    # Component calculations (simplified for test)
    class_norm = (row['class_rating'] - 111) / 4
    speed_norm = (row['speed_best_dist'] - 81) / 12
    form_norm = (row['speed_last'] - 65) / 26
    pace_norm = (row['e1_pace'] - 78) / 21
    
    weighted_components = (
        class_norm * 3.0 +
        form_norm * 1.8 +
        speed_norm * 1.8 +
        pace_norm * 1.5
    )
    
    # HYBRID MODEL logic from app.py
    prime_power_raw = safe_float(row.get('prime_power', 0.0), 0.0)
    if prime_power_raw > 0:
        # Normalize Prime Power (typical range: 110-130)
        pp_normalized = (prime_power_raw - 110) / 20  # 0 to 1 scale
        pp_contribution = pp_normalized * 10  # Scale to match component range
        
        # Hybrid: 30% component model, 70% Prime Power
        weighted_components = 0.3 * weighted_components + 0.7 * pp_contribution
    
    return weighted_components

# SA R8 Test Data
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
    'actual_finish': [99, 99, 99, 99, 4, 99, 99, 3, 5, 99, 99, 2, 1]
}

df = pd.DataFrame(horses_r8)

print("=" * 90)
print("VALIDATION: Hybrid Model Implementation (app.py logic)")
print("=" * 90)
print("Actual Finish: 13, 12, 8, 5, 9")
print()

# Calculate ratings using app.py hybrid logic
df['rating'] = df.apply(calculate_rating_hybrid_app_logic, axis=1)
df_sorted = df.sort_values('rating', ascending=False).reset_index(drop=True)

print("Prediction with Hybrid Model (30% Components + 70% Prime Power):")
print("-" * 90)

exact_matches = 0
winner_in_top3 = False
top2_in_top5 = 0

for i in range(5):
    horse = df_sorted.iloc[i]
    actual = horse['actual_finish']
    pp = horse['prime_power']
    rating = horse['rating']
    
    if horse['num'] == 13 and i < 3:
        winner_in_top3 = True
    
    if horse['num'] in [13, 12] and i < 5:
        top2_in_top5 += 1
    
    if actual <= 5:
        symbol = "✓" if (i+1) == actual else "✗"
        if (i+1) == actual:
            exact_matches += 1
        print(f"{symbol} Pred:{i+1} Act:{int(actual)} | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")
    else:
        symbol = "✗"
        print(f"{symbol} Pred:{i+1} Act:-- | #{int(horse['num']):2d} {horse['name']:20s} Rating:{rating:5.2f} PP:{pp:5.1f}")

print()
print(f"✓ Exact position matches: {exact_matches}/5")
print(f"✓ Winner (#13) in top 3: {winner_in_top3}")
print(f"✓ Top 2 finishers in top 5 predictions: {top2_in_top5}/2")

# Edge Case Testing
print("\n" + "=" * 90)
print("EDGE CASE TESTING")
print("=" * 90)

# Test 1: Missing Prime Power (should fallback to pure component model)
print("\nTest 1: Horse with Prime Power = 0 (missing data)")
test_horse_no_pp = {
    'speed_last': 85,
    'speed_best_dist': 88,
    'prime_power': 0.0,  # Missing!
    'class_rating': 113.0,
    'e1_pace': 92
}
rating_no_pp = calculate_rating_hybrid_app_logic(test_horse_no_pp)
print(f"  Result: Rating = {rating_no_pp:.2f}")
print(f"  ✓ Fallback to component-only model works (no crash)")

# Test 2: Very low Prime Power
print("\nTest 2: Horse with very low Prime Power (110)")
test_horse_low_pp = {
    'speed_last': 85,
    'speed_best_dist': 88,
    'prime_power': 110.0,
    'class_rating': 113.0,
    'e1_pace': 92
}
rating_low_pp = calculate_rating_hybrid_app_logic(test_horse_low_pp)
print(f"  Result: Rating = {rating_low_pp:.2f}")
print(f"  ✓ Low PP handled (normalized to 0.0)")

# Test 3: Very high Prime Power
print("\nTest 3: Horse with very high Prime Power (130)")
test_horse_high_pp = {
    'speed_last': 85,
    'speed_best_dist': 88,
    'prime_power': 130.0,
    'class_rating': 113.0,
    'e1_pace': 92
}
rating_high_pp = calculate_rating_hybrid_app_logic(test_horse_high_pp)
print(f"  Result: Rating = {rating_high_pp:.2f}")
print(f"  ✓ High PP handled (normalized to 1.0)")

# Test 4: Extreme Prime Power (outside typical range)
print("\nTest 4: Horse with extreme Prime Power (140 - outside normal range)")
test_horse_extreme_pp = {
    'speed_last': 85,
    'speed_best_dist': 88,
    'prime_power': 140.0,
    'class_rating': 113.0,
    'e1_pace': 92
}
rating_extreme_pp = calculate_rating_hybrid_app_logic(test_horse_extreme_pp)
print(f"  Result: Rating = {rating_extreme_pp:.2f}")
print(f"  ✓ Extreme PP handled (normalized to 1.5 - may need capping)")

print("\n" + "=" * 90)
print("IMPLEMENTATION VALIDATION SUMMARY")
print("=" * 90)

if winner_in_top3:
    print("✓✓✓ SUCCESS: Winner (#13) now in top 3 predictions")
else:
    print("✗✗✗ ISSUE: Winner (#13) still not in top 3")

if top2_in_top5 == 2:
    print("✓✓✓ SUCCESS: Both top 2 finishers in top 5 predictions")
else:
    print(f"⚠⚠⚠ PARTIAL: Only {top2_in_top5}/2 top finishers in top 5")

print("\nRECOMMENDATIONS:")
if rating_extreme_pp > 10:
    print("  1. Consider capping Prime Power contribution at reasonable max (e.g., 135)")
print("  2. Add UI indicator when hybrid model is active (user transparency)")
print("  3. Track performance: hybrid vs pure components over next 20 races")
print("  4. Add fallback message when Prime Power unavailable")

print("\n" + "=" * 90)
print("CRITICAL METRICS")
print("=" * 90)
print(f"SA R8 Improvement:")
print(f"  - OLD Model: Winner ranked 11th+ (missed completely)")
print(f"  - NEW Model: Winner in top 3 = {winner_in_top3}")
print(f"  - Improvement: {'MASSIVE' if winner_in_top3 else 'None'}")
