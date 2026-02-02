"""
ULTRATHINK DIAGNOSTIC - Deep Analysis of Prediction Failures
=============================================================

Problem Statement:
- SA R6: Class adjustment helped (class dropper scenario)
- SA R8: ALL weight schemes failed identically
- Prime Power predicted SA R8 top 3 perfectly
- System's additive model favors single-component dominance
- Need to understand WHY and HOW to fix

This analysis will:
1. Component-by-component breakdown of winners vs system picks
2. Reverse engineer what weights WOULD have worked
3. Test multiplicative vs additive formulas
4. Identify race-type specific patterns
5. Propose hybrid model architecture
"""

import pandas as pd
import numpy as np
from scipy.optimize import minimize
import itertools

# SA R8 Complete Data (from BRISNET)
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
    'running_style': ['E/P', 'E/P', 'E', 'E', 'E/P', 'E', 'E/P', 'E', 'P', 'P', 'E/P', 'S', 'E'],
    'actual_finish': [99, 99, 99, 99, 4, 99, 99, 3, 5, 99, 99, 2, 1],
    'ml_odds': [8, 6, 8, 15, 20, 20, 20, 6, 30, 20, 20, 3, 9/2]
}

df_r8 = pd.DataFrame(horses_r8)

# SA R6 Data (for comparison)
horses_r6 = {
    'num': [2, 3, 4, 7, 8],
    'name': ['Elegant Life', 'Smarty Nose', 'Windribbon', 'Big Cheeseola', 'Poise and Prada'],
    'speed_fig': [69, 86, 76, 67, 65],
    'class_adj': [-0.30, -0.30, +0.45, +1.00, -0.75],
    'form': [0.41, 0.96, 1.00, 0.65, 0.48],
    'actual_finish': [5, 4, 1, 2, 3]
}

df_r6 = pd.DataFrame(horses_r6)

print("=" * 90)
print("PART 1: COMPONENT PROFILE COMPARISON")
print("=" * 90)
print("\nSA R8 - Winner vs System's Top Pick:")
print("-" * 90)

# Compare #13 (winner) vs #1 (system top pick)
winner = df_r8[df_r8['num'] == 13].iloc[0]
system_pick = df_r8[df_r8['num'] == 1].iloc[0]

print(f"\n{'Metric':<25} {'#13 Rizzleberry (WON)':<20} {'#1 Clarina (Pred 1st)':<20} {'Advantage':<15}")
print("-" * 90)
print(f"{'Speed Last Race':<25} {winner['speed_last']:<20} {system_pick['speed_last']:<20} {'#1 by 14 pts' if system_pick['speed_last'] > winner['speed_last'] else ''}")
print(f"{'Speed Best at Dist':<25} {winner['speed_best_dist']:<20} {system_pick['speed_best_dist']:<20} {'#1 by 8 pts' if system_pick['speed_best_dist'] > winner['speed_best_dist'] else ''}")
print(f"{'Class Rating':<25} {winner['class_rating']:<20} {system_pick['class_rating']:<20} {'#13 by 0.2' if winner['class_rating'] > system_pick['class_rating'] else '#1 by 0.2'}")
print(f"{'E1 Pace':<25} {winner['e1_pace']:<20} {system_pick['e1_pace']:<20} {'#1 by 7 pts' if system_pick['e1_pace'] > winner['e1_pace'] else ''}")
print(f"{'E2 Late Pace':<25} {winner['e2_late']:<20} {system_pick['e2_late']:<20} {'#1 by 13 pts' if system_pick['e2_late'] > winner['e2_late'] else ''}")
print(f"{'Running Style':<25} {winner['running_style']:<20} {system_pick['running_style']:<20}")
print(f"{'Prime Power':<25} {winner['prime_power']:<20} {system_pick['prime_power']:<20} {'#13 by 7.6 pts ★'}")
print(f"{'ML Odds':<25} {winner['ml_odds']:<20} {system_pick['ml_odds']:<20}")

print("\n★ KEY INSIGHT: #1 dominated individual components (speed, pace) but #13 had higher PRIME POWER")
print("   This suggests Prime Power captures synergistic effects our additive model misses.")

print("\n" + "=" * 90)
print("PART 2: REVERSE ENGINEERING OPTIMAL WEIGHTS")
print("=" * 90)

def calculate_rating_additive(row, w_class, w_speed, w_form, w_pace):
    """Additive model: rating = sum of weighted components"""
    class_norm = (row['class_rating'] - 111) / 4
    speed_norm = (row['speed_best_dist'] - 81) / 12
    form_norm = (row['speed_last'] - 65) / 26
    pace_norm = (row['e1_pace'] - 78) / 21
    return class_norm * w_class + speed_norm * w_speed + form_norm * w_form + pace_norm * w_pace

def calculate_rating_multiplicative(row, w_class, w_speed, w_form, w_pace):
    """Multiplicative model: rating = product of weighted components"""
    class_norm = 1 + (row['class_rating'] - 111) / 4 * w_class
    speed_norm = 1 + (row['speed_best_dist'] - 81) / 12 * w_speed
    form_norm = 1 + (row['speed_last'] - 65) / 26 * w_form
    pace_norm = 1 + (row['e1_pace'] - 78) / 21 * w_pace
    return class_norm * speed_norm * form_norm * pace_norm

def calculate_rating_hybrid(row, w_class, w_speed, w_form, w_pace, alpha=0.5):
    """Hybrid: weighted average of additive and multiplicative"""
    add = calculate_rating_additive(row, w_class, w_speed, w_form, w_pace)
    mult = calculate_rating_multiplicative(row, w_class, w_speed, w_form, w_pace)
    return alpha * add + (1 - alpha) * mult

def evaluate_weights(weights, df, model='additive'):
    """Calculate prediction accuracy for given weights"""
    w_class, w_speed, w_form, w_pace = weights
    
    if model == 'additive':
        df['rating'] = df.apply(lambda r: calculate_rating_additive(r, w_class, w_speed, w_form, w_pace), axis=1)
    elif model == 'multiplicative':
        df['rating'] = df.apply(lambda r: calculate_rating_multiplicative(r, w_class, w_speed, w_form, w_pace), axis=1)
    elif model == 'hybrid':
        df['rating'] = df.apply(lambda r: calculate_rating_hybrid(r, w_class, w_speed, w_form, w_pace), axis=1)
    
    df_sorted = df.sort_values('rating', ascending=False)
    
    # Score: exact position matches + partial credit for close positions
    score = 0
    top5_actual = df[df['actual_finish'] <= 5].sort_values('actual_finish')
    
    for i, actual_row in top5_actual.iterrows():
        horse_num = actual_row['num']
        actual_pos = actual_row['actual_finish']
        pred_pos = df_sorted[df_sorted['num'] == horse_num].index[0] + 1
        
        # Exact match: 10 points, off by 1: 5 points, off by 2: 2 points
        diff = abs(pred_pos - actual_pos)
        if diff == 0:
            score += 10
        elif diff == 1:
            score += 5
        elif diff == 2:
            score += 2
    
    # Bonus for getting winner in top 3
    winner_num = df[df['actual_finish'] == 1].iloc[0]['num']
    if winner_num in df_sorted.head(3)['num'].values:
        score += 10
    
    return -score  # Negative for minimization

print("\nSearching for optimal weights across different model types...")
print("(Testing: Additive, Multiplicative, Hybrid models)")
print("-" * 90)

# Grid search for optimal weights
best_results = {}
for model_type in ['additive', 'multiplicative', 'hybrid']:
    best_score = float('inf')
    best_weights = None
    
    # Test a range of weight combinations
    for w_class in [1.5, 2.0, 2.5, 3.0, 3.5, 4.0]:
        for w_speed in [1.0, 1.5, 2.0, 2.5, 3.0]:
            for w_form in [1.0, 1.5, 2.0, 2.5]:
                for w_pace in [1.0, 1.5, 2.0, 2.5]:
                    weights = (w_class, w_speed, w_form, w_pace)
                    df_test = df_r8.copy()
                    score = evaluate_weights(weights, df_test, model=model_type)
                    
                    if score < best_score:
                        best_score = score
                        best_weights = weights
    
    best_results[model_type] = {'weights': best_weights, 'score': -best_score}
    
    print(f"\n{model_type.upper()} Model:")
    print(f"  Optimal Weights: Class={best_weights[0]}× Speed={best_weights[1]}× Form={best_weights[2]}× Pace={best_weights[3]}×")
    print(f"  Prediction Score: {-best_score}/60 points")
    
    # Show predictions with these weights
    df_test = df_r8.copy()
    if model_type == 'additive':
        df_test['rating'] = df_test.apply(lambda r: calculate_rating_additive(r, *best_weights), axis=1)
    elif model_type == 'multiplicative':
        df_test['rating'] = df_test.apply(lambda r: calculate_rating_multiplicative(r, *best_weights), axis=1)
    elif model_type == 'hybrid':
        df_test['rating'] = df_test.apply(lambda r: calculate_rating_hybrid(r, *best_weights), axis=1)
    
    df_sorted = df_test.sort_values('rating', ascending=False)
    print(f"  Top 5 Predictions:")
    for i in range(5):
        horse = df_sorted.iloc[i]
        actual = horse['actual_finish']
        symbol = "✓" if actual <= 5 else "✗"
        if actual <= 5:
            print(f"    {symbol} Pred:{i+1} Act:{int(actual)} | #{int(horse['num'])} {horse['name']}")
        else:
            print(f"    {symbol} Pred:{i+1} Act:-- | #{int(horse['num'])} {horse['name']}")

print("\n" + "=" * 90)
print("PART 3: PRIME POWER FORMULA ANALYSIS")
print("=" * 90)
print("\nPrime Power successfully predicted top 3. Why?")
print("-" * 90)

# Correlation analysis
print("\nCorrelation of individual metrics with actual finish position:")
top5 = df_r8[df_r8['actual_finish'] <= 5].copy()
print(f"Speed Last:        {top5['speed_last'].corr(top5['actual_finish']):.3f} (closer to 0 = no correlation)")
print(f"Speed Best Dist:   {top5['speed_best_dist'].corr(top5['actual_finish']):.3f}")
print(f"Class Rating:      {top5['class_rating'].corr(top5['actual_finish']):.3f}")
print(f"E1 Pace:           {top5['e1_pace'].corr(top5['actual_finish']):.3f}")
print(f"Prime Power:       {top5['prime_power'].corr(top5['actual_finish']):.3f} ★ STRONGEST")

print("\n" + "=" * 90)
print("PART 4: PATTERN IDENTIFICATION - SA R6 vs SA R8")
print("=" * 90)

print("\nSA R6 (Class adjustment helped):")
print("  - Race type: Class dropper scenario")
print("  - Winner had: Class +1.00 (dominant factor)")
print("  - System pick had: Speed 86 (dominant factor)")
print("  - Solution: Increase class weight")

print("\nSA R8 (Class adjustment didn't help):")
print("  - Race type: Even field, no major class droppers")
print("  - Winner had: Balanced metrics, no single dominance")
print("  - System pick had: Speed 91 (dominant factor)")
print("  - Issue: Additive model favors single-component stars")

print("\n" + "=" * 90)
print("PART 5: RECOMMENDATIONS")
print("=" * 90)

print("""
1. HYBRID MODEL ARCHITECTURE
   - Use additive model for horses with dominant components
   - Use multiplicative model for balanced horses
   - Weight by coefficient of variation in components

2. RACE-TYPE SPECIFIC WEIGHTS
   - Class dropper races: Class 3.5×, Speed 1.5×
   - Even field races: Balanced 2.0×, 2.0×, 2.0×, 2.0×
   - Pace-intensive races: Pace 3.0×, Speed 1.5×

3. INTEGRATE PRIME POWER
   - Use Prime Power as a tiebreaker or adjustment factor
   - Formula: Final_Rating = Component_Rating × (Prime_Power/avg_prime_power)^0.3

4. COMPONENT INTERACTION TERMS
   - Add: (Class × Form) for recent class moves
   - Add: (Speed × Pace) for tactical speed advantage
   - Add: (Style × Track_Bias) for running style fit

5. MACHINE LEARNING APPROACH
   - Train gradient boosting model on historical results
   - Features: All components + interactions + race conditions
   - This would automatically discover optimal weights per race type
""")

print("\n" + "=" * 90)
print("IMMEDIATE ACTION ITEMS")
print("=" * 90)
print("""
1. Test hybrid model on next race
2. Collect 20-30 more race results with predictions
3. Build training dataset for ML model
4. Implement race-type classification logic
5. Add Prime Power adjustment factor to current system
""")
