"""
Live Analysis: Pegasus World Cup G1 (Gulfstream Park, Jan 24, 2026)
Run the BRISNET PP through our model engine and show predictions.
"""

import pandas as pd
import numpy as np
import re
from typing import Dict, List, Tuple

# Read the PP text
with open('pegasus_wc_g1_pp.txt', 'r', encoding='utf-8') as f:
    pp_text = f.read()

print("="*80)
print("PEGASUS WORLD CUP G1 - MODEL ANALYSIS")
print("="*80)
print(f"\nPP Text Length: {len(pp_text)} characters")

# ============================================================================
# STEP 1: Extract Race Context
# ============================================================================
print("\n" + "="*80)
print("STEP 1: RACE CONTEXT EXTRACTION")
print("="*80)

# Extract header info
header_line = pp_text.strip().split('\n')[0]
print(f"\nHeader: {header_line}")

# Parse race metadata
race_info = {
    'track': 'Gulfstream Park',
    'race_type': 'PWCInvit-G1',
    'distance': '1⅛ Mile',
    'age_sex': '4&up',
    'date': 'Saturday, January 24, 2026',
    'race_number': 13
}

print(f"\nRace Type: {race_info['race_type']} (G1 Stakes)")
print(f"Distance: {race_info['distance']}")
print(f"Surface: Dirt")

# ============================================================================
# STEP 2: Parse All Horses
# ============================================================================
print("\n" + "="*80)
print("STEP 2: HORSE DATA EXTRACTION")
print("="*80)

# Extract horses with key data
horse_pattern = r'(\d+)\s+([A-Za-z\s]+?)\s+\(([ESP/]+)\s+(\d+)\)'
horses_data = []

lines = pp_text.split('\n')
i = 0
while i < len(lines):
    line = lines[i].strip()
    
    # Match horse header line
    match = re.match(r'^(\d+)\s+([A-Za-z\s]+?)\s+\(([ESP/]+)\s+(\d+)\)', line)
    if match:
        post = match.group(1)
        name = match.group(2).strip()
        style = match.group(3)
        quirin = match.group(4)
        
        # Get next few lines for additional data
        odds = '?'
        prime_power = 0.0
        speed_rating = 0
        earnings = 0
        
        # Look for Prime Power in next 10 lines
        for j in range(i+1, min(i+15, len(lines))):
            check_line = lines[j]
            
            # Prime Power
            pp_match = re.search(r'Prime Power:\s*([\d.]+)', check_line)
            if pp_match:
                prime_power = float(pp_match.group(1))
            
            # Speed rating from Life line
            life_match = re.search(r'Life:.*\$[\d,]+\s+(\d+)', check_line)
            if life_match:
                speed_rating = int(life_match.group(1))
            
            # Earnings
            earn_match = re.search(r'\$([\d,]+)', check_line)
            if earn_match and earnings == 0:
                earnings = int(earn_match.group(1).replace(',', ''))
            
            # ML Odds
            odds_match = re.search(r'(\d+/\d+)', check_line)
            if odds_match and odds == '?':
                odds = odds_match.group(1)
        
        horses_data.append({
            'post': int(post),
            'name': name,
            'style': style,
            'quirin': int(quirin),
            'ml_odds': odds,
            'prime_power': prime_power,
            'speed_rating': speed_rating,
            'earnings': earnings
        })
    
    i += 1

print(f"\n✓ Parsed {len(horses_data)} horses")
for h in horses_data:
    print(f"  #{h['post']:2d} {h['name']:20s} | PP: {h['prime_power']:6.1f} | Speed: {h['speed_rating']:3d} | Style: {h['style']:5s} | ML: {h['ml_odds']:>4s}")

# ============================================================================
# STEP 3: Calculate Component Ratings
# ============================================================================
print("\n" + "="*80)
print("STEP 3: COMPONENT RATING CALCULATION")
print("="*80)

# G1 Stakes Weights (from app.py RACE_CLASS_PARSER)
WEIGHTS = {
    'class': 10.0,   # Maximum for G1
    'speed': 1.8,
    'form': 1.8,
    'pace': 1.5,
    'style': 1.2,
    'post': 0.8
}

print(f"\nG1 Stakes Component Weights:")
for comp, weight in WEIGHTS.items():
    print(f"  {comp.capitalize():8s}: {weight:.1f}x")

# Calculate ratings for each horse
field_avg_pp = np.mean([h['prime_power'] for h in horses_data if h['prime_power'] > 0])
field_avg_speed = np.mean([h['speed_rating'] for h in horses_data if h['speed_rating'] > 0])

print(f"\nField Averages:")
print(f"  Prime Power: {field_avg_pp:.1f}")
print(f"  Speed Rating: {field_avg_speed:.1f}")

ratings = []

for h in horses_data:
    # Class Component (earnings-based)
    class_score = np.log10(max(h['earnings'], 1000)) - 4.5  # Normalize around $30k-$50k
    
    # Speed Component (relative to field)
    speed_score = (h['speed_rating'] - field_avg_speed) * 0.2
    
    # Prime Power Component (substitute for form)
    pp_score = (h['prime_power'] - field_avg_pp) * 0.1
    
    # Style Component (E/P favored in G1)
    style_score = 0.5 if 'E/P' in h['style'] else (0.3 if 'E' in h['style'] else -0.2)
    
    # Post Component (inside posts slightly favored)
    post_score = -0.1 * (h['post'] - 6.5)
    
    # Pace Component (high Quirin = speed)
    pace_score = (h['quirin'] - 4.5) * 0.15
    
    # Weighted Total
    total_rating = (
        class_score * WEIGHTS['class'] +
        speed_score * WEIGHTS['speed'] +
        pp_score * WEIGHTS['form'] +
        pace_score * WEIGHTS['pace'] +
        style_score * WEIGHTS['style'] +
        post_score * WEIGHTS['post']
    )
    
    ratings.append({
        'post': h['post'],
        'name': h['name'],
        'ml_odds': h['ml_odds'],
        'class': class_score,
        'speed': speed_score,
        'form': pp_score,
        'pace': pace_score,
        'style': style_score,
        'post_adj': post_score,
        'rating': total_rating,
        'prime_power': h['prime_power'],
        'speed_rating': h['speed_rating']
    })

# Sort by rating
ratings.sort(key=lambda x: x['rating'], reverse=True)

print("\n" + "="*80)
print("STEP 4: MODEL PREDICTIONS")
print("="*80)

print("\n🏆 TOP 5 CONTENDERS (by Model Rating):\n")
for i, r in enumerate(ratings[:5], 1):
    print(f"{i}. #{r['post']:2d} {r['name']:20s} | Rating: {r['rating']:6.2f} | ML: {r['ml_odds']:>4s}")
    print(f"    Components: Class={r['class']:+.2f} Speed={r['speed']:+.2f} Form={r['form']:+.2f} Pace={r['pace']:+.2f}")

# Calculate Fair Win Probabilities
print("\n📊 FAIR WIN PROBABILITIES:\n")

# Softmax conversion
rating_vals = np.array([r['rating'] for r in ratings])
exp_ratings = np.exp(rating_vals - np.max(rating_vals))  # Subtract max for numerical stability
probs = exp_ratings / np.sum(exp_ratings)

for i, (r, prob) in enumerate(zip(ratings, probs)):
    r['win_prob'] = prob

ratings.sort(key=lambda x: x['win_prob'], reverse=True)

for i, r in enumerate(ratings[:5], 1):
    fair_odds = (1.0 / r['win_prob']) - 1.0
    print(f"{i}. #{r['post']:2d} {r['name']:20s} | {r['win_prob']*100:5.1f}% | Fair Odds: {fair_odds:.1f}/1 | ML: {r['ml_odds']:>4s}")

# Most Likely Finishing Order (Sequential Selection)
print("\n🎯 MOST LIKELY FINISHING ORDER:\n")

remaining_probs = probs.copy()
remaining_indices = list(range(len(ratings)))
finishing_order = []

for pos in range(1, 6):
    # Normalize remaining probabilities
    if remaining_probs.sum() > 0:
        remaining_probs = remaining_probs / remaining_probs.sum()
    
    # Select highest probability
    best_idx = np.argmax(remaining_probs)
    horse_idx = remaining_indices[best_idx]
    
    finishing_order.append((ratings[horse_idx], remaining_probs[best_idx]))
    
    # Remove from pool
    remaining_indices.pop(best_idx)
    remaining_probs = np.delete(remaining_probs, best_idx)

position_emoji = {1: "🥇", 2: "🥈", 3: "🥉", 4: "4️⃣", 5: "5️⃣"}

for pos, (horse, cond_prob) in enumerate(finishing_order, 1):
    print(f"{position_emoji[pos]} {['Win', 'Place', 'Show', '4th', '5th'][pos-1]:5s} (#{pos}) • #{horse['post']:2d} {horse['name']:20s} (Odds: {horse['ml_odds']:>4s}) — {cond_prob*100:.1f}% conditional probability")

print("\n" + "="*80)
print("ANALYSIS COMPLETE")
print("="*80)
print("\n✅ Model predictions generated. Please provide actual race results for fine-tuning.")
print("\nExpected format: 1st place program number, 2nd, 3rd, 4th, 5th")
print("Example: 11, 1, 3, 7, 5")
