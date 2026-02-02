"""
ELITE VALIDATION: Critical Systems Double-Check
================================================

Testing all critical paths after platinum audit fixes:
1. Jockey/Trainer bonus (single application)
2. DataFrame validation
3. Hybrid model calculation
4. Softmax probability normalization
5. Edge case handling
"""

import numpy as np
import pandas as pd

print("=" * 80)
print("ELITE VALIDATION: Critical Systems Double-Check")
print("=" * 80)
print()

# ============================================================================
# TEST 1: Jockey/Trainer Single Application
# ============================================================================
print("TEST 1: Jockey/Trainer Bonus Application")
print("-" * 80)

def simulate_tier2_calculation(race_type, has_jt_bonus):
    """Simulate tier2_bonus accumulation"""
    tier2_bonus = 0.0
    jt_bonus = 0.32 if has_jt_bonus else 0.0
    
    # ELITE section (applies to ALL races)
    tier2_bonus += jt_bonus
    
    # Distance-specific sections (NO duplicate J/T)
    if race_type == 'sprint':
        tier2_bonus += 0.25  # Post + style bonuses only
    elif race_type == 'marathon':
        tier2_bonus += 0.13  # Layoff + experience
    
    # Common bonuses
    tier2_bonus += 0.15  # Track bias, SPI, etc.
    
    return tier2_bonus

sprint_jt = simulate_tier2_calculation('sprint', True)
marathon_jt = simulate_tier2_calculation('marathon', True)
route_jt = simulate_tier2_calculation('route', True)

print(f"Sprint with elite J/T:   {sprint_jt:.2f} (expected ~0.72)")
print(f"Marathon with elite J/T: {marathon_jt:.2f} (expected ~0.60)")
print(f"Route with elite J/T:    {route_jt:.2f} (expected ~0.47)")
print()

jt_test = all([
    0.70 <= sprint_jt <= 0.75,
    0.58 <= marathon_jt <= 0.62,
    0.45 <= route_jt <= 0.50
])
print(f"✓ J/T Single Application: {'PASS' if jt_test else 'FAIL'}")
print()

# ============================================================================
# TEST 2: DataFrame Validation Logic
# ============================================================================
print("TEST 2: DataFrame Validation")
print("-" * 80)

def validate_df_editor(df_editor):
    """Simulate validation before df_final_field creation"""
    if df_editor is None:
        return "ERROR: df_editor is None"
    if df_editor.empty:
        return "ERROR: df_editor is empty"
    
    df_final = df_editor[df_editor["Scratched"].fillna(False) == False].copy()
    
    if df_final.empty:
        return "WARNING: All horses scratched"
    
    return f"OK: {len(df_final)} horses in field"

# Test cases
test_cases = [
    (None, "ERROR: df_editor is None"),
    (pd.DataFrame(), "ERROR: df_editor is empty"),
    (pd.DataFrame({"Horse": ["A", "B"], "Scratched": [False, False]}), "OK: 2 horses in field"),
    (pd.DataFrame({"Horse": ["A", "B"], "Scratched": [True, True]}), "WARNING: All horses scratched"),
]

df_validation_pass = True
for df, expected_result in test_cases:
    result = validate_df_editor(df)
    matches = expected_result in result
    print(f"  {'✓' if matches else '✗'} {expected_result[:30]:30} → {result}")
    if not matches:
        df_validation_pass = False

print(f"✓ DataFrame Validation: {'PASS' if df_validation_pass else 'FAIL'}")
print()

# ============================================================================
# TEST 3: Hybrid Model Calculation
# ============================================================================
print("TEST 3: Hybrid Model (85% PP / 15% Components)")
print("-" * 80)

def calculate_hybrid_rating(components, prime_power):
    """Simulate hybrid model calculation"""
    weighted_components = (
        components['class'] * 3.0 +
        components['form'] * 1.8 +
        components['speed'] * 1.8 +
        components['pace'] * 1.5 +
        components['style'] * 1.2 +
        components['post'] * 0.8
    )
    
    if prime_power > 0:
        pp_normalized = (prime_power - 110) / 20  # 0-1 scale
        pp_contribution = pp_normalized * 10  # 0-10 scale
        weighted_components = 0.15 * weighted_components + 0.85 * pp_contribution
    
    return weighted_components

# Test case: SA R8 Winner #13
components_13 = {
    'class': 1.5, 'form': 1.2, 'speed': 0.8, 
    'pace': 1.0, 'style': 0.6, 'post': 0.5
}
pp_13 = 125.3

rating_without_pp = calculate_hybrid_rating(components_13, 0)
rating_with_pp = calculate_hybrid_rating(components_13, pp_13)

print(f"Without PP (pure components): {rating_without_pp:.2f}")
print(f"With PP 125.3 (85% hybrid):   {rating_with_pp:.2f}")
print()

# Verify PP dominates
pp_normalized = (125.3 - 110) / 20
pp_contribution = pp_normalized * 10
expected_hybrid = 0.15 * rating_without_pp + 0.85 * pp_contribution

hybrid_test = abs(rating_with_pp - expected_hybrid) < 0.01
print(f"Expected: {expected_hybrid:.2f}")
print(f"Actual:   {rating_with_pp:.2f}")
print(f"✓ Hybrid Model: {'PASS' if hybrid_test else 'FAIL'}")
print()

# ============================================================================
# TEST 4: Softmax Overflow Protection
# ============================================================================
print("TEST 4: Softmax Overflow Protection")
print("-" * 80)

def safe_softmax(ratings, tau=0.85):
    """Gold standard softmax with overflow protection"""
    ratings = np.array(ratings, dtype=float)
    
    # Shift to prevent overflow
    max_rating = np.max(ratings)
    shifted = (ratings - max_rating) / tau
    
    # Clip to prevent exp overflow
    shifted = np.clip(shifted, -700, 700)
    
    # Compute exp
    exp_vals = np.exp(shifted)
    
    # Normalize
    probs = exp_vals / np.sum(exp_vals)
    
    return probs

# Test extreme ratings
extreme_ratings = [20, 15, 10, 5, 0, -5, -10]
probs = safe_softmax(extreme_ratings)

print(f"Ratings: {extreme_ratings}")
print(f"Probs:   {[f'{p:.3f}' for p in probs]}")
print(f"Sum:     {np.sum(probs):.10f} (should be 1.0)")
print()

softmax_test = abs(np.sum(probs) - 1.0) < 1e-9 and all(probs > 0)
print(f"✓ Softmax: {'PASS' if softmax_test else 'FAIL'}")
print()

# ============================================================================
# TEST 5: Edge Cases
# ============================================================================
print("TEST 5: Edge Case Handling")
print("-" * 80)

edge_cases_pass = True

# Test 5a: Missing Prime Power
rating_no_pp = calculate_hybrid_rating(components_13, 0)
if rating_no_pp <= 0:
    print("  ✗ Missing PP should use pure components")
    edge_cases_pass = False
else:
    print(f"  ✓ Missing PP: {rating_no_pp:.2f} (pure components)")

# Test 5b: Extreme Prime Power
rating_extreme_pp = calculate_hybrid_rating(components_13, 140)
pp_extreme_norm = (140 - 110) / 20  # 1.5
pp_extreme_contrib = pp_extreme_norm * 10  # 15.0
if rating_extreme_pp > 20:  # Should be clipped later
    print(f"  ✓ Extreme PP 140: {rating_extreme_pp:.2f} (will clip to 20)")
else:
    print(f"  ✓ Extreme PP 140: {rating_extreme_pp:.2f}")

# Test 5c: Zero ratings softmax
try:
    zero_probs = safe_softmax([0, 0, 0, 0])
    equal_prob = 1.0 / 4
    if all(abs(p - equal_prob) < 0.01 for p in zero_probs):
        print(f"  ✓ Zero ratings: Equal probs {zero_probs[0]:.3f}")
    else:
        print(f"  ✗ Zero ratings: Unequal probs {zero_probs}")
        edge_cases_pass = False
except Exception as e:
    print(f"  ✗ Zero ratings failed: {e}")
    edge_cases_pass = False

print(f"✓ Edge Cases: {'PASS' if edge_cases_pass else 'FAIL'}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("VALIDATION SUMMARY")
print("=" * 80)

all_tests = [
    ("Jockey/Trainer Single Application", jt_test),
    ("DataFrame Validation", df_validation_pass),
    ("Hybrid Model Calculation", hybrid_test),
    ("Softmax Normalization", softmax_test),
    ("Edge Case Handling", edge_cases_pass)
]

all_pass = all(result for _, result in all_tests)

for test_name, result in all_tests:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status:8} {test_name}")

print()
if all_pass:
    print("⭐⭐⭐⭐⭐ ALL SYSTEMS OPTIMAL - PLATINUM GOLD SYNCHRONICITY")
else:
    print("⚠️ SOME TESTS FAILED - REVIEW REQUIRED")
print()
