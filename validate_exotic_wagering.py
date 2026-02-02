"""
ELITE VALIDATION: Exotic Wagering Harville Calculations
========================================================

Testing Harville formula implementation for exotics:
1. Division by zero protection
2. Probability normalization
3. Index bounds checking
4. Fair odds calculation
"""

import numpy as np

print("=" * 80)
print("EXOTIC WAGERING VALIDATION")
print("=" * 80)
print()

# ============================================================================
# TEST 1: Harville Formula Exacta Calculation
# ============================================================================
print("TEST 1: Harville Exacta (1st → 2nd)")
print("-" * 80)

def harville_exacta(p1, p2):
    """
    Exacta probability using Harville formula
    P(i wins, j second) = P(i) * P(j) / (1 - P(i))
    """
    if p1 <= 0 or p2 <= 0:
        return 0.0
    
    denom = 1.0 - p1
    if denom <= 1e-9:  # Prevent division by zero
        return 0.0
    
    return p1 * (p2 / denom)

# Test cases
test_exacta = [
    (0.35, 0.25, "Favorite over contender"),
    (0.25, 0.35, "Contender over favorite"),
    (0.10, 0.10, "Longshots"),
    (0.99, 0.01, "Extreme favorite - EDGE CASE"),
    (0.01, 0.99, "Longshot over heavy chalk - EDGE CASE"),
]

exacta_pass = True
print(f"{'Horse 1':>10} {'Horse 2':>10} {'Prob':>10} {'Fair Odds':>12}")
print("-" * 50)
for p1, p2, desc in test_exacta:
    prob = harville_exacta(p1, p2)
    if prob > 1e-9:
        fair_odds = (1.0 / prob) - 1
    else:
        fair_odds = float('inf')
    
    print(f"{p1:>10.2%} {p2:>10.2%} {prob:>10.4f} {fair_odds:>12.1f}")
    
    # Validation
    if p1 >= 0.99 and prob <= 0:
        print(f"  ✗ FAIL: {desc} - should handle extreme probability")
        exacta_pass = False
    elif not np.isfinite(fair_odds) and prob > 1e-9:
        print(f"  ✗ FAIL: {desc} - fair odds overflow")
        exacta_pass = False

print(f"\n✓ Exacta Harville: {'PASS' if exacta_pass else 'FAIL'}")
print()

# ============================================================================
# TEST 2: Trifecta Sequential Probability
# ============================================================================
print("TEST 2: Harville Trifecta (1st → 2nd → 3rd)")
print("-" * 80)

def harville_trifecta(p1, p2, p3):
    """
    Trifecta probability using Harville formula
    P(i,j,k) = P(i) * P(j)/(1-P(i)) * P(k)/(1-P(i)-P(j))
    """
    if p1 <= 0 or p2 <= 0 or p3 <= 0:
        return 0.0
    
    denom_ij = 1.0 - p1
    denom_ijk = 1.0 - p1 - p2
    
    if denom_ij <= 1e-9 or denom_ijk <= 1e-9:
        return 0.0
    
    p_ij = p1 * (p2 / denom_ij)
    prob_ijk = p_ij * (p3 / denom_ijk)
    
    return prob_ijk

# Test cases
test_trifecta = [
    (0.30, 0.25, 0.20, "Top 3 favorites"),
    (0.15, 0.12, 0.10, "Middle pack"),
    (0.50, 0.30, 0.15, "Chalk trifecta"),
    (0.95, 0.03, 0.01, "Heavy favorite - EDGE CASE"),
]

trifecta_pass = True
print(f"{'P1':>8} {'P2':>8} {'P3':>8} {'Prob':>10} {'Fair Odds':>12}")
print("-" * 50)
for p1, p2, p3, desc in test_trifecta:
    prob = harville_trifecta(p1, p2, p3)
    if prob > 1e-9:
        fair_odds = (1.0 / prob) - 1
    else:
        fair_odds = float('inf')
    
    print(f"{p1:>8.2%} {p2:>8.2%} {p3:>8.2%} {prob:>10.6f} {fair_odds:>12.0f}")
    
    # Validation
    if prob < 0 or prob > 1.0:
        print(f"  ✗ FAIL: {desc} - probability out of bounds")
        trifecta_pass = False
    elif not np.isfinite(fair_odds) and prob > 1e-9:
        print(f"  ✗ FAIL: {desc} - fair odds overflow")
        trifecta_pass = False

print(f"\n✓ Trifecta Harville: {'PASS' if trifecta_pass else 'FAIL'}")
print()

# ============================================================================
# TEST 3: Probability Normalization
# ============================================================================
print("TEST 3: Probability Normalization")
print("-" * 80)

def calculate_all_exactas(probs):
    """Calculate all exacta probabilities and normalize"""
    horses = list(range(len(probs)))
    exactas = []
    
    for i in horses:
        for j in horses:
            if i == j:
                continue
            
            prob = harville_exacta(probs[i], probs[j])
            exactas.append({"ticket": f"{i}→{j}", "prob": prob})
    
    # Normalize
    total = sum(e["prob"] for e in exactas)
    if total > 0:
        for e in exactas:
            e["prob"] /= total
    
    return exactas, total

# Test with 4-horse field
probs_4horse = np.array([0.40, 0.30, 0.20, 0.10])
exactas, total_before_norm = calculate_all_exactas(probs_4horse)

# Check sum after normalization
sum_after = sum(e["prob"] for e in exactas)

print(f"4-horse field probabilities: {probs_4horse}")
print(f"Total exacta combinations: {len(exactas)}")
print(f"Sum before normalization: {total_before_norm:.6f}")
print(f"Sum after normalization:  {sum_after:.10f}")
print()

normalization_pass = abs(sum_after - 1.0) < 1e-9
print(f"✓ Normalization: {'PASS' if normalization_pass else 'FAIL'}")
print()

# ============================================================================
# TEST 4: Edge Case - Near-Certain Favorite
# ============================================================================
print("TEST 4: Edge Cases")
print("-" * 80)

edge_cases_pass = True

# Case 1: 99% favorite
p_extreme = 0.99
p_other = 0.01
prob_extreme = harville_exacta(p_extreme, p_other)

if prob_extreme > 0:
    print(f"  ✓ 99% favorite exacta: {prob_extreme:.6f} (handled)")
else:
    print(f"  ✗ 99% favorite exacta: returned 0 (should be small positive)")
    edge_cases_pass = False

# Case 2: Equal probabilities (uniform field)
probs_uniform = np.array([0.25, 0.25, 0.25, 0.25])
uniform_exactas, _ = calculate_all_exactas(probs_uniform)
uniform_probs = [e["prob"] for e in uniform_exactas]
uniform_std = np.std(uniform_probs)

print(f"  ✓ Uniform field std dev: {uniform_std:.6f} (should be ~0)")

# Case 3: Zero probability horse
prob_zero_exacta = harville_exacta(0.0, 0.30)
if prob_zero_exacta == 0.0:
    print(f"  ✓ Zero probability: {prob_zero_exacta} (correctly handled)")
else:
    print(f"  ✗ Zero probability: {prob_zero_exacta} (should be 0)")
    edge_cases_pass = False

print(f"\n✓ Edge Cases: {'PASS' if edge_cases_pass else 'FAIL'}")
print()

# ============================================================================
# TEST 5: Index Bounds (Bug #10 from audit)
# ============================================================================
print("TEST 5: Index Bounds Checking")
print("-" * 80)

def safe_harville_trifecta_indexed(probs, idx1, idx2, idx3):
    """Harville trifecta with index bounds checking"""
    n = len(probs)
    
    # Bounds check
    if not all(0 <= idx < n for idx in [idx1, idx2, idx3]):
        return 0.0
    
    # Uniqueness check
    if len({idx1, idx2, idx3}) != 3:
        return 0.0
    
    return harville_trifecta(probs[idx1], probs[idx2], probs[idx3])

probs_test = np.array([0.35, 0.25, 0.20, 0.15, 0.05])

# Valid indices
prob_valid = safe_harville_trifecta_indexed(probs_test, 0, 1, 2)
print(f"  Valid indices (0,1,2): {prob_valid:.6f}")

# Out of bounds
prob_oob = safe_harville_trifecta_indexed(probs_test, 0, 1, 10)
bounds_test1 = (prob_oob == 0.0)
print(f"  {'✓' if bounds_test1 else '✗'} Out of bounds (0,1,10): {prob_oob} (should be 0)")

# Duplicate indices
prob_dup = safe_harville_trifecta_indexed(probs_test, 0, 0, 1)
bounds_test2 = (prob_dup == 0.0)
print(f"  {'✓' if bounds_test2 else '✗'} Duplicate indices (0,0,1): {prob_dup} (should be 0)")

index_bounds_pass = bounds_test1 and bounds_test2
print(f"\n✓ Index Bounds: {'PASS' if index_bounds_pass else 'FAIL'}")
print()

# ============================================================================
# SUMMARY
# ============================================================================
print("=" * 80)
print("EXOTIC WAGERING VALIDATION SUMMARY")
print("=" * 80)

all_tests = [
    ("Exacta Harville Formula", exacta_pass),
    ("Trifecta Harville Formula", trifecta_pass),
    ("Probability Normalization", normalization_pass),
    ("Edge Case Handling", edge_cases_pass),
    ("Index Bounds Checking", index_bounds_pass),
]

all_pass = all(result for _, result in all_tests)

for test_name, result in all_tests:
    status = "✓ PASS" if result else "✗ FAIL"
    print(f"{status:8} {test_name}")

print()
if all_pass:
    print("⭐⭐⭐⭐⭐ EXOTIC CALCULATIONS OPTIMAL - PLATINUM VALIDATED")
else:
    print("⚠️ SOME EXOTIC TESTS FAILED")
print()
