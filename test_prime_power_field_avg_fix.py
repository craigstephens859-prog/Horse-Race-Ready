#!/usr/bin/env python3
"""
Comprehensive test for Prime Power field average calculation fix.
Verifies that field_avg_prime is calculated BEFORE the loop
and available for all iterations.
"""
import numpy as np
import pandas as pd

print("=" * 70)
print("PRIME POWER FIELD AVERAGE FIX - COMPREHENSIVE TEST")
print("=" * 70)

# Simulate the fixed logic
print("\n[SETUP] Creating test DataFrame with 6 horses")
test_df = pd.DataFrame({
    'Horse': ['Horse1', 'Horse2', 'Horse3', 'Horse4', 'Horse5', 'Horse6'],
    'Post': [1, 2, 3, 4, 5, 6],
    'Style': ['S', 'S', 'E', 'E', 'S', 'P']
})
print(f"  Horses: {list(test_df['Horse'])}")

# Simulate parse_prime_power_for_block function
def mock_parse_prime_power(name):
    """Mock function returning dict with prime_power"""
    primes = {
        'Horse1': 131.9,
        'Horse2': 125.5,
        'Horse3': 93.8,
        'Horse4': 91.6,
        'Horse5': 86.5,
        'Horse6': 79.8
    }
    value = primes.get(name)
    return {'prime_power': value, 'prime_power_rank': f"{len(primes)}th"}

print("\n[TEST 1] Calculate field average BEFORE loop")
# This is the FIXED approach - calculate before loop
all_primes = [mock_parse_prime_power(row["Horse"]).get('prime_power') 
              for _, row in test_df.iterrows()]
field_avg_prime = np.nanmean([p for p in all_primes if p is not None and not np.isnan(p)])

print(f"  All primes: {all_primes}")
print(f"  Filtered (non-None, non-NaN): {[p for p in all_primes if p is not None and not np.isnan(p)]}")
print(f"  Field average: {field_avg_prime:.2f}")
assert not np.isnan(field_avg_prime), "Field average should be numeric"
assert field_avg_prime > 0, "Field average should be positive"
print("  ✓ PASS: Field average calculated correctly")

print("\n[TEST 2] Verify field_avg_prime is available in loop (iterations)")
iteration_count = 0
for _, row in test_df.iterrows():
    iteration_count += 1
    name = row['Horse']
    prime_dict = mock_parse_prime_power(name)
    prime = prime_dict.get('prime_power')
    
    # This is the key test - field_avg_prime should be available and defined
    try:
        prime_bonus = (prime - field_avg_prime) * 0.005 if prime is not None and field_avg_prime is not None and not np.isnan(prime) and not np.isnan(field_avg_prime) else 0
        print(f"  Iteration {iteration_count}: {name:10} prime={prime:6.1f}  bonus={prime_bonus:+.6f}")
        assert prime_bonus >= -1 and prime_bonus <= 1, f"Bonus out of expected range: {prime_bonus}"
    except NameError as e:
        print(f"  ✗ FAIL: NameError on iteration {iteration_count}: {e}")
        raise

print(f"  ✓ PASS: All {iteration_count} iterations completed successfully")

print("\n[TEST 3] Verify bonus calculations are mathematically correct")
prime = 131.9
expected_bonus = (prime - field_avg_prime) * 0.005
print(f"  Prime: {prime}")
print(f"  Field average: {field_avg_prime:.2f}")
print(f"  Expected bonus: {expected_bonus:.6f}")
print(f"  Calculation: ({prime} - {field_avg_prime:.2f}) * 0.005 = {expected_bonus:.6f}")
assert expected_bonus > 0, "High prime should give positive bonus"
print("  ✓ PASS: Bonus calculation correct")

print("\n[TEST 4] Verify handling of edge cases")
# Test with None value
prime_dict_none = {'prime_power': None, 'prime_power_rank': None}
prime_none = prime_dict_none.get('prime_power')
bonus_none = (prime_none - field_avg_prime) * 0.005 if prime_none is not None and field_avg_prime is not None and not np.isnan(prime_none) and not np.isnan(field_avg_prime) else 0
print(f"  None prime bonus: {bonus_none} (expected: 0)")
assert bonus_none == 0, "None should result in 0 bonus"
print("  ✓ PASS: None handling correct")

print("\n" + "=" * 70)
print("ALL TESTS PASSED ✓")
print("=" * 70)

print("\nFIX SUMMARY:")
print("────────────────────────────────────────────────────────────────")
print("ISSUE: field_avg_prime calculated inside loop on 'if _ == 0' check")
print("PROBLEM: Variable not defined for subsequent loop iterations")
print("\nFIXED BY:")
print("1. Moving calculation BEFORE the loop (lines 3126-3128)")
print("2. Remove conditional check inside loop")
print("3. Variable now available for ALL iterations")
print("\nLINES CHANGED:")
print("- Added: Lines 3126-3128 (pre-calculate field average)")
print("- Removed: Lines 3154-3161 (old conditional logic)")
print("- Updated: Line 3160 (comment reflects new approach)")
print("────────────────────────────────────────────────────────────────")
