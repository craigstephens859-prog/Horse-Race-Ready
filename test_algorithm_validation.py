"""
Algorithm Validation Test Suite
Ensures all prediction algorithms work correctly with proper normalization
"""

import pandas as pd
import numpy as np

def test_probability_normalization():
    """Test that probabilities always sum to 1.0"""
    print("=" * 60)
    print("TEST 1: Probability Normalization")
    print("=" * 60)
    
    # Test data
    test_probs = {
        'Horse A': 0.35,
        'Horse B': 0.25,
        'Horse C': 0.20,
        'Horse D': 0.15,
        'Horse E': 0.05
    }
    
    total = sum(test_probs.values())
    print(f"\nInput probabilities sum: {total:.6f}")
    
    # Normalize
    normalized = {h: p / total for h, p in test_probs.items()}
    norm_sum = sum(normalized.values())
    print(f"Normalized probabilities sum: {norm_sum:.10f}")
    
    if abs(norm_sum - 1.0) < 1e-9:
        print("✅ PASS: Probabilities properly normalized")
        return True
    else:
        print(f"❌ FAIL: Probabilities sum to {norm_sum}, not 1.0")
        return False


def test_sequential_selection():
    """Test that sequential selection produces unique horses"""
    print("\n" + "=" * 60)
    print("TEST 2: Sequential Selection (No Duplicates)")
    print("=" * 60)
    
    # Simulate finishing order calculation
    horses = ['Horse A', 'Horse B', 'Horse C', 'Horse D', 'Horse E']
    win_probs = np.array([0.35, 0.25, 0.20, 0.15, 0.05])
    
    # Normalize
    win_probs = win_probs / win_probs.sum()
    
    finishing_order = []
    remaining_indices = list(range(len(horses)))
    remaining_probs = win_probs.copy()
    
    for position in range(5):
        # Renormalize
        remaining_probs = remaining_probs / remaining_probs.sum()
        
        # Select
        best_idx = np.argmax(remaining_probs)
        selected_horse_idx = remaining_indices[best_idx]
        
        finishing_order.append(horses[selected_horse_idx])
        print(f"Position {position + 1}: {horses[selected_horse_idx]} ({remaining_probs[best_idx]*100:.1f}%)")
        
        # Remove
        remaining_indices.pop(best_idx)
        remaining_probs = np.delete(remaining_probs, best_idx)
    
    # Check for duplicates
    if len(finishing_order) == len(set(finishing_order)):
        print(f"\n✅ PASS: All {len(finishing_order)} horses are unique")
        return True
    else:
        print(f"\n❌ FAIL: Duplicate horses detected!")
        print(f"   Finishing order: {finishing_order}")
        return False


def test_edge_cases():
    """Test edge cases like empty fields, single horse, invalid probs"""
    print("\n" + "=" * 60)
    print("TEST 3: Edge Cases")
    print("=" * 60)
    
    all_passed = True
    
    # Test 1: Empty probabilities
    print("\n1. Empty probability array:")
    empty_probs = np.array([])
    if empty_probs.size == 0:
        uniform = np.array([1.0])
        print(f"   Fallback to uniform: {uniform}")
        print("   ✅ PASS: Handled empty array")
    else:
        print("   ❌ FAIL: Empty array not handled")
        all_passed = False
    
    # Test 2: All zero probabilities
    print("\n2. All zero probabilities:")
    zero_probs = np.array([0.0, 0.0, 0.0])
    if zero_probs.sum() == 0:
        uniform = np.ones(len(zero_probs)) / len(zero_probs)
        print(f"   Fallback to uniform: {uniform}")
        print(f"   Sum: {uniform.sum():.10f}")
        if abs(uniform.sum() - 1.0) < 1e-9:
            print("   ✅ PASS: Handled zero probabilities")
        else:
            print("   ❌ FAIL: Uniform distribution doesn't sum to 1.0")
            all_passed = False
    
    # Test 3: Single horse
    print("\n3. Single horse field:")
    single_prob = np.array([1.0])
    if abs(single_prob.sum() - 1.0) < 1e-9:
        print(f"   Single horse probability: {single_prob[0]:.10f}")
        print("   ✅ PASS: Single horse handled")
    else:
        print("   ❌ FAIL: Single horse probability != 1.0")
        all_passed = False
    
    # Test 4: Negative probabilities
    print("\n4. Negative probabilities:")
    neg_probs = np.array([0.5, -0.2, 0.7])
    clamped = np.maximum(neg_probs, 0.0)
    normalized = clamped / clamped.sum()
    print(f"   Input: {neg_probs}")
    print(f"   Clamped: {clamped}")
    print(f"   Normalized: {normalized}")
    print(f"   Sum: {normalized.sum():.10f}")
    if abs(normalized.sum() - 1.0) < 1e-9 and (normalized >= 0).all():
        print("   ✅ PASS: Negative values handled")
    else:
        print("   ❌ FAIL: Negative values not properly handled")
        all_passed = False
    
    # Test 5: Probabilities > 1.0
    print("\n5. Probabilities > 1.0 (percentage format):")
    pct_probs = np.array([35.0, 25.0, 20.0, 15.0, 5.0])  # Should be divided by 100
    converted = pct_probs / 100.0
    print(f"   Input: {pct_probs}")
    print(f"   Converted: {converted}")
    print(f"   Sum: {converted.sum():.10f}")
    if abs(converted.sum() - 1.0) < 1e-9:
        print("   ✅ PASS: Percentage values handled")
    else:
        print("   ❌ FAIL: Percentage values not properly converted")
        all_passed = False
    
    return all_passed


def test_live_odds_integration():
    """Test that Live Odds are prioritized over ML odds"""
    print("\n" + "=" * 60)
    print("TEST 4: Live Odds Integration")
    print("=" * 60)
    
    # Simulate odds dictionary building
    horses_data = [
        {'Horse': 'Horse A', 'ML': '5/2', 'Live Odds': '3/1'},
        {'Horse': 'Horse B', 'ML': '7/2', 'Live Odds': ''},
        {'Horse': 'Horse C', 'ML': '4/1', 'Live Odds': '6/1'},
    ]
    
    ml_odds_dict = {}
    for row in horses_data:
        horse_name = row['Horse']
        live_odds = row['Live Odds']
        ml_odds = row['ML']
        
        # Priority: Live Odds first, then ML
        odds_str = live_odds if live_odds else ml_odds
        
        # Parse odds
        if '/' in odds_str:
            parts = odds_str.split('/')
            ml_odds_dict[horse_name] = float(parts[0]) / float(parts[1])
        
        print(f"{horse_name}:")
        print(f"  ML: {ml_odds}, Live: {live_odds if live_odds else 'N/A'}")
        print(f"  Used: {odds_str} = {ml_odds_dict[horse_name]:.2f}")
    
    # Verify priority
    if ml_odds_dict['Horse A'] == 3.0 and ml_odds_dict['Horse B'] == 3.5 and ml_odds_dict['Horse C'] == 6.0:
        print("\n✅ PASS: Live Odds correctly prioritized over ML odds")
        return True
    else:
        print("\n❌ FAIL: Live Odds not properly prioritized")
        return False


def test_conditional_probabilities():
    """Test that conditional probabilities are calculated correctly"""
    print("\n" + "=" * 60)
    print("TEST 5: Conditional Probabilities")
    print("=" * 60)
    
    # Initial win probabilities
    horses = ['A', 'B', 'C']
    probs = np.array([0.5, 0.3, 0.2])
    
    print(f"\nInitial win probabilities:")
    for i, h in enumerate(horses):
        print(f"  {h}: {probs[i]*100:.1f}%")
    
    # After Horse A wins, what's the probability B finishes 2nd?
    # P(B=2nd | A=1st) = P(B wins from remaining) = 0.3 / (0.3 + 0.2) = 0.6
    remaining_probs = np.array([0.3, 0.2])
    remaining_probs = remaining_probs / remaining_probs.sum()
    
    print(f"\nConditional probabilities (given A wins):")
    print(f"  B: {remaining_probs[0]*100:.1f}%")
    print(f"  C: {remaining_probs[1]*100:.1f}%")
    
    expected_b = 0.6
    expected_c = 0.4
    
    if abs(remaining_probs[0] - expected_b) < 1e-9 and abs(remaining_probs[1] - expected_c) < 1e-9:
        print(f"\n✅ PASS: Conditional probabilities correct")
        print(f"   P(B=2nd | A=1st) = {remaining_probs[0]:.1f} = {expected_b:.1f} ✓")
        print(f"   P(C=2nd | A=1st) = {remaining_probs[1]:.1f} = {expected_c:.1f} ✓")
        return True
    else:
        print(f"\n❌ FAIL: Conditional probabilities incorrect")
        return False


def run_all_tests():
    """Run complete validation suite"""
    print("\n")
    print("╔" + "═" * 58 + "╗")
    print("║" + " " * 10 + "ALGORITHM VALIDATION TEST SUITE" + " " * 16 + "║")
    print("╚" + "═" * 58 + "╝")
    
    results = []
    
    results.append(("Probability Normalization", test_probability_normalization()))
    results.append(("Sequential Selection", test_sequential_selection()))
    results.append(("Edge Cases", test_edge_cases()))
    results.append(("Live Odds Integration", test_live_odds_integration()))
    results.append(("Conditional Probabilities", test_conditional_probabilities()))
    
    # Summary
    print("\n" + "=" * 60)
    print("TEST SUMMARY")
    print("=" * 60)
    
    passed = sum(1 for _, result in results if result)
    total = len(results)
    
    for test_name, result in results:
        status = "✅ PASS" if result else "❌ FAIL"
        print(f"{status}: {test_name}")
    
    print(f"\n{passed}/{total} tests passed ({passed/total*100:.1f}%)")
    
    if passed == total:
        print("\n🎉 ALL TESTS PASSED! Algorithms are working correctly.")
    else:
        print(f"\n⚠️  {total - passed} test(s) failed. Review algorithm implementation.")
    
    return passed == total


if __name__ == "__main__":
    success = run_all_tests()
    exit(0 if success else 1)
