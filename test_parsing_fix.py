#!/usr/bin/env python3
"""
Quick test to validate that new parsing functions work correctly
"""
import sys
sys.path.insert(0, r'C:\Users\C Stephens\Desktop\Horse Racing Picks')

print("="*80)
print("QUICK TEST: Parsing Functions Validation")
print("="*80)

# Import app functions
from app import (
    parse_pedigree_spi, 
    parse_pedigree_surface_stats,
    parse_awd_analysis,
    parse_track_bias_impact_values,
    calculate_spi_bonus,
    calculate_surface_specialty_bonus,
    calculate_awd_mismatch_penalty
)

# Test Case 1: SPI Parsing
print("\n[TEST 1] SPI Parsing")
test_block_spi = """
Pedigree: Fast 89, Off 84, Distance 92, Turf 88
Sire: Way of Appeal SPI .36
Dam Sire: Some Horse SPI .44
"""

result_spi = parse_pedigree_spi(test_block_spi)
print(f"  Input: Block with 'SPI .36' and 'SPI .44'")
print(f"  Output: {result_spi}")
if result_spi.get("spi") == 0.36:
    print("  ✓ PASS: SPI parsed correctly")
else:
    print(f"  ✗ FAIL: Expected spi=0.36, got {result_spi.get('spi')}")

# Test Case 2: Surface Stats Parsing
print("\n[TEST 2] Surface Specialty Stats Parsing")
test_block_surface = """
Sire: Way of Appeal Mud 23% Turf 19%
Dam-Sire: Some Horse Mud 13%
"""

result_surface = parse_pedigree_surface_stats(test_block_surface)
print(f"  Input: Block with 'Mud 23%', 'Turf 19%', 'Dam-Sire Mud 13%'")
print(f"  Output: {result_surface}")
if result_surface.get("sire_mud_pct") == 23.0:
    print("  ✓ PASS: Sire Mud %parsed correctly")
else:
    print(f"  ✗ FAIL: Expected sire_mud_pct=23.0, got {result_surface.get('sire_mud_pct')}")

# Test Case 3: AWD Parsing
print("\n[TEST 3] AWD Distance Analysis Parsing")
test_block_awd = """
Sire: Way of Appeal AWD 5.2f
Dam-Sire: Some Horse AWD 6.4f
"""

result_awd = parse_awd_analysis(test_block_awd)
print(f"  Input: Block with 'AWD 5.2f' and 'AWD 6.4f'")
print(f"  Output: {result_awd}")
if result_awd.get("sire_awd") == 5.2:
    print("  ✓ PASS: AWD parsed correctly")
else:
    print(f"  ✗ FAIL: Expected sire_awd=5.2, got {result_awd.get('sire_awd')}")

# Test Case 4: Impact Values Parsing
print("\n[TEST 4] Track Bias Impact Values Parsing")
test_block_impact = """
Running Style Impact: E 1.30, E/P 1.20, P 0.95, S 0.80
Post Position Impact: Rail 1.41, 1-3 1.10, 4-7 1.05, 8+ 0.95
"""

result_impact = parse_track_bias_impact_values(test_block_impact)
print(f"  Input: Block with Running Style and Post Position Impact Values")
print(f"  Output: {result_impact}")
if result_impact.get("running_style", {}).get("E") == 1.30:
    print("  ✓ PASS: E running style impact parsed")
else:
    print(f"  ✗ FAIL: Expected E=1.30, got {result_impact.get('running_style', {}).get('E')}")

# Test Case 5: SPI Bonus Calculator
print("\n[TEST 5] SPI Bonus Calculation")
bonus_weak = calculate_spi_bonus(0.36, None)  # Weak SPI
bonus_strong = calculate_spi_bonus(1.1, None)  # Strong SPI
print(f"  SPI 0.36 (weak) bonus: {bonus_weak} (expected ≈ -0.05)")
print(f"  SPI 1.1 (strong) bonus: {bonus_strong} (expected ≈ +0.06)")
if bonus_weak < 0:
    print("  ✓ PASS: Weak SPI gets penalty")
else:
    print("  ✗ FAIL: Weak SPI should be negative")

# Test Case 6: Surface Specialty Bonus
print("\n[TEST 6] Surface Specialty Bonus Calculation")
bonus_mud = calculate_surface_specialty_bonus(23, None, None, "muddy", "Dirt")
bonus_turf = calculate_surface_specialty_bonus(None, 35, None, "fast", "Turf")
print(f"  Sire Mud 23% on muddy track: {bonus_mud} (expected ≈ +0.08)")
print(f"  Sire Turf 35% on turf: {bonus_turf} (expected ≈ +0.08)")
if bonus_mud > 0:
    print("  ✓ PASS: Mud specialist gets bonus on muddy track")
else:
    print("  ✗ FAIL: Should get bonus")

# Test Case 7: AWD Mismatch Penalty
print("\n[TEST 7] AWD Mismatch Penalty Calculation")
penalty_small = calculate_awd_mismatch_penalty(5.2, None, "5.5 Furlongs")
penalty_large = calculate_awd_mismatch_penalty(7.5, None, "5.5 Furlongs")
print(f"  Sire AWD 5.2f on 5.5f race: {penalty_small} (expected ≈ 0 or -0.02)")
print(f"  Sire AWD 7.5f on 5.5f race: {penalty_large} (expected ≈ -0.08)")
if penalty_large < -0.07:
    print("  ✓ PASS: Large mismatch gets significant penalty")
else:
    print("  ✗ FAIL: Should get -0.08 penalty")

print("\n" + "="*80)
print("END OF TESTS")
print("="*80)
