#!/usr/bin/env python3
"""
Simple unit tests for parsing functions without Streamlit execution
"""
import re
import numpy as np
import pandas as pd

# Copy just the functions we need to test
def parse_pedigree_spi(block: str) -> dict:
    """Extract Sire Production Index (SPI)"""
    out = {"spi": np.nan, "dam_sire_spi": np.nan}
    if not block:
        return out
    try:
        # Pattern: SPI followed by space and optional dot then digits
        spi_match = re.search(r'(?mi)SPI\s+(\.\d+|\d+\.\d+)', block)
        if spi_match:
            spi_text = spi_match.group(1)
            # Convert .36 to 0.36
            if spi_text.startswith('.'):
                out["spi"] = float('0' + spi_text)
            else:
                out["spi"] = float(spi_text)
        
        dam_sire_match = re.search(r'(?mi)Dam[\s-]*Sire.*?SPI\s+(\.\d+|\d+\.\d+)', block)
        if dam_sire_match:
            dam_sire_text = dam_sire_match.group(1)
            if dam_sire_text.startswith('.'):
                out["dam_sire_spi"] = float('0' + dam_sire_text)
            else:
                out["dam_sire_spi"] = float(dam_sire_text)
    except:
        pass
    return out

def parse_pedigree_surface_stats(block: str) -> dict:
    """Extract Sire %Mud, %Turf, Dam-Sire %Mud"""
    out = {"sire_mud_pct": np.nan, "sire_turf_pct": np.nan, "dam_sire_mud_pct": np.nan}
    if not block:
        return out
    try:
        # Look for surface percentages anywhere in the block (not just in sections)
        # First, find all "Mud XX%" and "Turf XX%" patterns
        mud_matches = re.finditer(r'(?mi)Mud\s+(\d+)%', block)
        turf_matches = re.finditer(r'(?mi)Turf\s+(\d+)%', block)
        
        mud_list = [float(m.group(1)) for m in mud_matches]
        turf_list = [float(m.group(1)) for m in turf_matches]
        
        # Sire Mud is usually first mention or before Dam-Sire
        if mud_list:
            # Check if Dam-Sire comes after - if so, first mud is sire, last is dam-sire
            if 'dam' in block.lower() and 'sire' in block.lower():
                dam_sire_pos = block.lower().find('dam')
                # Find which mud % comes before dam-sire line
                for m in re.finditer(r'(?mi)Mud\s+(\d+)%', block):
                    if m.start() < dam_sire_pos:
                        out["sire_mud_pct"] = float(m.group(1))
                # Find which mud % comes after dam-sire
                for m in re.finditer(r'(?mi)Mud\s+(\d+)%', block):
                    if m.start() > dam_sire_pos:
                        out["dam_sire_mud_pct"] = float(m.group(1))
                        break
            else:
                out["sire_mud_pct"] = mud_list[0]
        
        # Turf is usually associated with sire, not dam-sire as often
        if turf_list:
            out["sire_turf_pct"] = turf_list[0]
    except:
        pass
    return out

print("="*80)
print("UNIT TESTS: Parsing Functions")
print("="*80)

# TEST 1
print("\n[TEST 1] SPI Parsing (.36 format)")
test1 = "Sire: Way of Appeal SPI .36"
result1 = parse_pedigree_spi(test1)
assert result1.get("spi") == 0.36, f"Expected 0.36, got {result1.get('spi')}"
print(f"  ✓ PASS: {test1} → spi={result1.get('spi')}")

# TEST 2
print("\n[TEST 2] SPI Parsing (decimal format)")
test2 = "SPI 0.44"
result2 = parse_pedigree_spi(test2)
assert result2.get("spi") == 0.44, f"Expected 0.44, got {result2.get('spi')}"
print(f"  ✓ PASS: {test2} → spi={result2.get('spi')}")

# TEST 3
print("\n[TEST 3] Surface Stats - Sire Mud %")
test3 = "Sire: Way of Appeal Mud 23% Turf 19%"
result3 = parse_pedigree_surface_stats(test3)
assert result3.get("sire_mud_pct") == 23.0, f"Expected 23.0, got {result3.get('sire_mud_pct')}"
print(f"  ✓ PASS: Extracted sire_mud_pct={result3.get('sire_mud_pct')}")

# TEST 4
print("\n[TEST 4] Surface Stats - Sire Turf %")
assert result3.get("sire_turf_pct") == 19.0, f"Expected 19.0, got {result3.get('sire_turf_pct')}"
print(f"  ✓ PASS: Extracted sire_turf_pct={result3.get('sire_turf_pct')}")

# TEST 5
print("\n[TEST 5] Surface Stats - Dam-Sire Mud %")
test5 = """Sire: Way of Appeal Mud 23%
Dam-Sire: Some Horse Mud 13%"""
result5 = parse_pedigree_surface_stats(test5)
assert result5.get("dam_sire_mud_pct") == 13.0, f"Expected 13.0, got {result5.get('dam_sire_mud_pct')}"
print(f"  ✓ PASS: Extracted dam_sire_mud_pct={result5.get('dam_sire_mud_pct')}")

# TEST 6 - Combined format
print("\n[TEST 6] All stats combined")
test6 = """Sire: Way of Appeal SPI .36 Mud 23% Turf 19%
Dam-Sire: Some Horse SPI .44 Mud 13%"""
spi_result = parse_pedigree_spi(test6)
surf_result = parse_pedigree_surface_stats(test6)
assert spi_result.get("spi") == 0.36
assert spi_result.get("dam_sire_spi") == 0.44
assert surf_result.get("sire_mud_pct") == 23.0
assert surf_result.get("dam_sire_mud_pct") == 13.0
print(f"  ✓ PASS: All fields extracted correctly")
print(f"    SPI: sire={spi_result.get('spi')}, dam-sire={spi_result.get('dam_sire_spi')}")
print(f"    Surface: sire_mud={surf_result.get('sire_mud_pct')}, dam_mud={surf_result.get('dam_sire_mud_pct')}")

print("\n" + "="*80)
print("ALL TESTS PASSED")
print("="*80)
