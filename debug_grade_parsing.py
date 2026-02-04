#!/usr/bin/env python3
"""Debug graded race parsing"""

from race_class_parser import parse_brisnet_header, parse_race_conditions, get_hierarchy_level

# Test PP text with Grade 1
pp_text = "Ultimate PP's w/ QuickPlay Comments | Churchill Downs | Grade1 1000000 | 1 1/4 Mile | 3yo | Race 11"

print("Testing Grade 1 parsing:")
print("=" * 80)
print(f"PP Text: {pp_text}")
print()

# Step 1: Parse header
print("Step 1: Parse header")
header = parse_brisnet_header(pp_text)
print(f"Header: {header}")
print()

# Step 2: Parse race conditions
print("Step 2: Parse race conditions")
race_type = header.get('race_type', '') or header.get('race_conditions', '')
purse = header.get('purse_amount', 0)
print(f"Input - race_type: '{race_type}', purse: {purse}")

conditions = parse_race_conditions(race_type, purse)
print(f"Conditions output: {conditions}")
print()

# Step 3: Get hierarchy level
print("Step 3: Get hierarchy level")
class_type = conditions.get('class_type', 'Unknown')
grade_level = conditions.get('grade_level')
purse_amt = conditions.get('purse_amount', 0)
print(f"Input - class_type: '{class_type}', grade_level: {grade_level}, purse: {purse_amt}")

hierarchy = get_hierarchy_level(class_type, grade_level, purse_amt)
print(f"Hierarchy output: {hierarchy}")
