"""
Test Brisnet PP Parser with your example inputs
"""

from brisnet_pp_parser import parse_brisnet_pp_to_json

# Example 1: Handicap race
example1 = """Ultimate PP's w/ QuickPlay Comments Gulfstream Park Hcp 100000 Ì 1m70yds 4&up Saturday, January 24, 2026 Race 6"""

# Example 2: Graded stakes with turf
example2 = """Ultimate PP's w/ QuickPlay Comments Gulfstream Park ™PWCFMTIv-G2 1ˆ Mile (T) 4&up, F & M Saturday, January 24, 2026 Race 10"""

print("=" * 80)
print("EXAMPLE 1: Handicap Race")
print("=" * 80)
print(parse_brisnet_pp_to_json(example1))

print("\n" + "=" * 80)
print("EXAMPLE 2: Grade 2 Stakes (Turf, Fillies & Mares)")
print("=" * 80)
print(parse_brisnet_pp_to_json(example2))
