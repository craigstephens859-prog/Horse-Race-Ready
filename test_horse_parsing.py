#!/usr/bin/env python3
"""Test horse header regex against sample data"""
import re

# The current regex from app.py
HORSE_HDR_RE = re.compile(
    r"""(?mi)^\s*
    (\d+)              # post/program
    \s+([A-Za-z0-9'.\-\s&]+?)   # horse name
    \s*\(\s*
    (E\/P|EP|E|P|S|NA)      # style
    (?:\s+(\d+))?           # optional quirin
    \s*\)\s*$              #
    """, re.VERBOSE
)

# Sample data from your screenshot
sample_headers = """
4 Zipadeedooda (E 4)
Own: Becky Gibbs
9/2 Yellow, Yellow Cap

5 Lastshotatlightnin (S 3)
Own: J Michael Baird
10/1 Black, Black Cap

6 Zees Clozure (P 4)
Own: Becky Gibbs
4/1 Lime, Lime Cap
"""

print("Testing HORSE_HDR_RE against sample headers:")
print("=" * 60)

matches = list(HORSE_HDR_RE.finditer(sample_headers))
print(f"Found {len(matches)} matches\n")

for i, m in enumerate(matches, 1):
    print(f"Match {i}:")
    print(f"  Post: {m.group(1)}")
    print(f"  Horse: {m.group(2)}")
    print(f"  Style: {m.group(3)}")
    print(f"  Quirin: {m.group(4)}")
    print(f"  Full match: '{m.group(0)}'")
    print()

# Now let's test line-by-line to see which ones match
print("\n" + "=" * 60)
print("Line-by-line analysis:")
print("=" * 60 + "\n")

for line in sample_headers.strip().split('\n'):
    match = HORSE_HDR_RE.match(line)
    status = "✓ MATCH" if match else "✗ NO MATCH"
    print(f"{status:10} | {line}")
