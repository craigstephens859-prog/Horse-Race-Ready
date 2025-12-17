#!/usr/bin/env python3
"""Test with full multi-horse PP text"""
import re

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

def split_into_horse_chunks(pp_text: str):
    chunks = []
    matches = list(HORSE_HDR_RE.finditer(pp_text or ""))
    for i, m in enumerate(matches):
        start = m.end()
        end = matches[i+1].start() if i+1 < len(matches) else len(pp_text)
        post = m.group(1).strip()
        name = m.group(2).strip()
        block = pp_text[start:end]
        chunks.append((post, name, block))
    return chunks

# Full PP text - horses 1, 2, 3 shown in table, then horses 4, 5, 6 as separate detailed blocks
full_text = """
Race 2 Ultimate PP's Mountaineer ™'Mdn 16.5k 5½ Furlongs 3&up, F & M Wednesday, August 20, 2025

1 Way of Appeal (S 3)
Own: Someone
9/2 odds

2 Spuns Kitten (S 3)
Own: Someone
3/1 odds

3 Emily Katherine (E 5)
Own: Someone
2/1 odds

4 Zipadeedooda (E 4)
Own: Becky Gibbs
9/2 Yellow, Yellow Cap
TAPARA BRANDON (121 17-15-21 14%)
Ch. f. 3 (Apr)
Sire : Unbridled Energy (Unbridled's Song)
Prime Power: 85.9 (4th) Life: 8 0 - 1 - 1 $4,345 51

5 Lastshotatlightnin (S 3)
Own: J Michael Baird
10/1 Black, Black Cap
BARBARAN ERIK (142 29-26-21 20%)
Ch. f. 4 (Mar)
Sire : Cal Nation (Distorted Humor) $2,500
Prime Power: 78.6 (6th) Life: 12 0 - 0 - 2 $6,585 52

6 Zees Clozure (P 4)
Own: Becky Gibbs
4/1 Lime, Lime Cap
STOKES JOE (90 14-9-15 16%)
Ch. f. 3 (May)
Sire : Unbridled Energy (Unbridled's Song)
Prime Power: 86.3 (3rd) Life: 1 0 - 0 - 0 $825 46
"""

print("Testing split_into_horse_chunks with full race text:")
print("=" * 70)

chunks = split_into_horse_chunks(full_text)
print(f"Found {len(chunks)} horse chunks\n")

for post, name, block in chunks:
    print(f"Post #{post} - {name}")
    print(f"  Block length: {len(block)} chars")
    print(f"  First 100 chars: {block[:100]!r}")
    print()

print("=" * 70)
print("\nAll matches found by regex:")
for m in HORSE_HDR_RE.finditer(full_text):
    print(f"  Post {m.group(1)} - {m.group(2).strip()} ({m.group(3)} {m.group(4) or ''})")
