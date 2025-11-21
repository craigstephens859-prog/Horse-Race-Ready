"""
Test script for running style parsing from BRISNET PP sample
"""
import re

def parse_running_style_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses running style from a horse's PP text block header.
    
    BRISNET Format in header line:
    - "1 Omnipontet (S 1)" - S = Sustained
    - "2 Nay V Belle (E/P 3)" - E/P = Early/Presser
    - "4 Queen Maxima (P 1)" - P = Presser
    - "5 Sunglow (E 8)" - E = Early
    - "7 Puro Magic (NA 0)" - NA = Not Available
    
    Returns dict with key: 'running_style'
    """
    result = {
        'running_style': ''
    }
    
    if not block:
        return result
    
    # Parse running style from header - appears in parentheses after horse name
    # Format: "Post# HorseName (STYLE #)"
    # Style can be: E, E/P, P, S, or NA
    style_match = re.search(r'^\s*\d+\s+[A-Za-z\s=\'-]+\s+\(([A-Z/]+)\s+\d+\)', block, re.MULTILINE)
    if style_match:
        running_style = style_match.group(1).strip()
        result['running_style'] = running_style
        
        if debug:
            print(f"  Running Style found: '{running_style}'")
    
    return result


# Test data from the sample PP
sample_blocks = {
    "Omnipontet": """1 Omnipontet (S 1)
Own: R Unicorn Stable
20/1 White, Black Blocks, Yellow Stripe And Cuffs On Blue Sleeves,
KIMURA KAZUSHI (0 0-0-0 0%)""",
    
    "Nay V Belle": """2 Nay V Belle (E/P 3)
Own: Saints Or Sinners
8/1 Red, White Angelic & Dark Horses On Black Ball Front
RISPOLI UMBERTO (0 0-0-0 0%)""",
    
    "Tahini": """3 Tahini (E/P 7)
Own: Kretz Racing Llc
30/1 Black, White 'scorpion' And Emblem On Back Black Bars
DEMURO MIRCO (0 0-0-0 0%)""",
    
    "Queen Maxima": """4 Queen Maxima (P 1)
Own: Dutch Girl Holdings Llc And Irving Ventu
5/2 Black, Orange And White Diamonds, Orange 'ai,' Orange
HERNANDEZ JUAN J (0 0-0-0 0%)""",
    
    "Sunglow": """5 Sunglow (E 8)
Own: Diamond T Racing Llc Hoffman Thoroughbre
15/1 Burgundy, Gold Bar On Navy Sleeves, Navy Cap
ROSARIO JOEL (0 0-0-0 0%)""",
    
    "Shoot It True": """6 Shoot It True (E/P 4)
Own: Ice Wine Stable Or Smart Cookie Stable
3/1 Burgundy And Light Blue Diamonds, Light Blue Bars On
MACHADO LUAN (0 0-0-0 0%)""",
    
    "Puro Magic": """7 Puro Magic (NA 0)
Own: Three H Racing
4/1 Pink, Blue Sash, White Sleeves, White Cap
YOSHIHARA HIROTO (0 0-0-0 0%)""",
    
    "Jungle Peace": """8 Jungle Peace (E/P 5)
Own: Cybt Mclean Racing Stables Or Mcclanahan
10/1 Silver & White Halves, White Circle "n" On Black Ball, Black,
PRAT FLAVIEN (0 0-0-0 0%)""",
    
    "Great Venezuela": """9 Great Venezuela (E/P 4)
Own: Orlyana Farm
6/1 Red And Blue Halves, Yellow Emblem, Blue Sleeves, Yellow
ALVARADO JUNIOR (0 0-0-0 0%)""",
    
    "Marian Cross": """10 Marian Cross (E 6)
Own: Qatar Racing Llc
20/1 Claret, Gold Frogs, Claret Cap
MURPHY OISIN (0 0-0-0 0%)"""
}

# Expected results
expected = {
    "Omnipontet": "S",
    "Nay V Belle": "E/P",
    "Tahini": "E/P",
    "Queen Maxima": "P",
    "Sunglow": "E",
    "Shoot It True": "E/P",
    "Puro Magic": "NA",
    "Jungle Peace": "E/P",
    "Great Venezuela": "E/P",
    "Marian Cross": "E"
}

print("="*60)
print("Testing Running Style Parsing")
print("="*60)

all_passed = True
for horse, block in sample_blocks.items():
    print(f"\n🐎 Testing: {horse}")
    print("-"*60)
    
    result = parse_running_style_for_block(block, debug=True)
    expected_style = expected[horse]
    
    style_match = result['running_style'] == expected_style
    
    print(f"  Running Style: {'✅' if style_match else '❌'} Expected: '{expected_style}', Got: '{result['running_style']}'")
    
    if not style_match:
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("✅ ALL TESTS PASSED!")
    print(f"Successfully parsed {len(sample_blocks)} horses")
else:
    print("❌ SOME TESTS FAILED")
print("="*60)
