"""
Test script for jockey/trainer parsing from BRISNET PP sample
"""
import re

def parse_jockey_trainer_for_block(block: str, debug: bool = False) -> dict:
    """
    Parses jockey and trainer names from a horse's PP text block.
    
    BRISNET Format:
    - Jockey: "KIMURA KAZUSHI (0 0-0-0 0%)"
    - Trainer: "Trnr: DAmato Philip (0 0-0-0 0%)"
    
    Returns dict with keys: 'jockey', 'trainer'
    """
    result = {
        'jockey': '',
        'trainer': ''
    }
    
    if not block:
        return result
    
    # Parse jockey - appears on a line by itself in ALL CAPS before "Trnr:"
    # Format: "KIMURA KAZUSHI (0 0-0-0 0%)" or "RISPOLI UMBERTO (0 0-0-0 0%)"
    jockey_match = re.search(r'^([A-Z][A-Z\s\'.-]+?)\s*\([\d\s-]+%\)', block, re.MULTILINE)
    if jockey_match:
        jockey_name = jockey_match.group(1).strip()
        # Clean up extra spaces and convert to title case for readability
        jockey_name = ' '.join(jockey_name.split())
        result['jockey'] = jockey_name
        
        if debug:
            print(f"  Jockey found: '{jockey_name}'")
    
    # Parse trainer - appears on line starting with "Trnr:"
    # Format: "Trnr: DAmato Philip (0 0-0-0 0%)"
    trainer_match = re.search(r'Trnr:\s*([A-Za-z][A-Za-z\s,\'.-]+?)\s*\([\d\s-]+%\)', block, re.MULTILINE)
    if trainer_match:
        trainer_name = trainer_match.group(1).strip()
        # Clean up extra spaces
        trainer_name = ' '.join(trainer_name.split())
        result['trainer'] = trainer_name
        
        if debug:
            print(f"  Trainer found: '{trainer_name}'")
    
    return result


# Test data from the sample PP
sample_blocks = {
    "Omnipontet": """1 Omnipontet (S 1)
Own: R Unicorn Stable
20/1 White, Black Blocks, Yellow Stripe And Cuffs On Blue Sleeves,
KIMURA KAZUSHI (0 0-0-0 0%)
Dkbbr. m. 5
Sire : Hat Trick (JPN) (Sunday Silence) $5,000
Dam: =Believable Winner (BRZ) (Put It Back)
Brdr: Haras Springfield (BRZ)
Trnr: DAmato Philip (0 0-0-0 0%)
Prime Power: 131.9 (7th) Life: 11 3 - 3 - 1 $70,038 89""",
    
    "Nay V Belle": """2 Nay V Belle (E/P 3)
Own: Saints Or Sinners
8/1 Red, White Angelic & Dark Horses On Black Ball Front
RISPOLI UMBERTO (0 0-0-0 0%)
Ch. f. 4 OBSAPR 2023 $200k
Sire : Midshipman (Unbridled's Song) $15,000
Dam: Cafe Belle (Medaglia d'Oro)
Brdr: Scott Clarke Reynard Bloodstock NickyDrion S Murat & Mich (KY)
Trnr: Glatt Mark (0 0-0-0 0%)
Prime Power: 139.3 (4th) Life: 13 3 - 4 - 0 $267,960 96""",
    
    "Queen Maxima": """4 Queen Maxima (P 1)
Own: Dutch Girl Holdings Llc And Irving Ventu
5/2 Black, Orange And White Diamonds, Orange 'ai,' Orange
HERNANDEZ JUAN J (0 0-0-0 0%)
Ch. f. 4 OBSOPN 2023 $40k
Sire : Bucchero (Kantharos) $10,000
Dam: Corfu Lady (Corfu)
Brdr: Saul Rosas (FL)
Trnr: Mullins Jeff (0 0-0-0 0%)
Prime Power: 155.2 (1st) Life: 11 6 - 2 - 0 $450,460 99"""
}

# Expected results
expected = {
    "Omnipontet": {"jockey": "KIMURA KAZUSHI", "trainer": "DAmato Philip"},
    "Nay V Belle": {"jockey": "RISPOLI UMBERTO", "trainer": "Glatt Mark"},
    "Queen Maxima": {"jockey": "HERNANDEZ JUAN J", "trainer": "Mullins Jeff"}
}

print("="*60)
print("Testing Jockey/Trainer Parsing")
print("="*60)

all_passed = True
for horse, block in sample_blocks.items():
    print(f"\n🐎 Testing: {horse}")
    print("-"*60)
    
    result = parse_jockey_trainer_for_block(block, debug=True)
    expected_result = expected[horse]
    
    jockey_match = result['jockey'] == expected_result['jockey']
    trainer_match = result['trainer'] == expected_result['trainer']
    
    print(f"  Jockey: {'✅' if jockey_match else '❌'} Expected: '{expected_result['jockey']}', Got: '{result['jockey']}'")
    print(f"  Trainer: {'✅' if trainer_match else '❌'} Expected: '{expected_result['trainer']}', Got: '{result['trainer']}'")
    
    if not (jockey_match and trainer_match):
        all_passed = False

print("\n" + "="*60)
if all_passed:
    print("✅ ALL TESTS PASSED!")
else:
    print("❌ SOME TESTS FAILED")
print("="*60)
