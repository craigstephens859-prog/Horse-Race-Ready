import re

text = """28Oct Mnr 4f ft :47ª B
07Apr Mnr (d) 4f my :49« B
01Apr Mnr 5f ft 1:03© Bg
25Apr CT 4f ft :49© B"""

# Test different patterns
patterns = [
    r'(?m)^\d{2}[A-Za-z]{3}\s+\w+\s+\d+f.*?Bg?(?:\s|$)',
    r'(?m)^\d{2}[A-Za-z]{3}.*?\d+f.*?\sB(?:g|\s|$)',
    r'(?m)^\d{2}[A-Za-z]{3}.*?\sB(?:g|\b)',
]

for i, pattern in enumerate(patterns, 1):
    matches = re.findall(pattern, text)
    print(f"Pattern {i}: Found {len(matches)} bullets")
    for m in matches[:5]:
        print(f"  - {repr(m[:60])}")
    print()
