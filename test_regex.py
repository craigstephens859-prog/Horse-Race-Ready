import re

# Sample line from debug output
test_line = "11Oct25SA« 1m ft :22« :46ª1:11 1:36« | ¨¨¬ OC50k/n1x-N ¨¨ 92 101/ 93 +4 +4 98 2 4© 4© 3¨ 2¨ 2¨"

print("Test line:")
print(test_line)
print()

# Show what's after the pipe
pipe_idx = test_line.find('|')
if pipe_idx >= 0:
    after_pipe = test_line[pipe_idx:]
    print(f"After pipe: {after_pipe}")
    print()
    
    # Show character codes
    print("First 50 chars after pipe with codes:")
    for i, c in enumerate(after_pipe[:50]):
        print(f"  [{i}] '{c}' = {ord(c)}")
    print()

# Try current regex
pattern = re.compile(
    r"\|"
    r"(?:[^\d]|\d(?!/\s*\d{2,3}\s+[+-]))*?"
    r"(\d{2,3})\s+"
    r"(\d{2,3})/?\s+"
    r"(\d{2,3})\s+"
    r"[+-]?\d+\s+"
    r"[+-]?\d+\s+"
    r"(\d{2,3})"
    r"\s+\d(?:\s|©|ª|«|¬)"
)

matches = list(pattern.finditer(test_line))
print(f"Current regex matches: {len(matches)}")
for m in matches:
    print(f"  Match: {m.group(0)}")
    print(f"  Groups: {m.groups()}")
print()

# Try simpler pattern
simple_pattern = re.compile(r"\|\s+(.{60})")
simple_match = simple_pattern.search(test_line)
if simple_match:
    print(f"60 chars after pipe: '{simple_match.group(1)}'")
print()

# Try to find the number sequence manually
number_pattern = re.compile(r"(\d{2,3})\s+(\d{2,3})/?\s+(\d{2,3})\s+[+-]?\d+\s+[+-]?\d+\s+(\d{2,3})")
number_matches = list(number_pattern.finditer(after_pipe))
print(f"Direct number pattern matches in text after pipe: {len(number_matches)}")
for m in number_matches:
    print(f"  Match: {m.group(0)}")
    print(f"  RR={m.group(1)}, CR={m.group(2)}, E1={m.group(3)}, SPD={m.group(4)}")
