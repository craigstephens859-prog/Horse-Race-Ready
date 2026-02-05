"""Find all broken f-strings in app.py"""
import re

with open('app.py', 'r', encoding='utf-8') as f:
    lines = f.readlines()

broken_fstrings = []
i = 0
while i < len(lines):
    line = lines[i]
    # Look for f" followed by { at end of line (incomplete f-string)
    if 'f"' in line and line.rstrip().endswith('{'):
        broken_fstrings.append({
            'line_num': i + 1,
            'context': line.strip()[:100],
            'next_lines': ''.join(lines[i+1:min(i+5, len(lines))])[:200]
        })
    # Also check for st.info/success/error with opening paren but f" on next line
    if re.search(r'st\.(info|success|error|warning)\(\s*$', line):
        if i+1 < len(lines) and lines[i+1].strip().startswith('f"'):
            broken_fstrings.append({
                'line_num': i + 1,
                'context': f'st.X( -> {lines[i+1].strip()[:80]}',
                'next_lines': ''.join(lines[i+1:min(i+5, len(lines))])[:200]
            })
    i += 1

print(f'ULTRATHINK F-STRING ANALYSIS')
print('=' * 70)
print(f'Found {len(broken_fstrings)} potentially broken f-strings:\n')

for item in broken_fstrings:
    print(f'Line {item["line_num"]}:')
    print(f'  {item["context"]}')
    print()
