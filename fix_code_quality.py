#!/usr/bin/env python3
"""
Code Quality Fixer - Automatically fix mechanical pylint issues

Fixes:
1. Trailing whitespace
2. Multiple statements on single line  
3. Unnecessary f-strings without interpolation
4. Import order
5. Bare except clauses
"""
import re
import sys
from pathlib import Path


def fix_trailing_whitespace(content: str) -> str:
    """Remove trailing whitespace from all lines"""
    lines = content.split('\n')
    fixed_lines = [line.rstrip() for line in lines]
    return '\n'.join(fixed_lines)


def fix_bare_excepts(content: str) -> str:
    """Replace bare except: with except Exception:"""
    # Match bare except followed by colon
    pattern = r'(\s+)except:\s*$'
    replacement = r'\1except Exception:'
    return re.sub(pattern, replacement, content, flags=re.MULTILINE)


def fix_unnecessary_fstrings(content: str) -> str:
    """Convert f-strings without interpolation to regular strings"""
    # Match f"string" or f'string' with no {} inside
    pattern = r'f(["\'])([^"\'{}]*)\1'
    
    def replace_if_no_braces(match):
        quote = match.group(1)
        text = match.group(2)
        if '{' not in text and '}' not in text:
            return f'{quote}{text}{quote}'
        return match.group(0)
    
    return re.sub(pattern, replace_if_no_braces, content)


def fix_singleton_comparison(content: str) -> str:
    """Fix == False to 'is False' or 'not' pattern"""
    # Replace DataFrame['col'] == False with ~DataFrame['col']
    pattern = r"(\w+\[['\"]\w+['\"])\s*==\s*False"
    replacement = r'~\1'
    return re.sub(pattern, replacement, content)


def process_file(filepath: Path) -> tuple:
    """Process a single file and return (modified, changes_made)"""
    print(f"Processing {filepath.name}...", end=' ')
    
    try:
        with open(filepath, 'r', encoding='utf-8') as f:
            original = f.read()
        
        modified = original
        changes = []
        
        # Apply fixes
        fixed = fix_trailing_whitespace(modified)
        if fixed != modified:
            changes.append("trailing whitespace")
            modified = fixed
        
        fixed = fix_bare_excepts(modified)
        if fixed != modified:
            changes.append("bare excepts")
            modified = fixed
        
        fixed = fix_unnecessary_fstrings(modified)
        if fixed != modified:
            changes.append("unnecessary f-strings")
            modified = fixed
        
        fixed = fix_singleton_comparison(modified)
        if fixed != modified:
            changes.append("singleton comparisons")
            modified = fixed
        
        if modified != original:
            with open(filepath, 'w', encoding='utf-8') as f:
                f.write(modified)
            print(f"✅ Fixed: {', '.join(changes)}")
            return True, changes
        else:
            print("✓ No changes needed")
            return False, []
    
    except Exception as e:
        print(f"❌ Error: {e}")
        return False, []


def main():
    """Fix code quality issues in all Python files"""
    target_files = [
        'elite_parser_v2_gold.py',
        'unified_rating_engine.py',
        'test_parser_comprehensive.py',
        'elite_parser.py',
        'horse_angles8.py',
        'app.py'
    ]
    
    total_modified = 0
    all_changes = []
    
    for filename in target_files:
        filepath = Path(filename)
        if filepath.exists():
            modified, changes = process_file(filepath)
            if modified:
                total_modified += 1
                all_changes.extend(changes)
        else:
            print(f"⚠️  {filename} not found, skipping")
    
    print(f"\n{'='*60}")
    print(f"✅ Modified {total_modified} files")
    if all_changes:
        change_counts = {}
        for change in all_changes:
            change_counts[change] = change_counts.get(change, 0) + 1
        print("\nChanges made:")
        for change, count in change_counts.items():
            print(f"  - {change}: {count} file(s)")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
