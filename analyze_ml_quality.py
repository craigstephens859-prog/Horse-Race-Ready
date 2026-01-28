#!/usr/bin/env python3
"""Quick code quality analyzer for ML prediction engine"""

import ast
import re

def analyze_code_quality(filename):
    """Analyze code quality metrics"""
    with open(filename, 'r', encoding='utf-8') as f:
        content = f.read()
        lines = content.split('\n')
    
    issues = []
    
    # Check trailing whitespace
    trailing_ws = sum(1 for line in lines if line and line[-1] in ' \t')
    if trailing_ws > 0:
        issues.append(f"❌ {trailing_ws} lines with trailing whitespace")
    
    # Check long lines
    long_lines = sum(1 for line in lines if len(line) > 120)
    if long_lines > 0:
        issues.append(f"⚠️  {long_lines} lines exceeding 120 chars")
    
    # Check docstrings
    try:
        tree = ast.parse(content)
        functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
        classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]
        
        funcs_without_docs = sum(1 for f in functions if not ast.get_docstring(f))
        classes_without_docs = sum(1 for c in classes if not ast.get_docstring(c))
        
        if funcs_without_docs > 0:
            issues.append(f"⚠️  {funcs_without_docs}/{len(functions)} functions missing docstrings")
        if classes_without_docs > 0:
            issues.append(f"⚠️  {classes_without_docs}/{len(classes)} classes missing docstrings")
    except SyntaxError as e:
        issues.append(f"❌ Syntax error: {e}")
    
    # Check for bare excepts
    bare_excepts = len(re.findall(r'except\s*:', content))
    if bare_excepts > 0:
        issues.append(f"❌ {bare_excepts} bare except clauses")
    
    # Check for f-strings without interpolation
    bad_fstrings = len(re.findall(r'f["\'][^"\']*["\']', content))
    if bad_fstrings > 5:  # Allow some in docstrings
        issues.append(f"⚠️  {bad_fstrings} potential f-strings without interpolation")
    
    # Calculate score
    total_deductions = len(issues) * 0.5
    base_score = 10.0
    final_score = max(5.0, base_score - total_deductions)
    
    print(f"\n{'='*80}")
    print(f"CODE QUALITY ANALYSIS: {filename}")
    print(f"{'='*80}")
    print(f"\n📊 SCORE: {final_score:.2f}/10.0")
    print(f"\n🔍 ISSUES FOUND ({len(issues)}):")
    for issue in issues:
        print(f"  {issue}")
    
    if final_score >= 9.0:
        print(f"\n✅ GRADE: A+ (EXCELLENT)")
    elif final_score >= 8.0:
        print(f"\n✅ GRADE: A (GOOD)")
    elif final_score >= 7.0:
        print(f"\n⚠️  GRADE: B (ACCEPTABLE)")
    else:
        print(f"\n❌ GRADE: C/D (NEEDS WORK)")
    
    print(f"{'='*80}\n")
    return final_score

if __name__ == "__main__":
    analyze_code_quality("ml_prediction_engine_elite.py")
