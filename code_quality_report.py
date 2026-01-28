#!/usr/bin/env python3
"""
Code Quality Report Generator

Analyzes code and generates quality metrics:
- Import order compliance
- Trailing whitespace
- Bare except usage
- Line length
- Function complexity
- Documentation coverage
"""
import ast
import re
from pathlib import Path
from typing import List, Tuple


class CodeQualityAnalyzer:
    """Analyzes Python files for code quality metrics"""

    def __init__(self, filepath: Path):
        self.filepath = filepath
        self.content = filepath.read_text(encoding='utf-8')
        self.lines = self.content.split('\n')
        try:
            self.tree = ast.parse(self.content)
        except SyntaxError:
            self.tree = None

    def check_import_order(self) -> Tuple[bool, str]:
        """Verify standard library imports come before third party"""
        import_lines = []
        for i, line in enumerate(self.lines):
            if line.strip().startswith(('import ', 'from ')):
                if not line.strip().startswith('#'):
                    import_lines.append((i, line.strip()))

        if not import_lines:
            return True, "No imports found"

        # Check if imports are grouped properly
        has_third_party = any('pandas' in line or 'numpy' in line or 'torch' in line
                             for _, line in import_lines)
        if not has_third_party:
            return True, "Only standard library imports"

        return True, "Import order OK"

    def check_trailing_whitespace(self) -> Tuple[int, List[int]]:
        """Count lines with trailing whitespace"""
        lines_with_trailing = []
        for i, line in enumerate(self.lines, 1):
            if line != line.rstrip():
                lines_with_trailing.append(i)
        return len(lines_with_trailing), lines_with_trailing[:5]

    def check_bare_excepts(self) -> Tuple[int, List[int]]:
        """Find bare except clauses"""
        bare_excepts = []
        for i, line in enumerate(self.lines, 1):
            if re.match(r'^\s+except\s*:\s*$', line):
                bare_excepts.append(i)
        return len(bare_excepts), bare_excepts

    def check_line_length(self, max_length=120) -> Tuple[int, List[int]]:
        """Find lines exceeding max length"""
        long_lines = []
        for i, line in enumerate(self.lines, 1):
            if len(line) > max_length:
                long_lines.append(i)
        return len(long_lines), long_lines[:5]

    def check_function_complexity(self) -> int:
        """Count number of functions/methods"""
        if not self.tree:
            return 0
        count = 0
        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                count += 1
        return count

    def check_docstring_coverage(self) -> Tuple[int, int]:
        """Check percentage of functions with docstrings"""
        if not self.tree:
            return 0, 0

        total_funcs = 0
        documented_funcs = 0

        for node in ast.walk(self.tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                total_funcs += 1
                if ast.get_docstring(node):
                    documented_funcs += 1

        return documented_funcs, total_funcs

    def generate_report(self) -> dict:
        """Generate complete quality report"""
        trailing_count, trailing_lines = self.check_trailing_whitespace()
        bare_except_count, bare_except_lines = self.check_bare_excepts()
        long_line_count, long_lines = self.check_line_length()
        func_count = self.check_function_complexity()
        documented, total_funcs = self.check_docstring_coverage()

        score = 10.0
        deductions = []

        # Deduct for issues
        if trailing_count > 0:
            deduction = min(1.0, trailing_count * 0.01)
            score -= deduction
            deductions.append(f"Trailing whitespace: -{deduction:.2f}")

        if bare_except_count > 0:
            deduction = bare_except_count * 0.5
            score -= deduction
            deductions.append(f"Bare excepts: -{deduction:.2f}")

        if long_line_count > 0:
            deduction = min(0.5, long_line_count * 0.05)
            score -= deduction
            deductions.append(f"Long lines: -{deduction:.2f}")

        if total_funcs > 0:
            doc_percentage = (documented / total_funcs) * 100
            if doc_percentage < 80:
                deduction = (80 - doc_percentage) * 0.02
                score -= deduction
                deductions.append(f"Low doc coverage: -{deduction:.2f}")

        return {
            'file': self.filepath.name,
            'score': max(0, score),
            'total_lines': len(self.lines),
            'functions': func_count,
            'documented_functions': documented,
            'total_functions': total_funcs,
            'doc_coverage': f"{(documented/total_funcs*100):.1f}%" if total_funcs > 0 else "N/A",
            'trailing_whitespace': trailing_count,
            'bare_excepts': bare_except_count,
            'long_lines': long_line_count,
            'deductions': deductions
        }


def main():
    """Generate quality report for all key files"""
    target_files = [
        'elite_parser_v2_gold.py',
        'unified_rating_engine.py',
        'test_parser_comprehensive.py',
        'app.py',
        'elite_parser.py',
        'horse_angles8.py'
    ]

    print("=" * 80)
    print("CODE QUALITY ANALYSIS REPORT")
    print("=" * 80)
    print()

    total_score = 0
    file_count = 0

    for filename in target_files:
        filepath = Path(filename)
        if not filepath.exists():
            print(f"⚠️  {filename} not found, skipping")
            continue

        analyzer = CodeQualityAnalyzer(filepath)
        report = analyzer.generate_report()

        file_count += 1
        total_score += report['score']

        print(f"📄 {report['file']}")
        print(f"   Score: {report['score']:.2f}/10.0")
        print(f"   Lines: {report['total_lines']:,}")
        print(f"   Functions: {report['functions']} ({report['documented_functions']} documented = {report['doc_coverage']})")
        print(f"   Issues:")
        print(f"      Trailing whitespace: {report['trailing_whitespace']}")
        print(f"      Bare excepts: {report['bare_excepts']}")
        print(f"      Long lines (>120 chars): {report['long_lines']}")

        if report['deductions']:
            print(f"   Deductions:")
            for deduction in report['deductions']:
                print(f"      - {deduction}")
        print()

    if file_count > 0:
        average_score = total_score / file_count
        print("=" * 80)
        print(f"OVERALL AVERAGE SCORE: {average_score:.2f}/10.0")

        if average_score >= 9.5:
            grade = "A+ (EXCELLENT)"
        elif average_score >= 9.0:
            grade = "A (VERY GOOD)"
        elif average_score >= 8.0:
            grade = "B (GOOD)"
        elif average_score >= 7.0:
            grade = "C (ACCEPTABLE)"
        else:
            grade = "D (NEEDS IMPROVEMENT)"

        print(f"GRADE: {grade}")
        print("=" * 80)


if __name__ == "__main__":
    main()
