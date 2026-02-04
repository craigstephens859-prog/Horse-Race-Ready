"""
Comprehensive Brisnet Ultimate Past Performances Parser

Parses pasted text from Brisnet PP files or screenshots into structured JSON.
Handles copyright notices, headers, and rating tables.
"""

import re
import json
from typing import Dict, List, Any, Optional
from race_class_parser import parse_brisnet_header, parse_race_conditions


def parse_brisnet_pp_to_json(pp_text: str) -> str:
    """
    Parse Brisnet PP text into structured JSON.
    
    Args:
        pp_text: Raw pasted text from Brisnet PP
        
    Returns:
        JSON string with parsed data
    """
    result = parse_brisnet_pp(pp_text)
    return json.dumps(result, indent=2)


def parse_brisnet_pp(pp_text: str) -> Dict[str, Any]:
    """
    Parse Brisnet PP text into structured dictionary.
    
    Args:
        pp_text: Raw pasted text from Brisnet PP
        
    Returns:
        Dictionary with header and categories
    """
    result: Dict[str, Any] = {
        "header": {},
        "categories": []
    }
    
    lines = pp_text.split('\n')
    
    # Step 1: Extract copyright notice
    copyright_notice = extract_copyright(lines)
    if copyright_notice:
        result["header"]["copyright_notice"] = copyright_notice
    
    # Step 2: Parse main header
    header_data = parse_header(pp_text)
    result["header"].update(header_data)
    
    # Step 3: Parse rating tables
    categories = parse_rating_tables(pp_text)
    result["categories"] = categories
    
    return result


def extract_copyright(lines: List[str]) -> Optional[str]:
    """
    Extract copyright notice from first few lines.
    
    Args:
        lines: List of text lines
        
    Returns:
        Copyright text or None
    """
    for line in lines[:5]:
        if re.search(r'\(c\)\s*Copyright|\u00a9\s*Copyright|Copyright\s+\d{4}', line, re.IGNORECASE):
            return line.strip()
    return None


def parse_header(pp_text: str) -> Dict[str, Any]:
    """
    Parse the main header line using race_class_parser.
    
    Args:
        pp_text: Raw PP text
        
    Returns:
        Dictionary with header fields
    """
    header_result: Dict[str, Any] = {}
    
    try:
        # Use existing race_class_parser
        parsed = parse_brisnet_header(pp_text)
        
        # Map to output format
        header_result["product_type"] = "Ultimate PP's w/ QuickPlay Comments"
        header_result["track"] = parsed.get("track_name", "")
        
        # Clean up race conditions (remove symbols)
        race_cond = parsed.get("race_type", "")
        header_result["race_conditions"] = clean_race_conditions(race_cond)
        header_result["race_conditions_raw"] = race_cond
        
        header_result["distance"] = parsed.get("distance", "")
        header_result["distance_furlongs"] = parsed.get("distance_furlongs", 0.0)
        
        # Combine age/sex
        age = parsed.get("age_restriction", "")
        header_result["age_sex"] = age
        
        # Format date
        day = parsed.get("day_of_week", "")
        date = parsed.get("race_date", "")
        if day and date:
            header_result["date"] = f"{day}, {date}"
        elif date:
            header_result["date"] = date
        else:
            header_result["date"] = ""
        
        header_result["race_number"] = parsed.get("race_number", 0)
        
        # Add parsed class data
        conditions = parse_race_conditions(race_cond, 0)
        header_result["class_type"] = conditions.get("class_type", "Unknown")
        header_result["grade_level"] = conditions.get("grade_level")
        header_result["is_stakes"] = conditions.get("is_stakes", False)
        header_result["is_graded"] = conditions.get("is_graded_stakes", False)
        
    except Exception as e:
        header_result["parse_error"] = str(e)
    
    return header_result


def clean_race_conditions(race_cond: str) -> str:
    """
    Clean up race condition symbols and abbreviations.
    
    Args:
        race_cond: Raw race conditions string
        
    Returns:
        Cleaned conditions string
    """
    # Remove copyright symbols that prefix race types
    cleaned = race_cond.replace('©', '').replace('™', '').replace('®', '')
    
    # Expand common abbreviations
    expansions = {
        'Hcp': 'Handicap',
        'Clm': 'Claiming',
        'Mdn': 'Maiden',
        'Alw': 'Allowance',
        'Stk': 'Stakes',
        'Inv': 'Invitational',
    }
    
    for abbr, full in expansions.items():
        if abbr in cleaned:
            cleaned = cleaned.replace(abbr, full)
    
    return cleaned.strip()


def parse_rating_tables(pp_text: str) -> List[Dict[str, Any]]:
    """
    Parse rating tables (Speed Last Race, Prime Power, etc.).
    
    Args:
        pp_text: Raw PP text
        
    Returns:
        List of category dictionaries
    """
    categories: List[Dict[str, Any]] = []
    
    # Common rating table headers
    table_patterns = [
        r'#\s*Speed\s+Last\s+Race',
        r'#\s*Prime\s+Power',
        r'#\s*Class\s+Rating',
        r'#\s*Best\s+Speed\s+at\s+Dist',
        r'Speed\s+Last\s+Race',
        r'Prime\s+Power',
        r'Class\s+Rating',
    ]
    
    lines = pp_text.split('\n')
    
    for i, line in enumerate(lines):
        # Check if this line is a table header
        for pattern in table_patterns:
            if re.search(pattern, line, re.IGNORECASE):
                category_name = extract_category_name(line)
                rankings = extract_rankings(lines, i + 1)
                
                if rankings:
                    categories.append({
                        "category_name": category_name,
                        "rankings": rankings
                    })
                break
    
    return categories


def extract_category_name(line: str) -> str:
    """
    Extract clean category name from header line.
    
    Args:
        line: Header line
        
    Returns:
        Clean category name
    """
    # Remove leading # symbols and clean up
    name = re.sub(r'^#\s*', '', line)
    name = re.sub(r'\s+', ' ', name).strip()
    return name


def extract_rankings(lines: List[str], start_idx: int) -> List[Dict[str, Any]]:
    """
    Extract ranking rows following a category header.
    
    Args:
        lines: All text lines
        start_idx: Index to start reading from
        
    Returns:
        List of ranking dictionaries
    """
    rankings: List[Dict[str, Any]] = []
    
    # Read next 5-20 lines looking for ranking patterns
    for i in range(start_idx, min(start_idx + 20, len(lines))):
        line = lines[i].strip()
        
        # Stop at empty lines or next header
        if not line or line.startswith('#') or re.search(r'^[A-Z\s]{10,}$', line):
            if rankings:  # Only stop if we've found some rankings
                break
            continue
        
        # Match pattern: "NUMBER HORSE_NAME RATING"
        # Examples: "2 British Isles 105", "11 White Abarrio 147.8"
        match = re.match(r'^(\d+)\s+([A-Za-z\s\']+?)\s+([\d.]+)\s*$', line)
        if match:
            rankings.append({
                "number": int(match.group(1)),
                "name": match.group(2).strip(),
                "rating": float(match.group(3))
            })
    
    return rankings


# Convenience function for command-line testing
def main():
    """Test parser with sample input."""
    import sys
    
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r', encoding='utf-8') as f:
            pp_text = f.read()
    else:
        # Use sample data
        pp_text = """Ultimate PP's w/ QuickPlay Comments Gulfstream Park PWCInvit-G1 1„ Mile 4&up Saturday, January 24, 2026 Race 13
# Speed Last Race # Prime Power # Class Rating # Best Speed at Dist
2 British Isles 105
10 Mika 102
11 White Abarrio 100
11 White Abarrio 147.8
3 Full Serrano 145.2
1 Disco Time 144.3
11 White Abarrio 120.9
1 Disco Time 120.4
3 Full Serrano 120.0
5 Skippylongstocking 107
11 White Abarrio 107
2 British Isles 105
"""
    
    result = parse_brisnet_pp_to_json(pp_text)
    print(result)


if __name__ == '__main__':
    main()
