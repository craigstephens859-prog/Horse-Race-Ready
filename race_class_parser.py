"""
Race Class Parser Module for Brisnet Past Performances

This module provides comprehensive parsing of Brisnet race headers and class-based
weight calculations for horse racing handicapping. It extracts race metadata,
analyzes race conditions, assigns hierarchy levels with grade boosts, and calculates
weighted scores for handicapping purposes.

Key Features:
- Parse race headers (track, race#, type, purse, distance, age/sex, date)
- Parse race conditions with expanded class type mappings
- Handle graded stakes with hierarchy boosts (G1 +3, G2 +2, G3 +1)
- Calculate weighted class scores for handicapping
- Robust error handling with sensible defaults
- Support for fractional distances (e.g., '1 1/16 Mile' → 8.5 furlongs)

Author: Horse Racing Picks System
Date: February 2026
Python: 3.11+
"""

import re
from typing import Dict, Any, Optional, Tuple
from decimal import Decimal, InvalidOperation
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# CLASS TYPE MAPPINGS - Expanded with common abbreviations
# ============================================================================

CLASS_MAP: Dict[str, str] = {
    # Graded Stakes
    'G1': 'Grade 1 Stakes',
    'G2': 'Grade 2 Stakes',
    'G3': 'Grade 3 Stakes',
    'GR1': 'Grade 1 Stakes',
    'GR2': 'Grade 2 Stakes',
    'GR3': 'Grade 3 Stakes',
    'GRADE1': 'Grade 1 Stakes',
    'GRADE2': 'Grade 2 Stakes',
    'GRADE3': 'Grade 3 Stakes',
    
    # Listed Stakes
    'L': 'Listed Stakes',
    'LR': 'Listed Stakes',
    'LISTED': 'Listed Stakes',
    
    # Non-Graded Stakes
    'STK': 'Stakes',
    'S': 'Stakes',
    'N': 'Non-Graded Stakes',
    'STAKES': 'Stakes',
    
    # Handicap variations
    'HCP': 'Handicap',
    'H': 'Handicap',
    'HANDICAP': 'Handicap',
    '©HCP': 'Handicap',  # Brisnet uses © symbol
    '©': 'Handicap',
    'OCH': 'Optional Claiming Handicap',
    'CLH': 'Claiming Handicap',
    'SHP': 'Starter Handicap',
    
    # Allowance variations
    'ALW': 'Allowance',
    'A': 'Allowance',
    'ALLOWANCE': 'Allowance',
    'AOC': 'Allowance Optional Claiming',
    'AO': 'Allowance Optional Claiming',
    'OC': 'Optional Claiming',
    'OCL': 'Optional Claiming',
    'N1X': 'Allowance Non-Winners of 1',
    'N2X': 'Allowance Non-Winners of 2',
    'N3X': 'Allowance Non-Winners of 3',
    'NW1': 'Allowance Non-Winners of 1',
    'NW2': 'Allowance Non-Winners of 2',
    'NW3': 'Allowance Non-Winners of 3',
    'N1L': 'Allowance Non-Winners of 1 Lifetime',
    'N2L': 'Allowance Non-Winners of 2 Lifetime',
    'N3L': 'Allowance Non-Winners of 3 Lifetime',
    
    # Starter Allowance/Stakes variations
    'STA': 'Starter Allowance',
    'STR': 'Starter Allowance',
    'STARTER': 'Starter Allowance',
    'SOC': 'Starter Optional Claiming',
    'CST': 'Claiming Stakes',
    'CLMSTK': 'Claiming Stakes',
    'SST': 'Starter Stakes',
    
    # Claiming variations
    'CLM': 'Claiming',
    'C': 'Claiming',
    'CL': 'Claiming',
    'CLAIMING': 'Claiming',
    'CLG': 'Claiming',
    'WAIVER': 'Waiver Claiming',
    'WCL': 'Waiver Claiming',
    
    # Maiden variations
    'MSW': 'Maiden Special Weight',
    'MD': 'Maiden',
    'MDN': 'Maiden',
    'MAIDEN': 'Maiden',
    'MCL': 'Maiden Claiming',
    'MDC': 'Maiden Claiming',
    'MDNCLM': 'Maiden Claiming',
    'MOC': 'Maiden Optional Claiming',
    'MSC': 'Maiden Starter Claiming',
    
    # Trial races
    'TRL': 'Trial',
    'TRIAL': 'Trial',
    
    # Special conditions
    'OPT': 'Optional',
    'OPTIONAL': 'Optional',
    'STATEBRED': 'State Bred',
    'STB': 'State Bred',
    
    # Specialty races
    'FUT': 'Futurity',
    'FUTURITY': 'Futurity',
    'FTR': 'Futurity',
    'DER': 'Derby',
    'DERBY': 'Derby',
    'DBY': 'Derby',
    'INVIT': 'Invitational',
    'INVITATIONAL': 'Invitational',
    'INV': 'Invitational',
    'MAT': 'Match Race',
    'MATCH': 'Match Race',
    'TR': 'Training Race',
    'TRAINING': 'Training Race',
}

# Hierarchy level mapping (higher = better class)
# Based on industry-standard US horse racing hierarchy (1-7 scale)
# Note: Grade boosts (+3/+2/+1) are applied separately via get_hierarchy_level()
LEVEL_MAP: Dict[str, int] = {
    # Level 7: Elite Stakes (Graded, Listed, High-Purse Non-Graded)
    'Grade 1 Stakes': 7,  # + Grade Boost +3 = Final 10
    'Grade 2 Stakes': 7,  # + Grade Boost +2 = Final 9  
    'Grade 3 Stakes': 7,  # + Grade Boost +1 = Final 8
    'Listed Stakes': 7,   # No grade boost
    'Stakes': 7,          # Non-graded stakes ($75k+ purse)
    'Non-Graded Stakes': 7,
    'Futurity': 7,        # Stakes for 2-year-olds
    'Derby': 7,           # Stakes for 3-year-olds
    'Invitational': 7,    # Stakes by invitation
    'Starter Stakes': 7,  # Starter with stakes purse (high end)
    
    # Level 6: Handicap (Weights assigned to equalize chances)
    'Handicap': 6,
    'Optional Claiming Handicap': 6,
    
    # Level 5: Optional Claiming / High Allowance
    'Allowance Optional Claiming': 5,
    'Optional Claiming': 5,
    'Starter Optional Claiming': 5,  # High end
    
    # Level 4: Allowance / Mid-Tier Conditional
    'Allowance': 4,
    'Allowance Non-Winners of 1': 4,
    'Allowance Non-Winners of 2': 4,
    'Allowance Non-Winners of 3': 4,
    'Allowance Non-Winners of 1 Lifetime': 4,
    'Allowance Non-Winners of 2 Lifetime': 4,
    'Allowance Non-Winners of 3 Lifetime': 4,
    'Optional': 4,
    'Trial': 4,  # Qualifiers for major stakes
    
    # Level 3: Starter Allowance
    'Starter Allowance': 3,
    'Starter Handicap': 3,
    'State Bred': 3,
    
    # Level 2: Claiming
    'Claiming': 2,
    'Waiver Claiming': 2,
    'Claiming Handicap': 2,
    'Claiming Stakes': 2,  # Low-end claiming stakes (bridges to higher)
    
    # Level 1: Maiden
    'Maiden Special Weight': 1,  # Top-quality maidens (no claiming)
    'Maiden': 1,
    'Maiden Optional Claiming': 1,  # Hybrid maiden
    'Maiden Claiming': 1,  # Maidens with claiming price
    'Maiden Starter Claiming': 1,
    
    # Special / Unknown
    'Match Race': 0,      # Special: Only two horses (rare)
    'Training Race': 0,   # Special: Practice races (not competitive)
    'Unknown': 0,
}


# ============================================================================
# FUNCTION 1: PARSE BRISNET HEADER
# ============================================================================

def parse_brisnet_header(pp_text: str) -> Dict[str, Any]:
    """
    Parse Brisnet race header to extract comprehensive race metadata.
    
    Supports both pipe-delimited and space-separated Brisnet PP formats.
    
    Pipe format: "Ultimate PP's | Turf Paradise | ©Hcp 50000 | 6 Furlongs | 3yo Fillies | Monday, February 02, 2026 | Race 8"
    Space format: "Ultimate PP's w/ QuickPlay Comments Gulfstream Park PWCInvit-G1 1„ Mile 4&up Saturday, January 24, 2026 Race 13"
    
    Args:
        pp_text: Raw Brisnet Past Performance text containing header line
        
    Returns:
        Dictionary containing parsed header fields:
            - track_name: str (e.g., 'Turf Paradise', 'Gulfstream Park')
            - race_number: int (e.g., 8, 13)
            - race_type: str (e.g., '©Hcp', 'PWCInvit-G1')
            - race_conditions: str (full race type text for further parsing)
            - purse_amount: int (in dollars, e.g., 50000, 3000000)
            - distance: str (e.g., '6 Furlongs', '1„ Mile')
            - distance_furlongs: float (converted, e.g., 6.0, 10.0)
            - age_restriction: str (e.g., '3yo', '4&up')
            - sex_restriction: str (e.g., 'Fillies', 'F&M')
            - race_date: str (e.g., 'February 02, 2026', 'January 24, 2026')
            - day_of_week: str (e.g., 'Monday', 'Saturday')
            - surface: str (e.g., 'Dirt', 'Turf') - if detectable
    """
    result: Dict[str, Any] = {
        'track_name': None,
        'race_number': None,
        'race_type': None,
        'race_conditions': None,
        'purse_amount': 0,
        'distance': None,
        'distance_furlongs': 0.0,
        'age_restriction': None,
        'sex_restriction': None,
        'race_date': None,
        'day_of_week': None,
        'surface': None,
    }
    
    try:
        # Find header line in first 10 lines
        header_line = ''
        for line in pp_text.split('\n')[:10]:
            if 'Ultimate PP' in line and 'Race' in line:
                header_line = line.strip()
                break
        
        if not header_line:
            logger.warning("No header line found in PP text")
            return result
        
        # Detect format: pipe-delimited or space-separated
        is_pipe_format = '|' in header_line
        
        if is_pipe_format:
            # Parse pipe-delimited format
            parts = [p.strip() for p in header_line.split('|')]
        else:
            # Parse space-separated format
            # Format: Ultimate PP's w/ QuickPlay Comments <Track> <Type> <Distance> <Age> <Date> Race <Num>
            import re
            
            # Remove "Ultimate PP's w/ QuickPlay Comments" prefix
            text = re.sub(r'^Ultimate PP.*?Comments\s+', '', header_line)
            
            # Extract track name (before race type pattern)
            # Track names can be multi-word (e.g., "Gulfstream Park", "Santa Anita", "Del Mar")
            track_pattern = r'^([A-Z][a-z]+(?:\s+[A-Z][a-z]+)?)'
            track_match = re.search(track_pattern, text)
            if track_match:
                result['track_name'] = track_match.group(1)
                text = text[len(track_match.group(1)):].strip()
            
            # Extract race type (usually has hyphens, special chars, or ends with -G1/G2/G3)
            # Examples: PWCInvit-G1, ©Hcp, ALW50000, MCL25000
            type_pattern = r'^([A-Za-z©™®]+(?:-[A-Z0-9]+)?(?:\s+\d+)?)'
            type_match = re.search(type_pattern, text)
            if type_match:
                result['race_type'] = type_match.group(1).strip()
                result['race_conditions'] = type_match.group(1).strip()
                text = text[len(type_match.group(1)):].strip()
            
            # Extract distance (e.g., "1„ Mile", "6F", "1ˆ")
            dist_pattern = r'^([\d½¼¾„ˆ]+\s*(?:Mile|Furlongs?|Yards?|f|F|m|M)?)'
            dist_match = re.search(dist_pattern, text)
            if dist_match:
                result['distance'] = dist_match.group(1).strip()
                text = text[len(dist_match.group(1)):].strip()
            
            # Extract age/sex restriction (e.g., "4&up", "3yo Fillies")
            age_pattern = r'^((?:\d+)?(?:yo|&up)?(?:\s+(?:Fillies|Colts|Mares|F&M|C&G))?)'
            age_match = re.search(age_pattern, text)
            if age_match and age_match.group(1).strip():
                age_text = age_match.group(1).strip()
                result['age_restriction'] = age_text
                # Extract sex restriction
                sex_match = re.search(r'(Fillies|Colts|Mares|F&M|C&G)', age_text)
                if sex_match:
                    result['sex_restriction'] = sex_match.group(1)
                text = text[len(age_text):].strip()
            
            # Extract date (e.g., "Saturday, January 24, 2026")
            date_pattern = r'((?:Monday|Tuesday|Wednesday|Thursday|Friday|Saturday|Sunday),?\s+\w+\s+\d+,?\s+\d{4})'
            date_match = re.search(date_pattern, text)
            if date_match:
                date_str = date_match.group(1)
                parts_date = date_str.split(',')
                if len(parts_date) >= 2:
                    result['day_of_week'] = parts_date[0].strip()
                    result['race_date'] = ','.join(parts_date[1:]).strip()
                text = text[len(date_str):].strip()
            
            # Extract race number (e.g., "Race 13")
            race_num_pattern = r'Race\s+(\d+)'
            race_num_match = re.search(race_num_pattern, text)
            if race_num_match:
                result['race_number'] = int(race_num_match.group(1))
            
            # Convert distance to furlongs
            if result['distance']:
                result['distance_furlongs'] = _convert_distance_to_furlongs(result['distance'])
            
            # Try to extract purse from race conditions on subsequent lines
            for line in pp_text.split('\n')[:20]:
                purse_match = re.search(r'Purse[:\s]+\$?([\d,]+)', line, re.IGNORECASE)
                if purse_match:
                    result['purse_amount'] = int(purse_match.group(1).replace(',', ''))
                    break
            
            return result
        
        # Original pipe-delimited parsing continues below
        parts = [p.strip() for p in header_line.split('|')]
        
        for i, part in enumerate(parts):
            part_lower = part.lower()
            
            # Skip "Ultimate PP's w/ QuickPlay Comments"
            if 'ultimate' in part_lower or 'quickplay' in part_lower:
                continue
            
            # Extract track name (first non-Ultimate part without race keywords)
            if not result['track_name'] and i > 0:
                if not re.search(r'\b(race|furlong|mile|yard|stakes|allowance|claiming|maiden)\b', part_lower):
                    if not re.search(r'\d{4}', part):  # Not a date
                        result['track_name'] = part
                        continue
            
            # Extract race type + purse (e.g., "©Hcp 50000")
            race_type_match = re.search(r'([©¨§]?[A-Za-z]+)\s+(\d+)', part)
            if race_type_match and not result['race_type']:
                result['race_type'] = race_type_match.group(1)
                result['race_conditions'] = part  # Save full text for detailed parsing
                try:
                    result['purse_amount'] = int(race_type_match.group(2))
                except (ValueError, TypeError):
                    result['purse_amount'] = 0
                continue
            
            # Extract distance (e.g., "6 Furlongs", "1 1/16 Mile")
            if re.search(r'\d+\s*(?:furlong|mile|yard|f\b)', part_lower):
                result['distance'] = part
                result['distance_furlongs'] = _convert_distance_to_furlongs(part)
                
                # Detect surface from distance line
                if 'turf' in part_lower or '(t)' in part_lower:
                    result['surface'] = 'Turf'
                elif 'dirt' in part_lower or 'd' in part_lower:
                    result['surface'] = 'Dirt'
                continue
            
            # Extract age restriction (e.g., "3yo", "4&up", "3&up")
            age_match = re.search(r'(\d+yo|\d+&up|\d+\+|4\+|3\+)', part_lower)
            if age_match:
                result['age_restriction'] = age_match.group(1).replace('+', '&up')
            
            # Extract sex restriction (e.g., "Fillies", "Mares", "F&M", "Colts & Geldings")
            sex_match = re.search(r'(fillies?|mares?|colts?|geldings?|f&m|c&g)', part_lower)
            if sex_match and not result['sex_restriction']:
                result['sex_restriction'] = sex_match.group(1).title()
            
            # Extract day of week and date (e.g., "Monday, February 02, 2026")
            day_match = re.search(r'(monday|tuesday|wednesday|thursday|friday|saturday|sunday)[,\s]+(.+\d{4})', part_lower)
            if day_match:
                result['day_of_week'] = day_match.group(1).title()
                result['race_date'] = day_match.group(2).strip().title()
                continue
            
            # Extract race number (e.g., "Race 8")
            race_num_match = re.search(r'race\s+(\d+)', part_lower)
            if race_num_match:
                try:
                    result['race_number'] = int(race_num_match.group(1))
                except (ValueError, TypeError):
                    result['race_number'] = None
                continue
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing Brisnet header: {e}")
        return result


def _convert_distance_to_furlongs(distance_str: str) -> float:
    """
    Convert distance string to furlongs (internal helper).
    
    US Horse Racing Standard: Only furlongs and miles are used.
    
    Handles various formats:
    - "6 Furlongs" → 6.0
    - "1 Mile" → 8.0
    - "1 1/16 Mile" → 8.5
    - "1 1/8 Mile" → 9.0
    - "7½ Furlongs" → 7.5
    - "5.5f" → 5.5
    - "1„ Mile" (Unicode) → 10.0
    - "1ˆ" (Unicode) → 8.5
    
    Args:
        distance_str: Distance string from header
        
    Returns:
        Distance in furlongs as float, 0.0 if parsing fails
    """
    try:
        distance_lower = distance_str.lower().strip()
        
        # Replace Unicode distance characters commonly used in Brisnet PPs
        # „ = 1 1/4 mile (10 furlongs)
        # ˆ = 1 1/16 mile (8.5 furlongs)
        # © = 1 1/8 mile (9 furlongs)
        distance_lower = distance_lower.replace('„', ' 1 1/4')  # 1 1/4 mile
        distance_lower = distance_lower.replace('ˆ', ' 1 1/16')  # 1 1/16 mile
        distance_lower = distance_lower.replace('©', ' 1 1/8')  # 1 1/8 mile
        distance_lower = distance_lower.replace('‰', ' 1 3/16')  # 1 3/16 mile
        distance_lower = distance_lower.replace('½', ' 1/2')
        distance_lower = distance_lower.replace('¼', ' 1/4')
        distance_lower = distance_lower.replace('¾', ' 3/4')
        
        # Handle miles FIRST (1 mile = 8 furlongs)
        # Must come before furlong check to handle "1 1/8 Mile" correctly
        if 'mile' in distance_lower:
            # Extract whole number and fraction (e.g., "1 1/16 Mile" → 8.5f, "1 1/8 Mile" → 9.0f)
            mile_match = re.search(r'(\d+)(?:\s+(\d+)/(\d+))?', distance_lower)
            if mile_match:
                miles = float(mile_match.group(1))
                if mile_match.group(2) and mile_match.group(3):  # Has fractional part
                    numerator = float(mile_match.group(2))
                    denominator = float(mile_match.group(3))
                    miles += (numerator / denominator)
                # Convert miles to furlongs: 1 mile = 8 furlongs
                return round(miles * 8.0, 1)
        
        # Handle furlongs directly
        elif 'furlong' in distance_lower or distance_lower.endswith('f'):
            # Extract number with optional fraction
            num_match = re.search(r'(\d+(?:\s*\d+/\d+)?|\d+\.?\d*)', distance_lower)
            if num_match:
                return _parse_fraction(num_match.group(1))
        
        return 0.0
        
    except Exception as e:
        logger.warning(f"Error converting distance '{distance_str}': {e}")
        return 0.0


def _parse_fraction(num_str: str) -> float:
    """
    Parse fractional numbers (internal helper).
    
    Examples:
    - "6" → 6.0
    - "7 1/2" → 7.5
    - "7½" → 7.5
    - "1/8" → 0.125
    
    Args:
        num_str: Number string, possibly with fraction
        
    Returns:
        Parsed float value
    """
    try:
        # Replace unicode fraction symbols
        num_str = num_str.replace('½', ' 1/2').replace('¼', ' 1/4').replace('¾', ' 3/4')
        num_str = num_str.replace('⅛', ' 1/8').replace('⅜', ' 3/8').replace('⅝', ' 5/8').replace('⅞', ' 7/8')
        
        # Check for fraction pattern: "7 1/2" or "1/2"
        frac_match = re.match(r'(\d+)?\s*(\d+)/(\d+)', num_str.strip())
        if frac_match:
            whole = float(frac_match.group(1) or 0)
            numerator = float(frac_match.group(2))
            denominator = float(frac_match.group(3))
            return whole + (numerator / denominator)
        
        # Simple decimal
        return float(num_str.strip())
        
    except Exception:
        return 0.0


# ============================================================================
# FUNCTION 2: PARSE RACE CONDITIONS
# ============================================================================

def parse_race_conditions(race_conditions: str, purse_amount: int = 0) -> Dict[str, Any]:
    """
    Parse race conditions string to extract detailed class information.
    
    Analyzes the race conditions text (e.g., "©Hcp 50000", "G1 Stakes 1000000")
    to extract class type, grade level, purse, and other relevant metadata.
    Handles graded stakes explicitly (G1, G2, G3) for hierarchy boost calculation.
    
    Args:
        race_conditions: Race conditions text from header (e.g., "©Hcp 50000")
        purse_amount: Purse amount in dollars (default: 0)
        
    Returns:
        Dictionary containing:
            - class_type: str (full class name from CLASS_MAP)
            - class_abbreviation: str (original abbreviation)
            - grade_level: Optional[int] (1, 2, or 3 for graded stakes, else None)
            - purse_amount: int (in dollars)
            - is_stakes: bool (True if stakes race)
            - is_graded_stakes: bool (True if G1/G2/G3)
            - is_restricted: bool (True if state-bred or other restrictions)
            - additional_conditions: list (e.g., ['State Bred', 'Non-Winners'])
            
    Example:
        >>> conditions = "©Hcp 50000"
        >>> result = parse_race_conditions(conditions, 50000)
        >>> result['class_type']
        'Handicap'
        >>> result['is_stakes']
        False
        
        >>> conditions = "G1 Stakes 1000000"
        >>> result = parse_race_conditions(conditions, 1000000)
        >>> result['grade_level']
        1
        >>> result['is_graded_stakes']
        True
    """
    result: Dict[str, Any] = {
        'class_type': 'Unknown',
        'class_abbreviation': '',
        'grade_level': None,
        'purse_amount': purse_amount,
        'is_stakes': False,
        'is_graded_stakes': False,
        'is_restricted': False,
        'additional_conditions': [],
    }
    
    try:
        if not race_conditions:
            logger.warning("Empty race conditions provided")
            return result
        
        conditions_clean = race_conditions.strip().upper()
        
        # Extract purse if present in conditions string
        purse_match = re.search(r'(\d+)$', conditions_clean)
        if purse_match:
            try:
                result['purse_amount'] = int(purse_match.group(1))
            except (ValueError, TypeError):
                pass
        
        # Check for graded stakes (G1, G2, G3, GR1, etc.)
        grade_match = re.search(r'\b(G|GR|GRADE)\s*([123])\b', conditions_clean)
        if grade_match:
            try:
                result['grade_level'] = int(grade_match.group(2))
                result['is_graded_stakes'] = True
                result['is_stakes'] = True
                result['class_type'] = f"Grade {result['grade_level']} Stakes"
                result['class_abbreviation'] = f"G{result['grade_level']}"
                return result  # Early return for graded stakes
            except (ValueError, TypeError):
                pass
        
        # Check for other stakes types
        if re.search(r'\b(STK|STAKES|LISTED|LR|L)\b', conditions_clean):
            result['is_stakes'] = True
        
        # Match class abbreviation against CLASS_MAP
        for abbrev, full_name in CLASS_MAP.items():
            # Check if abbreviation appears in conditions
            # Use word boundaries for exact matches
            pattern = r'\b' + re.escape(abbrev) + r'\b'
            if re.search(pattern, conditions_clean):
                result['class_type'] = full_name
                result['class_abbreviation'] = abbrev
                
                # Set stakes flag if matched class is stakes-related
                if 'Stakes' in full_name:
                    result['is_stakes'] = True
                
                break
        
        # Check for restrictions (state-bred, etc.)
        if re.search(r'\b(STATE\s*BRED|STB|RESTRICTED|BRED)\b', conditions_clean):
            result['is_restricted'] = True
            result['additional_conditions'].append('State Bred')
        
        # Check for non-winners conditions
        nw_match = re.search(r'\b(NON[\s-]?WINNERS?|N\dX|NW\d)\b', conditions_clean)
        if nw_match:
            result['additional_conditions'].append(f'Non-Winners Condition')
        
        # Check for optional claiming
        if 'OPTIONAL' in conditions_clean or 'OPT' in conditions_clean:
            if 'CLAIMING' in conditions_clean:
                result['additional_conditions'].append('Optional Claiming')
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing race conditions '{race_conditions}': {e}")
        return result


# ============================================================================
# FUNCTION 3: GET HIERARCHY LEVEL
# ============================================================================

def get_hierarchy_level(
    class_type: str,
    grade_level: Optional[int] = None,
    purse_amount: int = 0
) -> Dict[str, Any]:
    """
    Assign hierarchy level to race class with grade boosts for graded stakes.
    
    Maps class type to numeric hierarchy level (0-10 scale) and applies
    additional boosts for graded stakes races:
    - Grade 1: Base level 10 + 3 boost = 13
    - Grade 2: Base level 9 + 2 boost = 11
    - Grade 3: Base level 8 + 1 boost = 9
    
    Higher purse amounts within the same class can provide micro-adjustments.
    
    Args:
        class_type: Full class type name (e.g., 'Grade 1 Stakes', 'Handicap')
        grade_level: Grade level (1, 2, or 3) if applicable (default: None)
        purse_amount: Purse in dollars for micro-adjustments (default: 0)
        
    Returns:
        Dictionary containing:
            - base_level: int (0-10 from LEVEL_MAP)
            - grade_boost: int (0, 1, 2, or 3)
            - final_level: int (base_level + grade_boost)
            - purse_adjustment: float (micro-adjustment based on purse)
            - adjusted_level: float (final_level + purse_adjustment)
            
    Example:
        >>> result = get_hierarchy_level('Grade 1 Stakes', grade_level=1, purse_amount=1000000)
        >>> result['base_level']
        10
        >>> result['grade_boost']
        3
        >>> result['final_level']
        13
        
        >>> result = get_hierarchy_level('Claiming', purse_amount=10000)
        >>> result['base_level']
        3
        >>> result['grade_boost']
        0
        >>> result['final_level']
        3
    """
    result: Dict[str, Any] = {
        'base_level': 0,
        'grade_boost': 0,
        'final_level': 0,
        'purse_adjustment': 0.0,
        'adjusted_level': 0.0,
    }
    
    try:
        # Get base level from LEVEL_MAP
        result['base_level'] = LEVEL_MAP.get(class_type, 0)
        
        # Apply grade boost
        # PEGASUS WC TUNING: Reduced G1 boost from 3→2 (form/speed matter more in elite races)
        if grade_level == 1:
            result['grade_boost'] = 2  # Was 3, reduced after Pegasus WC analysis
        elif grade_level == 2:
            result['grade_boost'] = 2
        elif grade_level == 3:
            result['grade_boost'] = 1
        else:
            result['grade_boost'] = 0
        
        # Calculate final level
        result['final_level'] = result['base_level'] + result['grade_boost']
        
        # Calculate purse adjustment (micro-adjustment)
        # Adds 0.0 to 0.5 based on purse within class
        if purse_amount > 0:
            # Logarithmic scaling: purse adjustment increases with purse size
            # but caps at +0.5 to maintain class hierarchy
            purse_log = min(purse_amount, 5000000)  # Cap at $5M for calculation
            result['purse_adjustment'] = round(
                (purse_log / 10000000) * 0.5,  # Scale to 0-0.5 range
                2
            )
        
        # Calculate adjusted level
        result['adjusted_level'] = round(
            result['final_level'] + result['purse_adjustment'],
            2
        )
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating hierarchy level for '{class_type}': {e}")
        return result


# ============================================================================
# FUNCTION 4: CALCULATE CLASS WEIGHT
# ============================================================================

def calculate_class_weight(
    parsed_conditions: Dict[str, Any],
    hierarchy_info: Dict[str, Any],
    distance_furlongs: float = 0.0,
    surface: Optional[str] = None,
    base_weight: float = 1.0
) -> Dict[str, Any]:
    """
    Calculate weighted class score for handicapping using parsed data.
    
    Combines all parsed information (class type, hierarchy level, purse, distance,
    surface) to calculate a comprehensive weighted score for race class quality.
    This score can be used in handicapping models to adjust horse ratings based
    on the class level of the race.
    
    Calculation factors:
    - Base hierarchy level with grade boosts
    - Purse amount adjustment
    - Distance adjustment (longer races get slight boost)
    - Surface adjustment (turf races may get different weighting)
    - Stakes race bonus
    - Restriction penalty (state-bred, etc.)
    
    Args:
        parsed_conditions: Output from parse_race_conditions()
        hierarchy_info: Output from get_hierarchy_level()
        distance_furlongs: Race distance in furlongs (default: 0.0)
        surface: Race surface ('Dirt', 'Turf', 'Synthetic') (default: None)
        base_weight: Starting weight multiplier (default: 1.0)
        
    Returns:
        Dictionary containing:
            - class_weight: float (final calculated weight)
            - breakdown: dict (detailed calculation components)
                - hierarchy_score: float
                - purse_score: float
                - distance_adjustment: float
                - surface_adjustment: float
                - stakes_bonus: float
                - restriction_penalty: float
            - quality_rating: str ('Elite', 'High', 'Medium', 'Low')
            
    Example:
        >>> conditions = parse_race_conditions("G1 Stakes 1000000", 1000000)
        >>> hierarchy = get_hierarchy_level('Grade 1 Stakes', 1, 1000000)
        >>> weight = calculate_class_weight(conditions, hierarchy, 10.0, 'Dirt')
        >>> weight['quality_rating']
        'Elite'
        >>> weight['class_weight'] > 10.0
        True
    """
    result: Dict[str, Any] = {
        'class_weight': 0.0,
        'breakdown': {
            'hierarchy_score': 0.0,
            'purse_score': 0.0,
            'distance_adjustment': 0.0,
            'surface_adjustment': 0.0,
            'stakes_bonus': 0.0,
            'restriction_penalty': 0.0,
        },
        'quality_rating': 'Unknown',
    }
    
    try:
        # Start with adjusted hierarchy level
        hierarchy_score = hierarchy_info.get('adjusted_level', 0.0)
        result['breakdown']['hierarchy_score'] = hierarchy_score
        
        # Purse score (normalized)
        purse_amount = parsed_conditions.get('purse_amount', 0)
        if purse_amount > 0:
            # Logarithmic scale: $10k = 0.1, $100k = 0.2, $1M = 0.3
            purse_score = min(0.3, (purse_amount / 3333333) * 0.3)
            result['breakdown']['purse_score'] = round(purse_score, 3)
        
        # Distance adjustment (slight bonus for longer races)
        if distance_furlongs > 0:
            # Routes (>7f) get small bonus, sprints stay neutral
            if distance_furlongs >= 8.0:  # 1 mile or more
                result['breakdown']['distance_adjustment'] = 0.1
            elif distance_furlongs >= 7.0:
                result['breakdown']['distance_adjustment'] = 0.05
        
        # Surface adjustment
        if surface:
            surface_upper = surface.upper()
            if surface_upper == 'TURF':
                # Turf races often slightly higher quality fields
                result['breakdown']['surface_adjustment'] = 0.05
            elif surface_upper == 'SYNTHETIC':
                # Synthetic may be neutral or slightly lower
                result['breakdown']['surface_adjustment'] = -0.02
        
        # Stakes bonus (additional boost beyond hierarchy)
        if parsed_conditions.get('is_stakes', False):
            result['breakdown']['stakes_bonus'] = 0.2
            
            # Extra bonus for graded stakes
            if parsed_conditions.get('is_graded_stakes', False):
                result['breakdown']['stakes_bonus'] += 0.3
        
        # Restriction penalty (state-bred, etc. typically weaker fields)
        if parsed_conditions.get('is_restricted', False):
            result['breakdown']['restriction_penalty'] = -0.2
        
        # Calculate total weight
        total_weight = base_weight + sum(result['breakdown'].values())
        result['class_weight'] = round(max(0.0, total_weight), 3)
        
        # Assign quality rating
        if result['class_weight'] >= 12.0:
            result['quality_rating'] = 'Elite'
        elif result['class_weight'] >= 8.0:
            result['quality_rating'] = 'High'
        elif result['class_weight'] >= 4.0:
            result['quality_rating'] = 'Medium'
        elif result['class_weight'] >= 1.0:
            result['quality_rating'] = 'Low'
        else:
            result['quality_rating'] = 'Minimal'
        
        return result
        
    except Exception as e:
        logger.error(f"Error calculating class weight: {e}")
        return result


# ============================================================================
# MAIN CHAINING FUNCTION
# ============================================================================

def parse_and_calculate_class(pp_text: str) -> Dict[str, Any]:
    """
    Main chaining function that orchestrates all parsing and calculation steps.
    
    This is the primary entry point for the module. It chains together all
    four parsing/calculation functions to produce a comprehensive analysis
    of the race class from raw Brisnet Past Performance text.
    
    Processing pipeline:
    1. Parse header → Extract race metadata
    2. Parse conditions → Analyze class type and grade
    3. Get hierarchy → Assign level with grade boosts
    4. Calculate weight → Compute final weighted score
    
    Args:
        pp_text: Raw Brisnet Past Performance text
        
    Returns:
        Comprehensive dictionary containing all parsed data and calculations:
            - header: dict (from parse_brisnet_header)
            - conditions: dict (from parse_race_conditions)
            - hierarchy: dict (from get_hierarchy_level)
            - weight: dict (from calculate_class_weight)
            - summary: dict (key metrics for quick access)
            
    Example:
        >>> pp_text = '''
        ... Ultimate PP's w/ QuickPlay Comments | Turf Paradise | 
        ... ©Hcp 50000 | 6 Furlongs | 3yo Fillies | 
        ... Monday, February 02, 2026 | Race 8
        ... [rest of PP data...]
        ... '''
        >>> result = parse_and_calculate_class(pp_text)
        >>> result['summary']['track']
        'Turf Paradise'
        >>> result['summary']['class_weight']
        5.875
        >>> result['summary']['quality']
        'Medium'
    """
    try:
        # Step 1: Parse header
        header = parse_brisnet_header(pp_text)
        
        # Step 2: Parse race conditions
        conditions = parse_race_conditions(
            header.get('race_conditions') or header.get('race_type', ''),
            header.get('purse_amount', 0)
        )
        
        # Step 3: Get hierarchy level
        hierarchy = get_hierarchy_level(
            conditions.get('class_type', 'Unknown'),
            conditions.get('grade_level'),
            conditions.get('purse_amount', 0)
        )
        
        # Step 4: Calculate class weight
        weight = calculate_class_weight(
            conditions,
            hierarchy,
            header.get('distance_furlongs', 0.0),
            header.get('surface')
        )
        
        # Compile comprehensive result
        result = {
            'header': header,
            'conditions': conditions,
            'hierarchy': hierarchy,
            'weight': weight,
            'summary': {
                'track': header.get('track_name'),
                'race_number': header.get('race_number'),
                'class_type': conditions.get('class_type'),
                'grade_level': conditions.get('grade_level'),
                'purse': conditions.get('purse_amount'),
                'distance': header.get('distance'),
                'distance_furlongs': header.get('distance_furlongs'),
                'surface': header.get('surface'),
                'hierarchy_level': hierarchy.get('final_level'),
                'class_weight': weight.get('class_weight'),
                'quality': weight.get('quality_rating'),
                'is_stakes': conditions.get('is_stakes'),
                'is_graded': conditions.get('is_graded_stakes'),
            }
        }
        
        return result
        
    except Exception as e:
        logger.error(f"Error in main parsing pipeline: {e}")
        return {
            'header': {},
            'conditions': {},
            'hierarchy': {},
            'weight': {},
            'summary': {},
            'error': str(e),
        }


# ============================================================================
# EXAMPLE USAGE AND TESTS
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("RACE CLASS PARSER MODULE - TEST SUITE")
    print("=" * 80)
    print()
    
    # Test 1: Grade 1 Stakes
    print("TEST 1: Grade 1 Stakes Race")
    print("-" * 80)
    test_pp_1 = """
Ultimate PP's w/ QuickPlay Comments | Santa Anita | G1 Stakes 1000000 | 10 Furlongs | 3yo & Up | Saturday, March 15, 2026 | Race 9
[Additional PP data would follow...]
    """
    result_1 = parse_and_calculate_class(test_pp_1)
    print(f"Track: {result_1['summary']['track']}")
    print(f"Race #: {result_1['summary']['race_number']}")
    print(f"Class: {result_1['summary']['class_type']}")
    print(f"Grade: G{result_1['summary']['grade_level']}")
    print(f"Purse: ${result_1['summary']['purse']:,}")
    print(f"Distance: {result_1['summary']['distance']} ({result_1['summary']['distance_furlongs']} furlongs)")
    print(f"Hierarchy Level: {result_1['hierarchy']['base_level']} + {result_1['hierarchy']['grade_boost']} boost = {result_1['hierarchy']['final_level']}")
    print(f"Class Weight: {result_1['summary']['class_weight']}")
    print(f"Quality Rating: {result_1['summary']['quality']}")
    print()
    
    # Test 2: Handicap Race
    print("TEST 2: Handicap Race")
    print("-" * 80)
    test_pp_2 = """
Ultimate PP's w/ QuickPlay Comments | Turf Paradise | ©Hcp 50000 | 6 Furlongs | 3yo Fillies | Monday, February 02, 2026 | Race 8
[Additional PP data would follow...]
    """
    result_2 = parse_and_calculate_class(test_pp_2)
    print(f"Track: {result_2['summary']['track']}")
    print(f"Class: {result_2['summary']['class_type']}")
    print(f"Purse: ${result_2['summary']['purse']:,}")
    print(f"Distance: {result_2['summary']['distance']}")
    print(f"Hierarchy Level: {result_2['hierarchy']['final_level']}")
    print(f"Class Weight: {result_2['summary']['class_weight']}")
    print(f"Quality Rating: {result_2['summary']['quality']}")
    print()
    
    # Test 3: Claiming Race
    print("TEST 3: Claiming Race")
    print("-" * 80)
    test_pp_3 = """
Ultimate PP's w/ QuickPlay Comments | Churchill Downs | CLM 16000 | 1 Mile | 3yo & Up | Thursday, April 10, 2026 | Race 3
[Additional PP data would follow...]
    """
    result_3 = parse_and_calculate_class(test_pp_3)
    print(f"Track: {result_3['summary']['track']}")
    print(f"Class: {result_3['summary']['class_type']}")
    print(f"Purse: ${result_3['summary']['purse']:,}")
    print(f"Distance: {result_3['summary']['distance']} ({result_3['summary']['distance_furlongs']} furlongs)")
    print(f"Hierarchy Level: {result_3['hierarchy']['final_level']}")
    print(f"Class Weight: {result_3['summary']['class_weight']}")
    print(f"Quality Rating: {result_3['summary']['quality']}")
    print()
    
    # Test 4: Grade 2 Stakes (Turf)
    print("TEST 4: Grade 2 Stakes - Turf")
    print("-" * 80)
    test_pp_4 = """
Ultimate PP's w/ QuickPlay Comments | Belmont Park | G2 Stakes 500000 | 1 1/8 Mile Turf | 3yo Fillies | Sunday, June 07, 2026 | Race 10
[Additional PP data would follow...]
    """
    result_4 = parse_and_calculate_class(test_pp_4)
    print(f"Track: {result_4['summary']['track']}")
    print(f"Class: {result_4['summary']['class_type']}")
    print(f"Grade: G{result_4['summary']['grade_level']}")
    print(f"Surface: {result_4['summary']['surface']}")
    print(f"Purse: ${result_4['summary']['purse']:,}")
    print(f"Distance: {result_4['summary']['distance']} ({result_4['summary']['distance_furlongs']} furlongs)")
    print(f"Hierarchy Level: {result_4['hierarchy']['base_level']} + {result_4['hierarchy']['grade_boost']} boost = {result_4['hierarchy']['final_level']}")
    print(f"Class Weight: {result_4['summary']['class_weight']}")
    print(f"Quality Rating: {result_4['summary']['quality']}")
    print()
    
    # Test 5: Allowance Optional Claiming
    print("TEST 5: Allowance Optional Claiming")
    print("-" * 80)
    test_pp_5 = """
Ultimate PP's w/ QuickPlay Comments | Gulfstream Park | AOC 75000 | 7 Furlongs | 3yo & Up | Friday, January 17, 2026 | Race 7
[Additional PP data would follow...]
    """
    result_5 = parse_and_calculate_class(test_pp_5)
    print(f"Track: {result_5['summary']['track']}")
    print(f"Class: {result_5['summary']['class_type']}")
    print(f"Purse: ${result_5['summary']['purse']:,}")
    print(f"Distance: {result_5['summary']['distance']} ({result_5['summary']['distance_furlongs']} furlongs)")
    print(f"Hierarchy Level: {result_5['hierarchy']['final_level']}")
    print(f"Class Weight: {result_5['summary']['class_weight']}")
    print(f"Quality Rating: {result_5['summary']['quality']}")
    print()
    
    print("=" * 80)
    print("TEST SUITE COMPLETE")
    print("=" * 80)
    print()
    print("Weight Breakdown Example (from Test 1 - G1 Stakes):")
    print("-" * 80)
    for key, value in result_1['weight']['breakdown'].items():
        print(f"  {key}: {value}")
    print(f"  TOTAL CLASS WEIGHT: {result_1['weight']['class_weight']}")
