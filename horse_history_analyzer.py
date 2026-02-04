"""
Horse History Analyzer Module for Brisnet Past Performances

This module analyzes each individual horse's past performance history to identify
critical handicapping factors including class movement, distance changes, jockey/trainer
changes, surface switches, and workout quality (bullets).

Key Features:
- Parse individual horse past performances from Brisnet PP text
- Detect class movement (up/down in class)
- Detect distance changes (stretching out/sprinting)
- Identify jockey and trainer changes
- Track surface changes (dirt ↔ turf)
- Analyze recent workouts for bullet works
- Determine form cycles and trends

Author: Horse Racing Picks System
Date: February 2026
Python: 3.11+
"""

import re
from typing import Dict, Any, List, Optional, Tuple
from datetime import datetime, timedelta
import logging

# Import from race_class_parser for class hierarchy
try:
    from race_class_parser import (
        parse_race_conditions, 
        get_hierarchy_level,
        CLASS_MAP,
        LEVEL_MAP
    )
except ImportError:
    # Fallback if module not available
    LEVEL_MAP = {
        'Grade 1 Stakes': 10,
        'Grade 2 Stakes': 9,
        'Grade 3 Stakes': 8,
        'Stakes': 6,
        'Allowance': 4,
        'Claiming': 3,
        'Maiden': 2,
    }

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# DISTANCE CONVERSION UTILITIES
# ============================================================================

def parse_distance_to_furlongs(distance_str: str) -> float:
    """
    Parse ANY US horse racing distance format to furlongs.
    
    Handles all standard formats:
    - "6f" or "6 f" → 6.0
    - "6 Furlongs" → 6.0
    - "1m" or "1 Mile" → 8.0
    - "1 1/16m" or "1 1/16 Miles" → 8.5
    - "1 1/8m" or "1 1/8 Miles" → 9.0
    - "1 1/4m" or "1 1/4 Miles" → 10.0
    - "7½f" → 7.5
    - "220y" or "220 Yards" → 1.0
    
    Args:
        distance_str: Distance string in any US racing format
        
    Returns:
        Distance in furlongs (float)
    """
    try:
        dist_clean = distance_str.lower().strip()
        
        # Replace unicode fractions
        dist_clean = dist_clean.replace('½', ' 1/2').replace('¼', ' 1/4').replace('¾', ' 3/4')
        dist_clean = dist_clean.replace('⅛', ' 1/8').replace('⅜', ' 3/8').replace('⅝', ' 5/8').replace('⅞', ' 7/8')
        dist_clean = dist_clean.replace('⅙', ' 1/16')
        
        # FURLONGS (most common format)
        if 'furlong' in dist_clean or dist_clean.endswith('f') or ' f' in dist_clean:
            # Extract: "6f", "7 1/2f", "6.5 furlongs"
            num_match = re.search(r'(\d+(?:\.\d+)?|\d+\s+\d+/\d+)', dist_clean)
            if num_match:
                return _parse_number_with_fraction(num_match.group(1))
        
        # MILES (convert to furlongs: 1 mile = 8 furlongs)
        elif 'mile' in dist_clean or dist_clean.endswith('m'):
            # Extract: "1m", "1 1/8 miles", "1.25m"
            mile_match = re.search(r'(\d+)(?:\s+(\d+)/(\d+))?', dist_clean)
            if mile_match:
                miles = float(mile_match.group(1))
                if mile_match.group(2):  # Has fractional part
                    miles += float(mile_match.group(2)) / float(mile_match.group(3))
                return round(miles * 8.0, 1)
        
        # YARDS (220 yards = 1 furlong)
        elif 'yard' in dist_clean or dist_clean.endswith('y'):
            yard_match = re.search(r'(\d+)', dist_clean)
            if yard_match:
                yards = float(yard_match.group(1))
                return round(yards / 220.0, 1)
        
        return 0.0
        
    except Exception as e:
        logger.warning(f"Error parsing distance '{distance_str}': {e}")
        return 0.0


def _parse_number_with_fraction(num_str: str) -> float:
    """Parse number with optional fraction (e.g., '7', '7 1/2', '7.5')"""
    try:
        num_str = num_str.strip()
        
        # Check for fraction: "7 1/2"
        frac_match = re.match(r'(\d+)?\s*(\d+)/(\d+)', num_str)
        if frac_match:
            whole = float(frac_match.group(1) or 0)
            return whole + (float(frac_match.group(2)) / float(frac_match.group(3)))
        
        return float(num_str)
    except Exception:
        return 0.0


def categorize_distance(furlongs: float) -> str:
    """
    Categorize distance into racing terms.
    
    Args:
        furlongs: Distance in furlongs
        
    Returns:
        Category: 'Sprint', 'Mile', 'Route', 'Marathon'
    """
    if furlongs < 7.0:
        return 'Sprint'
    elif furlongs < 8.5:
        return 'Mile'
    elif furlongs < 11.0:
        return 'Route'
    else:
        return 'Marathon'


# ============================================================================
# HORSE PAST PERFORMANCE PARSER
# ============================================================================

def parse_horse_past_performances(horse_pp_text: str) -> Dict[str, Any]:
    """
    Parse a single horse's complete past performance section from Brisnet PP.
    
    Extracts:
    - Horse name, jockey, trainer
    - Past race lines (date, track, distance, surface, class, finish)
    - Recent workouts
    - Current connections vs. previous races
    
    Args:
        horse_pp_text: Text block for one horse from Brisnet PP
        
    Returns:
        Dictionary with horse info and past race history
    """
    result = {
        'horse_name': None,
        'current_jockey': None,
        'current_trainer': None,
        'past_races': [],
        'recent_workouts': [],
        'career_summary': {},
    }
    
    try:
        lines = horse_pp_text.split('\n')
        
        # Parse horse name (typically first line or after program number)
        for line in lines[:5]:
            # Look for horse name pattern (usually capitalized, may have program number)
            name_match = re.search(r'^\s*\d+\s+([A-Z][A-Za-z\s\']+)', line)
            if name_match:
                result['horse_name'] = name_match.group(1).strip()
                break
        
        # Parse past race lines
        # Brisnet format: Date Track Dist Surface Class/Type Finish Position etc.
        # Example: "29Dec25Tup 6½ ft :22¨ :45ª1:11 4000Clm 3 5 4 2¨"
        for line in lines:
            race_line = _parse_race_line(line)
            if race_line:
                result['past_races'].append(race_line)
        
        # Parse workouts (look for "WORKOUTS:" section or workout pattern)
        in_workout_section = False
        for line in lines:
            if 'workout' in line.lower() or 'wk' in line.lower():
                in_workout_section = True
                continue
            
            if in_workout_section:
                workout = _parse_workout_line(line)
                if workout:
                    result['recent_workouts'].append(workout)
        
        # Extract current jockey/trainer (from most recent race or header)
        if result['past_races']:
            result['current_jockey'] = result['past_races'][0].get('jockey')
            result['current_trainer'] = result['past_races'][0].get('trainer')
        
        return result
        
    except Exception as e:
        logger.error(f"Error parsing horse PP: {e}")
        return result


def _parse_race_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse a single past performance race line.
    
    Brisnet format example:
    "29Dec25Tup 6½ ft :22¨ :45ª1:11 4000Clm 3 5 4 2¨"
    
    Extracts: date, track, distance, surface, class, finish position
    """
    try:
        # Date pattern: DDMmmYYTrack (e.g., "29Dec25Tup")
        date_track_match = re.search(r'(\d{2}[A-Za-z]{3}\d{2})([A-Za-z]{2,4})', line)
        if not date_track_match:
            return None
        
        race_data = {
            'date': date_track_match.group(1),
            'track_code': date_track_match.group(2),
            'distance_furlongs': 0.0,
            'surface': None,
            'class_type': None,
            'class_level': 0,
            'finish_position': None,
            'raw_line': line,
        }
        
        # Distance (e.g., "6½", "1 1/16m")
        dist_match = re.search(r'(\d+(?:[½¼¾⅛⅜⅝⅞]|\.?\d*))\s*([fm])?', line)
        if dist_match:
            race_data['distance_furlongs'] = parse_distance_to_furlongs(dist_match.group(0))
        
        # Surface condition (ft=fast, gd=good, sy=sloppy, etc.)
        surf_match = re.search(r'\b(ft|fm|gd|sy|sl|my|hy|wf)\b', line.lower())
        if surf_match:
            race_data['surface'] = surf_match.group(1).upper()
        
        # Class type (e.g., "4000Clm", "G1", "Alw")
        class_match = re.search(r'(\d+)?([A-Za-z©§¨]+)', line)
        if class_match:
            class_amount = class_match.group(1)
            class_abbrev = class_match.group(2)
            race_data['class_type'] = class_abbrev
            
            # Map to hierarchy level
            if class_abbrev.upper() in LEVEL_MAP:
                race_data['class_level'] = LEVEL_MAP[class_abbrev.upper()]
        
        # Finish position (look for single digit followed by position indicators)
        finish_match = re.search(r'\b(\d{1,2})(?:[¨ª©§]|\s|$)', line)
        if finish_match:
            race_data['finish_position'] = int(finish_match.group(1))
        
        return race_data
        
    except Exception as e:
        logger.warning(f"Error parsing race line '{line}': {e}")
        return None


def _parse_workout_line(line: str) -> Optional[Dict[str, Any]]:
    """
    Parse workout line to extract date, distance, time, and bullet indicator.
    
    Format example: "28Jan26 Tup 4f ft :48.2 B"
    """
    try:
        workout = {
            'date': None,
            'track': None,
            'distance': None,
            'time': None,
            'is_bullet': False,
        }
        
        # Date
        date_match = re.search(r'(\d{2}[A-Za-z]{3}\d{2})', line)
        if date_match:
            workout['date'] = date_match.group(1)
        
        # Distance
        dist_match = re.search(r'(\d+\.?\d*)\s*f', line.lower())
        if dist_match:
            workout['distance'] = float(dist_match.group(1))
        
        # Time
        time_match = re.search(r':?(\d{1,2})\.(\d{1,2})', line)
        if time_match:
            workout['time'] = f"{time_match.group(1)}.{time_match.group(2)}"
        
        # Bullet indicator (B or "bullet")
        if re.search(r'\bB\b|bullet', line, re.IGNORECASE):
            workout['is_bullet'] = True
        
        return workout if workout['date'] else None
        
    except Exception as e:
        logger.warning(f"Error parsing workout line '{line}': {e}")
        return None


# ============================================================================
# CLASS MOVEMENT ANALYZER
# ============================================================================

def analyze_class_movement(
    current_class_level: int,
    past_races: List[Dict[str, Any]],
    lookback_races: int = 3
) -> Dict[str, Any]:
    """
    Determine if horse is moving up or down in class.
    
    Compares today's race class level to average of recent past races.
    
    Args:
        current_class_level: Today's race hierarchy level (0-13)
        past_races: List of past race dictionaries from parse_horse_past_performances
        lookback_races: How many recent races to analyze (default: 3)
        
    Returns:
        Dictionary with:
            - movement: str ('Up', 'Down', 'Same', 'Unknown')
            - class_change: int (positive = moving up, negative = moving down)
            - avg_recent_class: float
            - is_significant: bool (change >= 2 levels)
    """
    result = {
        'movement': 'Unknown',
        'class_change': 0,
        'avg_recent_class': 0.0,
        'is_significant': False,
        'recent_classes': [],
    }
    
    try:
        if not past_races:
            return result
        
        # Get class levels from recent races
        recent_classes = []
        for race in past_races[:lookback_races]:
            class_level = race.get('class_level', 0)
            if class_level > 0:
                recent_classes.append(class_level)
        
        if not recent_classes:
            return result
        
        result['recent_classes'] = recent_classes
        result['avg_recent_class'] = round(sum(recent_classes) / len(recent_classes), 1)
        
        # Calculate class change
        result['class_change'] = current_class_level - int(result['avg_recent_class'])
        
        # Determine movement
        if result['class_change'] >= 1:
            result['movement'] = 'Up'
        elif result['class_change'] <= -1:
            result['movement'] = 'Down'
        else:
            result['movement'] = 'Same'
        
        # Significant if change >= 2 levels
        result['is_significant'] = abs(result['class_change']) >= 2
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing class movement: {e}")
        return result


# ============================================================================
# DISTANCE MOVEMENT ANALYZER
# ============================================================================

def analyze_distance_movement(
    current_distance_furlongs: float,
    past_races: List[Dict[str, Any]],
    lookback_races: int = 3
) -> Dict[str, Any]:
    """
    Determine if horse is stretching out or sprinting back.
    
    Args:
        current_distance_furlongs: Today's race distance in furlongs
        past_races: List of past race dictionaries
        lookback_races: How many recent races to analyze (default: 3)
        
    Returns:
        Dictionary with:
            - movement: str ('Stretching Out', 'Sprinting Back', 'Same', 'Unknown')
            - distance_change: float (furlongs difference)
            - avg_recent_distance: float
            - current_category: str ('Sprint', 'Mile', 'Route', 'Marathon')
            - recent_category: str
            - category_change: bool (changed categories)
    """
    result = {
        'movement': 'Unknown',
        'distance_change': 0.0,
        'avg_recent_distance': 0.0,
        'current_category': categorize_distance(current_distance_furlongs),
        'recent_category': None,
        'category_change': False,
    }
    
    try:
        if not past_races:
            return result
        
        # Get distances from recent races
        recent_distances = []
        for race in past_races[:lookback_races]:
            dist = race.get('distance_furlongs', 0.0)
            if dist > 0:
                recent_distances.append(dist)
        
        if not recent_distances:
            return result
        
        result['avg_recent_distance'] = round(sum(recent_distances) / len(recent_distances), 1)
        result['recent_category'] = categorize_distance(result['avg_recent_distance'])
        
        # Calculate distance change
        result['distance_change'] = round(current_distance_furlongs - result['avg_recent_distance'], 1)
        
        # Determine movement
        if result['distance_change'] >= 1.0:
            result['movement'] = 'Stretching Out'
        elif result['distance_change'] <= -1.0:
            result['movement'] = 'Sprinting Back'
        else:
            result['movement'] = 'Same'
        
        # Check for category change (Sprint → Route, etc.)
        result['category_change'] = (result['current_category'] != result['recent_category'])
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing distance movement: {e}")
        return result


# ============================================================================
# JOCKEY/TRAINER CHANGE ANALYZER
# ============================================================================

def analyze_connections_changes(
    current_jockey: Optional[str],
    current_trainer: Optional[str],
    past_races: List[Dict[str, Any]],
    lookback_races: int = 3
) -> Dict[str, Any]:
    """
    Detect jockey and trainer changes.
    
    Args:
        current_jockey: Today's jockey name
        current_trainer: Today's trainer name
        past_races: List of past race dictionaries
        lookback_races: How many recent races to check
        
    Returns:
        Dictionary with:
            - jockey_change: bool
            - trainer_change: bool
            - previous_jockey: str
            - previous_trainer: str
            - jockey_consistent: bool (same jockey in all recent races)
            - trainer_consistent: bool
    """
    result = {
        'jockey_change': False,
        'trainer_change': False,
        'previous_jockey': None,
        'previous_trainer': None,
        'jockey_consistent': True,
        'trainer_consistent': True,
        'recent_jockeys': [],
        'recent_trainers': [],
    }
    
    try:
        if not past_races:
            return result
        
        # Get recent jockeys/trainers
        for race in past_races[:lookback_races]:
            jockey = race.get('jockey')
            trainer = race.get('trainer')
            
            if jockey:
                result['recent_jockeys'].append(jockey)
            if trainer:
                result['recent_trainers'].append(trainer)
        
        # Check for changes
        if result['recent_jockeys']:
            result['previous_jockey'] = result['recent_jockeys'][0]
            result['jockey_change'] = (current_jockey != result['previous_jockey'])
            result['jockey_consistent'] = len(set(result['recent_jockeys'])) == 1
        
        if result['recent_trainers']:
            result['previous_trainer'] = result['recent_trainers'][0]
            result['trainer_change'] = (current_trainer != result['previous_trainer'])
            result['trainer_consistent'] = len(set(result['recent_trainers'])) == 1
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing connections changes: {e}")
        return result


# ============================================================================
# SURFACE CHANGE ANALYZER
# ============================================================================

def analyze_surface_change(
    current_surface: Optional[str],
    past_races: List[Dict[str, Any]],
    lookback_races: int = 3
) -> Dict[str, Any]:
    """
    Detect surface changes (Dirt ↔ Turf ↔ Synthetic).
    
    Args:
        current_surface: Today's surface ('Dirt', 'Turf', 'Synthetic')
        past_races: List of past race dictionaries
        lookback_races: How many recent races to check
        
    Returns:
        Dictionary with:
            - surface_change: bool
            - previous_surface: str
            - change_type: str ('Dirt to Turf', 'Turf to Dirt', etc.)
            - turf_experience: int (number of turf races in history)
            - dirt_experience: int
    """
    result = {
        'surface_change': False,
        'previous_surface': None,
        'change_type': None,
        'turf_experience': 0,
        'dirt_experience': 0,
        'synthetic_experience': 0,
        'surface_consistency': True,
    }
    
    try:
        if not past_races or not current_surface:
            return result
        
        # Normalize surface names
        current_surf = _normalize_surface(current_surface)
        
        # Count surface types in history
        recent_surfaces = []
        for race in past_races[:lookback_races]:
            surf = race.get('surface')
            if surf:
                surf_norm = _normalize_surface(surf)
                recent_surfaces.append(surf_norm)
                
                if surf_norm == 'Turf':
                    result['turf_experience'] += 1
                elif surf_norm == 'Dirt':
                    result['dirt_experience'] += 1
                elif surf_norm == 'Synthetic':
                    result['synthetic_experience'] += 1
        
        if recent_surfaces:
            result['previous_surface'] = recent_surfaces[0]
            result['surface_change'] = (current_surf != result['previous_surface'])
            result['surface_consistency'] = len(set(recent_surfaces)) == 1
            
            if result['surface_change']:
                result['change_type'] = f"{result['previous_surface']} to {current_surf}"
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing surface change: {e}")
        return result


def _normalize_surface(surface_str: str) -> str:
    """Normalize surface string to standard names."""
    surf_lower = surface_str.lower()
    if 'turf' in surf_lower or 't' == surf_lower or 'grass' in surf_lower:
        return 'Turf'
    elif 'synth' in surf_lower or 'poly' in surf_lower or 'tapseta' in surf_lower:
        return 'Synthetic'
    else:
        return 'Dirt'


# ============================================================================
# WORKOUT QUALITY ANALYZER
# ============================================================================

def analyze_recent_workouts(
    workouts: List[Dict[str, Any]],
    days_back: int = 14
) -> Dict[str, Any]:
    """
    Analyze recent workout quality and bullet works.
    
    Args:
        workouts: List of workout dictionaries from parse_horse_past_performances
        days_back: How many days to look back (default: 14)
        
    Returns:
        Dictionary with:
            - total_workouts: int
            - bullet_count: int
            - bullet_percentage: float
            - most_recent_workout_days_ago: int
            - has_recent_bullet: bool (bullet within last 7 days)
            - workout_frequency: str ('Active', 'Moderate', 'Light')
    """
    result = {
        'total_workouts': 0,
        'bullet_count': 0,
        'bullet_percentage': 0.0,
        'most_recent_workout_days_ago': None,
        'has_recent_bullet': False,
        'workout_frequency': 'Unknown',
        'bullet_works': [],
    }
    
    try:
        if not workouts:
            return result
        
        result['total_workouts'] = len(workouts)
        
        # Count bullets
        for workout in workouts:
            if workout.get('is_bullet'):
                result['bullet_count'] += 1
                result['bullet_works'].append(workout)
        
        if result['total_workouts'] > 0:
            result['bullet_percentage'] = round(
                (result['bullet_count'] / result['total_workouts']) * 100, 1
            )
        
        # Check for recent bullet (within last 7 days)
        for workout in workouts[:3]:  # Check 3 most recent
            if workout.get('is_bullet'):
                result['has_recent_bullet'] = True
                break
        
        # Determine workout frequency
        if result['total_workouts'] >= 4:
            result['workout_frequency'] = 'Active'
        elif result['total_workouts'] >= 2:
            result['workout_frequency'] = 'Moderate'
        else:
            result['workout_frequency'] = 'Light'
        
        return result
        
    except Exception as e:
        logger.error(f"Error analyzing workouts: {e}")
        return result


# ============================================================================
# COMPREHENSIVE HORSE ANALYZER (MAIN FUNCTION)
# ============================================================================

def analyze_horse_form_cycle(
    horse_pp_text: str,
    current_race_class_level: int,
    current_race_distance_furlongs: float,
    current_race_surface: Optional[str] = None
) -> Dict[str, Any]:
    """
    Comprehensive analysis of single horse's form cycle and trends.
    
    This is the main entry point for analyzing one horse. It chains together
    all analysis functions to produce a complete form cycle assessment.
    
    Args:
        horse_pp_text: Brisnet PP text for one horse
        current_race_class_level: Today's race hierarchy level
        current_race_distance_furlongs: Today's race distance
        current_race_surface: Today's surface ('Dirt', 'Turf', 'Synthetic')
        
    Returns:
        Comprehensive dictionary with all analyses:
            - horse_info: Basic horse information
            - class_analysis: Class movement analysis
            - distance_analysis: Distance movement analysis
            - connections_analysis: Jockey/trainer changes
            - surface_analysis: Surface change analysis
            - workout_analysis: Recent workout quality
            - form_summary: Overall form assessment
    """
    try:
        # Step 1: Parse horse PP
        horse_data = parse_horse_past_performances(horse_pp_text)
        
        # Step 2: Analyze class movement
        class_analysis = analyze_class_movement(
            current_race_class_level,
            horse_data['past_races']
        )
        
        # Step 3: Analyze distance movement
        distance_analysis = analyze_distance_movement(
            current_race_distance_furlongs,
            horse_data['past_races']
        )
        
        # Step 4: Analyze connections changes
        connections_analysis = analyze_connections_changes(
            horse_data['current_jockey'],
            horse_data['current_trainer'],
            horse_data['past_races']
        )
        
        # Step 5: Analyze surface change
        surface_analysis = analyze_surface_change(
            current_race_surface,
            horse_data['past_races']
        )
        
        # Step 6: Analyze recent workouts
        workout_analysis = analyze_recent_workouts(
            horse_data['recent_workouts']
        )
        
        # Step 7: Generate form summary
        form_summary = _generate_form_summary(
            class_analysis,
            distance_analysis,
            connections_analysis,
            surface_analysis,
            workout_analysis
        )
        
        return {
            'horse_info': {
                'name': horse_data['horse_name'],
                'jockey': horse_data['current_jockey'],
                'trainer': horse_data['current_trainer'],
            },
            'class_analysis': class_analysis,
            'distance_analysis': distance_analysis,
            'connections_analysis': connections_analysis,
            'surface_analysis': surface_analysis,
            'workout_analysis': workout_analysis,
            'form_summary': form_summary,
        }
        
    except Exception as e:
        logger.error(f"Error in comprehensive horse analysis: {e}")
        return {
            'error': str(e),
            'horse_info': {},
            'form_summary': {'overall_rating': 'Unknown'},
        }


def _generate_form_summary(
    class_analysis: Dict,
    distance_analysis: Dict,
    connections_analysis: Dict,
    surface_analysis: Dict,
    workout_analysis: Dict
) -> Dict[str, Any]:
    """
    Generate overall form cycle summary based on all factors.
    
    Returns:
        Dictionary with:
            - overall_rating: str ('Positive', 'Neutral', 'Negative')
            - key_factors: list (notable factors affecting form)
            - confidence: str ('High', 'Medium', 'Low')
    """
    positive_factors = []
    negative_factors = []
    neutral_factors = []
    
    # Class movement
    if class_analysis['movement'] == 'Down':
        positive_factors.append(f"Dropping in class ({class_analysis['class_change']} levels)")
    elif class_analysis['movement'] == 'Up':
        if class_analysis['is_significant']:
            negative_factors.append(f"Significant class rise (+{class_analysis['class_change']} levels)")
        else:
            neutral_factors.append("Slight class rise")
    
    # Distance
    if distance_analysis['category_change']:
        negative_factors.append(f"Distance category change: {distance_analysis['change_type']}")
    
    # Surface
    if surface_analysis['surface_change']:
        change_type = surface_analysis['change_type']
        if 'Turf' in change_type:
            if surface_analysis['turf_experience'] < 2:
                negative_factors.append(f"Limited turf experience ({change_type})")
            else:
                neutral_factors.append(f"Surface change: {change_type}")
    
    # Workouts
    if workout_analysis['has_recent_bullet']:
        positive_factors.append("Recent bullet workout")
    elif workout_analysis['bullet_percentage'] >= 50:
        positive_factors.append(f"High bullet rate ({workout_analysis['bullet_percentage']}%)")
    
    # Connections
    if connections_analysis['jockey_change'] or connections_analysis['trainer_change']:
        neutral_factors.append("Jockey or trainer change")
    
    # Determine overall rating
    pos_count = len(positive_factors)
    neg_count = len(negative_factors)
    
    if pos_count > neg_count:
        overall_rating = 'Positive'
        confidence = 'High' if pos_count >= 2 else 'Medium'
    elif neg_count > pos_count:
        overall_rating = 'Negative'
        confidence = 'High' if neg_count >= 2 else 'Medium'
    else:
        overall_rating = 'Neutral'
        confidence = 'Medium' if (pos_count + neg_count) >= 2 else 'Low'
    
    return {
        'overall_rating': overall_rating,
        'positive_factors': positive_factors,
        'negative_factors': negative_factors,
        'neutral_factors': neutral_factors,
        'confidence': confidence,
    }


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == '__main__':
    print("=" * 80)
    print("HORSE HISTORY ANALYZER - TEST SUITE")
    print("=" * 80)
    print()
    
    # Test distance parsing
    print("TEST 1: Distance Parsing")
    print("-" * 80)
    test_distances = [
        "6f", "6 Furlongs", "7½f",
        "1m", "1 Mile", "1 1/16m", "1 1/8 Miles", "1 1/4m",
        "220y", "330 Yards"
    ]
    for dist in test_distances:
        furlongs = parse_distance_to_furlongs(dist)
        category = categorize_distance(furlongs)
        print(f"  {dist:20s} → {furlongs:5.1f} furlongs ({category})")
    print()
    
    # Test horse PP parsing (sample)
    print("TEST 2: Sample Horse Analysis")
    print("-" * 80)
    sample_horse_pp = """
5 SAMPLE HORSE                  J. Smith / T. Jones
29Dec25Tup 6½ ft :22¨ :45ª1:11 4000Clm 3 5 4 2¨
15Nov25GP 6f ft :22¨ :45¨1:10¨ 5000Clm 1 2 1 1©
01Oct25Kee 7f gd :23 :46ª1:23 6000Alw 5 7 6 4ª
WORKOUTS: 28Jan26 Tup 4f ft :48.2 B | 21Jan26 Tup 5f ft 1:01.1
    """
    
    result = analyze_horse_form_cycle(
        sample_horse_pp,
        current_race_class_level=3,  # Claiming race
        current_race_distance_furlongs=6.0,
        current_race_surface='Dirt'
    )
    
    print(f"Horse: {result['horse_info']['name']}")
    print(f"Class Movement: {result['class_analysis']['movement']} ({result['class_analysis']['class_change']} levels)")
    print(f"Distance Movement: {result['distance_analysis']['movement']} ({result['distance_analysis']['distance_change']} furlongs)")
    print(f"Workout Quality: {result['workout_analysis']['bullet_count']} bullets / {result['workout_analysis']['total_workouts']} works")
    print(f"Form Rating: {result['form_summary']['overall_rating']} (Confidence: {result['form_summary']['confidence']})")
    print()
    print("Key Factors:")
    for factor in result['form_summary']['positive_factors']:
        print(f"  ✓ {factor}")
    for factor in result['form_summary']['negative_factors']:
        print(f"  ✗ {factor}")
    
    print()
    print("=" * 80)
    print("TESTS COMPLETE")
    print("=" * 80)
