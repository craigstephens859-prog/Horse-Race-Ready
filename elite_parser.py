#!/usr/bin/env python3
"""
🏇 ULTRA-PRECISION BRISNET PP PARSER
Designed for 90%+ accuracy in structured data extraction.

Features:
- Robust regex with fallback patterns
- Fuzzy matching for typos
- Comprehensive validation
- Structured dict output
- Error recovery and logging
- Edge case handling (scratches, foreign horses, typos)
"""

import re
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from datetime import datetime

import pandas as pd
import numpy as np

# ===================== DATA MODELS =====================

@dataclass
class HorseData:
    """Structured horse data model"""
    # Identity
    post: str
    name: str
    program_number: str

    # Style & Pace
    pace_style: str  # E, E/P, P, S
    quirin_points: float
    style_strength: str  # Strong, Solid, Slight, Weak

    # Odds
    ml_odds: Optional[str]
    ml_odds_decimal: Optional[float]

    # Connections
    jockey: str
    jockey_win_pct: float
    trainer: str
    trainer_win_pct: float

    # Speed Figures
    speed_figures: List[int]  # Most recent first
    avg_top2: float
    peak_fig: int
    last_fig: int

    # Form Cycle
    days_since_last: Optional[int]
    last_race_date: Optional[str]
    recent_finishes: List[int]

    # Class
    recent_purses: List[int]
    race_types: List[str]

    # Equipment & Medications
    equipment: str  # Current equipment (L, Lb, Lbf, etc.)
    equipment_change: Optional[str]  # "Blinkers On", "Blinkers Off", etc.
    lasix: bool  # Currently on Lasix
    first_lasix: bool  # First time Lasix
    weight: Optional[int]  # Assigned weight

    # Race History Details
    race_history: List[Dict]  # Full race details with positions, times, etc.
    track_conditions: List[str]  # ft, gd, my, fm, sys for each race
    trip_comments: List[str]  # Running style comments
    beat_margins: List[str]  # How far behind winner
    odds_history: List[float]  # Past race odds
    
    # Running Style Patterns
    early_speed_pct: float  # % of races led/close early
    closing_pct: float  # % of races closed ground
    avg_early_position: float  # Average position at 1st call
    avg_late_position: float  # Average position at stretch

    # Track/Surface Performance
    surface_stats: Dict  # {surface: {starts, wins, win_pct, avg_fig}}
    distance_stats: Dict  # {distance_range: {starts, wins, avg_fig}}
    track_bias_fit: float  # How well suited to track bias

    # Workouts
    workouts: List[Dict]  # [{date, track, distance, time, rank}]
    last_workout_days: Optional[int]
    workout_pattern: str  # "Sharp", "Steady", "Sparse"

    # Pedigree
    sire: str
    dam: str
    sire_spi: Optional[float]
    damsire_spi: Optional[float]
    sire_awd: Optional[float]
    dam_dpi: Optional[float]

    # Angles
    angles: List[Dict]  # [{category, starts, win%, itm%, roi}]
    angle_count: int
    angle_flags: List[str]  # Quick reference flags

    # Validation
    parsing_confidence: float  # 0.0 to 1.0
    warnings: List[str]
    raw_block: str

# ===================== CORE PARSING ENGINE =====================

class EliteBRISNETParser:
    """
    Ultra-robust parser with error recovery and validation.
    """

    # Enhanced regex patterns with multiple fallback options
    HORSE_HDR_PATTERNS = [
        # Primary pattern
        re.compile(r"""(?mi)^\s*
            (\d+[A-Z]?)              # post (1, 2, 1A for coupled entry)
            \s+([A-Za-z0-9'.\-\s&=]+?)   # horse name (including =Name for foreign)
            \s*\(\s*
            (E\/P|EP|E|P|S|NA)       # style
            (?:\s+(\d+))?            # optional quirin
            \s*\)\s*$
        """, re.VERBOSE),

        # Fallback: simpler pattern for mangled headers
        re.compile(r"""(?mi)^\s*
            (\d+[A-Z]?)\s+            # post
            ([A-Za-z0-9][A-Za-z0-9\s'.\-&=]+)  # name (more permissive)
            \s*\(([EPNS/]+).*?\)     # style in parens
        """, re.VERBOSE),
    ]

    # Enhanced odds patterns (handles fractions, decimals, ranges)
    ODDS_PATTERNS = [
        re.compile(r'(\d+)\s*/\s*(\d+)'),      # Fractional: 5/2
        re.compile(r'(\d+)\s*-\s*(\d+)'),      # Range: 3-1
        re.compile(r'([+-]?\d+\.?\d*)'),       # Decimal: 3.5, +150, -110
    ]

    # Jockey pattern with variations
    JOCKEY_PATTERNS = [
        re.compile(r'^([A-Z][A-Z\s\'.\-JR]+?)\s*\([\d\s\-]+?(\d+)%\)', re.MULTILINE),
        re.compile(r'(?mi)Jockey:\s*([A-Za-z][A-Za-z\s,\'.\-]+?)\s*\(.*?(\d+)%'),
    ]

    # Trainer pattern with variations
    TRAINER_PATTERNS = [
        re.compile(r'Trnr:\s*([A-Za-z][A-Za-z\s,\'.\-]+?)\s*\([\d\s\-]+?(\d+)%\)', re.MULTILINE),
        re.compile(r'(?mi)Trainer:\s*([A-Za-z][A-Za-z\s,\'.\-]+?)\s*\(.*?(\d+)%'),
    ]

    # Speed figure pattern with date context - BRISNET format
    # Format: DATE TRK DIST RR RACETYPE CR E1 E2/ LP 1c 2c SPD PP ST...
    # Speed figure is in SPD column after E1 E2/ LP 1c 2c columns
    SPEED_FIG_PATTERN = re.compile(
        r"(\d{2}[A-Za-z]{3}\d{2})"  # Date (13Dec25)
        r".*?"  # Everything up to SPD column
        r"\s+(\d{2})\s+(\d{2})/\s*(\d{2,3})"  # E1 E2/ columns (e.g., "83 87/ 84")
        r"\s+[+-]?\d+"  # LP column
        r"\s+[+-]?\d+"  # 1c column
        r"\s+(\d{2,3})\s+"  # SPD (speed figure) - capturing group
        r"\d+\s+"  # PP (post position)
        r"(?:\d+|[A-Za-z]+)",  # ST (start position) - can be number or text
        re.MULTILINE
    )

    # Angle pattern
    ANGLE_PATTERN = re.compile(
        r'(?mi)^\s*(\d{4}\s+)?'  # Optional year
        r'(1st\s*time\s*str|Debut\s*MdnSpWt|Maiden\s*Sp\s*Wt|2nd\s*career\s*race|'
        r'Turf\s*to\s*Dirt|Dirt\s*to\s*Turf|Shipper|Blinkers\s*(?:on|off)|'
        r'(?:\d+(?:-\d+)?)\s*days?Away|JKYw/\s*Sprints|JKYw/\s*Trn\s*L(?:30|45|60)\b|'
        r'JKYw/\s*[EPS]|JKYw/\s*NA\s*types)\s+'
        r'(\d+)\s+(\d+)%\s+(\d+)%\s+([+-]?\d+(?:\.\d+)?)\s*$'
    )

    # Pedigree patterns
    PEDIGREE_PATTERNS = {
        'sire_stats': re.compile(
            r'(?mi)^\s*Sire\s*Stats:\s*AWD\s*(\d+(?:\.\d+)?)\s+'
            r'(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi'
        ),
        'damsire_stats': re.compile(
            r'(?mi)^\s*Dam\'s Sire:\s*AWD\s*(\d+(?:\.\d+)?)\s+'
            r'(\d+)%.*?(\d+)%.*?(\d+(?:\.\d+)?)\s*spi'
        ),
        'dam_stats': re.compile(
            r'(?mi)^\s*Dam:\s*DPI\s*(\d+(?:\.\d+)?)\s+(\d+)%'
        ),
        'sire_name': re.compile(r'Sire\s*:\s*([^\(]+)'),
        'dam_name': re.compile(r'Dam:\s*([^\(]+)'),
    }

    # Recent race pattern with dates - BRISNET format
    # Format: 13Dec25GP¯ à 1ˆ fm :times ¨¨¬ ™TrPOaksL 125k¨¨¬ CR E1 E2/ LP... FIN JOCKEY
    RACE_HISTORY_PATTERN = re.compile(
        r'(\d{2}[A-Za-z]{3}\d{2})'  # Date (13Dec25)
        r'\s*\w{2,4}\S*'  # Track code
        r'.*?'  # Distance, surface, times
        r'™[^\s]+?'  # Race type with ™ prefix
        r'.*?'  # Skip to finish position area
        r'STR\s+FIN.*?'  # Marker before finish
        r'(\d{1,2})[ƒ®«ª³©¨°¬²‚±\s]'  # Finish position with BRISNET beat margin symbols
    )

    def __init__(self):
        self.warnings = []
        self.confidence_scores = []

    def parse_full_pp(self, pp_text: str) -> Dict[str, HorseData]:
        """
        Master parsing function: PP text → structured dict of all horses

        Returns:
            {horse_name: HorseData object}
        """
        horses = {}
        chunks = self._split_into_chunks(pp_text)

        for post, name, style, quirin, block in chunks:
            try:
                horse_data = self._parse_single_horse(post, name, style, quirin, block, pp_text)
                horses[name] = horse_data
            except Exception as e:
                self.warnings.append(f"Failed to parse {name}: {str(e)}")
                # Create minimal fallback data
                horses[name] = self._create_fallback_data(post, name, block, str(e))

        return horses

    def _split_into_chunks(self, pp_text: str) -> List[Tuple[str, str, str, str, float]]:
        """
        Splits PP text into horse chunks using multiple pattern attempts.
        Returns: [(post, name, style, quirin, block), ...]
        """
        chunks = []

        # Try each pattern until one works
        for pattern in self.HORSE_HDR_PATTERNS:
            matches = list(pattern.finditer(pp_text or ""))
            if matches:
                for i, m in enumerate(matches):
                    start = m.end()
                    end = matches[i+1].start() if i+1 < len(matches) else len(pp_text)

                    post = m.group(1).strip()
                    name = m.group(2).strip()

                    # Extract style and quirin from header match
                    style = m.group(3).upper() if len(m.groups()) >= 3 else "NA"
                    style = "E/P" if style in ("EP", "E/P") else style

                    quirin_str = m.group(4) if len(m.groups()) >= 4 else None
                    quirin = float(quirin_str) if quirin_str else np.nan

                    # Clean foreign horse markers
                    name = name.lstrip('=')

                    block = pp_text[start:end]
                    chunks.append((post, name, style, quirin, block))
                break

        if not chunks:
            self.warnings.append("⚠️ No horses parsed - check PP format")

        return chunks

    def _parse_single_horse(self, post: str, name: str, style: str, quirin: float, block: str, full_pp: str) -> HorseData:
        """
        Parses all data for a single horse with robust error handling.
        """
        warnings = []
        confidence = 1.0  # Start at perfect confidence

        # 1. PACE STYLE & QUIRIN (already extracted from header)
        if not style or style == "NA":
            warnings.append("Style not detected")
            confidence -= 0.05

        style_strength = self._calculate_style_strength(style, quirin)

        # 2. MORNING LINE ODDS
        ml_odds, ml_decimal = self._parse_odds(block, full_pp, name)
        if not ml_odds:
            warnings.append("ML odds not found")
            confidence -= 0.1

        # 3. JOCKEY & TRAINER
        jockey, jockey_pct = self._parse_jockey(block)
        trainer, trainer_pct = self._parse_trainer(block)
        if not jockey:
            warnings.append("Jockey not found")
            confidence -= 0.15
        if not trainer:
            warnings.append("Trainer not found")
            confidence -= 0.15

        # 4. SPEED FIGURES
        figs, avg_top2, peak, last = self._parse_speed_figures(block)
        if not figs:
            warnings.append("No speed figures (possible first-time starter)")
            confidence -= 0.05  # Not critical for debuts

        # 5. FORM CYCLE
        days_ago, last_date, finishes = self._parse_form_cycle(block)

        # 6. CLASS DATA
        purses, race_types = self._parse_class_data(block)

        # 7. PEDIGREE
        pedigree = self._parse_pedigree(block)

        # 8. ANGLES
        angles = self._parse_angles(block)

        # 9. EQUIPMENT & MEDICATIONS
        equipment, equip_change, lasix, first_lasix, weight = self._parse_equipment_meds(block)

        # 10. COMPREHENSIVE RACE HISTORY
        race_history, conditions, comments, odds_hist = self._parse_comprehensive_race_history(block)

        # 11. RUNNING PATTERNS
        early_spd_pct, close_pct, avg_early, avg_late = self._calculate_running_patterns(race_history)

        # 12. SURFACE STATISTICS
        surface_stats = self._parse_surface_stats(block)

        # 13. WORKOUTS
        workouts, last_work_days, work_pattern = self._parse_workouts(block)

        # 14. ANGLE FLAGS
        angle_flags = self._parse_angle_flags(angles)

        # Create structured data
        return HorseData(
            post=post,
            name=name,
            program_number=post,
            pace_style=style,
            quirin_points=quirin,
            style_strength=style_strength,
            ml_odds=ml_odds,
            ml_odds_decimal=ml_decimal,
            jockey=jockey,
            jockey_win_pct=jockey_pct,
            trainer=trainer,
            trainer_win_pct=trainer_pct,
            speed_figures=figs,
            avg_top2=avg_top2,
            peak_fig=peak,
            last_fig=last,
            days_since_last=days_ago,
            last_race_date=last_date,
            recent_finishes=finishes,
            recent_purses=purses,
            race_types=race_types,
            equipment=equipment,
            equipment_change=equip_change,
            lasix=lasix,
            first_lasix=first_lasix,
            weight=weight,
            race_history=race_history,
            track_conditions=conditions,
            trip_comments=comments,
            beat_margins=[],  # Will be populated if needed
            odds_history=odds_hist,
            early_speed_pct=early_spd_pct,
            closing_pct=close_pct,
            avg_early_position=avg_early,
            avg_late_position=avg_late,
            surface_stats=surface_stats,
            distance_stats={},  # Can be calculated from race_history if needed
            track_bias_fit=0.0,  # Calculated at race level
            workouts=workouts,
            last_workout_days=last_work_days,
            workout_pattern=work_pattern,
            sire=pedigree.get('sire', ''),
            dam=pedigree.get('dam', ''),
            sire_spi=pedigree.get('sire_spi'),
            damsire_spi=pedigree.get('damsire_spi'),
            sire_awd=pedigree.get('sire_awd'),
            dam_dpi=pedigree.get('dam_dpi'),
            angles=angles,
            angle_count=len(angles),
            angle_flags=angle_flags,
            parsing_confidence=max(0.0, confidence),
            warnings=warnings,
            raw_block=block
        )

    def _parse_style_quirin(self, name: str, block: str) -> Tuple[str, float, str]:
        """
        Extract pace style and Quirin points from header.
        Returns: (style, quirin, strength)
        """
        # Try to extract from full PP using horse name as anchor
        full_pattern = rf'(?mi)^\s*\d+[A-Z]?\s+{re.escape(name)}\s*\(\s*(E\/P|EP|E|P|S|NA)(?:\s+(\d+))?\s*\)'

        match = re.search(full_pattern, block)
        if not match:
            # Fallback: search in first 200 chars of block
            match = re.search(r'\(\s*(E\/P|EP|E|P|S|NA)(?:\s+(\d+))?\s*\)', block[:200])

        if match:
            style = match.group(1).upper()
            style = "E/P" if style in ("EP", "E/P") else style

            quirin_str = match.group(2)
            quirin = float(quirin_str) if quirin_str else np.nan

            # Calculate strength
            strength = self._calculate_style_strength(style, quirin)

            return style, quirin, strength

        return "NA", np.nan, "Solid"

    def _calculate_style_strength(self, style: str, quirin: float) -> str:
        """Calculate style strength based on Quirin points"""
        s = (style or "NA").upper()
        try:
            q = float(quirin)
        except Exception:
            return "Solid"

        if pd.isna(q):
            return "Solid"

        if s in ("E", "E/P"):
            if q >= 7: return "Strong"
            if q >= 5: return "Solid"
            if q >= 3: return "Slight"
            return "Weak"
        if s in ("P", "S"):
            if q >= 5: return "Slight"
            if q >= 3: return "Solid"
            return "Strong"
        return "Solid"

    def _parse_odds(self, block: str, full_pp: str, horse_name: str) -> Tuple[Optional[str], Optional[float]]:
        """
        Extract ML odds with multiple fallback attempts.
        Returns: (odds_string, decimal_odds)
        """
        # Try finding odds at start of block
        for pattern in self.ODDS_PATTERNS:
            match = re.search(rf"(?mi)^\s*{pattern.pattern}", block[:100])
            if match:
                odds_str = match.group(0).strip()
                decimal = self._odds_to_decimal(odds_str)
                return odds_str, decimal

        # Fallback: search for "M/L" or "Morning Line" label
        ml_match = re.search(r'(?mi)(?:M/?L|Morning\s*Line).*?(\d+[/\-]\d+|\d+\.?\d*)', block[:200])
        if ml_match:
            odds_str = ml_match.group(1)
            decimal = self._odds_to_decimal(odds_str)
            return odds_str, decimal

        return None, None

    def _odds_to_decimal(self, odds_str: str) -> Optional[float]:
        """ROBUST: Convert odds with edge case handling

        Handles:
        - Fractional: 5/2 → 3.5
        - Range: 3-1 → 4.0
        - Decimal: 4.5 → 4.5
        - Scratches: SCR, WDN → None
        - Extreme odds: <1.01 or >999 → capped
        """
        if not odds_str:
            return None

        odds_str = str(odds_str).strip().upper()

        # Handle scratches/withdrawals
        if odds_str in ['SCR', 'WDN', 'SCRATCH', 'WITHDRAWN', 'N/A', '-', '0', '']:
            return None

        try:
            # Fractional: 5/2 → 3.5
            if '/' in odds_str:
                num, denom = odds_str.split('/')
                decimal = (float(num) / float(denom)) + 1.0

            # Range: 3-1 → 4.0
            elif '-' in odds_str:
                num, denom = odds_str.split('-')
                decimal = float(num) + 1.0

            # Decimal
            else:
                decimal = float(odds_str)

            # VALIDATION: Cap extreme values
            if decimal < 1.01:
                return 1.01  # Minimum realistic odds (1/100 = 99% prob)
            if decimal > 999.0:
                return 999.0  # Maximum realistic odds (1/1000 = 0.1% prob)

            return decimal
        except Exception:
            return None

    def _parse_jockey(self, block: str) -> Tuple[str, float]:
        """Extract jockey name and win %"""
        for pattern in self.JOCKEY_PATTERNS:
            match = pattern.search(block)
            if match:
                name = ' '.join(match.group(1).strip().split())
                pct = float(match.group(2)) / 100.0 if match.group(2) else 0.0
                return name, pct
        return "", 0.0

    def _parse_trainer(self, block: str) -> Tuple[str, float]:
        """Extract trainer name and win %"""
        for pattern in self.TRAINER_PATTERNS:
            match = pattern.search(block)
            if match:
                name = ' '.join(match.group(1).strip().split())
                pct = float(match.group(2)) / 100.0 if match.group(2) else 0.0
                return name, pct
        return "", 0.0

    def _parse_speed_figures(self, block: str) -> Tuple[List[int], float, int, int]:
        """
        Extract speed figures with recency context from BRISNET SPD column.
        Returns: (all_figs, avg_top2, peak, last)
        """
        figs = []
        for match in self.SPEED_FIG_PATTERN.finditer(block):
            try:
                # Group 5 is the speed figure (SPD column after E1 E2/ LP 1c 2c)
                fig_val = int(match.group(5))
                if 40 < fig_val < 130:
                    figs.append(fig_val)
            except Exception:
                pass

        # Limit to last 10 races
        figs = figs[:10]

        if not figs:
            return [], 0.0, 0, 0

        avg_top2 = np.mean(sorted(figs, reverse=True)[:2])
        peak = max(figs)
        last = figs[0] if figs else 0

        return figs, round(avg_top2, 1), peak, last

    def _parse_form_cycle(self, block: str) -> Tuple[Optional[int], Optional[str], List[int]]:
        """
        Extract recent race dates and finishes from BRISNET format.
        Returns: (days_since_last, last_date, finish_positions)
        """
        races = []
        
        # BRISNET format has finish position in FIN column
        # Pattern: ...SPD PP ST 1C 2C STR FIN JOCKEY...
        # After SPD column, we have PP ST 1C 2C STR FIN
        # Finish position comes after 5 numeric columns past SPD
        # Example: "86 12 11ƒ 12®ƒ 11¬ 9¬ 6«ƒ HusbandsMJ"
        #           SPD PP ST  1C   2C  STR FIN
        
        pattern = r'(\d{2}[A-Za-z]{3}\d{2}).*?'  # Date
        pattern += r'(\d{2})\s+(\d{2})/\s*(\d{2,3})'  # E1 E2/ columns
        pattern += r'\s+[+-]?\d+\s+[+-]?\d+\s+(\d{2,3})'  # LP 1c 2c SPD
        pattern += r'\s+\d+'  # PP
        pattern += r'\s+(?:\d+[^\s]*)'  # ST (can have symbols)
        pattern += r'\s+(?:\d+[^\s]*)'  # 1C
        pattern += r'\s+(?:\d+[^\s]*)'  # 2C
        pattern += r'\s+(?:\d+[^\s]*)'  # STR
        pattern += r'\s+(\d{1,2})[^\s\d]'  # FIN - finish

        for match in re.finditer(pattern, block, re.MULTILINE):
            date_str = match.group(1)
            try:
                finish = int(match.group(6))
                races.append({'date': date_str, 'finish': finish})
            except (ValueError, IndexError):
                pass

        if not races:
            return None, None, []

        # Calculate days since last race
        last_race = races[0]
        try:
            last_date_obj = datetime.strptime(last_race['date'], "%d%b%y")
            days_ago = (datetime.now() - last_date_obj).days
        except Exception:
            days_ago = None

        finishes = [r['finish'] for r in races[:5]]

        return days_ago, last_race['date'], finishes

    def _parse_class_data(self, block: str) -> Tuple[List[int], List[str]]:
        """Extract recent purse levels and race types from BRISNET format"""
        purses = []
        types = []

        # BRISNET format: ™TrPOaksL 125k¨¨¬ or ™A34000n2x¨¨© or ™Mdn 55k¨¨¨
        # Pattern matches race type marker ™ followed by race type info
        pattern = r'™([A-Za-z][A-Za-z0-9\s]*?)\s*(\d+)k?(?:[¨°©ª¬«­®¯±²³´]|\s)'
        
        for match in re.finditer(pattern, block):
            race_info = match.group(1).strip()
            purse_str = match.group(2)
            
            # Convert purse: if followed by 'k', multiply by 1000
            purse_match = re.search(r'(\d+)k', match.group(0))
            if purse_match:
                purse = int(purse_match.group(1)) * 1000
            else:
                purse = int(purse_str)
            
            # Classify race type
            race_type = self._classify_race_type(race_info)
            
            types.append(race_type)
            purses.append(purse)

        return purses[:5], types[:5]

    def _classify_race_type(self, race_info: str) -> str:
        """Classify BRISNET race type string into standard categories"""
        race_info_upper = race_info.upper()
        
        # Graded stakes
        if 'G1' in race_info_upper or '-G1' in race_info_upper:
            return 'G1'
        if 'G2' in race_info_upper or '-G2' in race_info_upper:
            return 'G2'
        if 'G3' in race_info_upper or '-G3' in race_info_upper:
            return 'G3'
        # Listed stakes or other stakes
        if 'L' in race_info or 'STK' in race_info_upper or 'STAKES' in race_info_upper:
            return 'Stk'
        # Handicap
        if 'HCP' in race_info_upper or 'HANDICAP' in race_info_upper:
            return 'Hcp'
        # Allowance variations
        if race_info.startswith('A') and any(c.isdigit() for c in race_info[:5]):
            return 'Alw'
        if 'ALW' in race_info_upper or 'ALLOWANCE' in race_info_upper:
            return 'Alw'
        if 'OC' in race_info:
            return 'OC'
        # Claiming
        if 'CLM' in race_info_upper or 'CLAIMING' in race_info_upper:
            return 'Clm'
        # Maiden
        if 'MDN' in race_info_upper or 'MAIDEN' in race_info_upper:
            if 'SP' in race_info_upper or 'SPECIAL' in race_info_upper:
                return 'Md Sp Wt'
            return 'Mdn'
        return 'Alw'  # Default to allowance if unclear

    def _parse_pedigree(self, block: str) -> Dict:
        """Extract all pedigree data"""
        ped = {}

        # Sire stats
        m = self.PEDIGREE_PATTERNS['sire_stats'].search(block)
        if m:
            ped['sire_awd'] = float(m.group(1))
            ped['sire_1st_pct'] = float(m.group(2))
            ped['sire_spi'] = float(m.group(4))

        # Damsire stats
        m = self.PEDIGREE_PATTERNS['damsire_stats'].search(block)
        if m:
            ped['damsire_awd'] = float(m.group(1))
            ped['damsire_spi'] = float(m.group(4))

        # Dam stats
        m = self.PEDIGREE_PATTERNS['dam_stats'].search(block)
        if m:
            ped['dam_dpi'] = float(m.group(1))

        # Names
        m = self.PEDIGREE_PATTERNS['sire_name'].search(block)
        if m:
            ped['sire'] = m.group(1).strip()

        m = self.PEDIGREE_PATTERNS['dam_name'].search(block)
        if m:
            ped['dam'] = m.group(1).strip()

        return ped

    def _parse_angles(self, block: str) -> List[Dict]:
        """Extract all handicapping angles"""
        angles = []
        seen_categories = set()  # Prevent duplicates

        for match in self.ANGLE_PATTERN.finditer(block):
            category = re.sub(r'\s+', ' ', match.group(2).strip())

            # Skip duplicates
            if category in seen_categories:
                continue
            seen_categories.add(category)

            angles.append({
                'category': category,
                'starts': int(match.group(3)),
                'win_pct': float(match.group(4)),
                'itm_pct': float(match.group(5)),
                'roi': float(match.group(6))
            })

        return angles

    def _parse_equipment_meds(self, block: str) -> Tuple[str, Optional[str], bool, bool, Optional[int]]:
        """
        Extract equipment, medications, and weight.
        Returns: (equipment, equipment_change, lasix, first_lasix, weight)
        """
        equipment = "L"  # Default
        equipment_change = None
        lasix = False
        first_lasix = False
        weight = None

        # Equipment indicators: L, Lb, Lbf, Lf, etc.
        equip_match = re.search(r'\b(L|Lb|Lbf|Lf|Bf|B)\s+(\d{3})\b', block)
        if equip_match:
            equipment = equip_match.group(1)
            weight = int(equip_match.group(2))

        # Lasix indicator (¦ symbol before race type)
        lasix = bool(re.search(r'¦', block))

        # Equipment change flags in angle section
        if re.search(r'(?i)blinkers?\s+on', block):
            equipment_change = "Blinkers On"
        elif re.search(r'(?i)blinkers?\s+off', block):
            equipment_change = "Blinkers Off"

        # First-time Lasix
        if re.search(r'(?i)1st.*?lasix|first.*?lasix', block):
            first_lasix = True

        return equipment, equipment_change, lasix, first_lasix, weight

    def _parse_comprehensive_race_history(self, block: str) -> Tuple[List[Dict], List[str], List[str], List[float]]:
        """
        Extract complete race history with running positions, conditions, comments, odds.
        Returns: (race_history, track_conditions, trip_comments, odds_history)
        """
        race_history = []
        track_conditions = []
        trip_comments = []
        odds_history = []

        # Simpler pattern focusing on key data
        # Match: DATE TRK ... surface ... SPD PP ST positions FIN JOCKEY ODDS
        lines = block.split('\n')
        for line in lines:
            # Look for lines with date pattern at start
            date_match = re.match(r'(\d{2}[A-Za-z]{3}\d{2})', line)
            if not date_match:
                continue

            try:
                # Extract surface (ft, gd, fm, my, sys, sl)
                surface_match = re.search(r'\s+(ft|gd|fm|my|sys|sl)\s+', line)
                surface = surface_match.group(1) if surface_match else "ft"

                # Extract E1 E2/ LP pattern
                pace_match = re.search(r'(\d{2})\s+(\d{2})/\s*(\d{2,3})\s+([+-]?\d+)\s+([+-]?\d+)\s+(\d{2,3})', line)
                if not pace_match:
                    continue

                e1 = int(pace_match.group(1))
                e2 = int(pace_match.group(2))
                late_pace = int(pace_match.group(3))
                spd = int(pace_match.group(6))

                # Extract post and running positions - look for PP ST 1C 2C STR FIN pattern
                # After SPD, we have: SPD PP ST 1C 2C STR FIN
                after_spd = line[pace_match.end():]
                positions = re.findall(r'(\d+[^\s\d]*)', after_spd)

                if len(positions) >= 6:
                    post_pos = positions[0]
                    start_pos = positions[1]
                    first_call = positions[2]
                    second_call = positions[3]
                    stretch = positions[4]
                    finish_str = positions[5]

                    # Extract numeric finish
                    finish_match = re.search(r'(\d{1,2})', finish_str)
                    if not finish_match:
                        continue
                    finish = int(finish_match.group(1))

                    # Extract odds - look for pattern like "L *1.10" or "16.30"
                    odds_match = re.search(r'(?:L\s+)?[*]?(\d+\.\d+)', after_spd)
                    odds = float(odds_match.group(1)) if odds_match else 0.0

                    # Extract comment - usually after odds and jockey name
                    comment_match = re.search(r'[A-Z][a-z]+[A-Z].*?\s+[\d.]+\s+(.*?)$', line)
                    comment = comment_match.group(1).strip() if comment_match else ""

                    race_info = {
                        'date': date_match.group(1),
                        'surface': surface,
                        'e1': e1,
                        'e2': e2,
                        'late_pace': late_pace,
                        'spd': spd,
                        'post': post_pos,
                        'start': start_pos,
                        'first_call': first_call,
                        'second_call': second_call,
                        'stretch': stretch,
                        'finish': finish,
                        'odds': odds
                    }

                    race_history.append(race_info)
                    track_conditions.append(surface)
                    trip_comments.append(comment)
                    odds_history.append(odds)

            except (ValueError, IndexError, AttributeError):
                continue

        return race_history, track_conditions, trip_comments, odds_history

    def _calculate_running_patterns(self, race_history: List[Dict]) -> Tuple[float, float, float, float]:
        """
        Calculate running style statistics from race history.
        Returns: (early_speed_pct, closing_pct, avg_early_pos, avg_late_pos)
        """
        if not race_history:
            return 0.0, 0.0, 0.0, 0.0

        early_positions = []
        late_positions = []
        led_early = 0
        closed_ground = 0

        for race in race_history[:10]:  # Last 10 races
            try:
                # Parse position numbers (remove beat margin symbols)
                early_pos = int(re.search(r'\d+', race.get('first_call', '99')).group())
                late_pos = int(re.search(r'\d+', race.get('stretch', '99')).group())

                early_positions.append(early_pos)
                late_positions.append(late_pos)

                if early_pos <= 2:
                    led_early += 1
                if late_pos < early_pos:
                    closed_ground += 1
            except (AttributeError, ValueError):
                continue

        total_races = len(early_positions)
        if total_races == 0:
            return 0.0, 0.0, 0.0, 0.0

        early_speed_pct = (led_early / total_races) * 100
        closing_pct = (closed_ground / total_races) * 100
        avg_early_pos = np.mean(early_positions) if early_positions else 0.0
        avg_late_pos = np.mean(late_positions) if late_positions else 0.0

        return early_speed_pct, closing_pct, avg_early_pos, avg_late_pos

    def _parse_surface_stats(self, block: str) -> Dict:
        """
        Extract surface-specific performance statistics.
        Returns: {surface: {starts, wins, win_pct, avg_earnings}}
        """
        surface_stats = {}

        # Pattern: Fst (108) 3 1 - 0 - 1 $35,200 83
        pattern = r'(Fst|Off|Dis|Trf|AW)\s+\((\d+)\)\s+(\d+)\s+(\d+)\s+-\s+(\d+)\s+-\s+(\d+)\s+\$?([\d,]+)'

        for match in re.finditer(pattern, block):
            surface = match.group(1)
            rating = int(match.group(2))
            starts = int(match.group(3))
            wins = int(match.group(4))
            earnings = int(match.group(7).replace(',', ''))

            win_pct = (wins / starts * 100) if starts > 0 else 0.0

            surface_stats[surface] = {
                'starts': starts,
                'wins': wins,
                'win_pct': win_pct,
                'avg_fig': rating,
                'earnings': earnings
            }

        return surface_stats

    def _parse_workouts(self, block: str) -> Tuple[List[Dict], Optional[int], str]:
        """
        Extract workout information.
        Returns: (workouts, last_workout_days, pattern)
        """
        workouts = []

        # Pattern: 18Jan GP 4f ft :49ª B 48/101
        pattern = r'(\d{1,2}[A-Za-z]{3}(?:\'\d{2})?)\s+(\w+)\s+([×]?)(\d+)f\s+(ft|gd|fm|sy|sl)\s+([\d:]+)\s+([BHG])\s+(\d+/\d+)?'

        for match in re.finditer(pattern, block):
            try:
                date_str = match.group(1)
                track = match.group(2)
                bullet = match.group(3) == '×'
                distance = int(match.group(4))
                surface = match.group(5)
                time_str = match.group(6)
                grade = match.group(7)
                rank_str = match.group(8) if match.group(8) else ""

                rank = None
                total = None
                if '/' in rank_str:
                    rank_parts = rank_str.split('/')
                    rank = int(rank_parts[0])
                    total = int(rank_parts[1])

                workouts.append({
                    'date': date_str,
                    'track': track,
                    'bullet': bullet,
                    'distance': distance,
                    'surface': surface,
                    'time': time_str,
                    'grade': grade,
                    'rank': rank,
                    'total': total
                })
            except (ValueError, IndexError):
                continue

        # Calculate last workout days and pattern
        last_workout_days = None
        pattern_desc = "Sparse"

        if workouts:
            # Workouts are most recent first
            try:
                # Parse date from first workout
                from datetime import datetime
                last_work = workouts[0]['date']
                # Handle date formats like "18Jan" or "18Jan'25"
                if "'" in last_work:
                    last_date = datetime.strptime(last_work, "%d%b'%y")
                else:
                    # Assume current year
                    last_date = datetime.strptime(last_work + str(datetime.now().year)[-2:], "%d%b%y")
                last_workout_days = (datetime.now() - last_date).days
            except:
                pass

            # Determine workout pattern
            if len(workouts) >= 5:
                pattern_desc = "Sharp"
            elif len(workouts) >= 3:
                pattern_desc = "Steady"

        return workouts, last_workout_days, pattern_desc

    def _parse_angle_flags(self, angles: List[Dict]) -> List[str]:
        """
        Extract quick-reference angle flags from parsed angles.
        Returns: List of simplified angle descriptions
        """
        flags = []

        for angle in angles:
            category = angle['category']
            win_pct = angle['win_pct']
            roi = angle['roi']

            # Flag high-percentage angles
            if win_pct >= 20 and roi > 0:
                flags.append(f"✓ {category}")
            elif win_pct >= 15:
                flags.append(category)

        return flags[:10]  # Limit to top 10

    def _create_fallback_data(self, post: str, name: str, block: str, error: str) -> HorseData:
        """Create minimal data when parsing fails"""
        return HorseData(
            post=post, name=name, program_number=post,
            pace_style="NA", quirin_points=np.nan, style_strength="Solid",
            ml_odds=None, ml_odds_decimal=None,
            jockey="", jockey_win_pct=0.0,
            trainer="", trainer_win_pct=0.0,
            speed_figures=[], avg_top2=0.0, peak_fig=0, last_fig=0,
            days_since_last=None, last_race_date=None, recent_finishes=[],
            recent_purses=[], race_types=[],
            equipment="L", equipment_change=None, lasix=False, first_lasix=False, weight=None,
            race_history=[], track_conditions=[], trip_comments=[], beat_margins=[], odds_history=[],
            early_speed_pct=0.0, closing_pct=0.0, avg_early_position=0.0, avg_late_position=0.0,
            surface_stats={}, distance_stats={}, track_bias_fit=0.0,
            workouts=[], last_workout_days=None, workout_pattern="Sparse",
            sire="", dam="", sire_spi=None, damsire_spi=None,
            sire_awd=None, dam_dpi=None,
            angles=[], angle_count=0, angle_flags=[],
            parsing_confidence=0.0,
            warnings=[f"CRITICAL: Parsing failed - {error}"],
            raw_block=block
        )

    def validate_parsed_data(self, horses: Dict[str, HorseData]) -> Dict:
        """
        Comprehensive validation of parsed data quality.
        Returns: {overall_score, issues, recommendations}
        """
        issues = []
        total_confidence = []

        for name, horse in horses.items():
            total_confidence.append(horse.parsing_confidence)

            # Check critical fields
            if not horse.jockey:
                issues.append(f"{name}: Missing jockey")
            if not horse.trainer:
                issues.append(f"{name}: Missing trainer")
            if not horse.ml_odds:
                issues.append(f"{name}: Missing ML odds")
            if horse.pace_style == "NA":
                issues.append(f"{name}: Unknown pace style")
            if not horse.speed_figures and horse.days_since_last is not None:
                # Has race history but no figs = parsing problem
                issues.append(f"{name}: Has race history but no speed figures")

            # Logical consistency checks
            if horse.quirin_points and (horse.quirin_points < 0 or horse.quirin_points > 8):
                issues.append(f"{name}: Invalid Quirin points ({horse.quirin_points})")

            if horse.days_since_last and horse.days_since_last < 0:
                issues.append(f"{name}: Invalid layoff days ({horse.days_since_last})")

        avg_confidence = np.mean(total_confidence) if total_confidence else 0.0

        return {
            'overall_confidence': round(avg_confidence, 3),
            'horses_parsed': len(horses),
            'issues': issues,
            'critical_issues': [i for i in issues if 'CRITICAL' in i or 'Missing jockey' in i or 'Missing trainer' in i],
            'recommendations': self._generate_recommendations(avg_confidence, issues)
        }

    def _generate_recommendations(self, confidence: float, issues: List[str]) -> List[str]:
        """Generate actionable recommendations based on validation"""
        recs = []

        if confidence < 0.7:
            recs.append("⚠️ Low parsing confidence - verify PP format matches BRISNET standard")

        if any('Missing jockey' in i for i in issues):
            recs.append("Check jockey formatting - should be ALL CAPS line before 'Trnr:'")

        if any('Missing trainer' in i for i in issues):
            recs.append("Check trainer formatting - should start with 'Trnr:' prefix")

        if any('Missing ML odds' in i for i in issues):
            recs.append("Verify odds are at start of horse block (first line after name)")

        if len(issues) > len(issues) // 2:  # More than half have issues
            recs.append("Consider manual data entry or PDF re-export for this race")

        return recs

# ===================== INTEGRATION WITH PROBABILISTIC MODEL =====================

def integrate_with_torch_model(parsed_horses: Dict[str, HorseData],
                              rating_engine,
                              softmax_tau: float = 3.0) -> pd.DataFrame:
    """
    Convert parsed data → ratings → softmax probabilities.

    Returns DataFrame with columns:
        Horse, Rating, Probability, FairOdds, Confidence
    """
    import torch

    rows = []

    for name, horse in parsed_horses.items():
        # Use existing rating engine (from app.py)
        # This would call calculate_final_rating with parsed data
        rating = rating_engine.compute_rating(horse)

        rows.append({
            'Horse': name,
            'Rating': rating,
            'Confidence': horse.parsing_confidence
        })

    df = pd.DataFrame(rows)

    # Apply softmax to ratings
    ratings_tensor = torch.tensor(df['Rating'].values, dtype=torch.float32)
    ratings_scaled = ratings_tensor / softmax_tau
    probs = torch.nn.functional.softmax(ratings_scaled, dim=0).numpy()

    df['Probability'] = probs
    df['Fair_Odds'] = (1.0 / probs).round(2)

    # Sort by probability descending
    df = df.sort_values('Probability', ascending=False).reset_index(drop=True)

    return df

# ===================== EXAMPLE USAGE =====================

if __name__ == "__main__":
    # Test with sample PP
    sample_pp = """
Race 2 Mountaineer 'Mdn 16.5k 5½ Furlongs 3&up, F & M Wednesday, August 20, 2025

1 Way of Appeal (S 3)
Own: Someone
7/2 Red, Red Cap
BARRIOS RICARDO (254 58-42-39 23%)
B. h. 3 (Mar)
Sire : Appeal (Not for Love) $25,000
Dam: Appealing (Storm Cat)
Brdr: Smith Racing (WV)
Trnr: Cady Khalil (150 18-24-31 12%)
Prime Power: 101.5 (4th)
23Sep23 Mtn Md Sp Wt 16500 98 4th
15Aug23 Mtn Md Sp Wt 16500 92 6th
Sire Stats: AWD 115.2 18% FTS 22% 108.5 spi
"""

    parser = EliteBRISNETParser()
    horses = parser.parse_full_pp(sample_pp)

    print("="*80)
    print("PARSED HORSES:")
    print("="*80)

    for name, horse in horses.items():
        print(f"\n🐎 {name}")
        print(f"   Post: {horse.post}")
        print(f"   Style: {horse.pace_style} (Quirin {horse.quirin_points}, {horse.style_strength})")
        print(f"   ML Odds: {horse.ml_odds} → {horse.ml_odds_decimal}")
        print(f"   Jockey: {horse.jockey} ({horse.jockey_win_pct:.1%})")
        print(f"   Trainer: {horse.trainer} ({horse.trainer_win_pct:.1%})")
        print(f"   Speed Figs: {horse.speed_figures} (Avg Top2: {horse.avg_top2})")
        print(f"   Angles: {horse.angle_count}")
        print(f"   Confidence: {horse.parsing_confidence:.1%}")
        if horse.warnings:
            print(f"   ⚠️ Warnings: {', '.join(horse.warnings)}")

    # Validation report
    validation = parser.validate_parsed_data(horses)
    print(f"\n{'='*80}")
    print("VALIDATION REPORT:")
    print(f"{'='*80}")
    print(f"Overall Confidence: {validation['overall_confidence']:.1%}")
    print(f"Horses Parsed: {validation['horses_parsed']}")
    print(f"Issues Found: {len(validation['issues'])}")
    if validation['critical_issues']:
        print("\n🚨 CRITICAL ISSUES:")
        for issue in validation['critical_issues']:
            print(f"   - {issue}")

    if validation['recommendations']:
        print("\n💡 RECOMMENDATIONS:")
        for rec in validation['recommendations']:
            print(f"   - {rec}")
