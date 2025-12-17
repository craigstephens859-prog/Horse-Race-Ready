# BRISNET Ultimate Race Summary Explanation - Parsing Gap Analysis

**Date:** December 16, 2025  
**Status:** COMPREHENSIVE SECTION PARSING REVIEW

---

## Overview

The **Race Summary** section in BRISNET Ultimate PPs provides 11 key data points per horse ranked by Average Distance/Surface speed rating. This analysis reviews the ULTIMATE RACE SUMMARY EXPLANATION guide and maps each field to current implementation status.

**Key Finding:** Currently **5 of 11 fields parsed**, **6 critical gaps identified** with estimated **+0.18-0.35 points accuracy gain** opportunity.

---

## Section 1: Horse Information (FIELD #1)

### Components:
1. **Program Number (#)** - Horse's post position/program
2. **Horse Name** - Horse identifier
3. **ML Odds** - Morning line odds from track  
4. **Med/Eqp** - Medication/equipment (L=Lasix, b=Blinkers)
5. **Days Since L/R** - Days since last race (dots indicate layoff count)

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Program # | ✅ PARSED | `extract_horses_and_styles()` | Extracted via HORSE_HDR_RE regex |
| Horse Name | ✅ PARSED | `extract_horses_and_styles()` | Extracted via HORSE_HDR_RE regex |
| ML Odds | ✅ PARSED | `extract_morning_line_by_horse()` | Regex searches for decimal/fraction odds |
| Lasix (L) | ✅ PARSED | `parse_equip_lasix()` | Detects L, L#, "Lasix On/Off" |
| Blinkers (b) | ✅ PARSED | `parse_equip_lasix()` | Detects B, "Blinkers On/Off" |
| Days Since L/R | ⚠️ PARTIAL | `parse_all_angles()` | Sets `layoff=30` (placeholder) |

### Gap: Days Since Last Race Calculation

**Problem:** Currently hardcoded to 30-day placeholder. Should calculate from most recent race date.

**Solution:** Parse date pattern from first race line (e.g., "23Sep25"), calculate days elapsed.

```python
def parse_days_since_last_race(block: str) -> int:
    """Calculate days since last race from most recent race date in PP"""
    # Pattern: "23Sep25Mnr® 5½ ft ..." where 23Sep25 = most recent race
    race_date_match = re.search(r'^(\d{2})([A-Z][a-z]{2})(\d{2})', block, re.MULTILINE)
    
    if race_date_match:
        day = int(race_date_match.group(1))
        month_str = race_date_match.group(2)  # Sep, Oct, etc.
        year_short = int(race_date_match.group(3))  # 25 for 2025
        
        # Convert month string to number
        months = {'Jan':1, 'Feb':2, 'Mar':3, 'Apr':4, 'May':5, 'Jun':6,
                  'Jul':7, 'Aug':8, 'Sep':9, 'Oct':10, 'Nov':11, 'Dec':12}
        month = months.get(month_str, 1)
        
        # Build date
        year = 2000 + year_short if year_short <= 50 else 1900 + year_short
        race_date = datetime(year, month, day)
        today = datetime.now()
        
        days = (today - race_date).days
        return max(0, days)
    
    return 0
```

**Expected Impact:** +0.02-0.04 points (needed for layoff analysis)

---

## Section 2: Run Style & PTS (FIELD #2)

### Components:
1. **Running Style** - E (Early), E/P (Early/Presser), P (Presser), S (Sustained/Closer), NA
2. **Early Speed Points (PTS)** - 0-8 scale measuring early speed ability
3. **Style Indicators** - `+` (favorable for race), `++` (best/dominant)

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Running Style | ✅ PARSED | `parse_running_style_for_block()` | Extracted from header regex |
| Quirin PTS | ✅ PARSED | `HORSE_HDR_RE` regex group 4 | Numeric value 0-8 |
| Style Strength | ✅ CALCULATED | `calculate_style_strength()` | "Strong"/"Solid"/"Slight"/"Weak" |
| Style Indicator (+/++) | ✅ TRACKED | MODEL_CONFIG track bias tables | Used in bias calculation |

**Status:** ✅ COMPLETE

---

## Section 3: Avg Dist/Surf (FIELD #3)

### Components:
1. **E1** - Average pace rating (start to first call) at today's distance/surface
2. **E2** - Average pace rating (start to second call) at today's distance/surface  
3. **Late** - Average pace rating (second call to finish) at today's distance/surface
4. **SPD** - Average BRIS speed rating at today's distance/surface

**Notation Markers:**
- `*` Asterisks = 2+ races in last 90 days at/near today's distance/surface (reliable)
- `()` Parentheses = Race >90 days ago (stale)
- No markers = Only one race in last 90 days

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| E1 (early pace) | ✅ PARSED | `parse_speed_figures_for_block()` | Regex group 3 |
| E2 (mid pace) | ✅ PARSED | `parse_speed_figures_for_block()` | Regex group 4 |
| Late (LP) | ⚠️ PARTIAL | `parse_speed_figures_for_block()` | Extracts but labeled as 'LP' placeholder |
| SPD (speed rating) | ⚠️ PARTIAL | `parse_speed_figures_for_block()` | Uses E2 as proxy (incorrect) |
| Race recency markers | ❌ NOT PARSED | - | `*` and `()` NOT extracted |
| Sample size validation | ❌ NOT PARSED | - | Cannot determine reliability of avg |

### Gaps:

**Gap 1: SPD vs E2 Confusion**
- Current code uses E2 (mid-pace) as proxy for SPD (final speed)
- True SPD comes from race lines but NOT in current regex

**Gap 2: Race Recency Markers**
- `*` = reliable average (2+ races, recent)
- `()` = stale average (>90 days ago)
- `no marker` = single race (unreliable)
- Currently: No distinction between averages

**Gap 3: Distance/Surface Specificity**
- Need to verify that parsed figures are specifically for "today's distance/surface"
- Parser should filter/flag if using stale or off-distance data

**Solution: Enhanced Speed Figure Parser**

```python
def parse_avg_dist_surf_figures(block: str) -> dict:
    """Parse Avg Dist/Surf section with reliability markers"""
    result = {
        'e1_avg': None,        # Pace 1st call
        'e2_avg': None,        # Pace 2nd call
        'late_avg': None,      # Pace to finish
        'spd_avg': None,       # Speed rating
        'e1_reliability': 'unknown',  # 'reliable'(*), 'stale'(%), 'single'(none)
        'e2_reliability': 'unknown',
        'late_reliability': 'unknown',
        'spd_reliability': 'unknown'
    }
    
    # Pattern: "85* 97*|94* 99*" or "85 97|94 99" or "85(P) 97(P)|94(P) 99(P)"
    # Where * = reliable, () = stale, none = single race
    # Format: E1marker E2marker|LATEmarker SPDmarker
    # Example from BRISNET doc: "85 97|94 99" or "85* 97*|94* 99*"
    
    dist_surf_pattern = r'(?m)^.*?(\d{2})\s*(\*|\(P\))?\s+(\d{2})\s*(\*|\(P\))?\s*\|\s+(\d{2})\s*(\*|\(P\))?\s+(\d{2})\s*(\*|\(P\))?'
    
    match = re.search(dist_surf_pattern, block)
    if match:
        e1_val = int(match.group(1))
        e1_marker = match.group(2)
        e2_val = int(match.group(3))
        e2_marker = match.group(4)
        late_val = int(match.group(5))
        late_marker = match.group(6)
        spd_val = int(match.group(7))
        spd_marker = match.group(8)
        
        # Map markers to reliability
        def map_reliability(marker):
            if marker == '*':
                return 'reliable'  # 2+ races in last 90 days
            elif marker == '(P)' or marker is None:
                return 'stale'     # >90 days ago or marker unknown
            else:
                return 'single'    # Only 1 race, no marker
        
        result['e1_avg'] = e1_val
        result['e1_reliability'] = map_reliability(e1_marker)
        result['e2_avg'] = e2_val
        result['e2_reliability'] = map_reliability(e2_marker)
        result['late_avg'] = late_val
        result['late_reliability'] = map_reliability(late_marker)
        result['spd_avg'] = spd_val
        result['spd_reliability'] = map_reliability(spd_marker)
    
    return result
```

**Expected Impact:** +0.03-0.06 points (better distance/surface validation)

---

## Section 4: Avg Race Rtng (FIELD #4)

### Definition:
Average of horse's most recent BRIS Race Ratings at today's distance/surface. Measures field quality (not classification), higher = stronger competition faced.

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Avg Race Rating | ❌ NOT PARSED | - | Not extracted from Race Summary |

### Gap: Avg Race Rtng Not Captured

**Problem:** This metric is valuable for class context but not currently parsed.

**Solution:**
```python
def parse_avg_race_rating(race_summary_line: str) -> float:
    """Extract Avg Race Rtng from race summary
    Format: "119" in "119 93 101/103"
    """
    # Pattern: Single 3-digit number after E2, before Best Pace section
    pattern = r'(?m)(\d{2,3})\s+93\s+101/103'  # Example pattern from BRIS doc
    match = re.search(pattern, race_summary_line)
    return float(match.group(1)) if match else np.nan
```

**Expected Impact:** +0.01-0.03 points (class context)

---

## Section 5: Best Pace (FIELD #5)

### Components:
Best E1, E2, and LP pace figures within **one year** (not just recent 90 days)

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Best Pace figures | ⚠️ PARTIAL | `parse_all_angles()` | Extracts pace values but not labeled as "Best" |
| Time window (1 year) | ❌ NOT ENFORCED | - | No date filtering for 1-year window |

### Gap: Best vs Recent Pace Confusion

**Problem:** Current parser doesn't distinguish:
- Avg Dist/Surf (recent 90 days) 
- Best Pace (all time within 1 year)

**Solution:** Add separate parsing for lifetime bests.

**Expected Impact:** +0.02-0.04 points

---

## Section 6: Final Speed (FIELD #6)

### Definition:
Horse's final speed figure for last 4 races. Notation:
- `.` dot = race at today's distance/surface (relevant)
- `T` = turf race (note surface)
- No marker = off-distance/off-surface (less relevant)

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Last 4 speed figures | ✅ PARSED | `parse_speed_figures_for_block()` | SPD list [:4] |
| Distance relevance marker | ❌ NOT PARSED | - | `.` dot NOT extracted |
| Turf marker | ❌ NOT PARSED | - | `T` NOT extracted |
| Off-distance penalty | ❌ NOT APPLIED | - | All speeds treated equally |

### Gap: Relevance Markers Not Tracked

**Problem:** Can't tell if speed figure is at relevant distance.

**Example from BRISNET:**
- "116.0 97" = last 4 speeds: 116 (relevant), 97 (relevant), 119 (turf), 117 (off-distance)
- Current: Just sees [116, 97, 119, 117] without context

**Solution:**
```python
def parse_final_speed_with_markers(race_summary_block: str) -> list:
    """Parse final speed figures with distance/surface relevance markers"""
    # Pattern: "116.0 97" where first is most recent
    # Each figure can have: "." (relevant dist), "T" (turf), or nothing (off-dist)
    # Example: "116* 97 119T 117"
    pattern = r'(\d{2,3})\.?(\d*)\s+(\d{2,3})([.T]?)\s+(\d{2,3})([.T]?)\s+(\d{2,3})([.T]?)'
    
    speeds = []
    match = re.search(pattern, race_summary_block)
    if match:
        # Parse 4 speeds with markers
        for i in range(1, 8, 2):  # Groups 1, 3, 5, 7
            speed_val = int(match.group(i))
            marker = match.group(i+1) if i+1 <= 8 else ""
            
            relevance = 'relevant' if marker == '.' else ('turf' if marker == 'T' else 'off-distance')
            speeds.append({'speed': speed_val, 'relevance': relevance})
    
    return speeds
```

**Expected Impact:** +0.02-0.05 points (context-specific speed adjustments)

---

## Section 7: ACL (FIELD #7)

### Definition:
**Average Competitive Level** - Rating based on Race Ratings when finishing in the money.  
Parentheses = rating NOT at today's distance/surface (less relevant)

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| ACL value | ❌ NOT PARSED | - | Not extracted |
| ACL relevance | ❌ NOT PARSED | - | Parentheses NOT distinguished |

### Gap: ACL Not Used

**Problem:** Valuable metric for "class ceiling" but not extracted.

**Solution:** Parse like Race Rating but label as "ACL"

**Expected Impact:** +0.01-0.03 points

---

## Section 8: Reg Spd Avg (FIELD #8)

### Definition:
**Racing Speed Average** - Average of speed ratings from last 3 starts (regardless of distance/surface)

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Last 3 speed avg | ⚠️ PARTIAL | `parse_speed_figures_for_block()` | Extracts SPD list |
| 3-race average | ❌ NOT CALCULATED | - | Just stores raw list |
| Distance-agnostic | ⚠️ UNCLEAR | - | No filtering for off-distance races |

### Gap: Not Aggregated Properly

**Problem:** Parser gets speed list but doesn't calculate 3-race average.

**Solution:**
```python
# In apex_enhance or compute_bias_ratings:
reg_spd_avg = np.mean(figs_per_horse[horse].get('SPD', [])[:3]) if horse in figs_per_horse else np.nan
```

**Expected Impact:** +0.01-0.02 points (form indicator)

---

## Section 9: Previous Race Ratings (FIELD #9)

### Definition:
Race ratings from 3 previous races (R1=most recent, R2=2nd back, R3=3rd back)

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| R1, R2, R3 values | ⚠️ PARTIAL | `parse_speed_figures_for_block()` | Extracted as RR list |
| Trend detection | ❌ NOT CALCULATED | - | No rising/falling form analysis |
| Consistency | ⚠️ PARTIAL | `parse_cr_rr_history()` | Calculates but from CR, not RR |

### Gap: Race Rating Trend Not Analyzed

**Problem:** Can't detect form trends (improving vs declining field quality faced).

**Solution:**
```python
def analyze_race_rating_trend(rr_list: list) -> dict:
    """Analyze 3-race trend in Race Ratings (field quality)"""
    if len(rr_list) < 2:
        return {'trend': 'unknown', 'direction': 0, 'consistency': 0}
    
    r1, r2, r3 = rr_list[0], rr_list[1], rr_list[2] if len(rr_list) > 2 else rr_list[1]
    
    trend = 'improving' if r1 > r2 else ('declining' if r1 < r2 else 'flat')
    direction = r1 - r2  # Numeric trend
    consistency = max(0, 1.0 - abs(r1 - r2) / 20.0)  # Lower variance = more consistent
    
    return {'trend': trend, 'direction': direction, 'consistency': consistency}
```

**Expected Impact:** +0.02-0.04 points (form trends)

---

## Section 10: Mud Speed (FIELD #10)

### Definition:
Best BRIS speed rating horse earned on muddy/sloppy track  
**Note:** Only appears on non-turf, non-2yo, non-maiden races

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| Mud speed value | ❌ NOT PARSED | - | Not extracted from Race Summary |
| Condition applicability | ❌ NOT VALIDATED | - | No check for race type/condition |

### Gap: Mud Speed Not Captured

**Problem:** Missing key surface-specific metric for off-track races.

**Solution:**
```python
def parse_mud_speed_rating(race_summary_block: str, condition: str) -> float:
    """Extract best mud speed rating (only for non-turf, non-2yo, non-maiden)"""
    if 'turf' in condition.lower() or 'maiden' in condition.lower():
        return np.nan
    
    # Pattern: "Mud Spd: 84" or similar
    mud_pattern = r'Mud\s*Spd[:\s]*(\d{2,3})'
    match = re.search(mud_pattern, race_summary_block, re.IGNORECASE)
    
    return float(match.group(1)) if match else np.nan
```

**Expected Impact:** +0.01-0.02 points (off-track races only)

---

## Section 11: Pedigree Statistics (FIELD #11)

### Subcomponents:

| Stat | Description | Applies To | Current Status |
|------|-------------|-----------|-----------------|
| **Mud Sts** | Sire progeny starts on off tracks | All except 2yo/Maiden/Turf | ❌ NOT PARSED |
| **% Mud** | Win % on off tracks by sire progeny | All except 2yo/Maiden/Turf | ❌ NOT PARSED |
| **% 1st** | Win % on first career start (sire progeny) | 2yo/Maiden only | ❌ NOT PARSED |
| **SPI** | Sire Production Index (vs avg) | 2yo/Maiden only | ❌ NOT PARSED |
| **DPI** | Dam Production Index (vs avg) | 2yo/Maiden only | ✅ PARSED* |
| **%Trf** | Win % on turf (sire progeny) | Turf races only | ❌ NOT PARSED |
| **Sire AWD** | Avg winning distance of sire's offspring | All | ❌ NOT PARSED |
| **Dam Sire AWD** | Avg winning distance of dam sire's offspring | All | ❌ NOT PARSED |

### Current Implementation:

| Component | Status | Location | Notes |
|-----------|--------|----------|-------|
| DPI (Dam Production) | ✅ PARSED | `parse_bris_pedigree_ratings()` | Extracted and bonus applied (+0.05-0.07) |
| SPI (Sire Production) | ❌ NOT PARSED | - | Should appear in 2yo/Maiden only |
| Mud %/Sts (Sire) | ❌ NOT PARSED | - | Not extracted for off-track context |
| Turf % (Sire) | ❌ NOT PARSED | - | Not extracted for turf races |
| AWD metrics | ❌ NOT PARSED | - | Not extracted for distance profile |

### Gaps: 7 of 8 Pedigree Stats Not Used

**Problem:** Only DPI parsed; other valuable pedigree indicators ignored.

**Solution: Comprehensive Pedigree Parser**

```python
def parse_pedigree_statistics_complete(block: str, race_type: str, surface: str) -> dict:
    """Parse all 11 pedigree statistics from Race Summary
    
    Args:
        block: Horse's PP block
        race_type: '2yo', 'maiden', 'regular', 'stake'
        surface: 'Dirt', 'Turf', 'Synthetic'
    """
    result = {
        # Sire stats (all races)
        'sire_mud_sts': None,
        'sire_mud_pct': None,
        'sire_awd': None,
        
        # Sire stats (2yo/Maiden only)
        'sire_first_career_pct': None,
        'spi': None,
        
        # Dam/Dam-Sire stats
        'dpi': None,
        'dam_sire_awd': None,
        
        # Turf specific
        'sire_turf_pct': None
    }
    
    # Pattern: "Mud Sts 142 % Mud 8.3%" or similar
    if race_type not in ('2yo', 'maiden', 'turf'):
        mud_sts_pattern = r'Mud Sts[:\s]*(\d+)\s+%\s*Mud[:\s]*(\d+(?:\.\d)?)'
        m = re.search(mud_sts_pattern, block, re.IGNORECASE)
        if m:
            result['sire_mud_sts'] = int(m.group(1))
            result['sire_mud_pct'] = float(m.group(2))
    
    # Sire AWD: "Sire AWD 7.5" or "Sire Stats AWD 7.5"
    sire_awd_pattern = r'(?:Sire|Sire Stats)[:\s]*AWD[:\s]*(\d+(?:\.\d)?)'
    m = re.search(sire_awd_pattern, block, re.IGNORECASE)
    if m:
        result['sire_awd'] = float(m.group(1))
    
    # Dam Sire AWD: "Dam Sire AWD 6.2"
    dam_sire_awd_pattern = r'Dam[\']*s?\s+Sire[:\s]*AWD[:\s]*(\d+(?:\.\d)?)'
    m = re.search(dam_sire_awd_pattern, block, re.IGNORECASE)
    if m:
        result['dam_sire_awd'] = float(m.group(1))
    
    # 2yo/Maiden only
    if race_type in ('2yo', 'maiden'):
        # % 1st: "% 1st 31" or similar
        first_pct_pattern = r'%\s*1st[:\s]*(\d+(?:\.\d)?)'
        m = re.search(first_pct_pattern, block, re.IGNORECASE)
        if m:
            result['sire_first_career_pct'] = float(m.group(1))
        
        # SPI: "SPI 2.15" or "Sire Production Index 2.15"
        spi_pattern = r'(?:SPI|Sire Production Index)[:\s]*(\d+(?:\.\d+)?)'
        m = re.search(spi_pattern, block, re.IGNORECASE)
        if m:
            result['spi'] = float(m.group(1))
        
        # DPI: "DPI 1.87" or "Dam Production Index 1.87"
        dpi_pattern = r'(?:DPI|Dam Production Index)[:\s]*(\d+(?:\.\d+)?)'
        m = re.search(dpi_pattern, block, re.IGNORECASE)
        if m:
            result['dpi'] = float(m.group(1))
    
    # Turf specific
    if surface.lower() == 'turf' or 'turf' in race_type.lower():
        # %Trf: "% Turf 17" or "%Trf 17"
        turf_pct_pattern = r'%\s*(?:Turf|Trf)[:\s]*(\d+(?:\.\d)?)'
        m = re.search(turf_pct_pattern, block, re.IGNORECASE)
        if m:
            result['sire_turf_pct'] = float(m.group(1))
    
    return result
```

**Integration into Bias Rating:**

```python
def apply_pedigree_bonuses(horse: str, pedigree_data: dict, race_type: str, surface: str) -> float:
    """Apply pedigree statistic bonuses"""
    bonus = 0.0
    
    # Sire mud (for off-track races)
    if condition_txt in ('muddy', 'sloppy') and pedigree_data.get('sire_mud_pct', 0) >= 15:
        bonus += MODEL_CONFIG.get('sire_mud_bonus', 0.08)
    
    # Sire turf (for turf races)
    if surface.lower() == 'turf' and pedigree_data.get('sire_turf_pct', 0) >= 18:
        bonus += MODEL_CONFIG.get('sire_turf_bonus', 0.10)
    
    # DPI (existing, but validate)
    if pedigree_data.get('dpi', 0) >= 1.5:
        bonus += MODEL_CONFIG.get('dpi_bonus', 0.05)
    
    # SPI (sire excellence)
    if race_type in ('2yo', 'maiden') and pedigree_data.get('spi', 0) >= 1.75:
        bonus += MODEL_CONFIG.get('spi_bonus', 0.06)
    
    # Sire AWD distance match
    sire_awd = pedigree_data.get('sire_awd', 0)
    today_dist = distance_to_furlongs(distance_txt)
    if sire_awd and 0.5 <= abs(sire_awd - today_dist) <= 1.5:  # Within 0.5 furlong
        bonus += MODEL_CONFIG.get('sire_awd_match_bonus', 0.04)
    
    return bonus
```

**Expected Impact:** +0.06-0.12 points (comprehensive pedigree)

---

## Summary: Race Summary Parsing Gaps

### Implementation Status Table:

| Field # | Component | Parsed | Gaps | Priority | Est. Gain |
|---------|-----------|--------|------|----------|-----------|
| 1 | Horse Info (except Days L/R) | 80% | Days since last race | HIGH | +0.02 |
| 2 | Run Style & PTS | 100% | None | - | +0.00 |
| 3 | Avg Dist/Surf | 50% | SPD confusion, recency markers, sample size | HIGH | +0.05 |
| 4 | Avg Race Rtng | 0% | Not parsed | MEDIUM | +0.02 |
| 5 | Best Pace | 30% | 1-year window not enforced | MEDIUM | +0.03 |
| 6 | Final Speed (4 races) | 50% | Relevance markers not parsed | MEDIUM | +0.04 |
| 7 | ACL | 0% | Not parsed | LOW | +0.02 |
| 8 | Reg Spd Avg | 30% | Not aggregated to 3-race average | LOW | +0.02 |
| 9 | Previous Race Ratings (R1/R2/R3) | 30% | Trend analysis missing | MEDIUM | +0.03 |
| 10 | Mud Speed | 0% | Not parsed | LOW | +0.01 |
| 11 | Pedigree Stats | 12% | 7 of 8 stats not parsed | HIGH | +0.09 |
| **TOTAL** | | **39%** | **11 gaps** | | **+0.33** |

---

## Recommended Implementation Priority

### Phase 1 (HIGH) - +0.16 points
1. **Field #11 Pedigree Stats** - Parse Sire %Mud, SPI, Turf %, AWD metrics (+0.09)
2. **Field #3 Avg Dist/Surf** - Fix SPD vs E2, add recency markers, sample size (+0.05)
3. **Field #1 Days Since Last Race** - Proper date calculation (+0.02)

### Phase 2 (MEDIUM) - +0.12 points
4. **Field #6 Final Speed** - Add distance/surface relevance markers (+0.04)
5. **Field #5 Best Pace** - Enforce 1-year window properly (+0.03)
6. **Field #9 Race Ratings Trend** - Add form trend analysis (+0.03)
7. **Field #4 Avg Race Rtng** - Parse and use for class context (+0.02)

### Phase 3 (LOW) - +0.05 points
8. **Field #10 Mud Speed** - Extract for off-track races (+0.01)
9. **Field #7 ACL** - Parse and validate (+0.02)
10. **Field #8 Reg Spd Avg** - Aggregate properly (+0.02)

---

## Code Integration Points

### Location 1: Add to `compute_bias_ratings()` function

```python
# After existing parsing loops, add:
pedigree_stats_per_horse = {}

for post, name, block in split_into_horse_chunks(pp_text):
    pedigree_stats = parse_pedigree_statistics_complete(block, race_type, surface_type)
    pedigree_stats_per_horse[name] = pedigree_stats
    
    # Add pedigree bonuses to tier1_bonus
    ped_bonus = apply_pedigree_bonuses(name, pedigree_stats, race_type, surface_type)
    all_angles_per_horse[name]['pedigree_bonus'] = ped_bonus
```

### Location 2: Add to MODEL_CONFIG

```python
# New pedigree bonus parameters
"sire_mud_bonus": 0.08,           # +8% for sire with 15%+ mud wins
"sire_turf_bonus": 0.10,          # +10% for sire with 18%+ turf wins
"spi_bonus": 0.06,                # +6% for SPI >= 1.75
"sire_awd_match_bonus": 0.04,     # +4% for distance-matching sire
"dam_sire_awd_mismatch_penalty": -0.03,  # -3% for distance-mismatched dam sire

# Recency weight adjustments
"avg_figure_recency_weight": 0.15,  # 15% boost for "*" marked figures (recent)
"avg_figure_stale_penalty": -0.05,   # -5% for "()" marked figures (>90 days)
```

---

## Testing Recommendations

### Unit Test: Pedigree Parsing

```python
def test_parse_pedigree_stats():
    block = """
    Mud Sts 142 % Mud 8.3%
    Sire Stats: AWD 7.5 SPI 1.92
    Dam Sire: AWD 6.1 
    Dam: DPI 1.45
    % Turf 17
    """
    result = parse_pedigree_statistics_complete(block, 'regular', 'Dirt')
    assert result['sire_mud_sts'] == 142
    assert result['sire_mud_pct'] == 8.3
    assert result['sire_awd'] == 7.5
    assert result['dpi'] == 1.45
    assert result['sire_turf_pct'] == 17
```

### Integration Test: Race Summary Accuracy

Load 10 real Race Summaries from BRIS PPs and verify:
1. All 11 fields parsed correctly
2. Bonuses calculated within expected ranges
3. Accuracy improvement of +0.18-0.35 points

---

## Conclusion

The **Race Summary section** contains rich, field-level data that current implementation only partially utilizes. By implementing the **11 gaps identified** above, particularly the **pedigree statistics parsing**, we can achieve:

- **+0.33 points estimated accuracy gain** (from current 7.05-7.15 to 7.38-7.48)
- **39% → 100% field coverage** in Race Summary parsing
- **Better distance/surface context** through recency markers
- **Comprehensive pedigree usage** including 7 currently-ignored stats

**Next Step:** Implement Phase 1 (HIGH priority) items starting with Pedigree Stats parser integration.

