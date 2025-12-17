# BRISNET Track Bias Part 3 - Detailed Analysis & Implementation Guide

## Format Discovery from Real Example

**Source:** Del Mar OC $50,000 1-Mile Turf, November 1, 2025, Race 1

### Key Findings from Example Data

The track bias section contains **TWO separate datasets:**
1. **MEET Totals** - Cumulative stats for the meet period (07/18 - 09/07)
2. **WEEK Totals** - Recent stats for the week (09/01 - 09/07)

Both datasets provide the same structure:
- Race count and date range
- Wire-to-wire percentage
- Speed bias percentage
- Winner's average beaten lengths (1stCall, 2ndCall)
- Running style Impact Values and win percentages
- Post position Impact Values and win percentages

---

## Real Data Example Analysis

### MEET TOTALS (Del Mar Turf 8.5f, 20 races)

**Running Styles (Early Speed vs Late Speed):**
```
E (Early):       Impact 1.77 (++)  ← DOMINANT front-runner style
E/P (Presser):   Impact 0.28       ← WEAK (less than 1/4 expected wins)
P (Middle pack):  Impact 1.35 (+)   ← FAVORABLE
S (Closer):      Impact 0.27       ← VERY WEAK (only 27% expected wins)
```

**Interpretation:** 
- Early runners win 35% of races (outperforming due to 1.77 impact)
- E runners are HUGELY dominant at this distance/track
- Closers are heavily disadvantaged (impact 0.27 = only 27% of expected wins)
- This is a FRONT-RUNNER BIASED track

**Post Positions:**
```
RAIL:   Impact 0.88  → Slight disadvantage
1-3:    Impact 0.88  → Slight disadvantage  
4-7:    Impact 1.01  → Neutral
8+:     Impact 1.17  (+) → FAVORABLE (outside posts advantage)
```

**Interpretation:** Counter-intuitive! Outside posts (8+) are BEST at Del Mar turf 8.5f
- Rail and inner posts show consistent 0.88 impact (10% each win rate)
- Outside posts show 1.17 impact (12% win rate)
- Suggests wider course allows passing room for outside posts

---

### WEEK TOTALS (Only 1 race in sample week!)

```
E (Early):  Impact 3.20 (++) → Won the 1 race that week
P, S, E/P:  Impact 0.00    → Didn't win
```

**Critical Issue:** Sample size of 1 race is statistically meaningless! 

**Implementation Decision:** MEET TOTALS are far more reliable (20 races vs 1 race)

---

## 11 Components - Detailed Extraction Strategy

### 1. **Surface, Distance & Race Count** ✅ 
**Extraction:**
- Line: "Turf 8.5f" 
- Race count: "# Races: 20"
- Date range: "07/18 - 09/07"

**Confidence Weighting:**
```python
if races >= 50:
    confidence = 1.0
elif races >= 20:
    confidence = 0.8  # Like this example
elif races >= 10:
    confidence = 0.6
else:
    confidence = 0.4  # Very low - use sparingly
```

---

### 2. **% Wire (Wire-to-Wire Winners)** ✅
**Extraction:** 
- Line: "%Wire: 25%"

**Use Case:** At Del Mar Turf 8.5f, 25% of winners led wire-to-wire

**Bonus Logic:**
```python
if wire_pct > 30:  # High wire-to-wire percentage
    # Front-runners winning from the start
    bonus_for_early_styles = +0.06
    penalty_for_closers = -0.08
    
elif wire_pct < 10:  # Low wire-to-wire percentage  
    # Most winners don't lead from start (closers can win)
    bonus_for_closers = +0.04
    penalty_for_early = -0.04
```

**This Example (25%):** Moderate front-runner advantage

---

### 3. **Speed Bias** ✅
**Extraction:**
- Line: "Speed Bias: 40%" (MEET) vs "Speed Bias: 100%" (WEEK)

**Definition:** Percentage of races won by E + E/P combined (early speed runners)

```
MEET: 40% = E(35%) + E/P(5%) = Only 40% early speed wins
      Means 60% of races won by P or S (pressers/closers)
      → CLOSER-FRIENDLY track
      
WEEK: 100% = E(100%) + E/P(0%) = All races won by early runners
      → EXTREMELY speed-biased (but only 1 race - unreliable!)
```

**Bonus Logic:**
```python
if speed_bias > 60:  # Heavy early speed bias
    bonus_for_early = +0.08
    penalty_for_closer = -0.06
    
elif speed_bias < 40:  # Closer-friendly
    bonus_for_closer = +0.07
    penalty_for_early = -0.05
    
else:  # Balanced 40-60%
    neutral_weighting()
```

**This Example (40% MEET):** Moderately closer-friendly (but contradicted by E impact 1.77!)

---

### 4. **Date Range** ✅
**Extraction:** "07/18 - 09/07" (MEET) and "09/01 - 09/07" (WEEK)

**Use:** Prefer MEET totals (more stable) unless race is in latest week

---

### 5. **WnrAvgBL (Winner's Average Beaten Lengths)** ✅
**Extraction:**
```
1stCall: 3.0 lengths (MEET) vs 0.0 (WEEK)
2ndCall: 2.0 lengths (MEET) vs 0.0 (WEEK)
```

**Interpretation (MEET):**
- Winners are +3.0 lengths BEHIND at 1st call
- Winners still +2.0 lengths behind at 2nd call
- → Winners make up ground in stretch (CLOSER-FAVORABLE race structure)

**Contradicts E impact 1.77!** Why?

**Possible Explanation:** 
- Early runners (E impact 1.77) WIN races BUT don't lead by much
- E runners are pressed hard and only win close races
- This is "speed presence" not "dominating front"

**Bonus Logic:**
```python
if first_call_bl > 2.5:  # Winners way back early
    bonus_for_closer = +0.06
    bonus_for_presser = +0.04
    
elif first_call_bl < 0:  # Winners ahead early
    bonus_for_early = +0.07

# Also check tightness between 1stCall and 2ndCall
if abs(first_call_bl - second_call_bl) > 2:
    # Big swing between first and second call
    # Suggests pace changes significantly
    factor_for_pace_adjustment = 1.2
```

---

### 6. **Runstyle (E, E/P, P, S, NA)** ✅
**Extraction:**
```
Runstyle: E   E/P  P   S
          +   --   +   --
Impact:   1.77 0.28 1.35 0.27
%Won:     35%  5%   50%  10%
```

**Data Structure:**
```python
runstyle_bias = {
    "E": {"impact": 1.77, "pct_won": 0.35, "symbol": "++"},
    "E/P": {"impact": 0.28, "pct_won": 0.05, "symbol": ""},
    "P": {"impact": 1.35, "pct_won": 0.50, "symbol": "+"},
    "S": {"impact": 0.27, "pct_won": 0.10, "symbol": ""}
}
```

**Interesting Finding:** P (Presser) has 50% win rate but 1.35 impact
- This means pressers win MORE than expected (150% of expected)
- Even though impact 1.35 < 1.77, the win% (50%) is highest
- Indicates pressers are the most consistent winners at this track

---

### 7. **Impact Value** ✅ CRITICAL EXTRACTION
**Extraction:** Direct from data (1.77, 0.28, 1.35, 0.27, 0.88, 1.17, etc.)

**Del Mar Turf 8.5f Example:**

| Metric | Impact | Effect |
|--------|--------|--------|
| E Early | 1.77 | Winning 177% of expected (++) |
| E/P Presser | 0.28 | Winning only 28% of expected |
| P Middle | 1.35 | Winning 135% of expected (+) |
| S Closer | 0.27 | Winning only 27% of expected |
| Outside (8+) | 1.17 | Posts 8+ winning 117% of expected (+) |

**Bonus Mapping:**
```python
def impact_to_bonus(impact_value, confidence=1.0):
    if impact_value >= 1.80:
        return +0.12 * confidence  # Dominant
    elif impact_value >= 1.50:
        return +0.10 * confidence  # Strong
    elif impact_value >= 1.20:
        return +0.07 * confidence  # Favorable
    elif impact_value >= 1.00:
        return +0.03 * confidence  # Slight edge
    elif impact_value >= 0.80:
        return -0.04 * confidence  # Slight disadvantage
    elif impact_value >= 0.50:
        return -0.08 * confidence  # Significant disadvantage
    else:
        return -0.12 * confidence  # Very weak
```

**This Example Bonuses:**
- E runner: +0.12 (1.77 is dominant) ✓
- E/P runner: -0.08 (0.28 is very weak) ✗ Bad pick
- P runner: +0.10 (1.35 is strong) ✓✓
- S runner: -0.12 (0.27 is very weak) ✗✗ Closers don't work here
- Outside post: +0.07 (1.17 is favorable) ✓

---

### 8. **% Races Won** ✅
**Extraction:** 35%, 5%, 50%, 10% (running styles) and 10%, 10%, 11%, 12% (posts)

**Validation Check:**
```
E: 35% (1.77 impact) - Makes sense
E/P: 5% (0.28 impact) - Makes sense (lowest win%, lowest impact)
P: 50% (1.35 impact) - INTERESTING! Highest win% but moderate impact
S: 10% (0.27 impact) - Makes sense (low win%, low impact)

Sum: 35 + 5 + 50 + 10 = 100% ✓ Validates

Post sum: 10 + 10 + 11 + 12 = 43% (doesn't sum to 100)
This is AVERAGE win%, not total
```

**Insight:** P runners winning 50% doesn't match impact 1.35

Why? Because there are more P runners in the field than E runners.
- Impact Value = (Win% of style) / (Expected% based on field composition)
- So P runners win 50% but that's only 135% of what's expected given their prevalence

---

### 9. **Post Bias** ✅
**Extraction:**
```
Post Bias: RAIL  1-3   4-7  8+
Impact:    0.88  0.88  1.01 1.17 (+)
Avg Win%:  10%   10%   11%  12%
```

**Del Mar Pattern:**
- RAIL & 1-3 both disadvantaged (0.88 = 88% of expected)
- 4-7 neutral (1.01 = just over average)
- 8+ favorable (1.17 = 117% of expected) ← Outside posts best!

**Bonus Mapping for Posts:**
```python
def post_impact_to_bonus(impact_value, confidence=1.0):
    if impact_value >= 1.30:
        return +0.10 * confidence
    elif impact_value >= 1.10:
        return +0.06 * confidence
    elif impact_value >= 0.90:
        return +0.00 * confidence  # Neutral
    elif impact_value >= 0.80:
        return -0.05 * confidence
    else:
        return -0.10 * confidence
```

**This Example:**
- Rail (post 1): -0.05 penalty
- Inner (1-3): -0.05 penalty
- Middle (4-7): 0.00 neutral
- Outside (8+): +0.06 bonus

---

### 10. **Avg Win %** ✅
**Extraction:** 10%, 10%, 11%, 12% for posts (from example)

**Use:** Backup validation when impact unclear

**Correlation Check:**
- Post 1-3: 0.88 impact, 10% win% → Consistent (below average)
- Post 8+: 1.17 impact, 12% win% → Consistent (above average)
- Strong validation

---

### 11. **Track Bias Symbols** ✅
**Extraction:**
- `++` indicates DOMINANT style/post
- `+` indicates FAVORABLE style/post
- `-` or blank indicates UNFAVORABLE or neutral

**From Example:**
```
E (Early): ++      → MOST dominant style
P (Middle): +      → Favorable but not dominant
8+ (Outside): +    → Favorable post bias
```

**Interpretation:**
- This is clearly a track where **Early runners dominate** (++)
- But **Middle-pack runners also do well** (+)
- And **Outside post is advantaged** (+)

**Best Profile:** E runner from post 8+ or higher
- E impact: +0.12
- Post 8+ impact: +0.06
- Bonus sum: +0.18 (huge advantage)

**Worst Profile:** S closer from rail
- S impact: -0.12
- Rail impact: -0.05
- Penalty sum: -0.17 (significant disadvantage)

---

## Parsing Implementation Strategy

### Function Signature:
```python
def parse_track_bias_report(report_text: str) -> dict:
    """
    Parse BRIS track bias report from PDF-extracted text.
    Handles both MEET TOTALS and WEEK TOTALS sections.
    
    Returns: Dictionary with track bias data for both periods
    """
```

### Parsing Sections:

**1. Identify MEET vs WEEK sections**
```python
meet_section = re.search(r'\* MEET Totals \*(.*?)(\* WEEK Totals \*|$)', 
                         report_text, re.DOTALL)
week_section = re.search(r'\* WEEK Totals \*(.*?)$', 
                         report_text, re.DOTALL)
```

**2. Extract surface/distance/race count**
```python
# Line: "Turf 8.5f Speed Bias: 40% WnrAvgBL"
# Line: "# Races: 20 07/18 - 09/07"
```

**3. Extract running style impact values**
```python
# Runstyle: E    E/P  P    S
# Impact:   1.77 0.28 1.35 0.27
```

**4. Extract post position impact values**
```python
# Post Bias: RAIL 1-3  4-7  8+
# Impact:    0.88 0.88 1.01 1.17
```

**5. Extract symbols (++, +, etc.)**

---

## Data Structure for Storage

```python
track_bias_cache = {
    ("DelMar", "Turf", "8.5f"): {
        "meet": {
            "races": 20,
            "date_start": "07/18",
            "date_end": "09/07",
            "confidence": 0.8,
            "wire_pct": 0.25,
            "speed_bias": 0.40,
            "winner_avg_bl_1st": 3.0,
            "winner_avg_bl_2nd": 2.0,
            "runstyles": {
                "E": {"impact": 1.77, "pct_won": 0.35, "symbol": "++"},
                "E/P": {"impact": 0.28, "pct_won": 0.05, "symbol": ""},
                "P": {"impact": 1.35, "pct_won": 0.50, "symbol": "+"},
                "S": {"impact": 0.27, "pct_won": 0.10, "symbol": ""}
            },
            "posts": {
                "RAIL": {"impact": 0.88, "pct_won": 0.10, "symbol": ""},
                "1-3": {"impact": 0.88, "pct_won": 0.10, "symbol": ""},
                "4-7": {"impact": 1.01, "pct_won": 0.11, "symbol": ""},
                "8+": {"impact": 1.17, "pct_won": 0.12, "symbol": "+"}
            }
        },
        "week": {
            "races": 1,
            "confidence": 0.1,  # Very low!
            # ... same structure, but use with caution
        }
    }
}
```

---

## Integration into compute_bias_ratings()

**Current:** Generic bonuses from MODEL_CONFIG
**New:** Extract from actual track bias report for the race

```python
def compute_bias_ratings(...):
    ...
    # Look up track bias data
    track_key = (track_name, surface_type, distance_txt)
    track_bias = track_bias_cache.get(track_key, {})
    
    for _, row in df_styles.iterrows():
        name = str(row.get("Horse"))
        style = _style_norm(row.get("Style") or ...)
        post = str(row.get("Post", row.get("#", "")))
        
        # GET IMPACT-BASED BONUSES (REPLACES GENERIC POST_BIAS_BONUS)
        meet_data = track_bias.get("meet", {})
        runstyle_impact = meet_data.get("runstyles", {}).get(style, {})
        post_group = categorize_post(post)  # Convert post 1 → "RAIL", post 5 → "4-7"
        post_impact = meet_data.get("posts", {}).get(post_group, {})
        
        impact_bonus = impact_to_bonus(runstyle_impact.get("impact", 1.0), 
                                       meet_data.get("confidence", 0.5))
        post_bonus = post_impact_to_bonus(post_impact.get("impact", 1.0),
                                          meet_data.get("confidence", 0.5))
        
        # Apply bonuses instead of generic values
        track_bias_bonus = impact_bonus + post_bonus
        
        arace = ... + track_bias_bonus
```

---

## Potential Accuracy Gain

### Current Generic Approach:
- Post bonuses: ±0.15 to ±0.40 (crude, one-size-fits-all)
- Running style: ±0.05 to ±0.10 (basic matching)
- **Total contribution:** ~±0.25 average

### With Impact Value Approach:
- Running style impact: ±0.08 to ±0.12 (precise to track)
- Post impact: ±0.04 to ±0.10 (precise to track)
- Wire/Speed bias modifiers: ±0.04 to ±0.08
- WnrAvgBL race profile: ±0.04 to ±0.07
- **Total contribution:** ~±0.20 to ±0.37 (more accurate distribution)

### Why Better:
1. **Data-driven:** Based on actual track data, not assumptions
2. **Track-specific:** Del Mar turf ≠ Keeneland dirt
3. **Adjusted for confidence:** Low sample size (n=1) reduced confidence
4. **Holistic:** Combines multiple signals (impact, wire%, speed bias)

**Estimated Gain:** +0.05-0.12 from track bias optimization alone

---

## Next Steps

1. **Parse track bias reports** from example text format
2. **Build lookup cache** for major tracks
3. **Replace generic post bonuses** with impact values
4. **Replace generic running style bonuses** with impact values
5. **Add wire%, speed bias, WnrAvgBL** contextual modifiers
6. **Test against known races** to validate accuracy

---

## Questions to Resolve

1. **Data availability:** Which tracks have bias reports? How are they obtained?
2. **Update frequency:** Daily? Weekly? Seasonal? How recent is the data?
3. **Format consistency:** Is the PDF format always the same, or varies by track?
4. **Multiple track configurations:** Does Del Mar have different biases for main track vs inner turf?
5. **Race conditions:** Are turf biases the same for firm vs yielding? Or separate?

**Answers will determine final implementation strategy.**
