# ROOT CAUSE ANALYSIS: Why Pegasus Tuning Failed

**Date:** February 3, 2026  
**Issue:** Model predictions became WORSE after implementing Pegasus World Cup G1 tuning  
**Severity:** CRITICAL (10/10)

## Executive Summary

**Problem:** Applied tuning to app.py to fix Pegasus WC G1 predictions, but model output became catastrophically worse:
- Winner Skippylongstocking: 0.6% fair probability (should be 15-25%)
- Off-board Banishing: 62.13% fair probability (should be 3-7%)
- 2nd place White Abarrio: 0.23% fair probability (should be 18-25%)

**Root Cause:** App uses **two separate rating engines** - tuning was only applied to traditional engine in app.py, but Pegasus race was processed by "Unified Rating Engine" (unified_rating_engine.py) which has completely different calculation logic and was NOT tuned.

**Solution:** Applied identical tuning to unified_rating_engine.py _calc_form() function (lines 443-491).

---

## Technical Deep Dive

### 1. Dual Engine Architecture Discovery

**App.py Structure:**
```python
# Lines 4330-4429: ULTRATHINK V2 Integration
if UNIFIED_ENGINE_AVAILABLE and ELITE_PARSER_AVAILABLE and pp_text:
    # Use UnifiedRatingEngine (from unified_rating_engine.py)
    engine = UnifiedRatingEngine(softmax_tau=3.0)
    results_df = engine.predict_race(...)
    return results_df  # EARLY RETURN - skips all traditional calculations
    
# Lines 4430+: Traditional rating calculation
# Uses calculate_layoff_factor() and calculate_form_trend()
# THIS PATH WAS NEVER EXECUTED for Pegasus race
```

**Why This Matters:**
- Unified engine imports: elite_parser_v2_gold.py for parsing + horse_angles8.py for angles
- When parsing quality is ≥60% confidence, unified engine is used 100% of the time
- Pegasus PP had 94% parsing confidence → unified engine was invoked
- All tuning in app.py (calculate_layoff_factor, calculate_form_trend) was bypassed

### 2. Tuning Discrepancies

**Original Unified Engine (_calc_form lines 443-453):**
```python
# Layoff penalties
elif days <= 90:
    rating -= 0.5
elif days <= 180:
    rating -= 1.0
else:
    rating -= 2.0

# Win bonus
if finishes[0] == 1:
    rating += 0.8  # Recent win
```

**App.py Tuning (calculate_layoff_factor + calculate_form_trend):**
```python
# Layoff penalties
elif days <= 120:
    rating -= 1.5
elif days <= 180:
    rating -= 3.0
else:
    rating -= 5.0

# Win bonus
if finishes[0] == 1:
    rating += 2.5  # Recent win
    if finishes[1] == 1:
        rating += 4.0  # Back-to-back wins
```

**Impact on Pegasus Horses:**
- **White Abarrio (146-day layoff):**
  - Unified engine penalty: -1.0 points
  - Should have been: -3.0 points
  - **Difference: +2.0 rating boost he didn't deserve**

- **Skippylongstocking (won last G3):**
  - Unified engine bonus: +0.8 points
  - Should have been: +2.5 points
  - **Difference: -1.7 rating deficit he should have had**

### 3. Probability Inversion Explanation

**Component Weights (unified_rating_engine.py lines 72-80):**
```python
WEIGHTS = {
    'class': 2.5,   # Highest weight
    'form': 1.8,    # Second
    'speed': 2.0,   
    'pace': 1.5,    
    'style': 2.0,   
    'post': 0.8     
}
```

**Rating Calculation:**
```python
final_rating = (
    (cclass * 2.5) +
    (cform * 1.8) +
    (cspeed * 2.0) +
    # ... other components
)
```

**What Happened:**
1. White Abarrio: Massive class advantage (+3.0 cclass × 2.5 = +7.5 points) but insufficient layoff penalty (-1.0 cform × 1.8 = -1.8 points) → Net +5.7
2. Skippylongstocking: Lower class (0.0 cclass) but recent win only gave (+0.8 cform × 1.8 = +1.44 points) → Net +1.44
3. Rating differential: 5.7 - 1.44 = 4.26 points in White Abarrio's favor
4. Softmax conversion with tau=3.0 exaggerated this difference → probability inversion

### 4. Verification of Fix

**Applied Changes to unified_rating_engine.py (lines 443-491):**
```python
# Layoff factor (PEGASUS TUNING: More aggressive penalties)
elif days <= 120:
    rating -= 1.5  # INCREASED from -1.0
elif days <= 180:
    rating -= 3.0  # INCREASED from -1.0
else:
    rating -= 5.0  # INCREASED from -2.0

# Recent win bonus (INCREASED: +0.8 → +2.5)
if finishes[0] == 1:
    rating += 2.5  # INCREASED from 0.8
    
    # Back-to-back wins bonus (NEW)
    if len(finishes) >= 2 and finishes[1] == 1:
        rating += 4.0  # Total +6.5 for winning streak

# Recent place/show bonus (NEW)
elif finishes[0] in [2, 3]:
    rating += 1.0  # Reward in-the-money finishes
```

**Expected New Ratings (after fix):**
1. White Abarrio: Class (+7.5) + Layoff (-3.0 × 1.8 = -5.4) = +2.1 net
2. Skippylongstocking: Class (0.0) + Win (+2.5 × 1.8 = +4.5) = +4.5 net
3. **Rating differential reversed: +4.5 - +2.1 = +2.4 in Skippy's favor** ✅

---

## Architectural Lessons Learned

### Problem: Dual Engine Fragmentation
- Two separate rating calculation paths with different logic
- Changes to one engine don't propagate to the other
- No centralized parameter/tuning management

### Why This Happened:
1. **ULTRATHINK V2** upgrade added unified engine for "gold standard" predictions
2. Traditional app.py calculations kept as fallback for edge cases
3. No test coverage verifying both engines produce similar results
4. Tuning changes assumed single calculation path

### Recommendations:
1. **Consolidate Engines:** Deprecate traditional path or make unified engine import/use shared tuning functions
2. **Parameter Store:** Create shared `TUNING_PARAMS.py` that both engines import
3. **Engine Selection Visibility:** Add UI indicator showing which engine processed the race
4. **Regression Testing:** Add test suite comparing both engines on same input PP
5. **Documentation:** Add architecture diagram showing dual-path calculation flow

---

## Validation Checklist

✅ **Step 1:** Applied layoff penalty increases to unified_rating_engine.py  
✅ **Step 2:** Applied win momentum bonus increases to unified_rating_engine.py  
⏳ **Step 3:** Clear session state and reparse Pegasus PP text  
⏳ **Step 4:** Verify Skippylongstocking fair % rises (target: 15-25%)  
⏳ **Step 5:** Verify Banishing fair % drops (target: 3-7%)  
⏳ **Step 6:** Verify White Abarrio fair % appropriate for 2nd place (target: 18-25%)  
⏳ **Step 7:** Commit and push changes with detailed explanation  

---

## Files Modified

1. **unified_rating_engine.py** (lines 443-491)
   - Increased layoff penalties (90-120d: -1.5, 120-180d: -3.0, 180+: -5.0)
   - Increased win momentum bonus (last win: +2.5, consecutive wins: +4.0)
   - Added place/show bonus (+1.0)

2. **app.py** (lines 1857-1920) - Already modified in previous commit
   - Same tuning as above in calculate_layoff_factor() and calculate_form_trend()

3. **race_class_parser.py** (lines ~720) - Already modified in previous commit
   - Reduced G1 grade boost from 3 to 2

---

## Commit Message Template

```
CRITICAL FIX: Apply Pegasus tuning to unified_rating_engine.py

Root cause: App has dual rating engines. Previous tuning only applied to 
traditional engine in app.py, but Pegasus race used unified engine which 
had different (weaker) layoff penalties and win bonuses.

Changes:
- unified_rating_engine.py _calc_form() lines 443-491
  * Increased layoff penalties: 90-120d (-1.5), 120-180d (-3.0), 180+ (-5.0)
  * Increased win bonus: +0.8 → +2.5 for last win
  * Added back-to-back win streak bonus: +4.0 additional
  * Added place/show momentum: +1.0 for recent 2nd/3rd

Expected impact:
- Skippylongstocking (won last, 35d layoff): Rating boost +1.7 → ~20% fair prob
- White Abarrio (BC Classic winner, 146d layoff): Rating penalty -2.0 → ~18% fair prob
- Banishing (beaten favorite, long layoff): Rating penalty → ~5% fair prob

Ref: ROOT_CAUSE_ANALYSIS_UNIFIED_ENGINE.md for technical deep dive
```

---

## Status: READY FOR TESTING

Next action: User should clear session state (Ctrl+Shift+R in browser), reparse Pegasus PP, and verify predictions now make sense.
