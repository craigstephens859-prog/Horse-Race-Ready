# CRITICAL MODEL ANALYSIS - Pegasus World Cup G1
## Analysis Date: February 4, 2026

## 🚨 SEVERE MODEL FAILURE DETECTED

### Section A Analysis (Race Setup Table)
**Parsed Data:**
- ✅ Horses correctly parsed (5-14 visible)
- ✅ ML odds correctly captured
- ✅ Live odds = ML odds (no market movement)
- ✅ Running styles correctly identified
- ✅ Quirin speed points captured

**Key Observations:**
- #5 Skippylongstocking: E/P, Quirin 4, 15/1
- #11 White Abarrio: E/P, Quirin 4, 4/1 favorite
- #6 Madaket Road: SCRATCHED (not shown but mentioned in PP)
- #14 Catalytic: SCRATCHED (not shown but mentioned in PP)

### Section C Analysis (Overlays Table)
**CRITICAL FAILURE - Fair Probabilities Completely Inverted:**

| Horse | Fair % | Actual Result | Status |
|-------|--------|---------------|--------|
| **Banishing** | **62.13%** | Off Board | ❌ MASSIVE OVERESTIMATE |
| **Captain Cook** | **34.95%** | 4th | ⚠️ Overestimated |
| **Skippylongstocking** | **0.6%** | **WON** | ❌ **CATASTROPHIC UNDERESTIMATE** |
| **White Abarrio** | **0.23%** | 2nd | ❌ **CATASTROPHIC UNDERESTIMATE** |
| British Isles | 1.33% | 5th | ⚠️ Slightly underestimated |
| Tappan Street | 0.12% | DNF | ✓ Reasonable |
| Brotha Keny | 0.49% | DNF | ✓ Reasonable |

### ROOT CAUSE ANALYSIS

#### Problem 1: Model Favoring Wrong Horse (Banishing 62.13%)
**Why is Banishing so heavily favored?**

Looking at PP data:
- Banishing: 20/1 ML, Prime Power 141.5 (7th), beaten as favorite last
- Model giving 62.13% win probability = -164 fair odds

**Hypothesis:** Model is NOT applying the following correctly:
1. ❌ "Beaten as favorite" penalty not implemented
2. ❌ Class weight reduction (G1) not taking effect
3. ❌ Form trend analysis broken

#### Problem 2: Winner Severely Underestimated (Skippy 0.6%)
**Why is Skippylongstocking at 0.6%?**

Looking at PP data:
- Skippylongstocking: Won last race (HarlnHdy-G3) ✓
- 35-day layoff (ideal range) ✓
- Prime Power 140.9 ✓
- Hot jockey/trainer combo ✓

**Hypothesis:** Model is NOT applying:
1. ❌ Win momentum bonus (+2.5 pts) - NOT WORKING
2. ❌ Recent form weighting - NOT WORKING
3. ❌ Optimal layoff bonus - NOT WORKING

#### Problem 3: Elite Horse Underestimated (White Abarrio 0.23%)
**Why is White Abarrio at 0.23%?**

Looking at PP data:
- Highest Prime Power (147.8) ✓
- Won BC Classic G1 2023 ✓
- Won Pegasus last year ✓
- BUT: 146-day layoff (should be -3.0 penalty)

**Hypothesis:**
- Model correctly applying layoff penalty (-3.0)
- BUT over-penalizing to the point of eliminating from contention
- Suggests layoff penalty is TOO SEVERE or being applied multiple times

---

## 🔍 TECHNICAL DIAGNOSIS

### Issue 1: Rating Calculation Path
**Question:** Are the tuned adjustments in app.py being executed?

The tuning I implemented:
1. `calculate_layoff_factor()` - Lines 1857-1877
2. `calculate_form_trend()` - Lines 1880-1920  
3. G1 class weight reduction in race_class_parser.py

**Verification Needed:**
- Are these functions being called?
- Are the return values being used in final rating?
- Is there an alternate calculation path bypassing these?

### Issue 2: Probability Conversion
**The overlays table shows extremely polarized probabilities:**
- Banishing: 62.13%
- All others: <35%
- Top contenders: <1%

This suggests:
1. **Softmax temperature issue**: Ratings might be too extreme
2. **Rating scale problem**: One horse has vastly inflated rating
3. **Probability normalization failure**: Probabilities don't sum to 100%

### Issue 3: Component Weighting
**Hypothesis:** The rating engine might be using OLD weights, not tuned weights.

Need to verify:
- Is race_class_parser returning correct G1 class weight?
- Are form/layoff adjustments being added to rating?
- Is there a cache preventing new calculations?

---

## 🎯 IMMEDIATE ACTION ITEMS

### Priority 1: Verify Tuning is Active
1. Add debug logging to `calculate_layoff_factor()`
2. Add debug logging to `calculate_form_trend()`
3. Print intermediate rating values before/after adjustments

### Priority 2: Check Rating Calculation Path
1. Find where final ratings are calculated
2. Trace whether tuned functions are called
3. Check for alternate calculation bypassing app.py changes

### Priority 3: Investigate Probability Conversion
1. Check softmax calculation in rating-to-probability conversion
2. Verify probabilities sum to 100%
3. Check for extreme rating values causing polarization

### Priority 4: Session State Cache
1. Clear Streamlit session state
2. Force re-calculation of all ratings
3. Verify "Reset / Parse New" clears cached values

---

## 🔬 COMPARISON: Expected vs Actual

### Expected Results (Post-Tuning):
| Rank | Horse | Expected Fair % | Reasoning |
|------|-------|-----------------|-----------|
| 1 | Skippylongstocking | ~25-35% | Won last, optimal layoff, +2.5 bonus |
| 2 | White Abarrio | ~20-30% | Highest PP but -3.0 layoff penalty |
| 3 | Full Serrano | ~15-20% | 2nd in last, BC winner, +1.0 place bonus |
| 4 | Banishing | ~8-12% | Mid-tier, no bonuses, G1 weight reduced |

### Actual Results (Current Model):
| Rank | Horse | Actual Fair % | Error |
|------|-------|---------------|-------|
| 1 | Banishing | 62.13% | +50-54% ERROR |
| 2 | Captain Cook | 34.95% | +15-25% ERROR |
| 3 | Skippylongstocking | 0.6% | -24-34% ERROR |
| 4 | White Abarrio | 0.23% | -20-30% ERROR |

**Magnitude of Errors:** 
- Skippylongstocking: **40-60x underestimated**
- White Abarrio: **90-130x underestimated**
- Banishing: **5-7x overestimated**

---

## 💀 SEVERITY ASSESSMENT

**Critical Level: 10/10 - SYSTEM BROKEN**

The model is producing results that are:
1. **Inverted**: Winner has lowest probability
2. **Polarized**: One horse dominates with 62%
3. **Irrational**: 4/1 favorite has 0.23% win chance
4. **Unusable**: Overlays are completely misleading

**Impact:**
- Model cannot be used for betting decisions
- Users following overlay recommendations would lose money
- Credibility of entire system compromised

---

## 🛠️ NEXT STEPS

1. **IMMEDIATE**: Add detailed logging to rating calculation
2. **URGENT**: Trace execution path for Skippylongstocking
3. **HIGH**: Check if tuning code is actually being executed
4. **HIGH**: Verify probability calculation isn't cached
5. **MEDIUM**: Add validation checks (probabilities sum to 100%)
6. **MEDIUM**: Add sanity checks (no horse >80% unless odds-on favorite)

---

## 📊 VALIDATION QUERIES

Run these checks:
```python
# 1. Are layoff days being calculated correctly?
print(f"Skippy layoff: {skippy_layoff_days} days")
print(f"White Abarrio layoff: {wa_layoff_days} days")

# 2. Are bonuses being applied?
print(f"Skippy form bonus: {skippy_form_trend}")
print(f"Skippy layoff factor: {skippy_layoff_factor}")

# 3. What are the raw ratings before probability conversion?
print(f"Skippy rating: {skippy_rating}")
print(f"Banishing rating: {banishing_rating}")
print(f"White Abarrio rating: {wa_rating}")

# 4. What are the fair probabilities?
print(f"Sum of all probabilities: {sum(all_probs)}")
```

---

## CONCLUSION

**The model is fundamentally broken.** The tuning I implemented is either:
1. Not being executed at all
2. Being overridden by cached values
3. Being applied in the wrong calculation path
4. Creating unintended side effects that invert results

**Immediate action required** to trace execution and identify why ratings are inverted.
