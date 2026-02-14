# DEAD FUNCTIONS - QUICK REFERENCE GUIDE
**Immediate Action Items for Restoration**

---

## 🎯 PRIORITY MATRIX

| Function | Status | Value | Integration | Time | Lines |
|----------|--------|-------|-------------|------|-------|
| `calculate_hot_combo_bonus` | 🔴 DEAD | **CRITICAL** | Line 7954 | 1-2h | 6283-6303 |
| `analyze_class_movement` | 🔴 DEAD | **HIGH** | Line 4807 | 2-3h | 6305-6400 |
| `analyze_form_cycle` | 🔴 DEAD | **HIGH** | Line 4823 | 2-3h | 7019-7095 |
| `parse_race_info` | 🔴 DEAD | **LOW** | N/A | N/A | 5889-6044 |

---

## ✅ SUPERSEDED FUNCTIONS (CONFIRMED ACTIVE)

✅ **calculate_workout_bonus_v2** - Line 6046, called at run_audit.py:751, 1145  
✅ **post_bias_score_multi** - Line 7151, called at line 7829  
✅ **style_match_score_multi** - Line 4951, called at line 7828  

**Verdict:** All v2/multi functions properly integrated. No action needed.

---

## 🔴 P0: calculate_hot_combo_bonus (CRITICAL)

### Why Dead Function is SUPERIOR
**Current (calculate_jockey_trainer_impact):**
- ❌ No combo_pct parameter (L60 combo win rate)
- ❌ Only 2 threshold tiers
- ❌ Misses 20%-30% combo range
- ❌ Jockey parsing not implemented

**Dead Function:**
- ✅ Uses actual combo_pct (40%+ L60 = KEY metric!)
- ✅ 3 threshold tiers (40%, 30%, 20%)
- ✅ Separate bonuses stack properly
- ✅ Cap at 0.25 prevents inflation

### Integration Code
```python
# Location: Line 7954

# BEFORE:
tier2_bonus += calculate_jockey_trainer_impact(name, pp_text)

# AFTER:
jt_bonus = calculate_jockey_trainer_impact(name, pp_text)
tier2_bonus += jt_bonus

# NEW: Extract combo data and apply dead function
trainer_pct, jockey_pct = extract_jt_rates(pp_text, name)  # NEW helper needed
combo_pct = parse_jt_combo_pct(pp_text, name)  # NEW helper needed
combo_bonus = calculate_hot_combo_bonus(trainer_pct, jockey_pct, combo_pct)
tier2_bonus += combo_bonus
```

### Helper Functions Needed
```python
def parse_jt_combo_pct(pp_text: str, horse_name: str) -> float:
    """Extract J/T combo L60 win % from PP angles section."""
    # Pattern: "JT L60: XX%" or similar
    # Search in horse section (800 chars after horse_name)
    pass

def extract_jt_rates(pp_text: str, horse_name: str) -> tuple[float, float]:
    """Extract trainer_pct and jockey_pct from existing parsing."""
    # Reuse calculate_jockey_trainer_impact logic
    # Return (trainer_win_rate, jockey_win_rate)
    pass
```

### Expected Impact
- **Litigation win:** 40% L60 combo was KEY factor (now captured!)
- **Rating boost:** +0.10 to +0.25 for elite combos
- **Accuracy gain:** +2-5% on races with strong connections

---

## 🟡 P1: analyze_class_movement (HIGH VALUE)

### Missing Features
**Current (calculate_comprehensive_class_rating):**
- ✅ Today's class hierarchy
- ✅ Purse comparison
- ❌ **MISSING:** Class trend pattern (rising/dropping)
- ❌ **MISSING:** Progression bonus (+0.05 for positive trend)
- ❌ **MISSING:** Class shopping bonus (+0.08 for finding easier spots)

**Dead Function Adds:**
```python
# Pattern detection (last 3 races)
if recent_levels[0] > recent_levels[1] > recent_levels[2]:
    pattern = "rising"
    bonus += 0.05  # Positive progression
elif recent_levels[0] < recent_levels[1] < recent_levels[2]:
    pattern = "dropping"
    bonus += 0.08  # Finding easier spots (proven angle!)
```

### Integration Code
```python
# Location: Line 4807

# AFTER calculating comprehensive_class:
past_races = parse_recent_races_detailed(horse_block)
class_movement_data = analyze_class_movement(past_races, race_type_detected, purse_val)
comprehensive_class += class_movement_data["bonus"]
```

### Expected Impact
- **Class droppers:** +0.08 to +0.12 bonus (key angle!)
- **Rising stars:** +0.05 bonus
- **Overmatched:** -0.10 to -0.15 penalty
- **Accuracy gain:** +3-5%

---

## 🟡 P1: analyze_form_cycle (HIGH VALUE)

### Missing Features
**Current (calculate_form_cycle_rating):**
- ✅ Layoff factor
- ✅ Recent form trend
- ❌ **MISSING:** Progressive trend detection (1 < 2 < 3 finishes)
- ❌ **MISSING:** Figure trend analysis (speed fig progression)
- ❌ **MISSING:** Cycle classification ("improving", "peaking", etc.)
- ❌ **MISSING:** Combined finish + figure trend scoring

**Dead Function Adds:**
```python
# Finish trend (lower is better)
if finishes[0] < finishes[1] < finishes[2]:
    finish_trend = +1.0  # Improving finishes

# Figure trend (higher is better)
if figs[0] > figs[1] > figs[2]:
    fig_trend = +1.0  # Improving figures

# Combined trend score
trend_score = (finish_trend + fig_trend) / 2.0

if trend_score >= 0.75:
    cycle = "improving"
    bonus = +0.15
```

### Integration Code
```python
# Location: Line 4823

# AFTER calculating form_rating:
past_races = parse_recent_races_detailed(horse_block)
form_cycle_data = analyze_form_cycle(past_races)
form_rating += form_cycle_data["bonus"]
```

### Expected Impact
- **Improving horses:** +0.10 to +0.15 bonus
- **Peaking horses:** +0.08 to +0.10 bonus
- **Declining horses:** -0.15 to -0.20 penalty
- **Accuracy gain:** +2-4%

---

## ❌ P2: parse_race_info (LOW PRIORITY)

### Assessment
**Current (detect_race_type) is adequate:**
- ✅ Handles all major race types
- ✅ Works with track bias profiles
- ✅ No critical failures traced to race type detection

**Dead function has advantages:**
- Better FTS detection
- More comprehensive mapping
- Structured output

**Recommendation:** Do NOT restore immediately
- Refactoring risk outweighs benefit
- Merge FTS detection into existing code instead
- Use as reference for future enhancements

---

## 📊 IMPLEMENTATION TIMELINE

### Week 1: Critical Integration
**Day 1-2:** calculate_hot_combo_bonus (P0)
- Create helper functions (2 hours)
- Integrate at line 7954 (1 hour)
- Test on 5 recent races (1 hour)
- **Total: 4 hours**

### Week 2: High Value Integration
**Day 1:** analyze_class_movement (P1)
- Integrate at line 4807 (1 hour)
- Test on 10 recent races (2 hours)
- **Total: 3 hours**

**Day 2:** analyze_form_cycle (P1)
- Integrate at line 4823 (1 hour)
- Test on 10 recent races (2 hours)
- **Total: 3 hours**

### Week 3: Database & Reporting
**Day 1-2:** Database schema updates
- Add new columns (1 hour)
- Update INSERT statements (2 hours)
- **Total: 3 hours**

**Day 3:** High IQ Classic report enhancements
- Add component breakdown (1 hour)
- Add detailed analysis text (1 hour)
- **Total: 2 hours**

**GRAND TOTAL: 15 hours**

---

## 🎯 EXPECTED ROI

### Accuracy Improvements
- **P0 (Hot Combo):** +2-5% on races with elite connections
- **P1 (Class Movement):** +3-5% on class droppers/risers
- **P1 (Form Cycle):** +2-4% on improving form horses
- **TOTAL ESTIMATED GAIN:** +7-14% overall accuracy

### Risk Assessment
- **Implementation risk:** LOW (additive bonuses, no breaking changes)
- **Testing burden:** MODERATE (15-20 races required)
- **Maintenance burden:** LOW (functions already exist, just dead)

### Business Value
- **Litigation-type scenarios:** Now properly capture 40%+ L60 combo edge
- **Class dropping angles:** Automated detection + bonus
- **Form improvement patterns:** Systematic identification
- **Report quality:** Enhanced High IQ Classic explanations

---

## 📋 TESTING CHECKLIST

### P0: Hot Combo Bonus
- [ ] Test on race with 40%+ L60 combo (e.g., Litigation scenario)
- [ ] Test on race with 20-30% combo (mid-tier)
- [ ] Test on race with no combo data (fallback to 0)
- [ ] Verify bonus stacks with jockey_trainer_impact properly
- [ ] Confirm cap at 0.25 works

### P1: Class Movement
- [ ] Test on horse dropping from Stakes to Allowance
- [ ] Test on horse rising from Claiming to Allowance
- [ ] Test on horse with stable class (same level 3+ races)
- [ ] Test on first-time starter (no past races)
- [ ] Verify bonus integrates with comprehensive_class

### P1: Form Cycle
- [ ] Test on horse with improving finishes (3-2-1 pattern)
- [ ] Test on horse with declining finishes (1-3-5 pattern)
- [ ] Test on horse with improving figures (70-75-80)
- [ ] Test on horse with declining figures (85-78-72)
- [ ] Verify bonus integrates with form_rating

---

## 🚀 QUICK START COMMANDS

### Find Integration Points
```bash
# Find where to add hot combo bonus
grep -n "calculate_jockey_trainer_impact" app.py

# Find where to add class movement
grep -n "comprehensive_class = calculate_comprehensive_class_rating" app.py

# Find where to add form cycle
grep -n "form_rating = calculate_form_cycle_rating" app.py
```

### Verify Superseded Functions Active
```bash
# Should show calls in run_audit.py
grep -n "calculate_workout_bonus_v2" app.py

# Should show call at line 7829
grep -n "post_bias_score_multi" app.py

# Should show call at line 7828
grep -n "style_match_score_multi" app.py
```

### Locate Dead Functions
```bash
# Lines 6283-6303
grep -n "def calculate_hot_combo_bonus" app.py

# Lines 6305-6400
grep -n "def analyze_class_movement" app.py

# Lines 7019-7095
grep -n "def analyze_form_cycle" app.py

# Lines 5889-6044
grep -n "def parse_race_info" app.py
```

---

## 📞 SUPPORT REFERENCES

**Full Analysis Report:** [DEAD_FUNCTIONS_ANALYSIS_REPORT.md](DEAD_FUNCTIONS_ANALYSIS_REPORT.md)

**Related Documents:**
- [ANGLE_WEIGHTS_REFERENCE.md](ANGLE_WEIGHTS_REFERENCE.md) - Angle bonus values
- [database_schema_gold.md](database_schema_gold.md) - Database structure
- [ULTRATHINK_PLATINUM_AUDIT_COMPLETE.md](ULTRATHINK_PLATINUM_AUDIT_COMPLETE.md) - Current system audit

**Key Functions:**
- [calculate_comprehensive_class_rating](app.py#L3894) - Base class rating
- [calculate_form_cycle_rating](app.py#L3416) - Base form rating
- [calculate_jockey_trainer_impact](app.py#L5638) - Current J/T bonus
- [detect_race_type](app.py#L1085) - Race type parser

---

**Last Updated:** February 13, 2026  
**Status:** Ready for immediate implementation  
**Priority:** P0 first, then P1, skip P2
