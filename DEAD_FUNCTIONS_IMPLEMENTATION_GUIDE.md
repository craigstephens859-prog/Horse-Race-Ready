# DEAD FUNCTIONS - IMPLEMENTATION GUIDE
**Copy-Paste Ready Code for Immediate Integration**

---

## 📋 IMPLEMENTATION CHECKLIST

- [ ] **Phase 1:** Hot Combo Bonus (CRITICAL - 2 hours)
- [ ] **Phase 2:** Class Movement Bonus (HIGH VALUE - 2 hours)
- [ ] **Phase 3:** Form Cycle Bonus (HIGH VALUE - 2 hours)
- [ ] **Phase 4:** Database Updates (OPTIONAL - 3 hours)
- [ ] **Phase 5:** Testing (REQUIRED - 3 hours)

**Total Time:** 12 hours (Phases 1-3 + testing)

---

## PHASE 1: HOT COMBO BONUS (P0 - CRITICAL)

### Step 1.1: Create Helper Functions

**Add after line 5750 (after calculate_track_condition_granular):**

```python
def parse_jt_combo_pct(pp_text: str, horse_name: str) -> float:
    """
    Extract jockey-trainer combo L60 win % from BRISNET PP angles section.
    
    Patterns to match:
    - "JT L60: 40%" (direct format)
    - "Combo: 10 4-2-1 40%" (combo stats)
    - "JKY/TRN L60 40%" (abbreviated)
    
    Returns: Win percentage as float (0.40 for 40%)
    """
    if not pp_text or not horse_name:
        return 0.0
    
    # Find horse section (search 1000 chars after horse name)
    horse_idx = pp_text.find(horse_name)
    if horse_idx == -1:
        return 0.0
    
    section = pp_text[horse_idx:horse_idx + 1000]
    
    # Try multiple patterns
    patterns = [
        r"JT\s+L60[:\s]+(\d+)%",          # "JT L60: 40%"
        r"Combo[:\s]+\d+\s+\d+-\d+-\d+\s+(\d+)%",  # "Combo: 10 4-2-1 40%"
        r"JKY/TRN\s+L60\s+(\d+)%",        # "JKY/TRN L60 40%"
        r"T/J\s+L60[:\s]+(\d+)%",         # "T/J L60: 40%"
    ]
    
    for pattern in patterns:
        match = re.search(pattern, section, re.IGNORECASE)
        if match:
            win_pct = int(match.group(1))
            return float(win_pct) / 100.0
    
    return 0.0


def extract_jt_rates(pp_text: str, horse_name: str) -> tuple[float, float]:
    """
    Extract individual trainer and jockey win rates from BRISNET PP.
    
    Reuses parsing logic from calculate_jockey_trainer_impact.
    Returns: (trainer_win_rate, jockey_win_rate) as floats
    """
    if not pp_text or not horse_name:
        return (0.0, 0.0)
    
    trainer_win_rate = 0.0
    jockey_win_rate = 0.0
    
    # Find horse section
    horse_section_start = pp_text.find(horse_name)
    if horse_section_start == -1:
        return (0.0, 0.0)
    
    section = pp_text[horse_section_start : horse_section_start + 800]
    
    # TRAINER: "Trnr: LastName FirstName (starts wins-places-shows win%)"
    trainer_pattern = r"Trnr:.*?\((\d+)\s+(\d+)-(\d+)-(\d+)\s+(\d+)%\)"
    trainer_match = re.search(trainer_pattern, section)
    if trainer_match:
        t_starts = int(trainer_match.group(1))
        t_win_pct_reported = int(trainer_match.group(5)) / 100.0
        
        if t_starts >= 20:
            trainer_win_rate = t_win_pct_reported
    
    # JOCKEY: Try multiple patterns
    jockey_patterns = [
        r"Jky:.*?\((\d+)\s+(\d+)-(\d+)-(\d+)\s+(\d+)%\)",  # Standard format
        r"JKYw/.*?(\d+)\s+(\d+)%",  # Angle format: "JKYw/ Sprints 50 22%"
    ]
    
    for jockey_pattern in jockey_patterns:
        jockey_match = re.search(jockey_pattern, section)
        if jockey_match:
            # Handle both formats
            if len(jockey_match.groups()) >= 5:  # Standard format
                j_starts = int(jockey_match.group(1))
                j_win_pct_reported = int(jockey_match.group(5)) / 100.0
            else:  # Angle format
                j_starts = int(jockey_match.group(1))
                j_win_pct_reported = int(jockey_match.group(2)) / 100.0
            
            if j_starts >= 20:
                jockey_win_rate = j_win_pct_reported
                break
    
    return (trainer_win_rate, jockey_win_rate)
```

### Step 1.2: Integrate at Line 7954

**FIND THIS CODE (around line 7954):**
```python
        # ELITE: Jockey/Trainer Performance Impact
        tier2_bonus += calculate_jockey_trainer_impact(name, pp_text)

        # ELITE: Track Condition Granularity
        track_info = st.session_state.get("track_condition_detail", None)
```

**REPLACE WITH:**
```python
        # ELITE: Jockey/Trainer Performance Impact
        jt_bonus = calculate_jockey_trainer_impact(name, pp_text)
        tier2_bonus += jt_bonus

        # ELITE: Hot Jockey/Trainer Combo Bonus (40%+ L60 was KEY to Litigation!)
        trainer_pct, jockey_pct = extract_jt_rates(pp_text, name)
        combo_pct = parse_jt_combo_pct(pp_text, name)
        
        if combo_pct > 0 or trainer_pct > 0 or jockey_pct > 0:
            combo_bonus = calculate_hot_combo_bonus(trainer_pct, jockey_pct, combo_pct)
            tier2_bonus += combo_bonus
            
            # Log for debugging (optional - comment out in production)
            if combo_pct >= 0.30:
                logger.info(
                    f"🔥 {name}: Hot combo {combo_pct*100:.0f}% L60 "
                    f"(T:{trainer_pct*100:.0f}% J:{jockey_pct*100:.0f}%) "
                    f"bonus={combo_bonus:+.2f}"
                )

        # ELITE: Track Condition Granularity
        track_info = st.session_state.get("track_condition_detail", None)
```

### Step 1.3: Test Hot Combo Integration

**Create test script: test_hot_combo.py**
```python
import re
import numpy as np

# Copy the three functions here (calculate_hot_combo_bonus, parse_jt_combo_pct, extract_jt_rates)

# Test case 1: Elite combo (Litigation scenario)
test_pp_1 = """
Horse Name: Litigation
Trnr: Smith John (50 11-7-4 22%)
JT L60: 40%
"""

trainer_pct, jockey_pct = extract_jt_rates(test_pp_1, "Litigation")
combo_pct = parse_jt_combo_pct(test_pp_1, "Litigation")
bonus = calculate_hot_combo_bonus(trainer_pct, jockey_pct, combo_pct)

print(f"Test 1 - Elite Combo (40% L60):")
print(f"  Trainer: {trainer_pct*100:.0f}%")
print(f"  Jockey: {jockey_pct*100:.0f}%")
print(f"  Combo: {combo_pct*100:.0f}%")
print(f"  Bonus: {bonus:+.2f}")
print(f"  Expected: +0.20 to +0.25")
print()

# Test case 2: Mid-tier combo
test_pp_2 = """
Horse Name: Test Horse
Trnr: Jones Bob (40 8-5-3 20%)
Combo: 15 4-3-2 27%
"""

trainer_pct, jockey_pct = extract_jt_rates(test_pp_2, "Test Horse")
combo_pct = parse_jt_combo_pct(test_pp_2, "Test Horse")
bonus = calculate_hot_combo_bonus(trainer_pct, jockey_pct, combo_pct)

print(f"Test 2 - Mid-Tier Combo (27% L60):")
print(f"  Trainer: {trainer_pct*100:.0f}%")
print(f"  Jockey: {jockey_pct*100:.0f}%")
print(f"  Combo: {combo_pct*100:.0f}%")
print(f"  Bonus: {bonus:+.2f}")
print(f"  Expected: +0.10 to +0.15")
print()

# Test case 3: No combo data (fallback)
test_pp_3 = """
Horse Name: Unknown Horse
Trnr: Brown Sam (25 3-4-2 12%)
"""

trainer_pct, jockey_pct = extract_jt_rates(test_pp_3, "Unknown Horse")
combo_pct = parse_jt_combo_pct(test_pp_3, "Unknown Horse")
bonus = calculate_hot_combo_bonus(trainer_pct, jockey_pct, combo_pct)

print(f"Test 3 - No Combo Data:")
print(f"  Trainer: {trainer_pct*100:.0f}%")
print(f"  Jockey: {jockey_pct*100:.0f}%")
print(f"  Combo: {combo_pct*100:.0f}%")
print(f"  Bonus: {bonus:+.2f}")
print(f"  Expected: 0.00 to +0.05")
```

**Run test:**
```bash
python test_hot_combo.py
```

---

## PHASE 2: CLASS MOVEMENT BONUS (P1 - HIGH VALUE)

### Step 2.1: Integration at Line 4807

**FIND THIS CODE (around line 4807):**
```python
    # Calculate comprehensive class rating
    # NOW includes PP text for race class parser to properly understand race acronyms
    comprehensive_class = calculate_comprehensive_class_rating(
        today_purse=purse_val,
        today_race_type=race_type_detected,
        horse_block=horse_block,
        pedigree=ped,
        angles_df=ang if ang is not None else pd.DataFrame(),
        pp_text=pp_text,  # NEW: Full PP for race analysis
        _distance_furlongs=distance_to_furlongs(distance_txt),  # NEW: Distance
        _surface_type=race_surface,  # NEW: Surface type
    )

    # Add pedigree/angle tweaks on top of class
    tweak = _angles_pedigree_tweak(name, race_surface, race_bucket, race_cond)
    cclass_total = comprehensive_class + tweak
```

**REPLACE WITH:**
```python
    # Calculate comprehensive class rating
    # NOW includes PP text for race class parser to properly understand race acronyms
    comprehensive_class = calculate_comprehensive_class_rating(
        today_purse=purse_val,
        today_race_type=race_type_detected,
        horse_block=horse_block,
        pedigree=ped,
        angles_df=ang if ang is not None else pd.DataFrame(),
        pp_text=pp_text,  # NEW: Full PP for race analysis
        _distance_furlongs=distance_to_furlongs(distance_txt),  # NEW: Distance
        _surface_type=race_surface,  # NEW: Surface type
    )

    # CLASS MOVEMENT BONUS: Analyze if horse is moving up/down in class
    past_races = parse_recent_races_detailed(horse_block)
    class_movement_data = analyze_class_movement(past_races, race_type_detected, purse_val)
    class_movement_bonus = class_movement_data["bonus"]
    
    # Log significant class changes (optional - comment out in production)
    if abs(class_movement_bonus) >= 0.08:
        movement_type = class_movement_data["class_change"]
        pattern = class_movement_data["pattern"]
        logger.info(
            f"📊 {name}: Class {movement_type} ({pattern}) "
            f"bonus={class_movement_bonus:+.2f}"
        )

    # Add pedigree/angle tweaks on top of class
    tweak = _angles_pedigree_tweak(name, race_surface, race_bucket, race_cond)
    cclass_total = comprehensive_class + tweak + class_movement_bonus
```

### Step 2.2: Test Class Movement Integration

**Test on known class dropper (e.g., Stakes to Allowance):**
```python
# In your test script or notebook:
test_block = """
Horse Name: Class Dropper
Recent Races:
- 2025-01-15: Stakes G3 $200,000 (5th place)
- 2025-12-10: Stakes Listed $150,000 (6th place)
- 2025-11-20: Stakes $100,000 (4th place)
"""

# Today's race: Allowance $50,000
class_movement = analyze_class_movement(
    past_races=[
        {"class": "Stakes G3", "purse": 200000, "finish": 5},
        {"class": "Stakes Listed", "purse": 150000, "finish": 6},
        {"class": "Stakes", "purse": 100000, "finish": 4},
    ],
    today_class="Allowance",
    today_purse=50000
)

print(f"Class Change: {class_movement['class_change']}")  # Should be "down"
print(f"Pattern: {class_movement['pattern']}")  # Should be "dropping" or "stable"
print(f"Bonus: {class_movement['bonus']:+.2f}")  # Should be +0.08 to +0.12
```

---

## PHASE 3: FORM CYCLE BONUS (P1 - HIGH VALUE)

### Step 3.1: Integration at Line 4823

**FIND THIS CODE (around line 4823):**
```python
    # Calculate form cycle rating
    form_rating = calculate_form_cycle_rating(
        horse_block=horse_block,
        pedigree=ped,
        angles_df=ang if ang is not None else pd.DataFrame(),
    )

    # Track bias style adjustments
    style_adjustment = 0.0
```

**REPLACE WITH:**
```python
    # Calculate form cycle rating
    form_rating = calculate_form_cycle_rating(
        horse_block=horse_block,
        pedigree=ped,
        angles_df=ang if ang is not None else pd.DataFrame(),
    )

    # FORM CYCLE BONUS: Analyze progressive improvement/decline
    past_races = parse_recent_races_detailed(horse_block)
    form_cycle_data = analyze_form_cycle(past_races)
    form_cycle_bonus = form_cycle_data["bonus"]
    
    # Log significant form cycles (optional - comment out in production)
    if abs(form_cycle_bonus) >= 0.10:
        cycle_type = form_cycle_data["cycle"]
        trend_score = form_cycle_data["trend_score"]
        logger.info(
            f"📈 {name}: Form cycle {cycle_type} (trend={trend_score:+.2f}) "
            f"bonus={form_cycle_bonus:+.2f}"
        )
    
    form_rating += form_cycle_bonus

    # Track bias style adjustments
    style_adjustment = 0.0
```

### Step 3.2: Test Form Cycle Integration

**Test on improving horse (finishes: 5-3-1):**
```python
# Test case: Improving finishes + improving figures
test_races = [
    {"finish": 1, "speed_fig": 85, "days_ago": 14},
    {"finish": 3, "speed_fig": 80, "days_ago": 35},
    {"finish": 5, "speed_fig": 75, "days_ago": 56},
]

form_cycle = analyze_form_cycle(test_races)

print(f"Cycle: {form_cycle['cycle']}")  # Should be "improving"
print(f"Trend Score: {form_cycle['trend_score']:+.2f}")  # Should be +0.75 to +1.0
print(f"Bonus: {form_cycle['bonus']:+.2f}")  # Should be +0.10 to +0.15
print(f"Last 3 Finishes: {form_cycle['last_3_finishes']}")  # [1, 3, 5]
print(f"Last 3 Figs: {form_cycle['last_3_figs']}")  # [85, 80, 75]
```

---

## PHASE 4: DATABASE UPDATES (OPTIONAL)

### Step 4.1: Add Database Columns

**Create migration script: migrate_dead_functions.sql**
```sql
-- Add class movement columns
ALTER TABLE runners ADD COLUMN class_movement_trend TEXT; -- 'up', 'down', 'same', 'unknown'
ALTER TABLE runners ADD COLUMN class_movement_pattern TEXT; -- 'rising', 'dropping', 'stable'
ALTER TABLE runners ADD COLUMN class_movement_bonus FLOAT DEFAULT 0.0;

-- Add form cycle columns
ALTER TABLE runners ADD COLUMN form_cycle_type TEXT; -- 'improving', 'peaking', 'declining', 'bottoming', 'stable'
ALTER TABLE runners ADD COLUMN form_cycle_trend_score FLOAT DEFAULT 0.0;
ALTER TABLE runners ADD COLUMN form_cycle_bonus FLOAT DEFAULT 0.0;

-- Add hot combo columns
ALTER TABLE runners ADD COLUMN jt_combo_win_pct FLOAT DEFAULT 0.0; -- L60 combo win %
ALTER TABLE runners ADD COLUMN hot_combo_bonus FLOAT DEFAULT 0.0;

-- Create indexes for analysis queries
CREATE INDEX IF NOT EXISTS idx_class_movement_trend ON runners(class_movement_trend);
CREATE INDEX IF NOT EXISTS idx_form_cycle_type ON runners(form_cycle_type);
CREATE INDEX IF NOT EXISTS idx_hot_combo ON runners(jt_combo_win_pct) WHERE jt_combo_win_pct >= 0.20;
```

**Run migration:**
```bash
sqlite3 gold_high_iq.db < migrate_dead_functions.sql
```

### Step 4.2: Update INSERT Statements

**Find all INSERT statements (4 files):**
```bash
grep -n "INSERT INTO gold_high_iq" *.py
grep -n "INSERT INTO runners" *.py
```

**Add to each INSERT (example):**
```python
# BEFORE INSERT, calculate and store values:
class_movement_data = analyze_class_movement(past_races, race_type, purse)
form_cycle_data = analyze_form_cycle(past_races)
trainer_pct, jockey_pct = extract_jt_rates(pp_text, horse_name)
combo_pct = parse_jt_combo_pct(pp_text, horse_name)
hot_combo_bonus = calculate_hot_combo_bonus(trainer_pct, jockey_pct, combo_pct)

# Add to INSERT statement:
cursor.execute('''
    INSERT INTO runners (
        ...,
        class_movement_trend,
        class_movement_pattern,
        class_movement_bonus,
        form_cycle_type,
        form_cycle_trend_score,
        form_cycle_bonus,
        jt_combo_win_pct,
        hot_combo_bonus
    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ...)
''', (
    ...,
    class_movement_data["class_change"],
    class_movement_data["pattern"],
    class_movement_data["bonus"],
    form_cycle_data["cycle"],
    form_cycle_data["trend_score"],
    form_cycle_data["bonus"],
    combo_pct,
    hot_combo_bonus
))
```

---

## PHASE 5: TESTING (REQUIRED)

### Test Suite: test_dead_functions_integration.py

```python
"""
Comprehensive test suite for dead function restoration.
Run after all integrations complete.
"""
import sys
sys.path.append(".")
import app as APP

def test_hot_combo_bonus():
    """Test calculate_hot_combo_bonus integration"""
    print("=" * 60)
    print("TEST 1: HOT COMBO BONUS")
    print("=" * 60)
    
    # Test case 1: Elite combo (40%+)
    bonus1 = APP.calculate_hot_combo_bonus(0.22, 0.18, 0.40)
    assert 0.20 <= bonus1 <= 0.25, f"Elite combo bonus out of range: {bonus1}"
    print(f"✅ Elite combo (40% L60): {bonus1:+.2f}")
    
    # Test case 2: Mid-tier combo (30%)
    bonus2 = APP.calculate_hot_combo_bonus(0.20, 0.15, 0.30)
    assert 0.15 <= bonus2 <= 0.20, f"Mid combo bonus out of range: {bonus2}"
    print(f"✅ Mid combo (30% L60): {bonus2:+.2f}")
    
    # Test case 3: Low combo (20%)
    bonus3 = APP.calculate_hot_combo_bonus(0.15, 0.12, 0.20)
    assert 0.10 <= bonus3 <= 0.15, f"Low combo bonus out of range: {bonus3}"
    print(f"✅ Low combo (20% L60): {bonus3:+.2f}")
    
    # Test case 4: No combo data
    bonus4 = APP.calculate_hot_combo_bonus(0.10, 0.08, 0.0)
    assert 0.0 <= bonus4 <= 0.05, f"No combo bonus out of range: {bonus4}"
    print(f"✅ No combo data: {bonus4:+.2f}")
    
    print()

def test_class_movement():
    """Test analyze_class_movement integration"""
    print("=" * 60)
    print("TEST 2: CLASS MOVEMENT BONUS")
    print("=" * 60)
    
    # Test case 1: Class dropper (Stakes to Allowance)
    past1 = [
        {"class": "Stakes G3", "purse": 200000},
        {"class": "Stakes", "purse": 150000},
        {"class": "Allowance", "purse": 80000},
    ]
    result1 = APP.analyze_class_movement(past1, "Allowance", 50000)
    print(f"✅ Class dropper: {result1['class_change']} ({result1['pattern']}) bonus={result1['bonus']:+.2f}")
    assert result1['class_change'] == 'down', "Should detect class drop"
    assert result1['bonus'] > 0, "Class drop should have positive bonus"
    
    # Test case 2: Class riser (Claiming to Allowance)
    past2 = [
        {"class": "Claiming", "purse": 20000},
        {"class": "Claiming", "purse": 25000},
        {"class": "Claiming", "purse": 20000},
    ]
    result2 = APP.analyze_class_movement(past2, "Allowance", 50000)
    print(f"✅ Class riser: {result2['class_change']} ({result2['pattern']}) bonus={result2['bonus']:+.2f}")
    assert result2['class_change'] == 'up', "Should detect class rise"
    assert result2['bonus'] < 0, "Class rise should have negative bonus (tougher)"
    
    # Test case 3: Stable class
    past3 = [
        {"class": "Allowance", "purse": 50000},
        {"class": "Allowance", "purse": 55000},
        {"class": "Allowance", "purse": 48000},
    ]
    result3 = APP.analyze_class_movement(past3, "Allowance", 50000)
    print(f"✅ Stable class: {result3['class_change']} ({result3['pattern']}) bonus={result3['bonus']:+.2f}")
    assert result3['class_change'] == 'same', "Should detect stable class"
    
    print()

def test_form_cycle():
    """Test analyze_form_cycle integration"""
    print("=" * 60)
    print("TEST 3: FORM CYCLE BONUS")
    print("=" * 60)
    
    # Test case 1: Improving form (finishes: 1-3-5, figs: 85-80-75)
    past1 = [
        {"finish": 1, "speed_fig": 85},
        {"finish": 3, "speed_fig": 80},
        {"finish": 5, "speed_fig": 75},
    ]
    result1 = APP.analyze_form_cycle(past1)
    print(f"✅ Improving: {result1['cycle']} (trend={result1['trend_score']:+.2f}) bonus={result1['bonus']:+.2f}")
    assert result1['cycle'] in ['improving', 'peaking'], "Should detect improving form"
    assert result1['bonus'] > 0, "Improving form should have positive bonus"
    
    # Test case 2: Declining form (finishes: 5-3-1, figs: 75-80-85)
    past2 = [
        {"finish": 5, "speed_fig": 75},
        {"finish": 3, "speed_fig": 80},
        {"finish": 1, "speed_fig": 85},
    ]
    result2 = APP.analyze_form_cycle(past2)
    print(f"✅ Declining: {result2['cycle']} (trend={result2['trend_score']:+.2f}) bonus={result2['bonus']:+.2f}")
    assert result2['cycle'] in ['declining', 'bottoming'], "Should detect declining form"
    assert result2['bonus'] < 0, "Declining form should have negative bonus"
    
    # Test case 3: Stable form (finishes: 3-3-3, figs: 80-80-80)
    past3 = [
        {"finish": 3, "speed_fig": 80},
        {"finish": 3, "speed_fig": 80},
        {"finish": 3, "speed_fig": 80},
    ]
    result3 = APP.analyze_form_cycle(past3)
    print(f"✅ Stable: {result3['cycle']} (trend={result3['trend_score']:+.2f}) bonus={result3['bonus']:+.2f}")
    assert result3['cycle'] == 'stable', "Should detect stable form"
    
    print()

def test_parsing_helpers():
    """Test new parsing helper functions"""
    print("=" * 60)
    print("TEST 4: PARSING HELPERS")
    print("=" * 60)
    
    test_pp = """
    Horse Name: Test Horse
    Trnr: Smith John (50 11-7-4 22%)
    JT L60: 35%
    """
    
    # Test parse_jt_combo_pct
    combo_pct = APP.parse_jt_combo_pct(test_pp, "Test Horse")
    print(f"✅ Combo %: {combo_pct*100:.0f}%")
    assert combo_pct == 0.35, f"Should extract 35%, got {combo_pct*100:.0f}%"
    
    # Test extract_jt_rates
    trainer_pct, jockey_pct = APP.extract_jt_rates(test_pp, "Test Horse")
    print(f"✅ Trainer %: {trainer_pct*100:.0f}%")
    assert trainer_pct == 0.22, f"Should extract 22%, got {trainer_pct*100:.0f}%"
    
    print()

if __name__ == "__main__":
    print("\n" + "=" * 60)
    print("DEAD FUNCTIONS INTEGRATION TEST SUITE")
    print("=" * 60 + "\n")
    
    try:
        test_hot_combo_bonus()
        test_class_movement()
        test_form_cycle()
        test_parsing_helpers()
        
        print("=" * 60)
        print("✅ ALL TESTS PASSED")
        print("=" * 60)
        print()
        print("Next steps:")
        print("1. Test on 5-10 recent races")
        print("2. Compare ratings before/after integration")
        print("3. Verify bonus ranges are reasonable")
        print("4. Update database schema (Phase 4)")
        
    except AssertionError as e:
        print(f"\n❌ TEST FAILED: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ ERROR: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
```

**Run test suite:**
```bash
python test_dead_functions_integration.py
```

---

## 📊 VALIDATION CHECKLIST

After implementing all phases:

### Functional Tests
- [ ] Hot combo bonus: 40%+ combo → +0.20 to +0.25
- [ ] Hot combo bonus: 20-30% combo → +0.10 to +0.15
- [ ] Hot combo bonus: No data → 0.00
- [ ] Class movement: Stakes to Allowance → +0.08 to +0.12
- [ ] Class movement: Claiming to Allowance → -0.10 to -0.15
- [ ] Form cycle: Improving (1-3-5) → +0.10 to +0.15
- [ ] Form cycle: Declining (5-3-1) → -0.15 to -0.20

### Integration Tests
- [ ] Bonuses stack properly (no double-counting)
- [ ] Bonuses respect caps (0.25 for combo, 0.20 for class, 0.20 for form)
- [ ] Logging output is readable
- [ ] No crashes on missing data
- [ ] Fallback to 0.0 when parsing fails

### Regression Tests
- [ ] Run 10 recent races, compare ratings before/after
- [ ] Verify top pick unchanged for races where bonuses don't apply
- [ ] Verify top pick IMPROVED for races with elite combos/class movement
- [ ] Check database INSERT works (if Phase 4 completed)

### Performance Tests
- [ ] Page load time unchanged (<2 sec for full race)
- [ ] No memory leaks (run 50 races in sequence)
- [ ] Parsing functions fast (<10ms per horse)

---

## 🚀 DEPLOYMENT CHECKLIST

Before pushing to production:

- [ ] All functional tests pass
- [ ] All integration tests pass
- [ ] Test on 20+ recent races (mix of tracks/distances/surfaces)
- [ ] Backup app.py before changes
- [ ] Document all changes in git commit
- [ ] Update CHANGELOG.md
- [ ] Notify users of new features (if public-facing)

---

## 📞 TROUBLESHOOTING

### Issue: Parsing functions return 0.0 for all horses
**Cause:** PP text format doesn't match regex patterns  
**Fix:** Add debug logging to see actual PP text, adjust patterns

```python
# Add to parsing functions:
logger.debug(f"Section text for {horse_name}: {section[:200]}...")
```

### Issue: Bonuses too high (>0.50 total)
**Cause:** Multiple bonuses stacking without proper caps  
**Fix:** Add global cap after all tier2 bonuses

```python
# After all tier2 bonuses:
tier2_bonus = float(np.clip(tier2_bonus, -1.0, 1.0))
```

### Issue: Database INSERT fails
**Cause:** New columns don't exist  
**Fix:** Run migration script (Phase 4 Step 4.1)

```bash
sqlite3 gold_high_iq.db < migrate_dead_functions.sql
```

---

## 📚 REFERENCE LINKS

- **Full Analysis:** [DEAD_FUNCTIONS_ANALYSIS_REPORT.md](DEAD_FUNCTIONS_ANALYSIS_REPORT.md)
- **Quick Reference:** [DEAD_FUNCTIONS_QUICK_REFERENCE.md](DEAD_FUNCTIONS_QUICK_REFERENCE.md)
- **App.py Functions:**
  - [calculate_hot_combo_bonus](app.py#L6283)
  - [analyze_class_movement](app.py#L6305)
  - [analyze_form_cycle](app.py#L7019)
  - [calculate_jockey_trainer_impact](app.py#L5638)

---

**Last Updated:** February 13, 2026  
**Implementation Status:** Ready for execution  
**Estimated Total Time:** 12 hours (Phases 1-3 + testing)
