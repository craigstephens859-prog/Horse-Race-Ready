# ML Engine Optimization Plan: Achieving 90% Winner Accuracy

## Executive Summary
**Current Status Analysis:**
- ✅ Phase 1: Enhanced parsing (Track Bias Impact Values, SPI, Surface Stats, AWD)
- ✅ Phase 2: ML calibration system with neural network
- ❌ **CRITICAL GAPS IDENTIFIED:**
  1. Missing weather/track condition data
  2. No jockey/trainer performance metrics in ML features
  3. Limited historical form analysis
  4. Post-time odds not captured
  5. No velocity/fractional pace data
  6. Missing equipment changes tracking
  7. No workouts analysis in ML features

---

## PART 1: CURRENT SYSTEM DIAGNOSIS

### What We Have:
**Rating Components (Base Model):**
- **Cclass**: Class rating (2.5x weight) - purse, race type, angles
- **Cspeed**: Speed figures (2.0x weight) - AvgTop2 Beyer
- **Cpace**: Pace pressure (1.5x weight) - PPI calculation
- **Cstyle**: Running style match (1.2x weight) - E/EP/P/S vs track bias
- **Cpost**: Post position bias (0.8x weight)
- **Atrack**: Track-specific adjustments
- **Tier 2 Bonuses**: Impact Values, SPI, Surface Stats, AWD

**ML System (15-feature neural network):**
```python
Features Currently Used:
1. predicted_win_prob (from rating system)
2. rating_total
3. rating_class
4. rating_speed
5. rating_pace
6. rating_style
7. rating_post
8. rating_angles
9. rating_pedigree
10. final_odds
11. morning_line
12. post_position
13. running_style (encoded)
14. quirin_points
15. last_beyer/avg_beyer
```

### Why 3rd/4th Place Predictions Fail:

**Root Causes:**
1. **Over-Optimization for Winners** - Model uses winner-focused loss (BCE)
2. **Insufficient Separation Power** - Close ratings don't distinguish 2nd-4th
3. **Missing Late-Race Factors**:
   - Trip handicapping (trouble, wide trips)
   - Rider decisions/skill
   - Closing kick ability
   - Stamina at distance
4. **No Multi-Output Architecture** - Single win probability doesn't model placings
5. **Harville Formula Weakness** - Assumes independent probabilities for exotic positions

---

## PART 2: DATA COLLECTION ENHANCEMENT

### NEW DATA POINTS TO CAPTURE (Priority Order):

#### **TIER 1: CRITICAL ADDITIONS**
```python
# Add to race_history.db schema
ALTER TABLE races ADD COLUMN:
- weather TEXT (fast/wet/sloppy/frozen)
- track_condition_variant TEXT (sealed/cuppy/deep)
- rail_distance TEXT (normal/+15ft/out)
- wind_speed INTEGER
- temperature INTEGER
- race_class_numeric INTEGER (0-12 scale)
- field_competitiveness REAL (std dev of ratings)

ALTER TABLE horses ADD COLUMN:
- jockey_name TEXT
- jockey_win_pct REAL (current meet)
- jockey_roi REAL (current meet)
- trainer_name TEXT
- trainer_win_pct REAL
- trainer_roi REAL
- days_since_last REAL
- medication TEXT (Lasix, Bute)
- equipment TEXT (blinkers on/off/first time)
- weight_carried INTEGER
- claimed_last_out BOOLEAN
- trouble_line TEXT (parsed from comment)
- trip_rating INTEGER (1-5: 1=worst trip, 5=perfect)
- fractional_times TEXT (e.g., "22.1,45.2,1:09.4")
- late_pace_rating REAL (calculated from fractions)
- actual_place INTEGER (not just win, capture 2nd-4th)
- beaten_lengths REAL
- speed_rating INTEGER (track-specific speed fig)
- track_variant INTEGER (daily adjustment)
```

#### **TIER 2: COMPETITIVE INTELLIGENCE**
```python
# Comparative metrics within race
- rating_rank INTEGER (1-12 within field)
- odds_rank INTEGER (favorite = 1)
- class_drop_indicator REAL (purse differential)
- pace_scenario_role TEXT (presser/stalker/closer)
- expected_fractions TEXT (projected)
- form_cycle TEXT (improving/declining/peaking)
- layoff_category TEXT (fresh/rusty/long)
```

#### **TIER 3: ADVANCED ANALYTICS**
```python
# Pattern recognition
- turf_pedigree_score REAL
- mud_breeding_score REAL
- distance_breeding_score REAL
- trainer_pattern_match BOOLEAN (e.g., "2nd off layoff")
- jockey_trainer_combo_roi REAL
- post_position_track_bias REAL (track-specific)
- pace_shape_advantage REAL (speed figure adjusted for pace)
```

---

## PART 3: ALGORITHM ENHANCEMENTS

### 3.1 Multi-Output Neural Network Architecture

**Replace Single-Output Winner Model with 4-Output Model:**
```python
class MultiPlaceNet(nn.Module):
    def __init__(self, input_dim=45):  # Expanded features
        super().__init__()
        # Shared layers
        self.shared1 = nn.Linear(input_dim, 128)
        self.bn1 = nn.BatchNorm1d(128)
        self.dropout1 = nn.Dropout(0.3)
        
        self.shared2 = nn.Linear(128, 64)
        self.bn2 = nn.BatchNorm1d(64)
        self.dropout2 = nn.Dropout(0.2)
        
        # Separate heads for each finishing position
        self.win_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.place_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.show_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
        
        self.fourth_head = nn.Sequential(
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Linear(32, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # Shared feature extraction
        x = F.relu(self.bn1(self.shared1(x)))
        x = self.dropout1(x)
        x = F.relu(self.bn2(self.shared2(x)))
        x = self.dropout2(x)
        
        # Multi-task outputs
        win_prob = self.win_head(x)
        place_prob = self.place_head(x)
        show_prob = self.show_head(x)
        fourth_prob = self.fourth_head(x)
        
        return win_prob, place_prob, show_prob, fourth_prob

# Custom loss function
def multi_place_loss(outputs, targets, weights=[2.0, 1.5, 1.2, 1.0]):
    """Weighted loss giving more importance to win prediction."""
    win_out, place_out, show_out, fourth_out = outputs
    win_target, place_target, show_target, fourth_target = targets
    
    loss = (weights[0] * F.binary_cross_entropy(win_out, win_target) +
            weights[1] * F.binary_cross_entropy(place_out, place_target) +
            weights[2] * F.binary_cross_entropy(show_out, show_target) +
            weights[3] * F.binary_cross_entropy(fourth_out, fourth_target))
    
    return loss / sum(weights)
```

### 3.2 Ensemble Model Approach

**Combine multiple specialized models:**
```python
class EnsembleRacingModel:
    def __init__(self):
        self.winner_model = WinnerFocusedNN()  # Optimized for 1st
        self.placer_model = PlaceFocusedNN()   # Optimized for 2-3
        self.deep_closer_model = DeepCloserNN() # Optimized for 3-4
        self.odds_calibrator = OddsBasedNN()   # Market wisdom
        
    def predict(self, features):
        win_prob = self.winner_model(features) * 0.40
        place_prob = self.placer_model(features) * 0.30
        closer_prob = self.deep_closer_model(features) * 0.20
        odds_prob = self.odds_calibrator(features) * 0.10
        
        # Weighted ensemble
        final_prob = win_prob + place_prob + closer_prob + odds_prob
        return final_prob
```

### 3.3 Enhanced Rating Formula

**Add "Finishing Kick" component:**
```python
def compute_enhanced_ratings(df_styles, ..., late_pace_data):
    # Existing components
    base_rating = c_class + cstyle + cpost + cpace + a_track + tier2_bonus
    
    # NEW: Late pace component (Ckick)
    ckick = calculate_late_pace_rating(
        last_fraction=horse_data['last_fraction'],
        avg_late_pace=race_avg_late_pace,
        distance_category=distance_bucket,
        running_style=style
    )
    
    # NEW: Trip/luck component (Ctrip)
    ctrip = calculate_trip_rating(
        last_race_trouble=horse_data['trouble_line'],
        avg_beaten_lengths=horse_data['avg_beaten_lengths'],
        pace_scenario_fit=horse_data['pace_fit']
    )
    
    # NEW: Jockey/Trainer component (Chuman)
    chuman = calculate_human_factor(
        jockey_roi=horse_data['jockey_roi'],
        trainer_roi=horse_data['trainer_roi'],
        combo_roi=horse_data['combo_roi']
    )
    
    # UPDATED FORMULA
    final_rating = (
        base_rating * 0.70 +      # 70% fundamental rating
        ckick * 0.15 +             # 15% late pace ability
        ctrip * 0.10 +             # 10% trip/luck factors
        chuman * 0.05              # 5% human factors
    )
    
    return final_rating
```

---

## PART 4: SPECIFIC FIXES FOR 3rd/4th PLACE

### 4.1 "Logical Contender" Identification

```python
def identify_logical_contenders(ratings_df, fair_probs, odds_data):
    """
    Beyond top-rated, find hidden contenders for 2nd-4th.
    """
    contenders = []
    
    for idx, horse in ratings_df.iterrows():
        score = 0
        
        # Factor 1: Underbet horses (value overlay)
        if fair_probs[horse['Horse']] > (1 / odds_data[horse['Horse']]):
            score += 3
        
        # Factor 2: Recent form improving
        if horse.get('form_cycle') == 'improving':
            score += 2
        
        # Factor 3: Tactical advantage (pace fit)
        if horse.get('pace_advantage') > 0.5:
            score += 2
        
        # Factor 4: Late pace ability
        if horse.get('late_pace_rating') > race_avg_late_pace + 1.0:
            score += 2
        
        # Factor 5: Trainer pattern match
        if horse.get('trainer_pattern_match'):
            score += 1
        
        # Factor 6: Closing kick at odds
        if horse['Style'] in ['P', 'S'] and horse.get('late_pace_rating') > 85:
            score += 2
        
        if score >= 4:  # Threshold for "logical"
            contenders.append({
                'horse': horse['Horse'],
                'rating': horse['R'],
                'logic_score': score,
                'suggested_use': '2nd-4th place exotic'
            })
    
    return sorted(contenders, key=lambda x: x['logic_score'], reverse=True)
```

### 4.2 Pace-Based Exotic Optimization

```python
def optimize_exotic_selections(ratings_df, pace_scenario):
    """
    Use pace scenario to predict who benefits in stretch.
    """
    A_group = []  # Win contenders
    B_group = []  # Place contenders  
    C_group = []  # Show+ contenders
    
    if pace_scenario == 'fast_pace':
        # Favor closers for 2nd-4th
        for horse in ratings_df.iterrows():
            if horse['Style'] in ['P', 'S'] and horse['late_pace_rating'] > 80:
                C_group.append(horse['Horse'])
    
    elif pace_scenario == 'slow_pace':
        # Front-runners hold, pressers for place
        for horse in ratings_df.iterrows():
            if horse['Style'] in ['E/P'] and horse['Quirin'] >= 4:
                B_group.append(horse['Horse'])
    
    elif pace_scenario == 'contested':
        # Mid-pack stalkers benefit
        for horse in ratings_df.iterrows():
            if horse['Style'] == 'E/P' and horse['post_position'] in [4,5,6]:
                B_group.append(horse['Horse'])
    
    return {'A': A_group, 'B': B_group, 'C': C_group}
```

---

## PART 5: BACKTEST FRAMEWORK

### 5.1 Simulation Code

```python
class RacingBacktester:
    def __init__(self, db_path='race_history.db'):
        self.db = RaceDatabase(db_path)
        self.results = {
            'win_accuracy': [],
            'place_accuracy': [],
            'show_accuracy': [],
            'fourth_accuracy': [],
            'top2_capture': [],
            'top3_capture': [],
            'top4_capture': []
        }
    
    def run_backtest(self, min_races=100):
        """Simulate predictions on historical data."""
        races = self.db.get_races(limit=min_races)
        
        for race in races:
            horses = self.db.get_horses_for_race(race['race_id'])
            
            # Simulate prediction
            predictions = self.model.predict(horses)
            
            # Get actual results
            actual_winner = [h for h in horses if h['actual_finish'] == 1][0]
            actual_place = [h for h in horses if h['actual_finish'] <= 2]
            actual_show = [h for h in horses if h['actual_finish'] <= 3]
            actual_fourth = [h for h in horses if h['actual_finish'] <= 4]
            
            # Top prediction (sorted by probability)
            predicted_win = predictions[0]
            predicted_top2 = predictions[:2]
            predicted_top3 = predictions[:3]
            predicted_top4 = predictions[:4]
            
            # Accuracy checks
            self.results['win_accuracy'].append(
                1 if predicted_win['horse_name'] == actual_winner['horse_name'] else 0
            )
            
            self.results['top2_capture'].append(
                1 if any(h['horse_name'] in [p['horse_name'] for p in predicted_top2] 
                        for h in actual_place) else 0
            )
            
            # ... similar for top3, top4
        
        return self.compile_results()
    
    def compile_results(self):
        """Generate accuracy report."""
        return {
            'Win Accuracy': f"{np.mean(self.results['win_accuracy']) * 100:.1f}%",
            'Top-2 Capture': f"{np.mean(self.results['top2_capture']) * 100:.1f}%",
            'Top-3 Capture': f"{np.mean(self.results['top3_capture']) * 100:.1f}%",
            'Top-4 Capture': f"{np.mean(self.results['top4_capture']) * 100:.1f}%"
        }
```

### 5.2 Expected Baseline vs. Target

| Metric | Current (Est.) | Target | Strategy |
|--------|---------------|--------|----------|
| Win Accuracy | 25-30% | **90%** | Multi-output NN + ensemble |
| Top-2 Capture (2 picks) | 40-50% | **85%** | Pace-based selection |
| Top-3 Capture (3 picks) | 60-70% | **90%** | Logical contender ID |
| Top-4 Capture (4 picks) | 75-80% | **95%** | Value overlay + closers |

---

## PART 6: IMPLEMENTATION ROADMAP

### Phase 3A: Data Collection (Week 1-2)
- [ ] Update database schema with 30 new columns
- [ ] Add parsing functions for jockey/trainer stats
- [ ] Implement weather data capture
- [ ] Build fractional pace parser
- [ ] Create trip rating system

### Phase 3B: Algorithm Upgrade (Week 3-4)
- [ ] Implement Multi-Output Neural Network
- [ ] Train separate winner/placer/closer models
- [ ] Build ensemble combiner
- [ ] Add late pace (Ckick) component
- [ ] Integrate human factors (Chuman)

### Phase 3C: Exotic Optimizer (Week 5)
- [ ] Pace scenario analyzer
- [ ] Logical contender identifier
- [ ] A/B/C/D group auto-builder
- [ ] Greedy optimization for trifecta/superfecta

### Phase 3D: Backtesting & Validation (Week 6)
- [ ] Collect 200+ historical races
- [ ] Run simulation on past data
- [ ] Generate accuracy reports
- [ ] Fine-tune model weights
- [ ] Achieve 90% win accuracy target

---

## PART 7: IMMEDIATE ACTION ITEMS

**TODAY:**
1. Update race_history.db schema with critical fields
2. Add jockey_name, trainer_name to save_horse_predictions()
3. Parse weather from BRISNET PP text

**THIS WEEK:**
1. Implement late_pace_rating calculation
2. Build trip_rating parser from race comments
3. Create multi_place_loss function
4. Start collecting historical race data (aim for 200 races)

**NEXT WEEK:**
1. Train Multi-Output Neural Network
2. Build ensemble model
3. Run first backtest
4. Iterate based on results

---

## CONCLUSION

**Current System Strengths:**
✅ Solid fundamental rating (class, speed, pace, style, post)
✅ Track bias integration
✅ Pedigree enhancements

**Critical Gaps:**
❌ Missing weather/track condition data
❌ No jockey/trainer metrics in ML
❌ Single-output model (win only, not place/show/fourth)
❌ No late pace/closing kick analysis
❌ Limited trip handicapping

**Path to 90% Accuracy:**
1. **Expand data collection** (weather, human factors, fractions)
2. **Multi-output architecture** (separate heads for each position)
3. **Ensemble approach** (combine winner + placer + closer models)
4. **Pace-based exotic optimization** (logical contender identification)
5. **Extensive backtesting** (200+ historical races for validation)

**Estimated Timeline:** 6 weeks to full implementation + validation

**Expected Outcome:**
- Win prediction: 90%+ accuracy
- Top-2 capture: 85%+ with 2 picks
- Top-3 capture: 90%+ with 3 picks
- Top-4 capture: 95%+ with 4 picks
