# 📊 Angle Weights & Code Reference

## Optimized Angle Weights Table

### Core Rating Components

| Component | Symbol | Weight | Range | Formula | Priority |
|-----------|--------|--------|-------|---------|----------|
| **Class** | `Cclass` | **2.5** | -2 to +12 | base_hierarchy + log(purse) + movement + ped | ⭐⭐⭐⭐⭐ |
| **Speed** | `Cspeed` | **2.0** | -4 to +4 | (beyer-80)/15 weighted by recency | ⭐⭐⭐⭐⭐ |
| **Pace** | `Cpace` | **1.5** | -3 to +3 | PPI-derived style advantage | ⭐⭐⭐⭐ |
| **Style Bias** | `Cstyle` | **1.2** | -0.5 to +0.7 | Running style × track bias | ⭐⭐⭐ |
| **Post Position** | `Cpost` | **0.8** | 0 to +0.25 | Position × distance/surface bias | ⭐⭐ |
| **Angles** | `boost` | **varied** | 0 to +3 | Pattern-based additions | ⭐⭐⭐⭐ |
| **Pedigree** | `ped_boost` | **varied** | 0 to +1.2 | Surface/distance aptitude | ⭐⭐⭐ |

### Composite Rating Formula
```
R = (Cclass × 2.5) + (Cspeed × 2.0) + (Cpace × 1.5) + 
    (Cstyle × 1.2) + (Cpost × 0.8) + angle_boosts + pedigree_boosts
```

---

## Detailed Angle Boosts

### 1. ROI-Based Angles (Historical Performance)

| Angle ROI | Starts Required | Boost | Confidence |
|-----------|-----------------|-------|------------|
| **> 2.0** | 20+ | +0.6 | Very High |
| **1.5-2.0** | 15+ | +0.4 | High |
| **1.2-1.5** | 10+ | +0.2 | Medium |
| **< 1.2** | any | +0.0 | Ignore |

**Code:**
```python
strong_angles = angles_df[(angles_df["ROI"] > 1.5) & (angles_df["Starts"] >= 10)]
boost += len(strong_angles) * 0.4
```

### 2. Win%-Based Angles

| Win % | Starts Required | Boost | Use Case |
|-------|-----------------|-------|----------|
| **> 35%** | 20+ | +0.8 | Elite pattern |
| **30-35%** | 15+ | +0.6 | Strong pattern |
| **25-30%** | 15+ | +0.3 | Good pattern |
| **< 25%** | any | +0.0 | Ignore |

**Code:**
```python
hot_angles = angles_df[(angles_df["Win%"] > 25) & (angles_df["Starts"] >= 15)]
boost += len(hot_angles) * 0.3
```

### 3. Specific Power Angles

| Angle Category | Threshold | Boost | Notes |
|----------------|-----------|-------|-------|
| **1st Time Starter** | Win% > 20% | +0.6 | Trainer debut record |
| **Debut MdnSpWt** | Win% > 18% | +0.5 | Maiden special weight debut |
| **2nd Career Race** | Win% > 22% | +0.5 | Follow-up to debut |
| **Turf to Dirt** | Win% > 25% | +0.7 | Surface switch |
| **Dirt to Turf** | Win% > 25% | +0.7 | Surface switch |
| **Shipper** | Win% > 22% | +0.5 | Track change |
| **Blinkers On** | Win% > 24% | +0.6 | Equipment change |
| **Blinkers Off** | Win% > 20% | +0.4 | Equipment change |
| **30-60 Days Away** | Win% > 18% | +0.3 | Layoff pattern |
| **Jockey w/ Trainer L30** | Win% > 30% | +0.8 | Hot combo |
| **Jockey w/ E types** | Win% > 28% | +0.6 | Jockey + style match |

**Code:**
```python
for _, angle_row in angles_df.iterrows():
    cat = angle_row["Category"].lower()
    win_pct = angle_row["Win%"]
    starts = angle_row["Starts"]
    
    if "1st time" in cat or "debut" in cat:
        if win_pct > 20 and starts >= 10:
            boost += 0.6
    
    elif "jky" in cat and "trn" in cat:  # Trainer/Jockey combo
        if win_pct > 30 and starts >= 15:
            boost += 0.8
    
    elif "shipper" in cat:
        if win_pct > 22 and starts >= 12:
            boost += 0.5
    
    elif "blinkers on" in cat:
        if win_pct > 24 and starts >= 10:
            boost += 0.6
    
    elif "turf to dirt" in cat or "dirt to turf" in cat:
        if win_pct > 25 and starts >= 10:
            boost += 0.7
```

---

## Pedigree Weights

### Surface-Specific

| Surface | Metric | Threshold | Boost | Logic |
|---------|--------|-----------|-------|-------|
| **Turf** | Dam DPI | > 2.0 | +0.7 | Strong turf pedigree |
| **Turf** | Dam DPI | > 1.5 | +0.4 | Good turf pedigree |
| **Turf** | Sire AWD | > 1.3 | +0.3 | Sire turf record |
| **Synthetic** | Dam DPI | > 1.8 | +0.5 | Synthetic aptitude |

**Code:**
```python
if surface_type.lower() == "turf":
    dam_dpi = pedigree.get("dam_dpi", 0)
    if dam_dpi > 2.0:
        boost += 0.7
    elif dam_dpi > 1.5:
        boost += 0.4
```

### Distance-Specific

| Distance | Metric | Threshold | Boost | Logic |
|----------|--------|-----------|-------|-------|
| **Route (1+ mile)** | Damsire AWD | > 1.5 | +0.7 | Stamina pedigree |
| **Route (1+ mile)** | Damsire AWD | > 1.3 | +0.5 | Good stamina |
| **Route (1+ mile)** | Dam DPI | > 1.8 | +0.4 | Dam route success |
| **Sprint (< 1 mile)** | Sire 1st% | > 20% | +0.3 | Speed pedigree |

**Code:**
```python
is_route = any(x in distance_txt.lower() for x in ["mile", "1 1/", "1 3/", "1 5/"])
if is_route:
    damsire_awd = pedigree.get("damsire_awd", 0)
    if damsire_awd > 1.5:
        boost += 0.7
    elif damsire_awd > 1.3:
        boost += 0.5
```

### Stakes Races

| Race Type | Metric | Threshold | Boost | Logic |
|-----------|--------|-----------|-------|-------|
| **Stakes (any)** | Sire AWD | > 1.5 | +0.5 | Quality pedigree |
| **Stakes (any)** | Sire AWD | > 1.2 | +0.3 | Good pedigree |
| **Grade 1/2** | Sire 1st% | > 15% | +0.4 | Elite sire |

**Code:**
```python
if race_type.startswith("Stakes"):
    sire_awd = pedigree.get("sire_awd", 0)
    if sire_awd > 1.5:
        ped_boost += 0.5
    elif sire_awd > 1.2:
        ped_boost += 0.3
```

---

## Class Calculation Details

### Race Type Hierarchy

| Race Type | Base Value | Purse Range (typical) | Notes |
|-----------|------------|------------------------|-------|
| **Stakes (G1)** | 10.0 | $300K-$1M+ | Highest class |
| **Stakes (G2)** | 8.5 | $200K-$600K | Elite |
| **Stakes (G3)** | 7.0 | $150K-$400K | Top tier |
| **Stakes (Listed)** | 6.0 | $100K-$250K | Quality stakes |
| **Stakes** | 5.0 | $75K-$200K | Open stakes |
| **Allowance** | 3.5 | $40K-$100K | Restricted |
| **Mdn Sp Wt** | 2.5 | $30K-$80K | Maiden special |
| **Maiden (other)** | 2.0 | $20K-$50K | Other maiden |
| **Mdn Claiming** | 1.5 | $15K-$35K | Claiming maiden |
| **Other** | 2.0 | varies | Catch-all |

### Purse Factor (Logarithmic)

```python
purse_factor = log10(purse) - 4.0

# Examples:
$10,000  → log10(10000) - 4.0 = 0.0  (baseline)
$50,000  → log10(50000) - 4.0 = +0.7
$100,000 → log10(100000) - 4.0 = +1.0
$500,000 → log10(500000) - 4.0 = +1.7
```

### Class Movement Adjustments

| Scenario | Purse Ratio | Adjustment | Meaning |
|----------|-------------|------------|---------|
| **Big Drop** | Recent > Current × 1.3 | +1.5 | Major class relief |
| **Moderate Drop** | Recent > Current × 1.1 | +0.8 | Class relief |
| **Neutral** | Recent ≈ Current | +0.0 | Same level |
| **Moderate Rise** | Recent < Current × 0.9 | -0.5 | Stepping up |
| **Big Rise** | Recent < Current × 0.7 | -1.2 | Major step up |

**Code:**
```python
recent_purses = []  # Extracted from PP block
avg_recent = np.mean(recent_purses[:3])

if avg_recent > purse * 1.3:
    class_adj = +1.5  # Dropping down significantly
elif avg_recent > purse * 1.1:
    class_adj = +0.8  # Moderate drop
elif avg_recent < purse * 0.7:
    class_adj = -1.2  # Major step up (negative)
elif avg_recent < purse * 0.9:
    class_adj = -0.5  # Minor step up
else:
    class_adj = 0.0   # Neutral
```

---

## Speed Figure Calculation

### Beyer Normalization

```python
# Normalize to -2 to +2 scale (80 = average)
cspeed_last = (last_beyer - 80) / 15.0
cspeed_avg = (avg_beyer - 80) / 15.0
cspeed_top = (top_beyer - 80) / 20.0
cspeed_trend = beyer_trend / 10.0

# Weighted combination
Cspeed = (cspeed_last × 0.50) +    # Most recent (50%)
         (cspeed_avg × 0.30) +      # Average (30%)
         (cspeed_top × 0.15) +      # Peak ability (15%)
         (cspeed_trend × 0.05)      # Trend (5%)
```

### Beyer Range Examples

| Beyer | Normalized | Weighted (×2.0) | Class Level |
|-------|------------|-----------------|-------------|
| **105** | +1.67 | **+3.34** | Elite |
| **95** | +1.00 | **+2.00** | Strong |
| **85** | +0.33 | **+0.67** | Above avg |
| **80** | 0.00 | **0.00** | Average |
| **75** | -0.33 | **-0.67** | Below avg |
| **65** | -1.00 | **-2.00** | Weak |
| **55** | -1.67 | **-3.34** | Very weak |

---

## Pace Calculation (PPI-Based)

### PPI Formula
```python
E_count = count of E styles
EP_count = count of E/P styles
P_count = count of P styles
S_count = count of S styles

PPI = (E_count + EP_count - P_count - S_count) × 10 / field_size
```

### PPI Interpretation

| PPI Value | Pace Scenario | Who Benefits | Cpace Boost |
|-----------|---------------|--------------|-------------|
| **+4 to +6** | Extreme speed | Closers (P/S) | +1.5 to +2.0 |
| **+2 to +4** | Fast pace | Closers (P/S) | +0.8 to +1.5 |
| **-1 to +2** | Moderate | Tactical (E/P) | +0.2 to +0.8 |
| **-2 to -1** | Slow pace | Speed (E) | -0.3 to +0.2 |
| **-4 to -2** | Very slow | Speed (E) | -0.8 to -0.3 |

### Style-Specific Adjustments

```python
# Per-horse pace advantage
if style in ("E", "E/P"):
    cpace = +0.6 × style_strength × PPI
elif style == "S":
    cpace = -0.6 × style_strength × PPI
else:
    cpace = 0.0
```

**Style Strength Multipliers:**
- Strong: 1.0
- Solid: 0.8
- Slight: 0.5
- Weak: 0.3

---

## Running Style Bias Adjustments

### Speed Favoring Track

| Style | Cstyle Adjustment | Logic |
|-------|-------------------|-------|
| **E** | +0.70 | Speed holds up |
| **E/P** | +0.50 | Tactical speed wins |
| **P** | -0.20 | Closers struggle |
| **S** | -0.50 | Late closers fail |

### Closer Favoring Track

| Style | Cstyle Adjustment | Logic |
|-------|-------------------|-------|
| **E** | -0.50 | Speed tires |
| **E/P** | -0.20 | Tactical disadvantaged |
| **P** | +0.25 | Pressers benefit |
| **S** | +0.50 | Closers thrive |

### Neutral

| Style | Cstyle Adjustment |
|-------|-------------------|
| All | 0.0 |

---

## Post Position Bias

### Sprints (< 1 Mile)

| Bias Type | Posts Favored | Cpost Boost | Typical Tracks |
|-----------|---------------|-------------|----------------|
| **Favors Rail** | 1 | +0.25 | Santa Anita, Del Mar |
| **1-3** | 1-3 | +0.25 | Most 1-turn tracks |
| **4-7** | 4-7 | +0.25 | Churchill (dirt) |
| **8+** | 8+ | +0.25 | Rare, wide tracks |
| **None** | All | 0.0 | Fair tracks |

### Routes (1+ Mile)

| Bias Type | Posts Favored | Cpost Boost | Notes |
|-----------|---------------|-------------|-------|
| **Favors Rail** | 1 | +0.25 | Inside saves ground |
| **1-3** | 1-3 | +0.25 | Position advantage |
| **None** | All | 0.0 | Usually neutral |

---

## Adaptive Probability (Tau Adjustment)

### Field Quality Assessment

```python
rating_spread = max(ratings) - mean(ratings)

if rating_spread > 8:
    tau = 0.65      # Clear standout (sharper probs)
elif rating_spread > 5:
    tau = 0.75      # Strong favorite
elif rating_spread < 2:
    tau = 1.1       # Wide open (flatter probs)
else:
    tau = 0.85      # Default
```

### Field Size Adjustment

```python
# Larger fields = more uncertainty
tau *= (1.0 + (field_size - 8) × 0.02)

# Examples:
6-horse field: tau × 0.96  (sharper)
8-horse field: tau × 1.00  (baseline)
10-horse field: tau × 1.04 (flatter)
12-horse field: tau × 1.08 (flatter)
```

### Resulting Probability Distributions

| Scenario | Top Horse % | 2nd Place % | 3rd Place % | Notes |
|----------|-------------|-------------|-------------|-------|
| **Clear Standout** | 45-55% | 15-20% | 8-12% | Chalk race |
| **Strong Favorite** | 35-45% | 18-22% | 10-14% | Likely favorite |
| **Competitive** | 25-35% | 20-25% | 12-16% | Several contenders |
| **Wide Open** | 15-25% | 15-20% | 12-16% | Anyone can win |

---

## Code Snippets

### Complete Rating Calculation

```python
def compute_enhanced_rating(horse_data, race_context):
    """
    Comprehensive rating calculation.
    
    Args:
        horse_data: dict with parsed horse info
        race_context: dict with race conditions
    
    Returns:
        float: Final rating
    """
    # 1. Class (weight: 2.5)
    cclass = calculate_class_rating(
        purse=race_context["purse"],
        race_type=race_context["race_type"],
        horse_block=horse_data["pp_block"],
        pedigree=horse_data["pedigree"]
    )
    
    # 2. Speed (weight: 2.0)
    beyer_data = parse_beyer_figures(horse_data["pp_block"])
    cspeed = calculate_speed_rating(beyer_data)
    
    # 3. Pace (weight: 1.5)
    cpace = calculate_pace_advantage(
        style=horse_data["style"],
        ppi=race_context["ppi"],
        style_strength=horse_data["style_strength"]
    )
    
    # 4. Style Bias (weight: 1.2)
    cstyle = calculate_style_bias(
        style=horse_data["style"],
        bias_type=race_context["running_style_bias"]
    )
    
    # 5. Post (weight: 0.8)
    cpost = calculate_post_bias(
        post=horse_data["post"],
        bias_type=race_context["post_bias"],
        distance=race_context["distance"]
    )
    
    # 6. Angles (varied)
    angle_boost = calculate_angle_boosts(
        angles_df=horse_data["angles"],
        race_context=race_context
    )
    
    # 7. Pedigree (varied)
    ped_boost = calculate_pedigree_boost(
        pedigree=horse_data["pedigree"],
        surface=race_context["surface"],
        distance=race_context["distance"],
        race_type=race_context["race_type"]
    )
    
    # Composite rating
    rating = (cclass * 2.5 + 
              cspeed * 2.0 + 
              cpace * 1.5 + 
              cstyle * 1.2 + 
              cpost * 0.8 + 
              angle_boost + 
              ped_boost)
    
    return round(rating, 2)
```

### Probability to American Odds

```python
def prob_to_american_odds(prob):
    """Convert probability to American odds format."""
    if prob <= 0:
        return "+∞"
    if prob >= 1:
        return "-∞"
    
    decimal = 1.0 / prob
    if decimal >= 2.0:
        american = (decimal - 1) * 100
        return f"+{int(american)}"
    else:
        american = -100 / (decimal - 1)
        return f"{int(american)}"

# Examples:
prob_to_american_odds(0.40)  # "+150"  (2.5:1)
prob_to_american_odds(0.60)  # "-150"  (1:1.5)
prob_to_american_odds(0.25)  # "+300"  (3:1)
```

### Multi-Position Prediction

```python
def predict_finishing_order(ratings, horse_names):
    """
    Predict top 4 finishers with probabilities.
    
    Returns:
        dict with winner, place, show, superfecta contenders
    """
    # Get adaptive probabilities
    probs = adaptive_probabilities(ratings, len(ratings))
    sorted_idx = np.argsort(-probs)
    
    result = {
        "winner": {
            "horse": horse_names[sorted_idx[0]],
            "prob": probs[sorted_idx[0]],
            "confidence": get_confidence_level(probs[sorted_idx[0]])
        },
        "place_contenders": [],
        "show_contenders": [],
        "superfecta_contenders": []
    }
    
    # 2nd place (conditional)
    exclude = {sorted_idx[0]}
    probs_2nd = conditional_probs(probs, exclude)
    sorted_2nd = np.argsort(-probs_2nd)
    
    for i in range(min(3, len(sorted_2nd))):
        if probs_2nd[sorted_2nd[i]] > 0.05:
            result["place_contenders"].append({
                "horse": horse_names[sorted_2nd[i]],
                "prob": probs_2nd[sorted_2nd[i]]
            })
    
    # 3rd place (conditional on top 2)
    exclude.add(sorted_2nd[0])
    probs_3rd = conditional_probs(probs, exclude)
    sorted_3rd = np.argsort(-probs_3rd)
    
    for i in range(min(3, len(sorted_3rd))):
        if probs_3rd[sorted_3rd[i]] > 0.03:
            result["show_contenders"].append({
                "horse": horse_names[sorted_3rd[i]],
                "prob": probs_3rd[sorted_3rd[i]]
            })
    
    # 4th place (conditional on top 3)
    exclude.add(sorted_3rd[0])
    probs_4th = conditional_probs(probs, exclude)
    sorted_4th = np.argsort(-probs_4th)
    
    for i in range(min(3, len(sorted_4th))):
        if probs_4th[sorted_4th[i]] > 0.03:
            result["superfecta_contenders"].append({
                "horse": horse_names[sorted_4th[i]],
                "prob": probs_4th[sorted_4th[i]]
            })
    
    return result

def conditional_probs(probs, exclude_indices):
    """Calculate conditional probabilities excluding certain horses."""
    cond = np.array([p if i not in exclude_indices else 0 
                     for i, p in enumerate(probs)])
    return cond / cond.sum()

def get_confidence_level(prob):
    """Convert probability to confidence level."""
    if prob > 0.35:
        return "High"
    elif prob > 0.20:
        return "Medium"
    else:
        return "Low"
```

---

## Testing & Validation

### Unit Tests

```python
def test_class_calculation():
    # Test class drop
    assert calculate_class_rating(
        purse=50000, 
        race_type="Allowance",
        horse_block="$120K $110K $100K",
        pedigree={}
    ) > 5.0  # Should show advantage
    
    # Test class rise
    assert calculate_class_rating(
        purse=150000,
        race_type="Stakes (G2)",
        horse_block="$40K $35K $40K",
        pedigree={}
    ) < 7.0  # Should show disadvantage

def test_speed_calculation():
    block_with_beyers = "Last 5: 95 92 88 85 90"
    beyers = parse_beyer_figures(block_with_beyers)
    assert beyers["last_beyer"] == 95
    assert 88 <= beyers["avg_beyer"] <= 92

def test_probability_sum():
    ratings = np.array([15.0, 12.0, 10.0, 8.0])
    probs = adaptive_probabilities(ratings, len(ratings))
    assert abs(probs.sum() - 1.0) < 0.001  # Must sum to 1
```

---

## Performance Benchmarks

### Target Accuracy (500+ race sample)

| Metric | Target | Good | Excellent |
|--------|--------|------|-----------|
| **Win Prediction** | 85% | 88% | 90%+ |
| **Top 2 Prediction** | 70% | 75% | 80%+ |
| **Top 3 Prediction** | 60% | 65% | 70%+ |
| **Exacta Coverage** | 30% | 35% | 40%+ |
| **Trifecta Coverage** | 15% | 20% | 25%+ |

### ROI Expectations

| Bet Type | Conservative | Moderate | Aggressive |
|----------|--------------|----------|------------|
| **Win** | +8% | +12% | +18% |
| **Place** | +5% | +8% | +12% |
| **Exacta** | +10% | +15% | +22% |
| **Trifecta** | +15% | +25% | +35% |

---

**Version:** 2.0  
**Last Updated:** January 27, 2026
