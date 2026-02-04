# Complete Race Type Classification System
## North American Thoroughbred Racing - Levels 1-7

### Overview
This document defines the comprehensive race type classification system used by our dynamic weighting engine. Every race is assigned a unique score (1.0-8.0) based on its type and level, with additional purse-based quality adjustments within each category.

**Key Principle**: Each horse entered into our system receives a weighted value calculation based on:
1. **Race Type Level (1-7)** - Base classification
2. **Race Type Score (1.0-8.0)** - Granular quality tier
3. **Purse Amount** - Quality scaling within same type
4. **Claiming Price** - For claiming races, price indicates tier
5. **Conditions** - For allowance races (NW1X, NW2X, NW3X)

---

## Level 1: MAIDEN RACES (1.0-3.0)
**Definition**: Horses that have never won a race

### 1.1 Maiden Special Weight (MSW)
- **Score**: 3.0
- **Abbreviations**: MSW, MD SP WT, MDN SP WT, Maiden Special Weight
- **Baseline Bonus**: +0.2
- **Notes**: Top-quality maidens expected to win soon; no claiming price

### 1.2 Maiden Claiming (MCL)
- **Score**: 1.0
- **Abbreviations**: MCL, MDN CLM, MD CLM, Maiden Claiming
- **Baseline Bonus**: -0.5
- **Notes**: Lowest tier; maidens with claiming price (poor quality)

### 1.3 Maiden Optional Claiming (MOC)
- **Score**: 2.0
- **Abbreviations**: MOC, Maiden Optional Claiming, MDN OC
- **Baseline Bonus**: 0.0
- **Notes**: Hybrid between MSW and MCL; optional claim

---

## Level 2: CLAIMING (1.5-3.2)
**Definition**: Horses for sale at set price; price indicates quality

### 2.1 Claiming (CLM) - Tiered by Price
**Abbreviations**: CLM, Claiming, CLM Price

#### Claiming Price Tiers:
- **CLM <$10,000**: Score 1.5, Baseline -0.2
- **CLM $10,000-$15,999**: Score 2.0, Baseline 0.0
- **CLM $16,000-$24,999**: Score 2.2, Baseline 0.0
- **CLM $25,000-$31,999**: Score 2.5, Baseline 0.0
- **CLM $32,000-$39,999**: Score 2.8, Baseline 0.0
- **CLM $40,000-$49,999**: Score 3.0, Baseline 0.0
- **CLM $50,000+**: Score 3.2, Baseline 0.0

**Extraction Examples**:
- "Clm25000" → $25,000 claiming → Score 2.5
- "Claiming $40,000" → Score 3.0
- "CLM 50000" → Score 3.2

### 2.2 Claiming Handicap (CLH)
- **Score**: 2.2
- **Abbreviations**: CLH, Claiming Handicap, CLM Handicap
- **Baseline Bonus**: 0.0
- **Notes**: Claiming with handicapped weights

### 2.3 Claiming Stakes (CST)
- **Score**: 3.5
- **Abbreviations**: CST, Claiming Stakes, CLM Stakes
- **Baseline Bonus**: +0.3
- **Notes**: Bridges Level 2 and Level 7; claiming with stakes prestige

---

## Level 3: STARTER (3.5-3.8)
**Definition**: For horses from recent claiming races; no claiming price

### 3.1 Starter Allowance (STR/STA)
- **Score**: 3.5
- **Abbreviations**: STR, STA, Starter Allowance, STR ALW
- **Baseline Bonus**: +0.3
- **Notes**: Restricted to horses claimed or eligible to be claimed

### 3.2 Starter Handicap (SHP)
- **Score**: 3.6
- **Abbreviations**: SHP, Starter Handicap, STR HCP
- **Baseline Bonus**: +0.3
- **Notes**: Starter with handicapped weights

### 3.3 Starter Optional Claiming (SOC)
- **Score**: 3.8
- **Abbreviations**: SOC, Starter Optional Claiming, STR OC
- **Baseline Bonus**: +0.3
- **Notes**: Starter with optional claiming price

---

## Level 4: ALLOWANCE (4.0-4.5)
**Definition**: Non-selling condition races; quality tier above claiming

### 4.1 Standard Allowance (ALW)
- **Score**: 4.0
- **Abbreviations**: ALW, Allowance
- **Baseline Bonus**: +0.5
- **Notes**: Open allowance; no specific restrictions

### 4.2 Restricted Allowances (Condition-Based)
#### NW1X (Non-Winners of 1 Other Than)
- **Score**: 4.0
- **Abbreviations**: NW1X, Allowance NW1X, ALW NW1X
- **Baseline Bonus**: +0.5
- **Notes**: Restricted to horses with 0-1 wins (excluding maiden/claiming)

#### NW2X (Non-Winners of 2 Other Than)
- **Score**: 4.2
- **Abbreviations**: NW2X, Allowance NW2X, ALW NW2X
- **Baseline Bonus**: +0.5
- **Notes**: Restricted to horses with 0-2 wins

#### NW3X (Non-Winners of 3 Other Than)
- **Score**: 4.5
- **Abbreviations**: NW3X, Allowance NW3X, ALW NW3X
- **Baseline Bonus**: +1.0
- **Notes**: Top restricted allowance; horses with 0-3 wins

---

## Level 5: ALLOWANCE OPTIONAL CLAIMING (4.5-5.0)
**Definition**: Allowance with optional claiming price; hybrid quality

### 5.1 Allowance Optional Claiming (AOC/OC/OCL)
**Abbreviations**: AOC, OC, OCL, Optional Claiming, Allowance Optional Claiming

#### Price Tiers:
- **OC <$32,000**: Score 4.5, Baseline +1.0
- **OC $32,000-$39,999**: Score 4.7, Baseline +1.0
- **OC $40,000-$49,999**: Score 4.8, Baseline +1.0
- **OC $50,000+**: Score 5.0, Baseline +1.5

**Example**: "OC 25000 NW1X" → Optional claiming $25k with NW1X condition

---

## Level 6: HANDICAP (5.0-5.5)
**Definition**: Weights assigned to equalize chances; mid-to-high level

### 6.1 Optional Claiming Handicap (OCH)
- **Score**: 5.0
- **Abbreviations**: OCH, Optional Claiming Handicap, OC HCP
- **Baseline Bonus**: +1.5
- **Notes**: Handicap with optional claiming price

### 6.2 Standard Handicap (HCP)
- **Score**: 5.5
- **Abbreviations**: HCP, Handicap, H
- **Baseline Bonus**: +1.8
- **Notes**: Weights assigned by racing secretary to equalize chances

---

## Level 7: STAKES & GRADED (5.0-8.0)
**Definition**: High-purse prestigious races; top-tier competition

### 7.1 Stakes (Non-Graded) (STK/S)
- **Score**: 5.0
- **Abbreviations**: STK, S, Stakes, Stake
- **Baseline Bonus**: +1.5
- **Purse Range**: Typically $75,000+
- **Notes**: Open stakes; not graded by industry

### 7.2 Listed Stakes (LST)
- **Score**: 5.2
- **Abbreviations**: Listed, Listed Stakes, LST
- **Baseline Bonus**: +1.6
- **Notes**: Below graded but above regular stakes

### 7.3 Grade 3 Stakes (G3)
- **Score**: 6.0
- **Abbreviations**: G3, Grade 3, Grade III, GRIII, G3 Stakes
- **Baseline Bonus**: +2.0
- **Purse Range**: Typically $100,000+
- **Examples**: Sham Stakes, Robert B. Lewis Stakes
- **Notes**: Entry-level graded stakes

### 7.4 Grade 2 Stakes (G2)
- **Score**: 7.0
- **Abbreviations**: G2, Grade 2, Grade II, GRII, G2 Stakes
- **Baseline Bonus**: +2.5
- **Purse Range**: Typically $200,000+
- **Examples**: Santa Anita Handicap, San Felipe Stakes
- **Notes**: Mid-tier graded stakes

### 7.5 Grade 1 Stakes (G1)
- **Score**: 8.0
- **Abbreviations**: G1, Grade 1, Grade I, GRI, G1 Stakes
- **Baseline Bonus**: +3.0
- **Purse Range**: Typically $300,000+
- **Examples**: Kentucky Derby, Pegasus World Cup, Breeders' Cup Classic
- **Notes**: Elite championship caliber; highest classification

---

## International Equivalents

### Group System (European/International)
- **Group 1**: Score 8.0 (equivalent to G1)
- **Group 2**: Score 7.0 (equivalent to G2)
- **Group 3**: Score 6.0 (equivalent to G3)
- **Abbreviations**: Group 1, GR1, Group 2, GR2, Group 3, GR3

---

## Special Race Types

### Waiver Claiming (WCL)
- **Score**: 2.2
- **Abbreviations**: WCL, Waiver Claiming, Waiver
- **Baseline Bonus**: 0.0
- **Notes**: Special claiming conditions

### Trial Races
- **Score**: 4.8
- **Abbreviations**: Trial
- **Baseline Bonus**: +1.0
- **Notes**: Qualifying races for major events

### Futurity
- **Score**: 5.5
- **Abbreviations**: Futurity
- **Baseline Bonus**: +1.8
- **Notes**: Special 2-year-old stakes

### Derby (Non-Graded)
- **Score**: 6.5
- **Abbreviations**: Derby
- **Baseline Bonus**: +2.0
- **Notes**: Prestigious 3-year-old events (if not graded separately)

---

## Purse-Based Quality Adjustment

### Why Purse Matters
Within the same race type, higher purse indicates better quality horses. Our system applies additional bonuses based on purse relative to baseline for that level.

### Purse Scaling by Level

#### Graded Stakes (G1/G2/G3 - Score ≥6.0)
- **Baseline**: $150,000
- **Scaling**: +0.3 per $300k above baseline
- **Max Bonus**: +1.5
- **Example**: $1M G1 → base +3.0, purse +0.85 = +3.85 total

#### Stakes/Handicap (Score 5.0-5.5)
- **Baseline**: $75,000
- **Scaling**: +0.2 per $150k above baseline
- **Max Bonus**: +1.0
- **Example**: $200k Stakes → base +1.5, purse +0.17 = +1.67 total

#### Allowance/AOC (Score 4.0-5.0)
- **Baseline**: $40,000
- **Scaling**: +0.15 per $80k above baseline
- **Max Bonus**: +0.8
- **Example**: $80k Allowance → base +0.5, purse +0.075 = +0.575 total

#### Claiming (Score 1.5-3.2)
- **Baseline**: $15,000
- **Scaling**: +0.1 per $30k above baseline
- **Max Bonus**: +0.5
- **Example**: $35k CLM25000 → base 0.0, purse +0.067 = +0.067 total

---

## Dynamic Weight Adjustment by Race Type

### How It Works
Once race type and score are determined, the system applies level-specific weight modifiers to emphasize different handicapping components.

### Weight Profiles

#### Grade 1/Grade 2 Races (Scores 7.0-8.0)
**Philosophy**: Elite races emphasize class and speed; best horses rise to top
- **Class**: +20% (1.2× base 2.5 = 3.0)
- **Speed**: +30% (1.3× base 2.0 = 2.6)
- **Form**: Standard (1.0× base 1.8 = 1.8)
- **Pace**: -10% (0.9× base 1.5 = 1.35)
- **Style**: +10% (1.1× base 2.0 = 2.2)
- **Post**: Standard (1.0× base 0.8 = 0.8)

#### Grade 3 Stakes (Score 6.0)
**Philosophy**: Quality stakes emphasize class and speed moderately
- **Class**: +10% (2.75)
- **Speed**: +20% (2.4)
- **Form**: Standard (1.8)
- **Pace**: Standard (1.5)
- **Style**: Standard (2.0)
- **Post**: Standard (0.8)

#### Allowance Races (Scores 4.0-5.0)
**Philosophy**: Balanced handicapping; all factors matter equally with slight speed emphasis
- **Class**: Standard (2.5)
- **Speed**: +10% (2.2)
- **Form**: +10% (1.98)
- **Pace**: +10% (1.65)
- **Style**: Standard (2.0)
- **Post**: Standard (0.8)

#### Maiden Races (Scores 1.0-3.0)
**Philosophy**: Unreliable form history; emphasize pace scenario and running style
- **Class**: -20% (2.0)
- **Speed**: -10% (1.8)
- **Form**: -30% (1.26) ← Form unreliable for maidens
- **Pace**: +20% (1.8) ← Pace scenario critical
- **Style**: +10% (2.2) ← Running style important
- **Post**: Standard (0.8)

#### Claiming Races (Scores 1.5-3.2)
**Philosophy**: Current form and pace setup matter most; class already factored via price
- **Class**: Standard (2.5)
- **Speed**: Standard (2.0)
- **Form**: +30% (2.34) ← Recent form critical
- **Pace**: +20% (1.8) ← Pace setup important
- **Style**: Standard (2.0)
- **Post**: -10% (0.72) ← Post less important

---

## Examples

### Example 1: Pegasus World Cup Invitational - G1
**Input**: Race Type = "Grade 1 Stakes", Purse = $3,000,000

**Calculation**:
1. **Type Detection**: "Grade 1 Stakes" → substring match "grade 1" → Score 8.0
2. **Baseline Bonus**: Score ≥8.0 → +3.0
3. **Purse Bonus**: $3M vs $150k baseline → ($3M - $150k) / $300k × 0.3 = +2.85 (capped at +1.5) = +1.5
4. **Total Baseline**: 3.0 + 1.5 = **4.5**
5. **Dynamic Weights**: Grade_1_2 profile applied
   - Class: 3.0 (vs 2.5 base)
   - Speed: 2.6 (vs 2.0 base)
   - Pace: 1.35 (vs 1.5 base)

### Example 2: $25,000 Claiming Race
**Input**: Race Type = "Clm25000", Purse = $22,000

**Calculation**:
1. **Type Detection**: "Clm25000" → extract claiming price $25,000 → Score 2.5
2. **Baseline Bonus**: Score 2.5 → +0.0
3. **Purse Bonus**: $22k vs $15k baseline → ($22k - $15k) / $30k × 0.1 = +0.023
4. **Total Baseline**: 0.0 + 0.023 = **0.023**
5. **Dynamic Weights**: Claiming profile applied
   - Form: 2.34 (vs 1.8 base) ← Emphasis on current form
   - Pace: 1.8 (vs 1.5 base) ← Pace setup critical

### Example 3: Allowance NW2X
**Input**: Race Type = "Allowance NW2X", Purse = $55,000

**Calculation**:
1. **Type Detection**: Exact match "nw2x" → Score 4.2
2. **Baseline Bonus**: Score 4.2 → +0.5
3. **Purse Bonus**: $55k vs $40k baseline → ($55k - $40k) / $80k × 0.15 = +0.028
4. **Total Baseline**: 0.5 + 0.028 = **0.528**
5. **Dynamic Weights**: Allowance profile applied
   - Balanced emphasis across all components (+10% speed/form/pace)

### Example 4: Maiden Special Weight
**Input**: Race Type = "MSW", Purse = $40,000

**Calculation**:
1. **Type Detection**: Exact match "msw" → Score 3.0
2. **Baseline Bonus**: Score 3.0 → +0.2
3. **Purse Bonus**: Maidens use claiming baseline ($15k) → ($40k - $15k) / $30k × 0.1 = +0.083
4. **Total Baseline**: 0.2 + 0.083 = **0.283**
5. **Dynamic Weights**: Maiden profile applied
   - Form: 1.26 (vs 1.8 base) ← De-emphasized (unreliable)
   - Pace: 1.8 (vs 1.5 base) ← Emphasized (pace scenario matters)
   - Style: 2.2 (vs 2.0 base) ← Emphasized (running style important)

---

## Technical Implementation

### Race Type Detection Algorithm
```python
def detect_race_type(race_type_string, purse_amount):
    # Step 1: Normalize input
    normalized = race_type_string.lower().strip()
    
    # Step 2: Try exact match (fastest)
    if normalized in RACE_TYPE_SCORES:
        score = RACE_TYPE_SCORES[normalized]
    
    # Step 3: Intelligent substring matching
    else:
        score = parse_complex_format(normalized)
    
    # Step 4: Extract claiming price if applicable
    if 'clm' in normalized or 'claiming' in normalized:
        claiming_price = extract_claiming_price(normalized)
        if claiming_price > 0:
            score = calculate_claiming_score(claiming_price)
    
    # Step 5: Apply baseline bonus
    baseline = get_baseline_bonus(score)
    
    # Step 6: Apply purse-based quality adjustment
    purse_bonus = calculate_purse_bonus(score, purse_amount)
    
    # Step 7: Select dynamic weight profile
    weights = get_dynamic_weights(score, normalized)
    
    return score, baseline, purse_bonus, weights
```

### Claiming Price Extraction
```python
def extract_claiming_price(race_type_string):
    # Pattern 1: "Clm25000" format
    match = re.search(r'clm[^\d]*(\d+)', race_type_string)
    if match:
        return int(match.group(1))
    
    # Pattern 2: "Claiming $25,000" format
    match = re.search(r'\$[\d,]+', race_type_string)
    if match:
        return int(match.group(0).replace('$', '').replace(',', ''))
    
    return 0
```

---

## Summary

### Complete Coverage
Our system recognizes **70+ race type variations** across all 7 classification levels, ensuring every horse entered receives an accurate weighted value based on:
- Race type abbreviation (MSW, CLM, G1, etc.)
- Claiming price (for CLM races)
- Purse amount (quality scaling)
- Allowance conditions (NW1X, NW2X, NW3X)
- Dynamic weight emphasis appropriate for that level

### Adaptive Intelligence
The system "thinks differently" for each race type:
- **Grade 1**: Emphasizes class and speed (best horses)
- **Claiming**: Emphasizes current form and pace (recent performance)
- **Maiden**: Emphasizes pace and style, de-emphasizes form (unreliable history)
- **Allowance**: Balanced approach (all factors equal)

### Unique Calculations
**Every single race that is input into system will be different** because:
1. Race type varies (MSW vs G1 vs CLM25000)
2. Purse amount varies ($20k vs $3M)
3. Claiming price varies ($10k vs $50k)
4. Conditions vary (NW1X vs NW3X)
5. Dynamic weights vary by profile

This ensures **granular, accurate weighted values** for precise handicapping at Bill Benter standards.

---

*Last Updated: February 4, 2026*  
*System Version: 2.0 - Comprehensive Classification*
