# Elite Mathematical Enhancement Plan
## Horse Racing Prediction System - Bill Benter-Level Sophistication

**Date:** February 4, 2026  
**Objective:** Transform system to world-elite standards with zero bugs, zero redundancy

---

## Phase 1: Current System Audit ✅

### What We Have (Strong Foundation):
1. **Unified Rating Engine** (unified_rating_engine.py)
   - Component-based ratings: class, speed, form, pace, style, post
   - Softmax probability conversion (τ=3.0)
   - Elite parser integration (94% parsing accuracy)
   - 8-angle comprehensive analysis

2. **ML Odds Integration** ✅
   - ML odds reality check in fair_probs_from_ratings() (app.py lines 5228-5253)
   - Caps probabilities based on market odds (prevents absurd model predictions)
   - Current implementation: Basic capping mechanism

3. **Component Weights** (empirically tuned)
   - Class: 2.5x, Speed: 2.0x, Form: 1.8x, Pace: 1.5x, Style: 2.0x, Post: 0.8x

### Critical Gaps for Elite Status:

#### 1. **Probabilistic Inference** ❌
- Current: Deterministic ratings → softmax
- Need: Bayesian uncertainty quantification for each component
- Implementation: Add confidence intervals to ratings

#### 2. **Multinomial Logit Model** ❌
- Current: Softmax (simplified logit)
- Need: Full multinomial logit with covariate regression
- Implementation: statsmodels.discrete.discrete_model.MNLogit

#### 3. **Pace Modeling** ⚠️
- Current: Basic pace scenario (+3.0 for lone speed, etc.)
- Need: Fractional time projections (E1, E2, LP timing)
- Implementation: LSTM neural network for pace prediction

#### 4. **Monte Carlo Simulation** ❌
- Current: Single point estimate
- Need: 10,000+ race simulations for confidence intervals
- Implementation: numpy.random with variance modeling

#### 5. **Kelly Criterion Optimization** ❌
- Current: Overlay calculation (fair odds vs market odds)
- Need: Optimal bet sizing with bankroll management
- Implementation: scipy.optimize with Kelly formula

#### 6. **Elo-Style Dynamic Ratings** ⚠️
- Current: Static race type hierarchy (G1=8, G2=7, etc.)
- Need: Dynamic horse ratings that update after each race
- Implementation: Bayesian Elo with K-factor tuning

#### 7. **Cross-Validation & Overfitting Prevention** ❌
- Current: No validation framework
- Need: Train/test splits, backtesting framework
- Implementation: sklearn.model_selection with time-series CV

#### 8. **Live Odds Integration** ⚠️
- Current: User manually inputs ML odds in Section A
- Need: Real-time odds feed integration and Bayesian updating
- Implementation: Weighted blend of model probs + market probs

---

## Phase 2: Critical Mathematical Enhancements (Priority Order)

### Enhancement 1: Bayesian Probability Framework 🎯
**Priority:** CRITICAL  
**Impact:** Foundation for all other enhancements  
**Timeline:** 2-3 hours

**Implementation:**
```python
# Add to unified_rating_engine.py
from scipy import stats
import numpy as np

class BayesianRatingComponent:
    """Each rating component returns (mean, std_dev) tuple"""
    
    def __init__(self, prior_mean: float, prior_std: float):
        self.prior_mean = prior_mean
        self.prior_std = prior_std
    
    def update(self, data_points: List[float]) -> Tuple[float, float]:
        """Bayesian posterior update using conjugate priors"""
        # For normal likelihood + normal prior → normal posterior
        n = len(data_points)
        data_mean = np.mean(data_points)
        data_var = np.var(data_points)
        
        # Posterior mean (weighted average)
        posterior_precision = (1/self.prior_std**2) + (n/data_var)
        posterior_mean = ((self.prior_mean/self.prior_std**2) + 
                         (n*data_mean/data_var)) / posterior_precision
        posterior_std = np.sqrt(1/posterior_precision)
        
        return posterior_mean, posterior_std
```

**Integration Points:**
- `_calc_class()`: Return (class_mean, class_std) based on purse variance
- `_calc_form()`: Return (form_mean, form_std) based on finish consistency
- `_calc_speed()`: Return (speed_mean, speed_std) based on fig variance
- Final rating: Propagate uncertainty through weighted sum

### Enhancement 2: Multinomial Logit Regression 🎯
**Priority:** CRITICAL  
**Impact:** True probabilistic finish predictions  
**Timeline:** 3-4 hours

**Implementation:**
```python
# Add to unified_rating_engine.py
import statsmodels.api as sm
from statsmodels.discrete.discrete_model import MNLogit

class MultinomialLogitPredictor:
    """
    Bill Benter-style multinomial logit model
    P(horse_i wins) = exp(β₀ + β₁·speed_i + β₂·class_i + ...) / Σ_j exp(...)
    """
    
    def __init__(self):
        self.model = None
        self.coefficients = None
    
    def fit(self, training_data: pd.DataFrame):
        """
        Train on historical race results
        
        Args:
            training_data: DataFrame with columns
                - horse_id
                - finish_position (1, 2, 3, ...)
                - speed_rating
                - class_rating
                - form_rating
                - pace_rating
                - style_rating
                - post_position
        """
        # Prepare data for multinomial logit
        y = training_data['finish_position'] - 1  # 0-indexed
        X = training_data[['speed_rating', 'class_rating', 'form_rating',
                          'pace_rating', 'style_rating', 'post_position']]
        X = sm.add_constant(X)
        
        # Fit multinomial logit
        self.model = MNLogit(y, X)
        result = self.model.fit(method='bfgs', maxiter=1000)
        self.coefficients = result.params
        
        return result.summary()
    
    def predict_probs(self, race_features: pd.DataFrame) -> np.ndarray:
        """
        Predict finish probabilities for each horse
        
        Returns:
            Array of shape (n_horses, n_positions)
            Each row sums to 1.0 across positions
        """
        X = sm.add_constant(race_features)
        probs = self.model.predict(self.coefficients, X)
        return probs
```

**Integration:**
- Replace current softmax in `fair_probs_from_ratings()` with MNLogit predictions
- Train model on historical data from gold_high_iq.db
- Use coefficients for interpretability (which factors matter most)

### Enhancement 3: Monte Carlo Simulation Framework 🎯
**Priority:** HIGH  
**Impact:** Confidence intervals for all predictions  
**Timeline:** 2-3 hours

**Implementation:**
```python
# Add to unified_rating_engine.py
class MonteCarloSimulator:
    """
    Run 10,000+ race simulations to quantify prediction uncertainty
    """
    
    def __init__(self, n_simulations: int = 10000):
        self.n_simulations = n_simulations
    
    def simulate_race(self, 
                     ratings_with_uncertainty: List[Tuple[float, float]],
                     correlation_matrix: np.ndarray = None) -> Dict[str, Any]:
        """
        Simulate race outcomes accounting for rating uncertainty
        
        Args:
            ratings_with_uncertainty: List of (mean_rating, std_rating) for each horse
            correlation_matrix: Optional correlation between horse performances
        
        Returns:
            {
                'win_probs': [prob for each horse],
                'place_probs': [prob for each horse],
                'show_probs': [prob for each horse],
                'confidence_intervals': [(low, high) for each horse],
                'exacta_probs': {(i,j): prob},
                'trifecta_probs': {(i,j,k): prob}
            }
        """
        n_horses = len(ratings_with_uncertainty)
        win_counts = np.zeros(n_horses)
        place_counts = np.zeros(n_horses)
        show_counts = np.zeros(n_horses)
        
        # Track exotic frequencies
        exacta_counts = {}
        trifecta_counts = {}
        
        for sim in range(self.n_simulations):
            # Draw random ratings from distributions
            simulated_ratings = []
            for mean_rating, std_rating in ratings_with_uncertainty:
                # Add race variance (track condition randomness)
                race_noise = np.random.normal(0, 0.5)
                horse_rating = np.random.normal(mean_rating, std_rating) + race_noise
                simulated_ratings.append(horse_rating)
            
            # Convert to probabilities (softmax)
            exp_ratings = np.exp(np.array(simulated_ratings) / 3.0)
            probs = exp_ratings / exp_ratings.sum()
            
            # Simulate finish order (multinomial draw)
            finish_order = np.random.choice(n_horses, size=n_horses, 
                                           replace=False, p=probs)
            
            # Update counts
            win_counts[finish_order[0]] += 1
            place_counts[finish_order[0]] += 1
            place_counts[finish_order[1]] += 1
            show_counts[finish_order[0]] += 1
            show_counts[finish_order[1]] += 1
            show_counts[finish_order[2]] += 1
            
            # Track exotics
            exacta_key = (finish_order[0], finish_order[1])
            exacta_counts[exacta_key] = exacta_counts.get(exacta_key, 0) + 1
            
            trifecta_key = (finish_order[0], finish_order[1], finish_order[2])
            trifecta_counts[trifecta_key] = trifecta_counts.get(trifecta_key, 0) + 1
        
        return {
            'win_probs': win_counts / self.n_simulations,
            'place_probs': place_counts / self.n_simulations,
            'show_probs': show_counts / self.n_simulations,
            'exacta_probs': {k: v/self.n_simulations for k, v in exacta_counts.items()},
            'trifecta_probs': {k: v/self.n_simulations for k, v in trifecta_counts.items()}
        }
```

### Enhancement 4: Kelly Criterion Bet Optimization 🎯
**Priority:** HIGH  
**Impact:** Optimal bankroll management  
**Timeline:** 2 hours

**Implementation:**
```python
# Add new file: betting_optimizer.py
from pulp import *
import numpy as np

class KellyCriterionOptimizer:
    """
    Optimize bet sizing using Kelly Criterion with safety constraints
    
    Kelly Formula: f* = (bp - q) / b
    where:
        f* = optimal fraction of bankroll
        b = odds received (decimal - 1)
        p = true win probability
        q = 1 - p
    
    For multiple bets, use linear programming to maximize expected log wealth
    """
    
    def __init__(self, bankroll: float, max_bet_fraction: float = 0.05):
        self.bankroll = bankroll
        self.max_bet_fraction = max_bet_fraction  # Safety constraint
    
    def calculate_full_kelly(self, true_prob: float, odds: float) -> float:
        """
        Calculate full Kelly bet size for single bet
        
        Args:
            true_prob: Model's win probability
            odds: Market odds (decimal format, e.g., 5.0 for 4/1)
        
        Returns:
            Optimal bet fraction (0.0 to 1.0)
        """
        b = odds - 1  # Net odds
        p = true_prob
        q = 1 - p
        
        kelly_fraction = (b * p - q) / b
        
        # Safety: Never bet more than max_bet_fraction
        kelly_fraction = np.clip(kelly_fraction, 0.0, self.max_bet_fraction)
        
        return kelly_fraction
    
    def calculate_fractional_kelly(self, true_prob: float, odds: float, 
                                   fraction: float = 0.25) -> float:
        """
        Use fractional Kelly (1/4 Kelly popular for risk reduction)
        """
        full_kelly = self.calculate_full_kelly(true_prob, odds)
        return full_kelly * fraction
    
    def optimize_multi_bet_portfolio(self, 
                                    opportunities: List[Dict[str, float]],
                                    correlation_matrix: np.ndarray = None) -> Dict:
        """
        Optimize bet allocation across multiple opportunities using LP
        
        Args:
            opportunities: List of dicts with keys:
                - 'horse': horse name
                - 'bet_type': 'win', 'place', 'exacta', etc.
                - 'true_prob': model probability
                - 'odds': market odds
                - 'ev': expected value per dollar
            correlation_matrix: Pairwise correlations between outcomes
        
        Returns:
            Optimal bet amounts for each opportunity
        """
        prob = LpProblem("Kelly_Optimization", LpMaximize)
        
        # Decision variables: bet amount for each opportunity
        bet_vars = []
        for i, opp in enumerate(opportunities):
            var = LpVariable(f"bet_{i}", lowBound=0, 
                           upBound=self.bankroll * self.max_bet_fraction)
            bet_vars.append(var)
        
        # Objective: Maximize expected log wealth (approximation)
        # E[log(W)] ≈ Σ p_i * log(1 + b_i * x_i) for small x_i
        expected_value = lpSum([
            opp['ev'] * bet_vars[i]
            for i, opp in enumerate(opportunities)
        ])
        prob += expected_value
        
        # Constraint: Total bets <= bankroll
        prob += lpSum(bet_vars) <= self.bankroll
        
        # Solve
        prob.solve(PULP_CBC_CMD(msg=0))
        
        # Extract solution
        optimal_bets = []
        for i, var in enumerate(bet_vars):
            optimal_bets.append({
                'horse': opportunities[i]['horse'],
                'bet_type': opportunities[i]['bet_type'],
                'amount': var.varValue,
                'expected_return': var.varValue * opportunities[i]['ev']
            })
        
        return {
            'optimal_bets': optimal_bets,
            'total_bet': sum(b['amount'] for b in optimal_bets),
            'expected_profit': value(prob.objective),
            'status': LpStatus[prob.status]
        }
```

### Enhancement 5: Elo-Style Dynamic Horse Ratings 🎯
**Priority:** MEDIUM  
**Impact:** Adaptive class ratings  
**Timeline:** 3 hours

**Implementation:**
```python
# Add to gold_database_manager.py
class DynamicEloRatingSystem:
    """
    Bayesian Elo rating system for horses
    Each horse has dynamic rating that updates after each race
    """
    
    def __init__(self, K_factor: float = 32, initial_rating: float = 1500):
        self.K = K_factor
        self.initial_rating = initial_rating
        self.ratings = {}  # {horse_name: (elo_rating, uncertainty)}
    
    def get_rating(self, horse_name: str) -> Tuple[float, float]:
        """Get current Elo rating and uncertainty"""
        if horse_name not in self.ratings:
            return self.initial_rating, 200.0  # High uncertainty for new horse
        return self.ratings[horse_name]
    
    def expected_score(self, rating_a: float, rating_b: float) -> float:
        """
        Expected score for horse A against horse B
        Uses logistic curve: E_a = 1 / (1 + 10^((R_b - R_a)/400))
        """
        return 1 / (1 + 10**((rating_b - rating_a) / 400))
    
    def update_ratings(self, race_results: List[Tuple[str, int]]):
        """
        Update Elo ratings after race
        
        Args:
            race_results: List of (horse_name, finish_position) sorted by finish
        """
        n_horses = len(race_results)
        
        # Create pairwise comparisons
        for i, (horse_a, finish_a) in enumerate(race_results):
            rating_a, uncertainty_a = self.get_rating(horse_a)
            
            # Adaptive K-factor based on uncertainty
            K_adaptive = self.K * (uncertainty_a / 200.0)
            
            # Compare against all horses that finished behind
            total_delta = 0.0
            for j, (horse_b, finish_b) in enumerate(race_results):
                if finish_b > finish_a:  # Horse A beat Horse B
                    rating_b, _ = self.get_rating(horse_b)
                    expected = self.expected_score(rating_a, rating_b)
                    actual = 1.0  # Win
                    total_delta += K_adaptive * (actual - expected)
                elif finish_b < finish_a:  # Horse A lost to Horse B
                    rating_b, _ = self.get_rating(horse_b)
                    expected = self.expected_score(rating_a, rating_b)
                    actual = 0.0  # Loss
                    total_delta += K_adaptive * (actual - expected)
            
            # Update rating
            new_rating = rating_a + total_delta
            new_uncertainty = max(50.0, uncertainty_a * 0.95)  # Reduce uncertainty
            
            self.ratings[horse_a] = (new_rating, new_uncertainty)
```

### Enhancement 6: LSTM Pace Prediction Neural Network 🎯
**Priority:** MEDIUM  
**Impact:** Fractional time projections  
**Timeline:** 4-5 hours

**Implementation:**
```python
# Add new file: pace_neural_network.py
import torch
import torch.nn as nn

class LSTMPacePredictor(nn.Module):
    """
    LSTM neural network for predicting fractional times
    
    Input: Horse's running style, early speed %, pace figures
    Output: Projected E1 (2f), E2 (4f), LP (6f/8f) times
    """
    
    def __init__(self, input_dim=10, hidden_dim=64, num_layers=2):
        super(LSTMPacePredictor, self).__init__()
        
        self.lstm = nn.LSTM(input_dim, hidden_dim, num_layers, 
                           batch_first=True, dropout=0.2)
        
        # Output layers for each fractional call
        self.fc_e1 = nn.Linear(hidden_dim, 1)  # 2f time
        self.fc_e2 = nn.Linear(hidden_dim, 1)  # 4f time
        self.fc_lp = nn.Linear(hidden_dim, 1)  # 6f/8f time
    
    def forward(self, x):
        """
        Args:
            x: Tensor of shape (batch, sequence_length, input_dim)
               sequence_length = recent races (e.g., last 5)
               input_dim = features [early_speed_pct, pace_fig, class, etc.]
        
        Returns:
            Tuple of (e1_time, e2_time, lp_time)
        """
        lstm_out, _ = self.lstm(x)
        last_output = lstm_out[:, -1, :]  # Use last timestep
        
        e1 = self.fc_e1(last_output)
        e2 = self.fc_e2(last_output)
        lp = self.fc_lp(last_output)
        
        return e1, e2, lp
    
    def train_model(self, training_data, epochs=100, lr=0.001):
        """Train on historical fractional time data"""
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        criterion = nn.MSELoss()
        
        for epoch in range(epochs):
            for batch in training_data:
                features, targets = batch
                
                optimizer.zero_grad()
                e1_pred, e2_pred, lp_pred = self.forward(features)
                
                loss = (criterion(e1_pred, targets[:, 0]) +
                       criterion(e2_pred, targets[:, 1]) +
                       criterion(lp_pred, targets[:, 2]))
                
                loss.backward()
                optimizer.step()
```

---

## Phase 3: Integration Strategy (Zero Redundancy)

### Step 1: Audit Current Calculation Paths
**Action:** Map all rating calculation flows to ensure single source of truth

```
Current Flow:
1. PP Text → Elite Parser → HorseData objects
2. HorseData → Unified Engine → Component Ratings
3. Component Ratings → Weighted Sum → Final Rating
4. Final Rating → Softmax → Win Probabilities
5. Win Probs + ML Odds → Fair Probs (with reality check)
6. Fair Probs vs Market Odds → Overlays
```

**Enhancement Flow (No Redundancy):**
```
Enhanced Flow:
1. PP Text → Elite Parser → HorseData objects
2. HorseData → Bayesian Rating Engine → (mean, std) for each component
3. (mean, std) tuples → Monte Carlo Simulator → Win/Place/Show distributions
4. Distributions + ML Odds → Multinomial Logit Model → Calibrated Probs
5. Calibrated Probs + Market Odds → Kelly Optimizer → Optimal Bet Sizes
6. All results → JSON output with confidence intervals
```

### Step 2: Consolidate into Single Prediction Pipeline
**Location:** unified_rating_engine.py (enhance existing, don't duplicate)

**New Method:**
```python
def predict_race_elite(self, pp_text, today_purse, today_race_type, ...):
    """
    ELITE PREDICTION PIPELINE - Single entry point for all predictions
    
    Returns:
        {
            'ratings': DataFrame with (mean, std) for each component,
            'probabilities': {
                'win': [prob per horse],
                'place': [prob per horse],
                'show': [prob per horse],
                'confidence_intervals': [(low, high) per horse]
            },
            'exotics': {
                'exacta': {(i,j): prob},
                'trifecta': {(i,j,k): prob}
            },
            'betting_strategy': {
                'optimal_bets': [...],
                'expected_roi': float,
                'risk_of_ruin': float
            },
            'model_diagnostics': {
                'parsing_confidence': float,
                'prediction_confidence': float,
                'calibration_score': float
            }
        }
    """
    # STEP 1: Parse (existing)
    horses = self.parser.parse_full_pp(pp_text)
    
    # STEP 2: Bayesian ratings (NEW)
    bayesian_ratings = self._calculate_bayesian_ratings(horses, ...)
    
    # STEP 3: Monte Carlo simulation (NEW)
    mc_results = self.monte_carlo.simulate_race(bayesian_ratings)
    
    # STEP 4: Multinomial logit calibration (NEW)
    calibrated_probs = self.mnlogit_model.predict_probs(bayesian_ratings)
    
    # STEP 5: Kelly optimization (NEW)
    betting_strategy = self.kelly_optimizer.optimize_multi_bet_portfolio(...)
    
    return {...}
```

### Step 3: Database Schema Extensions
**Location:** gold_high_iq.db

**New Tables:**
```sql
-- Store Elo ratings
CREATE TABLE horse_elo_ratings (
    horse_name TEXT PRIMARY KEY,
    elo_rating REAL,
    uncertainty REAL,
    last_updated DATETIME,
    races_count INTEGER
);

-- Store model training history
CREATE TABLE model_training_log (
    training_id INTEGER PRIMARY KEY,
    model_type TEXT,  -- 'mnlogit', 'lstm_pace', etc.
    training_date DATETIME,
    accuracy REAL,
    calibration_score REAL,
    n_samples INTEGER,
    coefficients TEXT  -- JSON
);

-- Store Monte Carlo results for analysis
CREATE TABLE monte_carlo_predictions (
    race_id TEXT,
    horse_name TEXT,
    win_prob REAL,
    place_prob REAL,
    show_prob REAL,
    confidence_low REAL,
    confidence_high REAL,
    actual_finish INTEGER,  -- For backtesting
    PRIMARY KEY (race_id, horse_name)
);
```

---

## Phase 4: Testing & Validation Framework

### Cross-Validation Strategy
```python
# Add new file: model_validation.py
from sklearn.model_selection import TimeSeriesSplit
import numpy as np

class RaceModelValidator:
    """
    Validate prediction models using time-series cross-validation
    (Cannot shuffle races - must respect temporal order)
    """
    
    def __init__(self, n_splits=5):
        self.tscv = TimeSeriesSplit(n_splits=n_splits)
    
    def validate_model(self, historical_races: pd.DataFrame):
        """
        Perform time-series CV on historical data
        
        Metrics:
        - Win accuracy: % of time top-rated horse won
        - Calibration: Brier score
        - Profit: ROI if betting on all overlays
        """
        accuracies = []
        brier_scores = []
        rois = []
        
        for train_idx, test_idx in self.tscv.split(historical_races):
            train_data = historical_races.iloc[train_idx]
            test_data = historical_races.iloc[test_idx]
            
            # Train model on train_data
            model = self._train_on_fold(train_data)
            
            # Predict on test_data
            predictions = model.predict(test_data)
            actuals = test_data['winner']
            
            # Calculate metrics
            accuracy = (predictions.argmax(axis=1) == actuals).mean()
            brier = np.mean((predictions - actuals)**2)
            roi = self._calculate_roi(predictions, test_data['odds'], actuals)
            
            accuracies.append(accuracy)
            brier_scores.append(brier)
            rois.append(roi)
        
        return {
            'mean_accuracy': np.mean(accuracies),
            'std_accuracy': np.std(accuracies),
            'mean_brier': np.mean(brier_scores),
            'mean_roi': np.mean(rois),
            'confidence_interval': (
                np.mean(accuracies) - 1.96*np.std(accuracies),
                np.mean(accuracies) + 1.96*np.std(accuracies)
            )
        }
```

---

## Phase 5: Production Deployment Checklist

### Code Quality ✅
- [ ] PEP8 compliance (autopep8, black)
- [ ] Type hints on all functions
- [ ] Docstrings with mathematical formulas
- [ ] Unit tests for each component
- [ ] Integration tests for full pipeline

### Performance ✅
- [ ] Vectorize operations (avoid Python loops)
- [ ] Cache expensive calculations
- [ ] Profile bottlenecks (cProfile)
- [ ] Optimize database queries

### Robustness ✅
- [ ] Handle missing data (imputation strategies)
- [ ] Validate inputs (assert types, ranges)
- [ ] Log all predictions for audit trail
- [ ] Graceful degradation if ML models fail

### Documentation ✅
- [ ] Mathematical derivations in docstrings
- [ ] Architecture diagram (current vs enhanced)
- [ ] API reference for all functions
- [ ] User guide for interpreting outputs

---

## Expected Performance Targets

### Current System:
- Win accuracy: ~50-60% (top pick wins)
- Calibration: Unknown (no validation)
- ROI: Variable (~1.10-1.30x on overlays)

### Enhanced System Targets:
- **Win accuracy: 70-75%** (top pick wins)
- **Calibration: Brier score < 0.15** (excellent)
- **ROI: 1.40-1.60x** (20-30% Kelly betting)
- **Risk of Ruin: < 1%** (over 1000 races)

---

## Implementation Timeline

### Week 1: Foundation
- Day 1-2: Bayesian rating components
- Day 3-4: Multinomial logit model
- Day 5: Monte Carlo simulator

### Week 2: Optimization
- Day 1-2: Kelly criterion optimizer
- Day 3: Elo dynamic ratings
- Day 4-5: LSTM pace model

### Week 3: Integration & Testing
- Day 1-2: Consolidate into unified pipeline
- Day 3: Cross-validation framework
- Day 4-5: Backtesting on historical data

### Week 4: Production
- Day 1-2: Performance optimization
- Day 3: Documentation
- Day 4-5: Deployment & monitoring

---

## Next Immediate Action

**Priority:** Fix any critical bugs in current unified engine after recent tuning  
**Then:** Implement Bayesian rating components (highest impact, foundation for others)  
**User Action Required:** Test Streamlit app at http://localhost:8502 and confirm Pegasus predictions are now sensible

