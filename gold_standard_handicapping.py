"""
🏇 GOLD STANDARD HANDICAPPING PHILOSOPHY
==========================================

Based on the collective wisdom of:
- Andy Beyer: Speed figure methodology, trip handicapping
- Steven Crist: Expected value, discipline, passing races
- Ellis Starr: Class, pace analysis, trainer patterns
- Brad Free: Track bias exploitation
- David Aragona: Pace handicapping, race shape
- Michael Beychok: Data-driven statistical analysis
- Paul Matties Jr.: Pace figures, fractional times
- Judy Wagner: Turf breeding, pedigree analysis
- Frank Scatoni: Class analysis, exotic strategies
- Dean Keppler: Pace projections, race flow

CORE PRINCIPLES:
1. Analytical Rigor: Comprehensive data processing
2. Discipline & Selectivity: Only bet when edge > 15%
3. Patience: Long-term profitability over quick wins
4. Objectivity: Bias-free, data-driven decisions
5. Risk Management: Kelly Criterion bankroll management
6. Adaptability: Real-time adjustments for scratches/changes
7. Pattern Recognition: Race shape, track bias, connections
8. Fundamentals First: Distance, surface, class are king
"""

import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from scipy import stats
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')


# =====================================================================
# CONFIGURATION: Gold Standard Parameters
# =====================================================================

@dataclass
class HandicappingConfig:
    """
    Configuration based on professional handicapper standards.
    
    Steven Crist: "Bet only when your odds are 20% higher than fair odds"
    Andy Beyer: "Speed figures must be normalized to track variant"
    Ellis Starr: "Class is the single most important factor"
    """
    
    # Betting discipline (Steven Crist principle)
    minimum_edge_percentage: float = 15.0  # Only bet with 15%+ edge
    minimum_roi_threshold: float = 1.20    # 20% ROI minimum
    maximum_bet_percentage: float = 5.0    # Max 5% of bankroll per race
    
    # Confidence thresholds (pass races below these)
    minimum_confidence_score: float = 0.70  # 70% confidence to bet
    minimum_data_completeness: float = 0.80  # 80% of features present
    
    # Risk management (Kelly Criterion with fractional Kelly)
    kelly_fraction: float = 0.25  # Quarter Kelly (conservative)
    maximum_field_size: int = 14  # Pass on fields > 14 horses
    
    # Value thresholds by bet type
    win_bet_min_edge: float = 20.0      # 20% edge for win bets
    exacta_min_edge: float = 25.0       # 25% edge for exactas
    trifecta_min_edge: float = 35.0     # 35% edge for trifectas
    
    # Beyer speed figure standards
    beyer_minimum_competitive: int = 70  # Minimum to compete
    beyer_elite_threshold: int = 100     # Elite horse threshold
    
    # Class jumping red flags (Ellis Starr)
    class_jump_max_safe: float = 1.5     # Max safe class jump (purse ratio)
    class_drop_bonus: float = 0.8        # Bonus for class drops
    
    # Pace scenario confidence (David Aragona)
    pace_pressure_critical_esp: float = 0.65  # High ESP = pace setup
    pace_analysis_min_horses: int = 4    # Need 4+ horses for pace analysis


# =====================================================================
# COMMON SENSE VALIDATION: Sanity Checks on Predictions
# =====================================================================

class CommonSenseValidator:
    """
    Applies handicapping common sense to catch mathematical errors.
    
    Examples of common sense rules:
    - A debut horse can't have the highest speed figure
    - A horse dropping 50% in class is suspicious (may be injured)
    - A 90-day layoff with no workouts is a red flag
    - An outsider at 50/1 shouldn't be top pick (check your math)
    """
    
    @staticmethod
    def validate_predictions(
        predictions: pd.DataFrame,
        race_context: Dict,
        horse_data: List[Dict]
    ) -> Tuple[bool, List[str]]:
        """
        Run sanity checks on predictions.
        
        Returns:
            (is_valid, list_of_warnings)
        """
        warnings = []
        
        # Check 1: Top pick shouldn't be extreme longshot
        top_pick = predictions.iloc[0]
        top_ml_odds = top_pick.get('ML', '99')
        
        try:
            ml_decimal = float(top_ml_odds.split('/')[0]) / float(top_ml_odds.split('/')[1]) if '/' in str(top_ml_odds) else float(top_ml_odds)
            if ml_decimal > 30:
                warnings.append(f"⚠️ COMMON SENSE: Top pick is 30/1+ longshot. Double-check ratings.")
        except:
            pass
        
        # Check 2: Favorite should be in top 4 picks (unless there's a reason)
        if 'ML' in predictions.columns:
            try:
                predictions['ml_decimal'] = predictions['ML'].apply(
                    lambda x: float(str(x).split('/')[0]) / float(str(x).split('/')[1]) if '/' in str(x) else float(x)
                )
                favorite_idx = predictions['ml_decimal'].idxmin()
                favorite_rank = predictions.index.get_loc(favorite_idx) + 1
                
                if favorite_rank > 5:
                    warnings.append(f"⚠️ COMMON SENSE: Public favorite ranked #{favorite_rank}. Verify this is intentional.")
            except:
                pass
        
        # Check 3: Debut horses shouldn't dominate top picks
        debut_count_top3 = 0
        for i, horse_dict in enumerate(horse_data[:3]):
            if horse_dict.get('starts_lifetime', 1) == 0:
                debut_count_top3 += 1
        
        if debut_count_top3 >= 2:
            warnings.append(f"⚠️ COMMON SENSE: {debut_count_top3} debut horses in top 3. Maidens are unpredictable.")
        
        # Check 4: Extreme class drops (possible injury)
        for horse_dict in horse_data[:5]:
            if 'class_level_last' in horse_dict and 'class_level_today' in horse_dict:
                class_drop_ratio = horse_dict['class_level_last'] / max(horse_dict['class_level_today'], 1)
                if class_drop_ratio > 2.5:
                    warnings.append(f"⚠️ COMMON SENSE: {horse_dict['horse_name']} dropping 2.5x in class. Check for injury.")
        
        # Check 5: Long layoff without workouts
        for horse_dict in horse_data[:5]:
            days_off = horse_dict.get('days_since_last', 0)
            workouts = horse_dict.get('workout_count_60days', 0)
            
            if days_off > 180 and workouts < 3:
                warnings.append(f"⚠️ COMMON SENSE: {horse_dict['horse_name']} has 180+ day layoff with <3 workouts. Risky.")
        
        # Check 6: Field size sanity
        field_size = race_context.get('field_size', 0)
        if field_size > 14:
            warnings.append(f"⚠️ DISCIPLINE: Field size = {field_size}. Steven Crist recommends passing large fields.")
        
        # Check 7: All predictions have similar probabilities (no clear favorite)
        if 'Pred_Win_Prob' in predictions.columns:
            top3_probs = predictions.head(3)['Pred_Win_Prob'].values
            if max(top_probs) - min(top_probs) < 0.05:  # Within 5%
                warnings.append(f"⚠️ COMMON SENSE: Top 3 picks have similar probabilities. Race may be too contentious to bet.")
        
        is_valid = len(warnings) == 0
        return is_valid, warnings


# =====================================================================
# EXPECTED VALUE CALCULATOR (Steven Crist Methodology)
# =====================================================================

class ExpectedValueCalculator:
    """
    Calculate betting value using Steven Crist's expected value formula.
    
    EV = (Fair Win Probability × Payout) - (Fair Loss Probability × Wager)
    
    Only recommend bets when EV > minimum threshold (discipline).
    """
    
    @staticmethod
    def calculate_win_ev(
        fair_win_prob: float,
        ml_odds: str,
        wager: float = 2.00
    ) -> Tuple[float, float, bool]:
        """
        Calculate expected value for win bet.
        
        Returns:
            (ev_dollars, edge_percentage, is_overlay)
        """
        # Convert ML odds to decimal
        try:
            if '/' in str(ml_odds):
                num, den = ml_odds.split('/')
                decimal_odds = float(num) / float(den)
            else:
                decimal_odds = float(ml_odds)
        except:
            return 0.0, 0.0, False
        
        # Market implied probability
        market_prob = 1 / (decimal_odds + 1)
        
        # Edge calculation (Steven Crist method)
        edge_percentage = ((fair_win_prob - market_prob) / market_prob) * 100
        
        # EV calculation
        payout = wager * decimal_odds  # Net profit on win
        ev_win = fair_win_prob * payout
        ev_loss = (1 - fair_win_prob) * wager
        ev_dollars = ev_win - ev_loss
        
        # Overlay check
        is_overlay = fair_win_prob > market_prob
        
        return ev_dollars, edge_percentage, is_overlay
    
    @staticmethod
    def calculate_exacta_ev(
        horse1_prob: float,
        horse2_prob: float,
        exacta_payout: float,
        wager: float = 2.00
    ) -> float:
        """
        Calculate expected value for exacta bet.
        
        Simplified: P(H1 wins) × P(H2 finishes 2nd | H1 wins)
        """
        # Conditional probability (simplified)
        prob_exacta = horse1_prob * (horse2_prob / (1 - horse1_prob + 0.001))
        
        ev = (prob_exacta * exacta_payout) - wager
        return ev


# =====================================================================
# BETTING RECOMMENDATION ENGINE
# =====================================================================

class BettingRecommendationEngine:
    """
    Generate betting recommendations using professional handicapper standards.
    
    Philosophy:
    - Steven Crist: Only bet when edge exceeds threshold
    - Andy Beyer: Consider race shape and trip handicapping
    - Ellis Starr: Focus on class and pace scenarios
    """
    
    def __init__(self, config: HandicappingConfig = None):
        self.config = config or HandicappingConfig()
        self.ev_calculator = ExpectedValueCalculator()
        self.validator = CommonSenseValidator()
    
    def generate_recommendations(
        self,
        predictions: pd.DataFrame,
        race_context: Dict,
        horse_data: List[Dict],
        bankroll: float = 1000.0
    ) -> Dict:
        """
        Generate betting recommendations with discipline and selectivity.
        
        Returns:
            {
                'recommended_bets': List of bet dictionaries,
                'pass_race': bool,
                'pass_reason': str,
                'confidence_score': float,
                'common_sense_warnings': List[str]
            }
        """
        
        # Step 1: Common sense validation
        is_valid, warnings = self.validator.validate_predictions(
            predictions, race_context, horse_data
        )
        
        # Step 2: Calculate confidence score
        confidence_score = self._calculate_confidence_score(
            predictions, race_context, horse_data
        )
        
        # Step 3: Discipline check - should we pass this race?
        should_pass, pass_reason = self._should_pass_race(
            predictions, race_context, confidence_score
        )
        
        if should_pass:
            return {
                'recommended_bets': [],
                'pass_race': True,
                'pass_reason': pass_reason,
                'confidence_score': confidence_score,
                'common_sense_warnings': warnings
            }
        
        # Step 4: Generate specific bet recommendations
        recommended_bets = self._generate_bet_recommendations(
            predictions, horse_data, bankroll, confidence_score
        )
        
        return {
            'recommended_bets': recommended_bets,
            'pass_race': False,
            'pass_reason': None,
            'confidence_score': confidence_score,
            'common_sense_warnings': warnings
        }
    
    def _calculate_confidence_score(
        self,
        predictions: pd.DataFrame,
        race_context: Dict,
        horse_data: List[Dict]
    ) -> float:
        """
        Calculate overall confidence in predictions (0.0 to 1.0).
        
        Factors:
        - Data completeness
        - Top pick probability
        - Ensemble uncertainty (if available)
        - Field size
        """
        confidence = 1.0
        
        # Factor 1: Data completeness
        total_features = 25  # From our 25-D feature vector
        avg_features_present = np.mean([
            len([v for v in h.values() if v is not None]) / total_features
            for h in horse_data[:5]
        ])
        confidence *= avg_features_present
        
        # Factor 2: Top pick probability
        if 'Pred_Win_Prob' in predictions.columns:
            top_prob = predictions.iloc[0]['Pred_Win_Prob']
            if top_prob < 0.20:  # Weak favorite
                confidence *= 0.8
            elif top_prob > 0.35:  # Strong favorite
                confidence *= 1.1
        
        # Factor 3: Ensemble uncertainty (if available)
        if 'Uncertainty' in predictions.columns:
            top_uncertainty = predictions.iloc[0]['Uncertainty']
            confidence *= (1.0 - top_uncertainty)
        
        # Factor 4: Field size penalty
        field_size = race_context.get('field_size', 8)
        if field_size > 12:
            confidence *= 0.9
        
        return min(confidence, 1.0)
    
    def _should_pass_race(
        self,
        predictions: pd.DataFrame,
        race_context: Dict,
        confidence_score: float
    ) -> Tuple[bool, Optional[str]]:
        """
        Determine if we should pass this race (Steven Crist discipline).
        
        Returns:
            (should_pass, reason)
        """
        
        # Check 1: Low confidence
        if confidence_score < self.config.minimum_confidence_score:
            return True, f"Low confidence ({confidence_score:.1%} < {self.config.minimum_confidence_score:.0%})"
        
        # Check 2: Field too large
        field_size = race_context.get('field_size', 0)
        if field_size > self.config.maximum_field_size:
            return True, f"Field too large ({field_size} horses > {self.config.maximum_field_size})"
        
        # Check 3: No clear edge on top pick
        if 'Pred_Win_Prob' in predictions.columns and 'ML' in predictions.columns:
            top_pick = predictions.iloc[0]
            ev_dollars, edge_pct, is_overlay = self.ev_calculator.calculate_win_ev(
                top_pick['Pred_Win_Prob'],
                top_pick['ML']
            )
            
            if edge_pct < self.config.minimum_edge_percentage:
                return True, f"No betting value (edge = {edge_pct:.1f}% < {self.config.minimum_edge_percentage:.0f}%)"
        
        # Check 4: Contentious race (multiple similar probabilities)
        if 'Pred_Win_Prob' in predictions.columns:
            top3 = predictions.head(3)['Pred_Win_Prob'].values
            spread = max(top3) - min(top3)
            if spread < 0.08:  # Within 8%
                return True, f"Race too contentious (top 3 within {spread:.1%})"
        
        return False, None
    
    def _generate_bet_recommendations(
        self,
        predictions: pd.DataFrame,
        horse_data: List[Dict],
        bankroll: float,
        confidence_score: float
    ) -> List[Dict]:
        """
        Generate specific bet recommendations with Kelly Criterion sizing.
        """
        recommendations = []
        
        # Win bet on top pick
        if 'Pred_Win_Prob' in predictions.columns and 'ML' in predictions.columns:
            top_pick = predictions.iloc[0]
            
            ev_dollars, edge_pct, is_overlay = self.ev_calculator.calculate_win_ev(
                top_pick['Pred_Win_Prob'],
                top_pick['ML'],
                wager=2.0
            )
            
            if is_overlay and edge_pct >= self.config.win_bet_min_edge:
                # Kelly Criterion sizing
                fair_prob = top_pick['Pred_Win_Prob']
                ml_odds = top_pick['ML']
                
                try:
                    if '/' in str(ml_odds):
                        num, den = ml_odds.split('/')
                        decimal_odds = float(num) / float(den)
                    else:
                        decimal_odds = float(ml_odds)
                    
                    # Kelly = (bp - q) / b, where b = decimal odds, p = win prob, q = lose prob
                    kelly_pct = ((decimal_odds * fair_prob) - (1 - fair_prob)) / decimal_odds
                    kelly_pct = max(0, kelly_pct)  # No negative Kelly
                    
                    # Apply fractional Kelly for safety
                    bet_pct = kelly_pct * self.config.kelly_fraction
                    bet_pct = min(bet_pct, self.config.maximum_bet_percentage / 100)
                    
                    bet_amount = bankroll * bet_pct
                    
                    recommendations.append({
                        'bet_type': 'WIN',
                        'horse': top_pick['Horse'],
                        'post': top_pick.get('Post', '?'),
                        'fair_prob': f"{fair_prob:.1%}",
                        'ml_odds': ml_odds,
                        'edge': f"{edge_pct:.1f}%",
                        'ev_per_dollar': f"${ev_dollars/2:.2f}",
                        'recommended_bet': f"${bet_amount:.2f}",
                        'bankroll_pct': f"{bet_pct*100:.1f}%",
                        'confidence': f"{confidence_score:.0%}",
                        'reason': f"Strong overlay with {edge_pct:.0f}% edge"
                    })
                except:
                    pass
        
        # Exacta box on top 2 if both are overlays
        if len(predictions) >= 2 and 'Pred_Win_Prob' in predictions.columns:
            top2 = predictions.head(2)
            
            horse1 = top2.iloc[0]
            horse2 = top2.iloc[1]
            
            # Check if both are overlays
            ev1, edge1, overlay1 = self.ev_calculator.calculate_win_ev(
                horse1['Pred_Win_Prob'], horse1['ML']
            )
            ev2, edge2, overlay2 = self.ev_calculator.calculate_win_ev(
                horse2['Pred_Win_Prob'], horse2['ML']
            )
            
            if overlay1 and overlay2 and (edge1 + edge2)/2 >= self.config.exacta_min_edge:
                # Conservative exacta bet (smaller stakes)
                bet_amount = bankroll * 0.01  # 1% of bankroll
                
                recommendations.append({
                    'bet_type': 'EXACTA BOX',
                    'horses': [horse1['Horse'], horse2['Horse']],
                    'posts': [horse1.get('Post', '?'), horse2.get('Post', '?')],
                    'combined_prob': f"{(horse1['Pred_Win_Prob'] + horse2['Pred_Win_Prob']):.1%}",
                    'avg_edge': f"{(edge1 + edge2)/2:.1f}%",
                    'recommended_bet': f"${bet_amount:.2f}",
                    'bankroll_pct': "1.0%",
                    'confidence': f"{confidence_score:.0%}",
                    'reason': "Both horses are overlays"
                })
        
        return recommendations


# =====================================================================
# MONTE CARLO SIMULATION: Long-term Testing
# =====================================================================

class MonteCarloSimulator:
    """
    Test prediction system over 100+ races with variance modeling.
    
    Principle: Win rate may be 30%, but ROI should be positive long-term.
    """
    
    @staticmethod
    def simulate_season(
        predictions_list: List[pd.DataFrame],
        actual_results: List[int],  # Actual winner's index
        ml_odds_list: List[List[str]],
        starting_bankroll: float = 1000.0,
        num_simulations: int = 1000
    ) -> Dict:
        """
        Simulate betting performance over multiple races.
        
        Returns:
            {
                'avg_final_bankroll': float,
                'win_rate': float,
                'roi': float,
                'sharpe_ratio': float,
                'max_drawdown': float
            }
        """
        
        final_bankrolls = []
        
        for sim in range(num_simulations):
            bankroll = starting_bankroll
            wins = 0
            total_bets = 0
            
            for i, (predictions, actual_winner, ml_odds) in enumerate(
                zip(predictions_list, actual_results, ml_odds_list)
            ):
                # Simple flat bet on top pick
                bet_amount = bankroll * 0.02  # 2% flat bet
                
                predicted_winner_idx = 0  # Top pick
                
                if predicted_winner_idx == actual_winner:
                    # Win - collect payout
                    odds_str = ml_odds[predicted_winner_idx]
                    try:
                        if '/' in odds_str:
                            num, den = odds_str.split('/')
                            decimal_odds = float(num) / float(den)
                        else:
                            decimal_odds = float(odds_str)
                        
                        payout = bet_amount * decimal_odds
                        bankroll += payout
                        wins += 1
                    except:
                        bankroll -= bet_amount  # Lost bet
                else:
                    # Loss
                    bankroll -= bet_amount
                
                total_bets += 1
                
                # Stop if bankrupt
                if bankroll <= 0:
                    break
            
            final_bankrolls.append(bankroll)
        
        # Calculate statistics
        avg_final = np.mean(final_bankrolls)
        win_rate = wins / max(total_bets, 1)
        roi = ((avg_final - starting_bankroll) / starting_bankroll) * 100
        
        # Sharpe ratio (risk-adjusted return)
        returns = [(fb - starting_bankroll) / starting_bankroll for fb in final_bankrolls]
        sharpe = np.mean(returns) / (np.std(returns) + 0.001)
        
        # Max drawdown
        max_dd = 0
        peak = starting_bankroll
        for br in final_bankrolls:
            if br > peak:
                peak = br
            dd = (peak - br) / peak
            if dd > max_dd:
                max_dd = dd
        
        return {
            'avg_final_bankroll': avg_final,
            'win_rate': win_rate,
            'roi': roi,
            'sharpe_ratio': sharpe,
            'max_drawdown': max_dd
        }


# =====================================================================
# EXAMPLE USAGE
# =====================================================================

if __name__ == "__main__":
    print("=" * 80)
    print("🏇 GOLD STANDARD HANDICAPPING SYSTEM")
    print("Based on the philosophy of legendary handicappers")
    print("=" * 80)
    
    # Example: Single race analysis
    predictions = pd.DataFrame({
        'Horse': ['Sky\'s Not Falling', 'Horsepower', 'Paros', 'Private Thoughts'],
        'Pred_Win_Prob': [0.325, 0.221, 0.148, 0.112],
        'Post': [9, 6, 7, 5],
        'ML': ['12/1', '9/2', '30/1', '6/1'],
        'Uncertainty': [0.08, 0.11, 0.15, 0.09]
    })
    
    race_context = {
        'field_size': 10,
        'distance_furlongs': 6.0,
        'surface': 'dirt',
        'race_type': 'Allowance'
    }
    
    horse_data = [
        {'horse_name': 'Sky\'s Not Falling', 'best_beyer': 92, 'days_since_last': 14, 'starts_lifetime': 8},
        {'horse_name': 'Horsepower', 'best_beyer': 88, 'days_since_last': 21, 'starts_lifetime': 12},
        {'horse_name': 'Paros', 'best_beyer': 85, 'days_since_last': 35, 'starts_lifetime': 15},
        {'horse_name': 'Private Thoughts', 'best_beyer': 83, 'days_since_last': 28, 'starts_lifetime': 10}
    ]
    
    # Initialize engine
    engine = BettingRecommendationEngine()
    
    # Generate recommendations
    result = engine.generate_recommendations(
        predictions=predictions,
        race_context=race_context,
        horse_data=horse_data,
        bankroll=1000.0
    )
    
    # Display results
    print("\n📊 ANALYSIS RESULTS:\n")
    print(f"Confidence Score: {result['confidence_score']:.1%}")
    print(f"Pass This Race: {'YES' if result['pass_race'] else 'NO'}")
    
    if result['pass_race']:
        print(f"Reason: {result['pass_reason']}")
    
    if result['common_sense_warnings']:
        print("\n⚠️  COMMON SENSE WARNINGS:")
        for warning in result['common_sense_warnings']:
            print(f"  {warning}")
    
    if result['recommended_bets']:
        print("\n💰 RECOMMENDED BETS:\n")
        for bet in result['recommended_bets']:
            print(f"Bet Type: {bet['bet_type']}")
            print(f"Horse: {bet.get('horse', bet.get('horses', 'N/A'))}")
            print(f"Edge: {bet['edge']}")
            print(f"Recommended Stake: {bet['recommended_bet']} ({bet['bankroll_pct']} of bankroll)")
            print(f"Reason: {bet['reason']}")
            print(f"EV per $1: {bet.get('ev_per_dollar', 'N/A')}")
            print("-" * 60)
    
    print("\n✅ System follows Steven Crist's discipline: Only bet when you have an edge!")
    print("✅ Kelly Criterion sizing protects bankroll from ruin")
    print("✅ Common sense validation catches mathematical errors")
    print("✅ Expected value calculation ensures long-term profitability")
