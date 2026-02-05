"""
PHASE 3: Advanced Probability Engine
Bayesian confidence intervals, Place/Show modeling, and exotic bet probabilities
"""

import numpy as np
from scipy import stats
from scipy.special import comb
from typing import Dict, List, Tuple, Any
import pandas as pd


class Phase3ProbabilityEngine:
    """
    Advanced probability modeling for horse racing predictions.
    
    Features:
    - Bayesian confidence intervals on win probabilities
    - Place/Show probability distributions using order statistics
    - Exacta/Trifecta probability calculations
    - Fixed $50 bankroll per-race analysis
    """
    
    def __init__(self, bankroll: float = 50.0):
        """
        Initialize Phase 3 engine.
        
        Args:
            bankroll: Fixed betting bankroll per race (default $50)
        """
        self.bankroll = bankroll
        
    def calculate_confidence_intervals(
        self, 
        win_probs: np.ndarray,
        confidence_level: float = 0.95,
        sample_size: int = 100
    ) -> Dict[str, Any]:
        """
        Calculate Bayesian confidence intervals for win probabilities.
        
        Uses Dirichlet distribution (conjugate prior for multinomial) to model
        uncertainty in win probability estimates.
        
        Args:
            win_probs: Array of win probabilities for each horse
            confidence_level: Confidence level (default 0.95 for 95% CI)
            sample_size: Effective sample size for prior strength
            
        Returns:
            Dictionary with mean, std, and confidence intervals per horse
        """
        n_horses = len(win_probs)
        
        # Dirichlet parameters (add pseudo-counts for prior)
        # Higher sample_size = more confidence in current estimates
        alpha = win_probs * sample_size + 1  # Add 1 for uniform prior
        
        # Generate Monte Carlo samples from Dirichlet distribution
        n_samples = 10000
        samples = np.random.dirichlet(alpha, size=n_samples)
        
        # Calculate statistics
        means = samples.mean(axis=0)
        stds = samples.std(axis=0)
        
        # Confidence intervals
        lower_percentile = (1 - confidence_level) / 2 * 100
        upper_percentile = (1 + confidence_level) / 2 * 100
        
        lower_bounds = np.percentile(samples, lower_percentile, axis=0)
        upper_bounds = np.percentile(samples, upper_percentile, axis=0)
        
        return {
            'mean': means,
            'std': stds,
            'lower_bound': lower_bounds,
            'upper_bound': upper_bounds,
            'confidence_level': confidence_level,
            'samples': samples  # For advanced analysis
        }
    
    def calculate_place_show_probabilities(
        self,
        win_probs: np.ndarray,
        n_simulations: int = 10000
    ) -> Dict[str, np.ndarray]:
        """
        Calculate Place (top 2) and Show (top 3) probabilities using Monte Carlo.
        
        Uses order statistics from multinomial sampling to estimate finish positions.
        
        Args:
            win_probs: Win probabilities for each horse
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            Dictionary with win, place, show probabilities per horse
        """
        n_horses = len(win_probs)
        
        # Initialize counters
        win_count = np.zeros(n_horses)
        place_count = np.zeros(n_horses)  # Top 2
        show_count = np.zeros(n_horses)   # Top 3
        
        # Monte Carlo simulation
        for _ in range(n_simulations):
            # Sample race outcome based on win probabilities
            # Generate random utilities and rank them
            utilities = np.random.gumbel(loc=np.log(win_probs), scale=1.0)
            finish_order = np.argsort(-utilities)  # Descending order
            
            # Count finishes
            win_count[finish_order[0]] += 1
            place_count[finish_order[:2]] += 1
            if n_horses >= 3:
                show_count[finish_order[:3]] += 1
        
        # Convert to probabilities
        return {
            'win': win_count / n_simulations,
            'place': place_count / n_simulations,
            'show': show_count / n_simulations if n_horses >= 3 else np.zeros(n_horses)
        }
    
    def calculate_exacta_probabilities(
        self,
        win_probs: np.ndarray,
        top_n: int = 10,
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Calculate exacta (1st-2nd) probabilities for top combinations.
        
        Args:
            win_probs: Win probabilities for each horse
            top_n: Number of top exacta combinations to return
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            DataFrame with exacta combinations and probabilities
        """
        n_horses = len(win_probs)
        
        # Initialize exacta matrix
        exacta_counts = np.zeros((n_horses, n_horses))
        
        # Monte Carlo simulation
        for _ in range(n_simulations):
            utilities = np.random.gumbel(loc=np.log(win_probs), scale=1.0)
            finish_order = np.argsort(-utilities)
            
            # Record exacta (first two finishers)
            first = finish_order[0]
            second = finish_order[1]
            exacta_counts[first, second] += 1
        
        # Convert to probabilities
        exacta_probs = exacta_counts / n_simulations
        
        # Get top N combinations
        top_indices = np.argsort(-exacta_probs.flatten())[:top_n]
        top_combos = []
        
        for idx in top_indices:
            first = idx // n_horses
            second = idx % n_horses
            prob = exacta_probs[first, second]
            
            if prob > 0:  # Only include non-zero probabilities
                top_combos.append({
                    'first': first + 1,  # 1-indexed for horse numbers
                    'second': second + 1,
                    'probability': prob,
                    'confidence': 'high' if prob > 0.05 else 'medium' if prob > 0.02 else 'low'
                })
        
        return pd.DataFrame(top_combos)
    
    def calculate_trifecta_probabilities(
        self,
        win_probs: np.ndarray,
        top_n: int = 10,
        n_simulations: int = 10000
    ) -> pd.DataFrame:
        """
        Calculate trifecta (1st-2nd-3rd) probabilities for top combinations.
        
        Args:
            win_probs: Win probabilities for each horse
            top_n: Number of top trifecta combinations to return
            n_simulations: Number of Monte Carlo simulations
            
        Returns:
            DataFrame with trifecta combinations and probabilities
        """
        n_horses = len(win_probs)
        
        if n_horses < 3:
            return pd.DataFrame()  # Not enough horses for trifecta
        
        # Dictionary to store trifecta counts
        trifecta_counts = {}
        
        # Monte Carlo simulation
        for _ in range(n_simulations):
            utilities = np.random.gumbel(loc=np.log(win_probs), scale=1.0)
            finish_order = np.argsort(-utilities)
            
            # Record trifecta (first three finishers)
            first = finish_order[0]
            second = finish_order[1]
            third = finish_order[2]
            
            combo = (first, second, third)
            trifecta_counts[combo] = trifecta_counts.get(combo, 0) + 1
        
        # Convert to probabilities and get top N
        trifecta_list = []
        for combo, count in trifecta_counts.items():
            prob = count / n_simulations
            trifecta_list.append({
                'first': combo[0] + 1,  # 1-indexed
                'second': combo[1] + 1,
                'third': combo[2] + 1,
                'probability': prob,
                'confidence': 'high' if prob > 0.02 else 'medium' if prob > 0.01 else 'low'
            })
        
        # Sort by probability and return top N
        trifecta_df = pd.DataFrame(trifecta_list)
        if len(trifecta_df) > 0:
            trifecta_df = trifecta_df.sort_values('probability', ascending=False).head(top_n)
        
        return trifecta_df
    
    def analyze_race_comprehensive(
        self,
        win_probs: np.ndarray,
        horse_names: List[str] = None,
        confidence_level: float = 0.95
    ) -> Dict[str, Any]:
        """
        Comprehensive Phase 3 analysis for a single race.
        
        Args:
            win_probs: Win probabilities for each horse
            horse_names: Optional list of horse names
            confidence_level: Confidence level for intervals
            
        Returns:
            Complete Phase 3 analysis dictionary
        """
        n_horses = len(win_probs)
        
        if horse_names is None:
            horse_names = [f"Horse #{i+1}" for i in range(n_horses)]
        
        # 1. Confidence intervals
        ci_results = self.calculate_confidence_intervals(win_probs, confidence_level)
        
        # 2. Place/Show probabilities
        place_show = self.calculate_place_show_probabilities(win_probs)
        
        # 3. Exacta probabilities
        exacta_df = self.calculate_exacta_probabilities(win_probs, top_n=10)
        
        # 4. Trifecta probabilities
        trifecta_df = self.calculate_trifecta_probabilities(win_probs, top_n=10)
        
        # Build comprehensive results
        results = {
            'win_probabilities': {
                'horse': horse_names,
                'mean': ci_results['mean'],
                'std': ci_results['std'],
                'lower_95': ci_results['lower_bound'],
                'upper_95': ci_results['upper_bound']
            },
            'place_show': {
                'horse': horse_names,
                'win_prob': place_show['win'],
                'place_prob': place_show['place'],
                'show_prob': place_show['show']
            },
            'exacta_top10': exacta_df,
            'trifecta_top10': trifecta_df,
            'bankroll': self.bankroll,
            'analysis_type': 'Phase 3: Advanced Probability Modeling'
        }
        
        return results


def format_phase3_report(results: Dict[str, Any]) -> str:
    """
    Format Phase 3 results for Classic Report display.
    
    Args:
        results: Results from Phase3ProbabilityEngine.analyze_race_comprehensive()
        
    Returns:
        Formatted string for display
    """
    report = []
    report.append("=" * 70)
    report.append("PHASE 3: ADVANCED PROBABILITY ANALYSIS")
    report.append("=" * 70)
    report.append(f"Bankroll: ${results['bankroll']:.2f} per race")
    report.append("")
    
    # Win probabilities with confidence intervals
    report.append("WIN PROBABILITIES WITH 95% CONFIDENCE INTERVALS:")
    report.append("-" * 70)
    wp = results['win_probabilities']
    for i, horse in enumerate(wp['horse']):
        mean = wp['mean'][i]
        std = wp['std'][i]
        lower = wp['lower_95'][i]
        upper = wp['upper_95'][i]
        report.append(
            f"{horse:20s} {mean:6.1%} ± {std:5.1%}  "
            f"[{lower:5.1%} - {upper:5.1%}]"
        )
    report.append("")
    
    # Place/Show probabilities
    report.append("PLACE & SHOW PROBABILITIES:")
    report.append("-" * 70)
    report.append(f"{'Horse':<20} {'Win':>8} {'Place':>8} {'Show':>8}")
    report.append("-" * 70)
    ps = results['place_show']
    for i, horse in enumerate(ps['horse']):
        report.append(
            f"{horse:<20} {ps['win_prob'][i]:7.1%} {ps['place_prob'][i]:7.1%} "
            f"{ps['show_prob'][i]:7.1%}"
        )
    report.append("")
    
    # Top Exacta combinations
    if len(results['exacta_top10']) > 0:
        report.append("TOP 10 EXACTA COMBINATIONS:")
        report.append("-" * 70)
        report.append(f"{'Combination':<20} {'Probability':>12} {'Confidence':>12}")
        report.append("-" * 70)
        for _, row in results['exacta_top10'].iterrows():
            combo = f"#{row['first']} → #{row['second']}"
            report.append(
                f"{combo:<20} {row['probability']:11.1%} {row['confidence']:>12}"
            )
        report.append("")
    
    # Top Trifecta combinations
    if len(results['trifecta_top10']) > 0:
        report.append("TOP 10 TRIFECTA COMBINATIONS:")
        report.append("-" * 70)
        report.append(f"{'Combination':<25} {'Probability':>12} {'Confidence':>12}")
        report.append("-" * 70)
        for _, row in results['trifecta_top10'].iterrows():
            combo = f"#{row['first']} → #{row['second']} → #{row['third']}"
            report.append(
                f"{combo:<25} {row['probability']:11.1%} {row['confidence']:>12}"
            )
        report.append("")
    
    report.append("=" * 70)
    
    return "\n".join(report)
