"""
BAYESIAN RATING FRAMEWORK - Elite Mathematical Enhancement
=============================================================

Adds uncertainty quantification to all rating components using Bayesian inference.
Each rating now returns (mean, standard_deviation) tuple representing confidence.

Mathematical Foundation:
------------------------
Uses conjugate priors (Normal-Normal) for tractable posterior updates:
    Prior: μ ~ N(μ₀, σ₀²)
    Data: X₁,...,Xₙ ~ N(μ, σ²)
    Posterior: μ | X ~ N(μₙ, σₙ²)

    where:
        σₙ² = 1 / (1/σ₀² + n/σ²)  [posterior precision]
        μₙ = σₙ² * (μ₀/σ₀² + nX̄/σ²)  [weighted average]

Integration: Drop-in replacement for existing rating calculations in unified_rating_engine.py
"""

from dataclasses import dataclass

import numpy as np


@dataclass
class BayesianRating:
    """
    Represents a rating with uncertainty

    Attributes:
        mean: Point estimate (current rating value)
        std: Standard deviation (confidence/uncertainty)
        confidence_level: 0-1 score (high confidence = low std)
    """

    mean: float
    std: float

    @property
    def confidence_level(self) -> float:
        """
        Convert std to confidence score (0-1)
        High std → low confidence, Low std → high confidence
        """
        # Sigmoid transformation: confidence = 1 / (1 + std)
        return 1.0 / (1.0 + self.std)

    @property
    def confidence_interval_95(self) -> tuple[float, float]:
        """95% confidence interval for rating"""
        margin = 1.96 * self.std
        return (self.mean - margin, self.mean + margin)

    def sample(self, n_samples: int = 1) -> np.ndarray:
        """Draw random samples from rating distribution"""
        return np.random.normal(self.mean, self.std, size=n_samples)


class BayesianComponentRater:
    """
    Bayesian rating calculator for individual components

    Uses empirical Bayes: Prior parameters learned from historical data
    """

    def __init__(self, component_name: str):
        """
        Args:
            component_name: 'class', 'form', 'speed', 'pace', 'style', 'post'
        """
        self.component_name = component_name

        # Empirical priors (learned from historical races)
        # These are population means/stds for each component
        self.PRIORS = {
            "class": {"mean": 0.0, "std": 2.0},  # Range: -3 to +6
            "form": {"mean": 0.0, "std": 1.5},  # Range: -3 to +3
            "speed": {"mean": 0.0, "std": 1.0},  # Range: -2 to +2
            "pace": {"mean": 0.0, "std": 1.5},  # Range: -3 to +3
            "style": {"mean": 0.0, "std": 0.5},  # Range: -0.5 to +0.8
            "post": {"mean": 0.0, "std": 0.3},  # Range: -0.5 to +0.5
        }

    def calculate_with_uncertainty(
        self, point_estimate: float, data_quality_indicators: dict
    ) -> BayesianRating:
        """
        Convert deterministic rating to Bayesian rating with uncertainty

        Args:
            point_estimate: Current deterministic rating value
            data_quality_indicators: Dict with keys:
                - 'n_data_points': Number of races used (more = lower uncertainty)
                - 'data_variance': Variance in recent performances
                - 'parsing_confidence': Parser confidence (0-1)
                - 'time_since_last': Days since last race (more = higher uncertainty)

        Returns:
            BayesianRating with (mean, std)
        """
        # Get prior for this component
        prior = self.PRIORS[self.component_name]
        prior_mean = prior["mean"]
        prior_std = prior["std"]

        # Adjust uncertainty based on data quality
        data_uncertainty = self._calculate_data_uncertainty(data_quality_indicators)

        # Bayesian update: Combine prior with point estimate
        # Treat point_estimate as single observation from likelihood
        posterior_precision = (1 / prior_std**2) + (1 / data_uncertainty**2)
        posterior_std = np.sqrt(1 / posterior_precision)

        posterior_mean = posterior_std**2 * (
            (prior_mean / prior_std**2) + (point_estimate / data_uncertainty**2)
        )

        return BayesianRating(mean=posterior_mean, std=posterior_std)

    def _calculate_data_uncertainty(self, indicators: dict) -> float:
        """
        Estimate data uncertainty from quality indicators

        Higher values = more uncertainty = less confidence in rating

        CALIBRATION FIX v3 (Feb 7, 2026 - Race 4 Oaklawn tuning):
        v1: Original - crushed components to ~7% of deterministic
        v2: Reduced factors - retained ~54% of deterministic
        v3: Added 0.4x dampening - now retains ~88% of deterministic

        The core issue: deterministic calcs (class, speed, form, pace) are
        well-calibrated but the Bayesian prior (mean=0) pulls everything to
        zero. We need data_uncertainty << prior_std so the likelihood dominates.

        Race 4 validation: Winnemac Avenue had 0.02 class, 0.01 speed despite
        being a legitimate 7/2 contender who ran 2nd. Over-shrinkage made
        ALL horses look similar, causing bonuses to dominate rankings.
        """
        base_uncertainty = self.PRIORS[self.component_name]["std"]

        # Factor 1: Sample size (more races = lower uncertainty)
        n = indicators.get("n_data_points", 3)
        sample_size_factor = np.sqrt(max(n, 1)) / 2.0
        sample_size_factor = np.clip(sample_size_factor, 0.7, 3.0)

        # Factor 2: Data variance (erratic performances = higher uncertainty)
        variance = indicators.get("data_variance", 1.0)
        # Guard against NaN/negative variance (belt-and-suspenders)
        if not np.isfinite(variance) or variance < 0:
            variance = 1.0
        variance_factor = 1.0 + np.sqrt(variance) * 0.10  # v3: 0.15 -> 0.10
        variance_factor = np.clip(variance_factor, 0.8, 1.5)  # v3: max 2.0 -> 1.5

        # Factor 3: Parsing confidence
        parsing_conf = indicators.get("parsing_confidence", 0.9)
        parsing_factor = 1.0 + (1.0 - parsing_conf) * 0.3  # v3: 0.5 -> 0.3
        # conf=0.9 -> 1.03, conf=0.7 -> 1.09, conf=0.5 -> 1.15

        # Factor 4: Recency (long layoff = higher uncertainty)
        days = indicators.get("time_since_last", 30) or 30
        recency_factor = 1.0 + (days / 500.0)  # v3: /365 -> /500 (gentler)
        recency_factor = np.clip(recency_factor, 1.0, 1.3)  # v3: max 1.5 -> 1.3

        # Combine factors multiplicatively
        raw_uncertainty = (
            base_uncertainty
            * variance_factor
            / sample_size_factor
            * parsing_factor
            * recency_factor
        )

        # v3 CRITICAL: Apply global dampening so data dominates prior
        # Without this, data_unc ~= base_std -> 50% shrinkage toward prior(0)
        # v4 (Feb 21, 2026): Raised from 0.4 to 0.55 — horses with limited race
        # history (e.g., 2-3 starts) benefit more from Bayesian adjustment.
        # 0.55 dampening -> ~82% retention of deterministic value (was ~88%)
        DAMPENING = 0.55
        total_uncertainty = raw_uncertainty * DAMPENING

        return float(np.clip(total_uncertainty, 0.1, 5.0))


class BayesianRatingAggregator:
    """
    Aggregate multiple Bayesian ratings into final rating with uncertainty
    """

    @staticmethod
    def weighted_sum(ratings: list[tuple[BayesianRating, float]]) -> BayesianRating:
        """
        Calculate weighted sum of Bayesian ratings

        Args:
            ratings: List of (BayesianRating, weight) tuples

        Returns:
            Aggregated BayesianRating

        Mathematical derivation:
            Let Y = Σ w_i * X_i where X_i ~ N(μ_i, σ_i²)
            Then Y ~ N(Σ w_i*μ_i, Σ w_i²*σ_i²)  [assuming independence]
        """
        # Weighted mean
        weighted_mean = sum(rating.mean * weight for rating, weight in ratings)

        # Propagate uncertainty (variance adds for independent variables)
        weighted_variance = sum(
            (weight * rating.std) ** 2 for rating, weight in ratings
        )
        weighted_std = np.sqrt(weighted_variance)

        return BayesianRating(mean=weighted_mean, std=weighted_std)

    @staticmethod
    def calibrate_to_softmax(
        ratings: list[BayesianRating], tau: float = 3.0
    ) -> list[tuple[float, float]]:
        """
        Convert Bayesian ratings to win probabilities with confidence intervals

        Args:
            ratings: List of BayesianRating for each horse
            tau: Softmax temperature parameter

        Returns:
            List of (win_prob, confidence_interval_width) for each horse
        """
        results = []

        for rating in ratings:
            # Point estimate probability (standard softmax)
            exp_ratings = np.array([r.mean for r in ratings])
            point_prob = np.exp(rating.mean / tau) / np.sum(np.exp(exp_ratings / tau))

            # Confidence interval via Monte Carlo sampling
            n_samples = 1000
            prob_samples = []

            for _ in range(n_samples):
                # Sample ratings for all horses
                sampled_ratings = np.array([r.sample(1)[0] for r in ratings])
                # Calculate probability from sampled ratings
                exp_sampled = np.exp(sampled_ratings / tau)
                prob_sample = exp_sampled / exp_sampled.sum()
                # Extract this horse's probability
                horse_idx = ratings.index(rating)
                prob_samples.append(prob_sample[horse_idx])

            # Calculate confidence interval
            ci_low = np.percentile(prob_samples, 2.5)
            ci_high = np.percentile(prob_samples, 97.5)
            ci_width = ci_high - ci_low

            results.append((point_prob, ci_width))

        return results


# ============================================================================
# INTEGRATION FUNCTIONS - Drop-in replacements for unified_rating_engine.py
# ============================================================================


def enhance_rating_with_bayesian_uncertainty(
    original_rating: float,
    component_type: str,
    horse_data: dict,
    parsing_confidence: float = 0.9,
) -> BayesianRating:
    """
    Wrapper function to add Bayesian uncertainty to existing ratings

    INTEGRATION: Call this after calculating any component rating

    Example:
        # Old code:
        cform = self._calc_form(horse)  # Returns float

        # New code:
        cform_deterministic = self._calc_form(horse)
        cform_bayesian = enhance_rating_with_bayesian_uncertainty(
            cform_deterministic, 'form', horse.__dict__, parsing_confidence
        )
        cform = cform_bayesian.mean  # Use mean for backwards compatibility
        cform_std = cform_bayesian.std  # New: uncertainty measure

    Args:
        original_rating: Deterministic rating from existing calculation
        component_type: 'class', 'form', 'speed', 'pace', 'style', or 'post'
        horse_data: Dict with horse attributes (recent_finishes, days_since_last, etc.)
        parsing_confidence: Parser confidence score (0-1)

    Returns:
        BayesianRating with mean and std
    """
    rater = BayesianComponentRater(component_type)

    # Extract data quality indicators from horse_data
    # CRITICAL FIX (Feb 18, 2026): Guard np.var() against empty/single-element lists.
    # np.var([]) returns NaN which propagated through ALL Bayesian calculations,
    # zeroing out component scores for horses with sparse race histories
    # (e.g., Don't Get Cute with 2 starts, Just a Wizard with 4 starts).
    finishes = horse_data.get("recent_finishes") or []
    if len(finishes) < 2:
        # Default variance for horses with 0-1 finishes (neutral uncertainty)
        data_var = 4.0
    else:
        data_var = float(np.var(finishes))
        if np.isnan(data_var) or not np.isfinite(data_var):
            data_var = 4.0
    indicators = {
        "n_data_points": len(finishes) or 3,
        "data_variance": data_var,
        "parsing_confidence": parsing_confidence,
        "time_since_last": horse_data.get("days_since_last", 30) or 30,
    }

    return rater.calculate_with_uncertainty(original_rating, indicators)


def calculate_final_rating_with_uncertainty(
    component_ratings: dict, component_weights: dict
) -> tuple[float, float]:
    """
    Calculate weighted final rating with propagated uncertainty

    INTEGRATION: Replace final rating calculation in unified engine

    Example:
        # Old code:
        final_rating = (
            (cclass * 2.5) +
            (cform * 1.8) +
            (cspeed * 2.0) +
            ...
        )

        # New code:
        component_ratings = {
            'class': cclass_bayesian,  # BayesianRating objects
            'form': cform_bayesian,
            'speed': cspeed_bayesian,
            ...
        }
        component_weights = {
            'class': 2.5,
            'form': 1.8,
            'speed': 2.0,
            ...
        }
        final_mean, final_std = calculate_final_rating_with_uncertainty(
            component_ratings, component_weights
        )

    Args:
        component_ratings: Dict of {component_name: BayesianRating}
        component_weights: Dict of {component_name: weight_value}

    Returns:
        Tuple of (final_rating_mean, final_rating_std)
    """
    ratings_with_weights = [
        (rating, component_weights[name]) for name, rating in component_ratings.items()
    ]

    final_bayesian = BayesianRatingAggregator.weighted_sum(ratings_with_weights)
    return final_bayesian.mean, final_bayesian.std


# ============================================================================
# USAGE EXAMPLE - How to integrate into unified_rating_engine.py
# ============================================================================


def example_integration():
    """
    Example showing how to integrate Bayesian uncertainty into the rating engine.
    This is reference documentation — not called in production.
    """
    # BEFORE (existing simplified code):
    # cclass = calc_class(horse, today_purse, today_race_type)
    # cform = calc_form(horse)
    # cspeed = calc_speed(horse, horses_in_race)
    # final_rating = (cclass * 2.5) + (cform * 1.8) + (cspeed * 2.0)

    # AFTER (with Bayesian enhancement):
    from bayesian_rating_framework import (
        calculate_final_rating_with_uncertainty,
        enhance_rating_with_bayesian_uncertainty,
    )

    # Example deterministic values (would come from rating engine in production)
    cclass_val = 3.5
    cform_val = 2.1
    cspeed_val = 4.0
    horse_data = {"recent_finishes": [1, 2, 3], "days_since_last": 21}

    # Enhance with Bayesian uncertainty
    cclass_bayes = enhance_rating_with_bayesian_uncertainty(
        cclass_val, "class", horse_data, parsing_confidence=0.94
    )
    cform_bayes = enhance_rating_with_bayesian_uncertainty(
        cform_val, "form", horse_data, parsing_confidence=0.94
    )
    cspeed_bayes = enhance_rating_with_bayesian_uncertainty(
        cspeed_val, "speed", horse_data, parsing_confidence=0.94
    )

    # Aggregate with uncertainty propagation
    component_ratings = {
        "class": cclass_bayes,
        "form": cform_bayes,
        "speed": cspeed_bayes,
    }
    component_weights = {"class": 2.5, "form": 1.8, "speed": 2.0}

    final_mean, final_std = calculate_final_rating_with_uncertainty(
        component_ratings, component_weights
    )

    # Now you have both point estimate AND uncertainty!
    print(f"Final rating: {final_mean:.2f} ± {final_std:.2f}")
    print(f"Confidence: {(1.0 / (1.0 + final_std)):.1%}")


if __name__ == "__main__":
    # Demo: Compare deterministic vs Bayesian ratings
    print("=" * 70)
    print("BAYESIAN RATING FRAMEWORK - Demo")
    print("=" * 70)

    # Scenario: Two horses with same point estimate but different data quality

    # Horse A: Consistent recent form, well-documented
    horse_a_rating = 5.0
    horse_a_data = {
        "recent_finishes": [1, 2, 1, 2, 3],  # 5 races, consistent
        "days_since_last": 21,
        "parsing_confidence": 0.95,
    }

    # Horse B: Erratic form, limited data
    horse_b_rating = 5.0  # Same point estimate!
    horse_b_data = {
        "recent_finishes": [1, 9, 2],  # 3 races, erratic
        "days_since_last": 120,
        "parsing_confidence": 0.70,
    }

    # Calculate Bayesian ratings
    rater = BayesianComponentRater("form")

    horse_a_bayes = rater.calculate_with_uncertainty(
        horse_a_rating,
        {
            "n_data_points": 5,
            "data_variance": np.var([1, 2, 1, 2, 3]),
            "parsing_confidence": 0.95,
            "time_since_last": 21,
        },
    )

    horse_b_bayes = rater.calculate_with_uncertainty(
        horse_b_rating,
        {
            "n_data_points": 3,
            "data_variance": np.var([1, 9, 2]),
            "parsing_confidence": 0.70,
            "time_since_last": 120,
        },
    )

    print("\nHorse A (Consistent, Recent):")
    print(f"  Rating: {horse_a_bayes.mean:.2f} ± {horse_a_bayes.std:.2f}")
    print(f"  Confidence: {horse_a_bayes.confidence_level:.1%}")
    print(f"  95% CI: {horse_a_bayes.confidence_interval_95}")

    print("\nHorse B (Erratic, Long Layoff):")
    print(f"  Rating: {horse_b_bayes.mean:.2f} ± {horse_b_bayes.std:.2f}")
    print(f"  Confidence: {horse_b_bayes.confidence_level:.1%}")
    print(f"  95% CI: {horse_b_bayes.confidence_interval_95}")

    print(f"\n✅ Both have same point estimate ({horse_a_rating})")
    print(
        f"✅ But Horse B has {horse_b_bayes.std / horse_a_bayes.std:.1f}x more uncertainty"
    )
    print("✅ This reflects lower confidence in erratic, long-laid-off horses")

    print("\n" + "=" * 70)
