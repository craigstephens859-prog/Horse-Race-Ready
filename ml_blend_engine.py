"""
ML BLEND ENGINE — PyTorch Model Integration for Live Predictions
================================================================

Loads the latest retrained Plackett-Luce RankingNN checkpoint and
produces per-horse ranking scores that blend into the existing
angle-weight prediction pipeline as a 7th weighted component.

Architecture:
    final_score = w1*Class + w2*Speed + w3*Form + w4*Pace
                + w5*Style + w6*Post  + w7*ML_Score   (+ angles bonus)

The ML score is the raw output of the trained neural ranker,
z-score normalised across the field to match the [-3, +3] range
of the other components.

Usage:
    from ml_blend_engine import MLBlendEngine
    ml_engine = MLBlendEngine("gold_high_iq.db")
    scores = ml_engine.score_horses(horses_features_df)
    # scores: dict[horse_name, float]  — normalised ML ranking scores

Author: Senior PyTorch + Horse Racing Architect
Date: February 20, 2026
"""

import glob
import logging
import os

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════
# MODEL ARCHITECTURE (must match retrain_model.py exactly)
# ═══════════════════════════════════════════════════════════


class RankingNN(nn.Module):
    """Compact MLP for listwise ranking — mirror of retrain_model.RankingNN."""

    def __init__(self, n_features: int, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.4),
            nn.Linear(hidden_dim, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════
# ML BLEND ENGINE
# ═══════════════════════════════════════════════════════════


class MLBlendEngine:
    """
    Loads the latest trained PyTorch ranking model and provides
    normalised scores for blending into the unified rating engine.

    Thread-safe: model is loaded once and kept in eval mode.
    Graceful fallback: if no model exists, returns empty scores.
    """

    def __init__(self, db_path: str = "gold_high_iq.db", model_dir: str = "models"):
        self.db_path = db_path
        self.model_dir = model_dir
        self.model = None
        self.checkpoint = None
        self.feat_mean = None
        self.feat_std = None
        self.feature_cols = None
        self.n_features = 0
        self.model_path = None
        self._load_latest_model()

    def _load_latest_model(self):
        """Load the most recent .pt checkpoint from the models/ directory."""
        if not os.path.isdir(self.model_dir):
            logger.info("ML Blend: No models/ directory found — ML scoring disabled")
            return

        # Find latest .pt file by modification time
        pt_files = glob.glob(os.path.join(self.model_dir, "ranking_model_*.pt"))
        if not pt_files:
            logger.info("ML Blend: No trained models found — ML scoring disabled")
            return

        latest_pt = max(pt_files, key=os.path.getmtime)
        try:
            checkpoint = torch.load(latest_pt, map_location="cpu", weights_only=False)

            n_features = checkpoint["n_features"]
            hidden_dim = checkpoint.get("hidden_dim", 64)

            model = RankingNN(n_features=n_features, hidden_dim=hidden_dim)
            model.load_state_dict(checkpoint["model_state_dict"])
            model.eval()

            self.model = model
            self.checkpoint = checkpoint
            self.n_features = n_features
            self.feature_cols = checkpoint.get("feature_cols", [])
            self.model_path = latest_pt

            # Load feature standardisation stats
            feat_mean = checkpoint.get("feat_mean")
            feat_std = checkpoint.get("feat_std")
            if feat_mean is not None:
                self.feat_mean = np.array(feat_mean, dtype=np.float32)
            if feat_std is not None:
                self.feat_std = np.array(feat_std, dtype=np.float32)
                # Prevent division by zero
                self.feat_std[self.feat_std < 1e-8] = 1.0

            logger.info(
                f"ML Blend: Loaded model from {os.path.basename(latest_pt)} "
                f"({n_features} features, hidden={hidden_dim})"
            )

        except Exception as e:
            logger.warning(f"ML Blend: Failed to load model {latest_pt}: {e}")
            self.model = None

    @property
    def is_available(self) -> bool:
        """Whether a trained model is loaded and ready for scoring."""
        return self.model is not None

    def get_model_info(self) -> dict:
        """Return metadata about the loaded model for UI display."""
        if not self.is_available:
            return {"available": False}

        val_metrics = self.checkpoint.get("val_metrics", {})
        training_config = self.checkpoint.get("training_config", {})
        return {
            "available": True,
            "model_path": os.path.basename(self.model_path),
            "n_features": self.n_features,
            "hidden_dim": self.checkpoint.get("hidden_dim", 64),
            "best_epoch": self.checkpoint.get("best_epoch", "?"),
            "total_epochs": self.checkpoint.get("total_epochs_run", "?"),
            "val_winner_acc": val_metrics.get("winner_accuracy", 0),
            "val_top3_acc": val_metrics.get("top3_accuracy", 0),
            "val_top4_acc": val_metrics.get("top4_accuracy", 0),
            "train_races": training_config.get("train_races", 0),
            "val_races": training_config.get("val_races", 0),
        }

    def score_horses(
        self, horse_features: dict[str, dict[str, float]]
    ) -> dict[str, float]:
        """
        Score a field of horses using the trained PyTorch model.

        Args:
            horse_features: {horse_name: {feature_name: value, ...}, ...}
                           Feature names must match self.feature_cols.
                           Missing features default to 0.0.

        Returns:
            {horse_name: normalised_ml_score} where scores are z-normalised
            to approximately [-3, +3] to match the other rating components.
            Returns {} if no model is loaded.
        """
        if not self.is_available or not horse_features:
            return {}

        horse_names = list(horse_features.keys())
        n_horses = len(horse_names)

        if n_horses < 2:
            return {}

        # Build feature matrix (n_horses, n_features)
        feature_matrix = np.zeros((n_horses, self.n_features), dtype=np.float32)
        for i, name in enumerate(horse_names):
            feats = horse_features[name]
            for j, col in enumerate(self.feature_cols):
                feature_matrix[i, j] = float(feats.get(col, 0.0))

        # Apply z-score standardisation using training stats
        if self.feat_mean is not None and self.feat_std is not None:
            feature_matrix = (feature_matrix - self.feat_mean) / self.feat_std

        # Run inference
        with torch.no_grad():
            x = torch.FloatTensor(feature_matrix)
            raw_scores = self.model(x).numpy()  # (n_horses,)

        # Normalise raw scores to [-3, +3] range (matching other components)
        # Use z-score normalisation across the field
        score_mean = raw_scores.mean()
        score_std = raw_scores.std()
        if score_std < 1e-8:
            score_std = 1.0  # Prevent division by zero

        normalised = (raw_scores - score_mean) / score_std
        # Clip to match component ranges
        normalised = np.clip(normalised, -3.0, 3.0)

        return {name: float(normalised[i]) for i, name in enumerate(horse_names)}

    def score_from_gold_features(
        self, race_horses: list[dict], horse_names: list[str] | None = None
    ) -> dict[str, float]:
        """
        Score horses using the same feature set the model was trained on.
        Accepts the raw horse dicts from gold_high_iq / horses_analyzed data.

        Args:
            race_horses: list of dicts with keys matching gold_high_iq columns
                        (odds, prime_power, last_speed_rating, etc.)
            horse_names: Optional list of horse names (same order as race_horses).
                        If not provided, uses "horse_name" or "Horse" key from dicts.

        Returns:
            {horse_name: normalised_ml_score}
        """
        horse_features = {}
        for i, h in enumerate(race_horses):
            if horse_names and i < len(horse_names):
                name = horse_names[i]
            else:
                name = h.get("horse_name", h.get("Horse", f"horse_{i}"))
            if not name:
                continue
            features = {}
            for col in self.feature_cols:
                val = h.get(col, 0.0)
                try:
                    features[col] = float(val) if val is not None else 0.0
                except (ValueError, TypeError):
                    features[col] = 0.0
            horse_features[name] = features

        return self.score_horses(horse_features)

    def reload_model(self):
        """Force reload of the latest model (call after retraining)."""
        self._load_latest_model()
