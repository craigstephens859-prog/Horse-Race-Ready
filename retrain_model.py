"""
ML MODEL RETRAINING ENGINE
==========================

Retrain the racing prediction model using real race results
from the gold_high_iq database.

Uses PyTorch with Plackett-Luce ranking loss for optimal accuracy.
Optimized for small-dataset horse racing (67–500+ races).

Key design decisions:
  - Batch size = number of RACES per batch (not horses)
  - Feature z-score standardisation (speed ~100 vs jockey_pct ~0.15)
  - Early stopping (patience=15) to prevent overfitting on <100 races
  - Gradient clipping (max_norm=1.0) for PL loss stability
  - Hidden dim 64 (~4.5K params) sized for current dataset

Author: Top-Tier ML Engineer
Date: January 29, 2026
Updated: February 20, 2026 — parameter tuning for 67-race dataset
"""

import logging
import os
import random
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from gold_database_manager import GoldHighIQDatabase

# ═══════════════════════════════════════════════════════════
# REPRODUCIBILITY: Global seed for deterministic training
# Without this, weight init, dropout, and DataLoader shuffling
# produce different results every run — even with same data.
# ═══════════════════════════════════════════════════════════
SEED = 42

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def set_deterministic_seed(seed: int = SEED):
    """Set all random seeds for fully reproducible training."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


# ═══════════════════════════════════════════════════════════
# DATASET
# ═══════════════════════════════════════════════════════════


class RaceDataset(Dataset):
    """PyTorch Dataset for race data with listwise ranking.

    Each item is one race: a variable-length tensor of (n_horses, n_features)
    paired with integer finish positions.  Features are z-score standardised
    using training-set statistics to equalise scale across speed ratings
    (~50-120), percentages (~0-0.3), and counts (~0-50).
    """

    def __init__(
        self,
        features_df: pd.DataFrame,
        labels_df: pd.DataFrame,
        feat_mean: np.ndarray | None = None,
        feat_std: np.ndarray | None = None,
    ):
        """
        Args:
            features_df: DataFrame with race_id, horse_name, and feature columns
            labels_df:   DataFrame with race_id, horse_name, actual_finish_position
            feat_mean:   Optional pre-computed feature means (for val/test sets)
            feat_std:    Optional pre-computed feature stds  (for val/test sets)
        """
        self.races = []

        # Identify feature columns (exclude metadata)
        feature_cols = [
            c
            for c in features_df.columns
            if c
            not in ["race_id", "horse_name", "actual_finish", "actual_finish_position"]
        ]
        self.feature_cols = feature_cols

        # ── Z-score standardisation ──────────────────────────────
        # Compute stats from this DataFrame if not provided (train set).
        # Val/test sets MUST receive the train set's mean/std to prevent
        # data leakage.
        raw = features_df[feature_cols].values.astype(np.float32)

        if feat_mean is None or feat_std is None:
            self.feat_mean = np.nanmean(raw, axis=0)
            self.feat_std = np.nanstd(raw, axis=0)
            # Prevent division by zero for constant features
            self.feat_std[self.feat_std < 1e-8] = 1.0
        else:
            self.feat_mean = feat_mean
            self.feat_std = feat_std

        # Group by race and build tensors
        race_groups = features_df.groupby("race_id")

        for race_id, race_data in race_groups:
            race_labels = labels_df[labels_df["race_id"] == race_id]

            race_combined = race_data.merge(
                race_labels[["horse_name", "actual_finish_position"]],
                on="horse_name",
                how="inner",
            )

            if len(race_combined) < 2:
                continue  # Need at least 2 horses for ranking

            features = race_combined[feature_cols].values.astype(np.float32)
            rankings = race_combined["actual_finish_position"].values.astype(np.int64)

            # Apply z-score: (x - mean) / std
            features = (features - self.feat_mean) / self.feat_std

            self.races.append(
                {
                    "features": torch.FloatTensor(features),
                    "rankings": torch.LongTensor(rankings),
                    "race_id": race_id,
                }
            )

    def __len__(self):
        return len(self.races)

    def __getitem__(self, idx):
        return self.races[idx]


def collate_races(batch):
    """Custom collate: return list of dicts (variable-length races can't stack)."""
    return batch


# ═══════════════════════════════════════════════════════════
# MODEL
# ═══════════════════════════════════════════════════════════


class RankingNN(nn.Module):
    """Compact MLP for listwise ranking.

    Architecture sized for small datasets (67-500 races):
      Linear(n_features -> 64) -> ReLU -> Dropout(0.4)
      Linear(64 -> 64)         -> ReLU -> Dropout(0.4)
      Linear(64 -> 1)          -> score per horse

    ~4.5K params at 27 features.  Scales safely to 1000+ races.
    """

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
        """
        Args:
            x: (n_horses, n_features)
        Returns:
            scores: (n_horses,)  — one scalar score per horse
        """
        return self.net(x).squeeze(-1)


# ═══════════════════════════════════════════════════════════
# PLACKETT-LUCE LOSS
# ═══════════════════════════════════════════════════════════


def plackett_luce_loss(
    scores: torch.Tensor, true_rankings: torch.Tensor
) -> torch.Tensor:
    """
    Plackett-Luce listwise ranking loss (vectorised via reverse cumulative logsumexp).

    The PL model probability of an observed ranking (r1, r2, ..., rn) is:
        P = prod_i  exp(s_ri) / sum_{j>=i} exp(s_rj)

    Negative log-likelihood:
        L = -sum_i [s_ri - logsumexp(s_ri, s_{ri+1}, ..., s_rn)]

    Args:
        scores:        (n_horses,) model output scores
        true_rankings: (n_horses,) actual finish positions (1-based)

    Returns:
        loss: scalar tensor
    """
    # Sort scores into true finish order
    sorted_indices = torch.argsort(true_rankings)
    sorted_scores = scores[sorted_indices]

    n = len(sorted_scores)

    # Vectorised reverse cumulative logsumexp (list-based to avoid inplace ops)
    reversed_scores = sorted_scores.flip(0)
    cum_list = [reversed_scores[0]]
    for i in range(1, n):
        cum_list.append(torch.logaddexp(cum_list[-1], reversed_scores[i]))
    reverse_cum_lse = torch.stack(cum_list).flip(0)

    loss = -(sorted_scores - reverse_cum_lse).sum()
    return loss


# ═══════════════════════════════════════════════════════════
# TRAIN / EVAL FUNCTIONS
# ═══════════════════════════════════════════════════════════


def train_epoch(model, dataloader, optimizer, device, max_grad_norm=1.0):
    """Train for one epoch with gradient clipping.

    Accumulates loss over all races in a batch before stepping,
    giving a true batch gradient rather than per-race stepping.
    This is more stable for Plackett-Luce with variable field sizes.
    """
    model.train()
    total_loss = 0.0
    n_races = 0

    for batch in dataloader:
        optimizer.zero_grad()
        batch_loss = torch.tensor(0.0, device=device, requires_grad=True)

        for race in batch:
            features = race["features"].to(device)
            rankings = race["rankings"].to(device)

            scores = model(features)
            loss = plackett_luce_loss(scores, rankings)
            batch_loss = batch_loss + loss
            n_races += 1

        # Average over races in batch for consistent gradient magnitude
        batch_loss = batch_loss / len(batch)
        batch_loss.backward()

        # Clip gradients — PL loss on 14-horse fields can spike
        nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)

        optimizer.step()
        total_loss += batch_loss.item() * len(batch)

    return total_loss / max(n_races, 1)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set.

    Metrics:
      - winner_accuracy: % of races where model's #1 pick = actual winner
      - top3_accuracy:   avg |pred_top3 intersection actual_top3| / 3
      - top4_accuracy:   avg |pred_top4 intersection actual_top4| / 4
    """
    model.eval()
    total_loss = 0.0

    winner_correct = 0
    top3_correct = 0.0
    top4_correct = 0.0
    total_races = 0
    races_with_3plus = 0
    races_with_4plus = 0

    with torch.no_grad():
        for batch in dataloader:
            for race in batch:
                features = race["features"].to(device)
                rankings = race["rankings"].to(device)

                scores = model(features)
                loss = plackett_luce_loss(scores, rankings)
                total_loss += loss.item()

                predicted_order = torch.argsort(scores, descending=True)
                actual_order = torch.argsort(rankings)
                n_horses = len(predicted_order)

                # Winner accuracy
                if predicted_order[0] == actual_order[0]:
                    winner_correct += 1

                # Top-3 overlap
                if n_horses >= 3:
                    pred_top3 = set(predicted_order[:3].cpu().numpy())
                    actual_top3 = set(actual_order[:3].cpu().numpy())
                    top3_correct += len(pred_top3 & actual_top3) / 3.0
                    races_with_3plus += 1

                # Top-4 overlap
                if n_horses >= 4:
                    pred_top4 = set(predicted_order[:4].cpu().numpy())
                    actual_top4 = set(actual_order[:4].cpu().numpy())
                    top4_correct += len(pred_top4 & actual_top4) / 4.0
                    races_with_4plus += 1

                total_races += 1

    return {
        "loss": total_loss / max(total_races, 1),
        "winner_accuracy": winner_correct / max(total_races, 1),
        "top3_accuracy": top3_correct / max(races_with_3plus, 1),
        "top4_accuracy": top4_correct / max(races_with_4plus, 1),
    }


# ═══════════════════════════════════════════════════════════
# MAIN RETRAINING FUNCTION
# ═══════════════════════════════════════════════════════════


def retrain_model(
    db_path: str = "gold_high_iq.db",
    epochs: int = 100,
    learning_rate: float = 0.0005,
    batch_size: int = 4,
    hidden_dim: int = 64,
    train_split: float = 0.8,
    min_races: int = 50,
    early_stopping_patience: int = 15,
    weight_decay: float = 1e-3,
    max_grad_norm: float = 1.0,
) -> dict:
    """
    Main retraining function — optimised for small horse racing datasets.

    Args:
        db_path:     Path to gold_high_iq.db
        epochs:      Maximum training epochs (early stopping may exit sooner)
        learning_rate: Initial LR for AdamW (scheduler will reduce)
        batch_size:  Number of RACES per batch (not horses)
        hidden_dim:  MLP hidden layer width (64 recommended for <500 races)
        train_split: Fraction of races used for training (rest = validation)
        min_races:   Minimum completed races required to begin training
        early_stopping_patience: Stop after N epochs with no val loss improvement
        weight_decay: L2 regularisation strength for AdamW
        max_grad_norm: Gradient clipping threshold

    Returns:
        Dict with training results, metrics, and model path
    """
    logger.info("=" * 60)
    logger.info("🚀 STARTING ML MODEL RETRAINING")
    logger.info("=" * 60)

    set_deterministic_seed(SEED)
    logger.info(f"🎲 Deterministic seed: {SEED}")

    start_time = time.time()

    # ── 1. Load data from database ───────────────────────────
    db = GoldHighIQDatabase(db_path)
    training_data = db.get_training_data(min_races=min_races)

    if training_data is None:
        logger.error(f"❌ Need at least {min_races} completed races.")
        return {"error": f"Insufficient data — need {min_races}+ completed races"}

    features_df, labels_df = training_data
    n_total_races = len(features_df["race_id"].unique())
    logger.info(f"✅ Loaded {len(features_df)} horses from {n_total_races} races")

    # ── 2. Chronological train/val split ─────────────────────
    # Split by race_id order (preserves temporal structure) so the model
    # is validated on the most recent races — simulating real usage.
    race_ids = features_df["race_id"].unique()
    n_train = int(len(race_ids) * train_split)
    train_race_ids = set(race_ids[:n_train])
    val_race_ids = set(race_ids[n_train:])

    train_features = features_df[features_df["race_id"].isin(train_race_ids)]
    train_labels = labels_df[labels_df["race_id"].isin(train_race_ids)]
    val_features = features_df[features_df["race_id"].isin(val_race_ids)]
    val_labels = labels_df[labels_df["race_id"].isin(val_race_ids)]

    # ── 3. Create datasets (standardised) ────────────────────
    # Train set computes its own mean/std; val set reuses them (no leakage)
    train_dataset = RaceDataset(train_features, train_labels)
    val_dataset = RaceDataset(
        val_features,
        val_labels,
        feat_mean=train_dataset.feat_mean,
        feat_std=train_dataset.feat_std,
    )

    n_train_races = len(train_dataset)
    n_val_races = len(val_dataset)

    if n_val_races == 0:
        logger.error("❌ No validation races after split")
        return {"error": "Insufficient data for validation split"}

    # Seeded DataLoaders for reproducibility
    shuffle_gen = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=collate_races,
        generator=shuffle_gen,
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_races
    )

    logger.info(f"📊 Train: {n_train_races} races | Val: {n_val_races} races")

    # ── 4. Initialise model + optimiser + scheduler ──────────
    n_features = len(train_dataset.feature_cols)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device: {device} | Features: {n_features} | Hidden: {hidden_dim}")

    model = RankingNN(n_features=n_features, hidden_dim=hidden_dim).to(device)
    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"📐 Model parameters: {total_params:,}")

    optimizer = optim.AdamW(
        model.parameters(), lr=learning_rate, weight_decay=weight_decay
    )
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=7, min_lr=1e-6
    )

    # ── 5. Training loop with early stopping ─────────────────
    best_val_loss = float("inf")
    best_val_acc = 0.0
    best_model_state = None
    best_epoch = 0
    epochs_no_improve = 0

    logger.info(
        f"\n🏃 Training for up to {epochs} epochs "
        f"(early stop patience={early_stopping_patience})..."
    )

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device, max_grad_norm)
        val_metrics = evaluate(model, val_loader, device)
        val_loss = val_metrics["loss"]

        scheduler.step(val_loss)
        current_lr = optimizer.param_groups[0]["lr"]

        # Check for improvement (track val loss for early stopping)
        if val_loss < best_val_loss - 1e-6:
            best_val_loss = val_loss
            best_val_acc = val_metrics["winner_accuracy"]
            best_model_state = {k: v.clone() for k, v in model.state_dict().items()}
            best_epoch = epoch + 1
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

        # Log progress every 10 epochs or on new best
        if (epoch + 1) % 10 == 0 or epochs_no_improve == 0:
            logger.info(
                f"Epoch {epoch + 1:3d}/{epochs} | "
                f"Train: {train_loss:.4f} | Val: {val_loss:.4f} | "
                f"Win: {val_metrics['winner_accuracy']:.1%} | "
                f"Top3: {val_metrics['top3_accuracy']:.1%} | "
                f"Top4: {val_metrics['top4_accuracy']:.1%} | "
                f"LR: {current_lr:.6f}"
            )

        # Early stopping
        if epochs_no_improve >= early_stopping_patience:
            logger.info(
                f"⏹️  Early stopping at epoch {epoch + 1} "
                f"(no improvement for {early_stopping_patience} epochs). "
                f"Best epoch: {best_epoch}"
            )
            break

    actual_epochs = epoch + 1

    # ── 6. Restore best model and save ───────────────────────
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    os.makedirs("models", exist_ok=True)
    model_path = f"models/ranking_model_{int(time.time())}.pt"

    # Save everything needed for inference: model weights, architecture
    # config, feature standardisation stats, and validation metrics
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "hidden_dim": hidden_dim,
            "feature_cols": train_dataset.feature_cols,
            "feat_mean": train_dataset.feat_mean.tolist(),
            "feat_std": train_dataset.feat_std.tolist(),
            "val_metrics": val_metrics,
            "best_epoch": best_epoch,
            "total_epochs_run": actual_epochs,
            "training_config": {
                "learning_rate": learning_rate,
                "batch_size": batch_size,
                "weight_decay": weight_decay,
                "max_grad_norm": max_grad_norm,
                "early_stopping_patience": early_stopping_patience,
                "train_races": n_train_races,
                "val_races": n_val_races,
            },
        },
        model_path,
    )

    # ── 7. Final evaluation on best model ────────────────────
    final_metrics = evaluate(model, val_loader, device)
    duration = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Best Epoch:       {best_epoch} / {actual_epochs} run")
    logger.info(f"Winner Accuracy:  {final_metrics['winner_accuracy']:.1%}")
    logger.info(f"Top-3 Accuracy:   {final_metrics['top3_accuracy']:.1%}")
    logger.info(f"Top-4 Accuracy:   {final_metrics['top4_accuracy']:.1%}")
    logger.info(f"Validation Loss:  {final_metrics['loss']:.4f}")
    logger.info(f"Training Time:    {duration:.1f}s")
    logger.info(f"Model Saved:      {model_path}")
    logger.info("=" * 60)

    # ── 8. Log to database ───────────────────────────────────
    db.log_retraining(
        metrics={
            "total_races": n_train_races + n_val_races,
            "total_horses": len(features_df),
            "train_split_pct": train_split,
            "val_split_pct": 1 - train_split,
            "val_winner_accuracy": final_metrics["winner_accuracy"],
            "val_top3_accuracy": final_metrics["top3_accuracy"],
            "val_top5_accuracy": final_metrics["top4_accuracy"],
            "val_loss": final_metrics["loss"],
            "epochs": actual_epochs,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "training_duration": duration,
        },
        model_path=model_path,
    )

    return {
        "success": True,
        "model_path": model_path,
        "metrics": final_metrics,
        "duration": duration,
        "n_races": n_train_races + n_val_races,
        "n_horses": len(features_df),
        "best_epoch": best_epoch,
        "total_epochs_run": actual_epochs,
        "early_stopped": actual_epochs < epochs,
    }


if __name__ == "__main__":
    """Run retraining from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrain racing ML model")
    parser.add_argument("--epochs", type=int, default=100, help="Max training epochs")
    parser.add_argument("--lr", type=float, default=0.0005, help="Learning rate")
    parser.add_argument("--batch-size", type=int, default=4, help="Races per batch")
    parser.add_argument(
        "--min-races", type=int, default=50, help="Minimum races required"
    )

    args = parser.parse_args()

    results = retrain_model(
        epochs=args.epochs,
        learning_rate=args.lr,
        batch_size=args.batch_size,
        min_races=args.min_races,
    )

    if "error" in results:
        logger.error(f"❌ Retraining failed: {results['error']}")
    else:
        logger.info(f"✅ Retraining successful! Model: {results['model_path']}")
