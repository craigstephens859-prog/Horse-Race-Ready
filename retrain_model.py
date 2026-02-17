"""
ML MODEL RETRAINING ENGINE
==========================

Retrain the racing prediction model using real race results
from the gold_high_iq database.

Uses PyTorch with Plackett-Luce ranking loss for optimal accuracy.

Author: Top-Tier ML Engineer
Date: January 29, 2026
"""

import logging
import time

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from gold_database_manager import GoldHighIQDatabase

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class RaceDataset(Dataset):
    """PyTorch Dataset for race data with listwise ranking."""

    def __init__(self, features_df: pd.DataFrame, labels_df: pd.DataFrame):
        """
        Args:
            features_df: DataFrame with features and race_id
            labels_df: DataFrame with race_id, horse_name, actual_finish_position
        """
        self.races = []

        # Group by race
        race_groups = features_df.groupby("race_id")

        for race_id, race_data in race_groups:
            # Get actual finish positions for this race
            race_labels = labels_df[labels_df["race_id"] == race_id]

            # Merge features with labels
            race_combined = race_data.merge(
                race_labels[["horse_name", "actual_finish_position"]],
                on="horse_name",
                how="inner",
            )

            if len(race_combined) < 2:
                continue  # Skip races with less than 2 horses

            # Extract feature columns (exclude metadata)
            feature_cols = [
                c
                for c in race_combined.columns
                if c
                not in [
                    "race_id",
                    "horse_name",
                    "actual_finish",
                    "actual_finish_position",
                ]
            ]

            features = race_combined[feature_cols].values.astype(np.float32)
            rankings = race_combined["actual_finish_position"].values.astype(np.int64)

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
    """Custom collate function for variable-length races."""
    return batch  # Return list of dicts


class RankingNN(nn.Module):
    """Neural network for listwise ranking."""

    def __init__(self, n_features: int, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden_dim)
        self.dropout1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.dropout2 = nn.Dropout(0.3)
        self.fc3 = nn.Linear(hidden_dim, 1)  # Single score output

    def forward(self, x):
        """
        Args:
            x: (n_horses, n_features)

        Returns:
            scores: (n_horses,)
        """
        h1 = torch.relu(self.fc1(x))
        h1 = self.dropout1(h1)
        h2 = torch.relu(self.fc2(h1))
        h2 = self.dropout2(h2)
        scores = self.fc3(h2).squeeze(-1)
        return scores


def plackett_luce_loss(
    scores: torch.Tensor, true_rankings: torch.Tensor
) -> torch.Tensor:
    """
    Plackett-Luce listwise ranking loss.

    Args:
        scores: (n_horses,) - model output scores
        true_rankings: (n_horses,) - actual finish positions (1, 2, 3, ...)

    Returns:
        loss: scalar
    """
    # Sort by true ranking
    sorted_indices = torch.argsort(true_rankings)
    sorted_scores = scores[sorted_indices]

    # Plackett-Luce: L = -Σ [score_i - log_sum_exp(remaining)]
    loss = 0.0
    n = len(sorted_scores)

    for i in range(n):
        remaining_scores = sorted_scores[i:]
        log_sum_exp = torch.logsumexp(remaining_scores, dim=0)
        loss += sorted_scores[i] - log_sum_exp

    return -loss  # Negate for minimization


def train_epoch(model, dataloader, optimizer, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0.0

    for batch in dataloader:
        for race in batch:
            features = race["features"].to(device)
            rankings = race["rankings"].to(device)

            optimizer.zero_grad()
            scores = model(features)
            loss = plackett_luce_loss(scores, rankings)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()

    return total_loss / len(dataloader.dataset)


def evaluate(model, dataloader, device):
    """Evaluate model on validation set."""
    model.eval()
    total_loss = 0.0

    winner_correct = 0
    top3_correct = 0
    top5_correct = 0
    total_races = 0

    with torch.no_grad():
        for batch in dataloader:
            for race in batch:
                features = race["features"].to(device)
                rankings = race["rankings"].to(device)

                scores = model(features)
                loss = plackett_luce_loss(scores, rankings)
                total_loss += loss.item()

                # Calculate accuracy metrics
                predicted_order = torch.argsort(scores, descending=True)
                actual_order = torch.argsort(rankings)

                # Winner accuracy
                if predicted_order[0] == actual_order[0]:
                    winner_correct += 1

                # Top-3 accuracy
                pred_top3 = set(predicted_order[:3].cpu().numpy())
                actual_top3 = set(actual_order[:3].cpu().numpy())
                top3_correct += len(pred_top3 & actual_top3) / 3.0

                # Top-5 accuracy
                if len(predicted_order) >= 5:
                    pred_top5 = set(predicted_order[:5].cpu().numpy())
                    actual_top5 = set(actual_order[:5].cpu().numpy())
                    top5_correct += len(pred_top5 & actual_top5) / 5.0

                total_races += 1

    avg_loss = total_loss / len(dataloader.dataset)
    winner_acc = winner_correct / total_races if total_races > 0 else 0.0
    top3_acc = top3_correct / total_races if total_races > 0 else 0.0
    top5_acc = top5_correct / total_races if total_races > 0 else 0.0

    return {
        "loss": avg_loss,
        "winner_accuracy": winner_acc,
        "top3_accuracy": top3_acc,
        "top5_accuracy": top5_acc,
    }


def retrain_model(
    db_path: str = "gold_high_iq.db",
    epochs: int = 50,
    learning_rate: float = 0.001,
    batch_size: int = 8,
    hidden_dim: int = 128,
    train_split: float = 0.8,
    min_races: int = 50,
) -> dict:
    """
    Main retraining function.

    Args:
        db_path: Path to gold database
        epochs: Number of training epochs
        learning_rate: Learning rate for Adam optimizer
        batch_size: Batch size (number of races per batch)
        hidden_dim: Hidden layer dimension
        train_split: Train/val split ratio
        min_races: Minimum number of completed races required

    Returns:
        Dict with training results and metrics
    """
    logger.info("=" * 60)
    logger.info("🚀 STARTING ML MODEL RETRAINING")
    logger.info("=" * 60)

    start_time = time.time()

    # 1. Load data from database
    db = GoldHighIQDatabase(db_path)
    training_data = db.get_training_data(min_races=min_races)

    if training_data is None:
        logger.error(
            f"❌ Insufficient data. Need at least {min_races} completed races."
        )
        return {"error": "Insufficient data"}

    features_df, labels_df = training_data

    logger.info(
        f"✅ Loaded {len(features_df)} horses from {len(features_df['race_id'].unique())} races"
    )

    # 2. Create datasets
    dataset = RaceDataset(features_df, labels_df)

    # Train/val split
    n_train = int(len(dataset) * train_split)
    n_val = len(dataset) - n_train

    train_dataset, val_dataset = torch.utils.data.random_split(
        dataset, [n_train, n_val], generator=torch.Generator().manual_seed(42)
    )

    train_loader = DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True, collate_fn=collate_races
    )
    val_loader = DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False, collate_fn=collate_races
    )

    logger.info(f"📊 Train: {n_train} races | Val: {n_val} races")

    # 3. Initialize model
    n_features = len(
        [
            c
            for c in features_df.columns
            if c not in ["race_id", "horse_name", "actual_finish"]
        ]
    )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"🖥️  Device: {device}")

    model = RankingNN(n_features=n_features, hidden_dim=hidden_dim).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=5, verbose=True
    )

    # 4. Training loop
    best_val_acc = 0.0
    best_model_state = None

    logger.info(f"\n🏃 Training for {epochs} epochs...")

    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader, optimizer, device)
        val_metrics = evaluate(model, val_loader, device)

        scheduler.step(val_metrics["loss"])

        # Log progress
        if (epoch + 1) % 5 == 0:
            logger.info(
                f"Epoch {epoch + 1}/{epochs} | "
                f"Train Loss: {train_loss:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} | "
                f"Winner Acc: {val_metrics['winner_accuracy']:.1%} | "
                f"Top-3: {val_metrics['top3_accuracy']:.1%} | "
                f"Top-5: {val_metrics['top5_accuracy']:.1%}"
            )

        # Save best model
        if val_metrics["winner_accuracy"] > best_val_acc:
            best_val_acc = val_metrics["winner_accuracy"]
            best_model_state = model.state_dict().copy()

    # 5. Load best model and save
    if best_model_state is not None:
        model.load_state_dict(best_model_state)

    model_path = f"models/ranking_model_{int(time.time())}.pt"
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "n_features": n_features,
            "hidden_dim": hidden_dim,
            "val_metrics": val_metrics,
        },
        model_path,
    )

    # 6. Final evaluation
    final_metrics = evaluate(model, val_loader, device)

    duration = time.time() - start_time

    logger.info("\n" + "=" * 60)
    logger.info("✅ TRAINING COMPLETE")
    logger.info("=" * 60)
    logger.info(f"Winner Accuracy:  {final_metrics['winner_accuracy']:.1%}")
    logger.info(f"Top-3 Accuracy:   {final_metrics['top3_accuracy']:.1%}")
    logger.info(f"Top-5 Accuracy:   {final_metrics['top5_accuracy']:.1%}")
    logger.info(f"Validation Loss:  {final_metrics['loss']:.4f}")
    logger.info(f"Training Time:    {duration:.1f}s")
    logger.info(f"Model Saved:      {model_path}")
    logger.info("=" * 60)

    # 7. Log to database
    db.log_retraining(
        metrics={
            "total_races": len(dataset),
            "total_horses": len(features_df),
            "train_split_pct": train_split,
            "val_split_pct": 1 - train_split,
            "val_winner_accuracy": final_metrics["winner_accuracy"],
            "val_top3_accuracy": final_metrics["top3_accuracy"],
            "val_top5_accuracy": final_metrics["top5_accuracy"],
            "val_loss": final_metrics["loss"],
            "epochs": epochs,
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
        "n_races": len(dataset),
        "n_horses": len(features_df),
    }


if __name__ == "__main__":
    """Run retraining from command line."""
    import argparse

    parser = argparse.ArgumentParser(description="Retrain racing ML model")
    parser.add_argument(
        "--epochs", type=int, default=50, help="Number of training epochs"
    )
    parser.add_argument("--lr", type=float, default=0.001, help="Learning rate")
    parser.add_argument(
        "--min-races", type=int, default=50, help="Minimum races required"
    )

    args = parser.parse_args()

    results = retrain_model(
        epochs=args.epochs, learning_rate=args.lr, min_races=args.min_races
    )

    if "error" in results:
        logger.error(f"❌ Retraining failed: {results['error']}")
    else:
        logger.info(f"✅ Retraining successful! Model: {results['model_path']}")
