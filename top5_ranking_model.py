"""
PYTORCH TOP-5 RANKING MODEL
============================

Listwise ranking model optimized for predicting exact top-5 finishing order.
Target: 90%+ winner accuracy, 2+ correct for exacta, 2-3 correct for trifecta/superfecta.

Uses:
- Listwise ListNet loss (Plackett-Luce model)
- Pairwise RankNet loss (Bradley-Terry model)
- Attention mechanism for field-relative features
- Gradient boosting ensemble for final predictions
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
from typing import List, Tuple, Dict, Optional
from pathlib import Path
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# ============================================================================
# 1. DATASET LOADER
# ============================================================================

class RacingDataset(Dataset):
    """
    PyTorch Dataset for racing data.
    Each sample is a complete race (variable number of runners, 5-20 typical).
    """
    
    def __init__(self, parquet_path: str, races_parquet: str, 
                 include_results: bool = True):
        """
        Args:
            parquet_path: Path to features.parquet (runner-level features)
            races_parquet: Path to races.parquet (race metadata)
            include_results: If True, load finish positions (for training)
        """
        self.features_df = pd.read_parquet(parquet_path)
        self.races_df = pd.read_parquet(races_parquet)
        
        # Connect to database for results if needed
        self.include_results = include_results
        if include_results:
            import sqlite3
            db_path = str(Path(parquet_path).parent.parent / "historical_racing_gold.db")
            conn = sqlite3.connect(db_path)
            self.results_df = pd.read_sql("SELECT * FROM results", conn)
            conn.close()
        
        # Get unique races
        self.race_ids = self.features_df['runner_id'].apply(
            lambda x: '_'.join(x.split('_')[:-1])
        ).unique()
        
        # Feature columns (exclude IDs)
        self.feature_cols = [
            c for c in self.features_df.columns 
            if c not in ['runner_id', 'program_number', 'race_id']
        ]
        
        logger.info(f"Loaded {len(self.race_ids)} races with {len(self.feature_cols)} features per runner")
    
    def __len__(self):
        return len(self.race_ids)
    
    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        """
        Return one complete race.
        
        Returns:
            {
                'features': [field_size, n_features] tensor
                'finish_positions': [field_size] tensor (if include_results=True)
                'race_id': str
            }
        """
        race_id = self.race_ids[idx]
        
        # Get all runners for this race
        race_features = self.features_df[
            self.features_df['runner_id'].str.startswith(race_id)
        ].copy()
        
        # Sort by program number
        race_features = race_features.sort_values('program_number')
        
        # Extract features
        X = race_features[self.feature_cols].values.astype(np.float32)
        X = torch.from_numpy(X)
        
        result = {
            'features': X,
            'race_id': race_id,
            'program_numbers': torch.tensor(race_features['program_number'].values, dtype=torch.long)
        }
        
        # Get finish positions if available
        if self.include_results:
            runner_ids = race_features['runner_id'].values
            results = self.results_df[self.results_df['runner_id'].isin(runner_ids)]
            
            # Map runner_id to finish position
            finish_map = dict(zip(results['runner_id'], results['finish_position']))
            finish_positions = [finish_map.get(rid, 99) for rid in runner_ids]
            
            result['finish_positions'] = torch.tensor(finish_positions, dtype=torch.long)
        
        return result


def collate_races(batch: List[Dict]) -> Dict:
    """
    Custom collate function for variable-sized races.
    Pads races to max field size in batch.
    """
    max_field_size = max(item['features'].size(0) for item in batch)
    n_features = batch[0]['features'].size(1)
    batch_size = len(batch)
    
    # Padded tensors
    features_padded = torch.zeros(batch_size, max_field_size, n_features)
    finish_padded = torch.zeros(batch_size, max_field_size, dtype=torch.long) + 99  # 99 = DNF
    mask = torch.zeros(batch_size, max_field_size, dtype=torch.bool)
    
    race_ids = []
    
    for i, item in enumerate(batch):
        field_size = item['features'].size(0)
        features_padded[i, :field_size, :] = item['features']
        mask[i, :field_size] = True
        race_ids.append(item['race_id'])
        
        if 'finish_positions' in item:
            finish_padded[i, :field_size] = item['finish_positions']
    
    return {
        'features': features_padded,
        'finish_positions': finish_padded,
        'mask': mask,
        'race_ids': race_ids
    }


# ============================================================================
# 2. TOP-5 RANKING MODEL ARCHITECTURE
# ============================================================================

class Top5RankingModel(nn.Module):
    """
    Deep neural network for listwise ranking.
    
    Architecture:
    1. Runner-level encoder (MLP per horse)
    2. Field-level attention (cross-attention across all runners)
    3. Ranking head (output scores for PlackettLuce loss)
    """
    
    def __init__(self, n_features: int, hidden_dim: int = 256, 
                 dropout: float = 0.3):
        super().__init__()
        
        # Runner encoder (transforms raw features to embeddings)
        self.runner_encoder = nn.Sequential(
            nn.Linear(n_features, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # Field-level attention (compare horses to each other)
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_dim,
            num_heads=8,
            dropout=dropout,
            batch_first=True
        )
        
        # Ranking head (output single score per runner)
        self.ranking_head = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim // 2, 1)
        )
    
    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            features: [batch_size, max_field_size, n_features]
            mask: [batch_size, max_field_size] - True for real runners, False for padding
        
        Returns:
            scores: [batch_size, max_field_size] - ranking scores (higher = better)
        """
        batch_size, max_field, n_feat = features.shape
        
        # Encode each runner
        runner_embeddings = self.runner_encoder(features)  # [B, F, H]
        
        # Apply field-level attention
        # Attention mask: prevent attending to padded positions
        attn_mask = ~mask.unsqueeze(1).expand(-1, max_field, -1)  # [B, F, F]
        attn_mask = attn_mask.float() * -1e9  # Convert to additive mask
        
        attended, _ = self.attention(
            runner_embeddings, 
            runner_embeddings, 
            runner_embeddings,
            attn_mask=attn_mask
        )  # [B, F, H]
        
        # Predict ranking score for each runner
        scores = self.ranking_head(attended).squeeze(-1)  # [B, F]
        
        # Mask out padded positions
        scores = scores.masked_fill(~mask, float('-inf'))
        
        return scores


# ============================================================================
# 3. RANKING LOSS FUNCTIONS
# ============================================================================

def plackett_luce_loss(scores: torch.Tensor, finish_positions: torch.Tensor, 
                       mask: torch.Tensor, top_k: int = 5) -> torch.Tensor:
    """
    Plackett-Luce (ListNet) loss for listwise ranking.
    Optimizes for exact top-K ordering.
    
    Args:
        scores: [batch, field_size] - predicted scores
        finish_positions: [batch, field_size] - actual positions (1=winner, 2=2nd, etc.)
        mask: [batch, field_size] - True for valid runners
        top_k: Only optimize top K positions (default 5)
    
    Returns:
        loss: scalar
    """
    batch_size, field_size = scores.shape
    
    # Convert finish positions to rankings (1st place = 0, 2nd = 1, etc.)
    rankings = finish_positions - 1
    
    total_loss = 0.0
    valid_samples = 0
    
    for b in range(batch_size):
        # Get valid runners
        valid_mask = mask[b]
        if valid_mask.sum() < 2:
            continue
        
        batch_scores = scores[b, valid_mask]
        batch_rankings = rankings[b, valid_mask]
        
        # Filter out DNF (position 99)
        finished_mask = batch_rankings < 98
        if finished_mask.sum() < 2:
            continue
        
        batch_scores = batch_scores[finished_mask]
        batch_rankings = batch_rankings[finished_mask]
        
        # Sort by actual finish position
        sorted_indices = torch.argsort(batch_rankings)
        sorted_scores = batch_scores[sorted_indices]
        
        # Plackett-Luce log-likelihood for top K
        loss_k = 0.0
        for k in range(min(top_k, len(sorted_scores))):
            # Probability of picking position k from remaining candidates
            remaining_scores = sorted_scores[k:]
            log_prob = sorted_scores[k] - torch.logsumexp(remaining_scores, dim=0)
            loss_k -= log_prob
        
        total_loss += loss_k
        valid_samples += 1
    
    return total_loss / max(valid_samples, 1)


def pairwise_ranking_loss(scores: torch.Tensor, finish_positions: torch.Tensor, 
                         mask: torch.Tensor) -> torch.Tensor:
    """
    Pairwise ranking loss (RankNet).
    Penalizes incorrect pairwise orderings.
    
    For every pair (i, j) where horse i finished ahead of horse j,
    ensure score_i > score_j.
    """
    batch_size = scores.shape[0]
    
    total_loss = 0.0
    valid_pairs = 0
    
    for b in range(batch_size):
        valid_mask = mask[b]
        if valid_mask.sum() < 2:
            continue
        
        batch_scores = scores[b, valid_mask]
        batch_positions = finish_positions[b, valid_mask]
        
        # Filter DNF
        finished_mask = batch_positions < 98
        if finished_mask.sum() < 2:
            continue
        
        batch_scores = batch_scores[finished_mask]
        batch_positions = batch_positions[finished_mask]
        
        # Generate all pairs
        n = len(batch_scores)
        for i in range(n):
            for j in range(i+1, n):
                # If horse i finished ahead of horse j
                if batch_positions[i] < batch_positions[j]:
                    # We want score_i > score_j
                    diff = batch_scores[j] - batch_scores[i]
                    loss_ij = F.softplus(diff)  # max(0, diff) smooth version
                    total_loss += loss_ij
                    valid_pairs += 1
    
    return total_loss / max(valid_pairs, 1)


def combined_ranking_loss(scores: torch.Tensor, finish_positions: torch.Tensor, 
                         mask: torch.Tensor) -> torch.Tensor:
    """
    Combination of listwise + pairwise losses.
    """
    listwise = plackett_luce_loss(scores, finish_positions, mask, top_k=5)
    pairwise = pairwise_ranking_loss(scores, finish_positions, mask)
    
    # Weight listwise more heavily (exact top-5 order is primary goal)
    return 0.7 * listwise + 0.3 * pairwise


# ============================================================================
# 4. TRAINING LOOP
# ============================================================================

class Top5Trainer:
    """
    Training orchestrator for top-5 ranking model.
    """
    
    def __init__(self, model: Top5RankingModel, device: str = 'cuda'):
        self.model = model.to(device)
        self.device = device
        self.optimizer = torch.optim.AdamW(
            model.parameters(), 
            lr=1e-3, 
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, 
            mode='min', 
            factor=0.5, 
            patience=3
        )
    
    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train for one epoch"""
        self.model.train()
        total_loss = 0.0
        
        for batch in dataloader:
            features = batch['features'].to(self.device)
            finish_positions = batch['finish_positions'].to(self.device)
            mask = batch['mask'].to(self.device)
            
            # Forward
            scores = self.model(features, mask)
            
            # Loss
            loss = combined_ranking_loss(scores, finish_positions, mask)
            
            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()
            
            total_loss += loss.item()
        
        return total_loss / len(dataloader)
    
    def evaluate(self, dataloader: DataLoader) -> Dict[str, float]:
        """Evaluate on validation set"""
        self.model.eval()
        
        total_winner_correct = 0
        total_top2_correct = 0
        total_top3_correct = 0
        total_races = 0
        
        with torch.no_grad():
            for batch in dataloader:
                features = batch['features'].to(self.device)
                finish_positions = batch['finish_positions'].to(self.device)
                mask = batch['mask'].to(self.device)
                
                scores = self.model(features, mask)
                
                # Get predictions (top 5 by score)
                for b in range(scores.shape[0]):
                    valid_mask = mask[b]
                    if valid_mask.sum() < 2:
                        continue
                    
                    race_scores = scores[b, valid_mask]
                    race_positions = finish_positions[b, valid_mask]
                    
                    # Filter DNF
                    finished_mask = race_positions < 98
                    if finished_mask.sum() < 2:
                        continue
                    
                    race_scores = race_scores[finished_mask]
                    race_positions = race_positions[finished_mask]
                    
                    # Predicted top 5
                    _, pred_indices = torch.topk(race_scores, k=min(5, len(race_scores)))
                    
                    # Actual top 5
                    actual_top5_mask = race_positions <= 5
                    actual_top5_indices = torch.where(actual_top5_mask)[0]
                    
                    # Winner accuracy
                    actual_winner_idx = torch.argmin(race_positions)
                    if pred_indices[0] == actual_winner_idx:
                        total_winner_correct += 1
                    
                    # Top 2 accuracy (how many of top 2 are correct?)
                    pred_top2 = set(pred_indices[:2].cpu().numpy())
                    actual_top2 = set(torch.where(race_positions <= 2)[0].cpu().numpy())
                    total_top2_correct += len(pred_top2 & actual_top2)
                    
                    # Top 3 accuracy
                    pred_top3 = set(pred_indices[:3].cpu().numpy())
                    actual_top3 = set(torch.where(race_positions <= 3)[0].cpu().numpy())
                    total_top3_correct += len(pred_top3 & actual_top3)
                    
                    total_races += 1
        
        return {
            'winner_accuracy': total_winner_correct / total_races if total_races > 0 else 0,
            'avg_top2_correct': total_top2_correct / (total_races * 2) if total_races > 0 else 0,
            'avg_top3_correct': total_top3_correct / (total_races * 3) if total_races > 0 else 0,
            'total_races': total_races
        }
    
    def fit(self, train_loader: DataLoader, val_loader: DataLoader, 
            n_epochs: int = 50, checkpoint_path: str = "top5_model.pt"):
        """
        Full training loop with validation and checkpointing.
        """
        best_val_acc = 0.0
        
        for epoch in range(n_epochs):
            train_loss = self.train_epoch(train_loader)
            val_metrics = self.evaluate(val_loader)
            
            logger.info(f"Epoch {epoch+1}/{n_epochs}")
            logger.info(f"  Train Loss: {train_loss:.4f}")
            logger.info(f"  Winner Acc: {val_metrics['winner_accuracy']:.1%}")
            logger.info(f"  Avg Top-2 Correct: {val_metrics['avg_top2_correct']:.2f}/2")
            logger.info(f"  Avg Top-3 Correct: {val_metrics['avg_top3_correct']:.2f}/3")
            
            # Learning rate scheduling
            self.scheduler.step(train_loss)
            
            # Save best model
            if val_metrics['winner_accuracy'] > best_val_acc:
                best_val_acc = val_metrics['winner_accuracy']
                torch.save({
                    'epoch': epoch,
                    'model_state_dict': self.model.state_dict(),
                    'optimizer_state_dict': self.optimizer.state_dict(),
                    'val_metrics': val_metrics
                }, checkpoint_path)
                logger.info(f"  ✓ Saved best model (winner acc: {best_val_acc:.1%})")
        
        logger.info(f"\n[TRAINING COMPLETE] Best winner accuracy: {best_val_acc:.1%}")


# ============================================================================
# 5. INFERENCE / PREDICTION
# ============================================================================

class Top5Predictor:
    """
    Production inference for new races.
    """
    
    def __init__(self, model_path: str, device: str = 'cuda'):
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        
        # Infer model architecture from checkpoint
        # (In production, you'd save architecture config with checkpoint)
        self.model = Top5RankingModel(
            n_features=50,  # Update based on your actual feature count
            hidden_dim=256,
            dropout=0.3
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
        self.device = device
        logger.info(f"Loaded model from {model_path}")
        logger.info(f"Validation metrics: {checkpoint['val_metrics']}")
    
    def predict_top5(self, features: np.ndarray) -> List[Tuple[int, float]]:
        """
        Predict top 5 finishers for a single race.
        
        Args:
            features: [field_size, n_features] numpy array
        
        Returns:
            List of (position_index, confidence) tuples for top 5
        """
        with torch.no_grad():
            # Convert to tensor
            features_t = torch.from_numpy(features).float().unsqueeze(0).to(self.device)
            mask = torch.ones(1, features.shape[0], dtype=torch.bool).to(self.device)
            
            # Predict
            scores = self.model(features_t, mask).squeeze(0)
            
            # Convert scores to probabilities (softmax)
            probs = F.softmax(scores, dim=0)
            
            # Get top 5
            top5_probs, top5_indices = torch.topk(probs, k=min(5, len(probs)))
            
            results = [
                (idx.item(), prob.item()) 
                for idx, prob in zip(top5_indices, top5_probs)
            ]
            
            return results
    
    def predict_with_program_numbers(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """
        Predict top 5 with program numbers (user-friendly output).
        
        Args:
            features_df: DataFrame with features + 'program_number' column
        
        Returns:
            DataFrame with columns: [rank, program_number, confidence]
        """
        # Extract features (exclude program_number)
        feature_cols = [c for c in features_df.columns if c != 'program_number']
        features = features_df[feature_cols].values.astype(np.float32)
        
        # Predict
        top5 = self.predict_top5(features)
        
        # Map indices to program numbers
        results = []
        for rank, (idx, confidence) in enumerate(top5, 1):
            program_num = features_df.iloc[idx]['program_number']
            results.append({
                'rank': rank,
                'program_number': program_num,
                'confidence': confidence
            })
        
        return pd.DataFrame(results)


# ============================================================================
# USAGE EXAMPLE
# ============================================================================

if __name__ == "__main__":
    # Example training workflow
    
    # 1. Load data
    train_dataset = RacingDataset(
        parquet_path="training_data/features.parquet",
        races_parquet="training_data/races.parquet",
        include_results=True
    )
    
    # 2. Split train/val
    train_size = int(0.8 * len(train_dataset))
    val_size = len(train_dataset) - train_size
    train_data, val_data = torch.utils.data.random_split(
        train_dataset, [train_size, val_size]
    )
    
    # 3. Create dataloaders
    train_loader = DataLoader(
        train_data, 
        batch_size=32, 
        shuffle=True, 
        collate_fn=collate_races
    )
    val_loader = DataLoader(
        val_data, 
        batch_size=32, 
        shuffle=False, 
        collate_fn=collate_races
    )
    
    # 4. Initialize model
    n_features = len(train_dataset.feature_cols)
    model = Top5RankingModel(n_features=n_features, hidden_dim=256, dropout=0.3)
    
    # 5. Train
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    trainer = Top5Trainer(model, device=device)
    trainer.fit(train_loader, val_loader, n_epochs=50, checkpoint_path="top5_model.pt")
    
    # 6. Inference on new race
    # predictor = Top5Predictor("top5_model.pt", device=device)
    # predictions = predictor.predict_with_program_numbers(new_race_features_df)
    # print(predictions)
    
    logger.info("\n[COMPLETE] Top-5 ranking model trained")
    logger.info("Use Top5Predictor for production inference")
