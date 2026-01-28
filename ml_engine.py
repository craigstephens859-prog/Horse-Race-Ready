"""
ML-Based Probability Calibration Engine
Neural network for refining raw rating-based probabilities using historical data
"""

import numpy as np
import pandas as pd
import sqlite3
import os
from datetime import datetime
from typing import Dict, List, Tuple, Optional
import json

# Try PyTorch first, fallback to sklearn
try:
    import torch
    import torch.nn as nn
    import torch.optim as optim
    USING_TORCH = True
except ImportError:
    from sklearn.neural_network import MLPRegressor
    from sklearn.preprocessing import StandardScaler
    import pickle
    USING_TORCH = False

# ===================== Database Layer =====================

class RaceDatabase:
    """SQLite database for storing race history and results."""
    
    def __init__(self, db_path: str = "race_history.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Create tables if they don't exist."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Races table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS races (
                race_id TEXT PRIMARY KEY,
                date TEXT NOT NULL,
                track TEXT,
                surface TEXT,
                distance TEXT,
                condition TEXT,
                race_type TEXT,
                purse INTEGER,
                field_size INTEGER,
                ppi REAL,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Horses table (predictions + results)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS horses (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                race_id TEXT NOT NULL,
                horse_name TEXT NOT NULL,
                post_position INTEGER,
                morning_line TEXT,
                final_odds REAL,
                
                -- Rating components
                rating_total REAL,
                rating_class REAL,
                rating_speed REAL,
                rating_pace REAL,
                rating_style REAL,
                rating_post REAL,
                rating_angles REAL,
                rating_pedigree REAL,
                
                -- Predicted probabilities
                predicted_win_prob REAL,
                predicted_place_prob REAL,
                predicted_show_prob REAL,
                
                -- Actual results
                actual_finish INTEGER,
                actual_win INTEGER,
                actual_place INTEGER,
                actual_show INTEGER,
                
                -- Metadata
                running_style TEXT,
                quirin_points INTEGER,
                last_beyer REAL,
                avg_beyer REAL,
                
                FOREIGN KEY (race_id) REFERENCES races(race_id)
            )
        """)
        
        # Performance metrics table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS performance_metrics (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                date TEXT NOT NULL,
                metric_type TEXT,
                metric_value REAL,
                sample_size INTEGER,
                notes TEXT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        # Model versions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS model_versions (
                version_id INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                training_samples INTEGER,
                validation_accuracy REAL,
                brier_score REAL,
                model_params TEXT,
                notes TEXT
            )
        """)
        
        conn.commit()
        conn.close()
    
    def save_race(self, race_data: Dict) -> str:
        """Save race information and return race_id."""
        race_id = f"{race_data['track']}_{race_data['date']}_{race_data.get('race_number', 1)}"
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT OR REPLACE INTO races 
            (race_id, date, track, surface, distance, condition, race_type, purse, field_size, ppi)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            race_id,
            race_data['date'],
            race_data['track'],
            race_data['surface'],
            race_data['distance'],
            race_data['condition'],
            race_data['race_type'],
            race_data['purse'],
            race_data['field_size'],
            race_data.get('ppi', 0.0)
        ))
        
        conn.commit()
        conn.close()
        return race_id
    
    def save_horse_predictions(self, race_id: str, horses_data: List[Dict]):
        """Save horse predictions for a race."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for horse in horses_data:
            cursor.execute("""
                INSERT INTO horses (
                    race_id, horse_name, post_position, morning_line, final_odds,
                    rating_total, rating_class, rating_speed, rating_pace, 
                    rating_style, rating_post, rating_angles, rating_pedigree,
                    predicted_win_prob, predicted_place_prob, predicted_show_prob,
                    running_style, quirin_points, last_beyer, avg_beyer
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id,
                horse['horse_name'],
                horse.get('post_position'),
                horse.get('morning_line'),
                horse.get('final_odds'),
                horse.get('rating_total'),
                horse.get('rating_class'),
                horse.get('rating_speed'),
                horse.get('rating_pace'),
                horse.get('rating_style'),
                horse.get('rating_post'),
                horse.get('rating_angles', 0),
                horse.get('rating_pedigree', 0),
                horse.get('predicted_win_prob'),
                horse.get('predicted_place_prob'),
                horse.get('predicted_show_prob'),
                horse.get('running_style'),
                horse.get('quirin_points'),
                horse.get('last_beyer'),
                horse.get('avg_beyer')
            ))
        
        conn.commit()
        conn.close()
    
    def update_race_results(self, race_id: str, results: List[Tuple[str, int]]):
        """
        Update actual finishing positions.
        results: List of (horse_name, finish_position) tuples
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        for horse_name, finish in results:
            cursor.execute("""
                UPDATE horses 
                SET actual_finish = ?,
                    actual_win = ?,
                    actual_place = ?,
                    actual_show = ?
                WHERE race_id = ? AND horse_name = ?
            """, (
                finish,
                1 if finish == 1 else 0,
                1 if finish <= 2 else 0,
                1 if finish <= 3 else 0,
                race_id,
                horse_name
            ))
        
        conn.commit()
        conn.close()
    
    def get_training_data(self, min_samples: int = 50) -> Tuple[np.ndarray, np.ndarray]:
        """
        Fetch historical data for model training.
        Returns: (features, targets) where targets are actual win outcomes
        """
        conn = sqlite3.connect(self.db_path)
        
        query = """
            SELECT 
                h.rating_class, h.rating_speed, h.rating_pace, 
                h.rating_style, h.rating_post, h.rating_angles, h.rating_pedigree,
                h.predicted_win_prob, h.final_odds, h.quirin_points,
                h.last_beyer, h.avg_beyer,
                r.field_size, r.ppi, r.purse,
                h.actual_win
            FROM horses h
            JOIN races r ON h.race_id = r.race_id
            WHERE h.actual_finish IS NOT NULL
            ORDER BY r.date DESC
        """
        
        df = pd.read_sql_query(query, conn)
        conn.close()
        
        if len(df) < min_samples:
            return None, None
        
        # Features (15 dimensions)
        features = df[[
            'rating_class', 'rating_speed', 'rating_pace', 'rating_style', 'rating_post',
            'rating_angles', 'rating_pedigree', 'predicted_win_prob', 'final_odds',
            'quirin_points', 'last_beyer', 'avg_beyer', 'field_size', 'ppi', 'purse'
        ]].fillna(0).values
        
        # Targets (1 = won, 0 = lost)
        targets = df['actual_win'].values
        
        return features, targets
    
    def get_performance_stats(self) -> Dict:
        """Calculate overall performance statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Win prediction accuracy
        cursor.execute("""
            SELECT 
                COUNT(*) as total_races,
                SUM(CASE WHEN predicted_win_prob = (
                    SELECT MAX(predicted_win_prob) FROM horses h2 WHERE h2.race_id = horses.race_id
                ) AND actual_win = 1 THEN 1 ELSE 0 END) as correct_winners
            FROM horses
            WHERE actual_finish IS NOT NULL
        """)
        total, correct = cursor.fetchone()
        
        # Brier score (calibration metric)
        cursor.execute("""
            SELECT AVG((predicted_win_prob - actual_win) * (predicted_win_prob - actual_win)) as brier_score
            FROM horses
            WHERE actual_finish IS NOT NULL
        """)
        brier = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_races': total or 0,
            'correct_winners': correct or 0,
            'win_accuracy': (correct / total * 100) if total > 0 else 0,
            'brier_score': brier or 0,
            'samples_for_training': total
        }
    
    def get_race_summary(self) -> Dict:
        """Get summary statistics about stored races."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("SELECT COUNT(*) FROM races")
        total_races = cursor.fetchone()[0]
        
        cursor.execute("SELECT COUNT(*) FROM horses")
        total_horses = cursor.fetchone()[0]
        
        conn.close()
        
        return {
            'total_races': total_races or 0,
            'total_horses': total_horses or 0
        }
    
    def get_races(self, limit: int = 20) -> List[Dict]:
        """Get recent races with details."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT race_id, track, date, surface, distance, created_at
            FROM races
            ORDER BY created_at DESC
            LIMIT ?
        """, (limit,))
        
        rows = cursor.fetchall()
        conn.close()
        
        return [
            {
                'race_id': row[0],
                'track': row[1],
                'date': row[2],
                'surface': row[3],
                'distance': row[4],
                'created_at': row[5]
            }
            for row in rows
        ]

# ===================== Neural Network Models =====================

if USING_TORCH:
    class ProbabilityRefinementNet(nn.Module):
        """PyTorch neural network for probability calibration."""
        
        def __init__(self, input_dim: int = 15):
            super().__init__()
            self.fc1 = nn.Linear(input_dim, 64)
            self.bn1 = nn.BatchNorm1d(64)
            self.dropout1 = nn.Dropout(0.3)
            
            self.fc2 = nn.Linear(64, 32)
            self.bn2 = nn.BatchNorm1d(32)
            self.dropout2 = nn.Dropout(0.2)
            
            self.fc3 = nn.Linear(32, 16)
            self.fc4 = nn.Linear(16, 1)
            
        def forward(self, x):
            x = torch.relu(self.bn1(self.fc1(x)))
            x = self.dropout1(x)
            
            x = torch.relu(self.bn2(self.fc2(x)))
            x = self.dropout2(x)
            
            x = torch.relu(self.fc3(x))
            x = torch.sigmoid(self.fc4(x))
            return x

class MLCalibrator:
    """ML-based probability calibration using historical data."""
    
    def __init__(self, db_path: str = "race_history.db", model_path: str = "model.pth"):
        self.db = RaceDatabase(db_path)
        self.model_path = model_path
        self.model = None
        self.scaler = None
        self.is_trained = False
        
        if USING_TORCH:
            self.model = ProbabilityRefinementNet()
            self.load_model()
        else:
            self.scaler = StandardScaler()
            self.load_model()
    
    def load_model(self):
        """Load trained model from disk if exists."""
        if USING_TORCH:
            if os.path.exists(self.model_path):
                try:
                    self.model.load_state_dict(torch.load(self.model_path))
                    self.model.eval()
                    self.is_trained = True
                except Exception as e:
                    print(f"Could not load model: {e}")
        else:
            if os.path.exists(self.model_path):
                try:
                    with open(self.model_path, 'rb') as f:
                        saved = pickle.load(f)
                        self.model = saved['model']
                        self.scaler = saved['scaler']
                        self.is_trained = True
                except Exception as e:
                    print(f"Could not load model: {e}")
    
    def save_model(self):
        """Save trained model to disk."""
        if USING_TORCH:
            torch.save(self.model.state_dict(), self.model_path)
        else:
            with open(self.model_path, 'wb') as f:
                pickle.dump({'model': self.model, 'scaler': self.scaler}, f)
    
    def train(self, epochs: int = 100, learning_rate: float = 0.001) -> Dict:
        """Train the model on historical data."""
        features, targets = self.db.get_training_data(min_samples=50)
        
        if features is None:
            return {'status': 'insufficient_data', 'message': 'Need at least 50 race results'}
        
        # Split train/validation
        split = int(0.8 * len(features))
        X_train, X_val = features[:split], features[split:]
        y_train, y_val = targets[:split], targets[split:]
        
        if USING_TORCH:
            return self._train_torch(X_train, y_train, X_val, y_val, epochs, learning_rate)
        else:
            return self._train_sklearn(X_train, y_train, X_val, y_val)
    
    def _train_torch(self, X_train, y_train, X_val, y_val, epochs, lr):
        """Training with PyTorch."""
        # Normalize features
        mean = X_train.mean(axis=0)
        std = X_train.std(axis=0) + 1e-8
        X_train = (X_train - mean) / std
        X_val = (X_val - mean) / std
        
        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train)
        y_train_t = torch.FloatTensor(y_train).unsqueeze(1)
        X_val_t = torch.FloatTensor(X_val)
        y_val_t = torch.FloatTensor(y_val).unsqueeze(1)
        
        # Training setup
        criterion = nn.BCELoss()
        optimizer = optim.Adam(self.model.parameters(), lr=lr, weight_decay=1e-5)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=10, factor=0.5)
        
        best_val_loss = float('inf')
        patience_counter = 0
        max_patience = 20
        
        self.model.train()
        for epoch in range(epochs):
            # Forward pass
            optimizer.zero_grad()
            outputs = self.model(X_train_t)
            loss = criterion(outputs, y_train_t)
            
            # Backward pass
            loss.backward()
            optimizer.step()
            
            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t)
                val_loss = criterion(val_outputs, y_val_t)
                
                # Calculate accuracy (predicted winner = highest prob)
                val_accuracy = ((val_outputs > 0.5).float() == y_val_t).float().mean()
            
            self.model.train()
            scheduler.step(val_loss)
            
            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
                self.save_model()
            else:
                patience_counter += 1
                if patience_counter >= max_patience:
                    break
        
        self.model.eval()
        self.is_trained = True
        
        # Calculate final metrics
        with torch.no_grad():
            val_preds = self.model(X_val_t).numpy()
            brier = np.mean((val_preds.flatten() - y_val) ** 2)
        
        return {
            'status': 'success',
            'epochs_trained': epoch + 1,
            'final_val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy * 100),
            'brier_score': float(brier),
            'training_samples': len(X_train)
        }
    
    def _train_sklearn(self, X_train, y_train, X_val, y_val):
        """Training with sklearn MLPRegressor."""
        # Normalize
        X_train = self.scaler.fit_transform(X_train)
        X_val = self.scaler.transform(X_val)
        
        # Train model
        self.model = MLPRegressor(
            hidden_layer_sizes=(64, 32, 16),
            activation='relu',
            solver='adam',
            alpha=0.001,
            batch_size=32,
            learning_rate='adaptive',
            max_iter=200,
            early_stopping=True,
            validation_fraction=0.1,
            random_state=42
        )
        
        self.model.fit(X_train, y_train)
        
        # Evaluate
        val_preds = self.model.predict(X_val)
        val_preds = np.clip(val_preds, 0, 1)  # Ensure [0,1]
        
        brier = np.mean((val_preds - y_val) ** 2)
        accuracy = np.mean((val_preds > 0.5) == y_val) * 100
        
        self.save_model()
        self.is_trained = True
        
        return {
            'status': 'success',
            'iterations': self.model.n_iter_,
            'val_accuracy': accuracy,
            'brier_score': brier,
            'training_samples': len(X_train)
        }
    
    def refine_probabilities(self, features: np.ndarray, raw_probs: np.ndarray) -> np.ndarray:
        """
        Refine raw probabilities using trained model.
        
        Args:
            features: (n_horses, 15) array of features
            raw_probs: (n_horses,) array of raw softmax probabilities
        
        Returns:
            refined_probs: (n_horses,) array of refined probabilities
        """
        if not self.is_trained:
            return raw_probs  # Return unchanged if not trained
        
        # Add raw_prob as a feature (it's already in features at index 7, but this ensures it)
        if features.shape[1] == 14:
            features = np.column_stack([features, raw_probs])
        
        if USING_TORCH:
            # Normalize (use training stats if available)
            self.model.eval()
            with torch.no_grad():
                X = torch.FloatTensor(features)
                refined = self.model(X).numpy().flatten()
        else:
            X = self.scaler.transform(features)
            refined = self.model.predict(X)
            refined = np.clip(refined, 0.001, 0.999)
        
        # Renormalize to sum to 1
        refined = refined / refined.sum()
        
        return refined

# ===================== Helper Functions =====================

def create_feature_vector(horse_data: Dict, race_context: Dict, raw_prob: float) -> np.ndarray:
    """Create 15-dimensional feature vector for a horse."""
    return np.array([
        horse_data.get('rating_class', 0),
        horse_data.get('rating_speed', 0),
        horse_data.get('rating_pace', 0),
        horse_data.get('rating_style', 0),
        horse_data.get('rating_post', 0),
        horse_data.get('rating_angles', 0),
        horse_data.get('rating_pedigree', 0),
        raw_prob,  # Include raw probability as feature
        horse_data.get('final_odds', 5.0),
        horse_data.get('quirin_points', 0),
        horse_data.get('last_beyer', 75),
        horse_data.get('avg_beyer', 75),
        race_context.get('field_size', 8),
        race_context.get('ppi', 0),
        race_context.get('purse', 50000)
    ])

def export_training_data_csv(db_path: str = "race_history.db", output_path: str = "training_data.csv"):
    """Export training data to CSV for analysis."""
    conn = sqlite3.connect(db_path)
    query = """
        SELECT 
            r.date, r.track, r.surface, r.distance, r.race_type, r.purse, r.field_size,
            h.horse_name, h.post_position, h.final_odds,
            h.rating_total, h.rating_class, h.rating_speed, h.rating_pace,
            h.predicted_win_prob, h.actual_finish, h.actual_win
        FROM horses h
        JOIN races r ON h.race_id = r.race_id
        WHERE h.actual_finish IS NOT NULL
        ORDER BY r.date DESC
    """
    df = pd.read_sql_query(query, conn)
    conn.close()
    df.to_csv(output_path, index=False)
    return len(df)
