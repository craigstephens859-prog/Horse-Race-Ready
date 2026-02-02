"""
GOLD HIGH-IQ DATABASE MANAGER
=============================

Handles all database operations for the gold-standard ML retraining system.

Features:
- Auto-save race analysis data
- Clean result entry
- Optimized queries for ML retraining
- Performance tracking

Author: Top-Tier ML + Full Stack Engineer
Date: January 29, 2026
"""

import sqlite3
import json
from datetime import datetime
from typing import Dict, List, Optional, Tuple
import pandas as pd
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class GoldHighIQDatabase:
    """Production-grade database manager for ML retraining system."""
    
    def __init__(self, db_path: str = "gold_high_iq.db"):
        """
        Initialize database connection.
        
        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path
        self._init_schema()
    
    def _init_schema(self):
        """Initialize database schema from SQL file."""
        with open("gold_database_schema.sql", "r", encoding="utf-8") as f:
            schema_sql = f.read()
        
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        # Execute schema (multiple statements)
        cursor.executescript(schema_sql)
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Database initialized: {self.db_path}")
    
    def save_analyzed_race(
        self,
        race_id: str,
        race_metadata: Dict,
        horses_data: List[Dict],
        pp_text_raw: str
    ) -> bool:
        """
        Save complete race analysis data automatically after "Analyze This Race".
        
        Args:
            race_id: Unique race identifier (TRACK_YYYYMMDD_R#)
            race_metadata: Dict with track, date, race_num, race_type, etc.
            horses_data: List of dicts, one per horse with all features
            pp_text_raw: Raw BRISNET PP text
        
        Returns:
            True if successful, False otherwise
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # CHECK IF RACE ALREADY HAS RESULTS SUBMITTED
            cursor.execute("""
                SELECT COUNT(*) FROM gold_high_iq 
                WHERE race_id = ?
            """, (race_id,))
            has_results = cursor.fetchone()[0] > 0
            
            if has_results:
                logger.warning(f"⚠️ Race {race_id} already has results. Updating predictions but preserving results.")
            
            # 1. Insert race record (INSERT OR REPLACE updates if exists)
            cursor.execute("""
                INSERT OR REPLACE INTO races_analyzed 
                (race_id, track_code, race_date, race_number, race_type, 
                 surface, distance, track_condition, purse, field_size, 
                 pp_text_raw, analyzed_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id,
                race_metadata.get('track', 'UNK'),
                race_metadata.get('date', datetime.now().strftime('%Y-%m-%d')),
                race_metadata.get('race_num', 1),
                race_metadata.get('race_type', 'UNK'),
                race_metadata.get('surface', 'Dirt'),
                race_metadata.get('distance', '6F'),
                race_metadata.get('condition', 'Fast'),
                race_metadata.get('purse', 0.0),
                len(horses_data),
                pp_text_raw,
                datetime.now().isoformat()
            ))
            
            # 2. Insert all horses
            for horse in horses_data:
                horse_id = f"{race_id}_{horse['program_number']}"
                
                cursor.execute("""
                    INSERT OR REPLACE INTO horses_analyzed 
                    (horse_id, race_id, program_number, horse_name, post_position, 
                     morning_line_odds, jockey, trainer, owner, running_style,
                     prime_power, best_beyer, last_beyer, avg_beyer_3,
                     e1_pace, e2_pace, late_pace, days_since_last,
                     starts_lifetime, wins_lifetime, win_pct, earnings_lifetime,
                     class_rating, angle_early_speed, angle_class, angle_recency,
                     angle_work_pattern, angle_connections, angle_pedigree,
                     angle_runstyle_bias, angle_post, rating_class, rating_form,
                     rating_speed, rating_pace, rating_style, rating_post,
                     rating_angles_total, rating_tier2_bonus, rating_final,
                     rating_confidence, form_decay_score, pace_esp_score,
                     mud_adjustment, predicted_probability, predicted_rank, fair_odds)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    horse_id,
                    race_id,
                    horse.get('program_number', 0),
                    horse.get('horse_name', 'Unknown'),
                    horse.get('post_position', 0),
                    horse.get('morning_line_odds', 99.0),
                    horse.get('jockey', ''),
                    horse.get('trainer', ''),
                    horse.get('owner', ''),
                    horse.get('running_style', 'P'),
                    horse.get('prime_power', 0.0),
                    horse.get('best_beyer', 0),
                    horse.get('last_beyer', 0),
                    horse.get('avg_beyer_3', 0.0),
                    horse.get('e1_pace', 0.0),
                    horse.get('e2_pace', 0.0),
                    horse.get('late_pace', 0.0),
                    horse.get('days_since_last', 0),
                    horse.get('starts_lifetime', 0),
                    horse.get('wins_lifetime', 0),
                    horse.get('win_pct', 0.0),
                    horse.get('earnings_lifetime', 0.0),
                    horse.get('class_rating', 0.0),
                    horse.get('angle_early_speed', 0.0),
                    horse.get('angle_class', 0.0),
                    horse.get('angle_recency', 0.0),
                    horse.get('angle_work_pattern', 0.0),
                    horse.get('angle_connections', 0.0),
                    horse.get('angle_pedigree', 0.0),
                    horse.get('angle_runstyle_bias', 0.0),
                    horse.get('angle_post', 0.0),
                    horse.get('rating_class', 0.0),
                    horse.get('rating_form', 0.0),
                    horse.get('rating_speed', 0.0),
                    horse.get('rating_pace', 0.0),
                    horse.get('rating_style', 0.0),
                    horse.get('rating_post', 0.0),
                    horse.get('rating_angles_total', 0.0),
                    horse.get('rating_tier2_bonus', 0.0),
                    horse.get('rating_final', 0.0),
                    horse.get('rating_confidence', 0.5),
                    horse.get('form_decay_score', 0.0),
                    horse.get('pace_esp_score', 0.0),
                    horse.get('mud_adjustment', 0.0),
                    horse.get('predicted_probability', 0.0),
                    horse.get('predicted_rank', 99),
                    horse.get('fair_odds', 99.0)
                ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Saved race {race_id} with {len(horses_data)} horses")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving race {race_id}: {e}")
            # ROLLBACK ON ERROR - ensures database integrity
            try:
                conn.rollback()
                conn.close()
            except:
                pass
            return False
    
    def get_pending_races(self, limit: int = 20) -> List[Tuple]:
        """
        Get races that have been analyzed but results not entered yet.
        
        Returns:
            List of tuples: (race_id, track, date, race_num, field_size)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT race_id, track_code, race_date, race_number, field_size
            FROM v_pending_races
            LIMIT ?
        """, (limit,))
        
        pending = cursor.fetchall()
        conn.close()
        
        return pending
    
    def get_horses_for_race(self, race_id: str) -> List[Dict]:
        """
        Get all horses for a specific race.
        
        Returns:
            List of dicts with horse data
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                horse_id,
                program_number,
                horse_name,
                post_position,
                morning_line_odds,
                predicted_probability,
                predicted_rank,
                fair_odds
            FROM horses_analyzed
            WHERE race_id = ?
            ORDER BY program_number
        """, (race_id,))
        
        horses = [dict(row) for row in cursor.fetchall()]
        conn.close()
        
        return horses
    
    def submit_race_results(
        self,
        race_id: str,
        finish_order_programs: List[int]
    ) -> bool:
        """
        Submit actual race results (top 5 finish positions).
        
        Args:
            race_id: Race identifier
            finish_order_programs: List of program numbers [1st, 2nd, 3rd, 4th, 5th]
        
        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            
            # Get all horses for this race
            cursor.execute("""
                SELECT 
                    horse_id, program_number, horse_name, post_position,
                    rating_final, predicted_probability, predicted_rank,
                    rating_class, rating_form, rating_speed, rating_pace,
                    rating_style, rating_post, rating_angles_total,
                    rating_confidence, form_decay_score, pace_esp_score,
                    angle_early_speed, angle_class, angle_recency,
                    prime_power, best_beyer, running_style
                FROM horses_analyzed
                WHERE race_id = ?
            """, (race_id,))
            
            all_horses = cursor.fetchall()
            horse_dict = {row[1]: row for row in all_horses}  # program_number -> row
            
            # Insert results into gold_high_iq table
            for actual_position, program_num in enumerate(finish_order_programs, 1):
                if program_num not in horse_dict:
                    logger.warning(f"Program #{program_num} not found in race {race_id}")
                    continue
                
                horse_row = horse_dict[program_num]
                horse_id = horse_row[0]
                predicted_rank = horse_row[6]
                
                # Calculate metrics
                prediction_error = abs(predicted_rank - actual_position)
                was_top5_correct = (predicted_rank <= 5 and actual_position <= 5)
                
                # Build features JSON
                features = {
                    'rating_class': horse_row[7],
                    'rating_form': horse_row[8],
                    'rating_speed': horse_row[9],
                    'rating_pace': horse_row[10],
                    'rating_style': horse_row[11],
                    'rating_post': horse_row[12],
                    'rating_angles_total': horse_row[13],
                    'rating_confidence': horse_row[14],
                    'form_decay_score': horse_row[15],
                    'pace_esp_score': horse_row[16],
                    'angle_early_speed': horse_row[17],
                    'angle_class': horse_row[18],
                    'angle_recency': horse_row[19],
                    'prime_power': horse_row[20],
                    'best_beyer': horse_row[21],
                    'running_style': horse_row[22]
                }
                
                result_id = f"{race_id}_{program_num}"
                
                cursor.execute("""
                    INSERT OR REPLACE INTO gold_high_iq
                    (result_id, race_id, horse_id, actual_finish_position,
                     program_number, horse_name, post_position, rating_final,
                     predicted_probability, predicted_rank, features_json,
                     prediction_error, was_top5_correct, result_entered_timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    result_id, race_id, horse_id, actual_position,
                    program_num, horse_row[2], horse_row[3], horse_row[4],
                    horse_row[5], predicted_rank, json.dumps(features),
                    prediction_error, was_top5_correct,
                    datetime.now().isoformat()
                ))
            
            # Update race_results_summary
            winner_name = horse_dict[finish_order_programs[0]][2] if finish_order_programs[0] in horse_dict else 'Unknown'
            
            # Calculate accuracy metrics
            predicted_top5 = sorted(all_horses, key=lambda x: x[6])[:5]  # Sort by predicted_rank
            predicted_top5_programs = [h[1] for h in predicted_top5]
            
            top1_correct = (predicted_top5_programs[0] == finish_order_programs[0])
            top3_correct_count = len(set(predicted_top5_programs[:3]) & set(finish_order_programs[:3]))
            top5_correct_count = len(set(predicted_top5_programs[:5]) & set(finish_order_programs[:5]))
            
            cursor.execute("""
                INSERT OR REPLACE INTO race_results_summary
                (race_id, winner_name, second_name, third_name, fourth_name, fifth_name,
                 top1_predicted_correctly, top3_predicted_correctly, top5_predicted_correctly,
                 results_complete_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id,
                horse_dict[finish_order_programs[0]][2] if finish_order_programs[0] in horse_dict else 'Unknown',
                horse_dict[finish_order_programs[1]][2] if finish_order_programs[1] in horse_dict else 'Unknown',
                horse_dict[finish_order_programs[2]][2] if finish_order_programs[2] in horse_dict else 'Unknown',
                horse_dict[finish_order_programs[3]][2] if finish_order_programs[3] in horse_dict else 'Unknown',
                horse_dict[finish_order_programs[4]][2] if finish_order_programs[4] in horse_dict else 'Unknown',
                top1_correct,
                top3_correct_count,
                top5_correct_count,
                datetime.now().isoformat()
            ))
            
            conn.commit()
            conn.close()
            
            logger.info(f"✅ Results submitted for {race_id} | Winner: {winner_name}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error submitting results for {race_id}: {e}")
            import traceback
            traceback.print_exc()
            # ROLLBACK ON ERROR - ensures database integrity
            try:
                conn.rollback()
                conn.close()
            except:
                pass
            return False
    
    def get_training_data(
        self,
        min_races: int = 50
    ) -> Optional[Tuple[pd.DataFrame, pd.DataFrame]]:
        """
        Get training data for ML model retraining.
        
        Args:
            min_races: Minimum number of completed races required
        
        Returns:
            (features_df, labels_df) or None if insufficient data
        """
        conn = sqlite3.connect(self.db_path)
        
        # Check if we have enough data
        cursor = conn.cursor()
        cursor.execute("SELECT COUNT(DISTINCT race_id) FROM gold_high_iq")
        num_races = cursor.fetchone()[0]
        
        if num_races < min_races:
            logger.warning(f"⚠️ Only {num_races} completed races. Need {min_races} minimum.")
            conn.close()
            return None
        
        # Load all training data
        df = pd.read_sql_query("""
            SELECT 
                race_id,
                horse_id,
                actual_finish_position,
                features_json,
                predicted_probability,
                predicted_rank,
                prediction_error
            FROM gold_high_iq
            ORDER BY race_id, actual_finish_position
        """, conn)
        
        conn.close()
        
        # Parse features JSON
        features_list = []
        for _, row in df.iterrows():
            features = json.loads(row['features_json'])
            features['race_id'] = row['race_id']
            features['horse_id'] = row['horse_id']
            features['actual_finish'] = row['actual_finish_position']
            features_list.append(features)
        
        features_df = pd.DataFrame(features_list)
        labels_df = df[['race_id', 'horse_id', 'actual_finish_position']]
        
        logger.info(f"✅ Loaded {len(features_df)} horses from {num_races} races for training")
        
        return features_df, labels_df
    
    def get_accuracy_stats(self) -> Dict:
        """
        Get overall prediction accuracy statistics.
        
        Returns:
            Dict with accuracy metrics
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            SELECT 
                COUNT(DISTINCT race_id) as total_races,
                AVG(CASE WHEN top1_predicted_correctly THEN 1 ELSE 0 END) as winner_accuracy,
                AVG(top3_predicted_correctly) / 3.0 as top3_accuracy,
                AVG(top5_predicted_correctly) / 5.0 as top5_accuracy
            FROM race_results_summary
        """)
        
        stats = cursor.fetchone()
        conn.close()
        
        return {
            'total_races': stats[0] or 0,
            'winner_accuracy': stats[1] or 0.0,
            'top3_accuracy': stats[2] or 0.0,
            'top5_accuracy': stats[3] or 0.0
        }
    
    def log_retraining(
        self,
        metrics: Dict,
        model_path: str
    ) -> int:
        """
        Log a model retraining session.
        
        Args:
            metrics: Dict with training metrics
            model_path: Path to saved model checkpoint
        
        Returns:
            retrain_id
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        
        cursor.execute("""
            INSERT INTO retraining_history
            (retrain_timestamp, total_races_used, total_horses_used,
             train_split_pct, val_split_pct, val_winner_accuracy,
             val_top3_accuracy, val_top5_accuracy, val_loss,
             model_checkpoint_path, epochs_trained, learning_rate,
             batch_size, training_duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            datetime.now().isoformat(),
            metrics.get('total_races', 0),
            metrics.get('total_horses', 0),
            metrics.get('train_split_pct', 0.8),
            metrics.get('val_split_pct', 0.2),
            metrics.get('val_winner_accuracy', 0.0),
            metrics.get('val_top3_accuracy', 0.0),
            metrics.get('val_top5_accuracy', 0.0),
            metrics.get('val_loss', 0.0),
            model_path,
            metrics.get('epochs', 0),
            metrics.get('learning_rate', 0.001),
            metrics.get('batch_size', 32),
            metrics.get('training_duration', 0.0)
        ))
        
        retrain_id = cursor.lastrowid
        
        conn.commit()
        conn.close()
        
        logger.info(f"✅ Logged retraining session #{retrain_id}")
        return retrain_id
