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
        """Initialize database schema – robust to pre-existing tables with
        different column layouts (e.g. gold_high_iq created by an older
        helper script).  Each DDL statement is executed independently so
        one failure does not block creation of later tables / views."""
        import os

        schema_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                   "gold_database_schema.sql")
        try:
            with open(schema_path, "r", encoding="utf-8") as f:
                schema_sql = f.read()
        except FileNotFoundError:
            logger.warning("gold_database_schema.sql not found – using inline fallback")
            schema_sql = ""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ---------- execute the SQL file, statement-by-statement ----------
        if schema_sql:
            # Split on ';' and run each independently
            for stmt in schema_sql.split(';'):
                stmt = stmt.strip()
                if not stmt:
                    continue
                try:
                    cursor.execute(stmt)
                except Exception as stmt_err:
                    # Log but continue – e.g. index on a column that doesn't
                    # exist in an older table layout is non-fatal.
                    logger.debug(f"Schema stmt skipped: {stmt_err}")

        # ---------- guarantee critical tables exist (inline fallback) -----
        critical_tables = {
            "races_analyzed": """
                CREATE TABLE IF NOT EXISTS races_analyzed (
                    race_id TEXT PRIMARY KEY,
                    track_code TEXT NOT NULL,
                    race_date TEXT NOT NULL,
                    race_number INTEGER NOT NULL,
                    race_type TEXT, surface TEXT, distance TEXT,
                    track_condition TEXT, purse REAL, field_size INTEGER,
                    pp_text_raw TEXT,
                    analyzed_timestamp TEXT NOT NULL,
                    UNIQUE(track_code, race_date, race_number)
                )""",
            "horses_analyzed": """
                CREATE TABLE IF NOT EXISTS horses_analyzed (
                    horse_id TEXT PRIMARY KEY,
                    race_id TEXT NOT NULL,
                    program_number INTEGER NOT NULL,
                    horse_name TEXT NOT NULL,
                    post_position INTEGER,
                    morning_line_odds REAL,
                    jockey TEXT, trainer TEXT, owner TEXT,
                    weight REAL, medication TEXT, equipment TEXT,
                    running_style TEXT, prime_power REAL,
                    best_beyer INTEGER, last_beyer INTEGER, avg_beyer_3 REAL,
                    e1_pace REAL, e2_pace REAL, late_pace REAL,
                    days_since_last INTEGER, starts_lifetime INTEGER,
                    wins_lifetime INTEGER, win_pct REAL,
                    earnings_lifetime REAL, class_rating REAL,
                    angle_early_speed REAL, angle_class REAL,
                    angle_recency REAL, angle_work_pattern REAL,
                    angle_connections REAL, angle_pedigree REAL,
                    angle_runstyle_bias REAL, angle_post REAL,
                    rating_class REAL, rating_form REAL,
                    rating_speed REAL, rating_pace REAL,
                    rating_style REAL, rating_post REAL,
                    rating_angles_total REAL, rating_tier2_bonus REAL,
                    rating_final REAL, rating_confidence REAL,
                    form_decay_score REAL, pace_esp_score REAL,
                    mud_adjustment REAL,
                    predicted_probability REAL NOT NULL DEFAULT 0,
                    predicted_rank INTEGER, fair_odds REAL,
                    FOREIGN KEY (race_id) REFERENCES races_analyzed(race_id)
                )""",
            "race_results_summary": """
                CREATE TABLE IF NOT EXISTS race_results_summary (
                    race_id TEXT PRIMARY KEY,
                    winner_name TEXT, second_name TEXT,
                    third_name TEXT, fourth_name TEXT, fifth_name TEXT,
                    top1_predicted_correctly BOOLEAN,
                    top3_predicted_correctly INTEGER,
                    top5_predicted_correctly INTEGER,
                    winner_predicted_odds REAL,
                    winner_actual_payout REAL,
                    roi_if_bet_on_predicted_winner REAL,
                    results_complete_timestamp TEXT NOT NULL,
                    FOREIGN KEY (race_id) REFERENCES races_analyzed(race_id)
                )""",
        }
        for tbl_name, ddl in critical_tables.items():
            try:
                cursor.execute(ddl)
            except Exception as tbl_err:
                logger.warning(f"Could not ensure table {tbl_name}: {tbl_err}")

        # ---------- guarantee the v_pending_races view -------------------
        try:
            cursor.execute("""
                CREATE VIEW IF NOT EXISTS v_pending_races AS
                SELECT ra.race_id, ra.track_code, ra.race_date,
                       ra.race_number, ra.field_size
                FROM races_analyzed ra
                WHERE NOT EXISTS (
                    SELECT 1 FROM race_results_summary rs
                    WHERE rs.race_id = ra.race_id
                )
                ORDER BY ra.race_date DESC, ra.race_number
            """)
        except Exception:
            # View may already exist with different definition; that's OK
            pass

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
        conn = sqlite3.connect(self.db_path, timeout=5)
        cursor = conn.cursor()

        pending: list = []

        # Strategy 1: use the v_pending_races view (preferred)
        try:
            cursor.execute("""
                SELECT race_id, track_code, race_date, race_number, field_size
                FROM v_pending_races
                LIMIT ?
            """, (limit,))
            pending = cursor.fetchall()
        except Exception:
            pass

        # Strategy 2: direct query against races_analyzed (skip view)
        if not pending:
            try:
                cursor.execute("""
                    SELECT ra.race_id, ra.track_code, ra.race_date,
                           ra.race_number, ra.field_size
                    FROM races_analyzed ra
                    WHERE NOT EXISTS (
                        SELECT 1 FROM race_results_summary rs
                        WHERE rs.race_id = ra.race_id
                    )
                    ORDER BY ra.race_date DESC, ra.race_number
                    LIMIT ?
                """, (limit,))
                pending = cursor.fetchall()
            except Exception:
                pass

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
        finish_order_programs: List[int],
        horses_ui: Optional[List[Dict]] = None
    ) -> bool:
        """
        Submit actual race results (top 4 finish positions).

        Args:
            race_id: Race identifier
            finish_order_programs: List of program numbers [1st, 2nd, 3rd, 4th]
            horses_ui: Optional list of horse dicts from the UI (fallback when
                       horses_analyzed table is empty for this race)

        Returns:
            True if successful
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            cursor = conn.cursor()

            # ------ ensure race_results_summary table exists ------
            cursor.execute("""
                CREATE TABLE IF NOT EXISTS race_results_summary (
                    race_id TEXT PRIMARY KEY,
                    winner_name TEXT, second_name TEXT,
                    third_name TEXT, fourth_name TEXT, fifth_name TEXT,
                    top1_predicted_correctly BOOLEAN,
                    top3_predicted_correctly INTEGER,
                    top5_predicted_correctly INTEGER,
                    winner_predicted_odds REAL,
                    winner_actual_payout REAL,
                    roi_if_bet_on_predicted_winner REAL,
                    results_complete_timestamp TEXT NOT NULL
                )
            """)

            # ------ load horse data from horses_analyzed ----------
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

            all_horses_rows = cursor.fetchall()
            horse_dict = {row[1]: row for row in all_horses_rows}

            # ------ fallback: build horse_dict from UI data -------
            ui_dict: Dict[int, Dict] = {}
            if horses_ui:
                for h in horses_ui:
                    pn = int(h.get('program_number', h.get('post_position', 0)))
                    ui_dict[pn] = h

            # Choose a name-lookup helper that works with either source
            def _horse_name(prog: int) -> str:
                if prog in horse_dict:
                    return horse_dict[prog][2]
                if prog in ui_dict:
                    return ui_dict[prog].get('horse_name', f'Horse #{prog}')
                return f'Horse #{prog}'

            # ------ detect which gold_high_iq schema we have ------
            cursor.execute("PRAGMA table_info(gold_high_iq)")
            gold_cols = {row[1] for row in cursor.fetchall()}
            uses_new_schema = 'result_id' in gold_cols  # schema from gold_database_schema.sql
            uses_old_schema = 'track' in gold_cols and 'id' in gold_cols

            # ------ insert per-horse results ----------------------
            for actual_position, program_num in enumerate(finish_order_programs[:4], 1):
                if uses_new_schema and program_num in horse_dict:
                    # New schema path (result_id TEXT PRIMARY KEY)
                    horse_row = horse_dict[program_num]
                    horse_id = horse_row[0]
                    predicted_rank = horse_row[6] or 99
                    prediction_error = abs(predicted_rank - actual_position)
                    was_top4_correct = (predicted_rank <= 4 and actual_position <= 4)

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
                        'running_style': horse_row[22],
                    }
                    result_id = f"{race_id}_{program_num}"
                    cursor.execute("""
                        INSERT OR REPLACE INTO gold_high_iq
                        (result_id, race_id, horse_id, actual_finish_position,
                         program_number, horse_name, post_position, rating_final,
                         predicted_probability, predicted_rank, features_json,
                         prediction_error, was_top5_correct,
                         result_entered_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        result_id, race_id, horse_id, actual_position,
                        program_num, horse_row[2], horse_row[3], horse_row[4],
                        horse_row[5], predicted_rank, json.dumps(features),
                        prediction_error, was_top4_correct,
                        datetime.now().isoformat(),
                    ))

                elif uses_old_schema:
                    # Old schema path (id INTEGER PRIMARY KEY AUTOINCREMENT)
                    h = ui_dict.get(program_num, {}) if ui_dict else {}
                    db_h = horse_dict.get(program_num)
                    name = _horse_name(program_num)
                    predicted_rank = (db_h[6] if db_h else
                                      int(h.get('predicted_rank', 99)))
                    prediction_error = abs(predicted_rank - actual_position)

                    # Parse race_id components (TRACK_YYYYMMDD_R#)
                    parts = race_id.rsplit('_', 2)
                    track_code = parts[0] if len(parts) >= 3 else 'UNK'
                    r_date = parts[1] if len(parts) >= 3 else ''
                    r_num = parts[2].replace('R', '') if len(parts) >= 3 else '0'

                    cursor.execute("""
                        INSERT INTO gold_high_iq
                        (race_id, track, race_num, race_date,
                         horse_name, program_number,
                         actual_finish_position, predicted_finish_position,
                         prediction_error, post_position, field_size,
                         prime_power, running_style, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        race_id, track_code, int(r_num) if r_num.isdigit() else 0,
                        r_date, name, program_num,
                        actual_position, predicted_rank,
                        prediction_error,
                        int(h.get('post_position', 0)),
                        int(h.get('field_size', len(ui_dict))),
                        float(h.get('prime_power', 0)),
                        h.get('running_style', ''),
                        datetime.now().isoformat(),
                    ))

                else:
                    logger.warning(
                        f"Program #{program_num} not found – "
                        f"no matching schema or horse data")

            # ------ race_results_summary row ----------------------
            # Compute basic accuracy
            predicted_top4_programs: List[int] = []
            if horse_dict:
                predicted_top4 = sorted(
                    all_horses_rows, key=lambda x: (x[6] or 99))[:4]
                predicted_top4_programs = [h[1] for h in predicted_top4]
            elif ui_dict:
                # Sort UI horses by predicted_rank
                sorted_ui = sorted(
                    ui_dict.values(),
                    key=lambda h: int(h.get('predicted_rank', 99)))[:4]
                predicted_top4_programs = [
                    int(h.get('program_number', 0)) for h in sorted_ui]

            top1_correct = (predicted_top4_programs[0] == finish_order_programs[0]
                            if predicted_top4_programs else False)
            top3_hit = (len(set(predicted_top4_programs[:3])
                            & set(finish_order_programs[:3]))
                        if predicted_top4_programs else 0)
            top4_hit = (len(set(predicted_top4_programs[:4])
                            & set(finish_order_programs[:4]))
                        if predicted_top4_programs else 0)

            cursor.execute("""
                INSERT OR REPLACE INTO race_results_summary
                (race_id, winner_name, second_name, third_name,
                 fourth_name, fifth_name,
                 top1_predicted_correctly, top3_predicted_correctly,
                 top5_predicted_correctly, results_complete_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                race_id,
                _horse_name(finish_order_programs[0]),
                _horse_name(finish_order_programs[1]) if len(finish_order_programs) > 1 else 'N/A',
                _horse_name(finish_order_programs[2]) if len(finish_order_programs) > 2 else 'N/A',
                _horse_name(finish_order_programs[3]) if len(finish_order_programs) > 3 else 'N/A',
                'N/A',
                top1_correct,
                top3_hit,
                top4_hit,
                datetime.now().isoformat(),
            ))

            conn.commit()
            conn.close()

            logger.info(f"✅ Results submitted for {race_id} | "
                        f"Winner: {_horse_name(finish_order_programs[0])}")
            return True

        except Exception as e:
            logger.error(f"❌ Error submitting results for {race_id}: {e}")
            import traceback
            traceback.print_exc()
            try:
                conn.rollback()
                conn.close()
            except Exception:
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
        conn = sqlite3.connect(self.db_path, timeout=5)
        cursor = conn.cursor()

        try:
            cursor.execute("""
                SELECT
                    COUNT(DISTINCT race_id) as total_races,
                    AVG(CASE WHEN top1_predicted_correctly THEN 1 ELSE 0 END) as winner_accuracy,
                    AVG(top3_predicted_correctly) / 3.0 as top3_accuracy,
                    AVG(COALESCE(top5_predicted_correctly, 0)) / 5.0 as top5_accuracy
                FROM race_results_summary
            """)
            stats = cursor.fetchone()
        except Exception:
            # Table may not exist yet
            stats = (0, 0.0, 0.0, 0.0)

        conn.close()

        return {
            'total_races': stats[0] or 0,
            'winner_accuracy': stats[1] or 0.0,
            'top3_accuracy': stats[2] or 0.0,
            'top5_accuracy': stats[3] or 0.0,
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
