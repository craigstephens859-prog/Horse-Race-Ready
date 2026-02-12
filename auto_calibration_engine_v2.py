"""
AUTO-CALIBRATION ENGINE v2.0
============================

Automatically adjusts prediction weights based on historical prediction errors.
Uses gradient descent to minimize prediction error over time.

Key Features v2.0 (Updated Feb 2026 with Pegasus World Cup learnings):
- Learns from race results stored in gold_high_iq.db
- PERSISTS learned weights to database (survives restarts)
- Learns from odds drift patterns (smart money detection)
- Applies L2 regularization to prevent overfitting
- Exports calibrated weights for unified_rating_engine.py
- Provides real-time weight application to predictions

Author: Elite ML Engineer
Date: February 6, 2026
"""

import json
import logging
import sqlite3
from datetime import datetime

import numpy as np

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class AutoCalibrationEngine:
    """
    Automatic weight calibration using gradient descent.

    Learns optimal weights from historical prediction errors.
    v2.0: Now with persistent storage and odds drift learning.
    """

    def __init__(self, db_path: str = "gold_high_iq.db"):
        """Initialize calibration engine with persistent storage."""
        self.db_path = db_path
        self.learning_rate = 0.05  # Conservative learning
        self.regularization = 0.01  # L2 penalty

        # Initialize database tables for persistent learning
        self._init_learning_tables()

        # Load persisted weights (or use defaults)
        self.base_weights = self._load_learned_weights()

        # Odds drift learned adjustments
        self.odds_drift_weights = self._load_odds_drift_weights()

        # Race-type modifiers
        self.modifiers = {
            "grade_1_2": {
                "class": 1.0,
                "speed": 1.1,
                "form": 1.2,
                "pace": 1.1,
                "style": 1.2,
                "post": 1.0,
            }
        }

        # Track calibration history
        self.calibration_log = []

    def _init_learning_tables(self):
        """Create tables to persist learned weights across sessions."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Table for main component weights
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS learned_weights (
                weight_id TEXT PRIMARY KEY,
                weight_name TEXT NOT NULL,
                weight_value REAL NOT NULL,
                races_trained_on INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL DEFAULT 0.5,
                notes TEXT
            )
        """)

        # Table for odds drift learning
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS odds_drift_learning (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                odds_direction TEXT NOT NULL,
                drift_magnitude REAL NOT NULL,
                actual_finish INTEGER NOT NULL,
                expected_finish INTEGER NOT NULL,
                race_id TEXT,
                timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)

        # Table for calibration history
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS calibration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                calibration_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                races_analyzed INTEGER,
                winner_accuracy REAL,
                top3_accuracy REAL,
                weights_json TEXT,
                improvements_json TEXT
            )
        """)

        # ── Track-Specific Learned Weights ──────────────────────────
        # Each track across America gets its own set of calibrated weights,
        # enabling the system to learn unique track profiles over time.
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_learned_weights (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_code TEXT NOT NULL,
                weight_name TEXT NOT NULL,
                weight_value REAL NOT NULL,
                races_trained_on INTEGER DEFAULT 0,
                last_updated TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                confidence REAL DEFAULT 0.5,
                UNIQUE(track_code, weight_name)
            )
        """)

        # Track calibration history — per-track accuracy over time
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS track_calibration_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                track_code TEXT NOT NULL,
                calibration_timestamp TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                races_analyzed INTEGER,
                winner_accuracy REAL,
                top3_accuracy REAL,
                weights_json TEXT,
                improvements_json TEXT
            )
        """)

        conn.commit()
        conn.close()
        logger.info("✅ Learning tables initialized (incl. track-specific)")

    def _load_learned_weights(self) -> dict[str, float]:
        """Load persisted weights from database, or use defaults."""
        default_weights = {
            "class": 2.5,
            "speed": 2.0,
            "form": 1.8,
            "pace": 1.5,
            "style": 2.0,
            "post": 0.8,
            "angles": 0.10,
            # Odds drift weights (from Pegasus 2026 analysis)
            "odds_drift_penalty": 3.0,
            "smart_money_bonus": 2.5,
            "a_group_drift_gate": 2.0,
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute("SELECT weight_name, weight_value FROM learned_weights")
            rows = cursor.fetchall()
            conn.close()

            if rows:
                learned = {row[0]: row[1] for row in rows}
                merged = {**default_weights, **learned}
                logger.info(f"📚 Loaded {len(rows)} learned weights from database")
                return merged

        except Exception as e:
            logger.warning(f"Could not load learned weights: {e}")

        return default_weights

    def _save_learned_weights(self, races_trained: int = 0):
        """Persist current weights to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        for name, value in self.base_weights.items():
            cursor.execute(
                """
                INSERT OR REPLACE INTO learned_weights 
                (weight_id, weight_name, weight_value, races_trained_on, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    f"weight_{name}",
                    name,
                    value,
                    races_trained,
                    datetime.now().isoformat(),
                ),
            )

        conn.commit()
        conn.close()
        logger.info(f"💾 Saved {len(self.base_weights)} learned weights to database")

    # ═══════════════════════════════════════════════════════════════════
    #  TRACK-SPECIFIC CALIBRATION — Per-Track Learned Weights
    # ═══════════════════════════════════════════════════════════════════

    def _load_track_weights(self, track_code: str) -> dict[str, float] | None:
        """Load track-specific learned weights, or None if track has no data."""
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                "SELECT weight_name, weight_value FROM track_learned_weights "
                "WHERE track_code = ?",
                (track_code.upper(),),
            )
            rows = cursor.fetchall()
            conn.close()
            if rows:
                track_weights = {row[0]: row[1] for row in rows}
                logger.info(
                    f"📚 Loaded {len(rows)} track-specific weights for {track_code}"
                )
                return track_weights
        except Exception as e:
            logger.warning(f"Could not load track weights for {track_code}: {e}")
        return None

    def _save_track_weights(
        self,
        track_code: str,
        weights: dict[str, float],
        races_trained: int = 0,
    ):
        """Persist track-specific calibrated weights to database."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        for name, value in weights.items():
            if name in [
                "angles",
                "odds_drift_penalty",
                "smart_money_bonus",
                "a_group_drift_gate",
            ]:
                continue
            cursor.execute(
                """
                INSERT INTO track_learned_weights
                    (track_code, weight_name, weight_value, races_trained_on,
                     last_updated, confidence)
                VALUES (?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_code, weight_name)
                DO UPDATE SET weight_value = excluded.weight_value,
                              races_trained_on = excluded.races_trained_on,
                              last_updated = excluded.last_updated,
                              confidence = excluded.confidence
            """,
                (
                    track_code.upper(),
                    name,
                    round(value, 3),
                    races_trained,
                    now,
                    min(1.0, 0.3 + races_trained * 0.05),  # confidence grows with data
                ),
            )

        conn.commit()
        conn.close()
        logger.info(
            f"💾 Saved {len(weights)} track weights for {track_code} "
            f"({races_trained} races)"
        )

    def calibrate_track(self, track_code: str, num_races: int = 50) -> dict:
        """
        Calibrate weights for a SPECIFIC track using only races from that track.

        Uses the same gradient descent approach as global calibration but
        scoped to one track. Falls back to global weights as starting point
        when track has fewer than 3 races.
        """
        track_code = track_code.upper()
        logger.info(f"🏇 Track calibration: {track_code} (last {num_races} races)...")

        # Get ALL recent results, then filter by track
        all_races = self.get_recent_results(limit=200)
        track_races = [
            r for r in all_races if (r.get("track") or "").upper() == track_code
        ]

        if len(track_races) < 2:
            logger.info(
                f"⏳ {track_code}: only {len(track_races)} race(s), "
                f"need 2+ for track calibration"
            )
            return {
                "status": "insufficient_data",
                "track": track_code,
                "races_available": len(track_races),
                "min_required": 2,
            }

        # Start from existing track weights, or global weights
        existing_track = self._load_track_weights(track_code)
        if existing_track:
            track_weights = {**self.base_weights, **existing_track}
        else:
            track_weights = self.base_weights.copy()

        # Compute gradients from track-specific races
        gradients_list = []
        error_summary = []
        for race in track_races[:num_races]:
            gradients = self.calculate_prediction_error(race)
            if gradients:
                gradients_list.append(gradients)
                horses = sorted(race["horses"], key=lambda h: h["predicted_rank"])
                if horses:
                    error_summary.append(horses[0]["actual_finish"])

        if not gradients_list:
            return {
                "status": "no_gradients",
                "track": track_code,
                "races_available": len(track_races),
            }

        # Apply gradient descent with slightly higher learning rate for tracks
        # (since we have fewer samples, we want faster adaptation)
        old_track_weights = track_weights.copy()

        avg_gradients = {}
        for component in track_weights.keys():
            if component in [
                "angles",
                "odds_drift_penalty",
                "smart_money_bonus",
                "a_group_drift_gate",
            ]:
                continue
            grads = [g.get(component, 0.0) for g in gradients_list]
            avg_gradients[component] = np.mean(grads)

        for component, current_weight in track_weights.items():
            if component in [
                "angles",
                "odds_drift_penalty",
                "smart_money_bonus",
                "a_group_drift_gate",
            ]:
                continue
            gradient = avg_gradients.get(component, 0.0)
            # Regularize toward the GLOBAL weight instead of 2.0
            global_anchor = self.base_weights.get(component, 2.0)
            reg_term = self.regularization * (current_weight - global_anchor)
            new_weight = current_weight - self.learning_rate * (gradient + reg_term)
            new_weight = np.clip(new_weight, 0.5, 4.0)
            track_weights[component] = round(float(new_weight), 3)

        # Accuracy metrics
        winner_accuracy = sum(1 for e in error_summary if e == 1) / len(error_summary)
        top3_accuracy = sum(1 for e in error_summary if e <= 3) / len(error_summary)

        # Persist
        self._save_track_weights(track_code, track_weights, len(track_races))

        # Log to track calibration history
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO track_calibration_history
                (track_code, races_analyzed, winner_accuracy, top3_accuracy,
                 weights_json, improvements_json)
                VALUES (?, ?, ?, ?, ?, ?)
            """,
                (
                    track_code,
                    len(track_races),
                    winner_accuracy,
                    top3_accuracy,
                    json.dumps(track_weights),
                    json.dumps(
                        {
                            k: round(
                                track_weights.get(k, 0) - old_track_weights.get(k, 0), 3
                            )
                            for k in track_weights
                            if k
                            not in [
                                "angles",
                                "odds_drift_penalty",
                                "smart_money_bonus",
                                "a_group_drift_gate",
                            ]
                        }
                    ),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not save track calibration history: {e}")

        result = {
            "status": "success",
            "track": track_code,
            "races_analyzed": len(track_races),
            "winner_accuracy": winner_accuracy,
            "top3_accuracy": top3_accuracy,
            "old_weights": old_track_weights,
            "new_weights": track_weights,
            "weight_changes": {
                k: round(track_weights.get(k, 0) - old_track_weights.get(k, 0), 3)
                for k in track_weights
                if k
                not in [
                    "angles",
                    "odds_drift_penalty",
                    "smart_money_bonus",
                    "a_group_drift_gate",
                ]
            },
        }
        logger.info(
            f"✅ Track calibration [{track_code}]: "
            f"W={winner_accuracy:.0%} T3={top3_accuracy:.0%} "
            f"({len(track_races)} races)"
        )
        return result

    def get_track_weights_for_prediction(self, track_code: str) -> dict[str, float]:
        """
        Get the best available weights for a given track.

        Priority:
          1. Track-specific learned weights (blended with global)
          2. Falls back to global learned weights
        """
        track_code = track_code.upper()
        track_weights = self._load_track_weights(track_code)

        if track_weights:
            # Blend: start with global, overlay track-specific
            blended = self.base_weights.copy()
            blended.update(track_weights)
            logger.info(f"🎯 Using track-specific weights for {track_code}")
            return blended

        # No track data yet — use global
        return self.base_weights.copy()

    def get_all_track_calibrations(self) -> list[dict]:
        """
        Return a summary of all tracks with learned weights.
        Used for the Auto-Calibration Monitor UI.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute("""
                SELECT track_code, weight_name, weight_value, 
                       races_trained_on, last_updated, confidence
                FROM track_learned_weights
                ORDER BY track_code, weight_name
            """)
            rows = cursor.fetchall()
            conn.close()

            if not rows:
                return []

            # Group by track
            tracks: dict[str, dict] = {}
            for track_code, wname, wval, trained, updated, conf in rows:
                if track_code not in tracks:
                    tracks[track_code] = {
                        "track_code": track_code,
                        "weights": {},
                        "races_trained_on": trained,
                        "last_updated": updated,
                        "avg_confidence": 0.0,
                    }
                tracks[track_code]["weights"][wname] = round(wval, 3)
                tracks[track_code]["races_trained_on"] = max(
                    tracks[track_code]["races_trained_on"], trained
                )
                if updated and updated > (tracks[track_code]["last_updated"] or ""):
                    tracks[track_code]["last_updated"] = updated

            # Calculate avg confidence per track
            for tc in tracks:
                w = tracks[tc]["weights"]
                if w:
                    cursor_vals = [
                        r[5] for r in rows if r[0] == tc and r[5] is not None
                    ]
                    tracks[tc]["avg_confidence"] = (
                        round(sum(cursor_vals) / len(cursor_vals), 2)
                        if cursor_vals
                        else 0.5
                    )

            return list(tracks.values())

        except Exception as e:
            logger.warning(f"Could not load track calibrations: {e}")
            return []

    def _load_odds_drift_weights(self) -> dict[str, float]:
        """Load learned odds drift adjustments from historical patterns."""
        defaults = {
            "drift_out_2x_penalty": -3.0,
            "drift_out_50pct_penalty": -1.5,
            "drift_in_50pct_bonus": 2.5,
            "drift_in_2x_bonus": 4.0,
        }

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            # Calculate optimal drift penalties from historical data
            cursor.execute("""
                SELECT odds_direction, AVG(drift_magnitude), 
                       AVG(CASE WHEN actual_finish <= 3 THEN 1 ELSE 0 END) as hit_rate,
                       COUNT(*) as sample_size
                FROM odds_drift_learning
                GROUP BY odds_direction
                HAVING sample_size >= 5
            """)

            results = cursor.fetchall()
            conn.close()

            if results:
                for direction, avg_drift, hit_rate, sample_size in results:
                    if direction == "OUT" and hit_rate < 0.15:
                        defaults["drift_out_2x_penalty"] = -3.5
                    elif direction == "IN" and hit_rate > 0.40:
                        defaults["drift_in_50pct_bonus"] = 3.0

                logger.info(
                    f"📊 Loaded odds drift adjustments from {sum(r[3] for r in results)} samples"
                )

        except Exception as e:
            pass  # Table might not exist yet

        return defaults

    def record_odds_drift_outcome(
        self,
        ml_odds: float,
        live_odds: float,
        actual_finish: int,
        expected_finish: int,
        race_id: str = None,
    ):
        """Record an odds drift outcome for future learning."""
        if ml_odds <= 0 or live_odds <= 0:
            return

        drift_ratio = live_odds / ml_odds
        direction = "OUT" if drift_ratio > 1.0 else "IN"
        magnitude = drift_ratio if drift_ratio > 1.0 else (1.0 / drift_ratio)

        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()

            cursor.execute(
                """
                INSERT INTO odds_drift_learning 
                (odds_direction, drift_magnitude, actual_finish, expected_finish, race_id)
                VALUES (?, ?, ?, ?, ?)
            """,
                (direction, magnitude, actual_finish, expected_finish, race_id),
            )

            conn.commit()
            conn.close()

            logger.info(
                f"📝 Recorded odds drift: {direction} {magnitude:.2f}x, finished {actual_finish}"
            )
        except Exception as e:
            logger.warning(f"Could not record odds drift: {e}")

    def get_recent_results(self, limit: int = 50) -> list[dict]:
        """Fetch recent races with both predictions and actual results.

        Supports BOTH database schemas:
        - New schema: horses_analyzed + gold_high_iq with features_json
        - Old schema: gold_high_iq with individual columns (predicted_finish_position, etc.)
        """
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        rows = []
        schema_used = None

        # --- Strategy 1: New schema (horses_analyzed joined) ---
        try:
            cursor.execute(
                """
                SELECT
                    g.race_id,
                    r.race_type,
                    r.track_code,
                    r.purse,
                    r.field_size,
                    g.horse_name,
                    g.predicted_probability,
                    g.predicted_rank,
                    g.actual_finish_position,
                    g.features_json
                FROM gold_high_iq g
                LEFT JOIN races_analyzed r ON g.race_id = r.race_id
                WHERE g.actual_finish_position IS NOT NULL
                  AND g.predicted_rank IS NOT NULL
                ORDER BY g.result_entered_timestamp DESC
                LIMIT ?
            """,
                (limit * 12,),
            )
            rows = cursor.fetchall()
            if rows:
                schema_used = "new_with_features"
        except Exception:
            pass

        # --- Strategy 2: Old schema (individual columns) ---
        if not rows:
            try:
                cursor.execute("PRAGMA table_info(gold_high_iq)")
                col_names = {row[1] for row in cursor.fetchall()}

                if "predicted_finish_position" in col_names:
                    cursor.execute(
                        """
                        SELECT
                            race_id,
                            race_type,
                            track,
                            0 as purse,
                            field_size,
                            horse_name,
                            odds,
                            predicted_finish_position,
                            actual_finish_position,
                            class_rating,
                            last_speed_rating,
                            prime_power,
                            running_style,
                            post_position
                        FROM gold_high_iq
                        WHERE actual_finish_position IS NOT NULL
                        ORDER BY timestamp DESC
                        LIMIT ?
                    """,
                        (limit * 12,),
                    )
                    rows = cursor.fetchall()
                    if rows:
                        schema_used = "old_individual_columns"
            except Exception as e:
                logger.warning(f"Old schema query failed: {e}")

        conn.close()

        # Group by race_id
        races = {}
        for row in rows:
            race_id = row[0]
            if race_id not in races:
                races[race_id] = {
                    "race_id": race_id,
                    "race_type": row[1] or "CLM",
                    "track": row[2] or "UNK",
                    "purse": row[3] or 0,
                    "field_size": row[4] or 8,
                    "horses": [],
                }

            if schema_used == "old_individual_columns":
                # Old schema: build features from individual columns
                # row[6]=odds, row[7]=predicted_finish_position, row[8]=actual_finish
                # row[9]=class_rating, row[10]=last_speed_rating, row[11]=prime_power
                # row[12]=running_style, row[13]=post_position
                odds_val = row[6] or 10.0
                # Convert odds to approximate probability (1/odds normalized)
                predicted_prob = 1.0 / max(odds_val + 1, 1.0)
                predicted_rank = row[7] or 99
                actual_finish = row[8]

                # Build component ratings from available data
                class_rating = row[9] or 0
                speed_rating = row[10] or 0
                prime = row[11] or 0

                # Estimate component scores from available data
                # Normalize to typical 0-1 ranges used by the engine
                c_class = class_rating / 120.0 if class_rating else 0
                c_speed = speed_rating / 100.0 if speed_rating else 0
                c_form = 0.5  # No form data in old schema, use neutral
                c_pace = 0.5  # No pace data in old schema, use neutral
                c_style = 0.5  # No style data in old schema, use neutral
                c_post = 0.5  # No post rating in old schema, use neutral

                races[race_id]["horses"].append(
                    {
                        "name": row[5],
                        "predicted_prob": predicted_prob,
                        "predicted_rank": predicted_rank,
                        "actual_finish": actual_finish,
                        "c_class": c_class,
                        "c_speed": c_speed,
                        "c_form": c_form,
                        "c_pace": c_pace,
                        "c_style": c_style,
                        "c_post": c_post,
                    }
                )
            else:
                # New schema: parse features from JSON
                features = {}
                if row[9]:
                    try:
                        features = json.loads(row[9])
                    except Exception:
                        pass

                races[race_id]["horses"].append(
                    {
                        "name": row[5],
                        "predicted_prob": row[6] or 0.1,
                        "predicted_rank": row[7] or 99,
                        "actual_finish": row[8],
                        "c_class": features.get("rating_class", 0),
                        "c_speed": features.get("rating_speed", 0),
                        "c_form": features.get("rating_form", 0),
                        "c_pace": features.get("rating_pace", 0),
                        "c_style": features.get("rating_style", 0),
                        "c_post": features.get("rating_post", 0),
                    }
                )

        result = list(races.values())[:limit]
        if result:
            logger.info(f"📊 Loaded {len(result)} races via {schema_used} schema")
        return result

    def calculate_prediction_error(self, race: dict) -> dict[str, float]:
        """Calculate error gradient for each component weight."""
        horses = sorted(race["horses"], key=lambda h: h["actual_finish"])
        if not horses:
            return {}

        winner = horses[0]
        winner_predicted_rank = winner["predicted_rank"]
        rank_error = winner_predicted_rank - 1

        gradients = {}
        winner_components = {
            "class": winner.get("c_class", 0),
            "speed": winner.get("c_speed", 0),
            "form": winner.get("c_form", 0),
            "pace": winner.get("c_pace", 0),
            "style": winner.get("c_style", 0),
            "post": winner.get("c_post", 0),
        }

        for component, rating in winner_components.items():
            if rating > 0:
                gradients[component] = -rank_error * rating * 0.1
            elif rating < 0:
                gradients[component] = rank_error * abs(rating) * 0.05
            else:
                gradients[component] = 0.0

        return gradients

    def apply_weight_updates(
        self, gradients_list: list[dict[str, float]]
    ) -> dict[str, float]:
        """Apply accumulated gradients to base weights using gradient descent."""
        if not gradients_list:
            return self.base_weights

        avg_gradients = {}
        for component in self.base_weights.keys():
            if component in [
                "angles",
                "odds_drift_penalty",
                "smart_money_bonus",
                "a_group_drift_gate",
            ]:
                continue
            gradients = [g.get(component, 0.0) for g in gradients_list]
            avg_gradients[component] = np.mean(gradients)

        updated_weights = {}
        for component, current_weight in self.base_weights.items():
            if component in [
                "angles",
                "odds_drift_penalty",
                "smart_money_bonus",
                "a_group_drift_gate",
            ]:
                updated_weights[component] = current_weight
                continue

            gradient = avg_gradients.get(component, 0.0)
            reg_term = self.regularization * (current_weight - 2.0)
            new_weight = current_weight - self.learning_rate * (gradient + reg_term)
            new_weight = np.clip(new_weight, 0.5, 4.0)
            updated_weights[component] = round(new_weight, 2)

        return updated_weights

    def calibrate_from_recent_results(self, num_races: int = 20) -> dict:
        """Main calibration function - analyzes recent races and updates weights."""
        logger.info(f"🔄 Starting auto-calibration on {num_races} recent races...")

        races = self.get_recent_results(limit=num_races)

        if len(races) == 0:
            logger.warning("⚠️ No races with results found. Skipping calibration.")
            return {
                "status": "skipped",
                "reason": "No races with results",
                "weights": self.base_weights,
            }

        logger.info(f"📊 Analyzing {len(races)} races...")

        gradients_list = []
        error_summary = []

        for race in races:
            gradients = self.calculate_prediction_error(race)
            if gradients:
                gradients_list.append(gradients)
                horses = sorted(race["horses"], key=lambda h: h["predicted_rank"])
                if horses:
                    winner_predicted = horses[0]["actual_finish"]
                    error_summary.append(winner_predicted)

        if not error_summary:
            return {
                "status": "skipped",
                "reason": "No valid races for calibration",
                "weights": self.base_weights,
            }

        old_weights = self.base_weights.copy()
        self.base_weights = self.apply_weight_updates(gradients_list)

        # Calculate accuracy metrics
        winner_accuracy = sum(1 for e in error_summary if e == 1) / len(error_summary)
        top3_accuracy = sum(1 for e in error_summary if e <= 3) / len(error_summary)

        # PERSIST LEARNED WEIGHTS TO DATABASE
        self._save_learned_weights(races_trained=len(races))

        # Log to calibration history table
        try:
            conn = sqlite3.connect(self.db_path)
            cursor = conn.cursor()
            cursor.execute(
                """
                INSERT INTO calibration_history 
                (races_analyzed, winner_accuracy, top3_accuracy, weights_json, improvements_json)
                VALUES (?, ?, ?, ?, ?)
            """,
                (
                    len(races),
                    winner_accuracy,
                    top3_accuracy,
                    json.dumps(self.base_weights),
                    json.dumps(
                        {
                            k: round(self.base_weights[k] - old_weights.get(k, 0), 3)
                            for k in self.base_weights
                        }
                    ),
                ),
            )
            conn.commit()
            conn.close()
        except Exception as e:
            logger.warning(f"Could not save calibration history: {e}")

        calibration_event = {
            "status": "success",
            "timestamp": datetime.now().isoformat(),
            "races_analyzed": len(races),
            "winner_accuracy": winner_accuracy,
            "top3_accuracy": top3_accuracy,
            "old_weights": old_weights,
            "new_weights": self.base_weights,
            "weight_changes": {
                k: round(self.base_weights[k] - old_weights.get(k, 0), 3)
                for k in self.base_weights
                if k
                not in [
                    "angles",
                    "odds_drift_penalty",
                    "smart_money_bonus",
                    "a_group_drift_gate",
                ]
            },
        }

        self.calibration_log.append(calibration_event)

        logger.info("✅ Calibration complete!")
        logger.info(f"📈 Winner Accuracy: {winner_accuracy:.1%}")
        logger.info(f"📈 Top-3 Accuracy: {top3_accuracy:.1%}")
        logger.info("💾 Weights persisted to database!")

        return calibration_event

    def get_learned_weights_for_prediction(self) -> dict[str, float]:
        """
        Get the current learned weights to apply in live predictions.

        Returns:
            Dict of weight names to values
        """
        return self.base_weights.copy()

    def export_updated_weights(self) -> str:
        """Generate Python code to update unified_rating_engine.py."""
        code = "# AUTO-CALIBRATED WEIGHTS\n"
        code += f"# Last Updated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        code += "# Generated by auto_calibration_engine.py\n\n"
        code += "WEIGHTS = {\n"

        for component, weight in self.base_weights.items():
            comment = (
                f"  # Auto-tuned from {len(self.calibration_log)} calibration events"
            )
            code += f"    '{component}': {weight},{comment if len(self.calibration_log) > 0 else ''}\n"

        code += "}\n"

        return code

    def save_calibration_history(self, filepath: str = "calibration_history.json"):
        """Save calibration log to JSON file."""
        with open(filepath, "w") as f:
            json.dump(self.calibration_log, f, indent=2)
        logger.info(f"💾 Calibration history saved to {filepath}")


def auto_calibrate_on_result_submission(
    db_path: str = "gold_high_iq.db",
    track_code: str = "",
) -> dict:
    """
    Trigger function called after each result submission.

    Performs BOTH global calibration AND track-specific calibration.

    Usage: Call this from app.py after gold_db.save_race_results()

    Args:
        db_path: Path to the gold database.
        track_code: Track code for track-specific calibration (e.g. 'TAM', 'GP').

    Returns:
        Calibration result dict with accuracy metrics and track info.
    """
    calibrator = AutoCalibrationEngine(db_path)

    # 1) Global calibration (as before)
    result = calibrator.calibrate_from_recent_results(num_races=20)

    if result.get("status") == "skipped":
        return result

    # Save updated code snippet for manual review
    new_code = calibrator.export_updated_weights()
    with open("updated_weights.py", "w") as f:
        f.write(new_code)

    logger.info("📝 Updated weights saved to updated_weights.py")
    logger.info("✅ Weights also persisted to database for live use!")

    # 2) Track-specific calibration
    if track_code:
        try:
            track_result = calibrator.calibrate_track(track_code, num_races=50)
            result["track_calibration"] = track_result
            logger.info(
                f"🏇 Track calibration for {track_code}: "
                f"{track_result.get('status', 'unknown')}"
            )
        except Exception as te:
            logger.warning(f"Track calibration failed for {track_code}: {te}")
            result["track_calibration"] = {"status": "error", "error": str(te)}

    return result


def get_live_learned_weights(
    db_path: str = "gold_high_iq.db",
    track_code: str = "",
) -> dict[str, float]:
    """
    Get learned weights for use in live predictions.

    If track_code is provided and the track has enough calibration data,
    returns track-specific weights blended with global weights.
    Otherwise returns global learned weights.

    Args:
        db_path: Path to the gold database.
        track_code: Optional track code for track-specific weights.

    Returns:
        Dict of learned weights (track-specific if available, else global).
    """
    calibrator = AutoCalibrationEngine(db_path)

    if track_code:
        track_weights = calibrator.get_track_weights_for_prediction(track_code)
        if track_weights:
            return track_weights

    return calibrator.get_learned_weights_for_prediction()


def get_all_track_calibrations_summary(
    db_path: str = "gold_high_iq.db",
) -> list[dict]:
    """
    Get summary of ALL tracks with learned weights.
    Used for the Auto-Calibration Monitor UI.
    """
    calibrator = AutoCalibrationEngine(db_path)
    return calibrator.get_all_track_calibrations()


if __name__ == "__main__":
    # Test calibration
    result = auto_calibrate_on_result_submission()
    print(json.dumps(result, indent=2, default=str))
