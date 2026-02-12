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

import json
import logging
import sqlite3
from datetime import datetime

import pandas as pd

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

        schema_path = os.path.join(
            os.path.dirname(os.path.abspath(__file__)), "gold_database_schema.sql"
        )
        try:
            with open(schema_path, encoding="utf-8") as f:
                schema_sql = f.read()
        except FileNotFoundError:
            logger.warning("gold_database_schema.sql not found – using inline fallback")
            schema_sql = ""

        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # ---------- execute the SQL file, statement-by-statement ----------
        if schema_sql:
            # Split on ';' and run each independently
            for stmt in schema_sql.split(";"):
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

        # ---------- track pattern learning tables -------------------------
        track_pattern_tables = {
            "track_pattern_winners": """
                CREATE TABLE IF NOT EXISTS track_pattern_winners (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    race_id TEXT NOT NULL,
                    track_code TEXT NOT NULL,
                    surface TEXT NOT NULL,
                    distance TEXT NOT NULL,
                    distance_furlongs REAL,
                    race_type TEXT,
                    purse_amount REAL DEFAULT 0,
                    post_position INTEGER,
                    running_style TEXT,
                    quirin_speed_pts INTEGER,
                    class_rating REAL,
                    best_beyer INTEGER,
                    last_beyer INTEGER,
                    avg_beyer_3 REAL,
                    days_since_last INTEGER,
                    prime_power REAL,
                    morning_line_odds REAL,
                    predicted_rank INTEGER,
                    actual_finish INTEGER NOT NULL,
                    horse_name TEXT,
                    field_size INTEGER,
                    workout_pattern TEXT,
                    form_decay_score REAL,
                    pace_esp_score REAL,
                    jockey TEXT,
                    trainer TEXT,
                    race_date TEXT,
                    timestamp TEXT DEFAULT CURRENT_TIMESTAMP,
                    FOREIGN KEY (race_id) REFERENCES races_analyzed(race_id)
                )""",
            "track_pattern_stats": """
                CREATE TABLE IF NOT EXISTS track_pattern_stats (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    track_code TEXT NOT NULL,
                    surface TEXT NOT NULL,
                    distance_category TEXT NOT NULL,
                    stat_key TEXT NOT NULL,
                    stat_value TEXT NOT NULL,
                    sample_size INTEGER DEFAULT 0,
                    last_updated TEXT,
                    UNIQUE(track_code, surface, distance_category, stat_key)
                )""",
        }
        for tbl_name, ddl in track_pattern_tables.items():
            try:
                cursor.execute(ddl)
            except Exception as tbl_err:
                logger.debug(f"Track pattern table {tbl_name} skipped: {tbl_err}")

        # ---------- migrate existing track_pattern_winners if needed -----
        # Add columns that may not exist in older databases
        for col_def in [
            ("purse_amount", "REAL DEFAULT 0"),
            ("quirin_speed_pts", "INTEGER"),
        ]:
            try:
                cursor.execute(
                    f"ALTER TABLE track_pattern_winners ADD COLUMN {col_def[0]} {col_def[1]}"
                )
            except Exception:
                pass  # Column already exists

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
        race_metadata: dict,
        horses_data: list[dict],
        pp_text_raw: str,
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
            cursor.execute(
                """
                SELECT COUNT(*) FROM gold_high_iq 
                WHERE race_id = ?
            """,
                (race_id,),
            )
            has_results = cursor.fetchone()[0] > 0

            if has_results:
                logger.warning(
                    f"⚠️ Race {race_id} already has results. Updating predictions but preserving results."
                )

            # 1. Insert race record (INSERT OR REPLACE updates if exists)
            cursor.execute(
                """
                INSERT OR REPLACE INTO races_analyzed 
                (race_id, track_code, race_date, race_number, race_type, 
                 surface, distance, track_condition, purse, field_size, 
                 pp_text_raw, analyzed_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    race_id,
                    race_metadata.get("track", "UNK"),
                    race_metadata.get("date", datetime.now().strftime("%Y-%m-%d")),
                    race_metadata.get("race_num", 1),
                    race_metadata.get("race_type", "UNK"),
                    race_metadata.get("surface", "Dirt"),
                    race_metadata.get("distance", "6F"),
                    race_metadata.get("condition", "Fast"),
                    race_metadata.get("purse", 0.0),
                    len(horses_data),
                    pp_text_raw,
                    datetime.now().isoformat(),
                ),
            )

            # 2. Insert all horses
            for horse in horses_data:
                horse_id = f"{race_id}_{horse['program_number']}"

                cursor.execute(
                    """
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
                """,
                    (
                        horse_id,
                        race_id,
                        horse.get("program_number", 0),
                        horse.get("horse_name", "Unknown"),
                        horse.get("post_position", 0),
                        horse.get("morning_line_odds", 99.0),
                        horse.get("jockey", ""),
                        horse.get("trainer", ""),
                        horse.get("owner", ""),
                        horse.get("running_style", "P"),
                        horse.get("prime_power", 0.0),
                        horse.get("best_beyer", 0),
                        horse.get("last_beyer", 0),
                        horse.get("avg_beyer_3", 0.0),
                        horse.get("e1_pace", 0.0),
                        horse.get("e2_pace", 0.0),
                        horse.get("late_pace", 0.0),
                        horse.get("days_since_last", 0),
                        horse.get("starts_lifetime", 0),
                        horse.get("wins_lifetime", 0),
                        horse.get("win_pct", 0.0),
                        horse.get("earnings_lifetime", 0.0),
                        horse.get("class_rating", 0.0),
                        horse.get("angle_early_speed", 0.0),
                        horse.get("angle_class", 0.0),
                        horse.get("angle_recency", 0.0),
                        horse.get("angle_work_pattern", 0.0),
                        horse.get("angle_connections", 0.0),
                        horse.get("angle_pedigree", 0.0),
                        horse.get("angle_runstyle_bias", 0.0),
                        horse.get("angle_post", 0.0),
                        horse.get("rating_class", 0.0),
                        horse.get("rating_form", 0.0),
                        horse.get("rating_speed", 0.0),
                        horse.get("rating_pace", 0.0),
                        horse.get("rating_style", 0.0),
                        horse.get("rating_post", 0.0),
                        horse.get("rating_angles_total", 0.0),
                        horse.get("rating_tier2_bonus", 0.0),
                        horse.get("rating_final", 0.0),
                        horse.get("rating_confidence", 0.5),
                        horse.get("form_decay_score", 0.0),
                        horse.get("pace_esp_score", 0.0),
                        horse.get("mud_adjustment", 0.0),
                        horse.get("predicted_probability", 0.0),
                        horse.get("predicted_rank", 99),
                        horse.get("fair_odds", 99.0),
                    ),
                )

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

    def get_pending_races(self, limit: int = 20) -> list[tuple]:
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
            cursor.execute(
                """
                SELECT race_id, track_code, race_date, race_number, field_size
                FROM v_pending_races
                LIMIT ?
            """,
                (limit,),
            )
            pending = cursor.fetchall()
        except Exception:
            pass

        # Strategy 2: direct query against races_analyzed (skip view)
        if not pending:
            try:
                cursor.execute(
                    """
                    SELECT ra.race_id, ra.track_code, ra.race_date,
                           ra.race_number, ra.field_size
                    FROM races_analyzed ra
                    WHERE NOT EXISTS (
                        SELECT 1 FROM race_results_summary rs
                        WHERE rs.race_id = ra.race_id
                    )
                    ORDER BY ra.race_date DESC, ra.race_number
                    LIMIT ?
                """,
                    (limit,),
                )
                pending = cursor.fetchall()
            except Exception:
                pass

        conn.close()
        return pending

    def get_horses_for_race(self, race_id: str) -> list[dict]:
        """
        Get all horses for a specific race.

        Returns:
            List of dicts with horse data
        """
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row  # Return rows as dicts
        cursor = conn.cursor()

        cursor.execute(
            """
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
        """,
            (race_id,),
        )

        horses = [dict(row) for row in cursor.fetchall()]
        conn.close()

        return horses

    def submit_race_results(
        self,
        race_id: str,
        finish_order_programs: list[int],
        horses_ui: list[dict] | None = None,
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
            cursor.execute(
                """
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
            """,
                (race_id,),
            )

            all_horses_rows = cursor.fetchall()
            horse_dict = {row[1]: row for row in all_horses_rows}

            # ------ fallback: build horse_dict from UI data -------
            ui_dict: dict[int, dict] = {}
            if horses_ui:
                for h in horses_ui:
                    pn = int(h.get("program_number", h.get("post_position", 0)))
                    ui_dict[pn] = h

            # Choose a name-lookup helper that works with either source
            def _horse_name(prog: int) -> str:
                if prog in horse_dict:
                    return horse_dict[prog][2]
                if prog in ui_dict:
                    return ui_dict[prog].get("horse_name", f"Horse #{prog}")
                return f"Horse #{prog}"

            # ------ detect which gold_high_iq schema we have ------
            cursor.execute("PRAGMA table_info(gold_high_iq)")
            gold_cols = {row[1] for row in cursor.fetchall()}
            uses_new_schema = (
                "result_id" in gold_cols
            )  # schema from gold_database_schema.sql
            uses_old_schema = "track" in gold_cols and "id" in gold_cols

            # ------ insert per-horse results ----------------------
            for actual_position, program_num in enumerate(finish_order_programs[:4], 1):
                if uses_new_schema and program_num in horse_dict:
                    # New schema path (result_id TEXT PRIMARY KEY)
                    horse_row = horse_dict[program_num]
                    horse_id = horse_row[0]
                    predicted_rank = horse_row[6] or 99
                    prediction_error = abs(predicted_rank - actual_position)
                    was_top4_correct = predicted_rank <= 4 and actual_position <= 4

                    features = {
                        "rating_class": horse_row[7],
                        "rating_form": horse_row[8],
                        "rating_speed": horse_row[9],
                        "rating_pace": horse_row[10],
                        "rating_style": horse_row[11],
                        "rating_post": horse_row[12],
                        "rating_angles_total": horse_row[13],
                        "rating_confidence": horse_row[14],
                        "form_decay_score": horse_row[15],
                        "pace_esp_score": horse_row[16],
                        "angle_early_speed": horse_row[17],
                        "angle_class": horse_row[18],
                        "angle_recency": horse_row[19],
                        "prime_power": horse_row[20],
                        "best_beyer": horse_row[21],
                        "running_style": horse_row[22],
                    }
                    result_id = f"{race_id}_{program_num}"
                    cursor.execute(
                        """
                        INSERT OR REPLACE INTO gold_high_iq
                        (result_id, race_id, horse_id, actual_finish_position,
                         program_number, horse_name, post_position, rating_final,
                         predicted_probability, predicted_rank, features_json,
                         prediction_error, was_top5_correct,
                         result_entered_timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            result_id,
                            race_id,
                            horse_id,
                            actual_position,
                            program_num,
                            horse_row[2],
                            horse_row[3],
                            horse_row[4],
                            horse_row[5],
                            predicted_rank,
                            json.dumps(features),
                            prediction_error,
                            was_top4_correct,
                            datetime.now().isoformat(),
                        ),
                    )

                elif uses_old_schema:
                    # Old schema path (id INTEGER PRIMARY KEY AUTOINCREMENT)
                    h = ui_dict.get(program_num, {}) if ui_dict else {}
                    db_h = horse_dict.get(program_num)
                    name = _horse_name(program_num)
                    predicted_rank = (
                        db_h[6] if db_h else int(h.get("predicted_rank", 99))
                    )
                    prediction_error = abs(predicted_rank - actual_position)

                    # Parse race_id components (TRACK_YYYYMMDD_R#)
                    parts = race_id.rsplit("_", 2)
                    track_code = parts[0] if len(parts) >= 3 else "UNK"
                    r_date = parts[1] if len(parts) >= 3 else ""
                    r_num = parts[2].replace("R", "") if len(parts) >= 3 else "0"

                    cursor.execute(
                        """
                        INSERT INTO gold_high_iq
                        (race_id, track, race_num, race_date,
                         horse_name, program_number,
                         actual_finish_position, predicted_finish_position,
                         prediction_error, post_position, field_size,
                         prime_power, running_style, timestamp)
                        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """,
                        (
                            race_id,
                            track_code,
                            int(r_num) if r_num.isdigit() else 0,
                            r_date,
                            name,
                            program_num,
                            actual_position,
                            predicted_rank,
                            prediction_error,
                            int(h.get("post_position", 0)),
                            int(h.get("field_size", len(ui_dict))),
                            float(h.get("prime_power", 0)),
                            h.get("running_style", ""),
                            datetime.now().isoformat(),
                        ),
                    )

                else:
                    logger.warning(
                        f"Program #{program_num} not found – "
                        f"no matching schema or horse data"
                    )

            # ------ race_results_summary row ----------------------
            # Compute basic accuracy
            predicted_top4_programs: list[int] = []
            if horse_dict:
                predicted_top4 = sorted(all_horses_rows, key=lambda x: x[6] or 99)[:4]
                predicted_top4_programs = [h[1] for h in predicted_top4]
            elif ui_dict:
                # Sort UI horses by predicted_rank
                sorted_ui = sorted(
                    ui_dict.values(), key=lambda h: int(h.get("predicted_rank", 99))
                )[:4]
                predicted_top4_programs = [
                    int(h.get("program_number", 0)) for h in sorted_ui
                ]

            top1_correct = (
                predicted_top4_programs[0] == finish_order_programs[0]
                if predicted_top4_programs
                else False
            )
            top3_hit = (
                len(set(predicted_top4_programs[:3]) & set(finish_order_programs[:3]))
                if predicted_top4_programs
                else 0
            )
            top4_hit = (
                len(set(predicted_top4_programs[:4]) & set(finish_order_programs[:4]))
                if predicted_top4_programs
                else 0
            )

            cursor.execute(
                """
                INSERT OR REPLACE INTO race_results_summary
                (race_id, winner_name, second_name, third_name,
                 fourth_name, fifth_name,
                 top1_predicted_correctly, top3_predicted_correctly,
                 top5_predicted_correctly, results_complete_timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
                (
                    race_id,
                    _horse_name(finish_order_programs[0]),
                    _horse_name(finish_order_programs[1])
                    if len(finish_order_programs) > 1
                    else "N/A",
                    _horse_name(finish_order_programs[2])
                    if len(finish_order_programs) > 2
                    else "N/A",
                    _horse_name(finish_order_programs[3])
                    if len(finish_order_programs) > 3
                    else "N/A",
                    "N/A",
                    top1_correct,
                    top3_hit,
                    top4_hit,
                    datetime.now().isoformat(),
                ),
            )

            conn.commit()
            conn.close()

            logger.info(
                f"✅ Results submitted for {race_id} | "
                f"Winner: {_horse_name(finish_order_programs[0])}"
            )

            # TRACK PATTERN LEARNING: Store winner characteristics for pattern analysis
            try:
                self.store_winner_patterns(race_id, finish_order_programs, horses_ui)
            except Exception as pat_err:
                logger.warning(f"⚠️ Pattern storage failed (non-fatal): {pat_err}")

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
        self, min_races: int = 50
    ) -> tuple[pd.DataFrame, pd.DataFrame] | None:
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
            logger.warning(
                f"⚠️ Only {num_races} completed races. Need {min_races} minimum."
            )
            conn.close()
            return None

        # Load all training data
        df = pd.read_sql_query(
            """
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
        """,
            conn,
        )

        conn.close()

        # Parse features JSON
        features_list = []
        for _, row in df.iterrows():
            features = json.loads(row["features_json"])
            features["race_id"] = row["race_id"]
            features["horse_id"] = row["horse_id"]
            features["actual_finish"] = row["actual_finish_position"]
            features_list.append(features)

        features_df = pd.DataFrame(features_list)
        labels_df = df[["race_id", "horse_id", "actual_finish_position"]]

        logger.info(
            f"✅ Loaded {len(features_df)} horses from {num_races} races for training"
        )

        return features_df, labels_df

    def get_accuracy_stats(self) -> dict:
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
            "total_races": stats[0] or 0,
            "winner_accuracy": stats[1] or 0.0,
            "top3_accuracy": stats[2] or 0.0,
            "top5_accuracy": stats[3] or 0.0,
        }

    def log_retraining(self, metrics: dict, model_path: str) -> int:
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

        cursor.execute(
            """
            INSERT INTO retraining_history
            (retrain_timestamp, total_races_used, total_horses_used,
             train_split_pct, val_split_pct, val_winner_accuracy,
             val_top3_accuracy, val_top5_accuracy, val_loss,
             model_checkpoint_path, epochs_trained, learning_rate,
             batch_size, training_duration_seconds)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
            (
                datetime.now().isoformat(),
                metrics.get("total_races", 0),
                metrics.get("total_horses", 0),
                metrics.get("train_split_pct", 0.8),
                metrics.get("val_split_pct", 0.2),
                metrics.get("val_winner_accuracy", 0.0),
                metrics.get("val_top3_accuracy", 0.0),
                metrics.get("val_top5_accuracy", 0.0),
                metrics.get("val_loss", 0.0),
                model_path,
                metrics.get("epochs", 0),
                metrics.get("learning_rate", 0.001),
                metrics.get("batch_size", 32),
                metrics.get("training_duration", 0.0),
            ),
        )

        retrain_id = cursor.lastrowid

        conn.commit()
        conn.close()

        logger.info(f"✅ Logged retraining session #{retrain_id}")
        return retrain_id

    # ================================================================
    # TRACK PATTERN LEARNING SYSTEM
    # ================================================================

    @staticmethod
    def _distance_to_furlongs(distance_str: str) -> float:
        """Convert a distance string like '6F', '1 1/16M', '1M' to furlongs."""
        if not distance_str:
            return 0.0
        d = distance_str.strip().upper()
        try:
            if "F" in d:
                return float(d.replace("F", "").strip())
            if "M" in d:
                d = d.replace("M", "").strip()
                if "/" in d:
                    parts = d.split()
                    whole = float(parts[0]) if len(parts) > 1 else 0.0
                    frac_parts = (parts[1] if len(parts) > 1 else parts[0]).split("/")
                    frac = float(frac_parts[0]) / float(frac_parts[1])
                    return (whole + frac) * 8.0
                return float(d if d else 1) * 8.0
        except (ValueError, IndexError, ZeroDivisionError):
            pass
        return 0.0

    @staticmethod
    def _categorize_distance(furlongs: float) -> str:
        """Categorize distance into sprint/route/marathon."""
        if furlongs <= 0:
            return "unknown"
        if furlongs <= 6.5:
            return "sprint"
        if furlongs <= 9.0:
            return "route"
        return "marathon"

    def store_winner_patterns(
        self,
        race_id: str,
        finish_order_programs: list[int],
        horses_ui: list[dict] | None = None,
    ) -> bool:
        """
        After a race result is submitted, store detailed winner/placer
        characteristics into track_pattern_winners for pattern learning.

        Pulls data from races_analyzed + horses_analyzed tables, falling back
        to horses_ui when the DB rows are incomplete.
        """
        try:
            conn = sqlite3.connect(self.db_path, timeout=10)
            cursor = conn.cursor()

            # Get race metadata
            cursor.execute(
                "SELECT track_code, surface, distance, race_type, race_date, "
                "field_size, purse "
                "FROM races_analyzed WHERE race_id = ?",
                (race_id,),
            )
            race_row = cursor.fetchone()
            if not race_row:
                logger.warning(
                    f"⚠️ No races_analyzed row for {race_id}, skipping pattern store"
                )
                conn.close()
                return False

            (
                track_code,
                surface,
                distance,
                race_type,
                race_date,
                field_size,
                purse_amount,
            ) = race_row
            furlongs = self._distance_to_furlongs(distance)

            # Load horse data from horses_analyzed
            cursor.execute(
                "SELECT program_number, horse_name, post_position, running_style, "
                "class_rating, best_beyer, last_beyer, avg_beyer_3, "
                "days_since_last, prime_power, morning_line_odds, "
                "predicted_rank, angle_work_pattern, form_decay_score, "
                "pace_esp_score, jockey, trainer "
                "FROM horses_analyzed WHERE race_id = ?",
                (race_id,),
            )
            horse_rows = {row[0]: row for row in cursor.fetchall()}

            # Build UI fallback dict
            ui_dict: dict[int, dict] = {}
            if horses_ui:
                for h in horses_ui:
                    pn = int(h.get("program_number", h.get("post_position", 0)))
                    ui_dict[pn] = h

            # Insert top-4 finishers into track_pattern_winners
            for actual_pos, prog in enumerate(finish_order_programs[:4], 1):
                db_h = horse_rows.get(prog)
                ui_h = ui_dict.get(prog, {})

                def _val(db_idx, ui_key, default=None):
                    if db_h and db_h[db_idx] is not None:
                        return db_h[db_idx]
                    return ui_h.get(ui_key, default)

                # Determine workout_pattern from angle_work_pattern score
                work_score = _val(12, "angle_work_pattern", 0.0)
                if work_score and float(work_score) >= 0.7:
                    workout_pattern = "Sharp"
                elif work_score and float(work_score) >= 0.3:
                    workout_pattern = "Steady"
                else:
                    workout_pattern = "Sparse"

                cursor.execute(
                    """
                    INSERT INTO track_pattern_winners
                    (race_id, track_code, surface, distance, distance_furlongs,
                     race_type, purse_amount, post_position, running_style,
                     quirin_speed_pts, class_rating,
                     best_beyer, last_beyer, avg_beyer_3, days_since_last,
                     prime_power, morning_line_odds, predicted_rank,
                     actual_finish, horse_name, field_size, workout_pattern,
                     form_decay_score, pace_esp_score, jockey, trainer,
                     race_date, timestamp)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?,
                            ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                    (
                        race_id,
                        track_code,
                        surface or "Dirt",
                        distance or "6F",
                        furlongs,
                        race_type or "UNK",
                        float(purse_amount or 0),
                        int(_val(2, "post_position", 0) or 0),
                        _val(3, "running_style", "P"),
                        int(ui_h.get("quirin_speed_pts", 0) or 0),
                        float(_val(4, "class_rating", 0) or 0),
                        int(_val(5, "best_beyer", 0) or 0),
                        int(_val(6, "last_beyer", 0) or 0),
                        float(_val(7, "avg_beyer_3", 0) or 0),
                        int(_val(8, "days_since_last", 0) or 0),
                        float(_val(9, "prime_power", 0) or 0),
                        float(_val(10, "morning_line_odds", 99) or 99),
                        int(_val(11, "predicted_rank", 99) or 99),
                        actual_pos,
                        _val(1, "horse_name", f"Horse #{prog}"),
                        field_size or len(finish_order_programs),
                        workout_pattern,
                        float(_val(13, "form_decay_score", 0) or 0),
                        float(_val(14, "pace_esp_score", 0) or 0),
                        _val(15, "jockey", ""),
                        _val(16, "trainer", ""),
                        race_date or "",
                        datetime.now().isoformat(),
                    ),
                )

            conn.commit()
            conn.close()
            logger.info(
                f"✅ Stored winner patterns for {race_id} ({track_code} {surface} {distance})"
            )

            # Rebuild aggregated stats for this track/surface/distance combo
            self._rebuild_track_pattern_stats(track_code, surface, distance)
            return True

        except Exception as e:
            logger.error(f"❌ Error storing winner patterns for {race_id}: {e}")
            import traceback

            traceback.print_exc()
            try:
                conn.rollback()
                conn.close()
            except Exception:
                pass
            return False

    def _rebuild_track_pattern_stats(
        self,
        track_code: str,
        surface: str,
        distance: str,
    ) -> None:
        """
        Rebuild aggregated pattern statistics for a given
        track / surface / distance-category combination.

        Updates the track_pattern_stats table with:
          - Post position win rates
          - Running style win rates
          - Average winning speed figures
          - Average winning class rating
          - Average winning days-since-last (freshness)
          - Workout pattern distribution for winners
          - Avg field size
          - Jockey/trainer win leaders
        """
        furlongs = self._distance_to_furlongs(distance)
        dist_cat = self._categorize_distance(furlongs)

        conn = sqlite3.connect(self.db_path, timeout=10)
        cursor = conn.cursor()
        now = datetime.now().isoformat()

        # Helper to upsert a stat
        def _upsert(key: str, value: str, sample: int):
            cursor.execute(
                """
                INSERT INTO track_pattern_stats
                (track_code, surface, distance_category, stat_key, stat_value, sample_size, last_updated)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                ON CONFLICT(track_code, surface, distance_category, stat_key)
                DO UPDATE SET stat_value = excluded.stat_value,
                              sample_size = excluded.sample_size,
                              last_updated = excluded.last_updated
            """,
                (track_code, surface, dist_cat, key, value, sample, now),
            )

        # --- All rows for this track/surface/distance-category ---
        cursor.execute(
            """
            SELECT post_position, running_style, class_rating,
                   best_beyer, last_beyer, avg_beyer_3,
                   days_since_last, prime_power, morning_line_odds,
                   actual_finish, field_size, workout_pattern,
                   form_decay_score, pace_esp_score, jockey, trainer
            FROM track_pattern_winners
            WHERE track_code = ? AND surface = ?
              AND distance_furlongs BETWEEN ? AND ?
        """,
            (
                track_code,
                surface,
                max(0, furlongs - 1.0),
                furlongs + 1.0,
            ),
        )
        rows = cursor.fetchall()

        if not rows:
            conn.close()
            return

        total = len(rows)
        winners = [r for r in rows if r[9] == 1]  # actual_finish == 1
        top3 = [r for r in rows if r[9] <= 3]
        win_count = len(winners)

        # ---- 1. Post Position Win Rates ----
        post_wins: dict[int, int] = {}
        post_total: dict[int, int] = {}
        for r in rows:
            pp = r[0] or 0
            post_total[pp] = post_total.get(pp, 0) + 1
        for w in winners:
            pp = w[0] or 0
            post_wins[pp] = post_wins.get(pp, 0) + 1

        post_rates = {}
        for pp in sorted(post_total.keys()):
            if pp > 0:
                wins = post_wins.get(pp, 0)
                runs = post_total[pp]
                post_rates[str(pp)] = round(wins / max(runs, 1), 3)

        _upsert("post_win_rates", json.dumps(post_rates), win_count)

        # Best post positions (top 3 by win rate with >= 2 sample)
        best_posts = sorted(
            [
                (pp, rate)
                for pp, rate in post_rates.items()
                if post_total.get(int(pp), 0) >= 2
            ],
            key=lambda x: -x[1],
        )[:3]
        _upsert("best_posts", json.dumps([p[0] for p in best_posts]), win_count)

        # ---- 2. Running Style Win Rates ----
        style_wins: dict[str, int] = {}
        style_total: dict[str, int] = {}
        for r in rows:
            s = r[1] or "P"
            style_total[s] = style_total.get(s, 0) + 1
        for w in winners:
            s = w[1] or "P"
            style_wins[s] = style_wins.get(s, 0) + 1

        style_rates = {}
        for s in sorted(style_total.keys()):
            wins = style_wins.get(s, 0)
            runs = style_total[s]
            style_rates[s] = round(wins / max(runs, 1), 3)

        _upsert("style_win_rates", json.dumps(style_rates), win_count)

        dominant_style = max(style_rates, key=style_rates.get) if style_rates else "P"
        _upsert("dominant_winning_style", dominant_style, win_count)

        # ---- 3. Average Winning Speed Figures ----
        if winners:
            avg_best_beyer = sum(w[3] or 0 for w in winners) / win_count
            avg_last_beyer = sum(w[4] or 0 for w in winners) / win_count
            avg_avg3 = sum(w[5] or 0 for w in winners) / win_count
            _upsert("avg_winner_best_beyer", str(round(avg_best_beyer, 1)), win_count)
            _upsert("avg_winner_last_beyer", str(round(avg_last_beyer, 1)), win_count)
            _upsert("avg_winner_avg_beyer_3", str(round(avg_avg3, 1)), win_count)

        # ---- 4. Average Winning Class Rating ----
        if winners:
            avg_class = sum(w[2] or 0 for w in winners) / win_count
            _upsert("avg_winner_class_rating", str(round(avg_class, 2)), win_count)

        # ---- 5. Freshness / Days Since Last ----
        if winners:
            avg_dsl = sum(w[6] or 0 for w in winners) / win_count
            _upsert("avg_winner_days_since_last", str(round(avg_dsl, 1)), win_count)

        # ---- 6. Workout Pattern Distribution (winners) ----
        workout_dist: dict[str, int] = {}
        for w in winners:
            wp = w[11] or "Sparse"
            workout_dist[wp] = workout_dist.get(wp, 0) + 1
        _upsert("winner_workout_patterns", json.dumps(workout_dist), win_count)

        # ---- 7. Average Field Size ----
        avg_field = sum(r[10] or 8 for r in rows) / total
        _upsert("avg_field_size", str(round(avg_field, 1)), total)

        # ---- 8. Morning Line Odds of Winners ----
        if winners:
            avg_ml = sum(w[8] or 0 for w in winners) / win_count
            _upsert("avg_winner_ml_odds", str(round(avg_ml, 1)), win_count)

        # ---- 9. Prime Power of Winners ----
        if winners:
            avg_pp = sum(w[7] or 0 for w in winners) / win_count
            _upsert("avg_winner_prime_power", str(round(avg_pp, 1)), win_count)

        # ---- 10. Top Jockeys (by win count) ----
        jockey_wins: dict[str, int] = {}
        for w in winners:
            j = w[14] or ""
            if j:
                jockey_wins[j] = jockey_wins.get(j, 0) + 1
        top_jockeys = sorted(jockey_wins.items(), key=lambda x: -x[1])[:5]
        _upsert("top_jockeys", json.dumps(dict(top_jockeys)), win_count)

        # ---- 11. Top Trainers (by win count) ----
        trainer_wins: dict[str, int] = {}
        for w in winners:
            t = w[15] or ""
            if t:
                trainer_wins[t] = trainer_wins.get(t, 0) + 1
        top_trainers = sorted(trainer_wins.items(), key=lambda x: -x[1])[:5]
        _upsert("top_trainers", json.dumps(dict(top_trainers)), win_count)

        # ---- 12. Total races tracked ----
        # Count distinct race_ids for this combo
        cursor.execute(
            """
            SELECT COUNT(DISTINCT race_id) FROM track_pattern_winners
            WHERE track_code = ? AND surface = ?
              AND distance_furlongs BETWEEN ? AND ?
        """,
            (track_code, surface, max(0, furlongs - 1.0), furlongs + 1.0),
        )
        distinct_races = cursor.fetchone()[0]
        _upsert("total_races_analyzed", str(distinct_races), distinct_races)

        conn.commit()
        conn.close()
        logger.info(
            f"📊 Rebuilt pattern stats: {track_code} {surface} {dist_cat} "
            f"({distinct_races} races, {win_count} winners)"
        )

    def get_track_patterns(
        self,
        track_code: str,
        surface: str,
        distance: str,
    ) -> dict:
        """
        Retrieve learned track patterns for a given track/surface/distance.

        Returns a dict with all stat keys, or empty dict if no data.
        Used during handicapping to apply track-level pattern bonuses.
        """
        furlongs = self._distance_to_furlongs(distance)
        dist_cat = self._categorize_distance(furlongs)

        conn = sqlite3.connect(self.db_path, timeout=5)
        cursor = conn.cursor()

        cursor.execute(
            """
            SELECT stat_key, stat_value, sample_size
            FROM track_pattern_stats
            WHERE track_code = ? AND surface = ? AND distance_category = ?
        """,
            (track_code, surface, dist_cat),
        )

        rows = cursor.fetchall()
        conn.close()

        if not rows:
            return {}

        result: dict = {}
        for key, value, sample in rows:
            # Try to parse JSON values
            try:
                parsed = json.loads(value)
                result[key] = {"value": parsed, "sample_size": sample}
            except (json.JSONDecodeError, TypeError):
                try:
                    result[key] = {"value": float(value), "sample_size": sample}
                except ValueError:
                    result[key] = {"value": value, "sample_size": sample}

        return result

    def get_all_track_summaries(self) -> list[dict]:
        """
        Get a summary of all tracks with learned patterns.
        Useful for displaying in the UI.
        """
        conn = sqlite3.connect(self.db_path, timeout=5)
        cursor = conn.cursor()

        cursor.execute("""
            SELECT track_code, surface, distance_category,
                   COUNT(*) as stat_count,
                   MAX(last_updated) as last_updated,
                   MAX(CASE WHEN stat_key = 'total_races_analyzed'
                       THEN CAST(stat_value AS INTEGER) ELSE 0 END) as races
            FROM track_pattern_stats
            GROUP BY track_code, surface, distance_category
            ORDER BY track_code, surface, distance_category
        """)

        summaries = []
        for row in cursor.fetchall():
            summaries.append(
                {
                    "track_code": row[0],
                    "surface": row[1],
                    "distance_category": row[2],
                    "stat_count": row[3],
                    "last_updated": row[4],
                    "races_analyzed": row[5],
                }
            )

        conn.close()
        return summaries
