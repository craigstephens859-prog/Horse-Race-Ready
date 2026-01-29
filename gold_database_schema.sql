-- ============================================================================
-- GOLD-STANDARD HIGH-IQ DATABASE SCHEMA
-- Optimized for Maximum ML Accuracy & Clean Retraining
-- ============================================================================

-- TABLE 1: races_analyzed
-- Stores every race analyzed with "Analyze This Race" button
CREATE TABLE IF NOT EXISTS races_analyzed (
    race_id TEXT PRIMARY KEY,
    track_code TEXT NOT NULL,
    race_date TEXT NOT NULL,
    race_number INTEGER NOT NULL,
    race_type TEXT,
    surface TEXT,
    distance TEXT,
    track_condition TEXT,
    purse REAL,
    field_size INTEGER,
    pp_text_raw TEXT,  -- Full BRISNET PP text
    analyzed_timestamp TEXT NOT NULL,
    UNIQUE(track_code, race_date, race_number)
);

CREATE INDEX IF NOT EXISTS idx_races_date ON races_analyzed(race_date DESC);
CREATE INDEX IF NOT EXISTS idx_races_track ON races_analyzed(track_code, race_date);

-- TABLE 2: horses_analyzed
-- Every horse in every analyzed race with full feature set
CREATE TABLE IF NOT EXISTS horses_analyzed (
    horse_id TEXT PRIMARY KEY,
    race_id TEXT NOT NULL,
    program_number INTEGER NOT NULL,
    horse_name TEXT NOT NULL,
    post_position INTEGER,
    morning_line_odds REAL,
    
    -- Elite Parser Features (40+ fields)
    jockey TEXT,
    trainer TEXT,
    owner TEXT,
    weight REAL,
    medication TEXT,
    equipment TEXT,
    running_style TEXT,
    prime_power REAL,
    
    -- Speed Figures
    best_beyer INTEGER,
    last_beyer INTEGER,
    avg_beyer_3 REAL,
    e1_pace REAL,
    e2_pace REAL,
    late_pace REAL,
    
    -- Form & Class
    days_since_last INTEGER,
    starts_lifetime INTEGER,
    wins_lifetime INTEGER,
    win_pct REAL,
    earnings_lifetime REAL,
    class_rating REAL,
    
    -- 8 Angle Scores (normalized 0-1)
    angle_early_speed REAL,
    angle_class REAL,
    angle_recency REAL,
    angle_work_pattern REAL,
    angle_connections REAL,
    angle_pedigree REAL,
    angle_runstyle_bias REAL,
    angle_post REAL,
    
    -- Unified Rating Engine Components
    rating_class REAL,
    rating_form REAL,
    rating_speed REAL,
    rating_pace REAL,
    rating_style REAL,
    rating_post REAL,
    rating_angles_total REAL,
    rating_tier2_bonus REAL,
    rating_final REAL,
    
    -- PhD Enhancements (NEW)
    rating_confidence REAL,  -- Entropy-based confidence [0,1]
    form_decay_score REAL,   -- Exponential decay form rating
    pace_esp_score REAL,     -- Game-theoretic pace scenario
    mud_adjustment REAL,     -- Pedigree-based off-track adjustment
    
    -- Model Predictions
    predicted_probability REAL NOT NULL,  -- Softmax win probability
    predicted_rank INTEGER,  -- 1=favorite, 2=second choice, etc.
    fair_odds REAL,
    
    FOREIGN KEY (race_id) REFERENCES races_analyzed(race_id)
);

CREATE INDEX IF NOT EXISTS idx_horses_race ON horses_analyzed(race_id);
CREATE INDEX IF NOT EXISTS idx_horses_prob ON horses_analyzed(predicted_probability DESC);

-- TABLE 3: gold_high_iq 🏆
-- CRITICAL: Only completed races with actual results for ML retraining
-- This is the GOLD STANDARD training data
CREATE TABLE IF NOT EXISTS gold_high_iq (
    result_id TEXT PRIMARY KEY,
    race_id TEXT NOT NULL,
    horse_id TEXT NOT NULL,
    
    -- Actual Race Outcome
    actual_finish_position INTEGER NOT NULL,  -- 1=winner, 2=second, etc.
    beaten_lengths REAL,
    final_time REAL,
    
    -- Copy of all prediction features (from horses_analyzed)
    program_number INTEGER,
    horse_name TEXT,
    post_position INTEGER,
    
    -- All features needed for retraining
    rating_final REAL,
    predicted_probability REAL,
    predicted_rank INTEGER,
    
    -- Feature vector (serialized for ML)
    features_json TEXT,  -- JSON of all 50+ features
    
    -- Quality metrics
    prediction_error REAL,  -- abs(predicted_rank - actual_finish_position)
    was_top5_correct BOOLEAN,  -- Did we get this horse in predicted top 5?
    
    -- Timestamps
    result_entered_timestamp TEXT NOT NULL,
    
    FOREIGN KEY (race_id) REFERENCES races_analyzed(race_id),
    FOREIGN KEY (horse_id) REFERENCES horses_analyzed(horse_id),
    UNIQUE(race_id, horse_id)
);

CREATE INDEX IF NOT EXISTS idx_gold_race ON gold_high_iq(race_id);
CREATE INDEX IF NOT EXISTS idx_gold_finish ON gold_high_iq(actual_finish_position);
CREATE INDEX IF NOT EXISTS idx_gold_quality ON gold_high_iq(was_top5_correct, prediction_error);

-- TABLE 4: retraining_history
-- Track every ML retraining run for performance monitoring
CREATE TABLE IF NOT EXISTS retraining_history (
    retrain_id INTEGER PRIMARY KEY AUTOINCREMENT,
    retrain_timestamp TEXT NOT NULL,
    
    -- Training dataset info
    total_races_used INTEGER,
    total_horses_used INTEGER,
    train_split_pct REAL,
    val_split_pct REAL,
    
    -- Model performance metrics
    val_winner_accuracy REAL,
    val_top3_accuracy REAL,
    val_top5_accuracy REAL,
    val_loss REAL,
    
    -- Model file path
    model_checkpoint_path TEXT,
    
    -- Training config
    epochs_trained INTEGER,
    learning_rate REAL,
    batch_size INTEGER,
    
    -- Performance tracking
    training_duration_seconds REAL
);

-- TABLE 5: race_results_summary
-- Aggregate statistics per completed race for quick dashboard display
CREATE TABLE IF NOT EXISTS race_results_summary (
    race_id TEXT PRIMARY KEY,
    
    -- Actual top 5 finishers
    winner_name TEXT,
    second_name TEXT,
    third_name TEXT,
    fourth_name TEXT,
    fifth_name TEXT,
    
    -- Model accuracy metrics
    top1_predicted_correctly BOOLEAN,
    top3_predicted_correctly INTEGER,  -- How many of top 3 in our predicted top 3
    top5_predicted_correctly INTEGER,  -- How many of top 5 in our predicted top 5
    
    -- Value metrics
    winner_predicted_odds REAL,
    winner_actual_payout REAL,
    roi_if_bet_on_predicted_winner REAL,
    
    -- Completion timestamp
    results_complete_timestamp TEXT NOT NULL,
    
    FOREIGN KEY (race_id) REFERENCES races_analyzed(race_id)
);

-- ============================================================================
-- HELPER VIEWS FOR QUICK QUERIES
-- ============================================================================

-- View: Pending races (analyzed but no results yet)
CREATE VIEW IF NOT EXISTS v_pending_races AS
SELECT 
    ra.race_id,
    ra.track_code,
    ra.race_date,
    ra.race_number,
    ra.surface,
    ra.distance,
    ra.field_size,
    ra.analyzed_timestamp
FROM races_analyzed ra
WHERE NOT EXISTS (
    SELECT 1 FROM gold_high_iq g WHERE g.race_id = ra.race_id
)
ORDER BY ra.race_date DESC, ra.race_number;

-- View: Completed races with accuracy metrics
CREATE VIEW IF NOT EXISTS v_completed_races AS
SELECT 
    ra.race_id,
    ra.track_code,
    ra.race_date,
    ra.race_number,
    rs.winner_name,
    rs.top1_predicted_correctly,
    rs.top3_predicted_correctly,
    rs.top5_predicted_correctly,
    rs.roi_if_bet_on_predicted_winner,
    COUNT(g.horse_id) as horses_with_results
FROM races_analyzed ra
JOIN race_results_summary rs ON ra.race_id = rs.race_id
JOIN gold_high_iq g ON ra.race_id = g.race_id
GROUP BY ra.race_id
ORDER BY ra.race_date DESC, ra.race_number;

-- View: Model performance over time
CREATE VIEW IF NOT EXISTS v_model_performance AS
SELECT 
    retrain_timestamp,
    total_races_used,
    val_winner_accuracy,
    val_top3_accuracy,
    val_top5_accuracy,
    val_loss
FROM retraining_history
ORDER BY retrain_timestamp DESC;

-- ============================================================================
-- PERFORMANCE OPTIMIZATION INDEXES
-- ============================================================================

CREATE INDEX IF NOT EXISTS idx_gold_features ON gold_high_iq(predicted_probability, actual_finish_position);
CREATE INDEX IF NOT EXISTS idx_results_summary_accuracy ON race_results_summary(top1_predicted_correctly, top3_predicted_correctly);
CREATE INDEX IF NOT EXISTS idx_horses_analyzed_composite ON horses_analyzed(race_id, predicted_rank, predicted_probability);

-- ============================================================================
-- DATABASE METADATA
-- ============================================================================

CREATE TABLE IF NOT EXISTS schema_version (
    version INTEGER PRIMARY KEY,
    applied_timestamp TEXT NOT NULL,
    description TEXT
);

INSERT OR IGNORE INTO schema_version (version, applied_timestamp, description)
VALUES (1, datetime('now'), 'Initial gold-standard high-IQ schema with PhD enhancements');
