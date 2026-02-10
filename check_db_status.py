"""Quick database status check for adaptive learning system."""
import sqlite3
import os

DB_PATH = "gold_high_iq.db"

if not os.path.exists(DB_PATH):
    print("❌ DATABASE NOT FOUND")
    exit()

conn = sqlite3.connect(DB_PATH)
c = conn.cursor()

# List all tables
c.execute("SELECT name FROM sqlite_master WHERE type='table'")
tables = [t[0] for t in c.fetchall()]
print("=== TABLES ===")
for t in tables:
    print(f"  {t}")

# Check races_analyzed
print("\n=== RACES ANALYZED ===")
try:
    c.execute("SELECT race_id, track_code, race_number, race_date, race_type, field_size FROM races_analyzed ORDER BY created_timestamp DESC")
    rows = c.fetchall()
    print(f"Total: {len(rows)} races")
    for r in rows:
        print(f"  {r[0]} | Track: {r[1]} | R{r[2]} | {r[3]} | {r[4]} | FS: {r[5]}")
except Exception as e:
    print(f"  Error: {e}")

# Check gold_high_iq (results)
print("\n=== GOLD_HIGH_IQ (Saved Results) ===")
try:
    c.execute("SELECT race_id, horse_name, predicted_rank, actual_finish_position, predicted_probability, result_entered_timestamp FROM gold_high_iq ORDER BY result_entered_timestamp DESC LIMIT 50")
    rows = c.fetchall()
    print(f"Total rows: {len(rows)}")
    current_race = None
    for r in rows:
        if r[0] != current_race:
            current_race = r[0]
            print(f"\n  Race: {r[0]}")
        afp = r[3] if r[3] is not None else "N/A"
        prob = f"{r[4]:.1%}" if r[4] is not None else "N/A"
        ts = r[5] if r[5] else "N/A"
        print(f"    {str(r[1]):25s} | PredRank: {r[2]:2} | ActualFinish: {afp} | Prob: {prob} | Saved: {ts}")
except Exception as e:
    print(f"  Error: {e}")

# Check learned_weights
print("\n=== LEARNED WEIGHTS ===")
try:
    c.execute("SELECT weight_name, weight_value, races_trained_on, last_updated, confidence FROM learned_weights")
    rows = c.fetchall()
    if rows:
        for r in rows:
            print(f"  {r[0]:10s}: {r[1]:.4f} | Trained on: {r[2]} races | Updated: {r[3]} | Conf: {r[4]:.2f}")
    else:
        print("  (no learned weights yet)")
except Exception as e:
    print(f"  Table not found: {e}")

# Check calibration_history
print("\n=== CALIBRATION HISTORY ===")
try:
    c.execute("SELECT calibration_timestamp, races_analyzed, winner_accuracy, top3_accuracy, weights_json FROM calibration_history ORDER BY calibration_timestamp DESC LIMIT 5")
    rows = c.fetchall()
    print(f"Total calibration events: {len(rows)}")
    for r in rows:
        print(f"  {r[0]} | Races: {r[1]} | WinAcc: {r[2]*100:.0f}% | Top3: {r[3]*100:.0f}%")
        if r[4]:
            print(f"    Weights: {r[4][:100]}...")
except Exception as e:
    print(f"  Table not found: {e}")

# Check race_insights (intelligent learning)
print("\n=== RACE INSIGHTS (Intelligent Learning) ===")
try:
    c.execute("SELECT race_id, pattern_type, horse_name, description, confidence, weight_key, weight_adjustment FROM race_insights ORDER BY timestamp DESC LIMIT 20")
    rows = c.fetchall()
    print(f"Total insights: {len(rows)}")
    for r in rows:
        desc_short = (r[3] or "")[:60]
        print(f"  {r[0]} | {r[1]:30s} | {str(r[2]):20s} | Conf: {r[4]:.2f} | {desc_short}")
except Exception as e:
    print(f"  Table not found: {e}")

# Check pattern_frequency
print("\n=== PATTERN FREQUENCY ===")
try:
    c.execute("SELECT pattern_type, occurrence_count, avg_rank_improvement FROM pattern_frequency ORDER BY occurrence_count DESC")
    rows = c.fetchall()
    if rows:
        for r in rows:
            print(f"  {r[0]:35s} | Count: {r[1]:3d} | Avg Rank Improvement: {r[2]:.2f}")
    else:
        print("  (no patterns recorded yet)")
except Exception as e:
    print(f"  Table not found: {e}")

# Check odds_drift_learning
print("\n=== ODDS DRIFT LEARNING ===")
try:
    c.execute("SELECT COUNT(*) FROM odds_drift_learning")
    count = c.fetchone()[0]
    print(f"Total drift records: {count}")
except Exception as e:
    print(f"  Table not found: {e}")

conn.close()
print("\n✅ Database check complete")
